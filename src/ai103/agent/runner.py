"""Boucle d'exécution du prompt agent : le modèle demande des outils, notre code les exécute.

Chaque question ouvre une conversation Foundry, supprimée à la fin. La boucle s'arrête sur une
réponse finale, un budget d'appels d'outils, un plafond de coût ou un nombre maximal de tours.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from ai103.agent.tools import ToolBox
from ai103.observability import tracer
from ai103.rag.generator import Usage, is_content_filter
from ai103.rag.guardrails import AnswerStatus, CitationCheck, answer_status, check_citations
from ai103.rag.prompts import RagAnswer

BUDGET_MESSAGE = json.dumps(
    {
        "error": "Tool budget exhausted. Answer now from the passages you already have, or set "
        "insufficient_evidence to true."
    }
)


@dataclass
class ToolCall:
    round: int
    name: str
    arguments: str
    seconds: float
    executed: bool
    summary: str


@dataclass
class AgentResult:
    question: str
    status: AnswerStatus
    answer: RagAnswer
    checks: list[CitationCheck]
    toolbox: ToolBox
    tool_calls: list[ToolCall]
    usage: Usage
    rounds: int
    stop_reason: str
    seconds: float
    content_filter: object = None
    raw_output: str = ""
    response_ids: list[str] = field(default_factory=list)


def _summary(output: str) -> str:
    """Résumé court d'un résultat d'outil pour le journal (le texte complet reste en mémoire)."""
    data = json.loads(output)
    if "error" in data:
        return f"error: {str(data['error'])[:120]}"
    if "passages" in data:
        return f"{len(data['passages'])} passages, filter={data.get('filter')}"
    if "filings" in data:
        return f"{len(data['filings'])} filings for {data.get('company')}"
    return json.dumps(data)[:120]


def _add_usage(total: Usage, response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return total
    current = Usage.from_response(usage)
    return Usage(
        total.input_tokens + current.input_tokens,
        total.cached_input_tokens + current.cached_input_tokens,
        total.output_tokens + current.output_tokens,
        total.reasoning_tokens + current.reasoning_tokens,
    )


class AgentRunner:
    def __init__(
        self,
        client: OpenAI,
        toolbox_factory: Callable[[], ToolBox],
        *,
        max_searches: int = 6,
        max_tool_calls: int = 12,
        max_cost_usd: float = 0.10,
        max_rounds: int = 10,
    ) -> None:
        """Deux budgets : les recherches (coûteuses) et l'ensemble des appels d'outils.

        Le modèle peut demander plusieurs outils dans un même tour (appels parallèles) ;
        les calculs, gratuits, ne doivent pas épuiser le budget des recherches.
        """
        if not 1 <= max_searches <= max_tool_calls or max_rounds < 2 or max_cost_usd <= 0:
            raise ValueError("Limites de l'agent invalides.")
        self.max_searches = max_searches
        self.client = client
        self.toolbox_factory = toolbox_factory
        self.max_tool_calls = max_tool_calls
        self.max_cost_usd = max_cost_usd
        self.max_rounds = max_rounds

    def answer(self, question: str) -> AgentResult:
        if not question.strip():
            raise ValueError("Question vide.")
        started = perf_counter()
        toolbox = self.toolbox_factory()
        calls: list[ToolCall] = []
        usage = Usage(0, 0, 0, 0)
        response_ids: list[str] = []
        stop_reason, rounds, response = "final_answer", 0, None
        with tracer.start_as_current_span("agent.answer") as root:
            conversation = self.client.conversations.create()
            try:
                request: Any = question
                while True:
                    rounds += 1
                    with tracer.start_as_current_span("agent.round", attributes={"round": rounds}):
                        response = self.client.responses.create(
                            input=request, conversation=conversation.id
                        )
                    response_ids.append(response.id)
                    usage = _add_usage(usage, response)
                    pending = [item for item in response.output if item.type == "function_call"]
                    if not pending:
                        break
                    if usage.cost_usd >= self.max_cost_usd:
                        stop_reason = "cost_limit"
                        break
                    if rounds >= self.max_rounds:
                        stop_reason = "round_limit"
                        break
                    request = [self._execute(toolbox, call, rounds, calls) for call in pending]
            except BadRequestError as exc:
                if not is_content_filter(exc):
                    raise
                stop_reason = "content_filter"
                result = self._finish(
                    question, toolbox, calls, usage, rounds, stop_reason, started, response_ids
                )
                result.content_filter = exc.body
                return result
            finally:
                try:
                    self.client.conversations.delete(conversation.id)
                except Exception:  # noqa: BLE001 - nettoyage au mieux, sans masquer le résultat
                    pass
            root.set_attribute("ai103.tool_calls", len(calls))
            root.set_attribute("ai103.cost_usd", usage.cost_usd)
            return self._finish(
                question,
                toolbox,
                calls,
                usage,
                rounds,
                stop_reason,
                started,
                response_ids,
                raw_output=getattr(response, "output_text", "") or "",
            )

    def _execute(
        self, toolbox: ToolBox, call: Any, round_: int, calls: list[ToolCall]
    ) -> dict[str, str]:
        done = [c for c in calls if c.executed]
        searches = sum(c.name == "search_filings" for c in done)
        executed = len(done) < self.max_tool_calls and (
            call.name != "search_filings" or searches < self.max_searches
        )
        started = perf_counter()
        with tracer.start_as_current_span(f"tool.{call.name}", attributes={"round": round_}):
            output = toolbox.call(call.name, call.arguments) if executed else BUDGET_MESSAGE
        calls.append(
            ToolCall(
                round_,
                call.name,
                call.arguments,
                perf_counter() - started,
                executed,
                _summary(output),
            )
        )
        return {"type": "function_call_output", "call_id": call.call_id, "output": output}

    def _finish(
        self,
        question: str,
        toolbox: ToolBox,
        calls: list[ToolCall],
        usage: Usage,
        rounds: int,
        stop_reason: str,
        started: float,
        response_ids: list[str],
        raw_output: str = "",
    ) -> AgentResult:
        sources = toolbox.registry.sources
        checks: list[CitationCheck] = []
        if stop_reason == "content_filter":
            status = AnswerStatus.BLOCKED_BY_CONTENT_FILTER
            answer = RagAnswer(
                answer="Blocked by the content safety filter.",
                citations=[],
                insufficient_evidence=True,
            )
        elif stop_reason != "final_answer":
            status = AnswerStatus.LIMIT_REACHED
            answer = RagAnswer(
                answer=f"Stopped: {stop_reason}.", citations=[], insufficient_evidence=True
            )
        else:
            try:
                answer = RagAnswer.model_validate_json(raw_output)
            except ValidationError:
                # Refus ou texte libre au lieu du JSON demandé : un statut, pas une panne.
                status = AnswerStatus.INVALID_OUTPUT
                answer = RagAnswer(
                    answer=raw_output[:500], citations=[], insufficient_evidence=True
                )
            else:
                checks = check_citations(answer, sources)
                status = answer_status(answer, checks)
        return AgentResult(
            question=question,
            status=status,
            answer=answer,
            checks=checks,
            toolbox=toolbox,
            tool_calls=calls,
            usage=usage,
            rounds=rounds,
            stop_reason=stop_reason,
            seconds=perf_counter() - started,
            raw_output=raw_output,
            response_ids=response_ids,
        )


def agent_record(result: AgentResult, settings: dict[str, Any]) -> dict[str, Any]:
    """Même structure que les rapports du RAG : l'évaluation et la comparaison s'appliquent."""
    verified = {c.source for c in result.checks if c.status.value == "verified"}
    return {
        "settings": settings,
        "question": result.question,
        "status": result.status.value,
        "stop_reason": result.stop_reason,
        "content_filter": result.content_filter,
        "answer": result.answer.model_dump(),
        "citation_checks": [asdict(check) for check in result.checks],
        "sources": [
            {
                "number": source.number,
                "verified": source.number in verified,
                "document": source.document,
                "pages": source.pages,
                "chunk": asdict(source.chunk),
            }
            for source in result.toolbox.registry.sources
        ],
        "tool_calls": [asdict(call) for call in result.tool_calls],
        "rounds": result.rounds,
        "timings": {"total_seconds": result.seconds},
        "usage": asdict(result.usage),
        "cost_usd": result.usage.cost_usd,
        "response_ids": result.response_ids,
    }
