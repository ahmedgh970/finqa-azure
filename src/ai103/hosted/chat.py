"""Chatbot financier : le modèle répond seul aux questions générales et appelle nos outils
quand la question porte sur les rapports du corpus.

Le modèle est appelé directement (Responses API, sans prompt agent ni conversation Foundry) :
notre code garde les éléments de chaque tour et n'en stocke aucun côté serveur (`store=False`).
La mémoire entre les questions vient de l'historique fourni par la plateforme hosted.
"""

import json
import re
import unicodedata
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from opentelemetry.trace import Span

from ai103.agent.definition import ANSWER_SCHEMA, INSTRUCTIONS
from ai103.agent.runner import AgentResult, AgentRunner
from ai103.agent.tools import TOOL_DEFINITIONS
from ai103.observability import trace_headers, tracer
from ai103.rag.generator import Usage
from ai103.rag.guardrails import AnswerStatus

CHAT_INTRO = """\
You are a financial analyst assistant in a chat. You can hold a conversation, answer general \
questions (greetings, what you can do, definitions of financial concepts, how a ratio is \
computed) directly from your knowledge, and answer follow-up questions using the conversation \
history. Answer in the language of the user.

You also have tools over a fixed corpus of company filings (10-K, 10-Q, 8-K and earnings \
releases). Use them when the question needs figures, facts or analysis from a company's \
filings; do not use them for general questions. Never state a company's reported figures from \
your own knowledge: retrieve them with the tools.

If the user repeats or rephrases a question already answered in this conversation, or asks for \
figures that an earlier answer already gives, answer from that earlier answer without calling \
the tools, and say that it comes from earlier in the conversation. Call the tools again only for \
information not given yet, or when the user explicitly asks you to check or refresh it.

Source numbers refer only to the passages returned in the current turn: when you reuse an earlier \
answer or answer a general question, return no citations and set insufficient_evidence to \
false. Set basis to "filings" when the answer relies on passages retrieved in this turn, \
"conversation" when it reuses an earlier answer, and "general" otherwise.

For questions about the filings, follow this method:"""

# Même méthode que le prompt agent évalué (A2) : seul le préambule change.
CHAT_INSTRUCTIONS = CHAT_INTRO + "\n" + INSTRUCTIONS.split("Method:\n", 1)[1]

# Schéma d'A2 plus `basis` : ce sur quoi le modèle dit s'appuyer, contrôlé ensuite par le code.
CHAT_SCHEMA: dict[str, Any] = {
    **ANSWER_SCHEMA,
    "properties": {
        **ANSWER_SCHEMA["properties"],
        "basis": {"type": "string", "enum": ["filings", "conversation", "general"]},
    },
    "required": [*ANSWER_SCHEMA["required"], "basis"],
}
TEXT_FORMAT: dict[str, Any] = {
    "format": {"type": "json_schema", "name": "chat_answer", "schema": CHAT_SCHEMA, "strict": True}
}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

STATUS_LABELS = {
    AnswerStatus.VERIFIED: "réponse sourcée, extraits vérifiés",
    AnswerStatus.PARTIALLY_VERIFIED: "réponse sourcée, extraits en partie vérifiés",
    AnswerStatus.UNSUPPORTED: "extraits non retrouvés dans les passages",
    AnswerStatus.INSUFFICIENT_EVIDENCE: "informations insuffisantes dans les rapports",
    AnswerStatus.LIMIT_REACHED: "arrêt sur une limite (budget, tours ou coût)",
    AnswerStatus.BLOCKED_BY_CONTENT_FILTER: "bloquée par le filtre de contenu",
    AnswerStatus.INVALID_OUTPUT: "réponse au format inattendu",
}
REUSED_OK = "chiffres retrouvés dans la conversation"
# Réponses fiables à reprendre telles quelles ; un échec précédent se retente avec les outils.
RELIABLE_ANSWERS = (STATUS_LABELS[AnswerStatus.VERIFIED], REUSED_OK)
REPEAT_SIMILARITY = 0.9


def _normalized(text: str) -> str:
    """Minuscules, sans accents ni ponctuation : « Quel est… ? » et « quel est… » se valent."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain.lower()).split())


def answered_before(question: str, history: list[dict[str, Any]]) -> bool:
    """La même question (à 90 % près) a déjà reçu une réponse vérifiée dans la conversation."""
    asked = _normalized(question)
    for previous, reply in zip(history, history[1:], strict=False):
        if (
            previous["role"] == "user"
            and reply["role"] == "assistant"
            and any(label in str(reply["content"]) for label in RELIABLE_ANSWERS)
            and SequenceMatcher(None, asked, _normalized(str(previous["content"]))).ratio()
            >= REPEAT_SIMILARITY
        ):
            return True
    return False


ANSWER_START = re.compile(r'"answer"\s*:\s*"')
ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


class AnswerStream:
    """Extrait au fil de l'eau le champ `answer` d'une réponse JSON encore en cours d'écriture.

    Le modèle écrit `{"answer": "…", "citations": […], …}` (`answer` en premier dans le schéma) :
    seul ce texte est montré pendant la génération ; une séquence d'échappement coupée entre deux
    morceaux attend le suivant.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.position: int | None = None
        self.done = False

    def feed(self, delta: str) -> str:
        self.buffer += delta
        if self.done:
            return ""
        if self.position is None:
            match = ANSWER_START.search(self.buffer)
            if not match:
                return ""
            self.position = match.end()
        text, i = [], self.position
        while i < len(self.buffer):
            char = self.buffer[i]
            if char == '"':
                self.done = True
                break
            if char != "\\":
                text.append(char)
                i += 1
                continue
            escape = self._escape(i)
            if escape is None:
                break
            decoded, i = escape
            text.append(decoded)
        self.position = i
        return "".join(text)

    def _escape(self, i: int) -> tuple[str, int] | None:
        """Séquence d'échappement complète à la position `i`, ou None s'il manque la suite."""
        if i + 1 >= len(self.buffer):
            return None
        code = self.buffer[i + 1]
        if code != "u":
            return ESCAPES.get(code, code), i + 2
        # \uXXXX, ou une paire 😀 pour les caractères hors du plan de base.
        size = 12 if "d800" <= self.buffer[i + 2 : i + 6].lower() <= "dbff" else 6
        if i + size > len(self.buffer):
            return None
        return json.loads(f'"{self.buffer[i : i + size]}"'), i + size


# Rappel du texte en cours de génération, propre à chaque question (requêtes concurrentes).
_ON_TEXT: ContextVar[Callable[[str], None] | None] = ContextVar("on_text", default=None)


@dataclass
class Turn:
    """Éléments envoyés au modèle pour une question, outils permis et rappel du streaming."""

    items: list[dict[str, Any]]
    tool_choice: str = "auto"
    on_text: Callable[[str], None] | None = None


class DirectRunner(AgentRunner):
    """Même boucle, mêmes budgets et mêmes contrôles qu'A2, avec un appel direct au modèle."""

    def __init__(
        self, client: Any, toolbox_factory: Any, *, model: str, effort: str = "low", **limits: Any
    ) -> None:
        super().__init__(client, toolbox_factory, **limits)
        self.model = model
        self.effort = effort

    def answer(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentResult:
        """`on_text` reçoit le texte du modèle au fil de la génération (streaming)."""
        token = _ON_TEXT.set(on_text)
        try:
            return super().answer(question, history)
        finally:
            _ON_TEXT.reset(token)

    def _open(self, question: str, history: list[dict[str, Any]], root: Span) -> tuple[Any, Any]:
        # Question déjà résolue : le code interdit les outils, le modèle reprend sa réponse
        # (instructions seules insuffisantes : il relançait la recherche au test du 23 octobre).
        repeated = answered_before(question, history)
        root.set_attributes({"ai103.history_messages": len(history), "ai103.repeat": repeated})
        items = [*history, {"role": "user", "content": question}]
        return Turn(items, "none" if repeated else "auto", _ON_TEXT.get()), None

    def _step(self, session: Any, request: Any) -> Any:
        if request:
            session.items.extend(request)
        with tracer.start_as_current_span(
            f"chat {self.model}",
            attributes={"gen_ai.operation.name": "chat", "gen_ai.request.model": self.model},
        ) as span:
            # Paramètres en dictionnaires simples (schémas partagés avec le prompt agent).
            client: Any = self.client
            call: dict[str, Any] = {
                "model": self.model,
                "instructions": CHAT_INSTRUCTIONS,
                "input": session.items,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": session.tool_choice,
                "text": TEXT_FORMAT,
                "reasoning": {"effort": self.effort},
                # Rien n'est conservé côté serveur : le raisonnement chiffré est renvoyé au tour
                # suivant avec les appels d'outils, comme le ferait une conversation stockée.
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "extra_headers": trace_headers(),
            }
            if session.on_text is None:
                response = client.responses.create(**call)
            else:
                response = self._stream(client, call, session.on_text)
            # Aucun span serveur pour un appel direct au modèle : les tokens sont portés ici.
            if response.usage is not None:
                usage = Usage.from_response(response.usage)
                span.set_attributes(
                    {
                        "gen_ai.response.id": response.id,
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.cache_read.input_tokens": usage.cached_input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                        "gen_ai.usage.reasoning.output_tokens": usage.reasoning_tokens,
                    }
                )
        session.items.extend(
            item.model_dump(mode="json", exclude_none=True) for item in response.output
        )
        return response

    @staticmethod
    def _stream(client: Any, call: dict[str, Any], on_text: Callable[[str], None]) -> Any:
        """Même appel en streaming : le texte part vers `on_text`, la réponse finale est rendue."""
        final = None
        with client.responses.create(**call, stream=True) as events:
            for event in events:
                if event.type == "response.output_text.delta":
                    on_text(event.delta)
                elif event.type in FINAL_EVENTS:
                    final = event.response
        if final is None:
            raise RuntimeError("Flux du modèle terminé sans réponse finale.")
        return final

    def _close(self, session: Any) -> None:
        return None


def is_general(result: AgentResult) -> bool:
    """Réponse sans recherche ni citation : le modèle a répondu de lui-même."""
    searched = any(call.name == "search_filings" and call.executed for call in result.tool_calls)
    return not searched and not result.answer.citations and result.stop_reason == "final_answer"


def basis(result: AgentResult) -> str:
    """Ce sur quoi le modèle dit s'appuyer (`filings`, `conversation`, `general`)."""
    try:
        return str(json.loads(result.raw_output).get("basis", "general"))
    except (json.JSONDecodeError, AttributeError):
        return "general"


def _numbers(text: str) -> set[str]:
    # Chiffres seuls : « 0,85 » (français) et « 0.85 » (anglais) se retrouvent.
    return {re.sub(r"[,.]", "", n) for n in NUMBER.findall(text)}


def missing_figures(answer: str, history: list[dict[str, Any]]) -> list[str]:
    """Chiffres de la réponse absents de la conversation : une reprise ne doit rien inventer."""
    seen = set().union(*(_numbers(str(m["content"])) for m in history)) if history else set()
    return sorted(_numbers(answer) - seen)


def render(result: AgentResult, history: list[dict[str, Any]] | None = None) -> str:
    """Texte renvoyé à l'utilisateur : réponse, sources vérifiées par le code et statut.

    Une réponse reprise de la conversation n'est pas revérifiée dans les rapports, mais le code
    contrôle que ses chiffres figurent bien dans les échanges précédents.
    """
    history = history or []
    lines = [result.answer.answer]
    if is_general(result):
        if history and basis(result) == "conversation":
            missing = missing_figures(result.answer.answer, history)
            note = (
                "Reprise d'une réponse précédente de la conversation, sans nouvelle recherche ; "
                + (
                    f"attention, chiffres absents de la conversation : {', '.join(missing)}."
                    if missing
                    else f"{REUSED_OK}."
                )
            )
        else:
            note = "Réponse générale, sans recherche dans les rapports."
        lines += ["", f"_{note}_"]
        return "\n".join(lines)
    sources = {source.number: source for source in result.toolbox.registry.sources}
    if result.checks:
        lines += ["", "Sources :"]
        for check in result.checks:
            source = sources.get(check.source)
            where = f"{source.document}, p. {source.pages}" if source else "source inconnue"
            mark = "vérifié" if check.status.value == "verified" else "non retrouvé"
            lines.append(f"[{check.source}] {where} : « {check.quote} » ({mark})")
    lines += ["", f"_Statut : {STATUS_LABELS[result.status]}._"]
    return "\n".join(lines)


def history_messages(items: list[Any]) -> list[dict[str, Any]]:
    """Messages texte des tours précédents (utilisateur et agent), dans l'ordre."""
    messages = []
    for item in items:
        if item.get("type") != "message" or item.get("role") not in ("user", "assistant"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = "".join(part.get("text", "") for part in content or [])
        if text.strip():
            messages.append({"role": item["role"], "content": text})
    return messages
