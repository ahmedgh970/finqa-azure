"""Chaîne RAG : filtre, retrieval vectoriel, contexte numéroté, génération et vérification."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from ai103.observability import stage, tracer
from ai103.rag.context import Source, build_context, number_sources
from ai103.rag.generator import AzureGenerator, ContentFilterError, Generation
from ai103.rag.guardrails import (
    AnswerStatus,
    CheckStatus,
    CitationCheck,
    answer_status,
    check_citations,
)
from ai103.rag.prompts import SYSTEM_PROMPT, RagAnswer, user_message
from ai103.retriever.core import Chunk, Mode, Retriever

NO_SOURCE_ANSWER = "No source was retrieved for this question; it cannot be answered."
BLOCKED_ANSWER = "The request was blocked by the content safety filter; no answer was generated."


@dataclass(frozen=True)
class RagResult:
    question: str
    filter_expression: str | None
    sources: list[Source]
    answer: RagAnswer
    checks: list[CitationCheck]
    status: AnswerStatus
    timings: dict[str, float]
    generation: Generation[RagAnswer] | None = None
    # Détail renvoyé par Azure lorsqu'un garde-fou bloque la requête.
    content_filter: object = None

    @property
    def verified_sources(self) -> set[int]:
        return {c.source for c in self.checks if c.status is CheckStatus.VERIFIED}


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        embed: Callable[[str], list[float]],
        generator: AzureGenerator,
        *,
        filter_for: Callable[[str], str | None] | None = None,
        mode: Mode = Mode.VECTOR,
        k: int = 5,
        candidates: int = 50,
    ) -> None:
        if not 1 <= k <= candidates:
            raise ValueError("1 <= k <= candidates requis.")
        self.retriever = retriever
        self.embed = embed
        self.generator = generator
        self.filter_for = filter_for
        self.mode = Mode(mode)
        self.k = k
        self.candidates = candidates

    def answer(self, question: str, extra_chunks: Sequence[Chunk] = ()) -> RagResult:
        """`extra_chunks` sert uniquement aux tests de robustesse (injection simulée)."""
        if not question.strip():
            raise ValueError("Question vide.")
        started = perf_counter()
        timings: dict[str, float] = {}
        with tracer.start_as_current_span("rag.answer") as root:
            # Le filtre ne dépend que du texte de la question.
            with stage("filter", timings) as span:
                expression = self.filter_for(question) if self.filter_for else None
                span.set_attribute("ai103.filter", expression or "")
            vector = None
            if self.mode is not Mode.LEXICAL:
                with stage("embedding", timings, **{"gen_ai.operation.name": "embeddings"}):
                    vector = self.embed(question)
            with stage("search", timings, **{"ai103.mode": self.mode.value, "ai103.k": self.k}):
                chunks = self.retriever.retrieve(
                    question,
                    mode=self.mode,
                    k=self.k,
                    vector=vector,
                    candidates=self.candidates,
                    filter_expression=expression,
                )
            sources = number_sources([*chunks, *extra_chunks])
            root.set_attribute("ai103.sources", len(sources))
            if not sources:
                # Aucun appel au modèle : rien sur quoi ancrer une réponse.
                empty = RagAnswer(answer=NO_SOURCE_ANSWER, citations=[], insufficient_evidence=True)
                status = AnswerStatus.INSUFFICIENT_EVIDENCE
                root.set_attribute("ai103.status", status.value)
                timings["total_seconds"] = perf_counter() - started
                return RagResult(question, expression, [], empty, [], status, timings)
            with stage("generation", timings, **{"gen_ai.operation.name": "chat"}) as span:
                try:
                    generation = self.generator.generate(
                        SYSTEM_PROMPT, user_message(question, build_context(sources)), RagAnswer
                    )
                except ContentFilterError as exc:
                    status = AnswerStatus.BLOCKED_BY_CONTENT_FILTER
                    span.set_attribute("ai103.content_filter", str(exc))
                    root.set_attribute("ai103.status", status.value)
                    blocked = RagAnswer(
                        answer=BLOCKED_ANSWER, citations=[], insufficient_evidence=True
                    )
                    timings["total_seconds"] = perf_counter() - started
                    return RagResult(
                        question,
                        expression,
                        sources,
                        blocked,
                        [],
                        status,
                        timings,
                        content_filter=exc.details or str(exc),
                    )
                usage = generation.usage
                span.set_attributes(
                    {
                        "gen_ai.provider.name": "azure.ai.openai",
                        "gen_ai.response.model": generation.model,
                        "gen_ai.response.id": generation.response_id,
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.cache_read.input_tokens": usage.cached_input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                        "gen_ai.usage.reasoning.output_tokens": usage.reasoning_tokens,
                        "ai103.cost_usd": usage.cost_usd,
                    }
                )
            checks = check_citations(generation.output, sources)
            status = answer_status(generation.output, checks)
            root.set_attribute("ai103.status", status.value)
        timings["total_seconds"] = perf_counter() - started
        return RagResult(
            question=question,
            filter_expression=expression,
            sources=sources,
            answer=generation.output,
            checks=checks,
            status=status,
            generation=generation,
            timings=timings,
        )
