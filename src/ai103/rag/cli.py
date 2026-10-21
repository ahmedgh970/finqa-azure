"""Poser une question au RAG : retrieval filtré, contexte numéroté et réponse citée."""

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from ai103.observability import TracedCredential, configure_file_tracing
from ai103.rag.generator import AzureGenerator, openai_client
from ai103.rag.pipeline import RagPipeline, RagResult
from ai103.retriever.core import AzureEmbedder, Chunk, Mode, Retriever
from ai103.retriever.filters import FilterExtractor, Strategy, question_filter

REQUIRED = (
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
)


def report(result: RagResult, settings: dict[str, Any]) -> dict[str, Any]:
    generation = result.generation
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "question": result.question,
        "filter": result.filter_expression,
        "status": result.status.value,
        "content_filter": result.content_filter,
        "answer": result.answer.model_dump(),
        "citation_checks": [asdict(check) for check in result.checks],
        "sources": [
            {
                "number": source.number,
                "verified": source.number in result.verified_sources,
                "document": source.document,
                "pages": source.pages,
                "chunk": asdict(source.chunk),
            }
            for source in result.sources
        ],
        "timings": result.timings,
        "usage": asdict(generation.usage) if generation else None,
        "cost_usd": generation.usage.cost_usd if generation else 0.0,
        "model": generation.model if generation else None,
    }


def injection_chunk(text: str) -> Chunk:
    """Source fictive, clairement étiquetée, pour tester la résistance à l'injection."""
    return Chunk("injection-test", text, "https://test.invalid/injection-test.pdf", 0.0)


def show(result: RagResult) -> None:
    print(f"Filtre : {result.filter_expression or 'aucun'}")
    print(f"\nRéponse :\n{result.answer.answer}")
    print(f"\nStatut (calculé par le code) : {result.status.value}")
    if result.content_filter:
        print(f"Détail du filtre : {json.dumps(result.content_filter, ensure_ascii=False)[:800]}")
    for check in result.checks:
        print(f"  [{check.source}] {check.status.value} : « {check.quote} »")
    print("\nSources fournies (* = extrait vérifié) :")
    for source in result.sources:
        mark = "*" if source.number in result.verified_sources else " "
        print(f" {mark}[{source.number}] {source.document}, pages {source.pages}")
    timings = " | ".join(f"{name} {value:.2f} s" for name, value in result.timings.items())
    print(f"\nDurées : {timings}")
    if result.generation:
        usage = result.generation.usage
        print(
            f"Tokens : entrée {usage.input_tokens} (cache {usage.cached_input_tokens}), "
            f"sortie {usage.output_tokens} (raisonnement {usage.reasoning_tokens}) | "
            f"coût {usage.cost_usd:.4f} USD"
        )


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default="financial-chunks-v1")
    parser.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.VECTOR)
    parser.add_argument(
        "--filter-strategy",
        type=Strategy,
        choices=list(Strategy),
        default=Strategy.COMPANY_YEAR_TOLERANT,
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--pool", type=int, help="semantic_hybrid : reclasser seulement les N premiers hybrides."
    )
    parser.add_argument("--effort", default="low", choices=["none", "minimal", "low", "medium"])
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    parser.add_argument("--trace", type=Path, help="Ajouter les spans OpenTelemetry à ce JSONL.")
    parser.add_argument(
        "--semantic-configuration",
        help="Configuration sémantique de l'index (mode semantic_hybrid). Par défaut : "
        "AZURE_SEARCH_SEMANTIC_CONFIGURATION, sinon finance-semantic.",
    )


def check_environment(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.k <= 50:
        parser.error("1 <= k <= 50 requis")
    load_dotenv()
    if missing := [name for name in REQUIRED if not os.environ.get(name, "").strip()]:
        parser.error(f"Configurer dans .env : {', '.join(missing)}")
    # Résolu après load_dotenv : la valeur de .env doit être visible.
    args.semantic_configuration = args.semantic_configuration or os.environ.get(
        "AZURE_SEARCH_SEMANTIC_CONFIGURATION", "finance-semantic"
    )


def pipeline_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "index": args.index,
        "mode": args.mode.value,
        "filter_strategy": args.filter_strategy.value,
        "k": args.k,
        "pool": args.pool,
        "reasoning_effort": args.effort,
        "deployment": os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        "semantic_configuration": (
            args.semantic_configuration if args.mode is Mode.SEMANTIC_HYBRID else None
        ),
    }


def build_pipeline(
    stack: ExitStack, args: argparse.Namespace
) -> tuple[RagPipeline, TracedCredential]:
    """Clients Azure fermés par `stack` ; l'identité est chronométrée pour les traces."""
    if args.trace:
        provider, stream = configure_file_tracing(args.trace)
        stack.callback(stream.close)
        stack.callback(provider.shutdown)
    filter_for = None
    if args.filter_strategy is not Strategy.NONE:
        filter_for = question_filter(
            FilterExtractor.from_metadata(args.metadata), args.filter_strategy
        )
    credential = TracedCredential(stack.enter_context(DefaultAzureCredential()))
    embedder = AzureEmbedder(
        stack.enter_context(httpx.Client(timeout=30)),
        credential,
        os.environ["AZURE_OPENAI_ENDPOINT"],
        os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    )
    search = stack.enter_context(
        SearchClient(
            endpoint=os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
            index_name=args.index,
            credential=credential,
        )
    )
    client = stack.enter_context(openai_client(os.environ["AZURE_OPENAI_ENDPOINT"], credential))
    generator = AzureGenerator(
        client, os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], reasoning_effort=args.effort
    )
    retriever = Retriever(search, semantic_configuration=args.semantic_configuration)
    pipeline = RagPipeline(
        retriever,
        embedder,
        generator,
        filter_for=filter_for,
        mode=args.mode,
        k=args.k,
        pool=args.pool,
    )
    return pipeline, credential


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    add_pipeline_arguments(parser)
    parser.add_argument("--output", type=Path, help="Enregistrer le résultat complet en JSON.")
    parser.add_argument(
        "--inject-text",
        help="Test de robustesse : ajoute une source fictive contenant ce texte.",
    )
    args = parser.parse_args()
    check_environment(parser, args)
    settings = {**pipeline_settings(args), "inject_text": args.inject_text}
    extra = [injection_chunk(args.inject_text)] if args.inject_text else []
    with ExitStack() as stack:
        pipeline, credential = build_pipeline(stack, args)
        result = pipeline.answer(args.question, extra)
    show(result)
    print(f"Jetons Entra : {credential.calls} obtentions, {credential.seconds:.2f} s au total")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report(result, settings), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()
