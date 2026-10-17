"""CLI de comparaison ponctuelle FinanceBench, sans mutation Azure."""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from ai103.retriever.core import AzureEmbedder, Mode, Retriever, Runner
from ai103.retriever.evaluation import compare, load_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", default="financebench_id_03029")
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=[
            "financial-chunks-v1",
            "financial-chunks-layout-v1",
        ],
    )
    parser.add_argument("--modes", nargs="+", choices=list(Mode), default=list(Mode))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/retrieval-comparison.json")
    )
    args = parser.parse_args()
    if not 1 <= args.k <= args.candidates <= 1000:
        parser.error("1 <= k <= candidates <= 1000 requis")
    load_dotenv()
    question = load_question(args.golden, args.question_id)
    endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "").strip()
    if not endpoint:
        parser.error("Configurer AZURE_SEARCH_SERVICE_ENDPOINT dans .env.")
    if (
        any(mode != Mode.LEXICAL for mode in args.modes)
        and not os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    ):
        parser.error("Configurer AZURE_OPENAI_ENDPOINT dans .env.")
    with ExitStack() as stack:
        credential = stack.enter_context(DefaultAzureCredential())
        embedder = None
        if any(mode != Mode.LEXICAL for mode in args.modes):
            embedder = AzureEmbedder(
                stack.enter_context(httpx.Client(timeout=30)),
                credential,
                os.environ["AZURE_OPENAI_ENDPOINT"],
                os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            )
        retrievers = {
            name: Retriever(
                stack.enter_context(
                    SearchClient(
                        endpoint=endpoint,
                        index_name=name,
                        credential=credential,
                    )
                )
            )
            for name in args.indexes
        }
        started = perf_counter()
        results = Runner(retrievers, embedder).run(
            question["question"],
            modes=args.modes,
            k=args.k,
            candidates=args.candidates,
        )
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
            "question_id": args.question_id,
            "question": question["question"],
            "k": args.k,
            "vector_candidates": args.candidates,
            "dimensions": 1024,
            "filter": None,
            "elapsed_seconds": perf_counter() - started,
            "embedding_deployment": embedder.deployment if embedder else None,
            "embedding_usage": embedder.last_usage if embedder else None,
            "gold_evidence": question["evidence"],
            "results": {
                name: {
                    mode: {
                        "chunks": [asdict(chunk) for chunk in chunks],
                        "comparison": compare(chunks, question),
                    }
                    for mode, chunks in modes.items()
                }
                for name, modes in results.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, modes in report["results"].items():
        for mode, result in modes.items():
            print(name, mode, result["comparison"])
    print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()
