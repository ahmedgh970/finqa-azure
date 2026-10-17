"""Évaluation FinanceBench exhaustive avec checkpoints locaux par question."""

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import httpx
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from ai103.retriever.core import AzureEmbedder, Retriever, Runner
from ai103.retriever.evaluation import EVALUATION_VERSION, compare
from ai103.retriever.filters import ALIASES, FilterExtractor, Strategy, question_filter
from ai103.retriever.qrels_scoring import ReviewedQrels

KS = (5, 10, 20)
LABELS = {"lexical": "lexical", "vector": "dense", "hybrid": "hybrid"}


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [row["financebench_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Golden set vide ou identifiants répétés.")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Question vide ou invalide.")
        evidence = row.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not item.get("doc_name") or not item.get("evidence_text") for item in evidence)
        ):
            raise ValueError("Chaque question doit posséder des preuves documentées non vides.")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate_question(
    runner: Runner,
    question: dict[str, Any],
    candidates: int = 50,
    qrels: ReviewedQrels | None = None,
    filter_expression: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    results = runner.run(
        question["question"],
        k=max(KS),
        candidates=candidates,
        filter_expression=filter_expression,
    )
    return {
        "question_id": question["financebench_id"],
        "question": question["question"],
        "filter": filter_expression,
        "elapsed_seconds": perf_counter() - started,
        "results": {
            index: {
                LABELS[mode]: {
                    "chunks": [asdict(chunk) for chunk in chunks],
                    "at_k": {
                        str(k): {
                            **compare(chunks[:k], question),
                            **(
                                {
                                    "reviewed_qrels": qrels.score(
                                        index, question["financebench_id"], chunks[:k]
                                    )
                                }
                                if qrels
                                else {}
                            ),
                        }
                        for k in KS
                    },
                }
                for mode, chunks in modes.items()
            }
            for index, modes in results.items()
        },
    }


def aggregate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        for index, modes in record["results"].items():
            for mode, result in modes.items():
                for k, metrics in result["at_k"].items():
                    groups.setdefault((index, mode, k), []).append(metrics)
    rows = []
    for (index, mode, k), values in sorted(groups.items()):
        # Macro-average: each question has equal weight, even with several proofs.
        rows.append(
            {
                "index": index,
                "mode": mode,
                "k": int(k),
                "questions": len(values),
                "document_hit_rate": mean(float(v["document_hit_at_k"]) for v in values),
                "document_first_chunk_mrr_at_k": mean(
                    v["document_reciprocal_rank_at_k"] for v in values
                ),
                "gold_document_recall": mean(v["gold_document_recall_at_k"] for v in values),
                "evidence_exact_fraction_macro": mean(
                    mean(
                        float(e["normalized_exact_in_single_chunk"])
                        for e in v["evidence_diagnostics"]
                    )
                    for v in values
                ),
                "evidence_unique_token_coverage_macro": mean(
                    mean(e["unique_gold_token_coverage"] for e in v["evidence_diagnostics"])
                    for v in values
                ),
            }
        )
    for row in rows:
        values = groups[(row["index"], row["mode"], str(row["k"]))]
        annotations = [v["reviewed_qrels"] for v in values if "reviewed_qrels" in v]
        eligible = [v for v in annotations if v["annotation_available"]]
        row.update(
            {
                "qrels_annotated_questions": len(eligible),
                "qrels_insufficient_judgments_questions": sum(
                    v["retrieval_judgment"] == "insufficient_judgments" for v in annotations
                ),
                "qrels_confirmed_irrelevant_questions": sum(
                    v["retrieval_judgment"] == "verified_irrelevant" for v in annotations
                ),
                "qrels_sufficient_annotation_fraction_macro": mean(
                    v["sufficient_annotation_fraction"] for v in eligible
                )
                if eligible
                else None,
                "qrels_unannotated_questions": len(values) - len(eligible),
                "qrels_known_useful_hit_rate": mean(
                    float(v["known_useful_hit_at_k"]) for v in eligible
                )
                if eligible
                else None,
                "qrels_known_useful_mrr": mean(v["known_useful_rr_at_k"] for v in eligible)
                if eligible
                else None,
                "qrels_verified_evidence_coverage_lower_bound_macro": mean(
                    v["verified_evidence_coverage_lower_bound"] for v in eligible
                )
                if eligible
                else None,
                "qrels_all_evidence_verified_rate": mean(
                    float(v["all_gold_evidence_verified_at_k"]) for v in eligible
                )
                if eligible
                else None,
                "qrels_all_evidence_completion_mrr": mean(
                    v["all_evidence_completion_rr_at_k"] for v in eligible
                )
                if eligible
                else None,
                "qrels_judged_result_fraction_micro": sum(v["judged_chunks"] for v in annotations)
                / sum(v["returned_chunks"] for v in annotations)
                if sum(v["returned_chunks"] for v in annotations)
                else None,
            }
        )
    return rows


def run_benchmark(
    runner: Runner,
    questions: Sequence[dict[str, Any]],
    output: Path,
    protocol: dict[str, Any],
    *,
    resume: bool = False,
    candidates: int = 50,
    usage: Callable[[], dict[str, Any]] | None = None,
    qrels: ReviewedQrels | None = None,
    filter_for: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    protocol = {**protocol, "reviewed_qrels": qrels.provenance if qrels else None}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != protocol:
            raise ValueError("Dossier existant : --resume et protocole identique requis.")
    else:
        if any(output.iterdir()):
            raise ValueError("Choisir un dossier vide.")
        atomic_json(manifest, protocol)
    records: list[dict[str, Any]] = []
    for position, question in enumerate(questions, 1):
        # Hash avoids using untrusted question IDs as filesystem paths.
        filename = hashlib.sha256(question["financebench_id"].encode()).hexdigest() + ".json"
        path = output / filename
        if path.exists():
            record = json.loads(path.read_text())
            if record["question_id"] != question["financebench_id"]:
                raise ValueError("Checkpoint incohérent.")
        else:
            try:
                # Le filtre ne dépend que du texte de la question, jamais des réponses gold.
                expression = filter_for(question["question"]) if filter_for else None
                record = evaluate_question(runner, question, candidates, qrels, expression)
                record["embedding_usage"] = usage() if usage else None
                atomic_json(path, record)
            except Exception as exc:
                atomic_json(
                    output / "status.json",
                    {
                        "status": "failed",
                        "completed": len(records),
                        "total": len(questions),
                        "failed_question": question["financebench_id"],
                        "error_type": type(exc).__name__,
                    },
                )
                raise
        records.append(record)
        print(f"[{position}/{len(questions)}] {question['financebench_id']}", flush=True)
    summary: dict[str, Any] = {
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
        "questions": len(records),
        "protocol": protocol,
        "metrics": aggregate(records),
        "limitations": "Document metrics are not evidence relevance labels. Exact inclusion "
        "and token coverage are diagnostics, not proof recall or financial correctness. "
        "k=5/10 are prefixes of top20; vector candidates fixed. No Azure evaluators or LLM judge.",
    }
    atomic_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["metrics"][0]))
        writer.writeheader()
        writer.writerows(summary["metrics"])
    atomic_json(output / "status.json", {"status": "completed", "completed": len(records)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/reports/retrieval-benchmark"))
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=[
            "financial-chunks-v1",
            "financial-chunks-layout-v1",
        ],
    )
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qrels-dir", type=Path, default=Path("docs/reports/retrieval-qrels"))
    parser.add_argument(
        "--filter-strategy", type=Strategy, choices=list(Strategy), default=Strategy.NONE
    )
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    args = parser.parse_args()
    if not 20 <= args.candidates <= 1000 or len(set(args.indexes)) != len(args.indexes):
        parser.error("20 <= candidates <= 1000 et index distincts requis")
    questions = load_questions(args.golden)
    if len(questions) != 150:
        parser.error("Ce protocole exige les 150 questions originales.")
    qrels = ReviewedQrels.load(
        args.qrels_dir, questions, hashlib.sha256(args.golden.read_bytes()).hexdigest()
    )
    load_dotenv()
    for name in ("AZURE_SEARCH_SERVICE_ENDPOINT", "AZURE_OPENAI_ENDPOINT"):
        if not os.environ.get(name, "").strip():
            parser.error(f"Configurer {name} dans .env.")
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    protocol = {
        "version": 3,
        "evaluation_version": EVALUATION_VERSION,
        "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        "indexes": args.indexes,
        "ks": list(KS),
        "modes": list(LABELS.values()),
        "candidates": args.candidates,
        "dimensions": 1024,
        "deployment": deployment,
        "search_endpoint": os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
        "embedding_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "filter": None,
    }
    filter_for: Callable[[str], str | None] | None = None
    if args.filter_strategy is not Strategy.NONE:
        filter_for = question_filter(
            FilterExtractor.from_metadata(args.metadata), args.filter_strategy
        )
        protocol["filter"] = {
            "strategy": args.filter_strategy.value,
            "extractor": "rules-v1",
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "aliases_sha256": hashlib.sha256(
                json.dumps(ALIASES, sort_keys=True).encode()
            ).hexdigest(),
            "vector_filter_mode": "preFilter",
        }
    with ExitStack() as stack:
        credential = stack.enter_context(DefaultAzureCredential())
        embed = AzureEmbedder(
            stack.enter_context(httpx.Client(timeout=30)),
            credential,
            os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment,
        )
        clients = {
            name: Retriever(
                stack.enter_context(
                    SearchClient(
                        endpoint=os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                        index_name=name,
                        credential=credential,
                    )
                )
            )
            for name in args.indexes
        }
        summary = run_benchmark(
            Runner(clients, embed),
            questions,
            args.output,
            protocol,
            resume=args.resume,
            candidates=args.candidates,
            usage=lambda: dict(embed.last_usage),
            qrels=qrels,
            filter_for=filter_for,
        )
    print(f"{summary['questions']} questions évaluées. Résultats : {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
