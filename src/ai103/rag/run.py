"""Passer N questions du golden set dans le RAG, avec journal, traces et plafond de coût.

Seul le texte des questions est envoyé ; réponses et preuves gold ne sont pas lues ici.
"""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from ai103.rag.cli import (
    add_pipeline_arguments,
    build_pipeline,
    check_environment,
    pipeline_settings,
    report,
)
from ai103.rag.pipeline import RagPipeline
from ai103.retriever.benchmark import atomic_json, load_questions


def percentile_95(values: list[float]) -> float:
    return values[0] if len(values) == 1 else quantiles(values, n=20, method="inclusive")[-1]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in records if "error" not in r]
    stages = sorted({name for r in answered for name in r["timings"]})
    return {
        "questions": len(records),
        "errors": len(records) - len(answered),
        "status": dict(Counter(r["status"] for r in answered)),
        "cost_usd": sum(r["cost_usd"] for r in answered),
        "tokens": {
            name: sum((r["usage"] or {}).get(name, 0) for r in answered)
            for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
        },
        "latency_seconds": {
            stage: {
                "median": median(values),
                "p95": percentile_95(values),
            }
            for stage in stages
            if (values := [r["timings"][stage] for r in answered if stage in r["timings"]])
        },
    }


def run(
    pipeline: RagPipeline,
    questions: list[dict[str, Any]],
    output: Path,
    settings: dict[str, Any],
    max_cost_usd: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    spent = 0.0
    stopped = None
    with (output / "results.jsonl").open("w", encoding="utf-8") as journal:
        for position, question in enumerate(questions, 1):
            if spent >= max_cost_usd:
                stopped = f"Plafond de {max_cost_usd} USD atteint après {len(records)} questions."
                break
            record: dict[str, Any] = {"question_id": question["financebench_id"]}
            try:
                record |= report(pipeline.answer(question["question"]), settings)
                spent += record["cost_usd"]
            except Exception as exc:
                # Une erreur (réseau, filtre de contenu…) est journalisée sans arrêter le lot.
                record |= {"question": question["question"], "error": type(exc).__name__}
                record["error_message"] = str(exc)[:500]
            records.append(record)
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            status = record.get("status", record.get("error"))
            print(f"[{position}/{len(questions)}] {question['financebench_id']} {status}")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "max_cost_usd": max_cost_usd,
        "stopped": stopped,
        **summarize(records),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_arguments(parser)
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-cost-usd", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Dossier vide ; par défaut horodaté.")
    args = parser.parse_args()
    check_environment(parser, args)
    if not 1 <= args.limit <= 150 or args.max_cost_usd <= 0:
        parser.error("1 <= limit <= 150 et max-cost-usd > 0 requis")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("docs/reports/rag/runs") / stamp
    if output.exists() and any(output.iterdir()):
        parser.error("Choisir un dossier vide.")
    output.mkdir(parents=True, exist_ok=True)
    args.trace = args.trace or output / "traces.jsonl"
    questions = load_questions(args.golden)[: args.limit]
    settings = {**pipeline_settings(args), "limit": args.limit}
    with ExitStack() as stack:
        pipeline, credential = build_pipeline(stack, args)
        summary = run(pipeline, questions, output, settings, args.max_cost_usd)
    summary["entra_tokens"] = {"calls": credential.calls, "seconds": credential.seconds}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "cost_usd", "latency_seconds")}, indent=2))
    print(f"Jetons Entra : {credential.calls} obtentions, {credential.seconds:.2f} s")
    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()
