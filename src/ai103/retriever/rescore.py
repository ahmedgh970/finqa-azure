"""Relire un rapport ponctuel sauvegardé avec les qrels, sans appel Azure."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ai103.retriever.benchmark import atomic_json, load_questions
from ai103.retriever.core import Chunk
from ai103.retriever.qrels_scoring import ReviewedQrels


def score_report(
    report: dict[str, Any],
    questions: list[dict[str, Any]],
    qrels: ReviewedQrels,
) -> dict[str, Any]:
    question = next((q for q in questions if q["financebench_id"] == report["question_id"]), None)
    if question is None or question["question"] != report["question"]:
        raise ValueError("Question du rapport différente du golden set.")
    fields = (
        "chunk_id",
        "content",
        "source",
        "score",
        "company",
        "fiscal_year",
        "page_start",
        "page_end",
    )
    return {
        "question_id": report["question_id"],
        "requested_k": report.get("k"),
        "azure_calls": False,
        "qrels_provenance": qrels.provenance,
        "results": {
            index: {
                mode: qrels.score(
                    index,
                    report["question_id"],
                    [
                        Chunk(**{name: row[name] for name in fields if name in row})
                        for row in result["chunks"]
                    ],
                )
                for mode, result in modes.items()
            }
            for index, modes in report["results"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("docs/reports/retrieval-comparison.json")
    )
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--qrels-dir", type=Path, default=Path("docs/reports/retrieval-qrels"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/retrieval-qrels/demo-scoring.json")
    )
    args = parser.parse_args()
    questions = load_questions(args.golden)
    qrels = ReviewedQrels.load(
        args.qrels_dir, questions, hashlib.sha256(args.golden.read_bytes()).hexdigest()
    )
    report = json.loads(args.report.read_text())
    if report.get("golden_sha256") != hashlib.sha256(args.golden.read_bytes()).hexdigest():
        raise ValueError("Golden set différent de celui du rapport.")
    result = score_report(report, questions, qrels)
    result["source_report_sha256"] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
