"""Revérifier hors ligne les extraits cités d'une campagne avec les garde-fous actuels.

Aucun appel Azure : réponses et chunks sont relus depuis `results.jsonl`, qui reste intact.
Le résultat est écrit dans `results.rechecked.jsonl`, puis préféré par l'évaluation.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai103.rag.context import number_sources
from ai103.rag.guardrails import CheckStatus, answer_status, check_citations
from ai103.rag.prompts import RagAnswer
from ai103.retriever.benchmark import atomic_json
from ai103.retriever.core import Chunk

RECHECKED = "results.rechecked.jsonl"
KEEP = {"blocked_by_content_filter", "limit_reached", "invalid_output"}


def recheck_record(record: dict[str, Any]) -> dict[str, Any]:
    # Blocage, arrêt sur limite ou sortie non conforme : aucune réponse à revérifier.
    if "error" in record or record.get("status") in KEEP:
        return record
    answer = RagAnswer.model_validate(record["answer"])
    sources = number_sources([Chunk(**source["chunk"]) for source in record["sources"]])
    checks = check_citations(answer, sources)
    verified = {check.source for check in checks if check.status is CheckStatus.VERIFIED}
    return {
        **record,
        "status_before_recheck": record["status"],
        "status": answer_status(answer, checks).value if sources else record["status"],
        "citation_checks": [asdict(check) for check in checks],
        "sources": [
            {**source, "verified": source["number"] in verified} for source in record["sources"]
        ],
    }


def recheck_run(run: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rechecked = [recheck_record(record) for record in records]
    (run / RECHECKED).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rechecked), encoding="utf-8"
    )
    answered = [r for r in rechecked if "error" not in r]
    summary = {
        "before": dict(Counter(r.get("status_before_recheck", r["status"]) for r in answered)),
        "after": dict(Counter(r["status"] for r in answered)),
        "changed": [
            {
                "question_id": r["question_id"],
                "before": r["status_before_recheck"],
                "after": r["status"],
            }
            for r in answered
            if r.get("status_before_recheck") not in (None, r["status"])
        ],
    }
    atomic_json(run / "recheck.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summary = recheck_run(args.run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
