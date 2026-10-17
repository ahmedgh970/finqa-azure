"""Évaluer hors ligne l'extracteur de filtres, sans appel Azure.

Le golden set sert uniquement à noter l'extracteur ; seule la question est extraite.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai103.retriever.benchmark import atomic_json, load_questions
from ai103.retriever.filters import FilterExtractor, Strategy, filter_years, odata_filter


def admissible(strategy: Strategy, filters: Any, document: dict[str, Any]) -> bool:
    """Le document gold passe-t-il le filtre construit pour cette stratégie ?"""
    if odata_filter(filters, strategy) is None:
        return True
    if filters.company != document["company"]:
        return False
    if years := filter_years(filters, strategy):
        return document["fiscal_year"] is None or document["fiscal_year"] in years
    return True


def evaluate(
    questions: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], extractor: FilterExtractor
) -> dict[str, Any]:
    rows = []
    for question in questions:
        document = metadata[question["doc_name"]]
        filters = extractor.extract(question["question"])
        if filters.company is None:
            company = "ambiguous" if filters.matched_companies else "none"
        else:
            company = "correct" if filters.company == document["company"] else "wrong"
        rows.append(
            {
                "question_id": question["financebench_id"],
                "question": question["question"],
                "gold_doc_name": question["doc_name"],
                "gold_company": document["company"],
                "gold_fiscal_year": document["fiscal_year"],
                "extracted": asdict(filters),
                "company_outcome": company,
                "years_include_gold_fiscal_year": document["fiscal_year"] in filters.years
                if filters.years and document["fiscal_year"] is not None
                else None,
                "filters": {s.value: odata_filter(filters, s) for s in Strategy},
                "gold_document_admissible": {
                    s.value: admissible(s, filters, document) for s in Strategy
                },
            }
        )
    filtered = {s.value: sum(row["filters"][s.value] is not None for row in rows) for s in Strategy}
    return {
        "company_outcomes": dict(Counter(row["company_outcome"] for row in rows)),
        "questions_with_years": sum(bool(row["extracted"]["years"]) for row in rows),
        "years_include_gold_fiscal_year": dict(
            Counter(str(row["years_include_gold_fiscal_year"]) for row in rows)
        ),
        "filtered_questions": filtered,
        "gold_document_excluded": {
            s.value: sum(not row["gold_document_admissible"][s.value] for row in rows)
            for s in Strategy
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/retrieval-filters/extraction.json")
    )
    args = parser.parse_args()
    questions = load_questions(args.golden)
    lines = args.metadata.read_text(encoding="utf-8").splitlines()
    metadata = {row["doc_name"]: row for row in map(json.loads, filter(None, lines))}
    if missing := {q["doc_name"] for q in questions} - set(metadata):
        parser.error(f"Documents gold absents du référentiel : {sorted(missing)}")
    extractor = FilterExtractor((r["company"] for r in metadata.values()), strict=True)
    result = evaluate(questions, metadata, extractor)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "azure_calls": False,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    for key in ("company_outcomes", "filtered_questions", "gold_document_excluded"):
        print(key, report[key])
    print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()
