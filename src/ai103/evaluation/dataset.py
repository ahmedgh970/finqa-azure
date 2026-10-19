"""Construire le jeu d'évaluation à partir d'une campagne RAG et du golden set.

La réponse gold n'est lue qu'ici, après la génération : elle n'a jamais été vue par le RAG.
"""

import json
from pathlib import Path
from typing import Any

from ai103.rag.context import build_context, number_sources
from ai103.retriever.core import Chunk


def results_file(run: Path) -> Path:
    """Les résultats revérifiés hors ligne, s'ils existent, priment sur l'original."""
    rechecked = run / "results.rechecked.jsonl"
    return rechecked if rechecked.exists() else run / "results.jsonl"


def rag_statuses(run: Path) -> dict[str, str]:
    lines = results_file(run).read_text(encoding="utf-8").splitlines()
    return {r["question_id"]: r["status"] for r in map(json.loads, lines) if "status" in r}


def load_golden(path: Path) -> dict[str, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {row["financebench_id"]: row for row in map(json.loads, filter(None, lines))}


def evaluation_rows(
    results: Path, golden: dict[str, dict[str, Any]], limit: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Lignes évaluables et nombre de questions sans réponse (erreur de génération)."""
    rows, skipped = [], 0
    for line in results.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "error" in record:
            skipped += 1
            continue
        gold = golden[record["question_id"]]
        sources = number_sources([Chunk(**source["chunk"]) for source in record["sources"]])
        rows.append(
            {
                "question_id": record["question_id"],
                "question_type": gold["question_type"],
                "query": record["question"],
                "response": record["answer"]["answer"],
                # Le même contexte que celui donné au générateur.
                "context": build_context(sources) if sources else "No source retrieved.",
                "ground_truth": str(gold["answer"]),
                "rag_status": record["status"],
                "rag_cost_usd": record["cost_usd"],
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows, skipped


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
