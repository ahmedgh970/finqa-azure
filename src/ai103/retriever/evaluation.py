"""Diagnostics de récupération ; aucun label de pertinence inventé."""

import html
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ai103.metadata_enrichment import doc_name_from_source
from ai103.retriever.core import Chunk

EVALUATION_VERSION = 2


def load_question(path: Path, question_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        matches = [
            row
            for line in stream
            if line.strip()
            if (row := json.loads(line))["financebench_id"] == question_id
        ]
    if len(matches) != 1:
        raise ValueError("L'identifiant doit correspondre à exactement une question.")
    return cast(dict[str, Any], matches[0])


def tokens(text: str) -> list[str]:
    text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    # Conserver nombres complets, signes et parenthèses comptables. Ne pas arrondir.
    return re.findall(r"[+-]?\d+(?:[.,]\d+)*|\w+|[()$%]", text)


def compare(chunks: Sequence[Chunk], question: dict[str, Any]) -> dict[str, Any]:
    evidence = question["evidence"]
    if not evidence or any(not tokens(item["evidence_text"]) for item in evidence):
        raise ValueError("Preuves gold non vides requises.")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunks répétés : classement invalide.")
    gold_docs = {item["doc_name"] for item in evidence}
    ranks = [
        rank
        for rank, chunk in enumerate(chunks, 1)
        if doc_name_from_source(chunk.source) in gold_docs
    ]
    comparisons = []
    for item in evidence:
        gold = tokens(item["evidence_text"])
        gold_set = set(gold)
        matching = [
            chunk for chunk in chunks if doc_name_from_source(chunk.source) == item["doc_name"]
        ]
        retrieved = set().union(*(set(tokens(chunk.content)) for chunk in matching))
        comparisons.append(
            {
                "doc_name": item["doc_name"],
                "gold_page_num_raw": item.get("evidence_page_num"),
                "normalized_exact_in_single_chunk": bool(gold)
                and any(
                    (" " + " ".join(gold) + " ") in (" " + " ".join(tokens(chunk.content)) + " ")
                    for chunk in matching
                ),
                "unique_gold_token_coverage": (
                    len(gold_set & retrieved) / len(gold_set) if gold_set else None
                ),
            }
        )
    return {
        "evaluation_version": EVALUATION_VERSION,
        "document_hit_at_k": bool(ranks),
        "document_reciprocal_rank_at_k": 1 / ranks[0] if ranks else 0.0,
        "gold_document_recall_at_k": len(
            gold_docs & {doc_name_from_source(chunk.source) for chunk in chunks}
        )
        / len(gold_docs)
        if gold_docs
        else None,
        "evidence_diagnostics": comparisons,
        "limitations": "Document hit != evidence hit. Token coverage is diagnostic, not recall. "
        "Gold page numbering is preserved raw and not compared to physical 1-based pages.",
    }
