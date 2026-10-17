"""Contre-exemples : calcul juste != preuve financière suffisante."""

import pytest

from ai103.retriever import Chunk
from ai103.retriever.evaluation import compare


def chunk(text, doc="DOC", key="a"):
    return Chunk(key, text, f"https://s.blob.core.windows.net/c/{doc}.pdf", 1)


def gold(*texts):
    return {"evidence": [{"doc_name": "DOC", "evidence_text": text} for text in texts]}


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("cash 1234", "cash 123"),
        ("cash 123", "cash 1234"),
        ("loss 123", "loss -123"),
        ("loss 123", "loss (123)"),
        ("cash 1.23", "cash 1.2"),
        ("margin 12", "margin 12%"),
    ],
)
def test_exact_inclusion_rejects_numeric_false_positives(actual, expected):
    result = compare([chunk(actual)], gold(expected))["evidence_diagnostics"][0]
    assert not result["normalized_exact_in_single_chunk"]
    assert result["unique_gold_token_coverage"] < 1


def test_unicode_negative_and_html_keep_matching():
    result = compare([chunk("<td>LOSS</td><td>−123</td>")], gold("loss -123"))
    assert result["evidence_diagnostics"][0]["normalized_exact_in_single_chunk"]


def test_wrong_passage_from_correct_pdf_is_document_hit_only():
    result = compare([chunk("unrelated section")], gold("cash 123"))
    assert result["document_hit_at_k"]
    assert result["document_reciprocal_rank_at_k"] == 1
    assert not result["evidence_diagnostics"][0]["normalized_exact_in_single_chunk"]
    assert result["evidence_diagnostics"][0]["unique_gold_token_coverage"] == 0


def test_table_columns_permuted_still_have_full_token_coverage():
    result = compare([chunk("2022 200 2023 100")], gold("2022 100 2023 200"))
    diag = result["evidence_diagnostics"][0]
    assert diag["unique_gold_token_coverage"] == 1  # démontre la limite de ce diagnostic
    assert not diag["normalized_exact_in_single_chunk"]


def test_multiple_evidence_and_split_chunks_are_not_all_exact():
    result = compare(
        [chunk("cash 123"), chunk("debt", key="b"), chunk("456", key="c")],
        gold("cash 123", "debt 456"),
    )
    assert result["gold_document_recall_at_k"] == 1
    assert [e["normalized_exact_in_single_chunk"] for e in result["evidence_diagnostics"]] == [
        True,
        False,
    ]
    assert all(e["unique_gold_token_coverage"] == 1 for e in result["evidence_diagnostics"])


def test_document_denominator_and_first_chunk_rank():
    question = {
        "evidence": [
            {"doc_name": "DOC", "evidence_text": "cash"},
            {"doc_name": "SECOND", "evidence_text": "debt"},
        ]
    }
    result = compare(
        [chunk("wrong", doc="OTHER"), chunk("cash", key="b"), chunk("cash", key="c")], question
    )
    assert result["gold_document_recall_at_k"] == 0.5
    assert result["document_reciprocal_rank_at_k"] == 0.5


def test_empty_results_are_misses():
    result = compare([], gold("cash"))
    assert result["document_reciprocal_rank_at_k"] == 0
    assert result["gold_document_recall_at_k"] == 0


def test_rejects_duplicate_chunk_ids():
    with pytest.raises(ValueError):
        compare([chunk("cash"), chunk("cash")], gold("cash"))
