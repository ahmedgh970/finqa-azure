import csv
import gzip
import json

import pytest

from ai103.retriever.qrels import build, export_reviewed, prepare_snapshot, propose


def question():
    return {
        "financebench_id": "q1",
        "question": "cash?",
        "evidence": [
            {"doc_name": "DOC", "evidence_text": "cash -123", "evidence_page_num": 0},
        ],
    }


def snapshot(texts):
    return {
        "indexes": [
            {
                "index_name": "native",
                "chunk_count": len(texts),
                "chunks": [
                    {
                        "chunk_id": f"c{i}",
                        "source": "https://s.blob.core.windows.net/c/DOC.pdf",
                        "content": text,
                    }
                    for i, text in enumerate(texts)
                ],
            }
        ]
    }


def fixture_files(tmp_path, q=None):
    golden = tmp_path / "gold.jsonl"
    golden.write_text(json.dumps(q or question()) + "\n")
    chunks = tmp_path / "snapshot.json.gz"
    with gzip.open(chunks, "wt") as stream:
        json.dump(snapshot(["cash -123", "cash 123"]), stream)
    output = tmp_path / "qrels"
    return golden, chunks, output


def test_exact_only_automatic_grade_and_no_numeric_false_positive():
    passages = prepare_snapshot(snapshot(["cash -123", "cash 123", "cash -1234"]))
    rows = propose(question(), 0, passages["native"]["DOC"])
    assert rows[0]["automatic_grade"] == 2
    assert all(row["automatic_grade"] is None for row in rows[1:])


def test_keep_all_exact_even_above_limit():
    passages = prepare_snapshot(snapshot(["cash -123"] * 3))
    assert len(propose(question(), 0, passages["native"]["DOC"], limit=1)) == 3


def test_wrong_document_refused():
    passages = prepare_snapshot(snapshot(["cash -123"]))
    q = question()
    q["evidence"][0]["doc_name"] = "OTHER"
    with pytest.raises(ValueError):
        propose(q, 0, passages["native"]["DOC"])


def test_build_preserves_inputs_and_review_requires_explicit_grade(tmp_path):
    golden, chunks, output = fixture_files(tmp_path)
    before = golden.read_bytes(), chunks.read_bytes()
    summary = build(golden, chunks, output)
    assert summary["automatic_pairs"] == 1
    assert not (output / "qrels.reviewed.jsonl").exists()
    assert export_reviewed(output, output / "review.csv") == 0
    assert before == (golden.read_bytes(), chunks.read_bytes())
    with pytest.raises(ValueError):
        build(golden, chunks, output)


def fill_review(output, grade="2", reviewer="test", notes="checked full evidence"):
    path = output / "review.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(reviewed_grade=grade, reviewer=reviewer, notes=notes)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_export_reviewed_binds_content(tmp_path):
    golden, chunks, output = fixture_files(tmp_path)
    build(golden, chunks, output)
    assert export_reviewed(output, fill_review(output)) == 1
    rows = [json.loads(line) for line in (output / "qrels.reviewed.jsonl").read_text().splitlines()]
    assert rows[0]["judgment_status"] == "reviewed"
    assert rows[0]["evidence_judgments"][0]["reviewer"] == "test"
    with (output / "candidates.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError):
        export_reviewed(output, output / "review.csv")


@pytest.mark.parametrize(
    "grade,reviewer,notes", [("3", "test", "notes"), ("2", "", "notes"), ("2", "test", "")]
)
def test_invalid_reviews_rejected(tmp_path, grade, reviewer, notes):
    golden, chunks, output = fixture_files(tmp_path)
    build(golden, chunks, output)
    with pytest.raises(ValueError):
        export_reviewed(output, fill_review(output, grade, reviewer, notes))


def test_no_question_negative_until_all_evidence_judged(tmp_path):
    q = question()
    q["evidence"].append({"doc_name": "DOC", "evidence_text": "other evidence"})
    golden, chunks, output = fixture_files(tmp_path, q)
    build(golden, chunks, output)
    assert export_reviewed(output, fill_review(output, "0")) == 0
