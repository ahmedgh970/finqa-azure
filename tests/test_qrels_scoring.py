import csv
import gzip
import hashlib
import json

import pytest

from ai103.retriever import Chunk
from ai103.retriever.benchmark import aggregate, evaluate_question, run_benchmark
from ai103.retriever.qrels import build, export_reviewed
from ai103.retriever.qrels_scoring import ReviewedQrels

SOURCE = "https://s.blob.core.windows.net/c/DOC.pdf"


def chunk(key, content="cash"):
    return Chunk(key, content, SOURCE, 1)


def judgment(cid, eid="q:e0", grade=2):
    return {
        "chunk_id": cid,
        "relevance": grade,
        "source": SOURCE,
        "content_sha256": hashlib.sha256(b"cash").hexdigest(),
        "evidence_judgments": [{"evidence_id": eid, "relevance": grade}],
    }


def scorer():
    pairs = {"a": judgment("a", grade=1), "b": judgment("b", grade=1), "c": judgment("c", "q:e1")}
    group = {"evidence_id": "q:e0", "members": [{"chunk_id": "a"}, {"chunk_id": "b"}]}
    return ReviewedQrels(
        {("idx", "q"): pairs},
        {("idx", "q"): [group]},
        {"q": {"q:e0", "q:e1"}, "unjudged": {"unjudged:e0"}},
        {("idx", cid): (row["content_sha256"], SOURCE) for cid, row in pairs.items()},
        {"version": 1},
    )


def test_group_needs_all_members_and_preserves_real_ranks():
    scores = scorer()
    first = scores.score("idx", "q", [chunk("x"), chunk("a"), chunk("c")])
    assert first["known_useful_rr_at_k"] == 0.5
    assert first["verified_evidence_coverage_lower_bound"] == 0.5
    assert not first["all_gold_evidence_verified_at_k"]
    assert first["unjudged_chunks"] == 1
    complete = scores.score("idx", "q", [chunk("x"), chunk("a"), chunk("c"), chunk("b")])
    assert complete["verified_evidence_coverage_lower_bound"] == 1
    assert complete["all_evidence_completion_rr_at_k"] == 0.25
    assert complete["evidence_completion_ranks"] == {"q:e1": 3, "q:e0": 4}


def test_unannotated_is_none_not_zero_and_unknowns_not_negative():
    result = scorer().score("idx", "unjudged", [chunk("x")])
    assert result["known_useful_hit_at_k"] is None
    assert result["verified_evidence_coverage_lower_bound"] is None
    assert result["judged_chunks"] == 0
    assert result["unjudged_chunks"] == 1


def test_alternative_sufficient_chunk_and_group_count_proof_once():
    scores = scorer()
    scores.pairs[("idx", "q")]["d"] = judgment("d")
    result = scores.score("idx", "q", [chunk("d"), chunk("a"), chunk("b")])
    assert result["verified_evidence_count"] == 1
    assert result["evidence_completion_ranks"]["q:e0"] == 1


@pytest.mark.parametrize("content,source", [("changed", SOURCE), ("cash", SOURCE + "?secret=1")])
def test_content_and_source_drift_refused(content, source):
    with pytest.raises(ValueError, match="Contenu/source"):
        scorer().score("idx", "q", [Chunk("a", content, source, 1)])


def test_duplicate_results_refused():
    with pytest.raises(ValueError):
        scorer().score("idx", "q", [chunk("a"), chunk("a")])


def make_files(tmp_path):
    q = {
        "financebench_id": "q",
        "question": "cash?",
        "evidence": [{"doc_name": "DOC", "evidence_text": "cash"}],
    }
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(q) + "\n")
    snapshot = tmp_path / "snapshot.json.gz"
    with gzip.open(snapshot, "wt") as stream:
        json.dump(
            {
                "indexes": [
                    {
                        "index_name": "idx",
                        "chunk_count": 1,
                        "chunks": [{"chunk_id": "a", "source": SOURCE, "content": "cash"}],
                    }
                ]
            },
            stream,
        )
    output = tmp_path / "labels"
    build(gold, snapshot, output)
    review = output / "review.csv"
    with review.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(reviewed_grade="2", reviewer="test", notes="cash fact checked")
    with review.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export_reviewed(output, review)
    return output, [q], hashlib.sha256(gold.read_bytes()).hexdigest()


def test_loader_and_benchmark_integration(tmp_path):
    from unittest.mock import Mock

    directory, questions, digest = make_files(tmp_path)
    scores = ReviewedQrels.load(directory, questions, digest)
    runner = Mock()
    runner.run.return_value = {"idx": {"lexical": [chunk("a")]}}
    record = evaluate_question(runner, questions[0], qrels=scores)
    assert (
        record["results"]["idx"]["lexical"]["at_k"]["5"]["reviewed_qrels"][
            "verified_evidence_count"
        ]
        == 1
    )
    row = aggregate([record])[0]
    assert row["qrels_annotated_questions"] == 1
    assert row["qrels_known_useful_mrr"] == 1
    assert row["qrels_judged_result_fraction_micro"] == 1


def test_loader_rejects_automatic_labels_and_changed_gold(tmp_path):
    directory, questions, digest = make_files(tmp_path)
    with pytest.raises(ValueError, match="golden"):
        ReviewedQrels.load(directory, questions, "changed")
    path = directory / "qrels.reviewed.jsonl"
    row = json.loads(path.read_text())
    row["judgment_status"] = "automatic_exact_text"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="explicites"):
        ReviewedQrels.load(directory, questions, digest)


def test_loader_rejects_group_without_reviewed_useful_members(tmp_path):
    directory, questions, digest = make_files(tmp_path)
    path = directory / "qrels.reviewed.jsonl"
    member = json.loads(path.read_text())
    group = {
        "index": "idx",
        "question_id": "q",
        "group_id": "g",
        "evidence_id": "q:e0",
        "judgment_status": "reviewed_group",
        "relevance": 2,
        "rule": "all_members_required",
        "reviewer": "test",
        "notes": "checked",
        "members": [member, {**member, "chunk_id": "unknown"}],
    }
    (directory / "qrels.groups.reviewed.jsonl").write_text(json.dumps(group))
    with pytest.raises(ValueError):
        ReviewedQrels.load(directory, questions, digest)


def test_resume_refused_when_judgments_change(tmp_path):
    from unittest.mock import Mock

    directory, questions, digest = make_files(tmp_path)
    scores = ReviewedQrels.load(directory, questions, digest)
    runner = Mock()
    runner.run.return_value = {"idx": {"lexical": [chunk("a")]}}
    output = tmp_path / "run"
    run_benchmark(runner, questions, output, {}, qrels=scores)
    scores.provenance["reviewed_sha256"] = "changed"
    with pytest.raises(ValueError):
        run_benchmark(runner, questions, output, {}, qrels=scores, resume=True)


def test_all_unknown_results_are_not_confirmed_failure():
    result = scorer().score("idx", "q", [chunk("unknown")])
    assert result["retrieval_judgment"] == "insufficient_judgments"
    assert result["known_useful_hit_at_k"] is False  # no known success, not proven irrelevant


def test_all_judged_negative_results_are_explicitly_irrelevant():
    scores = scorer()
    negative = judgment("x", grade=0)
    negative["evidence_judgments"].append({"evidence_id": "q:e1", "relevance": 0})
    scores.pairs[("idx", "q")]["x"] = negative
    result = scores.score("idx", "q", [chunk("x")])
    assert result["retrieval_judgment"] == "verified_irrelevant"


def test_aggregation_reports_unannotated_questions_separately():
    from ai103.retriever.evaluation import compare

    question = {"evidence": [{"doc_name": "DOC", "evidence_text": "cash"}]}
    records = []
    for qid in ("q", "unjudged"):
        metrics = {
            **compare([chunk("c")], question),
            "reviewed_qrels": scorer().score("idx", qid, [chunk("c")]),
        }
        records.append({"results": {"idx": {"dense": {"at_k": {"5": metrics}}}}})
    result = aggregate(records)[0]
    assert result["qrels_annotated_questions"] == 1
    assert result["qrels_unannotated_questions"] == 1
    assert result["qrels_known_useful_mrr"] == 1
    assert result["qrels_judged_result_fraction_micro"] == 0.5


def test_saved_report_scoring_is_offline_and_validates_question():
    from dataclasses import asdict

    from ai103.retriever.rescore import score_report

    questions = [{"financebench_id": "q", "question": "cash?"}]
    report = {
        "question_id": "q",
        "question": "cash?",
        "k": 5,
        "results": {"idx": {"dense": {"chunks": [asdict(chunk("c"))]}}},
    }
    result = score_report(report, questions, scorer())
    assert result["azure_calls"] is False
    assert result["requested_k"] == 5
    report["question"] = "changed"
    with pytest.raises(ValueError):
        score_report(report, questions, scorer())
