from unittest.mock import Mock

import pytest

from ai103.retriever import Chunk
from ai103.retriever.benchmark import aggregate, evaluate_question, load_questions, run_benchmark


def question():
    return {
        "financebench_id": "q1",
        "question": "Where is cash?",
        "evidence": [{"doc_name": "DOC", "evidence_text": "cash 123"}],
    }


def runner():
    fake = Mock()
    chunks = [
        Chunk(str(i), "irrelevant", "https://s.blob.core.windows.net/c/OTHER.pdf", 1)
        for i in range(20)
    ]
    chunks[7] = Chunk("gold", "cash 123", "https://s.blob.core.windows.net/c/DOC.pdf", 1)
    fake.run.return_value = {"index": {mode: chunks for mode in ("lexical", "vector", "hybrid")}}
    return fake


def test_prefixes_and_aggregation():
    fake = runner()
    record = evaluate_question(fake, question())
    fake.run.assert_called_once_with("Where is cash?", k=20, candidates=50, filter_expression=None)
    assert record["filter"] is None
    result = record["results"]["index"]["dense"]["at_k"]
    assert not result["5"]["document_hit_at_k"]
    assert result["10"]["document_reciprocal_rank_at_k"] == 1 / 8
    rows = aggregate([record, record])
    assert len(rows) == 9
    dense10 = next(r for r in rows if r["mode"] == "dense" and r["k"] == 10)
    assert dense10["questions"] == 2
    assert dense10["document_hit_rate"] == 1
    assert dense10["evidence_exact_fraction_macro"] == 1


def test_resume_does_not_repeat_completed_questions(tmp_path):
    fake = runner()
    protocol = {"version": 1}
    run_benchmark(fake, [question()], tmp_path, protocol)
    fake.reset_mock()
    run_benchmark(fake, [question()], tmp_path, protocol, resume=True)
    fake.run.assert_not_called()
    assert (tmp_path / "summary.csv").exists()
    with pytest.raises(ValueError):
        run_benchmark(fake, [question()], tmp_path, {"version": 2}, resume=True)


def test_failure_saved_and_can_resume(tmp_path):
    fake = runner()
    original = fake.run.return_value
    fake.run.side_effect = RuntimeError("network")
    with pytest.raises(RuntimeError):
        run_benchmark(fake, [question()], tmp_path, {})
    import json

    assert json.loads((tmp_path / "status.json").read_text())["completed"] == 0
    assert not (tmp_path / "summary.json").exists()
    fake.run.side_effect = None
    fake.run.return_value = original
    assert run_benchmark(fake, [question()], tmp_path, {}, resume=True)["questions"] == 1


def test_rejects_duplicate_gold_ids(tmp_path):
    import json

    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(question()) + "\n" + json.dumps(question()))
    with pytest.raises(ValueError):
        load_questions(path)


def test_macro_average_weights_questions_equally():
    from ai103.retriever.benchmark import aggregate
    from ai103.retriever.evaluation import compare

    source = "https://s.blob.core.windows.net/c/DOC.pdf"
    first = {"evidence": [{"doc_name": "DOC", "evidence_text": "cash"}]}
    second = {
        "evidence": [
            {"doc_name": "DOC", "evidence_text": "cash"},
            {"doc_name": "DOC", "evidence_text": "debt"},
            {"doc_name": "DOC", "evidence_text": "profit"},
        ]
    }
    metrics = [compare([Chunk("a", "cash", source, 1)], q) for q in (first, second)]
    records = [{"results": {"index": {"dense": {"at_k": {"5": value}}}}} for value in metrics]
    row = aggregate(records)[0]
    assert row["evidence_exact_fraction_macro"] == pytest.approx((1 + 1 / 3) / 2)
    assert row["evidence_unique_token_coverage_macro"] == pytest.approx((1 + 1 / 3) / 2)


def test_filter_built_from_question_text_only(tmp_path):
    fake = runner()
    seen = []

    def filter_for(text):
        seen.append(text)
        return "company eq 'DOC'"

    run_benchmark(fake, [question()], tmp_path, {}, filter_for=filter_for)
    assert seen == ["Where is cash?"]
    fake.run.assert_called_once_with(
        "Where is cash?", k=20, candidates=50, filter_expression="company eq 'DOC'"
    )
    checkpoint = next(p for p in tmp_path.glob("*.json") if len(p.stem) == 64)
    assert "company eq 'DOC'" in checkpoint.read_text()
