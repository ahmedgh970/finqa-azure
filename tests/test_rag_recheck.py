import json

from ai103.evaluation.dataset import rag_statuses, results_file
from ai103.rag.recheck import recheck_run


def record(qid, quote, status="unsupported"):
    chunk = {
        "chunk_id": "c1",
        "content": "Item\xa08.01 Other Events. Revenue 1,577.",
        "source": "https://s/c/AMCOR.pdf",
        "score": 1.0,
    }
    return {
        "question_id": qid,
        "status": status,
        "answer": {
            "answer": "Item 8.01 [1].",
            "citations": [{"source": 1, "quote": quote}],
            "insufficient_evidence": False,
        },
        "sources": [{"number": 1, "chunk": chunk, "verified": False}],
    }


def test_recheck_upgrades_artifact_quotes_and_keeps_original(tmp_path):
    lines = [
        record("q1", "Item\x008.01 Other Events."),
        record("q2", "Revenue 1,578."),
        {"question_id": "q3", "error": "RuntimeError"},
    ]
    original = "".join(json.dumps(line) + "\n" for line in lines)
    (tmp_path / "results.jsonl").write_text(original)
    summary = recheck_run(tmp_path)
    assert (tmp_path / "results.jsonl").read_text() == original
    assert summary["changed"] == [
        {"question_id": "q1", "before": "unsupported", "after": "verified"}
    ]
    assert results_file(tmp_path).name == "results.rechecked.jsonl"
    assert rag_statuses(tmp_path) == {"q1": "verified", "q2": "unsupported"}
