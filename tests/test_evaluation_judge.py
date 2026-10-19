import json
from unittest.mock import Mock

import pytest

from ai103.evaluation.cli import summarize
from ai103.evaluation.dataset import evaluation_rows
from ai103.evaluation.judge import NumericMatchEvaluator, evaluators, model_config

SOURCE = "https://s/financial-pdfs/3M_2018_10K.pdf"


def record(qid, answer="Capex was 1,577 [1].", status="verified"):
    chunk = {"chunk_id": "c1", "content": "PP&E (1,577)", "source": SOURCE, "score": 1.0}
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "status": status,
        "answer": {"answer": answer, "citations": [], "insufficient_evidence": False},
        "sources": [{"chunk": chunk}],
        "cost_usd": 0.01,
    }


def test_rows_join_gold_after_generation_and_skip_errors(tmp_path):
    results = tmp_path / "results.jsonl"
    lines = [record("q1"), {"question_id": "q2", "error": "RuntimeError"}, record("q3")]
    results.write_text("".join(json.dumps(line) + "\n" for line in lines))
    golden = {
        "q1": {"answer": "$1577.00", "question_type": "metrics-generated"},
        "q3": {"answer": "Yes", "question_type": "domain-relevant"},
    }
    rows, skipped = evaluation_rows(results, golden)
    assert skipped == 1 and [r["question_id"] for r in rows] == ["q1", "q3"]
    assert rows[0]["ground_truth"] == "$1577.00"
    assert '<source id="1"' in rows[0]["context"] and "PP&E (1,577)" in rows[0]["context"]
    assert evaluation_rows(results, golden, limit=1)[0][0]["question_id"] == "q1"


def test_numeric_evaluator_is_not_applicable_to_text_answers():
    numeric = NumericMatchEvaluator()
    assert numeric(response="Capex was 1,577 [1].", ground_truth="$1577.00")["numeric_match"] == 1.0
    assert numeric(response="Capex was 1,373 [1].", ground_truth="$1577.00")["numeric_match"] == 0.0
    assert numeric(response="Yes [1].", ground_truth="Yes, stable")["numeric_match"] is None


def test_evaluators_use_a_reasoning_judge_without_calling_azure():
    built = evaluators(model_config("https://res.services.ai.azure.com", "gpt-5-mini"), Mock())
    assert set(built) == {"groundedness", "relevance", "completeness", "numeric"}


def row(qtype, g, r, c, numeric, status="verified", tokens=1000):
    data = {"inputs.question_type": qtype, "inputs.rag_status": status, "inputs.rag_cost_usd": 0.01}
    keys = (("groundedness", "groundedness", g), ("relevance", "relevance", r))
    for name, key, score in (*keys, ("completeness", "response_completeness", c)):
        data[f"outputs.{name}.{key}"] = score
        data[f"outputs.{name}.{key}_result"] = "pass" if score >= 3 else "fail"
        data[f"outputs.{name}.{key}_properties"] = {
            "prompt_tokens": tokens,
            "completion_tokens": 100,
        }
    data["outputs.numeric.numeric_match"] = numeric
    return data


def test_summary_means_pass_rates_numeric_accuracy_and_judge_cost():
    rows = [
        row("metrics-generated", 5, 4, 5, 1.0),
        row("metrics-generated", 3, 2, 1, 0.0, status="insufficient_evidence"),
        # Le SDK convertit le None de l'évaluateur numérique en NaN.
        row("domain-relevant", 4, 5, float("nan"), float("nan")),
    ]
    rows[2]["outputs.completeness.response_completeness_properties"] = float("nan")
    summary = summarize(rows)
    overall = summary["overall"]
    assert overall["groundedness"]["mean"] == pytest.approx(4)
    assert overall["relevance"]["pass_rate"] == pytest.approx(2 / 3)
    assert overall["completeness"]["scored"] == 2  # NaN = évaluation manquante, exclue
    assert overall["completeness"]["pass_rate"] == pytest.approx(1 / 2)
    assert overall["numeric_accuracy"] == {"mean": 0.5, "applicable": 2}
    assert summary["by_question_type"]["domain-relevant"]["numeric_accuracy"]["applicable"] == 0
    assert summary["rag_status"] == {"verified": 2, "insufficient_evidence": 1}
    assert summary["judge_tokens"] == {"input": 8000, "output": 800}
    assert summary["judge_cost_usd_upper_bound"] == pytest.approx((8000 * 0.45 + 800 * 3.6) / 1e6)
