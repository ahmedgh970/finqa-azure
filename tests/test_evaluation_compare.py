import pytest

from ai103.evaluation.compare import bootstrap_mean_diff, mcnemar_exact, paired, wrong_answers


def q(status="verified", completeness=4.0, numeric=None, response="1,577 [1]"):
    return {
        "type": "metrics-generated",
        "status": status,
        "completeness": completeness,
        "numeric": numeric,
        "ground_truth": "$1577.00",
        "response": response,
    }


def test_mcnemar_counts_only_discordant_questions():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(9, 2) == pytest.approx(0.0654, abs=1e-3)
    assert mcnemar_exact(7, 7) == 1.0


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    diffs = [1.0, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0, 1.0]
    low, high = bootstrap_mean_diff(diffs, samples=2000)
    assert low < sum(diffs) / len(diffs) < high
    assert (low, high) == bootstrap_mean_diff(diffs, samples=2000)


def test_paired_comparison_and_answer_transitions():
    base = {"a": q("insufficient_evidence", 1.0, 0.0), "b": q(completeness=4.0, numeric=1.0)}
    other = {"a": q(completeness=5.0, numeric=1.0), "b": q("insufficient_evidence", 2.0, 0.0)}
    result = paired(base, other)
    assert result["completeness_mean_diff"] == pytest.approx(1.0)
    assert (result["numeric_gains"], result["numeric_losses"]) == (1, 1)
    assert (result["newly_answered"], result["newly_abstained"]) == (1, 1)


def test_wrong_answers_ignore_abstentions():
    questions = {
        "a": q(numeric=0.0),
        "b": q(completeness=2.0),
        "c": q("insufficient_evidence", 1.0, 0.0),
        "d": q(completeness=5.0, numeric=1.0),
    }
    assert [w["question_id"] for w in wrong_answers(questions)] == ["a", "b"]
