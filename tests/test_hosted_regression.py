import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

from ai103.hosted import regression
from ai103.hosted.chat import STATUS_LABELS
from ai103.rag.guardrails import AnswerStatus


def a2_row(question_id, status="verified", answer="Capex was $1,577 million [1].", seconds=10.0):
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "status": status,
        "answer": {"answer": answer},
        "timings": {"total_seconds": seconds},
        "cost_usd": 0.02,
    }


GOLDEN = {
    "q1": {"answer": "$1577.00"},
    "q2": {"answer": "$1577.00"},
    "q3": {"answer": "Yes, the company grew."},
    "q4": {"answer": "Revenue declined."},
}


def test_sample_is_fixed_and_half_numeric():
    a2 = {qid: a2_row(qid) for qid in GOLDEN}
    first = regression.sample(a2, GOLDEN, size=2, seed=7)
    assert first == regression.sample(a2, GOLDEN, size=2, seed=7)
    assert sum(1 for qid in first if qid in ("q1", "q2")) == 1


def test_hosted_status_is_read_from_the_rendered_answer():
    verified = STATUS_LABELS[AnswerStatus.VERIFIED]
    assert regression.hosted_status(f"Réponse.\n\n_Statut : {verified}._") == "verified"
    assert regression.hosted_status("Bonjour.\n\n_Réponse générale, sans recherche…_") == "general"
    assert regression.hosted_status("Texte sans statut") == "unknown"


def test_score_lists_regressions_and_improvements():
    a2 = {
        "q1": a2_row("q1"),
        "q2": a2_row("q2", status="insufficient_evidence", answer="Not found."),
    }
    rows = [
        # q1 : vérifiée par A2, l'agent hébergé n'a plus le bon chiffre → régression.
        {
            "question_id": "q1",
            "status": "verified",
            "text": "Capex was $2,000 million.",
            "seconds": 20,
        },
        # q2 : A2 sans réponse, l'agent hébergé trouve le chiffre → amélioration.
        {
            "question_id": "q2",
            "status": "verified",
            "text": "Capex was $1,577 million.",
            "seconds": 30,
        },
    ]
    summary = regression.score(rows, a2, GOLDEN)
    assert summary["regressions"] == ["q1"] and summary["improvements"] == ["q2"]
    assert summary["numeric"] == {"applicable": 2, "a2_exact": 1, "hosted_exact": 1}
    assert summary["status"]["same"] == 1 and summary["seconds"]["hosted_median"] == 25


def test_an_error_counts_as_a_regression_without_stopping_the_score():
    summary = regression.score(
        [{"question_id": "q1", "error": "APIStatusError: 424"}], {"q1": a2_row("q1")}, GOLDEN
    )
    assert summary["errors"] == 1 and summary["regressions"] == ["q1"]


def test_simulation_makes_no_call(tmp_path, monkeypatch, capsys):
    a2_dir = tmp_path / "a2"
    a2_dir.mkdir()
    lines = [json.dumps(a2_row(qid)) for qid in GOLDEN]
    (a2_dir / "results.jsonl").write_text("\n".join(lines) + "\n")
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "".join(json.dumps({"financebench_id": k, **v}) + "\n" for k, v in GOLDEN.items())
    )
    factory = Mock()
    monkeypatch.setattr(regression, "agent_client", factory)
    args = argparse.Namespace(
        a2=a2_dir, golden=golden, size=2, seed=1, output=tmp_path / "out", apply=False
    )
    regression.run(args)
    factory.assert_not_called()
    assert not (tmp_path / "out").exists() and "Simulation" in capsys.readouterr().out

    # Avec --apply : une conversation neuve par question (aucun previous_response_id).
    client = Mock()
    verified = STATUS_LABELS[AnswerStatus.VERIFIED]
    client.responses.create.return_value = SimpleNamespace(
        id="caresp_1", output_text=f"Capex was $1,577 million.\n\n_Statut : {verified}._"
    )
    monkeypatch.setattr(regression, "agent_client", lambda args: client)
    regression.run(argparse.Namespace(**{**vars(args), "apply": True}))
    assert all(
        "previous_response_id" not in c.kwargs for c in client.responses.create.call_args_list
    )
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["questions"] == 2 and summary["errors"] == 0
    assert len((tmp_path / "out" / "results.jsonl").read_text().splitlines()) == 2
