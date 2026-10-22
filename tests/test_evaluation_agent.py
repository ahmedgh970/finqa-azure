import json
from unittest.mock import Mock

import pytest

from ai103.evaluation.agent_eval import (
    AGENT_JUDGES,
    agent_evaluators,
    agent_rows,
    conversation,
    passages_by_call,
    summarize,
    tool_definitions,
)
from ai103.evaluation.judge import model_config

ARGS = {"query": "balance sheet", "company": "3M", "fiscal_years": [2023], "k": 5}


def source(number, text):
    chunk = {"chunk_id": f"c{number}", "content": text, "source": "s", "score": 1.0}
    return {"number": number, "document": "3M_2023Q2_10Q", "pages": "4", "chunk": chunk}


def record(qid="q1", status="verified", insufficient=False, per_call=False):
    search = {
        "round": 2,
        "name": "search_filings",
        "arguments": json.dumps(ARGS),
        "seconds": 1.0,
        "executed": True,
        "summary": "2 passages, filter=company eq '3M'",
    }
    if per_call:
        search["sources"] = [2, 1]
    return {
        "question_id": qid,
        "question": "What is 3M's quick ratio?",
        "status": status,
        "answer": {
            "answer": "0.85x [1].",
            "citations": [{"source": 1, "quote": "Total current liabilities 10,936"}],
            "insufficient_evidence": insufficient,
        },
        "sources": [source(1, "Total current liabilities 10,936"), source(2, "Cash 4,258")],
        "tool_calls": [
            {
                "round": 1,
                "name": "list_filings",
                "arguments": '{"company": "3M"}',
                "seconds": 0.1,
                "executed": True,
                "summary": "3 filings for 3M",
            },
            search,
            {
                "round": 2,
                "name": "calculate",
                "arguments": '{"expression": "1 + 1"}',
                "seconds": 0.0,
                "executed": False,
                "summary": "error",
            },
        ],
    }


def test_conversation_uses_converter_format_grouped_by_round():
    query, response, exact = conversation(record())
    assert query[0]["role"] == "system" and "financial analyst agent" in query[0]["content"]
    assert query[1]["content"][0]["text"] == "What is 3M's quick ratio?"
    assistant_turns = [m for m in response if m["role"] == "assistant"]
    assert len(assistant_turns) == 3  # tour 1, tour 2 (2 appels parallèles), réponse finale
    second = assistant_turns[1]["content"]
    assert [c["name"] for c in second] == ["search_filings", "calculate"]
    assert second[0] == {
        "type": "tool_call",
        "tool_call_id": "call_2",
        "name": "search_filings",
        "arguments": ARGS,
    }
    tool_messages = [m for m in response if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2", "call_3"]
    assert "budget exhausted" in tool_messages[2]["content"][0]["tool_result"]
    # Le juge voit la réponse réelle : texte, citations et extraits.
    final = json.loads(response[-1]["content"][0]["text"])
    assert final["citations"][0]["quote"] == "Total current liabilities 10,936"
    # Le texte des passages renvoyés par la recherche est fourni au juge.
    found = json.loads(tool_messages[1]["content"][0]["tool_result"])
    assert [p["text"] for p in found["passages"]] == [
        "Total current liabilities 10,936",
        "Cash 4,258",
    ]
    assert found["filter"] == "company eq '3M'" and exact


def test_abstention_is_visible_to_the_judge():
    _, response, _ = conversation(record(insufficient=True))
    assert json.loads(response[-1]["content"][0]["text"])["insufficient_evidence"] is True


def test_recorded_per_call_sources_are_used_and_light_version_has_no_text():
    _, response, exact = conversation(record(per_call=True))
    found = json.loads([m for m in response if m["role"] == "tool"][1]["content"][0]["tool_result"])
    assert [p["source"] for p in found["passages"]] == [2, 1] and exact
    _, light, _ = conversation(record(), with_passages=False)
    assert [m for m in light if m["role"] == "tool"][1]["content"][0]["tool_result"].startswith(
        "2 passages"
    )


def test_duplicate_passages_make_the_reconstruction_approximate():
    duplicated = record()
    duplicated["tool_calls"].append(dict(duplicated["tool_calls"][1], round=3))
    assert passages_by_call(duplicated) == ({2: [1, 2], 4: []}, False)


def test_sdk_parser_recognizes_our_tool_calls_without_network():
    evaluators = agent_evaluators(model_config("https://res.services.ai.azure.com", "j"), Mock())
    accuracy = evaluators["tool_call_accuracy"]
    _, response, _ = conversation(record())
    calls = accuracy._parse_tools_from_response(response)
    assert [c["name"] for c in calls] == ["list_filings", "search_filings", "calculate"]
    assert "Cash 4,258" in calls[1]["tool_result"]
    needed = accuracy._extract_needed_tool_definitions(calls, tool_definitions(), None)
    assert {d["name"] for d in needed} == {"list_filings", "search_filings", "calculate"}


def test_rows_skip_generation_errors_and_honor_limit(tmp_path):
    lines = [record("q1"), {"question_id": "q2", "error": "RuntimeError"}, record("q3")]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    rows = agent_rows(tmp_path)
    assert [r["question_id"] for r in rows] == ["q1", "q3"]
    assert rows[0]["reconstruction"] == "exact" and "response_light" in rows[0]
    assert len(agent_rows(tmp_path, limit=1)) == 1
    assert {d["name"] for d in rows[0]["tool_definitions"]} == {
        "list_filings",
        "search_filings",
        "calculate",
    }


def row(status, scores, result="pass"):
    data = {"inputs.question_id": status + "-q", "inputs.status": status}
    for (name, key), score in zip(AGENT_JUDGES.items(), scores, strict=True):
        data[f"outputs.{name}.{key}"] = score
        data[f"outputs.{name}.{key}_result"] = result
        data[f"outputs.{name}.{key}_reason"] = f"reason {name}"
        data[f"outputs.{name}.{key}_properties"] = {"prompt_tokens": 1000, "completion_tokens": 100}
    return data


def test_summary_by_status_handles_nan_and_not_applicable():
    rows = [
        row("verified", (5, 1, 5)),
        row("verified", (3, 0, 4), result="fail"),
        row("limit_reached", (float("nan"), float("nan"), 1), result="not_applicable"),
    ]
    summary = summarize(rows)
    overall = summary["overall"]
    assert overall["tool_call_accuracy"]["mean"] == pytest.approx(4)
    assert overall["tool_call_accuracy"]["scored"] == 2
    assert overall["tool_call_accuracy"]["not_applicable"] == 1
    assert overall["intent_resolution"]["pass_rate"] == pytest.approx(0.5)
    assert set(summary["by_status"]) == {"verified", "limit_reached"}
    assert summary["lowest"][0]["score"] == 0
    assert summary["judge_tokens"] == {"input": 9000, "output": 900}
