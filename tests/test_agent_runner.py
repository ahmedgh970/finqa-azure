import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest
from openai.types.responses import ResponseUsage

from ai103.agent.definition import AGENT_NAME, ANSWER_SCHEMA, agent_definition
from ai103.agent.runner import AgentRunner, agent_record
from ai103.agent.tools import Filing, ToolBox
from ai103.rag.guardrails import AnswerStatus
from ai103.retriever.core import Chunk

CHUNK = Chunk(
    "c1",
    "Property, plant and equipment, net 8,738",
    "https://s/financial-pdfs/3M_2018_10K.pdf",
    1.0,
    "3M",
    2018,
    50,
    50,
)


def usage(tokens=1000):
    return ResponseUsage.model_validate(
        {
            "input_tokens": tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 100,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": tokens + 100,
        }
    )


def call(name="search_filings", call_id="call_1", **arguments):
    arguments = arguments or {"query": "net PP&E", "company": "3M", "fiscal_years": [2018], "k": 5}
    return SimpleNamespace(
        type="function_call", name=name, call_id=call_id, arguments=json.dumps(arguments)
    )


def reply(output, text="", tokens=1000, rid="resp"):
    return SimpleNamespace(id=rid, output=output, output_text=text, usage=usage(tokens))


FINAL = json.dumps(
    {
        "answer": "Net PP&E was 8,738 million [1].",
        "citations": [{"source": 1, "quote": "Property, plant and equipment, net 8,738"}],
        "insufficient_evidence": False,
    }
)


def runner(responses, **limits):
    client = Mock()
    client.conversations.create.return_value = SimpleNamespace(id="conv_1")
    client.responses.create.side_effect = responses
    retriever = Mock()
    retriever.retrieve.return_value = [CHUNK]
    filings = [Filing("3M_2018_10K", "3M", 2018)]
    agent = AgentRunner(
        client, lambda: ToolBox(retriever, lambda q: [0.1] * 1024, filings), **limits
    )
    return agent, client, retriever


def test_tool_loop_then_verified_final_answer():
    agent, client, retriever = runner([reply([call()]), reply([], FINAL, rid="resp2")])
    result = agent.answer("What is 3M net PP&E in FY2018?")
    assert result.status is AnswerStatus.VERIFIED and result.stop_reason == "final_answer"
    assert result.rounds == 2 and result.response_ids == ["resp", "resp2"]
    second = client.responses.create.call_args_list[1].kwargs
    assert second["conversation"] == "conv_1"
    assert second["input"][0]["type"] == "function_call_output"
    assert second["input"][0]["call_id"] == "call_1"
    assert retriever.retrieve.call_args.kwargs["filter_expression"] == (
        "company eq '3M' and (fiscal_year eq 2018 or fiscal_year eq null)"
    )
    assert result.usage.input_tokens == 2000
    client.conversations.delete.assert_called_once_with("conv_1")
    assert result.tool_calls[0].sources == [1]
    record = agent_record(result, {"agent": AGENT_NAME})
    assert record["sources"][0]["verified"] and record["tool_calls"][0]["executed"]
    assert record["cost_usd"] == pytest.approx(result.usage.cost_usd)


def test_search_budget_does_not_block_calculations():
    # Cas réel Activision : 3 recherches parallèles, puis 3 calculs ; seul le budget de
    # recherches doit limiter les recherches.
    searches = [call(call_id=f"s{i}") for i in range(3)]
    sums = [call("calculate", f"c{i}", expression=f"{i} + 1") for i in range(3)]
    agent, client, _ = runner(
        [reply(searches), reply(sums), reply([], FINAL)], max_searches=2, max_tool_calls=12
    )
    result = agent.answer("Question?")
    executed = {(c.name, c.executed) for c in result.tool_calls}
    assert [c.executed for c in result.tool_calls if c.name == "search_filings"] == [
        True,
        True,
        False,
    ]
    assert all(c.executed for c in result.tool_calls if c.name == "calculate")
    sent = client.responses.create.call_args_list[1].kwargs["input"]
    assert "Tool budget exhausted" in sent[2]["output"]
    assert ("calculate", True) in executed


def test_total_tool_budget_is_enforced_by_code():
    calls = [call("calculate", f"c{i}", expression="1 + 1") for i in range(3)]
    agent, _, _ = runner([reply(calls), reply([], FINAL)], max_searches=1, max_tool_calls=2)
    result = agent.answer("Question?")
    assert [c.executed for c in result.tool_calls] == [True, True, False]


def test_round_limit_stops_the_loop():
    agent, client, _ = runner([reply([call(call_id=str(i))]) for i in range(5)], max_rounds=3)
    result = agent.answer("Question?")
    assert result.status is AnswerStatus.LIMIT_REACHED and result.stop_reason == "round_limit"
    assert client.responses.create.call_count == 3
    client.conversations.delete.assert_called_once()


def test_cost_limit_stops_the_loop():
    agent, _, _ = runner([reply([call()], tokens=200_000)], max_cost_usd=0.10)
    result = agent.answer("Question?")
    assert result.stop_reason == "cost_limit" and result.status is AnswerStatus.LIMIT_REACHED


def test_refusal_or_free_text_is_a_status_not_a_crash():
    agent, _, _ = runner([reply([], "I'm sorry, but I cannot assist with that request.")])
    result = agent.answer("Question?")
    assert result.status is AnswerStatus.INVALID_OUTPUT
    assert "cannot assist" in result.answer.answer


def test_content_filter_block_cleans_up_the_conversation():
    request = httpx.Request("POST", "https://x/openai/v1/responses")
    blocked = openai.BadRequestError(
        "blocked", response=httpx.Response(400, request=request), body={"code": "content_filter"}
    )
    agent, client, _ = runner([blocked])
    result = agent.answer("Question?")
    assert result.status is AnswerStatus.BLOCKED_BY_CONTENT_FILTER
    assert result.content_filter == {"code": "content_filter"}
    client.conversations.delete.assert_called_once_with("conv_1")


def test_definition_sends_tool_contracts_and_strict_answer_schema():
    definition = agent_definition("gpt-5.2").as_dict()
    assert definition["kind"] == "prompt" and definition["model"] == "gpt-5.2"
    assert [tool["name"] for tool in definition["tools"]] == [
        "list_filings",
        "search_filings",
        "calculate",
    ]
    assert all(tool["strict"] for tool in definition["tools"])
    assert definition["text"]["format"]["schema"] == ANSWER_SCHEMA
    assert "Absence from the passages is not a zero" in definition["instructions"]


def test_monitor_snapshot_of_a_running_campaign():
    from ai103.agent.monitor import snapshot

    record = {
        "status": "verified",
        "stop_reason": "final_answer",
        "rounds": 3,
        "cost_usd": 0.03,
        "timings": {"total_seconds": 20.0},
        "tool_calls": [
            {"name": "search_filings", "executed": True},
            {"name": "calculate", "executed": False},
        ],
    }
    text = snapshot([record, {"question_id": "q2", "error": "RuntimeError"}])
    assert "erreurs : 1" in text and "appels refusés : 1" in text and "0.03 USD au total" in text
