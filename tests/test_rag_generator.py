from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.core.credentials import AccessToken
from openai.types.responses import ResponseUsage
from pydantic import BaseModel

from ai103.rag.generator import (
    AzureGenerator,
    ContentFilterError,
    GenerationError,
    Usage,
    openai_client,
)


class Answer(BaseModel):
    answer: str


def usage(cached=0, reasoning=0):
    return ResponseUsage.model_validate(
        {
            "input_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens": 100_000,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": 1_100_000,
        }
    )


def response(status="completed", parsed=None, incomplete=None):
    return SimpleNamespace(
        status=status,
        output_parsed=parsed,
        incomplete_details=incomplete,
        usage=usage(reasoning=40_000),
        id="resp_1",
        model="gpt-5.2-2025-12-11",
    )


def test_cost_counts_cached_input_and_reasoning_as_output():
    assert Usage.from_response(usage()).cost_usd == pytest.approx(1.75 + 1.4)
    cached = Usage.from_response(usage(cached=400_000, reasoning=60_000))
    assert cached.reasoning_tokens == 60_000
    assert cached.cost_usd == pytest.approx(0.6 * 1.75 + 0.4 * 0.175 + 1.4)


def test_generate_sends_structured_request_without_storage():
    client = Mock()
    client.responses.parse.return_value = response(parsed=Answer(answer="OK"))
    result = AzureGenerator(client, "gpt-5.2", reasoning_effort="low").generate(
        "instructions", "question", Answer
    )
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-5.2"
    assert kwargs["text_format"] is Answer
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["store"] is False
    assert result.output.answer == "OK"
    assert result.usage.reasoning_tokens == 40_000
    assert result.response_id == "resp_1"


@pytest.mark.parametrize(
    "bad",
    [
        response(status="incomplete", incomplete=SimpleNamespace(reason="max_output_tokens")),
        response(parsed=None),
    ],
)
def test_incomplete_or_unparsed_response_is_an_error(bad):
    client = Mock()
    client.responses.parse.return_value = bad
    with pytest.raises(GenerationError):
        AzureGenerator(client, "gpt-5.2").generate("i", "q", Answer)


def test_client_uses_v1_endpoint_and_entra_token():
    credential = Mock()
    credential.get_token.return_value = AccessToken("jwt-token", 9_999_999_999)
    client = openai_client("https://res.services.ai.azure.com/", credential)
    assert str(client.base_url) == "https://res.services.ai.azure.com/openai/v1/"
    with pytest.raises(ValueError):
        openai_client("http://res.services.ai.azure.com", credential)


def test_invalid_generator_settings_are_rejected():
    with pytest.raises(ValueError):
        AzureGenerator(Mock(), " ")
    with pytest.raises(ValueError):
        AzureGenerator(Mock(), "gpt-5.2", max_output_tokens=1)


def blocked_error(code="content_filter"):
    import httpx
    import openai

    request = httpx.Request("POST", "https://res.services.ai.azure.com/openai/v1/responses")
    body = {
        "code": code,
        "innererror": {
            "content_filter_result": {"indirect_attack": {"filtered": True, "detected": True}}
        },
    }
    return openai.BadRequestError(
        "blocked", response=httpx.Response(400, request=request), body=body
    )


def test_prompt_blocked_by_content_filter_keeps_azure_details():
    client = Mock()
    client.responses.parse.side_effect = blocked_error()
    with pytest.raises(ContentFilterError) as caught:
        AzureGenerator(client, "gpt-5.2").generate("i", "q", Answer)
    assert caught.value.details["innererror"]["content_filter_result"]["indirect_attack"][
        "filtered"
    ]


def test_other_bad_requests_are_not_hidden():
    import openai

    client = Mock()
    client.responses.parse.side_effect = blocked_error(code="invalid_request")
    with pytest.raises(openai.BadRequestError):
        AzureGenerator(client, "gpt-5.2").generate("i", "q", Answer)


def test_answer_interrupted_by_content_filter():
    client = Mock()
    client.responses.parse.return_value = response(
        status="incomplete", incomplete=SimpleNamespace(reason="content_filter")
    )
    with pytest.raises(ContentFilterError):
        AzureGenerator(client, "gpt-5.2").generate("i", "q", Answer)


def test_shield_check_collects_filter_annotations_anywhere():
    from ai103.rag.shield_check import filter_fields, message

    data = {
        "choices": [{"content_filter_results": {"hate": {"filtered": False}}}],
        "prompt_filter_results": [
            {"content_filter_results": {"indirect_attack": {"detected": True}}}
        ],
        "usage": {"total_tokens": 10},
    }
    found = filter_fields(data)
    assert set(found) == {"choices[0].content_filter_results", "prompt_filter_results"}
    assert message("doc").count('"""<documents>') == 1
