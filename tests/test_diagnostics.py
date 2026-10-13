"""Les appels HTTP/SDK sont simulés : aucune dépendance à l'infrastructure."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from azure.core.exceptions import HttpResponseError

from ai103.config import Settings
from ai103.diagnostics.foundry_checks import check_foundry_project
from ai103.diagnostics.runner import CheckResult, report_exit_code, run_check, run_diagnostics
from ai103.models.contracts import ChatMessage
from ai103.models.ollama import OllamaProvider, OllamaResponseError


def test_azure_error_does_not_leak_sdk_body() -> None:
    def fail() -> dict[str, object]:
        error = HttpResponseError(message="secret-token-in-response")
        error.status_code = 403
        raise error

    result = run_check("foundry", fail)
    assert result.status == "failed"
    assert "RBAC" in result.message
    assert "secret-token" not in json.dumps(result.to_dict())


def test_failed_check_does_not_stop_next_check() -> None:
    def fail() -> dict[str, object]:
        raise ConnectionError("sensitive-internal-details")

    results = [run_check("first", fail), run_check("second", lambda: {"ready": True})]
    assert [result.status for result in results] == ["failed", "passed"]
    assert report_exit_code(results) == 1
    assert "sensitive-internal-details" not in results[0].message


def test_optional_skip_is_not_failure() -> None:
    results = [CheckResult("foundry", "skipped", 0, "Non configuré.")]
    assert report_exit_code(results) == 0


def test_http_contracts_and_token_usage() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "model": "local-chat",
                    "message": {"content": "Connexion réussie."},
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                },
            )
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    with httpx.Client(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    ) as c:
        provider = OllamaProvider(c, chat_model="local-chat", embedding_model="local-embed")
        result = provider.generate([ChatMessage("user", "Bonjour")])
        vectors = provider.embed(["Un document"])
    assert result.text == "Connexion réussie."
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert len(vectors[0]) == 3
    chat_payload = json.loads(requests[0].content)
    embed_payload = json.loads(requests[1].content)
    assert chat_payload["model"] == "local-chat"
    assert chat_payload["stream"] is False
    assert embed_payload["model"] == "local-embed"
    assert embed_payload["truncate"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": []},
        {"embeddings": [[]]},
        {"embeddings": [[True, False]]},
        {"embeddings": [["not-a-number"]]},
        {"error": "internal-ollama-error"},
    ],
)
def test_invalid_embedding_response_cannot_pass(payload: dict[str, object]) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with httpx.Client(base_url="http://localhost:11434", transport=transport) as client:
        provider = OllamaProvider(client, chat_model="chat", embedding_model="embed")
        with pytest.raises(OllamaResponseError):
            provider.embed(["Test"])


def test_http_error_is_reported_as_failure() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"error": "missing"}))
    with httpx.Client(base_url="http://localhost:11434", transport=transport) as client:
        provider = OllamaProvider(client, chat_model="missing", embedding_model="embed")
        result = run_check("generation", lambda: {"result": provider.generate([])})
    assert result.status == "failed"
    assert "404" in result.message


def test_foundry_read_consumes_lazy_iterator_without_inference() -> None:
    settings = Settings.from_env(
        {"FOUNDRY_PROJECT_ENDPOINT": "https://demo.services.ai.azure.com/api/projects/course"}
    )
    observed: list[str] = []

    def connections():
        observed.append("network-request-executed")
        return
        yield  # Reproduit un itérateur SDK paresseux vide.

    with patch("ai103.diagnostics.foundry_checks.AIProjectClient") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.connections.list.return_value = connections()
        result = check_foundry_project(MagicMock(), settings)
        client.get_openai_client.assert_not_called()
    assert observed == ["network-request-executed"]
    assert result["connections_observed"] == 0
    assert result["project_accessible"] is True


def test_missing_models_fail_without_attempting_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    with (
        patch("ai103.diagnostics.runner.OllamaProvider") as provider_class,
        patch("ai103.diagnostics.runner.DefaultAzureCredential"),
        patch(
            "ai103.diagnostics.runner.check_azure_identity", return_value={"authenticated": True}
        ),
        patch(
            "ai103.diagnostics.runner.check_azure_subscription", return_value={"state": "Enabled"}
        ),
    ):
        provider = provider_class.return_value
        provider.list_local_models.return_value = ["other-model"]
        results = run_diagnostics(Settings.from_env({}))
        provider.generate.assert_not_called()
        provider.embed.assert_not_called()
    statuses = {result.name: result.status for result in results}
    assert statuses["ollama_generation"] == "failed"
    assert statuses["ollama_embeddings"] == "failed"
    assert statuses["foundry_project"] == "skipped"
    assert report_exit_code(results) == 1
