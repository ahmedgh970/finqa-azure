"""Validation sans accès au compte Azure ni au serveur Ollama."""

import pytest

from ai103.config import ConfigurationError, Settings


def test_defaults_allow_optional_foundry() -> None:
    settings = Settings.from_env({})
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.foundry_project_endpoint is None
    assert settings.azure_subscription_id is None
    assert settings.azure_token_credentials == "AzureCliCredential"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number", ""])
def test_invalid_timeout_is_rejected(value: str) -> None:
    with pytest.raises(ConfigurationError, match="OLLAMA_REQUEST_TIMEOUT_SECONDS"):
        Settings.from_env({"OLLAMA_REQUEST_TIMEOUT_SECONDS": value})


@pytest.mark.parametrize(
    "value",
    [
        "https://ollama.com",
        "http://example.com:11434",
        "http://localhost:11434/api",
        "http://user:secret@localhost:11434",
        "http://localhost:invalid",
    ],
)
def test_ollama_must_stay_local(value: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({"OLLAMA_BASE_URL": value})


def test_cloud_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="locaux"):
        Settings.from_env({"OLLAMA_CHAT_MODEL": "some-model:cloud"})


def test_model_endpoint_is_not_a_project_endpoint() -> None:
    with pytest.raises(ConfigurationError, match="projet Foundry"):
        Settings.from_env({"FOUNDRY_PROJECT_ENDPOINT": "https://example.openai.azure.com"})


def test_foundry_endpoint_normalization() -> None:
    settings = Settings.from_env(
        {"FOUNDRY_PROJECT_ENDPOINT": "https://demo.services.ai.azure.com/api/projects/course/"}
    )
    assert (
        settings.foundry_project_endpoint
        == "https://demo.services.ai.azure.com/api/projects/course"
    )


def test_invalid_uuid_does_not_echo_value() -> None:
    with pytest.raises(ConfigurationError) as captured:
        Settings.from_env({"AZURE_SUBSCRIPTION_ID": "private-invalid-value"})
    assert "private-invalid-value" not in str(captured.value)


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({"OLLAMA_EMBEDDING_MODEL": " "})
