"""Configuration explicite, validée sans afficher les valeurs confidentielles."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID


class ConfigurationError(ValueError):
    """Une variable de configuration est invalide."""


def _optional_uuid(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name, "").strip()
    if not value:
        return None
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ConfigurationError(f"{name} doit être un UUID valide.") from exc


def _timeout(values: Mapping[str, str], name: str, default: str) -> float:
    try:
        value = float(values.get(name, default))
    except ValueError as exc:
        raise ConfigurationError(f"{name} doit être un nombre positif.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} doit être un nombre fini strictement positif.")
    return value


def _endpoint(value: str, name: str, *, local: bool) -> str:
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        _ = parts.port  # Valide aussi le port avant toute requête.
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError
        if local:
            valid = (
                parts.scheme == "http"
                and parts.hostname in {"localhost", "127.0.0.1", "::1"}
                and not parts.path
            )
        else:
            segments = parts.path.split("/")
            valid = (
                parts.scheme == "https"
                and parts.hostname is not None
                and parts.hostname.endswith(".services.ai.azure.com")
                and len(segments) == 4
                and segments[1:3] == ["api", "projects"]
                and bool(segments[3])
            )
        if not valid:
            raise ValueError
    except ValueError as exc:
        message = (
            "doit viser Ollama local en HTTP, sans chemin (ex. http://localhost:11434)."
            if local
            else "doit être l'endpoint HTTPS d'un projet Foundry actuel : "
            "https://<resource>.services.ai.azure.com/api/projects/<project>."
        )
        raise ConfigurationError(f"{name} {message}") from exc
    return value


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_timeout_seconds: float
    azure_subscription_id: str | None
    azure_tenant_id: str | None
    azure_timeout_seconds: float
    azure_token_credentials: str
    foundry_project_endpoint: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        chat = values.get("OLLAMA_CHAT_MODEL", "granite4.1:3b").strip()
        embedding = values.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest").strip()
        if not chat or not embedding:
            raise ConfigurationError("Les noms des modèles Ollama ne doivent pas être vides.")
        if any("cloud" in model.lower() for model in (chat, embedding)):
            raise ConfigurationError("Les diagnostics Ollama requièrent des modèles locaux.")
        credential = values.get("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential").strip()
        if credential not in {"AzureCliCredential", "dev"}:
            raise ConfigurationError("AZURE_TOKEN_CREDENTIALS doit être AzureCliCredential ou dev.")
        foundry = values.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
        return cls(
            ollama_base_url=_endpoint(
                values.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                "OLLAMA_BASE_URL",
                local=True,
            ),
            ollama_chat_model=chat,
            ollama_embedding_model=embedding,
            ollama_timeout_seconds=_timeout(values, "OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"),
            azure_subscription_id=_optional_uuid(values, "AZURE_SUBSCRIPTION_ID"),
            azure_tenant_id=_optional_uuid(values, "AZURE_TENANT_ID"),
            azure_timeout_seconds=_timeout(values, "AZURE_REQUEST_TIMEOUT_SECONDS", "10"),
            azure_token_credentials=credential,
            foundry_project_endpoint=(
                _endpoint(foundry, "FOUNDRY_PROJECT_ENDPOINT", local=False) if foundry else None
            ),
        )
