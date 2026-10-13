"""Un échec est rapporté sans interrompre les contrôles indépendants."""

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Literal

import httpx
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.identity import DefaultAzureCredential

from ai103.config import Settings
from ai103.diagnostics.azure_checks import check_azure_identity, check_azure_subscription
from ai103.diagnostics.foundry_checks import check_foundry_project
from ai103.models.contracts import ChatMessage
from ai103.models.ollama import OllamaProvider

logger = logging.getLogger(__name__)
Status = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    duration_ms: float
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _failure_message(exc: Exception) -> str:
    # Les exceptions SDK peuvent contenir des requêtes, URLs ou corps sensibles.
    # On classe l'erreur plutôt que d'afficher aveuglément str(exc).
    if isinstance(exc, ClientAuthenticationError):
        return "Authentification Azure impossible. Vérifier az login et AZURE_TOKEN_CREDENTIALS."
    if isinstance(exc, HttpResponseError):
        if exc.status_code == 401:
            return "HTTP 401 : vérifier l'identité, le tenant et l'audience du jeton."
        if exc.status_code == 403:
            return "HTTP 403 : vérifier RBAC et les restrictions réseau du service."
        if exc.status_code == 404:
            return "HTTP 404 : vérifier l'identifiant de ressource ou l'endpoint du projet."
        return f"Échec HTTP Azure (statut {exc.status_code})."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Échec HTTP Ollama (statut {exc.response.status_code}). Vérifier le modèle choisi."
    if isinstance(exc, (httpx.TimeoutException, ServiceResponseError)):
        return "Délai de réponse dépassé. Vérifier le service et le timeout configuré."
    if isinstance(exc, (httpx.RequestError, ServiceRequestError)):
        return "Connexion impossible. Vérifier le serveur, le réseau et les restrictions d'accès."
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Échec inattendu ({type(exc).__name__})."


def run_check(name: str, action: Callable[[], dict[str, object]]) -> CheckResult:
    start = perf_counter()
    try:
        details = action()
        status: Status = "passed"
        message = "Contrôle réussi."
    except Exception as exc:
        details = {"error_type": type(exc).__name__}
        status = "failed"
        message = _failure_message(exc)
    duration = round((perf_counter() - start) * 1000, 2)
    logger.info("check=%s status=%s duration_ms=%.2f", name, status, duration)
    return CheckResult(name, status, duration, message, details)


def _skipped(name: str, reason: str) -> CheckResult:
    return CheckResult(name, "skipped", 0, reason)


def report_exit_code(results: list[CheckResult]) -> int:
    return 1 if any(result.status == "failed" for result in results) else 0


def _check_generation(provider: OllamaProvider) -> dict[str, object]:
    result = provider.generate([ChatMessage("user", "Réponds simplement : connexion réussie.")])
    return {
        "model": result.model,
        "response": result.text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def _check_embeddings(provider: OllamaProvider, model: str) -> dict[str, object]:
    vectors = provider.embed(["Azure AI Search retrouvera les documents pertinents."])
    return {"model": model, "dimension": len(vectors[0]), "vectors_received": len(vectors)}


def run_diagnostics(settings: Settings, *, quick: bool = False) -> list[CheckResult]:
    results = [CheckResult("configuration", "passed", 0, "Configuration valide.")]
    with httpx.Client(
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout_seconds, connect=5),
        trust_env=False,  # Un proxy d'entreprise ne doit pas recevoir les requêtes locales.
    ) as client:
        provider = OllamaProvider(
            client,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.ollama_embedding_model,
        )
        models: list[str] = []

        def list_models() -> dict[str, object]:
            models.extend(provider.list_local_models())
            return {"local_models": models}

        server = run_check("ollama_server", list_models)
        results.append(server)
        for name, model, action in (
            ("ollama_generation", settings.ollama_chat_model, lambda: _check_generation(provider)),
            (
                "ollama_embeddings",
                settings.ollama_embedding_model,
                lambda: _check_embeddings(provider, settings.ollama_embedding_model),
            ),
        ):
            if quick:
                results.append(_skipped(name, "Mode rapide : inférence locale non exécutée."))
            elif server.status != "passed":
                results.append(_skipped(name, "Le serveur Ollama est inaccessible."))
            elif model not in models:
                results.append(CheckResult(name, "failed", 0, f"Modèle local absent : {model}."))
            else:
                results.append(run_check(name, action))

    with DefaultAzureCredential(
        require_envvar=True, process_timeout=settings.azure_timeout_seconds
    ) as credential:
        identity = run_check("azure_identity", lambda: check_azure_identity(credential))
        results.append(identity)
        if identity.status == "passed":
            results.append(
                run_check(
                    "azure_subscription", lambda: check_azure_subscription(credential, settings)
                )
            )
        else:
            results.append(_skipped("azure_subscription", "Authentification Azure non validée."))
        if settings.foundry_project_endpoint is None:
            results.append(
                _skipped("foundry_project", "Projet facultatif : FOUNDRY_PROJECT_ENDPOINT absent.")
            )
        elif identity.status != "passed":
            results.append(_skipped("foundry_project", "Authentification Azure non validée."))
        else:
            results.append(
                run_check("foundry_project", lambda: check_foundry_project(credential, settings))
            )
    return results
