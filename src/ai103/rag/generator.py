"""Génération Azure OpenAI via l'endpoint v1, authentification Entra sans clé API."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from openai import BadRequestError, OpenAI
from openai.types.responses import ParsedResponse, ResponseUsage
from pydantic import BaseModel

SCOPE = "https://cognitiveservices.azure.com/.default"
# Tarif Global gpt-5.2 relevé le 18/10/2026 dans l'API Azure Retail Prices, USD par million.
PRICES_PER_MILLION = {"input": 1.75, "cached_input": 0.175, "output": 14.0}

Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class GenerationError(RuntimeError):
    """Réponse absente, tronquée ou non conforme au schéma demandé."""


class ContentFilterError(GenerationError):
    """Requête ou réponse bloquée par les garde-fous du déploiement (filtres, Prompt Shields)."""

    def __init__(self, message: str, details: object = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    @classmethod
    def from_response(cls, usage: ResponseUsage | None) -> "Usage":
        if usage is None:
            raise GenerationError("Usage des tokens absent de la réponse.")
        return cls(
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.input_tokens_details.cached_tokens or 0,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens or 0,
        )

    @property
    def cost_usd(self) -> float:
        """Estimation au tarif public ; les tokens de raisonnement sont des tokens de sortie."""
        uncached = self.input_tokens - self.cached_input_tokens
        return (
            uncached * PRICES_PER_MILLION["input"]
            + self.cached_input_tokens * PRICES_PER_MILLION["cached_input"]
            + self.output_tokens * PRICES_PER_MILLION["output"]
        ) / 1_000_000


@dataclass(frozen=True)
class Generation[T: BaseModel]:
    output: T
    usage: Usage
    latency_seconds: float
    response_id: str
    model: str


def is_content_filter(exc: BadRequestError) -> bool:
    """Codes documentés par Azure pour un blocage par les garde-fous, pas un simple mot-clé."""
    inner = exc.body.get("innererror") if isinstance(exc.body, dict) else None
    inner_code = inner.get("code") if isinstance(inner, dict) else None
    return exc.code == "content_filter" or inner_code == "ResponsibleAIPolicyViolation"


def openai_client(endpoint: str, credential: TokenCredential, timeout: float = 120) -> OpenAI:
    """Client OpenAI pointé sur l'endpoint v1 de la ressource Azure, jeton Entra renouvelé."""
    if not endpoint.startswith("https://"):
        raise ValueError("Endpoint HTTPS requis.")
    return OpenAI(
        base_url=endpoint.rstrip("/") + "/openai/v1/",
        api_key=get_bearer_token_provider(credential, SCOPE),
        timeout=timeout,
        max_retries=2,
    )


class AzureGenerator:
    """Appel Responses API avec sortie structurée ; le client appartient à l'appelant."""

    def __init__(
        self,
        client: OpenAI,
        deployment: str,
        *,
        reasoning_effort: Effort = "low",
        max_output_tokens: int = 4000,
    ) -> None:
        if not deployment.strip() or max_output_tokens < 16:
            raise ValueError("Déploiement et max_output_tokens >= 16 requis.")
        self.client = client
        self.deployment = deployment
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def generate[T: BaseModel](
        self, instructions: str, user_input: str, schema: type[T]
    ) -> Generation[T]:
        started = perf_counter()
        try:
            response = self._parse(instructions, user_input, schema)
        except BadRequestError as exc:
            if is_content_filter(exc):
                raise ContentFilterError("Prompt bloqué par le filtre.", exc.body) from exc
            raise
        latency = perf_counter() - started
        reason = getattr(response.incomplete_details, "reason", None)
        if reason == "content_filter":
            raise ContentFilterError("Réponse interrompue par le filtre de contenu.")
        if response.status != "completed" or response.output_parsed is None:
            raise GenerationError(f"Réponse inexploitable : statut={response.status}, {reason=}.")
        return Generation(
            output=response.output_parsed,
            usage=Usage.from_response(response.usage),
            latency_seconds=latency,
            response_id=response.id,
            model=response.model,
        )

    def _parse[T: BaseModel](
        self, instructions: str, user_input: str, schema: type[T]
    ) -> ParsedResponse[T]:
        return self.client.responses.parse(
            model=self.deployment,
            instructions=instructions,
            input=user_input,
            text_format=schema,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            # Aucune conservation côté service : chaque question est indépendante.
            store=False,
        )
