"""Évaluateurs Azure AI Evaluation (juge LLM distinct du générateur) et contrôle des chiffres."""

from typing import Any

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    GroundednessEvaluator,
    RelevanceEvaluator,
    ResponseCompletenessEvaluator,
)
from azure.core.credentials import TokenCredential

from ai103.evaluation.numeric import check

# La version par défaut des évaluateurs (2024-02-15-preview) ne gère pas les modèles de
# raisonnement ; on fixe celle de référence du SDK.
API_VERSION = "2025-04-01-preview"

EVALUATOR_CONFIG: dict[str, Any] = {
    "groundedness": {
        "column_mapping": {
            "query": "${data.query}",
            "context": "${data.context}",
            "response": "${data.response}",
        }
    },
    "relevance": {"column_mapping": {"query": "${data.query}", "response": "${data.response}"}},
    "completeness": {
        "column_mapping": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"}
    },
    "numeric": {
        "column_mapping": {"response": "${data.response}", "ground_truth": "${data.ground_truth}"}
    },
}


class NumericMatchEvaluator:
    """Évaluateur personnalisé, sans LLM : le chiffre gold figure-t-il dans la réponse directe ?"""

    def __call__(self, *, response: str, ground_truth: str) -> dict[str, Any]:
        result = check(response, ground_truth)
        return {
            "numeric_applicable": result.applicable,
            # None hors réponses chiffrées : ces questions ne comptent pas dans la moyenne.
            "numeric_match": (1.0 if result.matched else 0.0) if result.applicable else None,
        }


def model_config(endpoint: str, deployment: str) -> AzureOpenAIModelConfiguration:
    return AzureOpenAIModelConfiguration(
        azure_endpoint=endpoint, azure_deployment=deployment, api_version=API_VERSION
    )


def evaluators(
    config: AzureOpenAIModelConfiguration, credential: TokenCredential
) -> dict[str, Any]:
    """Trois juges LLM (modèle de raisonnement) et le contrôle déterministe."""
    common: dict[str, Any] = {"credential": credential, "is_reasoning_model": True}
    return {
        "groundedness": GroundednessEvaluator(config, **common),  # type: ignore[no-untyped-call]
        "relevance": RelevanceEvaluator(config, **common),  # type: ignore[no-untyped-call]
        "completeness": ResponseCompletenessEvaluator(config, **common),
        "numeric": NumericMatchEvaluator(),
    }
