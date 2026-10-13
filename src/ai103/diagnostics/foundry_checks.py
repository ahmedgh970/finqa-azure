"""Une lecture réelle de connexions, sans modèle ni quota d'inférence."""

from itertools import islice

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential

from ai103.config import Settings


def check_foundry_project(credential: TokenCredential, settings: Settings) -> dict[str, object]:
    if settings.foundry_project_endpoint is None:
        raise ValueError("Configurer FOUNDRY_PROJECT_ENDPOINT pour contrôler ce projet.")
    with AIProjectClient(
        endpoint=settings.foundry_project_endpoint,
        credential=credential,
        connection_timeout=settings.azure_timeout_seconds,
        read_timeout=settings.azure_timeout_seconds,
        retry_total=0,
    ) as client:
        # L'itérateur est paresseux : l'itérer déclenche effectivement la requête HTTP.
        # Pas de list_secrets(), ni de journalisation des endpoints/identifiants de connexions.
        observed = sum(1 for _ in islice(client.connections.list(), 10))
        return {
            "project_accessible": True,
            "connections_observed": observed,
            "connection_count_is_lower_bound": observed == 10,
        }
