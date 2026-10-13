"""Deux preuves distinctes : authentification Entra puis lecture ARM."""

from azure.core.credentials import TokenCredential
from azure.mgmt.resource.subscriptions import SubscriptionClient

from ai103.config import Settings


def check_azure_identity(credential: TokenCredential) -> dict[str, object]:
    # Le jeton est utilisé uniquement en mémoire, jamais retourné ni journalisé.
    token = credential.get_token("https://management.azure.com/.default")
    return {"authenticated": True, "token_expires_on": token.expires_on}


def check_azure_subscription(credential: TokenCredential, settings: Settings) -> dict[str, object]:
    if settings.azure_subscription_id is None:
        raise ValueError("Configurer AZURE_SUBSCRIPTION_ID depuis az account show.")
    with SubscriptionClient(
        credential=credential,
        connection_timeout=settings.azure_timeout_seconds,
        read_timeout=settings.azure_timeout_seconds,
        retry_total=0,
    ) as client:
        subscription = client.subscriptions.get(settings.azure_subscription_id)
        tenant_id = subscription.tenant_id
        if settings.azure_tenant_id and tenant_id != settings.azure_tenant_id:
            raise ValueError("Le tenant de la subscription diffère de AZURE_TENANT_ID.")
        state = str(subscription.state)
        # Les versions du SDK exposent soit un str, soit une enum str.
        if state not in {"Enabled", "SubscriptionState.ENABLED"}:
            raise ValueError("La subscription existe mais n'est pas active (Enabled).")
        return {
            "subscription_id": subscription.subscription_id,
            "subscription_name": subscription.display_name,
            "tenant_id": tenant_id,
            "state": "Enabled",
        }
