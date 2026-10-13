"""Inventaire de tout le corpus Blob : uv run python -m ai103.storage_inventory."""

import argparse
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from azure.core.exceptions import AzureError, ClientAuthenticationError, HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobProperties, BlobServiceClient
from dotenv import load_dotenv


@dataclass(frozen=True)
class Inventory:
    """Résultat calculé seulement après avoir parcouru tous les blobs."""

    pdf_count: int
    total_bytes: int
    other_blob_count: int


def storage_configuration() -> tuple[str, str]:
    """Lire les deux valeurs non secrètes et vérifier leur forme."""
    account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "").strip().rstrip("/")
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER_NAME", "").strip()
    if not re.fullmatch(r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", account_url):
        raise ValueError(
            "Configurer AZURE_STORAGE_ACCOUNT_URL : https://<compte>.blob.core.windows.net"
        )
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", container_name)
        or "--" in container_name
    ):
        raise ValueError("Configurer AZURE_STORAGE_CONTAINER_NAME avec un nom de conteneur valide.")
    # Même sélection explicite que le doctor ; les variables exportées gardent la priorité.
    choice = os.environ.setdefault("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    if choice not in {"AzureCliCredential", "dev"}:
        raise ValueError(
            "AZURE_TOKEN_CREDENTIALS doit être AzureCliCredential ou dev sur ce poste."
        )
    return account_url, container_name


def inventory_pdf_blobs(blobs: Iterable[BlobProperties]) -> Inventory:
    """Consommer tout l'itérateur paginé, sans limiter le corpus ni télécharger les PDF."""
    pdf_count = 0
    total_bytes = 0
    other_blob_count = 0
    for blob in blobs:
        if not blob.name.lower().endswith(".pdf"):
            other_blob_count += 1
            continue
        pdf_count += 1
        total_bytes += blob.size
        # repr rend aussi visibles les caractères spéciaux dans les noms de fichiers.
        print(f"{blob.name!r}\t{blob.size} octets")
    return Inventory(pdf_count, total_bytes, other_blob_count)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventaire complet des PDF dans Azure Blob Storage"
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--expected-count", type=int, default=368, help="Nombre de PDF attendu")
    args = parser.parse_args(argv)
    if args.expected_count < 0:
        parser.error("--expected-count doit être positif ou nul.")
    # Chemin explicite : aucune recherche de .env dans les dossiers parents.
    load_dotenv(args.env_file, override=False)
    try:
        account_url, container_name = storage_configuration()
    except ValueError as error:
        print(f"Configuration : {error}", file=sys.stderr)
        return 2

    try:
        # Le credential obtient un jeton Entra ; aucun jeton ou clé n'est affiché.
        with (
            DefaultAzureCredential() as credential,
            BlobServiceClient(
                account_url=account_url,
                credential=credential,
                connection_timeout=10,
                read_timeout=30,
                retry_total=2,
            ) as service,
        ):
            container = service.get_container_client(container_name)
            # L'appel retourne un itérateur : la boucle déclenche les requêtes réseau
            # et poursuit automatiquement la pagination jusqu'au dernier résultat.
            inventory = inventory_pdf_blobs(container.list_blobs())
    except ClientAuthenticationError:
        print("Authentification échouée : vérifier az login et le tenant.", file=sys.stderr)
        return 1
    except HttpResponseError as error:
        hints = {
            401: "Vérifier la session Azure CLI et le tenant.",
            403: "Vérifier le rôle Blob au bon scope, sa propagation et les règles réseau.",
            404: "Vérifier le compte Storage et le nom du conteneur.",
        }
        hint = hints.get(
            error.status_code or 0, "Vérifier la disponibilité du service et réessayer."
        )
        # Ne pas afficher le corps SDK : il peut contenir des informations confidentielles.
        print(f"Azure HTTP {error.status_code} : {hint}", file=sys.stderr)
        return 1
    except AzureError:
        print("Accès Azure échoué : vérifier le réseau et la disponibilité.", file=sys.stderr)
        return 1

    print(f"\nInventaire complet du conteneur {container_name!r}")
    print(f"PDF : {inventory.pdf_count} ; attendu : {args.expected_count}")
    print(f"Volume PDF : {inventory.total_bytes} octets ({inventory.total_bytes / 10**9:.3f} Go)")
    print(f"Autres blobs : {inventory.other_blob_count}")
    if inventory.pdf_count != args.expected_count:
        print(
            "Nombre de PDF différent du corpus attendu : vérifier les transferts.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
