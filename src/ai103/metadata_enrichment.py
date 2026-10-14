"""Préparer ou appliquer les métadonnées documentaires dans Azure AI Search."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import unquote, urlsplit

from azure.core.exceptions import AzureError, HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from dotenv import load_dotenv

DEFAULT_INDEXES = ("financial-chunks-v1", "financial-chunks-layout-v1")
SELECT_FIELDS = ["chunk_id", "source", "company", "fiscal_year"]


class EnrichmentError(ValueError):
    """Un contrôle empêche de poursuivre l'enrichissement."""


@dataclass(frozen=True)
class DocumentMetadata:
    company: str
    fiscal_year: int | None


@dataclass
class IndexPlan:
    index_name: str
    chunk_count: int
    documents: dict[str, int]
    patches: list[dict[str, Any]]
    sources: dict[str, str]
    chunk_sources: dict[str, str]
    applied_chunks: int = 0
    write_attempted: bool = False
    verified: bool = False
    throttling_retries: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "index_name": self.index_name,
            "chunk_count": self.chunk_count,
            "document_count": len(self.documents),
            "planned_updates": len(self.patches),
            "unchanged_chunks": self.chunk_count - len(self.patches),
            "applied_chunks": self.applied_chunks,
            "write_attempted": self.write_attempted,
            "verified": self.verified,
            "throttling_retries": self.throttling_retries,
            "failures": self.failures,
            "documents": self.documents,
        }


def load_metadata(path: Path, expected_count: int = 84) -> dict[str, DocumentMetadata]:
    """Lire uniquement le référentiel à trois champs, sans données de questions/réponses."""
    metadata: dict[str, DocumentMetadata] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnrichmentError(f"JSON invalide à la ligne {line_number}.") from exc
        if not isinstance(row, dict) or set(row) != {"doc_name", "company", "fiscal_year"}:
            raise EnrichmentError(f"Champs inattendus à la ligne {line_number}.")
        name, company, year = row["doc_name"], row["company"], row["fiscal_year"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or "/" in name
            or "\\" in name
            or name.lower().endswith(".pdf")
            or not isinstance(company, str)
            or not company.strip()
            or (year is not None and (type(year) is not int or not 1000 <= year <= 9999))
        ):
            raise EnrichmentError(f"Métadonnées invalides à la ligne {line_number}.")
        if name in metadata:
            raise EnrichmentError(f"Document dupliqué à la ligne {line_number}.")
        metadata[name] = DocumentMetadata(company.strip(), year)
    if len(metadata) != expected_count:
        raise EnrichmentError(
            f"Référentiel incomplet : {len(metadata)} documents, {expected_count} attendus."
        )
    return metadata


def doc_name_from_source(source: str) -> str:
    """Retrouver le nom du document en préservant sa casse et les underscores."""
    parts = urlsplit(source)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or not parts.hostname.endswith(".blob.core.windows.net")
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise EnrichmentError("Une source n'est pas une URL Blob HTTPS sans secret.")
    filename = unquote(parts.path.rsplit("/", 1)[-1])
    if not filename.lower().endswith(".pdf") or "/" in filename or "\\" in filename:
        raise EnrichmentError("Une source ne désigne pas un fichier PDF valide.")
    return filename[:-4]


def validate_schema(index: SearchIndex) -> None:
    """Les vecteurs stockés permettent d'omettre content_vector dans les merge."""
    fields = {item.name: item for item in index.fields}
    required = {
        "chunk_id": "Edm.String",
        "source": "Edm.String",
        "company": "Edm.String",
        "fiscal_year": "Edm.Int32",
    }
    for name, type_name in required.items():
        item = fields.get(name)
        if item is None or item.type != type_name or item.retrievable is False:
            raise EnrichmentError(f"Schéma incompatible : champ {name}.")
    if fields["chunk_id"].key is not True:
        raise EnrichmentError("chunk_id doit être la clé de l'index.")
    if any(fields[name].filterable is not True for name in ("chunk_id", "company", "fiscal_year")):
        raise EnrichmentError("chunk_id, company et fiscal_year doivent être filtrables.")
    vectors = [item for item in index.fields if item.vector_search_dimensions is not None]
    if not vectors or any(item.stored is not True for item in vectors):
        raise EnrichmentError("Les champs vectoriels doivent avoir stored=true avant un merge.")


def plan_index(
    client: SearchClient, index_name: str, metadata: Mapping[str, DocumentMetadata]
) -> IndexPlan:
    """Parcourir toutes les pages et refuser une couverture partielle ou incohérente."""
    before = client.get_document_count()
    counts: Counter[str] = Counter()
    sources: dict[str, str] = {}
    seen: set[str] = set()
    patches: list[dict[str, Any]] = []
    chunk_sources: dict[str, str] = {}
    while len(seen) < before:
        results = client.search(
            search_text="*",
            select=SELECT_FIELDS,
            include_total_count=True,
            skip=len(seen),
            top=1000,
        )
        if results.get_count() != before:
            raise EnrichmentError("Le nombre de chunks a changé pendant la lecture.")
        batch = list(results)
        if not batch:
            raise EnrichmentError("La pagination s'est arrêtée avant la fin de l'index.")
        for row in batch:
            key, source = row.get("chunk_id"), row.get("source")
            if not isinstance(key, str) or not key or not isinstance(source, str):
                raise EnrichmentError("Un chunk n'a pas de clé ou de source valide.")
            if key in seen:
                raise EnrichmentError("Clé répétée pendant la pagination ; lecture incohérente.")
            seen.add(key)
            chunk_sources[key] = source
            name = doc_name_from_source(source)
            if name not in metadata:
                raise EnrichmentError("Un document indexé est absent du référentiel.")
            if name in sources and sources[name] != source:
                raise EnrichmentError("Plusieurs sources différentes ont le même doc_name.")
            sources[name] = source
            counts[name] += 1
            target = metadata[name]
            if row.get("company") != target.company or row.get("fiscal_year") != target.fiscal_year:
                patches.append(
                    {"chunk_id": key, "company": target.company, "fiscal_year": target.fiscal_year}
                )
    if len(seen) != before or client.get_document_count() != before:
        raise EnrichmentError("Le nombre de chunks est incohérent à la fin de la lecture.")
    if set(counts) != set(metadata):
        raise EnrichmentError("Certains documents du référentiel n'ont aucun chunk dans cet index.")
    return IndexPlan(
        index_name, before, dict(sorted(counts.items())), patches, sources, chunk_sources
    )


def verify_plan(
    client: SearchClient,
    plan: IndexPlan,
    metadata: Mapping[str, DocumentMetadata],
    *,
    attempts: int = 5,
    retry_delay: float = 3,
) -> None:
    """Relire les clés exactes sans skip ; attendre brièvement leur visibilité Search."""
    if attempts < 1 or retry_delay < 0:
        raise EnrichmentError("Paramètres de vérification invalides.")
    if len(plan.chunk_sources) != plan.chunk_count:
        raise EnrichmentError("La liste des clés à vérifier est incomplète.")
    if client.get_document_count() != plan.chunk_count:
        raise EnrichmentError("Le nombre de chunks a changé avant la vérification.")
    keys = list(plan.chunk_sources)
    for offset in range(0, len(keys), 500):
        batch = keys[offset : offset + 500]
        # Les clés Search valides ne contiennent pas '|'. Aucun ordre n'est requis :
        # ce filtre ne peut retourner que les 500 clés demandées, en une requête.
        if any("|" in key for key in batch):
            raise EnrichmentError("Une clé contient le séparateur de vérification.")
        values = "|".join(batch).replace("'", "''")
        for attempt in range(attempts):
            rows = list(
                client.search(
                    search_text="*",
                    filter=f"search.in(chunk_id, '{values}', '|')",
                    select=SELECT_FIELDS,
                    top=len(batch),
                )
            )
            returned = [row.get("chunk_id") for row in rows]
            if len(set(returned)) != len(returned) or set(returned) - set(batch):
                raise EnrichmentError("Réponse de vérification avec des clés inattendues.")
            matches = set(returned) == set(batch)
            for row in rows:
                key = row["chunk_id"]
                source = plan.chunk_sources[key]
                if row.get("source") != source:
                    raise EnrichmentError("La source d'un chunk a changé pendant la vérification.")
                target = metadata[doc_name_from_source(source)]
                matches = matches and (
                    row.get("company") == target.company
                    and row.get("fiscal_year") == target.fiscal_year
                )
            if matches:
                break
            if attempt == attempts - 1:
                raise EnrichmentError(
                    "Métadonnées pas encore confirmées après plusieurs lectures par clé ; "
                    "relancer une simulation pour établir les mises à jour restantes."
                )
            sleep(retry_delay)
    if client.get_document_count() != plan.chunk_count:
        raise EnrichmentError("Le nombre de chunks a changé pendant la vérification.")
    plan.verified = True


def quota_error(error: HttpResponseError) -> bool:
    """Classifier sans enregistrer le corps SDK susceptible de contenir des secrets."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "quota",
            "storage limit",
            "vector limit",
            "maximum number of documents",
        )
    )


def retry_after_seconds(error: HttpResponseError) -> float | None:
    """Lire Retry-After, en secondes/date HTTP, ou son équivalent en millisecondes."""
    if error.response is None:
        return None
    response_headers: Mapping[str, str] = getattr(error.response, "headers", {})
    headers = {key.lower(): value for key, value in response_headers.items()}
    for name in ("retry-after-ms", "x-ms-retry-after-ms", "retry-after"):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) / (1000 if name.endswith("-ms") else 1)
        except (TypeError, ValueError):
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                delay = (date - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                continue
        if delay >= 0 and delay < float("inf"):
            return delay
    return None


def apply_plan(client: SearchClient, plan: IndexPlan, batch_size: int = 500) -> None:
    """Appliquer seulement des merge ; arrêter au premier lot en échec partiel."""
    if not 1 <= batch_size <= 1000:
        raise EnrichmentError("batch_size doit être compris entre 1 et 1000.")
    for offset in range(0, len(plan.patches), batch_size):
        batch = plan.patches[offset : offset + batch_size]
        if any(set(row) != {"chunk_id", "company", "fiscal_year"} for row in batch):
            raise EnrichmentError("Une mise à jour contient des champs non autorisés.")
        if offset:
            sleep(1)  # Envois séquentiels, au plus un nouveau lot par seconde.
        merge_batch(client, plan, batch)
        print(
            f"{plan.index_name} : {plan.applied_chunks}/{len(plan.patches)} "
            "mises à jour acquittées.",
            flush=True,
        )


def merge_batch(client: SearchClient, plan: IndexPlan, batch: list[dict[str, Any]]) -> None:
    """Merge contrôlé commun aux deux clients ; n'accepter que leurs champs dédiés."""
    allowed = ({"chunk_id", "company", "fiscal_year"}, {"chunk_id", "page_start", "page_end"})
    if not batch or any(set(row) not in allowed for row in batch):
        raise EnrichmentError("Champs de merge non autorisés.")
    expected_keys = {row["chunk_id"] for row in batch}
    if len(expected_keys) != len(batch):
        raise EnrichmentError("Clés répétées dans un lot de merge.")
    plan.write_attempted = True
    for attempt in range(5):
        try:
            results = client.merge_documents(documents=batch, retry_total=0)
            break
        except HttpResponseError as exc:
            if exc.status_code != 429 or quota_error(exc) or attempt == 4:
                raise
            delay = max(2 ** (attempt + 1), retry_after_seconds(exc) or 0)
            if delay > 60:
                raise
            plan.throttling_retries += 1
            print(f"Azure HTTP 429 : reprise {attempt + 1}/4 du lot après {delay:g} s.", flush=True)
            sleep(delay)
    returned_keys = [result.key for result in results]
    if len(returned_keys) != len(expected_keys) or set(returned_keys) != expected_keys:
        raise EnrichmentError("Réponse de merge incomplète ; résultat de ce lot à vérifier.")
    for result in results:
        if result.succeeded:
            plan.applied_chunks += 1
        else:
            plan.failures.append({"key": result.key, "status_code": result.status_code})
    if plan.failures:
        raise EnrichmentError("Un lot de merge a des échecs ; arrêt et bilan dans le rapport.")


def search_endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    if not re.fullmatch(r"https://[a-z0-9][a-z0-9-]*\.search\.windows\.net", value):
        raise EnrichmentError("Configurer AZURE_SEARCH_SERVICE_ENDPOINT ou --endpoint.")
    return value


def azure_error_message(error: AzureError) -> str:
    code = error.status_code if isinstance(error, HttpResponseError) else None
    if code == 403:
        return "Azure HTTP 403 : vérifier les rôles Search et les règles réseau."
    if code == 429:
        if isinstance(error, HttpResponseError) and quota_error(error):
            if "vector" in str(error).lower():
                return "Azure HTTP 429 : quota vectoriel Search signalé ; arrêt du lot."
            if "storage" in str(error).lower():
                return "Azure HTTP 429 : quota de stockage Search signalé ; arrêt du lot."
            return "Azure HTTP 429 : quota Search signalé ; arrêt sans nouvelle tentative du lot."
        return (
            "Azure HTTP 429 : limite Search persistante ; débit ou capacité à vérifier. "
            "Les lots acquittés sont conservés ; une simulation indique ce qui reste à modifier."
        )
    return f"Accès Azure échoué (HTTP {code}) ; aucun détail SDK confidentiel affiché."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrichissement des métadonnées Azure Search")
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/reports/metadata-enrichment.json")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--endpoint", help="Sinon AZURE_SEARCH_SERVICE_ENDPOINT dans .env")
    parser.add_argument(
        "--index", action="append", dest="indexes", help="Répétable ; les deux par défaut"
    )
    parser.add_argument("--expected-count", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Autoriser les écritures merge")
    mode.add_argument("--dry-run", action="store_true", help="Simulation, comportement par défaut")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file, override=False)
    indexes = args.indexes if args.indexes is not None else list(DEFAULT_INDEXES)
    try:
        if args.expected_count < 1 or not 1 <= args.batch_size <= 1000:
            raise EnrichmentError("Nombre attendu ou taille de lot invalide.")
        if len(set(indexes)) != len(indexes) or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,127}", name) for name in indexes
        ):
            raise EnrichmentError("Noms d'index invalides ou répétés.")
        if args.report.resolve() in {args.metadata.resolve(), args.env_file.resolve()}:
            raise EnrichmentError("Le rapport ne doit pas écraser un fichier d'entrée.")
        metadata = load_metadata(args.metadata, args.expected_count)
        endpoint = search_endpoint(
            args.endpoint or os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "")
        )
        choice = os.environ.setdefault("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
        if choice not in {"AzureCliCredential", "dev"}:
            raise EnrichmentError("AZURE_TOKEN_CREDENTIALS doit être AzureCliCredential ou dev.")
    except (ValueError, OSError, UnicodeError) as exc:
        message = str(exc) if isinstance(exc, EnrichmentError) else "Fichier d'entrée illisible."
        print(f"Configuration : {message}", file=sys.stderr)
        return 2

    plans: list[IndexPlan] = []
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "status": "preflight",
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "metadata_documents": len(metadata),
        "metadata_without_fiscal_year": sum(item.fiscal_year is None for item in metadata.values()),
    }
    exit_code = 1
    try:
        # Le rapport doit être accessible avant toute écriture Azure.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with ExitStack() as stack:
            credential = stack.enter_context(DefaultAzureCredential())
            schema_client = stack.enter_context(
                SearchIndexClient(
                    endpoint,
                    credential,
                    api_version="2026-04-01",
                    connection_timeout=10,
                    read_timeout=30,
                    retry_total=2,
                )
            )
            clients: dict[str, SearchClient] = {}
            for name in indexes:
                print(f"Lecture complète : {name}", flush=True)
                validate_schema(schema_client.get_index(name))
                client = stack.enter_context(
                    SearchClient(
                        endpoint,
                        name,
                        credential,
                        api_version="2026-04-01",
                        connection_timeout=10,
                        read_timeout=30,
                        retry_total=2,
                    )
                )
                clients[name] = client
                plan = plan_index(client, name, metadata)
                plans.append(plan)
                print(
                    f"{name} : {plan.chunk_count} chunks, {len(plan.documents)} PDF, "
                    f"{len(plan.patches)} mises à jour prévues.",
                    flush=True,
                )
            if any(plan.sources != plans[0].sources for plan in plans[1:]):
                raise EnrichmentError("Les index ne désignent pas les mêmes sources PDF.")
            report["status"] = "ready"
            report["indexes"] = [plan.summary() for plan in plans]
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if args.apply:
                for plan in plans:
                    print(f"Mise à jour merge : {plan.index_name}", flush=True)
                    if plan.patches:
                        # Une interruption peut désormais laisser des mises à jour appliquées.
                        report["status"] = "applying"
                        report["azure_writes_attempted"] = True
                        report["indexes"] = [item.summary() for item in plans]
                        args.report.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    report["active_index"] = plan.index_name
                    report["phase"] = "merge"
                    apply_plan(clients[plan.index_name], plan, args.batch_size)
                    report["phase"] = "verification"
                    verify_plan(clients[plan.index_name], plan, metadata)
                report["status"] = "applied_and_verified"
            else:
                report["status"] = "dry_run_complete"
            exit_code = 0
    except AzureError as exc:
        report["error"] = azure_error_message(exc)
    except (EnrichmentError, OSError) as exc:
        report["error"] = str(exc) if isinstance(exc, EnrichmentError) else "Rapport inaccessible."
    finally:
        report["indexes"] = [plan.summary() for plan in plans]
        report["azure_writes_attempted"] = any(plan.write_attempted for plan in plans)
        if exit_code:
            report["status"] = (
                "partial_or_unknown" if report["azure_writes_attempted"] else "preflight_failed"
            )
        try:
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            print("Impossible d'enregistrer le rapport final.", file=sys.stderr)
            exit_code = 1
    if exit_code:
        print(str(report.get("error", "Enrichissement interrompu.")), file=sys.stderr)
    else:
        print(f"Rapport : {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
