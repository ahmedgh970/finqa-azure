"""Appliquer les pages proposées aux chunks existants, avec contrôles de provenance."""

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from dotenv import load_dotenv

from ai103.metadata_enrichment import (
    DEFAULT_INDEXES,
    EnrichmentError,
    IndexPlan,
    azure_error_message,
    doc_name_from_source,
    merge_batch,
    search_endpoint,
    validate_schema,
)
from ai103.page_localization import METHOD_VERSION, write_json

LIVE_FIELDS = [
    "chunk_id",
    "document_id",
    "source",
    "content",
    "company",
    "fiscal_year",
    "page_start",
    "page_end",
]
PAGE_FIELDS = ("page_start", "page_end")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class Bundle:
    index_name: str
    rows: dict[str, dict[str, Any]]
    snapshot_sha256: str


@dataclass
class PreparedIndex:
    plan: IndexPlan
    expected: dict[str, dict[str, Any]]
    unresolved: int

    def summary(self) -> dict[str, Any]:
        counts = {
            field: sum(row.get(field) is not None for row in self.expected.values())
            for field in ("page_start", "page_end", "company", "fiscal_year")
        }
        return {
            **self.plan.summary(),
            "unresolved_proposals": self.unresolved,
            "target_non_null_counts": counts,
        }


def load_bundles(directory: Path, corpus_path: Path, endpoint: str) -> list[Bundle]:
    """Lier propositions, audit, snapshot et originaux avant tout accès en écriture."""
    summary = json.loads((directory / "summary.json").read_text())
    corpus = json.loads(corpus_path.read_text())["documents"]
    names = {row["doc_name"] for row in corpus}
    if len(corpus) != 84 or len(names) != 84 or names != set(summary["pdfs"]):
        raise EnrichmentError("Le corpus et les empreintes doivent couvrir exactement 84 PDF.")
    for doc in corpus:
        proof = summary["pdfs"][doc["doc_name"]]
        if (
            proof["pages"] != doc["pages"]
            or hashlib.sha256(Path(doc["local_path"]).read_bytes()).hexdigest() != proof["sha256"]
        ):
            raise EnrichmentError("Un PDF original a changé depuis la localisation.")
    with gzip.open(directory / "chunks.json.gz", "rt", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if snapshot["endpoint"] != endpoint:
        raise EnrichmentError("Endpoint différent de celui du snapshot.")
    indexes = snapshot["indexes"]
    if len(indexes) != 2 or {item["index_name"] for item in indexes} != set(DEFAULT_INDEXES):
        raise EnrichmentError("Les deux index attendus doivent être présents.")
    bundles = []
    for index in indexes:
        name = index["index_name"]
        proposals = json.loads((directory / f"{name}.pages.json").read_text())
        audit = json.loads((directory / f"{name}.audit.json").read_text())
        digest = hashlib.sha256(json.dumps(index["chunks"], sort_keys=True).encode()).hexdigest()
        if digest != index["sha256"] or digest != proposals["snapshot_sha256"]:
            raise EnrichmentError("Empreintes du snapshot et des propositions incompatibles.")
        if (
            proposals["index_name"] != name
            or proposals["method_version"] != METHOD_VERSION
            or proposals["page_numbering"] != "physical_1_based_cover_included"
        ):
            raise EnrichmentError("Convention ou version de localisation incompatible.")
        original = {row["chunk_id"]: row for row in index["chunks"]}
        proposed = {row["chunk_id"]: row for row in proposals["chunks"]}
        audited = {row["chunk_id"]: row for row in audit}
        if (
            len(original) != index["chunk_count"]
            or len(original) != len(index["chunks"])
            or len(proposed) != len(proposals["chunks"])
            or len(audited) != len(audit)
            or set(original) != set(proposed)
            or set(original) != set(audited)
        ):
            raise EnrichmentError("Clés manquantes, supplémentaires ou répétées dans les preuves.")
        rows = {}
        for key, row in original.items():
            proof, proposal = audited[key], proposed[key]
            doc_name = doc_name_from_source(row["source"])
            start, end = (proposal.get(field) for field in PAGE_FIELDS)
            if set(proposal) != {"chunk_id", *PAGE_FIELDS} or doc_name not in names:
                raise EnrichmentError("Proposition avec champs ou source non autorisés.")
            if (
                proof["content_sha256"] != sha_text(row["content"])
                or proof["doc_name"] != doc_name
                or any(proof[field] != proposal[field] for field in PAGE_FIELDS)
            ):
                raise EnrichmentError("Une proposition ne correspond plus à son audit.")
            if start is None:
                if end is not None:
                    raise EnrichmentError("Plage non résolue partiellement renseignée.")
            elif (
                type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= summary["pdfs"][doc_name]["pages"]
                or not proof["method"].startswith(("normalized_exact:", "anchored_alignment:"))
            ):
                raise EnrichmentError("Plage ou méthode de localisation invalide.")
            rows[key] = {
                "source": row["source"],
                "content_sha256": proof["content_sha256"],
                "doc_name": doc_name,
                "page_start": start,
                "page_end": end,
            }
        if {row["doc_name"] for row in rows.values()} != names:
            raise EnrichmentError("Un index ne couvre plus les 84 PDF.")
        bundles.append(Bundle(name, rows, digest))
    return bundles


def validate_page_schema(index: SearchIndex) -> None:
    validate_schema(index)
    fields = {field.name: field for field in index.fields}
    for name in PAGE_FIELDS:
        if (
            name not in fields
            or fields[name].type != "Edm.Int32"
            or fields[name].retrievable is False
        ):
            raise EnrichmentError(f"Champ de page incompatible : {name}.")


def read_keys(client: SearchClient, keys: list[str]) -> dict[str, dict[str, Any]]:
    if not keys or len(keys) > 500 or any(not key or "|" in key for key in keys):
        raise EnrichmentError("Lot de clés invalide.")
    values = "|".join(keys).replace("'", "''")
    rows = list(
        client.search(
            search_text="*",
            filter=f"search.in(chunk_id, '{values}', '|')",
            select=LIVE_FIELDS,
            top=len(keys),
        )
    )
    found = {row["chunk_id"]: dict(row) for row in rows}
    if len(found) != len(rows) or set(found) != set(keys):
        raise EnrichmentError("Lecture par clés incomplète ou incohérente.")
    return found


def unchanged_content(row: dict[str, Any], expected: dict[str, Any]) -> None:
    content = row.get("content")
    if (
        row.get("source") != expected["source"]
        or not isinstance(content, str)
        or sha_text(content) != expected["content_sha256"]
    ):
        raise EnrichmentError("La source ou le contenu d'un chunk a changé ; arrêt.")
    for name in ("document_id", "company", "fiscal_year"):
        if name in expected and row.get(name) != expected[name]:
            raise EnrichmentError("Une métadonnée à préserver a changé ; arrêt.")


def prepare_index(client: SearchClient, bundle: Bundle) -> PreparedIndex:
    if client.get_document_count() != len(bundle.rows):
        raise EnrichmentError("Comptage Azure différent du snapshot.")
    keys = list(bundle.rows)
    expected: dict[str, dict[str, Any]] = {}
    patches = []
    documents: Counter[str] = Counter()
    sources = {}
    unresolved = 0
    for offset in range(0, len(keys), 500):
        live = read_keys(client, keys[offset : offset + 500])
        for key, row in live.items():
            proof = bundle.rows[key]
            unchanged_content(row, proof)
            target = {
                **proof,
                **{field: row.get(field) for field in ("document_id", "company", "fiscal_year")},
            }
            documents[proof["doc_name"]] += 1
            sources[proof["doc_name"]] = proof["source"]
            if proof["page_start"] is None:
                unresolved += 1
                target.update({field: row.get(field) for field in PAGE_FIELDS})
            else:
                for field in PAGE_FIELDS:
                    if row.get(field) not in (None, proof[field]):
                        raise EnrichmentError("Une page déjà renseignée contredit la proposition.")
                if any(row.get(field) != proof[field] for field in PAGE_FIELDS):
                    patches.append(
                        {"chunk_id": key, **{field: proof[field] for field in PAGE_FIELDS}}
                    )
            expected[key] = target
        print(
            f"Précontrôle {bundle.index_name} : {min(offset + 500, len(keys))}/{len(keys)}",
            flush=True,
        )
    if client.get_document_count() != len(keys):
        raise EnrichmentError("Comptage modifié pendant le précontrôle.")
    plan = IndexPlan(
        bundle.index_name,
        len(keys),
        dict(documents),
        patches,
        sources,
        {key: row["source"] for key, row in bundle.rows.items()},
    )
    return PreparedIndex(plan, expected, unresolved)


def apply_pages(client: SearchClient, prepared: PreparedIndex, batch_size: int = 100) -> None:
    if not 1 <= batch_size <= 500:
        raise EnrichmentError("Taille de lot de pages invalide.")
    plan = prepared.plan
    for offset in range(0, len(plan.patches), batch_size):
        batch = plan.patches[offset : offset + batch_size]
        if any(
            set(row) != {"chunk_id", *PAGE_FIELDS}
            or any(type(row[field]) is not int for field in PAGE_FIELDS)
            for row in batch
        ):
            raise EnrichmentError("Lot de pages non autorisé.")
        if offset:
            sleep(1)
        # Recontrôle juste avant chaque merge, afin de détecter une modification concurrente.
        live = read_keys(client, [row["chunk_id"] for row in batch])
        for patch in batch:
            row = live[patch["chunk_id"]]
            unchanged_content(row, prepared.expected[patch["chunk_id"]])
            if any(row.get(field) not in (None, patch[field]) for field in PAGE_FIELDS):
                raise EnrichmentError("Une page a changé depuis le précontrôle.")
        merge_batch(client, plan, batch)
        print(
            f"{plan.index_name} : {plan.applied_chunks}/{len(plan.patches)} pages acquittées",
            flush=True,
        )


def verify_pages(client: SearchClient, prepared: PreparedIndex) -> None:
    keys = list(prepared.expected)
    for offset in range(0, len(keys), 500):
        for attempt in range(5):
            live = read_keys(client, keys[offset : offset + 500])
            matches = True
            for key, row in live.items():
                target = prepared.expected[key]
                unchanged_content(row, target)
                matches = matches and all(row.get(field) == target[field] for field in PAGE_FIELDS)
            if matches:
                break
            if attempt == 4:
                raise EnrichmentError(
                    "Les pages ne sont pas encore confirmées après cinq lectures."
                )
            sleep(3)
        print(
            f"Vérification {prepared.plan.index_name} : {min(offset + 500, len(keys))}/{len(keys)}",
            flush=True,
        )
    if client.get_document_count() != len(keys):
        raise EnrichmentError("Le nombre de chunks a changé après application.")
    prepared.plan.verified = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("docs/reports/page-localization"))
    parser.add_argument(
        "--corpus", type=Path, default=Path("docs/reports/financebench-84-documents.json")
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--report", type=Path, default=Path("docs/reports/page-localization/enrichment.json")
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(override=False)
    os.environ.setdefault("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    prepared: list[PreparedIndex] = []
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "azure_writes_attempted": False,
        "status": "preflight",
    }
    result = 1
    report_safe = False
    try:
        if not 1 <= args.batch_size <= 500:
            raise EnrichmentError("Taille de lot invalide (1 à 500).")
        protected = [
            args.corpus,
            *args.directory.glob("*.pages.json"),
            *args.directory.glob("*.audit.json"),
            args.directory / "summary.json",
            args.directory / "validation.json",
            args.directory / "adobe-boundary-review.json",
            args.directory / "chunks.json.gz",
        ]
        if args.report.suffix != ".json" or args.report.resolve() in {
            path.resolve() for path in protected
        }:
            raise EnrichmentError("Le rapport ne doit pas écraser un fichier de preuve.")
        report_safe = True
        endpoint = search_endpoint(args.endpoint)
        bundles = load_bundles(args.directory, args.corpus, endpoint)
        report["snapshot_sha256"] = {
            bundle.index_name: bundle.snapshot_sha256 for bundle in bundles
        }
        write_json(args.report, report)
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
            clients = {}
            for bundle in bundles:
                validate_page_schema(schema_client.get_index(bundle.index_name))
                client = stack.enter_context(
                    SearchClient(
                        endpoint,
                        bundle.index_name,
                        credential,
                        api_version="2026-04-01",
                        connection_timeout=10,
                        read_timeout=30,
                        retry_total=2,
                    )
                )
                clients[bundle.index_name] = client
                item = prepare_index(client, bundle)
                prepared.append(item)
                print(
                    f"{bundle.index_name} : {len(item.plan.patches)} mises à jour, "
                    f"{item.unresolved} propositions non résolues ignorées",
                    flush=True,
                )
            report["status"] = "ready"
            report["indexes"] = [item.summary() for item in prepared]
            write_json(args.report, report)
            if args.apply:
                for item in prepared:
                    report["active_index"] = item.plan.index_name
                    report["phase"] = "merge"
                    report["status"] = "applying"
                    report["azure_writes_attempted"] = bool(item.plan.patches) or any(
                        row.plan.write_attempted for row in prepared
                    )
                    write_json(args.report, report)
                    apply_pages(clients[item.plan.index_name], item, args.batch_size)
                    report["phase"] = "verification"
                    verify_pages(clients[item.plan.index_name], item)
                    report["indexes"] = [item.summary() for item in prepared]
                    report["azure_writes_attempted"] = any(
                        row.plan.write_attempted for row in prepared
                    )
                    write_json(args.report, report)
                report["status"] = "applied_and_verified"
            else:
                report["status"] = "dry_run_complete"
            result = 0
    except AzureError as exc:
        report["error"] = azure_error_message(exc)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report["error"] = (
            str(exc)
            if isinstance(exc, EnrichmentError)
            else "Fichier de preuve ou rapport invalide/inaccessible."
        )
    finally:
        report["indexes"] = [item.summary() for item in prepared]
        report["azure_writes_attempted"] = any(item.plan.write_attempted for item in prepared)
        if result:
            report["status"] = (
                "partial_or_unknown" if report["azure_writes_attempted"] else "preflight_failed"
            )
        if report_safe:
            try:
                write_json(args.report, report)
            except OSError:
                print("Impossible d'enregistrer le rapport final.", file=sys.stderr)
                result = 1
    if result:
        print(report.get("error", "Traitement interrompu."), file=sys.stderr)
    print(f"Rapport : {args.report}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
