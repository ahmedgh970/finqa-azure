"""Exporter les seuls champs utiles à la localisation ; aucune écriture Azure."""

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from ai103.metadata_enrichment import (
    DEFAULT_INDEXES,
    EnrichmentError,
    doc_name_from_source,
    load_metadata,
    search_endpoint,
)

FIELDS = ["chunk_id", "source", "content", "page_start", "page_end"]


def read_index(client: SearchClient, expected_names: set[str]) -> list[dict[str, Any]]:
    """Refuser une pagination incohérente plutôt que dédupliquer silencieusement."""
    count = client.get_document_count()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    names: Counter[str] = Counter()
    while len(rows) < count:
        result = client.search(
            search_text="*", select=FIELDS, top=1000, skip=len(rows), include_total_count=True
        )
        if result.get_count() != count:
            raise EnrichmentError("Le comptage a changé pendant l'export.")
        batch = list(result)
        if not batch:
            raise EnrichmentError("Export interrompu avant la fin de l'index.")
        for item in batch:
            key, source, content = (item.get(name) for name in FIELDS[:3])
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(source, str)
                or not source
                or not isinstance(content, str)
                or not content
            ):
                raise EnrichmentError("Clé, source ou contenu invalide.")
            if key in keys:
                raise EnrichmentError("Clé répétée pendant l'export ; aucune sortie publiée.")
            name = doc_name_from_source(source)
            if name not in expected_names:
                raise EnrichmentError("Source étrangère au corpus attendu.")
            names[name] += 1
            keys.add(key)
            rows.append({field: item.get(field) for field in FIELDS})
        print(f"Lecture seule : {len(rows)}/{count} chunks", flush=True)
    if len(rows) != count or client.get_document_count() != count or set(names) != expected_names:
        raise EnrichmentError("Export incomplet ou corpus modifié.")
    return sorted(rows, key=lambda row: row["chunk_id"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/page-localization/chunks.json.gz")
    )
    args = parser.parse_args(argv)
    load_dotenv(override=False)
    os.environ.setdefault("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    names = set(load_metadata(args.metadata))
    endpoint = search_endpoint(args.endpoint)
    snapshot: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "azure_writes": False,
        "indexes": [],
    }
    with DefaultAzureCredential() as credential:
        for name in DEFAULT_INDEXES:
            print(f"Export sans vecteurs : {name}", flush=True)
            with SearchClient(
                endpoint,
                name,
                credential,
                api_version="2026-04-01",
                retry_total=3,
                connection_timeout=10,
                read_timeout=60,
            ) as client:
                rows = read_index(client, names)
            digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            snapshot["indexes"].append(
                {"index_name": name, "chunk_count": len(rows), "sha256": digest, "chunks": rows}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False)
    temporary.replace(args.output)
    print(f"Snapshot complet : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
