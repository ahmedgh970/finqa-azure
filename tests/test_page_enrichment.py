"""Merge des seules pages : preuves, conservation et reprise idempotente."""

import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchField, SearchIndex
from azure.search.documents.models import IndexingResult

from ai103.metadata_enrichment import DEFAULT_INDEXES, EnrichmentError
from ai103.page_enrichment import (
    Bundle,
    apply_pages,
    load_bundles,
    main,
    prepare_index,
    sha_text,
    validate_page_schema,
    verify_pages,
)
from ai103.page_localization import METHOD_VERSION

ENDPOINT = "https://sample.search.windows.net"


def live_row(key: str, name: str = "ACME_2023") -> dict:
    return {
        "chunk_id": key,
        "document_id": name,
        "source": f"https://example.blob.core.windows.net/pdfs/{name}.pdf",
        "content": f"Financial original text for {key}",
        "company": "Acme",
        "fiscal_year": 2023,
        "page_start": None,
        "page_end": None,
    }


def bundle_for(rows: list[dict]) -> Bundle:
    proof = {
        row["chunk_id"]: {
            "source": row["source"],
            "content_sha256": sha_text(row["content"]),
            "doc_name": row["document_id"],
            "page_start": 2,
            "page_end": 3,
        }
        for row in rows
    }
    return Bundle("sample-index", proof, "test")


def client_for(rows: list[dict]) -> MagicMock:
    client = MagicMock(spec=SearchClient)
    client.get_document_count.return_value = len(rows)

    def search(**kwargs):
        assert "skip" not in kwargs and "content_vector" not in kwargs["select"]
        keys = (
            kwargs["filter"]
            .removeprefix("search.in(chunk_id, '")
            .removesuffix("', '|')")
            .split("|")
        )
        return [dict(row) for row in reversed(rows) if row["chunk_id"] in keys]

    def merge(documents, **kwargs):
        assert kwargs == {"retry_total": 0}
        results = []
        for doc in documents:
            assert set(doc) == {"chunk_id", "page_start", "page_end"}
            row = next(row for row in rows if row["chunk_id"] == doc["chunk_id"])
            row.update(doc)
            results.append(IndexingResult(key=doc["chunk_id"], succeeded=True, status_code=200))
        return results

    client.search.side_effect = search
    client.merge_documents.side_effect = merge
    return client


def schema() -> SearchIndex:
    return SearchIndex(
        name="sample-index",
        fields=[
            SearchField(name="chunk_id", type="Edm.String", key=True, filterable=True),
            SearchField(name="source", type="Edm.String"),
            SearchField(name="company", type="Edm.String", filterable=True),
            SearchField(name="fiscal_year", type="Edm.Int32", filterable=True),
            SearchField(name="page_start", type="Edm.Int32"),
            SearchField(name="page_end", type="Edm.Int32"),
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                stored=True,
                vector_search_dimensions=1024,
            ),
        ],
    )


def test_pages_merge_preserves_unresolved_and_other_fields_and_is_idempotent() -> None:
    rows = [live_row(str(i)) for i in range(3)]
    proof = bundle_for(rows)
    proof.rows["2"].update(page_start=None, page_end=None)
    rows[1].update(page_start=2, page_end=3)
    rows[2].update(page_start=9, page_end=9)
    original = [dict(row) for row in rows]
    client = client_for(rows)
    prepared = prepare_index(client, proof)
    assert prepared.plan.patches == [{"chunk_id": "0", "page_start": 2, "page_end": 3}]
    assert prepared.unresolved == 1
    apply_pages(client, prepared)
    verify_pages(client, prepared)
    assert prepared.plan.verified and prepared.plan.applied_chunks == 1
    assert rows[2]["page_start"] == 9
    for before, after in zip(original, rows, strict=True):
        assert {k: v for k, v in before.items() if k not in ("page_start", "page_end")} == {
            k: v for k, v in after.items() if k not in ("page_start", "page_end")
        }
    assert prepare_index(client, proof).plan.patches == []
    client.upload_documents.assert_not_called()
    client.merge_or_upload_documents.assert_not_called()


@pytest.mark.parametrize("problem", ["content", "source", "page", "count", "missing-key"])
def test_changed_provenance_blocks_preflight(problem: str) -> None:
    rows = [live_row("a")]
    proof = bundle_for(rows)
    client = client_for(rows)
    if problem == "content":
        rows[0]["content"] += " changed"
    elif problem == "source":
        rows[0]["source"] = "https://example.blob.core.windows.net/pdfs/OTHER.pdf"
    elif problem == "page":
        rows[0]["page_start"] = 99
    elif problem == "count":
        client.get_document_count.return_value = 2
    else:
        client.search.side_effect = lambda **kwargs: []
    with pytest.raises(EnrichmentError):
        prepare_index(client, proof)
    client.merge_documents.assert_not_called()


def test_change_after_preflight_blocks_write_and_verification_checks_company() -> None:
    rows = [live_row("a")]
    client = client_for(rows)
    prepared = prepare_index(client, bundle_for(rows))
    rows[0]["company"] = "Changed"
    with pytest.raises(EnrichmentError, match="préserver"):
        apply_pages(client, prepared)
    client.merge_documents.assert_not_called()
    with pytest.raises(EnrichmentError, match="préserver"):
        verify_pages(client, prepared)


def test_partial_failure_retains_acknowledged_pages_and_stops() -> None:
    rows = [live_row("a"), live_row("b")]
    client = client_for(rows)
    prepared = prepare_index(client, bundle_for(rows))
    client.merge_documents.side_effect = [
        [
            IndexingResult(key="a", succeeded=True, status_code=200),
            IndexingResult(key="b", succeeded=False, status_code=429, error_message="secret"),
        ]
    ]
    with pytest.raises(EnrichmentError):
        apply_pages(client, prepared)
    assert prepared.plan.applied_chunks == 1 and not prepared.plan.verified
    assert prepared.plan.failures == [{"key": "b", "status_code": 429}]
    assert "secret" not in json.dumps(prepared.summary())


def test_page_schema_requires_stored_vectors_and_integer_pages() -> None:
    validate_page_schema(schema())
    bad = schema()
    bad.fields[-1].stored = False
    with pytest.raises(EnrichmentError):
        validate_page_schema(bad)
    bad = schema()
    bad.fields[-2].type = "Edm.String"
    with pytest.raises(EnrichmentError):
        validate_page_schema(bad)


def proof_files(tmp_path: Path) -> tuple[Path, list[list[dict]]]:
    documents, pdfs, all_rows = [], {}, []
    for i in range(84):
        name = f"ACME_{2000 + i}"
        pdf = tmp_path / f"{name}.pdf"
        pdf.write_bytes(name.encode())
        documents.append({"doc_name": name, "local_path": str(pdf), "pages": 10})
        pdfs[name] = {"pages": 10, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"documents": documents}))
    (tmp_path / "summary.json").write_text(json.dumps({"pdfs": pdfs}))
    indexes = []
    for index_name in DEFAULT_INDEXES:
        rows = [live_row(f"{index_name}-{i}", doc["doc_name"]) for i, doc in enumerate(documents)]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        indexes.append(
            {"index_name": index_name, "sha256": digest, "chunk_count": 84, "chunks": rows}
        )
        proposed = [{"chunk_id": row["chunk_id"], "page_start": 2, "page_end": 3} for row in rows]
        audit = [
            {
                **proposal,
                "doc_name": row["document_id"],
                "content_sha256": sha_text(row["content"]),
                "method": "normalized_exact:test",
            }
            for proposal, row in zip(proposed, rows, strict=True)
        ]
        (tmp_path / f"{index_name}.pages.json").write_text(
            json.dumps(
                {
                    "index_name": index_name,
                    "method_version": METHOD_VERSION,
                    "page_numbering": "physical_1_based_cover_included",
                    "snapshot_sha256": digest,
                    "chunks": proposed,
                }
            )
        )
        (tmp_path / f"{index_name}.audit.json").write_text(json.dumps(audit))
        all_rows.append(rows)
    with gzip.open(tmp_path / "chunks.json.gz", "wt") as stream:
        json.dump({"endpoint": ENDPOINT, "indexes": indexes}, stream)
    return corpus, all_rows


@pytest.mark.parametrize("mode", ["dry-run", "apply", "second-preflight-fails"])
def test_cli_validates_both_indexes_before_writes(tmp_path: Path, mode: str) -> None:
    corpus, rows = proof_files(tmp_path)
    clients = [client_for(group) for group in rows]
    if mode == "second-preflight-fails":
        rows[1][0]["content"] = "Changed"
    report = tmp_path / "enrichment.json"
    with (
        patch("ai103.page_enrichment.DefaultAzureCredential"),
        patch("ai103.page_enrichment.SearchIndexClient") as schemas,
        patch("ai103.page_enrichment.SearchClient") as client_class,
        patch("ai103.page_enrichment.sleep"),
        patch("ai103.metadata_enrichment.sleep"),
    ):
        schemas.return_value.__enter__.return_value.get_index.return_value = schema()
        contexts = [MagicMock(), MagicMock()]
        for context, client in zip(contexts, clients, strict=True):
            context.__enter__.return_value = client
        client_class.side_effect = contexts
        result = main(
            [
                "--directory",
                str(tmp_path),
                "--corpus",
                str(corpus),
                "--endpoint",
                ENDPOINT,
                "--report",
                str(report),
                "--apply" if mode != "dry-run" else "--dry-run",
            ]
        )
    data = json.loads(report.read_text())
    if mode == "apply":
        assert result == 0 and data["status"] == "applied_and_verified"
        assert all(item["verified"] and item["applied_chunks"] == 84 for item in data["indexes"])
    else:
        assert result == (1 if mode == "second-preflight-fails" else 0)
        assert not data["azure_writes_attempted"]
        for client in clients:
            client.merge_documents.assert_not_called()


def test_invalid_proof_and_protected_report_never_overwrite_inputs(tmp_path: Path) -> None:
    corpus, _ = proof_files(tmp_path)
    target = tmp_path / "summary.json"
    before = target.read_bytes()
    assert (
        main(
            [
                "--directory",
                str(tmp_path),
                "--corpus",
                str(corpus),
                "--endpoint",
                ENDPOINT,
                "--report",
                str(target),
            ]
        )
        == 1
    )
    assert target.read_bytes() == before
    proposal = tmp_path / f"{DEFAULT_INDEXES[0]}.pages.json"
    data = json.loads(proposal.read_text())
    data["chunks"][0]["page_start"] = 0
    proposal.write_text(json.dumps(data))
    with pytest.raises(EnrichmentError):
        load_bundles(tmp_path, corpus, ENDPOINT)
