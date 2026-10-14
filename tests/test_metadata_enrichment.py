"""Mises à jour partielles simulées : aucun accès Azure réel."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchField, SearchIndex
from azure.search.documents.models import IndexingResult

from ai103.metadata_enrichment import (
    DocumentMetadata,
    EnrichmentError,
    apply_plan,
    azure_error_message,
    doc_name_from_source,
    load_metadata,
    main,
    plan_index,
    retry_after_seconds,
    validate_schema,
    verify_plan,
)


@pytest.fixture(autouse=True)
def no_real_wait():
    with patch("ai103.metadata_enrichment.sleep") as wait:
        yield wait


def http_error(code: int, message: str, headers: dict[str, str] | None = None) -> HttpResponseError:
    error = HttpResponseError(message=message)
    error.status_code = code
    error.response = MagicMock(headers=headers or {})
    return error


def schema(*, stored: bool = True) -> SearchIndex:
    return SearchIndex(
        name="sample-index",
        fields=[
            SearchField(
                name="chunk_id", type="Edm.String", key=True, retrievable=True, filterable=True
            ),
            SearchField(name="source", type="Edm.String", retrievable=True),
            SearchField(name="company", type="Edm.String", filterable=True, retrievable=True),
            SearchField(name="fiscal_year", type="Edm.Int32", filterable=True, retrievable=True),
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                vector_search_dimensions=1024,
                stored=stored,
            ),
        ],
    )


def chunk(key: str, name: str = "ACME_2023_10K") -> dict[str, object]:
    return {
        "chunk_id": key,
        "source": f"https://example.blob.core.windows.net/pdfs/{name}.pdf",
        "company": None,
        "fiscal_year": None,
    }


def paged(rows: list[dict[str, object]], count: int) -> MagicMock:
    result = MagicMock()
    result.get_count.return_value = count
    result.__iter__.return_value = iter(rows)
    return result


def client_for(rows: list[dict[str, object]]) -> MagicMock:
    client = MagicMock(spec=SearchClient)
    client.get_document_count.return_value = len(rows)

    def search(**kwargs: object) -> MagicMock:
        if "filter" in kwargs:
            values = str(kwargs["filter"]).removeprefix("search.in(chunk_id, '")
            keys = values.removesuffix("', '|')").replace("''", "'").split("|")
            selected = [row for row in rows if row["chunk_id"] in keys]
            return paged(selected, len(selected))
        skip, top = int(str(kwargs["skip"])), int(str(kwargs["top"]))
        return paged(rows[skip : skip + top], len(rows))

    client.search.side_effect = search
    return client


def metadata_file(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_reference_preserves_null_and_fiscal_year_different_from_filename(tmp_path: Path) -> None:
    path = metadata_file(
        tmp_path,
        [
            {"doc_name": "ULTA_2023_10K", "company": "Ulta", "fiscal_year": 2022},
            {"doc_name": "PEPSI_2023_8K", "company": "PepsiCo", "fiscal_year": None},
        ],
    )
    original = path.read_bytes()
    result = load_metadata(path, expected_count=2)
    assert result["ULTA_2023_10K"].fiscal_year == 2022
    assert result["PEPSI_2023_8K"].fiscal_year is None
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "rows",
    [
        [{"doc_name": "ACME", "company": "Acme", "fiscal_year": True}],
        [{"doc_name": "ACME", "company": "Acme", "fiscal_year": "2023"}],
        [{"doc_name": "ACME", "company": "Acme", "fiscal_year": 2023, "answer": "gold"}],
        [{"doc_name": "ACME", "company": "Acme", "fiscal_year": 2023}] * 2,
        [],
    ],
)
def test_invalid_or_incomplete_reference_is_rejected(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    with pytest.raises(EnrichmentError):
        load_metadata(metadata_file(tmp_path, rows), expected_count=1)


def test_source_decoding_preserves_case_and_company_underscores() -> None:
    assert (
        doc_name_from_source(
            "https://example.blob.core.windows.net/pdfs/JOHNSON_JOHNSON_2022_10K.pdf"
        )
        == "JOHNSON_JOHNSON_2022_10K"
    )
    assert (
        doc_name_from_source("https://example.blob.core.windows.net/pdfs/Pfizer%20report.PDF")
        == "Pfizer report"
    )
    with pytest.raises(EnrichmentError):
        doc_name_from_source("https://example.blob.core.windows.net/pdfs/a.pdf?sig=secret")


def test_plan_reads_every_page_and_sends_only_allowed_fields() -> None:
    rows = [chunk(str(i)) for i in range(1003)]
    rows[-1].update(company="Acme", fiscal_year=2023)
    client = client_for(rows)
    plan = plan_index(client, "sample-index", {"ACME_2023_10K": DocumentMetadata("Acme", 2023)})
    assert plan.chunk_count == 1003 and len(plan.patches) == 1002
    assert plan.documents == {"ACME_2023_10K": 1003}
    assert [c.kwargs["skip"] for c in client.search.call_args_list] == [0, 1000]
    assert all(set(row) == {"chunk_id", "company", "fiscal_year"} for row in plan.patches)
    assert all("content" not in call.kwargs["select"] for call in client.search.call_args_list)
    client.merge_documents.assert_not_called()


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "incomplete", "changed"])
def test_incoherent_scan_never_becomes_a_plan(problem: str) -> None:
    rows = [chunk("1"), chunk("2")]
    metadata = {"ACME_2023_10K": DocumentMetadata("Acme", 2023)}
    if problem == "duplicate":
        rows[1]["chunk_id"] = "1"
    elif problem == "unknown":
        rows[1] = chunk("2", "OTHER_2023_10K")
    elif problem == "incomplete":
        metadata["MISSING_2023_10K"] = DocumentMetadata("Missing", 2023)
    client = client_for(rows)
    if problem == "changed":
        client.get_document_count.side_effect = [2, 3]
    with pytest.raises(EnrichmentError):
        plan_index(client, "sample-index", metadata)
    client.merge_documents.assert_not_called()


def test_stored_false_blocks_metadata_only_merge() -> None:
    validate_schema(schema())
    with pytest.raises(EnrichmentError, match="stored=true"):
        validate_schema(schema(stored=False))


def test_verification_requires_filterable_key() -> None:
    index = schema()
    index.fields[0].filterable = False
    with pytest.raises(EnrichmentError, match="filtrables"):
        validate_schema(index)


@pytest.mark.parametrize("problem", ["missing", "stale", "changed-source", "changed-count"])
def test_verification_checks_all_keys_and_retries_reads_only(problem: str) -> None:
    rows = [chunk("1"), chunk("2")]
    client = client_for(rows)
    metadata = {"ACME_2023_10K": DocumentMetadata("Acme", None)}
    plan = plan_index(client, "sample-index", metadata)
    for row in rows:
        row.update(company="Acme")
    if problem == "missing":
        rows.pop()
    elif problem == "stale":
        rows[1]["company"] = None
    elif problem == "changed-source":
        rows[1]["source"] = chunk("2", "OTHER")["source"]
    else:
        client.get_document_count.return_value = 3
    client.search.reset_mock()
    with pytest.raises(EnrichmentError):
        verify_plan(client, plan, metadata, attempts=2, retry_delay=0)
    assert not plan.verified
    assert client.search.call_count == (
        2 if problem in {"missing", "stale"} else int(problem == "changed-source")
    )
    client.merge_documents.assert_not_called()


def test_verification_accepts_delayed_visibility_without_repeating_merge() -> None:
    rows = [chunk("1")]
    client = client_for(rows)
    metadata = {"ACME_2023_10K": DocumentMetadata("Acme", None)}
    plan = plan_index(client, "sample-index", metadata)
    client.search.side_effect = [
        paged([chunk("1")], 1),
        paged([{**chunk("1"), "company": "Acme"}], 1),
    ]
    verify_plan(client, plan, metadata, attempts=2, retry_delay=0)
    assert plan.verified
    client.merge_documents.assert_not_called()


def test_partial_merge_failure_records_success_and_stops_later_batches() -> None:
    client = client_for([chunk("1"), chunk("2"), chunk("3")])
    plan = plan_index(client, "sample-index", {"ACME_2023_10K": DocumentMetadata("Acme", None)})
    client.merge_documents.side_effect = [
        [IndexingResult(key="1", succeeded=True, status_code=200)],
        [IndexingResult(key="2", succeeded=False, status_code=404, error_message="secret-body")],
    ]
    with pytest.raises(EnrichmentError):
        apply_plan(client, plan, batch_size=1)
    assert plan.write_attempted and plan.applied_chunks == 1
    assert plan.failures == [{"key": "2", "status_code": 404}]
    assert client.merge_documents.call_count == 2
    assert client.merge_documents.call_args_list[0].kwargs["documents"] == [
        {"chunk_id": "1", "company": "Acme", "fiscal_year": None}
    ]
    assert "secret-body" not in json.dumps(plan.summary())
    client.upload_documents.assert_not_called()
    client.merge_or_upload_documents.assert_not_called()


def test_http_429_retries_same_merge_respects_retry_after_and_counts_once(no_real_wait) -> None:
    client = client_for([chunk("1"), chunk("2")])
    metadata = {"ACME_2023_10K": DocumentMetadata("Acme", None)}
    plan = plan_index(client, "sample-index", metadata)
    client.merge_documents.side_effect = [
        http_error(429, "Too many requests secret-body", {"Retry-After": "7"}),
        [IndexingResult(key="1", succeeded=True, status_code=200)],
        [IndexingResult(key="2", succeeded=True, status_code=200)],
    ]
    apply_plan(client, plan, batch_size=1)
    assert plan.applied_chunks == 2 and plan.throttling_retries == 1
    calls = client.merge_documents.call_args_list
    assert calls[0] == calls[1] and calls[0].kwargs["retry_total"] == 0
    assert [call.args[0] for call in no_real_wait.call_args_list] == [7, 1]
    assert "secret-body" not in json.dumps(plan.summary())


@pytest.mark.parametrize(
    ("error", "calls", "waits"),
    [
        (http_error(429, "Too many requests"), 5, [2, 4, 8, 16]),
        (http_error(429, "Vector quota has been exceeded secret-body"), 1, []),
        (http_error(429, "Too many requests", {"Retry-After": "120"}), 1, []),
        (http_error(403, "secret-body"), 1, []),
    ],
)
def test_retries_are_bounded_and_quota_or_long_wait_stops(
    error, calls: int, waits: list[int], no_real_wait
) -> None:
    client = client_for([chunk("1"), chunk("2")])
    plan = plan_index(client, "sample-index", {"ACME_2023_10K": DocumentMetadata("Acme", None)})
    client.merge_documents.side_effect = error
    with pytest.raises(HttpResponseError):
        apply_plan(client, plan, batch_size=1)
    assert client.merge_documents.call_count == calls
    assert plan.applied_chunks == 0 and plan.write_attempted
    assert [call.args[0] for call in no_real_wait.call_args_list] == waits
    client.upload_documents.assert_not_called()


def test_retry_after_supports_milliseconds_http_date_and_invalid_values() -> None:
    assert retry_after_seconds(http_error(429, "", {"X-Ms-Retry-After-Ms": "2500"})) == 2.5
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    delay = retry_after_seconds(http_error(429, "", {"Retry-After": date}))
    assert delay is not None and 28 <= delay <= 30
    for value in ("invalid", "-1", "nan", "inf"):
        assert retry_after_seconds(http_error(429, "", {"Retry-After": value})) is None


@pytest.mark.parametrize(
    ("detail", "label"),
    [("vector quota exceeded", "vectoriel"), ("storage quota exceeded", "stockage")],
)
def test_quota_diagnostic_identifies_capacity_without_exposing_sdk_body(
    detail: str, label: str
) -> None:
    message = azure_error_message(http_error(429, detail + " secret-token"))
    assert label in message and "secret-token" not in message


@pytest.mark.parametrize(
    "mode", ["dry-run", "apply", "reordered-apply", "resume-apply", "second-index-fails"]
)
def test_cli_preflights_all_indexes_before_writes_and_verifies_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    path = metadata_file(
        tmp_path, [{"doc_name": "ACME_2023_10K", "company": "Acme", "fiscal_year": 2023}]
    )
    report = tmp_path / "report.json"
    # Chaque index conserve son propre contenu et ses propres clés.
    rows = [[chunk("native")], [chunk("layout")]]
    if mode == "resume-apply":
        rows[0][0].update(company="Acme", fiscal_year=2023)
        rows[1] = [chunk("layout-done"), chunk("layout-todo")]
        rows[1][0].update(company="Acme", fiscal_year=2023)
    if mode == "reordered-apply":
        rows = [[chunk(f"{name}-{i}") for i in range(1003)] for name in ("native", "layout")]
    clients = [client_for(group) for group in rows]
    for client, group in zip(clients, rows, strict=True):

        def merge(documents, group=group, **kwargs):
            assert kwargs == {"retry_total": 0}
            results = []
            for document in documents:
                assert set(document) == {"chunk_id", "company", "fiscal_year"}
                row = next(row for row in group if row["chunk_id"] == document["chunk_id"])
                row.update(document)
                results.append(IndexingResult(key=row["chunk_id"], succeeded=True, status_code=200))
            return results

        client.merge_documents.side_effect = merge
        if mode == "reordered-apply":
            original_search = client.search.side_effect

            def unstable_search(*, group=group, original_search=original_search, **kwargs):
                # Après les merge, chaque requête peut changer l'ordre interne.
                # Une relecture skip/top reverrait donc des clés déjà parcourues.
                if group[0]["company"] is not None:
                    group.reverse()
                return original_search(**kwargs)

            client.search.side_effect = unstable_search
    if mode == "second-index-fails":
        clients[1].search.side_effect = HttpResponseError(message="secret-token-in-body")
    with (
        patch("ai103.metadata_enrichment.DefaultAzureCredential"),
        patch("ai103.metadata_enrichment.SearchIndexClient") as schema_class,
        patch("ai103.metadata_enrichment.SearchClient") as client_class,
    ):
        schema_class.return_value.__enter__.return_value.get_index.return_value = schema()
        contexts = client_class.side_effect = [MagicMock(), MagicMock()]
        for context, client in zip(contexts, clients, strict=True):
            context.__enter__.return_value = client
        args = [
            "--endpoint",
            "https://sample.search.windows.net",
            "--metadata",
            str(path),
            "--expected-count",
            "1",
            "--report",
            str(report),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
        if mode != "dry-run":
            args.append("--apply")
        result = main(args)
    data = json.loads(report.read_text())
    assert "secret-token" not in report.read_text()
    if mode in {"apply", "reordered-apply", "resume-apply"}:
        assert result == 0 and data["status"] == "applied_and_verified"
        assert all(item["verified"] for item in data["indexes"])
        assert all(row["company"] == "Acme" for group in rows for row in group)
        if mode == "resume-apply":
            clients[0].merge_documents.assert_not_called()
            clients[1].merge_documents.assert_called_once()
            assert (
                clients[1].merge_documents.call_args.kwargs["documents"][0]["chunk_id"]
                == "layout-todo"
            )
    else:
        assert result == (1 if mode == "second-index-fails" else 0)
        assert data["azure_writes_attempted"] is False
        for client in clients:
            client.merge_documents.assert_not_called()
