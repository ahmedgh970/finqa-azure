"""Localisation : pages physiques, ambiguïtés et exécution hors ligne."""

import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.search.documents import SearchClient
from pypdf import PdfWriter

from ai103.chunk_snapshot import FIELDS, read_index
from ai103.metadata_enrichment import DEFAULT_INDEXES, EnrichmentError
from ai103.page_localization import PageMatcher, compact, extract_document, main, occurrences


def passage(seed: str) -> str:
    return " ".join(f"{seed}{i} operating revenue growth" for i in range(45))


def test_physical_pages_include_cover_and_blank_pages() -> None:
    text = passage("unique")
    matcher = PageMatcher({"test": ["Cover", "", text, ""]})
    location = matcher.locate(text)
    assert (location.page_start, location.page_end) == (3, 3)


def test_chunk_spans_two_physical_pages() -> None:
    first, second = passage("alpha"), passage("beta")
    matcher = PageMatcher({"test": ["cover", first, second]})
    location = matcher.locate(first[-350:] + "\n" + second[:350])
    assert (location.page_start, location.page_end) == (2, 3)


def test_repeated_passage_on_different_pages_stays_null() -> None:
    text = passage("repeated")
    location = PageMatcher({"test": [text, passage("other"), text]}).locate(text)
    assert location.page_start is None and location.page_end is None
    assert location.method == "ambiguous_page_ranges"


def test_repetitions_on_the_same_page_have_unambiguous_page_metadata() -> None:
    text = passage("twice")
    location = PageMatcher({"test": [text + "\n" + text]}).locate(text)
    assert (location.page_start, location.page_end) == (1, 1)


def test_markup_and_whitespace_are_normalized_without_modifying_input() -> None:
    text = passage("table")
    azure = "<table><tr><td>" + text.replace(" ", "</td><td>") + "</td></tr></table>"
    original = azure
    location = PageMatcher({"test": ["cover", text]}).locate(azure)
    assert (location.page_start, location.page_end) == (2, 2)
    assert azure == original
    assert compact("ﬁnan-\ncial&nbsp;report", markup=True) == "financialreport"


def test_minor_internal_parsing_error_can_be_located_by_intact_boundaries() -> None:
    text = passage("financial")
    azure = text.replace("financial20", "incorrectOCR20")
    location = PageMatcher({"test": ["cover", text]}).locate(azure)
    assert (location.page_start, location.page_end) == (2, 2)
    assert location.method.startswith("anchored_alignment")
    assert location.query_coverage is not None and location.query_coverage >= 0.95


def test_matching_edges_do_not_validate_an_unrelated_body() -> None:
    text = passage("actual")
    unrelated = text[:150] + passage("invented") + text[-150:]
    location = PageMatcher({"test": [text]}).locate(unrelated)
    assert location.page_start is None


def test_boundary_missing_does_not_infer_a_page_from_internal_text() -> None:
    text = passage("body")
    location = PageMatcher({"test": [text]}).locate("Unknown beginning " + text)
    assert location.page_start is None


def test_disagreeing_extractors_and_short_chunks_abstain() -> None:
    text = passage("disagree")
    matcher = PageMatcher({"a": [text, "other"], "b": ["other", text]})
    assert matcher.locate(text).method == "ambiguous_page_ranges"
    assert matcher.locate("Annual report").method == "too_short"
    assert occurrences("abc" * 40, "abc") is None


def test_extraction_preserves_actual_pdf_blank_pages_and_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    writer.write(path)
    original = path.read_bytes()
    document = {"doc_name": "blank", "local_path": str(path), "bytes": len(original), "pages": 3}
    name, result = extract_document(document, tmp_path / "cache")
    assert name == "blank" and result["pages"] == 3
    assert set(result["variants"]) == {"reading_order", "layout", "content_stream"}
    assert all(
        len(pages) == 3 and all(not p.strip() for p in pages)
        for pages in result["variants"].values()
    )
    assert path.read_bytes() == original
    with patch(
        "ai103.page_localization.subprocess.run", side_effect=AssertionError("Cache attendu")
    ):
        assert extract_document(document, tmp_path / "cache")[1] == result


@pytest.mark.parametrize("problem", [None, "duplicate", "count", "source"])
def test_azure_export_is_read_only_and_rejects_incoherent_results(problem: str | None) -> None:
    client = MagicMock(spec=SearchClient)
    client.get_document_count.return_value = 2
    names = {"ACME_2023"}
    rows = [
        {
            "chunk_id": str(i),
            "source": "https://example.blob.core.windows.net/pdfs/ACME_2023.pdf",
            "content": passage(str(i)),
        }
        for i in range(2)
    ]
    if problem == "duplicate":
        rows[1]["chunk_id"] = "0"
    if problem == "source":
        names = {"OTHER"}
    result = MagicMock()
    result.get_count.return_value = 3 if problem == "count" else 2
    result.__iter__.return_value = iter(rows)
    client.search.return_value = result
    if problem is None:
        assert len(read_index(client, names)) == 2
    else:
        with pytest.raises(EnrichmentError):
            read_index(client, names)
    assert "content_vector" not in FIELDS
    client.merge_documents.assert_not_called()
    client.upload_documents.assert_not_called()
    client.delete_documents.assert_not_called()


def test_offline_cli_creates_all_keyed_proposals_and_audit_for_84_documents(tmp_path: Path) -> None:
    cache = tmp_path / "pdf-text-cache"
    cache.mkdir()
    documents = []
    chunks = []
    for i in range(84):
        name = f"ACME_{2000 + i}"
        local = tmp_path / f"{name}.pdf"
        local.write_bytes(f"fixture-{i}".encode())
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        text = passage(name)
        with gzip.open(cache / f"{digest}.json.gz", "wt") as stream:
            json.dump(
                {
                    "pdf_sha256": digest,
                    "pages": 2,
                    "variants": {
                        "reading_order": ["cover", text],
                        "layout": ["cover", text],
                        "content_stream": ["cover", text],
                    },
                },
                stream,
            )
        documents.append(
            {"doc_name": name, "local_path": str(local), "bytes": local.stat().st_size, "pages": 2}
        )
        chunks.append(
            {
                "chunk_id": str(i),
                "source": f"https://example.blob.core.windows.net/pdfs/{name}.pdf",
                "content": text,
            }
        )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"documents": documents}))
    snapshot = tmp_path / "snapshot.json.gz"
    indexes = [
        {
            "index_name": name,
            "chunk_count": 84,
            "sha256": hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest(),
            "chunks": chunks,
        }
        for name in DEFAULT_INDEXES
    ]
    with gzip.open(snapshot, "wt") as stream:
        json.dump({"created_at": "test", "indexes": indexes}, stream)
    with patch("socket.create_connection", side_effect=AssertionError("Aucun réseau autorisé")):
        assert (
            main(
                [
                    "--snapshot",
                    str(snapshot),
                    "--corpus",
                    str(corpus),
                    "--output-dir",
                    str(tmp_path),
                ]
            )
            == 0
        )
    for name in DEFAULT_INDEXES:
        output = json.loads((tmp_path / f"{name}.pages.json").read_text())
        assert output["azure_writes"] is False and len(output["chunks"]) == 84
        assert all(set(row) == {"chunk_id", "page_start", "page_end"} for row in output["chunks"])
        assert all(row["page_start"] == row["page_end"] == 2 for row in output["chunks"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert all(value["unresolved"] == 0 for value in summary["indexes"].values())
