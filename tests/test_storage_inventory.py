"""Inventaire simulé : aucun appel réseau ni PDF financier dans les tests."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from azure.core.exceptions import HttpResponseError
from azure.storage.blob import BlobProperties

from ai103.storage_inventory import inventory_pdf_blobs, main


def blob(name: str, size: int) -> BlobProperties:
    return BlobProperties(name=name, **{"Content-Length": size})


def test_complete_inventory_includes_nested_paths_uppercase_and_all_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Deux pages simulées, 368 PDF, des noms imbriqués et un blob non PDF.
    def pages() -> Iterator[BlobProperties]:
        for page in range(2):
            for index in range(184):
                yield blob(f"rapports/{page}/{index}.PDF", 1_000)
        yield blob("readme.txt", 10)

    result = inventory_pdf_blobs(pages())
    assert result.pdf_count == 368
    assert result.total_bytes == 368_000
    assert result.other_blob_count == 1
    assert "rapports/1/183.PDF" in capsys.readouterr().out


def test_failure_on_later_page_cannot_report_a_complete_inventory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://testaccount.blob.core.windows.net")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "financial-pdfs")
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")

    def incomplete_listing() -> Iterator[BlobProperties]:
        yield blob("premier.pdf", 1_000)
        error = HttpResponseError(message="sensitive-response-body")
        error.status_code = 403
        raise error

    with (
        patch("ai103.storage_inventory.DefaultAzureCredential"),
        patch("ai103.storage_inventory.BlobServiceClient") as client_class,
    ):
        service = client_class.return_value.__enter__.return_value
        container = service.get_container_client.return_value
        container.list_blobs.return_value = incomplete_listing()
        assert main(["--env-file", str(tmp_path / "absent.env")]) == 1
        container.download_blob.assert_not_called()
        client_class.return_value.__exit__.assert_called_once()
    output = capsys.readouterr()
    assert "Inventaire complet" not in output.out
    assert "403" in output.err and "scope" in output.err
    assert "sensitive-response-body" not in output.err


def test_missing_configuration_does_not_create_an_azure_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONTAINER_NAME", raising=False)
    with patch("ai103.storage_inventory.BlobServiceClient") as client_class:
        assert main(["--env-file", str(tmp_path / "absent.env")]) == 2
        client_class.assert_not_called()


def test_pdf_count_mismatch_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://testaccount.blob.core.windows.net")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "financial-pdfs")
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    with (
        patch("ai103.storage_inventory.DefaultAzureCredential"),
        patch("ai103.storage_inventory.BlobServiceClient") as client_class,
    ):
        service = client_class.return_value.__enter__.return_value
        service.get_container_client.return_value.list_blobs.return_value = [
            blob("unique.pdf", 1_000),
        ]
        assert main(["--env-file", str(tmp_path / "absent.env")]) == 1
    assert "différent" in capsys.readouterr().err


def test_success_keeps_exported_configuration_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AZURE_STORAGE_ACCOUNT_URL=https://otheraccount.blob.core.windows.net\n"
        "AZURE_STORAGE_CONTAINER_NAME=other-container\n"
        "AZURE_TOKEN_CREDENTIALS=dev\n"
    )
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://testaccount.blob.core.windows.net")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_NAME", "financial-pdfs")
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    with (
        patch("ai103.storage_inventory.DefaultAzureCredential"),
        patch("ai103.storage_inventory.BlobServiceClient") as client_class,
    ):
        service = client_class.return_value.__enter__.return_value
        service.get_container_client.return_value.list_blobs.return_value = (
            blob(f"rapport-{index}.pdf", 1_000_000) for index in range(368)
        )
        assert main(["--env-file", str(env_file)]) == 0
        assert client_class.call_args.kwargs["account_url"] == (
            "https://testaccount.blob.core.windows.net"
        )
        service.get_container_client.assert_called_once_with("financial-pdfs")
        service.get_container_client.return_value.download_blob.assert_not_called()
        assert os.environ["AZURE_TOKEN_CREDENTIALS"] == "AzureCliCredential"
    output = capsys.readouterr()
    assert "PDF : 368" in output.out
    assert "368000000 octets (0.368 Go)" in output.out
    assert not output.err
