"""Vrais petits PDF générés localement : aucun accès Azure ni corpus financier."""

import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from ai103.pdf_page_inventory import main


def make_pdf(path: Path, pages: int, password: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    if password is not None:
        writer.encrypt(password)
    with path.open("wb") as stream:
        writer.write(stream)


def test_counts_nested_uppercase_pdfs_and_preserves_originals(tmp_path: Path) -> None:
    directory = tmp_path / "corpus"
    first = directory / "first.pdf"
    second = directory / "nested" / "first.PDF"
    make_pdf(first, 2)
    make_pdf(second, 3, password="")
    (directory / "notes.txt").write_text("ignoré", encoding="utf-8")
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (first, second)}
    output = tmp_path / "reports" / "pages.json"

    assert main([str(directory), "--output", str(output), "--expected-count", "2"]) == 0
    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert report["total_pages"] == 5
    assert report["pdf_count"] == report["counted_pdf_count"] == 2
    assert report["failed_pdf_count"] == 0
    assert {entry["file"]: entry["pages"] for entry in report["files"]} == {
        "first.pdf": 2,
        "nested/first.PDF": 3,
    }
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in originals.items()
    )


def test_corrupt_and_password_protected_pdfs_do_not_hide_partial_total(tmp_path: Path) -> None:
    directory = tmp_path / "corpus"
    make_pdf(directory / "a-locked.pdf", 2, password="secret-test")
    (directory / "b-corrupt.pdf").write_bytes(b"%PDF-1.7\ninvalid")
    make_pdf(directory / "z-valid.pdf", 4)
    output = tmp_path / "pages.json"

    assert main([str(directory), "--output", str(output), "--expected-count", "3"]) == 1
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["total_pages"] is None
    assert report["total_pages_counted"] == 4
    assert report["failed_pdf_count"] == 2
    assert report["files"][-1]["pages"] == 4
    assert all(entry["pages"] is None and entry["error"] for entry in report["files"][:2])


@pytest.mark.parametrize("actual_count", [0, 1])
def test_missing_pdfs_cannot_confirm_the_expected_corpus(tmp_path: Path, actual_count: int) -> None:
    directory = tmp_path / "corpus"
    directory.mkdir()
    if actual_count:
        make_pdf(directory / "only.pdf", 2)
    output = tmp_path / "pages.json"
    assert main([str(directory), "--output", str(output), "--expected-count", "368"]) == 1
    report = json.loads(output.read_text())
    assert report["pdf_count"] == actual_count
    assert report["complete"] is False
    assert report["total_pages"] is None
