"""Compter les pages des PDF locaux et enregistrer un rapport JSON, sans appel Azure."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPageCount:
    """Une erreur reste explicite : elle ne devient jamais un compte de zéro page."""

    file: str
    pages: int | None
    error: str | None


def _raise_walk_error(error: OSError) -> None:
    raise error


def list_pdf_files(directory: Path) -> list[Path]:
    """Parcourir tous les sous-dossiers ; échouer si un dossier est inaccessible."""
    files = []
    for folder, _, names in os.walk(directory, onerror=_raise_walk_error, followlinks=False):
        for name in names:
            if Path(name).suffix.lower() == ".pdf":
                files.append(Path(folder) / name)
    return sorted(files)


def count_pdf_pages(path: Path, directory: Path) -> PdfPageCount:
    """Ouvrir le PDF en lecture seule, sans extraire le texte ou rendre les pages."""
    relative_name = path.relative_to(directory).as_posix()
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted and not reader.decrypt(""):
                return PdfPageCount(relative_name, None, "PDF protégé : mot de passe requis.")
            pages = len(reader.pages)
        if pages == 0:
            return PdfPageCount(relative_name, None, "PDF sans page : vérifier le fichier.")
        return PdfPageCount(relative_name, pages, None)
    except Exception as error:
        # Un fichier corrompu ou illisible ne doit pas empêcher d'examiner les suivants.
        return PdfPageCount(relative_name, None, f"{type(error).__name__}: {error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compter toutes les pages des PDF locaux et sauvegarder un rapport JSON."
    )
    parser.add_argument("directory", type=Path, help="Dossier local contenant les PDF")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/pdf-pages.json"), help="Rapport JSON"
    )
    parser.add_argument(
        "--expected-count", type=int, default=368, help="Nombre de PDF attendu (défaut : 368)"
    )
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"Dossier local introuvable : {directory}")
    if args.expected_count <= 0:
        parser.error("--expected-count doit être strictement positif.")
    if output.suffix.lower() != ".json":
        parser.error("--output doit désigner un fichier .json, jamais un PDF.")

    try:
        files = list_pdf_files(directory)
    except OSError as error:
        print(f"Parcours incomplet du dossier : {error}", file=sys.stderr)
        return 1

    results = []
    for index, path in enumerate(files, start=1):
        result = count_pdf_pages(path, directory)
        results.append(result)
        detail = f"{result.pages} pages" if result.error is None else f"ERREUR : {result.error}"
        print(f"[{index}/{len(files)}] {result.file!r} : {detail}", flush=True)

    failed = sum(result.error is not None for result in results)
    total_counted = sum(result.pages for result in results if result.pages is not None)
    complete = len(files) == args.expected_count and failed == 0
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "directory": str(directory),
        "expected_pdf_count": args.expected_count,
        "pdf_count": len(files),
        "counted_pdf_count": len(files) - failed,
        "failed_pdf_count": failed,
        "total_pages_counted": total_counted,
        "total_pages": total_counted if complete else None,
        "complete": complete,
        "files": [asdict(result) for result in results],
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Impossible d'enregistrer le rapport : {error}", file=sys.stderr)
        return 1

    print(f"\nPDF trouvés : {len(files)} / {args.expected_count} attendus ; erreurs : {failed}")
    print(f"Pages comptées : {total_counted}")
    print(f"Rapport : {output}")
    if not complete:
        print(
            "Rapport incomplet : le total du corpus n'est pas confirmé. "
            "Vérifier les erreurs et le nombre de PDF.",
            file=sys.stderr,
        )
        return 1
    print(f"Total confirmé du corpus : {total_counted} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
