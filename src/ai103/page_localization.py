"""Proposer des pages physiques pour les chunks, entièrement hors ligne et sans merge."""

import argparse
import bisect
import gzip
import hashlib
import html
import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai103.metadata_enrichment import DEFAULT_INDEXES, EnrichmentError, doc_name_from_source

METHOD_VERSION = "physical-page-alignment-v2"
MIN_CHARACTERS = 80
EDGE_CHARACTERS = 48
MIN_QUERY_COVERAGE = 0.95
MIN_WINDOW_COVERAGE = 0.85
MAX_PAGE_SPAN = 8


def clean_text(text: str, *, markup: bool = False) -> str:
    """Normaliser pour la localisation uniquement ; ne jamais réécrire le contenu Azure."""
    if markup:
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
        text = html.unescape(text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = unicodedata.normalize("NFKD", text).casefold()
    return "".join(char for char in text if not unicodedata.combining(char))


def compact(text: str, *, markup: bool = False) -> str:
    return re.sub(r"[^a-z0-9]", "", clean_text(text, markup=markup))


def tokens(text: str, *, markup: bool = False) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_text(text, markup=markup))


@dataclass(frozen=True)
class Location:
    page_start: int | None
    page_end: int | None
    method: str
    query_coverage: float | None = None
    window_coverage: float | None = None


def occurrences(text: str, needle: str, limit: int = 32) -> list[int] | None:
    found: list[int] = []
    offset = 0
    while (offset := text.find(needle, offset)) >= 0:
        found.append(offset)
        if len(found) > limit:
            return None
        offset += 1
    return found


class PageMatcher:
    """Plusieurs ordres d'extraction du PDF ; abstention si les plages divergent."""

    def __init__(self, variants: dict[str, list[str]]) -> None:
        if not variants or len({len(pages) for pages in variants.values()}) != 1:
            raise EnrichmentError("Extractions PDF de longueurs incompatibles.")
        self.variants: list[tuple[str, str, list[int], list[str]]] = []
        for name, pages in variants.items():
            parts = [compact(page) for page in pages]
            offsets = [0]
            for part in parts:
                offsets.append(offsets[-1] + len(part))
            self.variants.append((name, "".join(parts), offsets, pages))

    @staticmethod
    def page_at(offsets: list[int], position: int) -> int:
        # Les offsets répétés des pages blanches sont conservés, sans renumérotation.
        return bisect.bisect_right(offsets, position)

    def locate(self, content: str) -> Location:
        query = compact(content, markup=True)
        if len(query) < MIN_CHARACTERS:
            return Location(None, None, "too_short")
        query_tokens = tokens(content, markup=True)
        candidates: dict[tuple[int, int], Location] = {}
        saw_edges = False
        for name, document, offsets, pages in self.variants:
            exact = occurrences(document, query)
            if exact is None:
                return Location(None, None, "ambiguous_repetition")
            if exact:
                for start in exact:
                    end = start + len(query)
                    first, last = self.page_at(offsets, start), self.page_at(offsets, end - 1)
                    if last - first < MAX_PAGE_SPAN:
                        candidates[first, last] = Location(
                            first, last, f"normalized_exact:{name}", 1, 1
                        )
                continue
            heads = occurrences(document, query[:EDGE_CHARACTERS])
            tails = occurrences(document, query[-EDGE_CHARACTERS:])
            if heads is None or tails is None:
                return Location(None, None, "ambiguous_edges")
            saw_edges = saw_edges or bool(heads and tails)
            for start in heads:
                for tail in tails:
                    end = tail + EDGE_CHARACTERS
                    length = end - start
                    if not 0.6 * len(query) <= length <= 1.8 * len(query) + 256:
                        continue
                    first, last = self.page_at(offsets, start), self.page_at(offsets, end - 1)
                    if last - first >= MAX_PAGE_SPAN:
                        continue
                    # Retrouver aussi les espaces du passage pour contrôler son contenu.
                    window = "\n".join(pages[first - 1 : last])
                    window_tokens = tokens(window)
                    # Restreindre aux tokens dont le texte compact couvre les deux bornes.
                    # Les offsets sont calculés sur la même normalisation alphanumérique.
                    relative_start = start - offsets[first - 1]
                    relative_end = end - offsets[first - 1]
                    selected: list[str] = []
                    cursor = 0
                    for token in window_tokens:
                        token_end = cursor + len(token)
                        if token_end > relative_start and cursor < relative_end:
                            selected.append(token)
                        cursor = token_end
                    matched = sum(
                        block.size
                        for block in SequenceMatcher(
                            None, query_tokens, selected, autojunk=False
                        ).get_matching_blocks()
                    )
                    coverage = matched / max(len(query_tokens), 1)
                    precision = matched / max(len(selected), 1)
                    if coverage >= MIN_QUERY_COVERAGE and precision >= MIN_WINDOW_COVERAGE:
                        previous = candidates.get((first, last))
                        location = Location(
                            first,
                            last,
                            f"anchored_alignment:{name}",
                            round(coverage, 5),
                            round(precision, 5),
                        )
                        if previous is None or (previous.query_coverage or 0) < coverage:
                            candidates[first, last] = location
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        reason = (
            "ambiguous_page_ranges"
            if candidates
            else ("low_alignment_coverage" if saw_edges else "unmatched_boundaries")
        )
        return Location(None, None, reason)


def extract_document(document: dict[str, Any], cache: Path) -> tuple[str, dict[str, Any]]:
    """Lire les originaux existants ; jamais copier ni télécharger de PDF."""
    path = Path(document["local_path"])
    if path.stat().st_size != document["bytes"]:
        raise EnrichmentError(f"Taille locale modifiée : {document['doc_name']}.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = cache / f"{digest}.json.gz"
    variants: dict[str, list[str]] = {}
    if target.exists():
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            result = json.load(stream)
        if result["pages"] != document["pages"] or result["pdf_sha256"] != digest:
            raise EnrichmentError("Cache PDF incohérent.")
        variants = result["variants"]
        if any(
            len(pages) != document["pages"] or any(not isinstance(page, str) for page in pages)
            for pages in variants.values()
        ):
            raise EnrichmentError("Pages du cache PDF incohérentes.")
    for name, flags in (
        ("reading_order", []),
        ("layout", ["-layout"]),
        ("content_stream", ["-raw"]),
    ):
        if name in variants:
            continue
        process = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-eol", "unix", *flags, str(path), "-"],
            capture_output=True,
            timeout=180,
            check=True,
        )
        pages = process.stdout.decode("utf-8").split("\f")
        if pages and pages[-1] == "":
            pages.pop()
        if len(pages) != document["pages"]:
            raise EnrichmentError(f"Pages extraites incohérentes : {document['doc_name']}.")
        variants[name] = pages
    result = {"pdf_sha256": digest, "pages": document["pages"], "variants": variants}
    cache.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
    return document["doc_name"], result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("docs/reports/page-localization/chunks.json.gz")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("docs/reports/financebench-84-documents.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports/page-localization"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if shutil.which("pdftotext") is None or not 1 <= args.workers <= 8:
        parser.error("pdftotext requis ; workers doit être compris entre 1 et 8.")
    corpus = json.loads(args.corpus.read_text())["documents"]
    names = [item["doc_name"] for item in corpus]
    if len(names) != 84 or len(set(names)) != 84:
        raise EnrichmentError("Le manifeste doit contenir les 84 PDF distincts.")
    with gzip.open(args.snapshot, "rt", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if {item["index_name"] for item in snapshot["indexes"]} != set(DEFAULT_INDEXES):
        raise EnrichmentError("Les deux index attendus doivent être présents.")
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for index in snapshot["indexes"]:
        keys = [row["chunk_id"] for row in index["chunks"]]
        if len(set(keys)) != len(keys) or len(keys) != index["chunk_count"]:
            raise EnrichmentError("Snapshot incomplet ou clés répétées.")
        digest = hashlib.sha256(json.dumps(index["chunks"], sort_keys=True).encode()).hexdigest()
        if digest != index["sha256"]:
            raise EnrichmentError("Empreinte du snapshot différente.")
        observed = set()
        for row in index["chunks"]:
            name = doc_name_from_source(row["source"])
            observed.add(name)
            groups[name].append((index["index_name"], row))
        if observed != set(names):
            raise EnrichmentError("Les sources ne correspondent pas aux 84 PDF.")
    print("Extraction locale des 84 PDF, avec conservation des pages physiques.", flush=True)
    cache = args.output_dir / "pdf-text-cache"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        extracted = dict(executor.map(lambda doc: extract_document(doc, cache), corpus))
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in DEFAULT_INDEXES}
    audits: dict[str, list[dict[str, Any]]] = {name: [] for name in DEFAULT_INDEXES}
    for number, name in enumerate(names, 1):
        pdf = extracted[name]
        matcher = PageMatcher(pdf["variants"])
        resolved = 0
        for index_name, row in groups[name]:
            location = matcher.locate(row["content"])
            outputs[index_name].append(
                {
                    "chunk_id": row["chunk_id"],
                    "page_start": location.page_start,
                    "page_end": location.page_end,
                }
            )
            audits[index_name].append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_name": name,
                    "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                    **asdict(location),
                }
            )
            resolved += location.page_start is not None
        print(f"[{number}/84] {name} : {resolved}/{len(groups[name])} chunks localisés", flush=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "method_version": METHOD_VERSION,
        "status": "proposals_only_pending_review",
        "azure_writes": False,
        "page_numbering": "physical_1_based_cover_included",
        "snapshot_created_at": snapshot["created_at"],
        "thresholds": {
            "min_characters": MIN_CHARACTERS,
            "edge_characters": EDGE_CHARACTERS,
            "query_coverage": MIN_QUERY_COVERAGE,
            "window_coverage": MIN_WINDOW_COVERAGE,
            "max_page_span": MAX_PAGE_SPAN,
        },
        "pdfs": {
            name: {"sha256": item["pdf_sha256"], "pages": item["pages"]}
            for name, item in extracted.items()
        },
        "indexes": {},
    }
    for index in snapshot["indexes"]:
        name = index["index_name"]
        rows = sorted(outputs[name], key=lambda row: row["chunk_id"])
        detail = sorted(audits[name], key=lambda row: row["chunk_id"])
        counts = Counter(item["method"].split(":")[0] for item in detail)
        found = sum(row["page_start"] is not None for row in rows)
        summary["indexes"][name] = {
            "total": len(rows),
            "located": found,
            "unresolved": len(rows) - found,
            "methods": dict(counts),
        }
        write_json(
            args.output_dir / f"{name}.pages.json",
            {
                "index_name": name,
                "method_version": METHOD_VERSION,
                "status": "proposals_only_pending_review",
                "azure_writes": False,
                "page_numbering": summary["page_numbering"],
                "snapshot_sha256": index["sha256"],
                "chunks": rows,
            },
        )
        write_json(args.output_dir / f"{name}.audit.json", detail)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["indexes"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
