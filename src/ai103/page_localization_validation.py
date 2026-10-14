"""Contrôler la localisation avec des passages pypdf de pages physiques connues."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from ai103.page_localization import METHOD_VERSION, PageMatcher, compact, write_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("docs/reports/financebench-84-documents.json")
    )
    parser.add_argument("--directory", type=Path, default=Path("docs/reports/page-localization"))
    args = parser.parse_args(argv)
    corpus = json.loads(args.corpus.read_text())["documents"]
    samples: list[dict[str, Any]] = []
    for number, doc in enumerate(corpus, 1):
        path = Path(doc["local_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(args.directory / "pdf-text-cache" / f"{digest}.json.gz", "rt") as stream:
            cache = json.load(stream)
        matcher = PageMatcher(cache["variants"])
        reader = PdfReader(path)
        positions = sorted({0, len(reader.pages) // 2, len(reader.pages) - 1})
        for position in positions:
            text = reader.pages[position].extract_text() or ""
            if len(compact(text)) < 120:
                continue
            start = max(0, len(text) // 3)
            passage = text[start : start + 1500]
            if len(compact(passage)) < 80:
                continue
            # Ajouter du balisage et altérer un mot interne ; les bornes restent intactes.
            words = passage.split()
            if len(words) > 50:
                words[len(words) // 2] = "syntheticOCRerror"
                transformed = "<p>" + " ".join(words) + "</p>"
            else:
                transformed = passage
            location = matcher.locate(transformed)
            actual = [location.page_start, location.page_end]
            expected = [position + 1, position + 1]
            status = (
                "abstained"
                if location.page_start is None
                else ("correct" if actual == expected else "wrong")
            )
            samples.append(
                {
                    "doc_name": doc["doc_name"],
                    "kind": "independent_pypdf_with_markup_and_ocr_noise",
                    "expected": expected,
                    "actual": actual,
                    "status": status,
                    "method": location.method,
                }
            )
        middle = len(reader.pages) // 2
        if middle + 1 < len(reader.pages):
            first = reader.pages[middle].extract_text() or ""
            second = reader.pages[middle + 1].extract_text() or ""
            if len(compact(first[-400:])) >= 80 and len(compact(second[:400])) >= 80:
                location = matcher.locate(first[-400:] + "\n" + second[:400])
                expected = [middle + 1, middle + 2]
                actual = [location.page_start, location.page_end]
                status = (
                    "abstained"
                    if location.page_start is None
                    else ("correct" if actual == expected else "wrong")
                )
                samples.append(
                    {
                        "doc_name": doc["doc_name"],
                        "kind": "independent_pypdf_cross_page",
                        "expected": expected,
                        "actual": actual,
                        "status": status,
                        "method": location.method,
                    }
                )
        print(f"[{number}/{len(corpus)}] Contrôle pypdf : {doc['doc_name']}", flush=True)
    counts = Counter(sample["status"] for sample in samples)
    report = {
        "method_version": METHOD_VERSION,
        "azure_writes": False,
        "scope": "independent_local_pypdf_excerpts_not_a_global_ground_truth_for_azure_chunks",
        "documents": len(corpus),
        "samples": len(samples),
        "counts": dict(counts),
        "checks": samples,
    }
    write_json(args.directory / "validation.json", report)
    print(
        json.dumps({"documents": len(corpus), "samples": len(samples), "counts": counts}),
        flush=True,
    )
    return int(counts["wrong"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
