"""Rapprochement offline FinanceBench/chunks ; suggestions séparées des jugements."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai103.metadata_enrichment import doc_name_from_source
from ai103.retriever.benchmark import atomic_json, load_questions
from ai103.retriever.evaluation import tokens

METHOD = "evidence-chunk-candidates-v1"
REVIEW_FIELDS = [
    "candidate_id",
    "index",
    "question_id",
    "evidence_id",
    "chunk_id",
    "automatic_grade",
    "reviewed_grade",
    "reviewer",
    "notes",
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_id(question_id: str, position: int) -> str:
    return f"{question_id}:e{position}"


@dataclass
class Passage:
    row: dict[str, Any]
    doc_name: str
    words: list[str]
    text: str
    vocabulary: set[str]


def shingles(words: Sequence[str], size: int = 5) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def prepare_snapshot(snapshot: dict[str, Any]) -> dict[str, dict[str, list[Passage]]]:
    indexes: dict[str, dict[str, list[Passage]]] = {}
    for index in snapshot["indexes"]:
        name = index["index_name"]
        if name in indexes:
            raise ValueError("Index répété dans le snapshot.")
        groups: dict[str, list[Passage]] = defaultdict(list)
        seen = set()
        for row in index["chunks"]:
            if row["chunk_id"] in seen:
                raise ValueError("Clé répétée dans le snapshot.")
            seen.add(row["chunk_id"])
            doc_name = doc_name_from_source(row["source"])
            words = tokens(row["content"])
            groups[doc_name].append(
                Passage(
                    row,
                    doc_name,
                    words,
                    " " + " ".join(words) + " ",
                    set(words),
                )
            )
        if len(seen) != index["chunk_count"]:
            raise ValueError("Comptage du snapshot incohérent.")
        indexes[name] = groups
    return indexes


def propose(
    question: dict[str, Any],
    position: int,
    passages: Sequence[Passage],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Garder tous les exacts ; compléter par les meilleurs candidats lexicaux."""
    evidence = question["evidence"][position]
    words = tokens(evidence["evidence_text"])
    if not words or limit < 1:
        raise ValueError("Preuve tokenisée non vide et limite positive requises.")
    vocabulary, grams = set(words), shingles(words)
    needle = " " + " ".join(words) + " "
    scored = []
    for passage in passages:
        if passage.doc_name != evidence["doc_name"]:
            raise ValueError("Une proposition doit être limitée au document gold.")
        exact = needle in passage.text
        coverage = len(vocabulary & passage.vocabulary) / len(vocabulary)
        gram_coverage = len(grams & shingles(passage.words)) / len(grams) if grams else 0.0
        scored.append((exact, gram_coverage, coverage, passage))
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].row["chunk_id"]))
    selected = [item for rank, item in enumerate(scored) if item[0] or rank < limit]
    proposals = []
    for exact, gram_coverage, coverage, passage in selected:
        alignment = SequenceMatcher(None, words, passage.words, autojunk=True)
        blocks = [block for block in alignment.get_matching_blocks() if block.size]
        longest = max(blocks, key=lambda block: block.size, default=None)
        start, size = (longest.a, longest.size) if longest else (0, 0)
        eid = evidence_id(question["financebench_id"], position)
        cid = digest(json.dumps([eid, passage.row["chunk_id"]]))
        proposals.append(
            {
                "candidate_id": cid,
                "question_id": question["financebench_id"],
                "question": question["question"],
                "evidence_id": eid,
                "evidence_position": position,
                "doc_name": evidence["doc_name"],
                "gold_page_num_raw": evidence.get("evidence_page_num"),
                "evidence_text": evidence["evidence_text"],
                "evidence_sha256": digest(evidence["evidence_text"]),
                "chunk_id": passage.row["chunk_id"],
                "source": passage.row["source"],
                "content": passage.row["content"],
                "content_sha256": digest(passage.row["content"]),
                "automatic_grade": 2 if exact else None,
                "judgment_status": "automatic_exact_text" if exact else "unjudged",
                "unique_token_coverage": coverage,
                "fivegram_coverage": gram_coverage,
                "longest_matching_tokens": size,
                "gold_matching_excerpt": " ".join(words[start : start + size]),
                "reason": "Entire normalized evidence occurs in this chunk; not human-reviewed."
                if exact
                else "Lexical candidate only; no relevance grade inferred.",
            }
        )
    return proposals


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def export_reviewed(directory: Path, reviewed_csv: Path) -> int:
    """Exporter uniquement des jugements explicites, liés aux candidats immuables."""
    manifest = json.loads((directory / "manifest.json").read_text())
    candidates_path = directory / "candidates.jsonl"
    if file_digest(candidates_path) != manifest["candidates_sha256"]:
        raise ValueError("Le fichier de candidats a changé depuis sa préparation.")
    registry = {row["candidate_id"]: row for row in iter_jsonl(candidates_path)}
    judgments: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen = set()
    with reviewed_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cid = row["candidate_id"]
            if cid not in registry or cid in seen:
                raise ValueError("Candidat inconnu ou répété dans la revue.")
            seen.add(cid)
            candidate = registry[cid]
            if any(
                row[name] != str(candidate[name])
                for name in ("index", "question_id", "evidence_id", "chunk_id")
            ):
                raise ValueError("Identité du candidat modifiée dans la revue.")
            if not row["reviewed_grade"].strip():
                continue
            if row["reviewed_grade"] not in {"0", "1", "2"}:
                raise ValueError("Grade de revue autorisé : 0, 1 ou 2.")
            if not row["reviewer"].strip() or not row["notes"].strip():
                raise ValueError("Chaque jugement exige un reviewer et une justification.")
            key = (candidate["index"], candidate["question_id"], candidate["chunk_id"])
            judgment = judgments.setdefault(
                key,
                {
                    "index": key[0],
                    "question_id": key[1],
                    "chunk_id": key[2],
                    "content_sha256": candidate["content_sha256"],
                    "source": candidate["source"],
                    "relevance": 0,
                    "judgment_status": "reviewed",
                    "evidence_judgments": [],
                },
            )
            grade = int(row["reviewed_grade"])
            judgment["relevance"] = max(judgment["relevance"], grade)
            judgment["evidence_judgments"].append(
                {
                    "evidence_id": candidate["evidence_id"],
                    "relevance": grade,
                    "reviewer": row["reviewer"],
                    "notes": row["notes"],
                }
            )
    # A per-evidence negative is not a question-level negative while other evidence is unjudged.
    positive_or_complete = []
    evidence_counts = manifest["question_evidence_counts"]
    for judgment in judgments.values():
        if (
            judgment["relevance"] == 0
            and len(judgment["evidence_judgments"]) != evidence_counts[judgment["question_id"]]
        ):
            continue
        positive_or_complete.append(judgment)
    path = directory / "qrels.reviewed.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for judgment in positive_or_complete:
            stream.write(json.dumps(judgment, ensure_ascii=False) + "\n")
    atomic_json(
        directory / "review-status.json",
        {
            "judged_pairs": len(positive_or_complete),
            "review_csv_sha256": file_digest(reviewed_csv),
            "candidates_sha256": manifest["candidates_sha256"],
            "limitations": "Only explicitly reviewed pairs. Unlisted pairs remain unjudged. "
            "No exhaustive recall denominator implied.",
        },
    )
    return len(positive_or_complete)


def build(golden: Path, snapshot_path: Path, output: Path, limit: int = 5) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choisir un dossier vide pour préserver les revues existantes.")
    questions = load_questions(golden)
    with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    indexes = prepare_snapshot(snapshot)
    for question in questions:
        for evidence in question["evidence"]:
            if any(evidence["doc_name"] not in groups for groups in indexes.values()):
                raise ValueError("Document gold absent du snapshot.")
    output.mkdir(parents=True, exist_ok=True)
    all_units = sum(len(q["evidence"]) for q in questions)
    counts: Counter[str] = Counter()
    automatic: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidate_path = output / "candidates.jsonl"
    with (
        candidate_path.open("w", encoding="utf-8") as stream,
        (output / "review.csv").open("w", newline="", encoding="utf-8") as csv_stream,
    ):
        writer = csv.DictWriter(csv_stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for number, question in enumerate(questions, 1):
            for index, groups in indexes.items():
                covered = 0
                for position, evidence in enumerate(question["evidence"]):
                    candidates = propose(
                        question, position, groups[evidence["doc_name"]], limit=limit
                    )
                    has_exact = any(row["automatic_grade"] == 2 for row in candidates)
                    covered += has_exact
                    counts[f"{index}:evidence_exact"] += has_exact
                    for row in candidates:
                        row["index"] = index
                        row["candidate_id"] = digest(index + row["candidate_id"])
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        review_row = {name: row.get(name, "") for name in REVIEW_FIELDS}
                        review_row["automatic_grade"] = row["automatic_grade"] or ""
                        writer.writerow(review_row)
                        counts[f"{index}:candidates"] += 1
                        if row["automatic_grade"] == 2:
                            key = (index, row["question_id"], row["chunk_id"])
                            auto = automatic.setdefault(
                                key,
                                {
                                    "index": index,
                                    "question_id": key[1],
                                    "chunk_id": key[2],
                                    "source": row["source"],
                                    "content_sha256": row["content_sha256"],
                                    "relevance": 2,
                                    "judgment_status": "automatic_exact_text",
                                    "evidence_ids": [],
                                },
                            )
                            auto["evidence_ids"].append(row["evidence_id"])
                counts[f"{index}:questions_any_exact"] += covered > 0
                counts[f"{index}:questions_all_exact"] += covered == len(question["evidence"])
            print(f"[{number}/{len(questions)}] {question['financebench_id']}", flush=True)
    with (output / "qrels.automatic.jsonl").open("w", encoding="utf-8") as stream:
        for row in automatic.values():
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "method": METHOD,
        "golden_sha256": file_digest(golden),
        "snapshot_sha256": file_digest(snapshot_path),
        "snapshot_created_at": snapshot.get("created_at"),
        "candidates_sha256": file_digest(candidate_path),
        "azure_calls": False,
        "questions": len(questions),
        "evidence_per_index": all_units,
        "candidate_limit": limit,
        "question_evidence_counts": {q["financebench_id"]: len(q["evidence"]) for q in questions},
        "counts": dict(counts),
        "automatic_pairs": len(automatic),
        "label_definition": {
            "2": "direct sufficient evidence; automatic rule requires full gold text",
            "1": "useful partial evidence",
            "0": "irrelevant to all required gold evidence",
            "blank": "unjudged",
        },
        "limitations": "Automatic exact matches are silver labels, not human judgments. "
        "Other candidates have no grade. No negatives inferred from absence or weak overlap. "
        "Candidate pool is not exhaustive relevance ground truth. Pages not used as labels.",
    }
    atomic_json(output / "manifest.json", manifest)
    write_review_packets(output)
    return manifest


def write_review_packets(directory: Path) -> None:
    """Fournir des pages Markdown locales avec les textes complets à relire."""
    packets = directory / "review-packets"
    packets.mkdir(exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(directory / "candidates.jsonl"):
        groups[row["question_id"]].append(row)
    links = {}
    for qid, candidates in groups.items():
        blocks = [
            "# " + qid,
            candidates[0]["question"],
            "Les grades automatiques ne sont pas des jugements relus.",
        ]
        for row in candidates:
            # Long fences prevent source text from closing the literal content block.
            fence = "`" * (
                max(
                    (len(m) for m in re.findall(r"`+", row["content"] + row["evidence_text"])),
                    default=3,
                )
                + 1
            )
            blocks.extend(
                [
                    "## " + row["index"] + " / " + row["evidence_id"] + " / " + row["chunk_id"],
                    "Candidate ID: " + row["candidate_id"],
                    "Automatic grade: " + str(row["automatic_grade"]),
                    "Gold evidence:",
                    fence + "text",
                    row["evidence_text"],
                    fence,
                    "Chunk (snapshot):",
                    fence + "text",
                    row["content"],
                    fence,
                ]
            )
        filename = digest(qid) + ".md"
        (packets / filename).write_text("\n\n".join(blocks), encoding="utf-8")
        links[qid] = "review-packets/" + filename
    atomic_json(directory / "review-packets.json", links)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument(
        "--snapshot", type=Path, default=Path("docs/reports/page-localization/chunks.json.gz")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/reports/retrieval-qrels"))
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--reviewed-csv", type=Path)
    args = parser.parse_args()
    if args.reviewed_csv:
        print(f"{export_reviewed(args.output, args.reviewed_csv)} qrels reviewed exportées.")
    else:
        if args.candidates < 1:
            parser.error("candidates doit être positif")
        print(json.dumps(build(args.golden, args.snapshot, args.output, args.candidates), indent=2))


if __name__ == "__main__":
    main()
