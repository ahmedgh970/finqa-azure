"""Scorer des jugements explicites sans convertir les non-jugés en négatifs."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai103.metadata_enrichment import doc_name_from_source
from ai103.retriever.core import Chunk

SCORING_VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def valid_grade(value: Any) -> bool:
    return type(value) is int and value in (0, 1, 2)


@dataclass
class ReviewedQrels:
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]]
    groups: dict[tuple[str, str], list[dict[str, Any]]]
    evidence: dict[str, set[str]]
    fingerprints: dict[tuple[str, str], tuple[str, str]]
    provenance: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: Path,
        questions: Sequence[dict[str, Any]],
        golden_sha256: str,
    ) -> "ReviewedQrels":
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["golden_sha256"] != golden_sha256:
            raise ValueError("Les qrels ne correspondent pas au golden set.")
        candidates_path = directory / "candidates.jsonl"
        if sha256(candidates_path) != manifest["candidates_sha256"]:
            raise ValueError("Le pool de candidats des qrels a été modifié.")
        evidence = {
            q["financebench_id"]: {
                f"{q['financebench_id']}:e{i}" for i in range(len(q["evidence"]))
            }
            for q in questions
        }
        documents = {
            f"{q['financebench_id']}:e{i}": e["doc_name"]
            for q in questions
            for i, e in enumerate(q["evidence"])
        }
        registry: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in read_jsonl(candidates_path):
            registry_key = (row["index"], row["question_id"], row["chunk_id"], row["evidence_id"])
            if registry_key in registry or row["evidence_id"] not in evidence.get(
                row["question_id"], set()
            ):
                raise ValueError("Candidat répété ou preuve inconnue.")
            if (
                hashlib.sha256(row["content"].encode()).hexdigest() != row["content_sha256"]
                or doc_name_from_source(row["source"]) != documents[row["evidence_id"]]
            ):
                raise ValueError("Contenu ou source d'un candidat incohérent.")
            registry[registry_key] = row
        fingerprints: dict[tuple[str, str], tuple[str, str]] = {}

        def check_reference(row: dict[str, Any], member: dict[str, Any], eid: str) -> None:
            ref = registry.get((row["index"], row["question_id"], member["chunk_id"], eid))
            if ref is None or any(
                member.get(name) != ref[name] for name in ("content_sha256", "source")
            ):
                raise ValueError("Jugement sans candidat correspondant ou contenu modifié.")
            key = (row["index"], member["chunk_id"])
            fingerprint = (member["content_sha256"], member["source"])
            if key in fingerprints and fingerprints[key] != fingerprint:
                raise ValueError("Empreintes contradictoires pour une même clé.")
            fingerprints[key] = fingerprint

        pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        pairs_path = directory / "qrels.reviewed.jsonl"
        for row in read_jsonl(pairs_path):
            if row.get("judgment_status") != "reviewed" or not valid_grade(row.get("relevance")):
                raise ValueError("Seuls les jugements explicites 0/1/2 sont acceptés.")
            key = (row["index"], row["question_id"])
            if row["chunk_id"] in pairs[key]:
                raise ValueError("Qrel répétée.")
            eids = set()
            judgments = row.get("evidence_judgments", [])
            if not judgments:
                raise ValueError("Jugements par preuve requis.")
            for judgment in judgments:
                eid = judgment["evidence_id"]
                if eid in eids or not valid_grade(judgment.get("relevance")):
                    raise ValueError("Preuve répétée ou grade invalide.")
                if (
                    not judgment.get("reviewer", "").strip()
                    or not judgment.get("notes", "").strip()
                ):
                    raise ValueError("Reviewer et justification requis.")
                eids.add(eid)
                check_reference(row, row, eid)
            if row["relevance"] != max(j["relevance"] for j in judgments):
                raise ValueError("Grade agrégé incohérent.")
            if row["relevance"] == 0 and eids != evidence[row["question_id"]]:
                raise ValueError("Un négatif de question doit couvrir toutes ses preuves.")
            pairs[key][row["chunk_id"]] = row
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        groups_path = directory / "qrels.groups.reviewed.jsonl"
        seen_groups = set()
        if groups_path.exists():
            for row in read_jsonl(groups_path):
                if (
                    row.get("judgment_status") != "reviewed_group"
                    or (not valid_grade(row.get("relevance")) or row["relevance"] != 2)
                    or row.get("rule") != "all_members_required"
                    or not row.get("reviewer", "").strip()
                    or not row.get("notes", "").strip()
                ):
                    raise ValueError("Groupe suffisant explicitement justifié requis.")
                gid = (row["index"], row["question_id"], row["group_id"])
                members = row.get("members", [])
                if (
                    gid in seen_groups
                    or len(members) < 2
                    or len({m["chunk_id"] for m in members}) != len(members)
                ):
                    raise ValueError("Groupe répété ou membres invalides.")
                seen_groups.add(gid)
                for member in members:
                    check_reference(row, member, row["evidence_id"])
                    pair = pairs.get((row["index"], row["question_id"]), {}).get(member["chunk_id"])
                    if pair is None or not any(
                        j["evidence_id"] == row["evidence_id"] and j["relevance"] > 0
                        for j in pair["evidence_judgments"]
                    ):
                        raise ValueError("Les membres d'un groupe doivent avoir été jugés utiles.")
                groups[(row["index"], row["question_id"])].append(row)
        provenance = {
            "scoring_version": SCORING_VERSION,
            "manifest_sha256": sha256(manifest_path),
            "candidates_sha256": sha256(candidates_path),
            "reviewed_sha256": sha256(pairs_path),
            "groups_sha256": sha256(groups_path) if groups_path.exists() else None,
            "snapshot_sha256": manifest["snapshot_sha256"],
            "automatic_labels_used": False,
        }
        return cls(dict(pairs), dict(groups), evidence, fingerprints, provenance)

    def score(self, index: str, question_id: str, chunks: Sequence[Chunk]) -> dict[str, Any]:
        if question_id not in self.evidence:
            raise ValueError("Question absente du golden set des qrels.")
        ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(chunks, 1)}
        if len(ranks) != len(chunks):
            raise ValueError("Clés répétées dans les résultats.")
        for chunk in chunks:
            expected = self.fingerprints.get((index, chunk.chunk_id))
            actual = (hashlib.sha256(chunk.content.encode()).hexdigest(), chunk.source)
            if expected is not None and actual != expected:
                raise ValueError("Contenu/source récupéré différent de celui du jugement.")
        pairs = self.pairs.get((index, question_id), {})
        groups = self.groups.get((index, question_id), [])
        annotated = bool(pairs or groups)
        useful_ranks = [
            ranks[cid] for cid, row in pairs.items() if cid in ranks and row["relevance"] > 0
        ]
        completion: dict[str, int] = {}
        with_sufficient_judgment = set()
        for cid, row in pairs.items():
            for judgment in row["evidence_judgments"]:
                if judgment["relevance"] == 2:
                    eid = judgment["evidence_id"]
                    with_sufficient_judgment.add(eid)
                    if cid in ranks:
                        completion[eid] = min(completion.get(eid, ranks[cid]), ranks[cid])
        for group in groups:
            eid = group["evidence_id"]
            with_sufficient_judgment.add(eid)
            if all(m["chunk_id"] in ranks for m in group["members"]):
                rank = max(ranks[m["chunk_id"]] for m in group["members"])
                completion[eid] = min(completion.get(eid, rank), rank)
        all_eids = self.evidence[question_id]
        judged = sum(cid in pairs for cid in ranks)
        all_verified = len(completion) == len(all_eids)
        retrieval_judgment = (
            "empty_results"
            if not chunks
            else "verified_useful"
            if useful_ranks
            else "verified_irrelevant"
            if judged == len(chunks)
            else "insufficient_judgments"
        )
        return {
            "retrieval_judgment": retrieval_judgment,
            "scoring_version": SCORING_VERSION,
            "annotation_available": annotated,
            "returned_chunks": len(chunks),
            "judged_chunks": judged,
            "unjudged_chunks": len(chunks) - judged,
            "judged_fraction": judged / len(chunks) if chunks else None,
            "known_useful_hit_at_k": bool(useful_ranks) if annotated else None,
            "known_useful_rr_at_k": (1 / min(useful_ranks) if useful_ranks else 0.0)
            if annotated
            else None,
            "gold_evidence_total": len(all_eids),
            "evidence_with_sufficient_judgments": len(with_sufficient_judgment),
            "sufficient_annotation_fraction": len(with_sufficient_judgment) / len(all_eids),
            "verified_evidence_count": len(completion),
            "verified_evidence_coverage_lower_bound": len(completion) / len(all_eids)
            if annotated
            else None,
            "all_gold_evidence_verified_at_k": all_verified if annotated else None,
            "all_evidence_completion_rr_at_k": 1 / max(completion.values())
            if all_verified
            else (0.0 if annotated else None),
            "evidence_completion_ranks": completion,
            "unverified_evidence_ids": sorted(all_eids - completion.keys()),
            "limitations": "Scores use reviewed labels only; absent pairs are unjudged. "
            "A zero means no reviewed success observed, not a proven retrieval failure. "
            "Coverage is a lower bound for annotated evidence, not exhaustive chunk recall. "
            "Gold evidence may itself omit facts needed for a complete answer.",
        }
