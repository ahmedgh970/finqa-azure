"""Comparer plusieurs configurations RAG évaluées, question par question, sans appel Azure.

Les configurations portent sur les mêmes 150 questions : les écarts sont donc appariés.
"""

import argparse
import json
import math
import random
from collections import Counter
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any

from ai103.evaluation.dataset import rag_statuses
from ai103.retriever.benchmark import atomic_json

ABSTAIN = "insufficient_evidence"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return None if math.isnan(value) else float(value)


def load_config(run: Path) -> dict[str, dict[str, Any]]:
    """Une entrée par question : statut RAG revérifié, notes du juge, chiffres, coût."""
    statuses = rag_statuses(run)
    rows = json.loads((run / "evaluation" / "evaluation.json").read_text())["rows"]
    questions = {}
    for row in rows:
        qid = row["inputs.question_id"]
        questions[qid] = {
            "type": row["inputs.question_type"],
            "status": statuses.get(qid, row.get("inputs.rag_status")),
            "groundedness": _num(row.get("outputs.groundedness.groundedness")),
            "relevance": _num(row.get("outputs.relevance.relevance")),
            "completeness": _num(row.get("outputs.completeness.response_completeness")),
            "numeric": _num(row.get("outputs.numeric.numeric_match")),
            "rag_cost": _num(row.get("inputs.rag_cost_usd")) or 0.0,
            "ground_truth": row.get("inputs.ground_truth"),
            "response": row.get("inputs.response", ""),
        }
    return questions


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def describe(questions: dict[str, dict[str, Any]], generation: dict[str, Any]) -> dict[str, Any]:
    def scores(name: str, subset: list[dict[str, Any]]) -> list[float]:
        return [q[name] for q in subset if q[name] is not None]

    answered = [q for q in questions.values() if q["status"] != ABSTAIN]
    abstained = [q for q in questions.values() if q["status"] == ABSTAIN]
    everything = list(questions.values())
    numeric_all = scores("numeric", everything)
    numeric_answered = scores("numeric", answered)
    completeness_answered = scores("completeness", answered)
    by_type = {}
    for qtype in sorted({q["type"] for q in everything}):
        subset = [q for q in everything if q["type"] == qtype]
        by_type[qtype] = {
            "completeness": _avg(scores("completeness", subset)),
            "answer_rate": sum(q["status"] != ABSTAIN for q in subset) / len(subset),
        }
    latency = generation.get("latency_seconds", {})
    return {
        "questions": len(everything),
        "generation_errors": generation.get("errors", 0),
        "status": dict(Counter(q["status"] for q in everything)),
        "answer_rate": len(answered) / len(everything),
        "groundedness": _avg(scores("groundedness", everything)),
        "relevance": _avg(scores("relevance", everything)),
        "completeness": _avg(scores("completeness", everything)),
        "completeness_pass_rate": _avg([float(c >= 3) for c in scores("completeness", everything)]),
        "numeric_exact": sum(numeric_all),
        "numeric_applicable": len(numeric_all),
        "answered": {
            "count": len(answered),
            "completeness": _avg(completeness_answered),
            "completeness_high_rate": _avg([float(c >= 4) for c in completeness_answered]),
            "numeric_exact": sum(numeric_answered),
            "numeric_given": len(numeric_answered),
        },
        "abstained": {
            "count": len(abstained),
            "completeness": _avg(scores("completeness", abstained)),
        },
        "by_question_type": by_type,
        "generation_cost_usd": generation.get("cost_usd"),
        "tokens": generation.get("tokens"),
        "latency_total_median": latency.get("total_seconds", {}).get("median"),
        "latency_total_p95": latency.get("total_seconds", {}).get("p95"),
    }


def mcnemar_exact(gains: int, losses: int) -> float:
    """p-valeur bilatérale exacte : seules les questions qui changent d'issue comptent."""
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def bootstrap_mean_diff(
    diffs: list[float], samples: int = 10_000, seed: int = 103
) -> tuple[float, float]:
    """Intervalle à 95 % de la différence moyenne appariée (rééchantillonnage des questions)."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def paired(base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> dict[str, Any]:
    common = sorted(set(base) & set(other))
    diffs = [
        other[q]["completeness"] - base[q]["completeness"]
        for q in common
        if base[q]["completeness"] is not None and other[q]["completeness"] is not None
    ]
    numeric = [
        q for q in common if base[q]["numeric"] is not None and other[q]["numeric"] is not None
    ]
    gains = sum(other[q]["numeric"] > base[q]["numeric"] for q in numeric)
    losses = sum(other[q]["numeric"] < base[q]["numeric"] for q in numeric)
    newly_answered = sum(base[q]["status"] == ABSTAIN != other[q]["status"] for q in common)
    newly_abstained = sum(other[q]["status"] == ABSTAIN != base[q]["status"] for q in common)
    low, high = bootstrap_mean_diff(diffs) if diffs else (0.0, 0.0)
    return {
        "completeness_mean_diff": mean(diffs) if diffs else None,
        "completeness_diff_ci95": [low, high],
        "completeness_improved": sum(d > 0 for d in diffs),
        "completeness_worsened": sum(d < 0 for d in diffs),
        "numeric_gains": gains,
        "numeric_losses": losses,
        "numeric_mcnemar_p": mcnemar_exact(gains, losses),
        "newly_answered": newly_answered,
        "newly_abstained": newly_abstained,
    }


def wrong_answers(questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Réponses données (non abstenues) mais jugées fausses : chiffre inexact ou complétude ≤ 2."""
    wrong = []
    for qid, q in sorted(questions.items()):
        if q["status"] == ABSTAIN:
            continue
        if q["numeric"] == 0.0 or (q["completeness"] is not None and q["completeness"] <= 2):
            wrong.append(
                {
                    "question_id": qid,
                    "type": q["type"],
                    "status": q["status"],
                    "completeness": q["completeness"],
                    "numeric": q["numeric"],
                    "ground_truth": str(q["ground_truth"])[:160],
                    "response": q["response"][:220],
                }
            )
    return wrong


def compare(runs: list[Path]) -> dict[str, Any]:
    configs = {run.name: load_config(run) for run in runs}
    generation = {run.name: json.loads((run / "summary.json").read_text()) for run in runs}
    base_name = runs[0].name
    base = configs[base_name]
    reachable = {
        qid
        for questions in configs.values()
        for qid, q in questions.items()
        if q["status"] != ABSTAIN and (q["completeness"] or 0) >= 4
    }
    return {
        "baseline": base_name,
        "configs": {name: describe(q, generation[name]) for name, q in configs.items()},
        "paired_vs_baseline": {
            name: paired(base, q) for name, q in configs.items() if name != base_name
        },
        # Questions bien traitées par au moins une configuration : plafond d'un choix par question.
        "answered_well_by_any_config": len(reachable),
        "wrong_answers": {name: wrong_answers(q) for name, q in configs.items()},
        "median_of_medians_latency": median(
            d["latency_seconds"]["total_seconds"]["median"] for d in generation.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Dossiers de campagnes évaluées.")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/rag/eval/comparison.json")
    )
    args = parser.parse_args()
    result = compare(args.runs)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "wrong_answers"}, indent=2))
    print(f"Comparaison : {args.output}")


if __name__ == "__main__":
    main()
