"""Évaluer une campagne RAG : juges Azure AI Evaluation (gpt-5-mini) et contrôle des chiffres.

Exemple : uv run python scripts/evaluate_rag.py --run docs/reports/rag/eval/C1-native-vector-k5
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from azure.ai.evaluation import evaluate
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from ai103.evaluation.dataset import (
    evaluation_rows,
    load_golden,
    rag_statuses,
    write_jsonl,
)
from ai103.evaluation.judge import EVALUATOR_CONFIG, evaluators, model_config
from ai103.retriever.benchmark import atomic_json

# Nom de l'évaluateur dans evaluate() -> clé de résultat choisie par le SDK.
JUDGES = {
    "groundedness": "groundedness",
    "relevance": "relevance",
    "completeness": "response_completeness",
}
# Prix standard de gpt-5-mini absent de l'API Retail Prices : borne haute au tarif
# « priority processing » relevé le 19 octobre (USD par million de tokens).
JUDGE_PRICES = {"input": 0.45, "output": 3.6}


def _value(value: Any) -> float | None:
    """Le SDK transforme un None en NaN : les deux signifient « pas de valeur »."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _score(row: dict[str, Any], name: str) -> float | None:
    return _value(row.get(f"outputs.{name}.{JUDGES[name]}"))


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("inputs.question_type", "unknown")].append(row)

    def judge_scores(subset: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in JUDGES:
            scores = [s for row in subset if (s := _score(row, name)) is not None]
            # Taux calculé sur les questions notées : une évaluation en échec n'est pas un « fail ».
            passed = [
                row.get(f"outputs.{name}.{JUDGES[name]}_result") == "pass"
                for row in subset
                if _score(row, name) is not None
            ]
            result[name] = {
                "mean": _mean(scores),
                "pass_rate": _mean([float(p) for p in passed]),
                "scored": len(scores),
            }
        numeric = [
            float(value)
            for row in subset
            if (value := _value(row.get("outputs.numeric.numeric_match"))) is not None
        ]
        result["numeric_accuracy"] = {"mean": _mean(numeric), "applicable": len(numeric)}
        return result

    tokens = Counter[str]()
    for row in rows:
        for name in JUDGES:
            properties = row.get(f"outputs.{name}.{JUDGES[name]}_properties")
            if not isinstance(properties, dict):
                # Jugement en échec : le SDK met NaN à la place du détail des tokens.
                continue
            tokens["input"] += properties.get("prompt_tokens", 0)
            tokens["output"] += properties.get("completion_tokens", 0)
    return {
        "questions": len(rows),
        "overall": judge_scores(rows),
        "by_question_type": {
            name: judge_scores(subset) for name, subset in sorted(by_type.items())
        },
        "rag_status": dict(Counter(row.get("inputs.rag_status") for row in rows)),
        "rag_cost_usd": sum(row.get("inputs.rag_cost_usd") or 0 for row in rows),
        "judge_tokens": dict(tokens),
        "judge_cost_usd_upper_bound": sum(tokens[k] * JUDGE_PRICES[k] for k in tokens) / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Dossier d'une campagne rag_run.")
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--limit", type=int, help="Évaluer seulement les N premières réponses.")
    parser.add_argument(
        "--log-to-foundry", action="store_true", help="Publier aussi dans le projet Foundry."
    )
    parser.add_argument(
        "--resummarize", action="store_true", help="Recalculer summary.json sans appel Azure."
    )
    args = parser.parse_args()
    output = args.run / (f"evaluation-limit{args.limit}" if args.limit else "evaluation")
    if args.resummarize:
        if not (output / "evaluation.json").exists():
            parser.error(
                f"Aucune évaluation dans {output} : lancer d'abord la commande sans --resummarize."
            )
        saved = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        summary_path = output / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        kept = {k: previous[k] for k in ("created_at", "run", "judge_deployment") if k in previous}
        summary = {**kept, "resummarized_at": datetime.now(UTC).isoformat()}
        statuses = rag_statuses(args.run)
        for row in saved["rows"]:
            row["inputs.rag_status"] = statuses.get(
                row.get("inputs.question_id"), row.get("inputs.rag_status")
            )
        summary |= summarize(saved["rows"])
        atomic_json(summary_path, summary)
        print(json.dumps({k: summary[k] for k in ("questions", "overall")}, indent=2))
        return
    load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    judge = os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT", "gpt-5-mini").strip()
    if not endpoint:
        parser.error("Configurer AZURE_OPENAI_ENDPOINT dans .env.")
    if judge == os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"):
        parser.error("Le juge doit être distinct du modèle générateur.")
    project = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    if args.log_to_foundry and not project:
        parser.error("Configurer FOUNDRY_PROJECT_ENDPOINT pour publier dans Foundry.")
    rows, skipped = evaluation_rows(
        args.run / "results.jsonl", load_golden(args.golden), args.limit
    )
    if output.exists() and any(output.iterdir()):
        parser.error(f"Dossier non vide : {output}")
    write_jsonl(rows, output / "data.jsonl")
    credential = DefaultAzureCredential()
    result = evaluate(
        data=str(output / "data.jsonl"),
        evaluators=evaluators(model_config(endpoint, judge), credential),
        evaluator_config=EVALUATOR_CONFIG,
        evaluation_name=f"finance-rag-{args.run.name}",
        output_path=str(output / "evaluation.json"),
        azure_ai_project=project if args.log_to_foundry else None,
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "run": str(args.run),
        "judge_deployment": judge,
        "skipped_generation_errors": skipped,
        "foundry_url": result.get("studio_url"),
        **summarize(result["rows"]),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("questions", "overall", "rag_status")}, indent=2))
    print(f"Coût du juge (borne haute) : {summary['judge_cost_usd_upper_bound']:.4f} USD")
    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()
