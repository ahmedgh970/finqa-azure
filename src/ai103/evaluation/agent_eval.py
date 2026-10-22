"""Évaluer la trajectoire de l'agent : outils, respect des consignes, résolution de la demande.

Les conversations sont reconstruites à partir du journal de campagne, au format de messages
d'agent attendu par Azure AI Evaluation (format « converter » des appels d'outils). Aucun nouvel
appel à l'agent : seul le juge est sollicité.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
    evaluate,
)
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from ai103.agent.definition import INSTRUCTIONS
from ai103.agent.tools import TOOL_DEFINITIONS
from ai103.evaluation.dataset import results_file, write_jsonl
from ai103.evaluation.judge import model_config
from ai103.retriever.benchmark import atomic_json

# Nom de l'évaluateur dans evaluate() -> clé de résultat choisie par le SDK.
AGENT_JUDGES = {
    "tool_call_accuracy": "tool_call_accuracy",
    "task_adherence": "task_adherence",
    "intent_resolution": "intent_resolution",
}
JUDGE_PRICES = {"input": 0.45, "output": 3.6}  # borne haute gpt-5-mini, USD par million
# Le texte des passages sert à juger l'ancrage des arguments et le respect des consignes ;
# Intent Resolution n'en a pas besoin : version légère, moins coûteuse.
EVALUATOR_CONFIG: dict[str, Any] = {
    name: {
        "column_mapping": {
            "query": "${data.query}",
            "response": "${data.%s}"
            % ("response_light" if name == "intent_resolution" else "response"),
            "tool_definitions": "${data.tool_definitions}",
        }
    }
    for name in AGENT_JUDGES
}
PASSAGES = re.compile(r"^(\d+) passages")


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in TOOL_DEFINITIONS
    ]


def passages_by_call(record: dict[str, Any]) -> tuple[dict[int, list[int]], bool]:
    """Numéros des passages renvoyés par chaque appel de recherche, et exactitude du rattachement.

    Les campagnes récentes enregistrent les numéros par appel. Pour les anciennes, on les
    reconstitue dans l'ordre : exact si les recherches n'ont renvoyé aucun doublon, sinon approché.
    """
    calls = record.get("tool_calls", [])
    if any(call.get("sources") for call in calls):
        return {i: call.get("sources", []) for i, call in enumerate(calls, 1)}, True
    total = len(record.get("sources", []))
    mapping: dict[int, list[int]] = {}
    cursor, returned = 1, 0
    for index, call in enumerate(calls, 1):
        match = PASSAGES.match(call["summary"]) if call["executed"] else None
        if call["name"] != "search_filings" or match is None:
            continue
        count = int(match.group(1))
        returned += count
        mapping[index] = list(range(cursor, min(cursor + count, total + 1)))
        cursor += count
    return mapping, returned == total


def _search_result(record: dict[str, Any], numbers: list[int], filter_note: str) -> str:
    by_number = {source["number"]: source for source in record.get("sources", [])}
    passages = [
        {
            "source": number,
            "document": by_number[number]["document"],
            "pages": by_number[number]["pages"],
            "text": by_number[number]["chunk"]["content"],
        }
        for number in numbers
        if number in by_number
    ]
    return json.dumps({"filter": filter_note, "passages": passages}, ensure_ascii=False)


def conversation(
    record: dict[str, Any], *, with_passages: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """(query, response, exact) : consigne et question, tours d'outils, réponse JSON complète."""
    query: list[dict[str, Any]] = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": [{"type": "text", "text": record["question"]}]},
    ]
    passages, exact = passages_by_call(record)
    response: list[dict[str, Any]] = []
    by_round: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, call in enumerate(record.get("tool_calls", []), 1):
        by_round[call["round"]].append((index, call))
    for round_number in sorted(by_round):
        calls = by_round[round_number]
        response.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": f"call_{index}",
                        "name": call["name"],
                        "arguments": json.loads(call["arguments"]),
                    }
                    for index, call in calls
                ],
            }
        )
        for index, call in calls:
            if not call["executed"]:
                result = "Refused: tool budget exhausted."
            elif with_passages and index in passages:
                filter_note = call["summary"].split("filter=", 1)[-1]
                result = _search_result(record, passages[index], filter_note)
            else:
                result = call["summary"]
            response.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": [{"type": "tool_result", "tool_result": result}],
                }
            )
    # La réponse réelle de l'agent : texte, citations avec extraits, et indicateur d'abstention.
    final = json.dumps(record["answer"], ensure_ascii=False)
    response.append({"role": "assistant", "content": [{"type": "text", "text": final}]})
    return query, response, exact


def agent_rows(run: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in results_file(run).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "error" in record:
            continue
        query, response, exact = conversation(record)
        _, light, _ = conversation(record, with_passages=False)
        rows.append(
            {
                "question_id": record["question_id"],
                "status": record["status"],
                "reconstruction": "exact" if exact else "approximate",
                "query": query,
                "response": response,
                "response_light": light,
                "tool_definitions": tool_definitions(),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def agent_evaluators(
    config: AzureOpenAIModelConfiguration, credential: TokenCredential
) -> dict[str, Any]:
    common: dict[str, Any] = {"credential": credential, "is_reasoning_model": True}
    return {
        "tool_call_accuracy": ToolCallAccuracyEvaluator(config, **common),  # type: ignore[no-untyped-call]
        "task_adherence": TaskAdherenceEvaluator(config, **common),  # type: ignore[no-untyped-call]
        "intent_resolution": IntentResolutionEvaluator(config, **common),  # type: ignore[no-untyped-call]
    }


def _value(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def scores(subset: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, key in AGENT_JUDGES.items():
            values = [
                v
                for r in subset
                if isinstance(v := _value(r.get(f"outputs.{name}.{key}")), int | float)
            ]
            verdicts = [
                r.get(f"outputs.{name}.{key}_result")
                for r in subset
                if r.get(f"outputs.{name}.{key}_result") in ("pass", "fail")
            ]
            result[name] = {
                "mean": mean(values) if values else None,
                "pass_rate": mean(v == "pass" for v in verdicts) if verdicts else None,
                "scored": len(values),
                "not_applicable": sum(
                    r.get(f"outputs.{name}.{key}_result") == "not_applicable" for r in subset
                ),
            }
        return result

    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_status[row.get("inputs.status", "unknown")].append(row)
    tokens: Counter[str] = Counter()
    for row in rows:
        for name, key in AGENT_JUDGES.items():
            properties = row.get(f"outputs.{name}.{key}_properties")
            if isinstance(properties, dict):
                tokens["input"] += properties.get("prompt_tokens", 0)
                tokens["output"] += properties.get("completion_tokens", 0)
    worst = sorted(
        (
            (
                score,
                row.get("inputs.question_id"),
                name,
                str(row.get(f"outputs.{name}.{key}_reason")),
            )
            for row in rows
            for name, key in AGENT_JUDGES.items()
            if isinstance(score := _value(row.get(f"outputs.{name}.{key}")), int | float)
        ),
        key=lambda item: item[0],
    )[:10]
    return {
        "questions": len(rows),
        "overall": scores(rows),
        "by_status": {status: scores(subset) for status, subset in sorted(by_status.items())},
        "by_reconstruction": {
            kind: scores([r for r in rows if r.get("inputs.reconstruction") == kind])
            for kind in ("exact", "approximate")
            if any(r.get("inputs.reconstruction") == kind for r in rows)
        },
        "lowest": [
            {"score": s, "question_id": q, "evaluator": n, "reason": r[:400]}
            for s, q, n, r in worst
        ],
        "judge_tokens": dict(tokens),
        "judge_cost_usd_upper_bound": sum(tokens[k] * JUDGE_PRICES[k] for k in tokens) / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Dossier d'une campagne d'agent.")
    parser.add_argument("--limit", type=int, help="Évaluer seulement les N premières questions.")
    parser.add_argument("--output", type=Path, help="Dossier de sortie (vide).")
    args = parser.parse_args()
    load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    judge = os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT", "gpt-5-mini").strip()
    if not endpoint:
        parser.error("Configurer AZURE_OPENAI_ENDPOINT dans .env.")
    if judge == os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"):
        parser.error("Le juge doit être distinct du modèle générateur.")
    default = f"agent-evaluation-limit{args.limit}" if args.limit else "agent-evaluation"
    output = args.output or args.run / default
    if output.exists() and any(output.iterdir()):
        parser.error(f"Dossier non vide : {output}")
    write_jsonl(agent_rows(args.run, args.limit), output / "data.jsonl")
    result = evaluate(
        data=str(output / "data.jsonl"),
        evaluators=agent_evaluators(model_config(endpoint, judge), DefaultAzureCredential()),
        evaluator_config=EVALUATOR_CONFIG,
        evaluation_name=f"finance-agent-{args.run.name}",
        output_path=str(output / "evaluation.json"),
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "run": str(args.run),
        "judge_deployment": judge,
        **summarize(result["rows"]),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("questions", "overall")}, indent=2))
    print(f"Coût du juge (borne haute) : {summary['judge_cost_usd_upper_bound']:.4f} USD")
    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()
