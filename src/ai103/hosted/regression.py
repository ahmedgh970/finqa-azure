"""Non-régression de l'agent hébergé face à A2, sans juge LLM.

Un échantillon fixe des questions passées à A2, une conversation neuve par question. On compare
le statut (lu dans le texte rendu par l'agent), le chiffre gold (contrôle déterministe
d'`evaluation.numeric`) et la latence. Le golden set n'est lu qu'après les réponses, pour noter.

Sans `--apply`, le script affiche l'échantillon et l'estimation de coût, sans aucun appel.
"""

import argparse
import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai103.evaluation.dataset import load_golden, results_file
from ai103.evaluation.numeric import check, gold_number
from ai103.hosted.chat import REUSED_OK, STATUS_LABELS
from ai103.hosted.deploy import HOSTED_AGENT_NAME, agent_client
from ai103.retriever.benchmark import atomic_json

A2_RUN = Path("docs/reports/rag/eval/A2-agent-v2")
OUTPUT = Path("docs/reports/hosted/regression")
# Coût moyen mesuré d'une question d'A2 ; plafond par question imposé par l'agent lui-même.
MEAN_COST_USD = 0.039
MAX_QUESTION_COST_USD = 0.10
LABEL_STATUS = {label: status.value for status, label in STATUS_LABELS.items()}


def load_a2(run: Path) -> dict[str, dict[str, Any]]:
    """Réponses d'A2 sans erreur (résultats revérifiés s'ils existent)."""
    lines = results_file(run).read_text(encoding="utf-8").splitlines()
    return {r["question_id"]: r for r in map(json.loads, lines) if "error" not in r}


def sample(
    a2: dict[str, dict[str, Any]], golden: dict[str, dict[str, Any]], size: int, seed: int
) -> list[str]:
    """Tirage fixe, moitié questions à réponse chiffrée (contrôle exact), moitié autres."""
    ids = sorted(a2)
    numeric = [i for i in ids if gold_number(str(golden[i]["answer"])) is not None]
    others = [i for i in ids if i not in set(numeric)]
    rng = random.Random(seed)
    half = size // 2
    return sorted(rng.sample(numeric, half) + rng.sample(others, size - half))


def hosted_status(text: str) -> str:
    """Statut affiché par l'agent (dernière ligne du rendu), ou nature de la réponse."""
    for label, status in LABEL_STATUS.items():
        if f"_Statut : {label}._" in text:
            return status
    if "Réponse générale" in text:
        return "general"
    if "Reprise d'une réponse précédente" in text:
        return "conversation" if REUSED_OK in text else "conversation_unchecked"
    return "unknown"


def ask(client: Any, question: str) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    try:
        response = client.responses.create(model=HOSTED_AGENT_NAME, input=question)
    except Exception as error:  # une erreur ne doit pas arrêter la campagne
        return {"started_at": started_at, "error": f"{type(error).__name__}: {error}"}
    text = str(response.output_text)
    return {
        "started_at": started_at,
        "seconds": round(time.perf_counter() - started, 2),
        "response_id": str(response.id),
        "status": hosted_status(text),
        "verified_quotes": text.count("(vérifié)"),
        "text": text,
    }


def _quantile(values: list[float], q: float) -> float | None:
    return sorted(values)[round(q * (len(values) - 1))] if values else None


def score(
    rows: list[dict[str, Any]], a2: dict[str, dict[str, Any]], golden: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Comparaison question par question, puis bilan ; une régression est listée nommément."""
    compared, regressions, improvements = [], [], []
    for row in rows:
        reference = a2[row["question_id"]]
        gold = str(golden[row["question_id"]]["answer"])
        before = check(reference["answer"]["answer"], gold)
        after = check(row.get("text", ""), gold) if "error" not in row else None
        item = {
            "question_id": row["question_id"],
            "a2_status": reference["status"],
            "hosted_status": row.get("status", "error"),
            "numeric_applicable": before.applicable,
            "a2_numeric": before.matched,
            "hosted_numeric": after.matched if after else None,
            "a2_seconds": round(reference["timings"]["total_seconds"], 2),
            "hosted_seconds": row.get("seconds"),
            "a2_cost_usd": reference["cost_usd"],
        }
        compared.append(item)
        was_ok = item["a2_status"] == "verified" and item["a2_numeric"] is not False
        is_ok = item["hosted_status"] == "verified" and item["hosted_numeric"] is not False
        if was_ok and not is_ok:
            regressions.append(item["question_id"])
        elif is_ok and not was_ok:
            improvements.append(item["question_id"])

    def count(key: str, value: Any) -> int:
        return sum(1 for item in compared if item[key] == value)

    hosted_seconds = [i["hosted_seconds"] for i in compared if i["hosted_seconds"] is not None]
    a2_seconds = [i["a2_seconds"] for i in compared]
    return {
        "questions": len(compared),
        "errors": sum(1 for row in rows if "error" in row),
        "status": {
            "a2": dict(
                sorted(
                    {i["a2_status"]: count("a2_status", i["a2_status"]) for i in compared}.items()
                )
            ),
            "hosted": dict(
                sorted(
                    {
                        i["hosted_status"]: count("hosted_status", i["hosted_status"])
                        for i in compared
                    }.items()
                )
            ),
            "same": sum(1 for i in compared if i["a2_status"] == i["hosted_status"]),
        },
        "numeric": {
            "applicable": count("numeric_applicable", True),
            "a2_exact": count("a2_numeric", True),
            "hosted_exact": count("hosted_numeric", True),
        },
        # A2 est chronométré dans le runner local ; l'agent hébergé, côté client (réseau compris).
        "seconds": {
            "a2_median": statistics.median(a2_seconds) if a2_seconds else None,
            "hosted_median": statistics.median(hosted_seconds) if hosted_seconds else None,
            "hosted_p90": _quantile(hosted_seconds, 0.9),
            "hosted_max": max(hosted_seconds, default=None),
        },
        "a2_cost_usd": round(sum(i["a2_cost_usd"] for i in compared), 4),
        "regressions": regressions,
        "improvements": improvements,
        "compared": compared,
    }


def run(args: argparse.Namespace) -> None:
    a2 = load_a2(args.a2)
    golden = load_golden(args.golden)
    selected = sample(a2, golden, args.size, args.seed)
    print(f"{len(selected)} questions (graine {args.seed}) :")
    for question_id in selected:
        print(f"  {question_id}  A2 : {a2[question_id]['status']}")
    print(
        f"Coût estimé ≈ {len(selected) * MEAN_COST_USD:.2f} USD "
        f"(au plus {len(selected) * MAX_QUESTION_COST_USD:.2f} USD : plafond par question)."
    )
    if not args.apply:
        print("\nSimulation : aucun appel. Relancer avec --apply pour interroger l'agent déployé.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "results.jsonl"
    client = agent_client(argparse.Namespace(remote=True))
    rows = []
    with results.open("w", encoding="utf-8") as journal:
        for n, question_id in enumerate(selected, 1):
            # Texte de la question seulement : la réponse gold n'est lue qu'au moment de noter.
            row = {"question_id": question_id, **ask(client, a2[question_id]["question"])}
            rows.append(row)
            journal.write(json.dumps(row, ensure_ascii=False) + "\n")
            journal.flush()
            print(f"[{n}/{len(selected)}] {question_id} : {row.get('status', row.get('error'))}")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "agent": HOSTED_AGENT_NAME,
        "reference": str(args.a2),
        "seed": args.seed,
        **score(rows, a2, golden),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "compared"}, indent=2))


def add_parser(commands: Any) -> None:
    parser = commands.add_parser("regression", help="Comparer l'agent déployé à A2 (simulation).")
    parser.add_argument("--apply", action="store_true", help="Interroger réellement l'agent.")
    parser.add_argument("--a2", type=Path, default=A2_RUN)
    parser.add_argument("--golden", type=Path, default=Path("docs/financebench_open_source.jsonl"))
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--output", type=Path, default=OUTPUT)
