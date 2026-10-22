"""Suivre une campagne d'agent en cours, en relisant son journal (aucun appel Azure)."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def snapshot(records: list[dict[str, Any]]) -> str:
    answered = [r for r in records if "error" not in r]
    if not answered:
        return f"{len(records)} questions, aucune réponse encore."
    searches = [sum(c["name"] == "search_filings" for c in r["tool_calls"]) for r in answered]
    refused = sum(not c["executed"] for r in answered for c in r["tool_calls"])
    cost = sum(r["cost_usd"] for r in answered)
    durations = [r["timings"]["total_seconds"] for r in answered]
    return "\n".join(
        [
            f"Questions : {len(records)} (erreurs : {len(records) - len(answered)})",
            f"Statuts   : {dict(Counter(r['status'] for r in answered))}",
            f"Arrêts    : {dict(Counter(r['stop_reason'] for r in answered))}",
            f"Recherches par question : moyenne {mean(searches):.1f}, max {max(searches)} | "
            f"appels refusés : {refused}",
            f"Tours : moyenne {mean(r['rounds'] for r in answered):.1f} | durée médiane "
            f"{median(durations):.0f} s",
            f"Coût : {cost:.2f} USD au total, {cost / len(answered):.3f} USD par question, "
            f"max {max(r['cost_usd'] for r in answered):.3f}",
        ]
    )


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--every", type=int, default=30, help="Rafraîchir toutes les N secondes.")
    args = parser.parse_args()
    while True:
        print(
            f"\n=== {time.strftime('%H:%M:%S')} ===\n{snapshot(read(args.run / 'results.jsonl'))}"
        )
        if (args.run / "summary.json").exists():
            print("Campagne terminée.")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
