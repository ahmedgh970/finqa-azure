"""Commande : python -m ai103 doctor [--quick] [--json]."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ai103.config import ConfigurationError, Settings
from ai103.diagnostics.runner import CheckResult, report_exit_code, run_diagnostics
from ai103.logging_setup import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostics de l'environnement Finance RAG")
    parser.add_argument("command", choices=["doctor"])
    parser.add_argument("--quick", action="store_true", help="Sans génération ni embeddings")
    parser.add_argument("--json", action="store_true", help="Rapport JSON sur stdout")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    configure_logging()
    # Le fichier est explicite ; le programme ne cherche pas de .env dans les dossiers parents.
    # Les variables déjà exportées dans le shell ont priorité sur le fichier.
    load_dotenv(args.env_file, override=False)
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        results = [CheckResult("configuration", "failed", 0, str(exc))]
        exit_code = 2
    else:
        os.environ["AZURE_TOKEN_CREDENTIALS"] = settings.azure_token_credentials
        results = run_diagnostics(settings, quick=args.quick)
        exit_code = report_exit_code(results)
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "quick" if args.quick else "full",
                    "checks": [result.to_dict() for result in results],
                    "exit_code": exit_code,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for result in results:
            print(
                f"[{result.status.upper():7}] {result.name} "
                f"({result.duration_ms:.0f} ms) — {result.message}"
            )
            if result.details:
                print("          " + json.dumps(result.details, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
