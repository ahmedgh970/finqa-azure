"""Test de connexion à gpt-5.2 : un seul appel court, sans document du corpus."""

import argparse
import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel

from ai103.rag.generator import AzureGenerator, openai_client


class Ping(BaseModel):
    answer: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effort", default="low", choices=["none", "minimal", "low", "medium"])
    args = parser.parse_args()
    load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    if not endpoint or not deployment:
        parser.error("Configurer AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_CHAT_DEPLOYMENT dans .env.")
    with DefaultAzureCredential() as credential, openai_client(endpoint, credential) as client:
        generator = AzureGenerator(
            client, deployment, reasoning_effort=args.effort, max_output_tokens=200
        )
        result = generator.generate(
            "Réponds en un mot, au format demandé.", "Réponds simplement : OK", Ping
        )
    usage = result.usage
    print(f"Modèle : {result.model} | réponse : {result.output.answer!r}")
    print(f"Latence : {result.latency_seconds:.2f} s")
    print(
        f"Tokens : entrée {usage.input_tokens} (cache {usage.cached_input_tokens}), "
        f"sortie {usage.output_tokens} (raisonnement {usage.reasoning_tokens})"
    )
    print(f"Coût estimé : {usage.cost_usd:.6f} USD")


if __name__ == "__main__":
    main()
