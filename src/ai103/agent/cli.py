"""Agent Foundry : créer sa version (`create --apply`) ou lui poser une question (`ask`).

`create` sans `--apply` affiche seulement la définition qui serait envoyée à Foundry.
"""

import argparse
import json
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from ai103.agent.definition import AGENT_NAME, agent_definition
from ai103.agent.runner import AgentResult, AgentRunner, agent_record
from ai103.agent.tools import ToolBox, load_filings
from ai103.observability import TracedCredential, configure_file_tracing
from ai103.rag.run import run
from ai103.retriever.benchmark import atomic_json, load_questions
from ai103.retriever.core import AzureEmbedder, Retriever

VERSION_FILE = Path("docs/reports/agent/agent-version.json")


def _env(parser: argparse.ArgumentParser, *names: str) -> dict[str, str]:
    load_dotenv()
    values = {name: os.environ.get(name, "").strip() for name in names}
    if missing := [name for name, value in values.items() if not value]:
        parser.error(f"Configurer dans .env : {', '.join(missing)}")
    return values


def create(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    env = _env(parser, "FOUNDRY_PROJECT_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    definition = agent_definition(env["AZURE_OPENAI_CHAT_DEPLOYMENT"], args.effort)
    if not args.apply:
        print(json.dumps(definition.as_dict(), ensure_ascii=False, indent=2))
        print(f"\nSimulation : rien n'est créé. Ajouter --apply pour créer « {AGENT_NAME} ».")
        return
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=env["FOUNDRY_PROJECT_ENDPOINT"], credential=credential) as project,
    ):
        version = project.agents.create_version(
            agent_name=AGENT_NAME,
            definition=definition,
            description="FinanceBench agent: filings catalog, search (C5 retrieval), calculator.",
        )
    details = {
        "created_at": datetime.now(UTC).isoformat(),
        "name": version.name,
        "version": version.version,
        "id": version.id,
        "model": env["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        "reasoning_effort": args.effort,
    }
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(VERSION_FILE, details)
    print(json.dumps(details, indent=2))


def show(result: AgentResult) -> None:
    print("Appels d'outils :")
    for call in result.tool_calls:
        mark = "" if call.executed else " (refusé : budget épuisé)"
        print(f"  [tour {call.round}] {call.name}({call.arguments}) -> {call.summary}{mark}")
    print(f"\nRéponse :\n{result.answer.answer}")
    print(f"\nStatut (calculé par le code) : {result.status.value} | arrêt : {result.stop_reason}")
    for check in result.checks:
        print(f"  [{check.source}] {check.status.value} : « {check.quote} »")
    usage = result.usage
    print(
        f"\nTours : {result.rounds} | durée {result.seconds:.1f} s | tokens : entrée "
        f"{usage.input_tokens} (cache {usage.cached_input_tokens}), sortie {usage.output_tokens} "
        f"(raisonnement {usage.reasoning_tokens}) | coût {usage.cost_usd:.4f} USD"
    )


def build_runner(stack: ExitStack, args: argparse.Namespace, env: dict[str, str]) -> AgentRunner:
    if args.trace:
        provider, stream = configure_file_tracing(args.trace)
        stack.callback(stream.close)
        stack.callback(provider.shutdown)
    credential = TracedCredential(stack.enter_context(DefaultAzureCredential()))
    embedder = AzureEmbedder(
        stack.enter_context(httpx.Client(timeout=30)),
        credential,
        env["AZURE_OPENAI_ENDPOINT"],
        os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    )
    search = stack.enter_context(
        SearchClient(env["AZURE_SEARCH_SERVICE_ENDPOINT"], args.index, credential)
    )
    retriever = Retriever(
        search,
        semantic_configuration=os.environ.get(
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION", "finance-semantic"
        ),
    )
    filings = load_filings(args.metadata)
    project = stack.enter_context(
        AIProjectClient(
            endpoint=env["FOUNDRY_PROJECT_ENDPOINT"], credential=credential, allow_preview=True
        )
    )
    client = stack.enter_context(project.get_openai_client(agent_name=AGENT_NAME))
    return AgentRunner(
        client,
        lambda: ToolBox(retriever, embedder, filings),
        max_searches=args.max_searches,
        max_tool_calls=args.max_tool_calls,
        max_cost_usd=args.max_question_cost,
        max_rounds=args.max_rounds,
    )


def runner_settings(args: argparse.Namespace) -> dict[str, Any]:
    version = json.loads(VERSION_FILE.read_text()) if VERSION_FILE.exists() else {}
    return {
        "agent": AGENT_NAME,
        "agent_version": version.get("version"),
        "index": args.index,
        "retrieval": "semantic_hybrid, pool 20",
        "max_searches": args.max_searches,
        "max_tool_calls": args.max_tool_calls,
        "max_question_cost_usd": args.max_question_cost,
        "max_rounds": args.max_rounds,
    }


def add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default="financial-chunks-v1")
    parser.add_argument("--metadata", type=Path, default=Path("docs/corpus-metadata.jsonl"))
    parser.add_argument("--max-searches", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-question-cost", type=float, default=0.10)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--trace", type=Path, help="Ajouter les spans OpenTelemetry à ce JSONL.")


RUNNER_ENV = (
    "FOUNDRY_PROJECT_ENDPOINT",
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
)


def ask(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    env = _env(parser, *RUNNER_ENV)
    with ExitStack() as stack:
        result = build_runner(stack, args, env).answer(args.question)
    show(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = agent_record(result, runner_settings(args))
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapport : {args.output}")


def run_batch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Toutes les questions du golden set (texte seulement), journal et plafond de coût global."""
    env = _env(parser, *RUNNER_ENV)
    if not 1 <= args.limit <= 150 or args.max_cost_usd <= 0:
        parser.error("1 <= limit <= 150 et max-cost-usd > 0 requis")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("docs/reports/agent/runs") / stamp
    if output.exists() and any(output.iterdir()):
        parser.error("Choisir un dossier vide.")
    output.mkdir(parents=True, exist_ok=True)
    args.trace = args.trace or output / "traces.jsonl"
    questions = load_questions(args.golden)[: args.limit]
    settings = {**runner_settings(args), "limit": args.limit}
    with ExitStack() as stack:
        runner = build_runner(stack, args, env)
        summary = run(
            lambda question: agent_record(runner.answer(question), settings),
            questions,
            output,
            settings,
            args.max_cost_usd,
        )
    print(json.dumps({k: summary[k] for k in ("status", "cost_usd", "stopped")}, indent=2))
    print(f"Résultats : {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create", help="Créer une version de l'agent.")
    create_parser.add_argument("--apply", action="store_true", help="Créer réellement l'agent.")
    create_parser.add_argument("--effort", default="low", choices=["none", "low", "medium"])
    ask_parser = commands.add_parser("ask", help="Poser une question à l'agent.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--output", type=Path)
    add_runner_arguments(ask_parser)
    run_parser = commands.add_parser("run", help="Passer les questions du golden set à l'agent.")
    run_parser.add_argument(
        "--golden", type=Path, default=Path("docs/financebench_open_source.jsonl")
    )
    run_parser.add_argument("--limit", type=int, default=150)
    run_parser.add_argument("--max-cost-usd", type=float, default=8.0)
    run_parser.add_argument("--output", type=Path, help="Dossier vide ; par défaut horodaté.")
    add_runner_arguments(run_parser)
    args = parser.parse_args()
    if args.command == "create":
        create(args, parser)
    elif args.command == "ask":
        ask(args, parser)
    else:
        run_batch(args, parser)


if __name__ == "__main__":
    main()
