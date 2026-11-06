"""Empaqueter, déployer et interroger le hosted agent (déploiement du code en ZIP, sans conteneur).

`deploy` est en simulation par défaut ; `--apply` crée réellement une version de l'agent.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ai103.hosted.chat import FINAL_EVENTS
from ai103.retriever.benchmark import atomic_json

HOSTED_AGENT_NAME = "finance-chat-agent"
SOURCE = Path(__file__).resolve().parents[1]  # src/ai103
PACKAGES = ("agent", "hosted", "rag", "retriever")
MODULES = ("__init__.py", "observability.py")
# Outils d'évaluation lancés depuis ce poste : jamais dans l'agent déployé.
EXCLUDED = ("hosted/regression.py",)
REQUIREMENTS = (
    "azure-ai-agentserver-responses",
    "azure-ai-projects",
    "azure-identity",
    "azure-monitor-opentelemetry-exporter",
    "azure-search-documents",
    "httpx",
    "openai",
    "opentelemetry-sdk",
    "pydantic",
)
# Adresses et noms de déploiement, jamais de secret ni le choix d'identité locale
# (AZURE_TOKEN_CREDENTIALS forcerait Azure CLI au lieu de l'identité d'agent).
ENVIRONMENT = (
    "FOUNDRY_PROJECT_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_SEARCH_SERVICE_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
)
# Veille après 10 min sans requête (décision du 6 novembre) : facturé jusque-là, environ
# 0,01 USD par session ; évite le démarrage à froid (≈ 10 s) entre deux questions rapprochées.
IDLE_TIMEOUT = timedelta(minutes=10)
ENTRY_POINT = "from ai103.hosted.app import main\n\nmain()\n"
REPORTS = Path("docs/reports/hosted")


def requirements() -> str:
    """Versions installées et testées localement, dépendances transitives comprises.

    Épingler les seules dépendances directes ne suffit pas : le 6 novembre, le build distant a pris
    `azure-ai-agentserver-core` 2.3.0 (2.2.0 en local), incompatible avec `responses` 2.3.0.
    """
    pinned: dict[str, str] = {}
    pending = list(REQUIREMENTS)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in pinned:
            continue
        pinned[name] = importlib.metadata.version(name)
        for spec in importlib.metadata.requires(name) or []:
            requirement = Requirement(spec)
            # Marqueurs évalués sur ce poste (Linux, Python 3.13, comme le runtime), sans extras.
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return "".join(f"{name}=={pinned[name]}\n" for name in sorted(pinned))


def _add(archive: zipfile.ZipFile, name: str, content: str | bytes) -> None:
    # Fichier ordinaire lisible par tous (0644) : une entrée 0600 sans type, comme celles de
    # `writestr`, n'est pas vue par le build distant (« No Python dependency manifest found »).
    # Date fixe : le même code donne toujours le même SHA-256.
    entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    entry.external_attr = (stat.S_IFREG | 0o644) << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(entry, content)


def package(output: Path, metadata: Path) -> str:
    """ZIP plat : main.py, requirements.txt, notre code utile et le catalogue des rapports."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        _add(archive, "main.py", ENTRY_POINT)
        _add(archive, "requirements.txt", requirements())
        _add(archive, "corpus-metadata.jsonl", metadata.read_bytes())
        for module in MODULES:
            _add(archive, f"ai103/{module}", (SOURCE / module).read_bytes())
        for name in PACKAGES:
            for path in sorted((SOURCE / name).glob("*.py")):
                if f"{name}/{path.name}" not in EXCLUDED:
                    _add(archive, f"ai103/{name}/{path.name}", path.read_bytes())
    return hashlib.sha256(output.read_bytes()).hexdigest()


def environment() -> dict[str, str]:
    missing = [name for name in ENVIRONMENT if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(f"Variables absentes de .env : {', '.join(missing)}")
    return {name: os.environ[name].strip() for name in ENVIRONMENT}


def definition_document(env: dict[str, str]) -> dict[str, Any]:
    """Définition envoyée à Foundry, sous forme lisible pour la simulation et le rapport."""
    return {
        "kind": "hosted",
        "cpu": "0.5",
        "memory": "1Gi",
        "code_configuration": {
            "runtime": "python_3_13",
            "entry_point": ["python", "main.py"],
            "dependency_resolution": "remote_build",
        },
        "protocol_versions": [{"protocol": "responses", "version": "2.0.0"}],
        "session_configuration": {"idle_timeout_seconds": int(IDLE_TIMEOUT.total_seconds())},
        "environment_variables": env,
    }


def deploy(args: argparse.Namespace) -> None:
    env = environment()
    sha = package(args.zip, args.metadata)
    document = definition_document(env)
    print(f"ZIP : {args.zip} ({args.zip.stat().st_size // 1024} Ko), SHA-256 {sha}")
    print(json.dumps(document, indent=2))
    if not args.apply:
        print("\nSimulation : rien n'a été créé. Relancer avec --apply pour déployer.")
        return
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        CodeConfiguration,
        HostedAgentDefinition,
        ProtocolVersionRecord,
        SessionConfiguration,
    )
    from azure.identity import DefaultAzureCredential

    definition = HostedAgentDefinition(
        cpu=document["cpu"],
        memory=document["memory"],
        code_configuration=CodeConfiguration(**document["code_configuration"]),
        protocol_versions=[ProtocolVersionRecord(protocol="responses", version="2.0.0")],
        session_configuration=SessionConfiguration(idle_timeout_seconds=IDLE_TIMEOUT),
        environment_variables=env,
    )
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(env["FOUNDRY_PROJECT_ENDPOINT"], credential, allow_preview=True) as project,
        args.zip.open("rb") as code,
    ):
        created = project.agents.create_version_from_code(
            HOSTED_AGENT_NAME,
            definition=definition,
            code=code,
            code_zip_sha256=sha,
            description="Chatbot financier : mémoire de conversation et outils du corpus",
        )
        print(f"Version {created.version} créée, construction distante en cours…")
        status, deadline = "creating", time.monotonic() + 20 * 60
        while status not in ("active", "failed") and time.monotonic() < deadline:
            time.sleep(10)
            version = project.agents.get_version(HOSTED_AGENT_NAME, created.version)
            status = str(version["status"])
            print(f"  statut : {status}")
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "agent": HOSTED_AGENT_NAME,
            "version": created.version,
            "status": status,
            "error": version.get("error") if status == "failed" else None,
            "zip_sha256": sha,
            "definition": document,
        }
    REPORTS.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORTS / f"deploy-v{created.version}.json", report)
    print(json.dumps(report, indent=2, default=str))


def serve(args: argparse.Namespace) -> None:
    """Le serveur du hosted agent, lancé sur ce poste (http://localhost:8088)."""
    os.environ.setdefault("CORPUS_METADATA", str(args.metadata))
    if args.azure_monitor:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        from ai103.observability import application_insights_connection_string

        with (
            DefaultAzureCredential() as credential,
            AIProjectClient(os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential) as project,
        ):
            # La bibliothèque du protocole active l'export si cette variable existe.
            os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = (
                application_insights_connection_string(project)
            )
    from ai103.hosted.app import main

    main()


def agent_client(args: argparse.Namespace) -> Any:
    """Client Responses : le serveur local, ou l'agent déployé (jeton Entra)."""
    if args.remote:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        project = AIProjectClient(os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential)
        return project.get_openai_client(agent_name=HOSTED_AGENT_NAME)
    from openai import OpenAI

    return OpenAI(base_url=args.url, api_key="local")


def send(client: Any, question: str, previous: str | None) -> str:
    """Une question ; `previous` relie la réponse à la précédente, donc à toute la conversation."""
    started = time.perf_counter()
    # Champ omis au premier tour : Foundry refuse `previous_response_id: null` (HTTP 400).
    chained = {"previous_response_id": previous} if previous else {}
    response, first_text = None, None
    with client.responses.create(
        model=HOSTED_AGENT_NAME, input=question, stream=True, **chained
    ) as events:
        for event in events:
            if event.type == "response.output_text.delta":
                first_text = first_text or time.perf_counter() - started
                print(event.delta, end="", flush=True)
            elif event.type in FINAL_EVENTS:
                response = event.response
    if response is None:
        raise RuntimeError("Flux de l'agent terminé sans réponse finale.")
    shown = f"premier texte {first_text:.1f} s | " if first_text else ""
    print(f"\n\n[id : {response.id} | {shown}total {time.perf_counter() - started:.1f} s]")
    return str(response.id)


def ask(args: argparse.Namespace) -> None:
    """Une seule question (scripts, tests) ; `--previous` poursuit une conversation."""
    response_id = send(agent_client(args), args.question, args.previous)
    print(f"Question suivante dans la même conversation : --previous {response_id}")


def chat(args: argparse.Namespace) -> None:
    """Conversation interactive : l'outil garde l'identifiant, comme le ferait une interface."""
    client, previous = agent_client(args), None
    print("Conversation ouverte. « /nouveau » pour recommencer, ligne vide pour quitter.")
    while question := input("\nVous : ").strip():
        if question == "/nouveau":
            previous = None
            print("Nouvelle conversation : l'agent ne voit plus les échanges précédents.")
            continue
        print()
        previous = send(client, question, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = Path("docs/corpus-metadata.jsonl")
    deploy_parser = commands.add_parser("deploy", help="Empaqueter et déployer (simulation).")
    deploy_parser.add_argument("--apply", action="store_true", help="Créer réellement la version.")
    deploy_parser.add_argument("--zip", type=Path, default=Path(f"dist/{HOSTED_AGENT_NAME}.zip"))
    deploy_parser.add_argument("--metadata", type=Path, default=metadata)
    serve_parser = commands.add_parser("serve", help="Lancer le serveur hosted en local.")
    serve_parser.add_argument("--metadata", type=Path, default=metadata)
    serve_parser.add_argument("--azure-monitor", action="store_true")
    ask_parser = commands.add_parser("ask", help="Poser une question (local ou déployé).")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--previous", help="Identifiant de la réponse précédente.")
    ask_parser.add_argument("--remote", action="store_true", help="Agent déployé dans Foundry.")
    ask_parser.add_argument("--url", default="http://localhost:8088")
    chat_parser = commands.add_parser("chat", help="Conversation interactive (local ou déployé).")
    chat_parser.add_argument("--remote", action="store_true", help="Agent déployé dans Foundry.")
    chat_parser.add_argument("--url", default="http://localhost:8088")
    from ai103.hosted import regression  # importe ce module : import différé, sans cycle

    regression.add_parser(commands)
    args = parser.parse_args()
    load_dotenv()
    commands_by_name = {"deploy": deploy, "serve": serve, "ask": ask, "chat": chat}
    {**commands_by_name, "regression": regression.run}[args.command](args)
    sys.stdout.flush()
