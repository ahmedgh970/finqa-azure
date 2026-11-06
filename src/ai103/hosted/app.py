"""Serveur du hosted agent : protocole Responses de Foundry, une question par requête.

La plateforme fournit l'historique de la conversation (mémoire), l'identité d'agent et la
chaîne de connexion Application Insights ; la bibliothèque du protocole exporte alors les traces,
nos spans compris. En local, le même serveur écoute sur http://localhost:8088.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any

import httpx
from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from ai103.agent.runner import AgentResult
from ai103.agent.tools import ToolBox, load_filings
from ai103.hosted.chat import AnswerStream, DirectRunner, history_messages, render
from ai103.retriever.core import AzureEmbedder, Retriever

HISTORY_ITEMS = 20  # messages des tours précédents relus à chaque question

DEFAULTS = {
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.2",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
    "AZURE_SEARCH_INDEX": "financial-chunks-v1",
    "AZURE_SEARCH_SEMANTIC_CONFIGURATION": "finance-semantic",
    "CORPUS_METADATA": "corpus-metadata.jsonl",
}


def setting(name: str) -> str:
    value = os.environ.get(name, "").strip() or DEFAULTS.get(name, "")
    if not value:
        raise RuntimeError(f"Variable d'environnement manquante : {name}")
    return value


@cache
def runner() -> DirectRunner:
    """Clients créés une fois par processus et partagés par les requêtes de la session."""
    credential = DefaultAzureCredential()  # identité d'agent dans Foundry, Azure CLI en local
    project = AIProjectClient(endpoint=setting("FOUNDRY_PROJECT_ENDPOINT"), credential=credential)
    embedder = AzureEmbedder(
        httpx.Client(timeout=30),
        credential,
        setting("AZURE_OPENAI_ENDPOINT"),
        setting("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    search = SearchClient(
        setting("AZURE_SEARCH_SERVICE_ENDPOINT"), setting("AZURE_SEARCH_INDEX"), credential
    )
    retriever = Retriever(
        search, semantic_configuration=setting("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
    )
    filings = load_filings(Path(setting("CORPUS_METADATA")))
    return DirectRunner(
        project.get_openai_client(),
        lambda: ToolBox(retriever, embedder, filings),
        model=setting("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    )


async def respond(question: str, history: list[Any], agent: DirectRunner) -> str:
    """La boucle est synchrone : elle tourne dans un thread pour ne pas bloquer le serveur."""
    messages = history_messages(history)
    result = await asyncio.to_thread(agent.answer, question, messages)
    return render(result, messages)


async def stream_answer(
    question: str, history: list[Any], agent: DirectRunner
) -> AsyncIterator[str]:
    """Texte de la réponse au fil de la génération, puis sources vérifiées et statut.

    La boucle tourne dans un thread ; ses morceaux de texte passent par une file asyncio. Le texte
    total est celui de `respond` : si la réponse finale diffère de ce qui a été montré (arrêt sur
    une limite, filtre de contenu), elle est ajoutée en entier après un séparateur.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    messages = history_messages(history)
    extractor = AnswerStream()

    def on_text(delta: str) -> None:
        if text := extractor.feed(delta):
            loop.call_soon_threadsafe(queue.put_nowait, text)

    def run() -> AgentResult:
        # La fin passe par le même canal que le texte, depuis le même thread : elle arrive après
        # le dernier morceau (posée depuis la boucle, elle pouvait le devancer).
        try:
            return agent.answer(question, messages, on_text)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(run))
    shown = ""
    while (chunk := await queue.get()) is not None:
        shown += chunk
        yield chunk
    final = render(await task, messages)
    if final.startswith(shown):
        yield final[len(shown) :]
    else:
        yield f"\n\n---\n\n{final}"


def create_app(*, observability: bool = True) -> ResponsesAgentServerHost:
    """`observability=False` : pas de configuration des traces par la bibliothèque (tests)."""
    extra: dict[str, Any] = {} if observability else {"configure_observability": None}
    app = ResponsesAgentServerHost(
        options=ResponsesServerOptions(default_fetch_history_count=HISTORY_ITEMS), **extra
    )

    @app.response_handler
    async def handler(
        request: Any, context: ResponseContext, cancellation_signal: asyncio.Event
    ) -> TextResponse:
        question = await context.get_input_text()
        history = list(await context.get_history())
        # Streaming : le client voit la réponse s'écrire ; sans `stream`, le texte est regroupé.
        return TextResponse(context, request, text=stream_answer(question, history, runner()))

    return app


def main() -> None:
    create_app().run()
