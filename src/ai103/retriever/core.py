"""Recherche Azure en lecture seule ; aucune génération de réponse."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import httpx
from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

DIMENSIONS = 1024
FIELDS = ["chunk_id", "content", "source", "company", "fiscal_year", "page_start", "page_end"]


class Mode(StrEnum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"
    # Hybride puis reclassement des 50 premiers par le semantic ranker d'Azure AI Search.
    SEMANTIC_HYBRID = "semantic_hybrid"


BENCHMARK_MODES = (Mode.LEXICAL, Mode.VECTOR, Mode.HYBRID)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    source: str
    score: float
    company: str | None = None
    fiscal_year: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    # Score du semantic ranker (0 à 4), présent seulement en mode sémantique.
    reranker_score: float | None = None
    # Rang (1 = premier) dans le lot hybride avant reclassement, si un lot est imposé.
    initial_rank: int | None = None


def validate_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
        raise ValueError("Le vecteur doit contenir 1024 nombres finis.")
    if not any(vector):
        raise ValueError("Le vecteur nul est invalide.")
    return list(vector)


class AzureEmbedder:
    """Client embeddings v1 ; le client HTTP et l'identité appartiennent à l'appelant.

    Le jeton Entra est mis en cache et renouvelé avant expiration : en redemander un à chaque
    appel coûtait environ 0,8 s par question (mesure OpenTelemetry du 19 octobre).
    """

    def __init__(
        self, client: httpx.Client, credential: TokenCredential, endpoint: str, deployment: str
    ) -> None:
        if not endpoint.startswith("https://") or not deployment.strip():
            raise ValueError("Endpoint HTTPS et nom de déploiement requis.")
        self.client = client
        self.token = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        self.url = endpoint.rstrip("/") + "/openai/v1/embeddings"
        self.deployment = deployment
        self.last_usage: dict[str, Any] = {}

    def __call__(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Question vide.")
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token()}"},
            json={"model": self.deployment, "input": text, "dimensions": DIMENSIONS},
        )
        response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("usage", {})
        return validate_vector(body["data"][0]["embedding"])


class Retriever:
    def __init__(self, client: SearchClient, semantic_configuration: str | None = None) -> None:
        self.client = client
        self.semantic_configuration = semantic_configuration

    def retrieve(
        self,
        question: str,
        *,
        mode: Mode,
        k: int = 5,
        vector: Sequence[float] | None = None,
        candidates: int = 50,
        filter_expression: str | None = None,
        pool: int | None = None,
    ) -> list[Chunk]:
        """`pool` : nombre de candidats hybrides que le semantic ranker peut reclasser.

        Sans `pool`, Azure reclasse les 50 premiers résultats. Avec `pool`, une première
        requête hybride fixe le lot, puis la requête sémantique est filtrée sur ces chunks :
        on compare ainsi le reclassement et l'ordre RRF sur exactement les mêmes candidats.
        """
        mode = Mode(mode)
        if pool is not None:
            if mode is not Mode.SEMANTIC_HYBRID or not k <= pool <= 50:
                raise ValueError("pool réservé au mode semantic_hybrid, avec k <= pool <= 50.")
            first = self.retrieve(
                question,
                mode=Mode.HYBRID,
                k=pool,
                vector=vector,
                candidates=max(candidates, pool),
                filter_expression=filter_expression,
            )
            if not first:
                return []
            ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(first, 1)}
            ids = ",".join(ranks)
            reranked = self.retrieve(
                question,
                mode=Mode.SEMANTIC_HYBRID,
                k=k,
                vector=vector,
                candidates=max(candidates, pool),
                filter_expression=f"search.in(chunk_id, '{ids}', ',')",
            )
            return [replace(chunk, initial_rank=ranks.get(chunk.chunk_id)) for chunk in reranked]
        if not question.strip() or not 1 <= k <= 1000 or not k <= candidates <= 1000:
            raise ValueError("Question non vide et 1 <= k <= candidates <= 1000 requis.")
        options: dict[str, Any] = {
            "top": k,
            "select": FIELDS,
            "filter": filter_expression,
        }
        if mode is Mode.SEMANTIC_HYBRID:
            if not self.semantic_configuration:
                raise ValueError("Configuration sémantique de l'index requise.")
            # Le ranker ne reclasse que les 50 premiers résultats de la fusion RRF.
            options.update(
                search_fields=["content"],
                query_type="semantic",
                semantic_configuration_name=self.semantic_configuration,
            )
        elif mode != Mode.VECTOR:
            options.update(search_fields=["content"], query_type="simple")
        if mode != Mode.LEXICAL:
            if vector is None:
                raise ValueError("Embedding requis pour la recherche vectorielle/hybride.")
            options.update(
                vector_queries=[
                    VectorizedQuery(
                        vector=validate_vector(vector),
                        fields="content_vector",
                        k_nearest_neighbors=candidates,
                    )
                ],
                vector_filter_mode="preFilter",
            )
        rows = self.client.search(search_text=question if mode != Mode.VECTOR else None, **options)
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                source=row["source"],
                score=float(row["@search.score"]),
                company=row.get("company"),
                fiscal_year=row.get("fiscal_year"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                reranker_score=row.get("@search.reranker_score"),
            )
            for row in rows
        ]


@dataclass
class Runner:
    retrievers: dict[str, Retriever]
    embed: Callable[[str], list[float]] | None = None

    def run(
        self,
        question: str,
        *,
        modes: Sequence[Mode] = BENCHMARK_MODES,
        k: int = 5,
        candidates: int = 50,
        filter_expression: str | None = None,
    ) -> dict[str, dict[str, list[Chunk]]]:
        modes = tuple(Mode(mode) for mode in modes)
        if not question.strip() or not modes or not 1 <= k <= candidates <= 1000:
            raise ValueError("Question, modes et limites invalides.")
        vector = None
        if any(mode != Mode.LEXICAL for mode in modes):
            if self.embed is None:
                raise ValueError("Client embeddings requis.")
            vector = validate_vector(self.embed(question))
        return {
            name: {
                mode.value: retriever.retrieve(
                    question,
                    mode=mode,
                    k=k,
                    vector=vector,
                    candidates=candidates,
                    filter_expression=filter_expression,
                )
                for mode in modes
            }
            for name, retriever in self.retrievers.items()
        }
