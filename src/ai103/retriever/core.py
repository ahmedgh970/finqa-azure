"""Recherche Azure en lecture seule ; aucune génération de réponse."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
from azure.core.credentials import TokenCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

DIMENSIONS = 1024
FIELDS = ["chunk_id", "content", "source", "company", "fiscal_year", "page_start", "page_end"]


class Mode(StrEnum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


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


def validate_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
        raise ValueError("Le vecteur doit contenir 1024 nombres finis.")
    if not any(vector):
        raise ValueError("Le vecteur nul est invalide.")
    return list(vector)


class AzureEmbedder:
    """Client embeddings v1 ; le client HTTP et l'identité appartiennent à l'appelant."""

    def __init__(
        self, client: httpx.Client, credential: TokenCredential, endpoint: str, deployment: str
    ) -> None:
        if not endpoint.startswith("https://") or not deployment.strip():
            raise ValueError("Endpoint HTTPS et nom de déploiement requis.")
        self.client = client
        self.credential = credential
        self.url = endpoint.rstrip("/") + "/openai/v1/embeddings"
        self.deployment = deployment
        self.last_usage: dict[str, Any] = {}

    def __call__(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Question vide.")
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {token.token}"},
            json={"model": self.deployment, "input": text, "dimensions": DIMENSIONS},
        )
        response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("usage", {})
        return validate_vector(body["data"][0]["embedding"])


class Retriever:
    def __init__(self, client: SearchClient) -> None:
        self.client = client

    def retrieve(
        self,
        question: str,
        *,
        mode: Mode,
        k: int = 5,
        vector: Sequence[float] | None = None,
        candidates: int = 50,
        filter_expression: str | None = None,
    ) -> list[Chunk]:
        mode = Mode(mode)
        if not question.strip() or not 1 <= k <= 1000 or not k <= candidates <= 1000:
            raise ValueError("Question non vide et 1 <= k <= candidates <= 1000 requis.")
        options: dict[str, Any] = {
            "top": k,
            "select": FIELDS,
            "filter": filter_expression,
        }
        if mode != Mode.VECTOR:
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
        modes: Sequence[Mode] = tuple(Mode),
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
