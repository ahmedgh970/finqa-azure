"""Appels visibles à /api/tags, /api/chat et /api/embed."""

import math
from collections.abc import Sequence
from typing import Any

import httpx

from ai103.models.contracts import ChatMessage, GenerationResult


class OllamaResponseError(ValueError):
    """Ollama a répondu, mais la réponse ne permet pas de valider le contrôle."""


def _object(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    value: object = response.json()
    if not isinstance(value, dict):
        raise OllamaResponseError("Réponse JSON Ollama invalide : objet attendu.")
    if value.get("error"):
        raise OllamaResponseError("Ollama a renvoyé une erreur applicative.")
    return value


def _tokens(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class OllamaProvider:
    def __init__(self, client: httpx.Client, *, chat_model: str, embedding_model: str) -> None:
        self._client = client
        self._chat_model = chat_model
        self._embedding_model = embedding_model

    def list_local_models(self) -> list[str]:
        payload = _object(self._client.get("/api/tags"))
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Réponse Ollama sans liste de modèles.")
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                raise OllamaResponseError("Entrée de modèle Ollama invalide.")
            # Le diagnostic local exclut les modèles relayés au cloud.
            if not model.get("remote_host") and "cloud" not in model["name"].lower():
                names.append(model["name"])
        return names

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        payload = _object(
            self._client.post(
                "/api/chat",
                json={
                    "model": self._chat_model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 64},
                    "keep_alive": 0,
                },
            )
        )
        message = payload.get("message")
        if not isinstance(message, dict):
            raise OllamaResponseError("Réponse Ollama sans message.")
        content = message.get("content")
        if payload.get("done") is not True or not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("La génération Ollama n'a pas produit de texte complet.")
        model = payload.get("model")
        if not isinstance(model, str):
            raise OllamaResponseError("Réponse Ollama sans nom de modèle.")
        return GenerationResult(
            text=content,
            model=model,
            prompt_tokens=_tokens(payload.get("prompt_eval_count")),
            completion_tokens=_tokens(payload.get("eval_count")),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Au moins un texte est nécessaire pour calculer les embeddings.")
        payload = _object(
            self._client.post(
                "/api/embed",
                json={
                    "model": self._embedding_model,
                    "input": list(texts),
                    "truncate": False,
                    "keep_alive": 0,
                },
            )
        )
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaResponseError("Le nombre d'embeddings ne correspond pas aux textes.")
        vectors: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise OllamaResponseError("Vecteur d'embedding vide ou invalide.")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in vector
            ):
                raise OllamaResponseError("L'embedding doit contenir des nombres finis.")
            vectors.append([float(v) for v in vector])
        if len({len(vector) for vector in vectors}) != 1:
            raise OllamaResponseError("Les embeddings ont des dimensions différentes.")
        return vectors
