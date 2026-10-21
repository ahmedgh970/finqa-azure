from unittest.mock import Mock

import httpx
import pytest
from azure.core.credentials import AccessToken

from ai103.retriever import AzureEmbedder, Chunk, Mode, Retriever, Runner
from ai103.retriever.core import BENCHMARK_MODES
from ai103.retriever.evaluation import compare, load_question

VECTOR = [0.1] * 1024


@pytest.mark.parametrize("mode", BENCHMARK_MODES)
def test_query_modes(mode):
    client = Mock()
    client.search.return_value = [
        {
            "chunk_id": "a",
            "content": "cash flow",
            "source": "s",
            "@search.score": 0.5,
        }
    ]
    chunks = Retriever(client).retrieve("cash", mode=mode, vector=VECTOR, k=3)
    options = client.search.call_args.kwargs
    assert options["top"] == 3
    assert options["search_text"] == (None if mode == Mode.VECTOR else "cash")
    assert ("vector_queries" in options) == (mode != Mode.LEXICAL)
    if mode != Mode.LEXICAL:
        assert options["vector_queries"][0].k_nearest_neighbors == 50
        assert options["vector_queries"][0].fields == "content_vector"
    assert chunks[0].page_start is None
    assert options["query_type"] == "simple" if mode != Mode.VECTOR else "query_type" not in options


def test_semantic_hybrid_reranks_with_index_configuration():
    client = Mock()
    client.search.return_value = [
        {
            "chunk_id": "a",
            "content": "cash flow",
            "source": "s",
            "@search.score": 0.03,
            "@search.reranker_score": 2.7,
        }
    ]
    retriever = Retriever(client, semantic_configuration="finance-semantic")
    chunks = retriever.retrieve("cash", mode=Mode.SEMANTIC_HYBRID, vector=VECTOR, k=10)
    options = client.search.call_args.kwargs
    assert options["query_type"] == "semantic"
    assert options["semantic_configuration_name"] == "finance-semantic"
    assert options["search_text"] == "cash" and "vector_queries" in options
    assert chunks[0].reranker_score == 2.7 and chunks[0].score == 0.03
    with pytest.raises(ValueError):
        Retriever(client).retrieve("cash", mode=Mode.SEMANTIC_HYBRID, vector=VECTOR)


def test_default_runner_modes_exclude_semantic():
    embed = Mock(return_value=VECTOR)
    retriever = Mock()
    retriever.retrieve.return_value = []
    result = Runner({"index": retriever}, embed).run("cash")
    assert set(result["index"]) == {"lexical", "vector", "hybrid"}


def test_reuses_embedding_across_indexes_and_modes():
    embed = Mock(return_value=VECTOR)
    retriever = Mock()
    Runner({"native": retriever, "layout": retriever}, embed).run("question")
    embed.assert_called_once_with("question")
    assert retriever.retrieve.call_count == 6


def test_lexical_does_not_embed():
    embed = Mock()
    Runner({"a": Mock()}, embed).run("question", modes=[Mode.LEXICAL])
    embed.assert_not_called()


@pytest.mark.parametrize("vector", [None, [1.0], [float("nan")] * 1024, [0.0] * 1024])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        Retriever(Mock()).retrieve("q", mode=Mode.VECTOR, vector=vector)


def test_invalid_limits_do_not_call_embedding():
    embed = Mock()
    with pytest.raises(ValueError):
        Runner({}, embed).run("q", k=51, candidates=50)
    embed.assert_not_called()


def test_embedding_request():
    def respond(request):
        assert request.url.path == "/openai/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer fake-token"
        import json

        body = json.loads(request.content)
        assert body == {"model": "deployed", "input": "question", "dimensions": 1024}
        return httpx.Response(
            200, json={"data": [{"embedding": VECTOR}], "usage": {"total_tokens": 2}}
        )

    class Credential:
        calls = 0

        def get_token(self, *scopes, **kwargs):
            self.calls += 1
            return AccessToken("fake-token", 9_999_999_999)

    credential = Credential()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embed = AzureEmbedder(client, credential, "https://example.openai.azure.com", "deployed")
        assert embed("question") == VECTOR
        assert embed("question") == VECTOR
        assert embed.last_usage["total_tokens"] == 2
    # Jeton mis en cache : un seul appel à l'identité pour deux embeddings.
    assert credential.calls == 1


def test_evidence_diagnostic_requires_correct_document():
    question = {"evidence": [{"doc_name": "RIGHT", "evidence_text": "cash 123"}]}
    wrong = Chunk("a", "cash 123", "https://s.blob.core.windows.net/c/WRONG.pdf", 1.0)
    assert not compare([wrong], question)["document_hit_at_k"]
    assert compare([wrong], question)["evidence_diagnostics"][0]["unique_gold_token_coverage"] == 0
    right = Chunk(
        "b", "<td>cash</td><td>123</td>", "https://s.blob.core.windows.net/c/RIGHT.pdf", 0.1
    )
    result = compare([wrong, right], question)
    assert result["document_reciprocal_rank_at_k"] == 0.5
    assert result["evidence_diagnostics"][0]["normalized_exact_in_single_chunk"]


def test_unknown_question(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text('{"financebench_id":"a"}\n')
    with pytest.raises(ValueError):
        load_question(path, "missing")


def test_semantic_rerank_of_a_fixed_hybrid_pool():
    client = Mock()
    pool_rows = [
        {"chunk_id": f"c{i}", "content": "x", "source": "s", "@search.score": 1 / i}
        for i in range(1, 21)
    ]
    reranked = [
        {
            "chunk_id": cid,
            "content": "x",
            "source": "s",
            "@search.score": 0.1,
            "@search.reranker_score": score,
        }
        for cid, score in (("c15", 3.1), ("c2", 2.9), ("c1", 2.0))
    ]
    client.search.side_effect = [pool_rows, reranked]
    retriever = Retriever(client, semantic_configuration="finance-semantic")
    chunks = retriever.retrieve(
        "net PP&E",
        mode=Mode.SEMANTIC_HYBRID,
        vector=VECTOR,
        k=3,
        pool=20,
        filter_expression="company eq '3M'",
    )
    first, second = (call.kwargs for call in client.search.call_args_list)
    assert first["query_type"] == "simple" and first["top"] == 20
    assert first["filter"] == "company eq '3M'"
    assert second["query_type"] == "semantic" and second["top"] == 3
    assert second["filter"].startswith("search.in(chunk_id, 'c1,c2,c3,")
    assert second["filter"].endswith("c20', ',')")
    # Le passage reclassé premier venait de la 15e place du lot hybride.
    assert [(c.chunk_id, c.initial_rank) for c in chunks] == [("c15", 15), ("c2", 2), ("c1", 1)]


def test_pool_is_only_for_semantic_mode_and_bounded():
    retriever = Retriever(Mock(), semantic_configuration="finance-semantic")
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode=Mode.HYBRID, vector=VECTOR, k=10, pool=20)
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode=Mode.SEMANTIC_HYBRID, vector=VECTOR, k=10, pool=5)
