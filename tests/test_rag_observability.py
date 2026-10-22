import json
from unittest.mock import Mock

import pytest
from azure.core.credentials import AccessToken
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai103.observability import TracedCredential, stage, tracer_provider
from ai103.rag.cli import report
from ai103.rag.generator import Generation, Usage
from ai103.rag.pipeline import RagPipeline
from ai103.rag.prompts import Citation, RagAnswer
from ai103.rag.run import run, summarize
from ai103.retriever.core import Chunk

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider():
    # Le fournisseur global ne peut être fixé qu'une fois par processus.
    trace.set_tracer_provider(tracer_provider(EXPORTER))


@pytest.fixture(autouse=True)
def clear():
    EXPORTER.clear()


def test_stage_records_duration_and_span_attributes():
    timings = {}
    with stage("search", timings, **{"ai103.k": 5}):
        pass
    (span,) = EXPORTER.get_finished_spans()
    assert span.name == "search"
    assert span.attributes["ai103.k"] == 5
    assert timings["search_seconds"] >= 0


def test_traced_credential_counts_token_requests():
    inner = Mock()
    inner.get_token.return_value = AccessToken("jwt", 9_999_999_999)
    credential = TracedCredential(inner)
    assert credential.get_token("https://search.azure.com/.default").token == "jwt"
    credential.get_token("https://cognitiveservices.azure.com/.default")
    assert credential.calls == 2
    assert [s.name for s in EXPORTER.get_finished_spans()] == ["credential.token"] * 2


def fake_pipeline(cost_tokens=5000, fail_on=None):
    retriever, generator = Mock(), Mock()
    retriever.retrieve.return_value = [
        Chunk("c1", "Capex (1,577) in 2018", "https://s/c/3M.pdf", 1)
    ]
    usage = Usage(cost_tokens, 0, 100, 20)
    answer = RagAnswer(
        answer="1,577 [1]",
        citations=[Citation(source=1, quote="Capex (1,577) in 2018")],
        insufficient_evidence=False,
    )

    def generate(instructions, message, schema):
        if fail_on and fail_on in message:
            raise RuntimeError("content_filter")
        return Generation(answer, usage, 0.5, "resp", "gpt-5.2")

    generator.generate.side_effect = generate
    return RagPipeline(retriever, lambda q: [0.1] * 1024, generator)


def test_pipeline_emits_nested_spans_with_genai_usage():
    fake_pipeline().answer("Capex?")
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    assert set(spans) == {"rag.answer", "filter", "embedding", "search", "generation"}
    root = spans["rag.answer"]
    assert all(spans[n].parent.span_id == root.context.span_id for n in ("search", "generation"))
    generation = spans["generation"].attributes
    assert generation["gen_ai.usage.input_tokens"] == 5000
    assert generation["gen_ai.usage.reasoning.output_tokens"] == 20
    assert root.attributes["ai103.status"] == "verified"


def question(i, text="Capex?"):
    return {"financebench_id": f"q{i}", "question": text}


def test_run_stops_at_cost_cap_and_logs_errors(tmp_path):
    pipeline = fake_pipeline(cost_tokens=100_000, fail_on="BLOCK")
    questions = [question(1), question(2, "BLOCK me"), question(3), question(4)]
    # 100 000 tokens d'entrée et 100 de sortie : 0,1764 USD par question réussie.
    summary = run(
        lambda question: report(pipeline.answer(question), {"index": "x"}),
        questions,
        tmp_path,
        {"index": "x"},
        max_cost_usd=0.3,
    )
    lines = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [record["question_id"] for record in lines] == ["q1", "q2", "q3"]
    assert lines[1]["error"] == "RuntimeError"
    assert summary["errors"] == 1
    assert summary["status"] == {"verified": 2}
    assert summary["cost_usd"] == pytest.approx(2 * (100_000 * 1.75 + 100 * 14) / 1e6)
    assert "Plafond" in summary["stopped"]


def test_summary_latency_percentiles():
    records = [
        {"status": "verified", "cost_usd": 0.01, "usage": None, "timings": {"total_seconds": t}}
        for t in (1.0, 2.0, 3.0, 10.0)
    ]
    latency = summarize(records)["latency_seconds"]["total_seconds"]
    assert latency["median"] == 2.5
    assert 3.0 < latency["p95"] <= 10.0
