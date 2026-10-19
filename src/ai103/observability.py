"""Traces OpenTelemetry : export local en JSON, prêt à basculer vers Azure Monitor."""

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import IO, Any

from azure.core.credentials import AccessToken, TokenCredential
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

SERVICE_NAME = "finance-rag"
tracer = trace.get_tracer("ai103.rag")


def _json_line(span: ReadableSpan) -> str:
    return str(span.to_json(indent=None)) + "\n"


def tracer_provider(exporter: SpanExporter) -> TracerProvider:
    """Fournisseur de traces ; seul l'exporteur change pour viser Application Insights."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def file_exporter(stream: IO[str]) -> SpanExporter:
    """Un span par ligne JSON dans le fichier ouvert par l'appelant."""
    return ConsoleSpanExporter(service_name=SERVICE_NAME, out=stream, formatter=_json_line)


@contextmanager
def stage(
    name: str, timings: MutableMapping[str, float], **attributes: Any
) -> Iterator[trace.Span]:
    """Span OpenTelemetry et durée en secondes dans `timings[<name>_seconds]`."""
    started = perf_counter()
    try:
        with tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
    finally:
        timings[f"{name}_seconds"] = perf_counter() - started


class TracedCredential:
    """Enveloppe une identité Azure pour chronométrer chaque obtention de jeton Entra."""

    def __init__(self, credential: TokenCredential) -> None:
        self.credential = credential
        self.calls = 0
        self.seconds = 0.0

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        started = perf_counter()
        with tracer.start_as_current_span("credential.token", attributes={"scopes": scopes}):
            token = self.credential.get_token(*scopes, **kwargs)
        self.calls += 1
        self.seconds += perf_counter() - started
        return token


def configure_file_tracing(path: Path) -> tuple[TracerProvider, IO[str]]:
    """Active les traces globales vers `path` ; l'appelant ferme le fichier et le fournisseur."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8")
    provider = tracer_provider(file_exporter(stream))
    trace.set_tracer_provider(provider)
    return provider, stream
