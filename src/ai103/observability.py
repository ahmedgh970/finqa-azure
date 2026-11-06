"""Traces OpenTelemetry : export local en JSON et, sur demande, vers Application Insights."""

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import IO, Any

from azure.core.credentials import AccessToken, TokenCredential
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

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


def add_exporter(exporter: SpanExporter) -> tuple[TracerProvider, bool]:
    """Ajoute un export par lots au fournisseur global, créé au besoin.

    Renvoie le fournisseur et s'il vient d'être créé : l'appelant le ferme alors lui-même. Un
    fournisseur existant (export fichier) garde ses exports : les spans partent aux deux endroits.
    """
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        current.add_span_processor(BatchSpanProcessor(exporter))
        return current, False
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider, True


def application_insights_connection_string(project: Any) -> str:
    """Chaîne de connexion injectée par la plateforme (hosted agent), sinon celle du projet."""
    injected = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if injected:
        return injected
    return str(project.telemetry.get_application_insights_connection_string())


def azure_monitor_exporter(connection_string: str) -> SpanExporter:
    exporter: SpanExporter = AzureMonitorTraceExporter.from_connection_string(connection_string)
    return exporter


def trace_headers() -> dict[str, str]:
    """En-tête W3C `traceparent` du span courant, à joindre aux appels de service.

    Un service qui le lit rattache ses propres spans à notre trace : une seule trace de bout en
    bout au lieu de deux traces sans lien.
    """
    headers: dict[str, str] = {}
    propagate.inject(headers)
    return headers
