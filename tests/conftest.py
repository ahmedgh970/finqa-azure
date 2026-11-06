import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai103.observability import tracer_provider

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="session", autouse=True)
def tracing():
    # Le fournisseur global ne peut être fixé qu'une fois par processus.
    trace.set_tracer_provider(tracer_provider(EXPORTER))


@pytest.fixture
def spans():
    EXPORTER.clear()
    return EXPORTER
