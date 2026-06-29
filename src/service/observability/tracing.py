from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_SERVICE = "llm-service-kernel"


def setup_telemetry() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError:
        print("[otel] opentelemetry-exporter-otlp-proto-grpc not installed — tracing disabled", flush=True)
        return

    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", _SERVICE)})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    print(f"[otel] tracing enabled → {endpoint}", flush=True)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_SERVICE)
