"""The trace an assignment is part of (W3C Trace Context, through OpenTelemetry's API).

A server that traces says where in a trace a task was assigned (``Assign.traceparent``).
:func:`trace_scope` runs the assignment there: what its code asks of others (a rath client's
request, a child it assigns) then carries the trace on, and an app that set OpenTelemetry up
records the assignment as a span of its own.

Only the API is used: an app that set nothing up records nothing, and passes the trace on
exactly as it came. Setting it up is the app's choice, the standard way (``OTEL_*``, or an SDK
configured in code).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from opentelemetry import context, propagate, trace

TRACEPARENT = "traceparent"


class Traced(Protocol):
    """What :func:`trace_scope` reads of a task."""

    @property
    def id(self) -> str:
        """The task id."""
        ...

    @property
    def traceparent(self) -> str | None:
        """Where in a trace the task was assigned."""
        ...


@contextmanager
def trace_scope(task: Traced) -> Iterator[None]:
    """Run the block inside the task's trace, as a span of this agent."""
    parent = propagate.extract({TRACEPARENT: task.traceparent}) if task.traceparent else None
    entered = context.attach(parent) if parent is not None else None
    try:
        with trace.get_tracer("arkitekt_runtime").start_as_current_span("task", attributes={"rekuest.task": task.id}):
            yield
    finally:
        if entered is not None:
            context.detach(entered)


def current_traceparent() -> str | None:
    """Where in a trace the running code is, for whoever is asked next; ``None`` in none."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier.get(TRACEPARENT)
