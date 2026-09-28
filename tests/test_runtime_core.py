"""The core runs a declared app with no transport, client or task scope of its own."""

import asyncio
import ast
import contextlib
import contextvars
import subprocess
import sys
from types import SimpleNamespace
from collections.abc import Iterator
from typing import Any

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.actors.types import ActorContext
from arkitekt_runtime.agents.base import BaseAgent, NoCallerPostman
from arkitekt_spec.declare.agents.errors import NoCallerError
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.task import Task

from .agent_helpers import run_assignment
from .memory_transport import MemoryAgentTransport

HEAVY = {"rekuest", "rath", "websockets", "graphql", "fastapi", "fakts"}


def test_the_core_loads_no_transport_or_client() -> None:
    """Checked in a subprocess: the test session may have loaded anything."""
    code = (
        "import sys, arkitekt_runtime.agents.base, arkitekt_runtime.actors.actify, "
        "arkitekt_runtime.task, arkitekt_runtime.invoke; "
        "print(sorted({m.split('.')[0] for m in sys.modules}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert not set(ast.literal_eval(out.stdout.strip())) & HEAVY


def _assign(interface: str) -> messages.Assign:
    return messages.Assign(
        task=f"task-{interface}",
        interface=interface,
        args={},
        implementation="impl-1",
        action="action-1",
        reference="ref-1",
        user="user-1",
        org="org-1",
    )


CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("current", default=None)


@contextlib.contextmanager
def _scope(task: Any) -> Iterator[None]:  # noqa: ANN401
    token = CURRENT.set(task.id)
    try:
        yield
    finally:
        CURRENT.reset(token)


async def aread() -> str:
    """Reads the scope"""
    return CURRENT.get() or "none"


def read() -> str:
    """Reads the scope, across the thread hop"""
    return CURRENT.get() or "none"


@pytest.mark.parametrize("func", [aread, read])
async def test_the_agents_task_scopes_are_entered_around_an_assignment(func: Any) -> None:  # noqa: ANN401
    registry = AppRegistry()
    registry.register(func)
    agent = BaseAgent(transport=MemoryAgentTransport(), app_registry=registry, task_scopes=[_scope])
    agent.collect_from_registry()

    assert await run_assignment(agent, _assign(func.__name__)) == {"return0": f"task-{func.__name__}"}
    assert CURRENT.get() is None


async def test_without_a_scope_nothing_is_entered() -> None:
    registry = AppRegistry()
    registry.register(aread)
    agent = BaseAgent(transport=MemoryAgentTransport(), app_registry=registry)
    agent.collect_from_registry()

    assert await run_assignment(agent, _assign("aread")) == {"return0": "none"}


async def calls_another(task: Task) -> str:
    """Calls another action"""
    await task.acall(SimpleNamespace(id="action-2", args=[], returns=[]), {})  # type: ignore[attr-defined]
    return "never"


async def test_calling_another_action_without_a_caller_fails_at_once() -> None:
    """A runtime that cannot call other actions says so, instead of waiting forever."""
    agent = BaseAgent(transport=MemoryAgentTransport(), app_registry=AppRegistry())
    # Reading the caller never fails, so protocol checks (which read every member on
    # Python 3.11) still see an actor context; only calling through it fails.
    assert isinstance(agent.caller_postman, NoCallerPostman)
    assert not agent.caller_postman.connected
    assert isinstance(agent, ActorContext)
    with pytest.raises(NoCallerError):
        async for _ in agent.caller_postman.aassign(args={}):
            pass

    agent.app_registry.register(calls_another)
    agent.collect_from_registry()
    transport: MemoryAgentTransport = agent.transport  # type: ignore[assignment]
    with contextlib.suppress(Exception):
        await asyncio.wait_for(run_assignment(agent, _assign("calls_another")), timeout=5.0)
    (critical,) = transport.of_type(messages.Critical)
    assert "cannot call other actions" in critical.error
