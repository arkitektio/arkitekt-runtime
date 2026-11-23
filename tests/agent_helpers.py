"""Driving an agent through one assignment, for tests that exercise actors end to end."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import MemoryAgentTransport


async def sent_until(
    transport: MemoryAgentTransport, holds: Callable[[], bool]
) -> None:
    """Wait until ``holds()`` is true of what ``transport`` sent, woken by each send."""
    watcher = transport.watch()
    try:
        while not holds():
            await watcher.get()
    finally:
        transport.unwatch(watcher)


async def _run_loop(agent: BaseAgent) -> AsyncIterator[None]:
    async for message in agent.transport.areceive():
        await agent.process(message)
        yield


async def run_assignment(agent: BaseAgent, message: messages.Assign) -> dict[str, Any]:
    """Feed one assignment and return what the actor yielded."""
    transport: MemoryAgentTransport = agent.transport  # type: ignore[assignment]
    transport.feed(message)
    loop = _run_loop(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def until_reported() -> None:
        await sent_until(
            transport,
            lambda: bool(
                transport.of_type(messages.Yield)
                or transport.of_type(messages.Failed)
                or transport.of_type(messages.Critical)
            ),
        )

    await asyncio.wait_for(until_reported(), timeout=5.0)
    failures = transport.of_type(messages.Failed) + transport.of_type(messages.Critical)
    assert not failures, f"the assignment failed: {failures}"
    returns = transport.of_type(messages.Yield)[0].returns
    assert returns is not None
    return returns
