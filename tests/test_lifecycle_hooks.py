"""``@app.lifecycle``: one async generator as a startup hook and its shutdown."""

import asyncio
from collections.abc import AsyncIterator, Generator

import pytest
from arkitekt_spec.declare.agents.hooks.errors import StartupHookError

from arkitekt_runtime.agents.base import BaseAgent


@pytest.fixture(autouse=True)
def _isolated_hooks(mock_agent: BaseAgent) -> Generator[None, None, None]:
    registry = mock_agent.app_registry.hooks_registry
    registry.reset()
    yield
    registry.reset()


@pytest.mark.asyncio
async def test_what_follows_the_yield_runs_in_the_entering_task_after_the_shutdown_hooks(
    mock_agent: BaseAgent,
) -> None:
    registry = mock_agent.app_registry
    events: list[str] = []
    tasks: list[asyncio.Task | None] = []

    @registry.context
    class Connection:
        pass

    @registry.lifecycle
    async def first() -> AsyncIterator[Connection]:
        tasks.append(asyncio.current_task())
        events.append("enter first")
        try:
            yield Connection()
        finally:
            tasks.append(asyncio.current_task())
            events.append("exit first")

    @registry.lifecycle
    async def second() -> AsyncIterator[None]:
        events.append("enter second")
        yield
        events.append("exit second")

    @registry.shutdown
    async def shutdown(connection: Connection) -> None:
        events.append("shutdown")

    mock_agent.collect_from_registry()
    answer = await mock_agent.arun_startup_hooks()
    assert list(answer.contexts.values())[0].__class__ is Connection
    mock_agent.contexts.update(answer.contexts)
    mock_agent._ran_startup_hooks = True
    await mock_agent.arun_shutdown_hooks()

    assert events == ["enter first", "enter second", "shutdown", "exit second", "exit first"]
    assert tasks[0] is tasks[1], "left in the task that entered it"


@pytest.mark.asyncio
async def test_a_start_that_fails_leaves_what_was_entered(mock_agent: BaseAgent) -> None:
    registry = mock_agent.app_registry
    events: list[str] = []

    @registry.lifecycle
    async def entered() -> AsyncIterator[None]:
        events.append("enter")
        yield
        events.append("exit")

    @registry.startup
    async def fails():  # noqa: ANN202
        raise RuntimeError("no hardware")

    mock_agent.collect_from_registry()
    with pytest.raises(StartupHookError):
        await mock_agent.aactivate()

    assert events == ["enter", "exit"]
