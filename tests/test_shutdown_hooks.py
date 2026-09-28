"""No-Docker checks for the shutdown hooks the agent runs while tearing down.

These cover what ``BaseAgent.arun_shutdown_hooks`` guarantees: hooks are
collected from the app registry, run in the reverse of their registration order
with the live states and contexts injected by annotation, a failing hook does
not stop the remaining ones, and hooks only run for an agent that actually got
through its startup hooks — exactly once per start.
"""

from collections.abc import Generator

from arkitekt_runtime.agents.hooks.materialize import materialize_hook
import pytest

from arkitekt_spec.declare.app import AppRegistry
from arkitekt_runtime.agents.hooks.shutdown import (
    ThreadedShutdownHook,
    WrappedShutdownHook,
)
from arkitekt_runtime.agents.base import BaseAgent


# These contexts and states belong to a registry, as they would to an app.
# There is no process-wide one to fall into.
_REGISTRY = AppRegistry()


@_REGISTRY.context
class Connection:
    """A context holding something a shutdown hook has to close."""

    def __init__(self) -> None:
        self.closed = False


@_REGISTRY.state
class Counter:
    """A state a shutdown hook may want to read one last time."""

    count: int = 3


@pytest.fixture(autouse=True)
def _isolated_hooks(mock_agent: BaseAgent) -> Generator[None, None, None]:
    """The agent defaults to the global app registry, which other test modules also
    decorate into. Start (and leave) each test with an empty hooks registry."""
    registry = mock_agent.app_registry.hooks_registry
    registry.reset()
    yield
    registry.reset()


def test_register_shutdown_lands_in_the_registry(mock_agent: BaseAgent) -> None:
    def close_it() -> None:
        pass

    async def aclose_it() -> None:
        pass

    mock_agent.app_registry.shutdown(close_it)
    mock_agent.app_registry.shutdown(aclose_it)

    hooks = mock_agent.app_registry.hooks_registry.shutdown_hooks
    assert isinstance(materialize_hook(hooks["close_it"]), ThreadedShutdownHook)
    assert isinstance(materialize_hook(hooks["aclose_it"]), WrappedShutdownHook)


def test_app_registry_shutdown_decorator(mock_agent: BaseAgent) -> None:
    """The registry-bound decorator registers on that registry, under a given name."""
    registry = mock_agent.app_registry

    @registry.shutdown
    async def close_it() -> None:
        pass

    @registry.shutdown(name="renamed")
    async def close_that_too() -> None:
        pass

    hooks = registry.hooks_registry.shutdown_hooks
    assert isinstance(materialize_hook(hooks["close_it"]), WrappedShutdownHook)
    assert isinstance(materialize_hook(hooks["renamed"]), WrappedShutdownHook)


@pytest.mark.asyncio
async def test_shutdown_hooks_run_with_states_and_contexts(
    mock_agent: BaseAgent,
) -> None:
    """The live state and context objects are injected by annotation."""
    seen: list[object] = []

    async def close_connection(connection: Connection, counter: Counter) -> None:
        connection.closed = True
        seen.append(counter)

    # The state belongs to the module's registry; this agent's app takes it in.
    mock_agent.app_registry.merge(_REGISTRY)
    mock_agent.app_registry.shutdown(close_connection)

    agent = mock_agent
    connection, counter = Connection(), Counter()
    agent.contexts[_REGISTRY.structure_registry.context_for(Connection).name] = connection
    agent.states["Counter"] = counter

    agent.collect_from_registry()
    agent._ran_startup_hooks = True
    await agent.arun_shutdown_hooks()

    assert connection.closed, "The hook should have received the live context"
    assert seen == [counter], "The hook should have received the live state"


@pytest.mark.asyncio
async def test_threaded_shutdown_hook_runs_with_states_and_contexts(
    mock_agent: BaseAgent,
) -> None:
    """A sync hook runs in a thread, and gets the same injection as an async one."""
    seen: list[object] = []

    def close_connection(connection: Connection, counter: Counter) -> None:
        connection.closed = True
        seen.append(counter)

    # The state belongs to the module's registry; this agent's app takes it in.
    mock_agent.app_registry.merge(_REGISTRY)
    mock_agent.app_registry.shutdown(close_connection)

    agent = mock_agent
    connection, counter = Connection(), Counter()
    agent.contexts[_REGISTRY.structure_registry.context_for(Connection).name] = connection
    agent.states["Counter"] = counter

    agent.collect_from_registry()
    agent._ran_startup_hooks = True
    await agent.arun_shutdown_hooks()

    assert connection.closed, "The threaded hook should have received the live context"
    assert seen == [counter], "The threaded hook should have received the live state"


@pytest.mark.asyncio
async def test_shutdown_hooks_run_in_reverse_registration_order(
    mock_agent: BaseAgent,
) -> None:
    """Teardown unwinds in the reverse of the order things were set up."""
    calls: list[str] = []

    async def first() -> None:
        calls.append("first")

    async def second() -> None:
        calls.append("second")

    mock_agent.app_registry.shutdown(first)
    mock_agent.app_registry.shutdown(second)

    agent = mock_agent
    agent.collect_from_registry()
    agent._ran_startup_hooks = True
    await agent.arun_shutdown_hooks()

    assert calls == ["second", "first"]


@pytest.mark.asyncio
async def test_failing_shutdown_hook_does_not_stop_the_others(
    mock_agent: BaseAgent,
) -> None:
    """A hook that raises is logged; teardown carries on."""
    calls: list[str] = []

    async def survivor() -> None:
        calls.append("survivor")

    async def exploding() -> None:
        raise RuntimeError("nope")

    mock_agent.app_registry.shutdown(survivor)
    mock_agent.app_registry.shutdown(exploding)  # runs first (reverse order)

    agent = mock_agent
    agent.collect_from_registry()
    agent._ran_startup_hooks = True

    await agent.arun_shutdown_hooks()

    assert calls == ["survivor"], "The failing hook must not abort the remaining ones"


@pytest.mark.asyncio
async def test_shutdown_hooks_only_run_for_a_started_agent_and_only_once(
    mock_agent: BaseAgent,
) -> None:
    calls: list[str] = []

    async def close_it() -> None:
        calls.append("closed")

    mock_agent.app_registry.shutdown(close_it)

    agent = mock_agent
    agent.collect_from_registry()

    # Never started: teardown owes it nothing.
    await agent.arun_shutdown_hooks()
    assert calls == []

    agent._ran_startup_hooks = True
    await agent.arun_shutdown_hooks()
    await agent.arun_shutdown_hooks()
    assert calls == ["closed"], "Hooks must run exactly once per start"
