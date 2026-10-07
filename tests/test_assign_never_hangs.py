"""No-Docker checks that an assignment the agent receives can never go silent.

Each of these is a way a task used to stay non-terminal on the backend forever while the
agent looked perfectly healthy: the agent received the ``Assign`` and then — for a reason the
backend never learned about — simply never mentioned the task again.

They drive a real ``BaseAgent`` message loop over
:class:`arkitekt_runtime.local.MemoryAgentTransport`.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.types import TaskEventKind
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.errors import CriticalCallError
from arkitekt_runtime.calls import _astream_raw

from arkitekt_runtime.local import MemoryAgentTransport


# These states belong to a registry, as they would to an app. There is no
# process-wide one to fall into.
_REGISTRY = AppRegistry()


@_REGISTRY.state
@dataclass
class Stage:
    """A state some function depends on — and that the agent under test never initializes."""

    position: int = 0


@pytest.fixture()
def transport() -> MemoryAgentTransport:
    return MemoryAgentTransport()


@pytest.fixture()
def agent(transport: MemoryAgentTransport) -> BaseAgent:
    """A bare agent with its own registry, so nothing leaks between tests."""
    return BaseAgent(
        name="never-hangs-test", transport=transport, app_registry=AppRegistry()
    )


async def _run_loop(agent: BaseAgent) -> AsyncIterator[None]:
    async for message in agent.transport.areceive():
        await agent.process(message)
        yield


async def _pump(agent: BaseAgent, count: int) -> None:
    loop = _run_loop(agent)
    for _ in range(count):
        await asyncio.wait_for(loop.__anext__(), timeout=2.0)


async def _until(predicate, timeout: float = 2.0) -> None:
    waited = 0.0
    while not predicate() and waited < timeout:
        await asyncio.sleep(0.01)
        waited += 0.01
    assert predicate(), "condition was not reached in time"


def _assign(task: str, interface: str, **args: object) -> messages.Assign:
    return messages.Assign(
        task=task,
        interface=interface,
        args=args,
        implementation="impl-1",
        action="action-1",
        reference=f"ref-{task}",
        user="user-1",
        org="org-1",
    )


def _register(agent: BaseAgent, function) -> None:
    agent.app_registry.register(
        function,
    )
    agent.collect_from_registry()


@pytest.mark.asyncio
async def test_duplicate_assign_runs_once(
    agent: BaseAgent, transport: MemoryAgentTransport
) -> None:
    """Assigns are delivered at-least-once; the same task id must never execute twice.

    The backend redelivers an Assign it got no report for, and recovers frames a dying
    connection popped but never acked. Both can hand a *running* task to the agent again —
    which used to overwrite the running entry and start a second execution.
    """
    runs = 0
    release = asyncio.Event()

    async def aonce(x: int) -> int:
        """Count executions, and stay running until released."""
        nonlocal runs
        runs += 1
        await release.wait()
        return x

    _register(agent, aonce)

    transport.feed(_assign("task-1", "aonce", x=1))
    await _pump(agent, 1)
    await _until(lambda: runs == 1)

    transport.feed(_assign("task-1", "aonce", x=1))  # the redelivery
    await _pump(agent, 1)
    await asyncio.sleep(0.05)

    assert runs == 1, "a redelivered Assign must not start a second execution"
    # It must still *answer*: any report tells the backend's watchdog the task was picked up.
    logs = [m for m in transport.of_type(messages.Log) if m.task == "task-1"]
    assert logs, "the duplicate should be acknowledged with a report for the task"

    release.set()
    await _until(lambda: bool(transport.of_type(messages.Completed)))
    assert len(transport.of_type(messages.Completed)) == 1




@pytest.mark.asyncio
async def test_unknown_interface_is_reported_and_does_not_kill_the_agent(
    agent: BaseAgent, transport: MemoryAgentTransport
) -> None:
    """The Critical was always sent — and then the exception was re-raised into the message
    loop, tearing the whole agent down and orphaning every *other* task it was running."""
    transport.feed(_assign("task-3", "no-such-interface"))

    await _pump(agent, 1)  # must not raise

    criticals = [c for c in transport.of_type(messages.Critical) if c.task == "task-3"]
    assert criticals, "the backend must be told the assignment could not start"


@pytest.mark.asyncio
async def test_failure_outside_the_actors_own_error_handling_is_reported(
    agent: BaseAgent, transport: MemoryAgentTransport
) -> None:
    """``on_assign`` reports what it anticipates. Whatever else raises inside it (resolving
    the function's locals, the bound app, a lock on entry) used to be logged by the task's
    done-callback and nothing more — the backend waited forever."""

    # A function that needs a state this agent never initialized: resolving its locals
    # genuinely raises ``StateRequirementsNotMet`` — no patching involved.
    async def aneeds_state(x: int, stage: Stage) -> int:
        """Never reached."""
        return x

    agent.app_registry.merge(_REGISTRY)  # Stage is a state of this app too
    _register(agent, aneeds_state)

    transport.feed(_assign("task-4", "aneeds_state", x=1))
    await _pump(agent, 1)

    await _until(
        lambda: any(c.task == "task-4" for c in transport.of_type(messages.Critical))
    )
    critical = next(c for c in transport.of_type(messages.Critical) if c.task == "task-4")
    assert "State requirements not met" in critical.error


@dataclass
class _Event:
    """What the stream reads off a task event: its kind."""

    kind: TaskEventKind
    message: str | None = None


class _ScriptedPostman:
    """A postman whose ``aassign`` replays a fixed event script, then waits forever —
    exactly what the real one does once the backend has nothing more to say."""

    def __init__(self, *kinds: TaskEventKind) -> None:
        self.kinds = kinds

    async def aassign(self, **kwargs: object) -> AsyncIterator[_Event]:
        for kind in self.kinds:
            yield _Event(kind=kind)
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [TaskEventKind.CANCELLED, TaskEventKind.INTERRUPTED])
async def test_stream_ends_when_the_task_is_cancelled_or_interrupted(
    kind: TaskEventKind,
) -> None:
    """Someone else ending the task (the UI, an interrupt cascading down a tree) is terminal.
    The GraphQL caller only knew COMPLETED / FAILED / CRITICAL and hung on anything else."""

    async def consume() -> None:
        async for _ in _astream_raw(postman=_ScriptedPostman(TaskEventKind.PROGRESS, kind)):  # type: ignore[arg-type]
            pass

    with pytest.raises(CriticalCallError):
        await asyncio.wait_for(consume(), timeout=2.0)
