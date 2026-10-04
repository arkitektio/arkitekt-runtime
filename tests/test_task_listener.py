"""A task listener hears what happens to each task an agent runs.

A host that embeds an app (a control program with its own panel) shows who called
what and how it went. It is told through one typed event per step; the journal
stays the agent's own.
"""

import asyncio
import threading
from collections.abc import AsyncIterator

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_spec.declare.agents.connection import TaskEvent, TaskEventKind
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.task import Task

from .memory_transport import MemoryAgentTransport


@pytest.fixture()
def heard() -> list[TaskEvent]:
    return []


@pytest.fixture()
def transport() -> MemoryAgentTransport:
    return MemoryAgentTransport()


@pytest.fixture()
def agent(transport: MemoryAgentTransport, heard: list[TaskEvent]) -> BaseAgent:
    async def listener(event: TaskEvent) -> None:
        heard.append(event)

    agent = BaseAgent(
        name="listener-test",
        transport=transport,
        app_registry=AppRegistry(),
        task_listener=listener,
    )
    transport.set_transport_host(agent)
    return agent


async def _run_loop(agent: BaseAgent) -> AsyncIterator[None]:
    async for message in agent.transport.areceive():
        await agent.process(message)
        yield


async def _pump(agent: BaseAgent, count: int = 1) -> None:
    loop = _run_loop(agent)
    for _ in range(count):
        await asyncio.wait_for(loop.__anext__(), timeout=2.0)


def _assign(task: str, interface: str, **args: object) -> messages.Assign:
    return messages.Assign(
        task=task,
        interface=interface,
        args=args,
        implementation="impl-1",
        action="action-1",
        reference="ref-1",
        user="user-1",
        org="org-1",
    )


async def _until_ended(heard: list[TaskEvent]) -> None:
    ended = (TaskEventKind.DONE, TaskEventKind.FAILED, TaskEventKind.CANCELLED)

    async def wait() -> None:
        while not any(event.kind in ended for event in heard):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5.0)


@pytest.mark.asyncio
async def test_it_hears_the_arguments_the_progress_the_result_and_the_end(
    agent: BaseAgent, transport: MemoryAgentTransport, heard: list[TaskEvent]
) -> None:
    def move(x: int, task: Task) -> int:
        """Move."""
        task.progress(50, "halfway")
        return x

    agent.app_registry.register(move)
    agent.collect_from_registry()

    transport.feed(_assign("task-1", "move", x=3))
    await _pump(agent)
    await _until_ended(heard)

    # The first progress is the actor's own: the task was queued for running.
    assert [event.kind for event in heard] == [
        TaskEventKind.ASSIGNED,
        TaskEventKind.PROGRESS,
        TaskEventKind.PROGRESS,
        TaskEventKind.YIELDED,
        TaskEventKind.DONE,
    ]
    assert {event.task_id for event in heard} == {"task-1"}
    assert {event.action for event in heard} == {"move"}
    assert heard[0].arguments == {"x": 3}
    assert (heard[2].progress, heard[2].message) == (50, "halfway")


@pytest.mark.asyncio
async def test_a_failure_carries_its_error(
    agent: BaseAgent, transport: MemoryAgentTransport, heard: list[TaskEvent]
) -> None:
    def move(x: int) -> int:
        """Move."""
        raise ValueError("out of range")

    agent.app_registry.register(move)
    agent.collect_from_registry()

    transport.feed(_assign("task-1", "move", x=3))
    await _pump(agent)
    await _until_ended(heard)

    assert heard[-1].kind is TaskEventKind.FAILED
    assert "out of range" in (heard[-1].error or "")


@pytest.mark.asyncio
async def test_a_cancelled_sync_action_is_heard_as_cancelled_not_failed(
    agent: BaseAgent, transport: MemoryAgentTransport, heard: list[TaskEvent]
) -> None:
    """The cancellation reaches a sync body where it asks: ``task.check_cancelled()``."""
    started = threading.Event()
    stopped = threading.Event()

    def scan(x: int, task: Task) -> int:
        """Scan until told to stop."""
        started.set()
        try:
            while True:
                task.check_cancelled()
        finally:
            stopped.set()

    agent.app_registry.register(scan)
    agent.collect_from_registry()

    transport.feed(_assign("task-1", "scan", x=1))
    await _pump(agent)
    await asyncio.wait_for(asyncio.to_thread(started.wait, 2.0), timeout=3.0)

    transport.feed(messages.Cancel(task="task-1"))
    await _pump(agent)
    await _until_ended(heard)

    assert heard[-1].kind is TaskEventKind.CANCELLED
    assert await asyncio.to_thread(stopped.wait, 2.0)


@pytest.mark.asyncio
async def test_a_listener_that_raises_does_not_harm_the_task(
    transport: MemoryAgentTransport,
) -> None:
    async def listener(event: TaskEvent) -> None:
        raise RuntimeError("the panel is gone")

    agent = BaseAgent(
        name="listener-test",
        transport=transport,
        app_registry=AppRegistry(),
        task_listener=listener,
    )
    transport.set_transport_host(agent)

    def move(x: int) -> int:
        """Move."""
        return x

    agent.app_registry.register(move)
    agent.collect_from_registry()

    transport.feed(_assign("task-1", "move", x=3))
    await _pump(agent)

    async def wait() -> None:
        while not transport.of_type(messages.Completed):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5.0)
    assert transport.of_type(messages.Yield)[0].returns == {"return0": 3}
