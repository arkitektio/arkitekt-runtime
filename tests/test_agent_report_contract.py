"""The agent-report contract (rekuest's ``docs/design/journal.md``), as the runtime keeps it.

``fixtures/agent_wire.json`` is the server's canonical file, copied byte for byte: every
frame in it has to parse with the message models and dump back to itself. The rest checks
what the runtime numbers: ``pos`` per session, ``task_step`` per task (child calls
included), effects, shelving, and probes that are never numbered.
"""

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Annotated, Any

import janus
import pytest
from pydantic import Field, TypeAdapter

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.agents.journal import Journal, TaskGate
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.task import Task

from .agent_helpers import run_assignment
from arkitekt_runtime.local import MemoryAgentTransport

WIRE = json.loads((Path(__file__).parent / "fixtures" / "agent_wire.json").read_text())

FROM_AGENT: TypeAdapter[messages.FromAgentMessage] = TypeAdapter(
    Annotated[messages.FromAgentMessage, Field(discriminator="type")]
)
TO_AGENT: TypeAdapter[messages.ToAgentMessage] = TypeAdapter(
    Annotated[messages.ToAgentMessage, Field(discriminator="type")]
)


def _cases(*groups: str) -> list[Any]:
    return [
        pytest.param(case["frame"], id=f"{group}:{case['name']}")
        for group in groups
        for case in WIRE[group]
    ]


# ---------------------------------------------------------------- fixture --


@pytest.mark.parametrize("frame", _cases("numbered_from_agent", "unnumbered_from_agent"))
def test_every_frame_from_the_agent_round_trips(frame: dict[str, Any]) -> None:
    message = FROM_AGENT.validate_python(frame)
    assert message.model_dump(mode="json", by_alias=True, exclude_unset=True) == frame


@pytest.mark.parametrize("frame", _cases("numbered_from_agent"))
def test_numbered_frames_go_out_exactly_as_the_fixture(frame: dict[str, Any]) -> None:
    """What the agent actually sends: every field set, nothing unset on the wire."""
    message = FROM_AGENT.validate_python(frame)
    assert isinstance(message, messages.JournaledMessage)
    assert message.model_dump(mode="json", by_alias=True, exclude_none=True) == frame
    assert json.loads(message.model_dump_json(exclude_none=True)) == frame


@pytest.mark.parametrize("frame", _cases("to_agent"))
def test_every_frame_to_the_agent_round_trips(frame: dict[str, Any]) -> None:
    message = TO_AGENT.validate_python(frame)
    assert message.model_dump(mode="json", by_alias=True, exclude_none=True) == frame


def test_the_fixture_steps_are_what_the_journal_numbers() -> None:
    """Replaying the fixture's numbered frames through a journal gives the same
    ``pos`` and ``task_step`` (task 42's step 8 is its child call)."""
    journal = Journal()
    for task in ("42", "43", "44", "45", "46", "47"):
        journal.begin_task(task)
    for case in WIRE["numbered_from_agent"]:
        frame = dict(case["frame"])
        expected = frame.pop("task_step", None)
        for key in ("pos", "journal_session", "agent_ts"):
            frame.pop(key)
        if case["name"] == "state_patch":
            assert journal.take_step("42") == 8  # the child call of unnumbered_from_agent
        if case["name"] in ("failed", "critical", "cancelled", "interrupted", "paused"):
            # Tasks 43..47 started (step 1) before what the fixture shows of them.
            journal.append(messages.Started(task=frame["task"]))
        entry = journal.append(FROM_AGENT.validate_python(frame))
        assert entry is not None and entry.step == expected, case["name"]


# ---------------------------------------------------------------- journal --


def _journal(*tasks: str) -> Journal:
    journal = Journal()
    journal.append(messages.SessionInit(session_id="s", states={}))
    for task in tasks:
        journal.begin_task(task)
    return journal


def test_a_task_is_numbered_per_task_and_the_assign_takes_no_step() -> None:
    journal = _journal("t", "u")
    assign = messages.Assign(
        interface="i", task="t", args={}, user="u", org="o", action="a", implementation="m", reference="r"
    )
    assert journal.record_assign(assign, "i").step is None  # type: ignore[union-attr]
    started = journal.append(messages.Started(task="t"))
    other = journal.append(messages.Started(task="u"))
    lock = journal.append(messages.Lock(key="k", task="t"))
    assert journal.take_step("t") == 3  # a child call
    effect = journal.append(
        messages.Effect(task="t", effect=messages.EffectKind.NOW, value=1.0)
    )
    completed = journal.append(messages.Completed(task="t"))
    unlock = journal.append(messages.Unlock(key="k", task="t"))
    unshelve = journal.append(messages.Unshelve(ref="d", drawer="d"))
    assert [e.step for e in (started, other, lock, effect, completed, unlock)] == [  # type: ignore[union-attr]
        1,
        1,
        2,
        4,
        5,
        6,
    ], "gapless per task, child calls included; the UNLOCK after the end is the holder's"
    assert unlock.task_id == "t"  # type: ignore[union-attr]
    assert unshelve.step is None and unshelve.pos == 9  # type: ignore[union-attr]


def test_an_unlock_without_its_holder_takes_the_holders_step() -> None:
    journal = _journal("t")
    journal.append(messages.Lock(key="k", task="t"))
    unlock = journal.append(messages.Unlock(key="k"))
    assert (unlock.task_id, unlock.step) == ("t", 2)  # type: ignore[union-attr]


def test_a_probes_reports_are_never_numbered_but_its_patches_are() -> None:
    journal = _journal("p-abc")
    assert journal.append(messages.Log(task="p-abc", message="probing")) is None
    assert journal.append(messages.Completed(task="p-abc")) is None
    assert journal.take_step("p-abc") is None
    probe = messages.Assign(
        interface="i", task="p-abc", args={}, user="u", org="o", action="a", implementation="m", probe=True
    )
    assert journal.record_assign(probe, "i") is None
    assert journal.watermark().pos == 1  # type: ignore[union-attr]
    patch = journal.append(
        messages.StatePatch(
            session_id="s",
            global_rev=1,
            state_name="Camera",
            ts=0.0,
            op="replace",
            path="/exposure",
            value=2,
            old_value=None,
            task_id="p-abc",
        )
    )
    assert patch is not None and (patch.pos, patch.step, patch.task_id) == (2, None, "p-abc")


def test_a_report_about_a_task_this_process_never_ran_has_no_step() -> None:
    """E.g. answering an inquiry about a predecessor's task: numbered, but no step."""
    journal = _journal()
    entry = journal.append(messages.Critical(task="old", error="not running here"))
    assert entry is not None and (entry.pos, entry.step) == (2, None)
    assert journal.take_step("old") is None


def test_no_step_is_taken_before_the_session() -> None:
    assert Journal().take_step("t") is None


def test_a_task_begun_before_the_session_keeps_its_steps() -> None:
    """Steps are the task's, not the session's: the baseline does not reset them."""
    journal = Journal()
    journal.begin_task("t")
    journal.append(messages.SessionInit(session_id="s", states={}))
    assert journal.append(messages.Log(task="t", message="a")).step == 1  # type: ignore[union-attr]


# ------------------------------------------------------------------ agent --


def _agent(registry: AppRegistry | None = None) -> tuple[BaseAgent, MemoryAgentTransport]:
    transport = MemoryAgentTransport()
    agent = BaseAgent(transport=transport, app_registry=registry or AppRegistry())
    agent.collect_from_registry()
    return agent, transport


def _assign(interface: str, task: str = "42", probe: bool = False) -> messages.Assign:
    return messages.Assign(
        task=task,
        interface=interface,
        args={},
        implementation="impl-1",
        action="action-1",
        reference="ref-1",
        user="user-1",
        org="org-1",
        probe=probe,
    )


async def takes_effects(task: Task) -> str:
    """Takes the clock, randomness and a deadline"""
    now = await task.anow()
    drawn = await task.arandom(8)
    await task.asleep(0.01)
    assert isinstance(now, float) and len(drawn) == 16
    return drawn


def takes_effects_in_a_thread(task: Task) -> str:
    """Takes the clock, randomness and a deadline, from a worker thread"""
    task.now()
    drawn = task.random(4)
    task.sleep(0.01)
    return drawn


@pytest.mark.parametrize("func", [takes_effects, takes_effects_in_a_thread])
async def test_effects_are_recorded_as_numbered_task_steps(func: Any) -> None:  # noqa: ANN401
    registry = AppRegistry()
    registry.register(func)
    agent, transport = _agent(registry)
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))

    returns = await run_assignment(agent, _assign(func.__name__))

    effects = transport.of_type(messages.Effect)
    assert [e.effect for e in effects] == ["NOW", "RANDOM", "SLEEP"]
    # What a replay matches them by: the kind and its occurrence in the task.
    assert [e.key for e in effects] == ["NOW:1", "RANDOM:1", "SLEEP:1"]
    now, drawn, sleep = effects
    assert isinstance(now.value, float) and abs(now.value - time.time()) < 60
    assert drawn.value == returns["return0"]
    assert isinstance(sleep.value, float) and sleep.value >= now.value
    task_frames = [m for m in transport.sent if getattr(m, "task", None) == "42"]
    steps = [m.task_step for m in task_frames]  # type: ignore[union-attr]
    assert steps == list(range(1, len(steps) + 1)), f"gapless per task: {task_frames}"
    assert all(m.pos is not None and m.journal_session == "s" for m in task_frames)  # type: ignore[union-attr]
    wire = json.loads(sleep.model_dump_json(exclude_none=True))
    assert set(wire) == {
        "type",
        "id",
        "seq",
        "task",
        "effect",
        "value",
        "key",
        "pos",
        "journal_session",
        "agent_ts",
        "task_step",
    }


async def test_a_task_reports_started_as_its_first_step() -> None:
    """Until STARTED arrives the server counts a running task as QUEUED, and takes an
    idempotent one whose agent died for one it already re-queued."""
    registry = AppRegistry()
    registry.register(takes_effects)
    agent, transport = _agent(registry)
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))

    await run_assignment(agent, _assign("takes_effects"))

    task_frames = [m for m in transport.sent if getattr(m, "task", None) == "42"]
    assert isinstance(task_frames[0], messages.Started), task_frames
    assert task_frames[0].task_step == 1
    assert len(transport.of_type(messages.Started)) == 1


async def test_a_probes_frames_are_never_numbered() -> None:
    registry = AppRegistry()
    registry.register(takes_effects)
    agent, transport = _agent(registry)
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))

    await run_assignment(agent, _assign("takes_effects", task="p-1", probe=True))

    probe_frames = [m for m in transport.sent if getattr(m, "task", None) == "p-1"]
    assert probe_frames, "the probe reported"
    assert all(m.pos is None and m.task_step is None for m in probe_frames)  # type: ignore[union-attr]
    assert agent.journal.watermark().pos == 1  # type: ignore[union-attr]


async def test_a_child_call_takes_the_next_step_in_order_with_its_reports() -> None:
    """The step is taken on the ordered path: a report queued before the call (still
    waiting behind a patch) gets the earlier step."""
    agent, transport = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    agent._task_gates["42"] = TaskGate()
    agent.journal.begin_task("42")

    blocked = asyncio.Event()
    real = agent._aprocess_queued

    async def blocking_first(item: Any) -> None:  # noqa: ANN401
        await blocked.wait()
        await real(item)

    agent._aprocess_queued = blocking_first  # type: ignore[method-assign]
    agent._event_queue = janus.Queue()
    agent._patch_processor_task = asyncio.create_task(agent.apatch_event_loop())
    try:
        await agent._adispatch(messages.Log(task="42", message="before the call"))
        reserving = asyncio.create_task(agent.areserve_call_step("42"))
        await asyncio.sleep(0.01)
        assert not reserving.done(), "the step waits its turn"
        blocked.set()
        assert await asyncio.wait_for(reserving, 1.0) == 2
        await agent._adispatch(messages.Log(task="42", message="after the call"))
        await asyncio.wait_for(agent._event_queue.async_q.join(), 1.0)
        assert [m.task_step for m in transport.of_type(messages.Log)] == [1, 3]
    finally:
        agent._patch_processor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent._patch_processor_task
        queue, agent._event_queue = agent._event_queue, None
        queue.close()


async def test_a_probe_takes_no_step_for_its_calls() -> None:
    agent, _ = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    assert await agent.areserve_call_step("p-123") is None


class Image:
    """A value only this agent can hold."""

    async def aget_label(self) -> str:
        return "an image"


async def test_shelving_is_one_numbered_frame_and_nothing_waits_for_a_reply() -> None:
    agent, transport = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    image = Image()

    resource_id = await agent.aput_on_shelve("@mikro/image", image)

    (shelve,) = transport.of_type(messages.Shelve)
    assert agent.shelve[resource_id] is image
    assert len(resource_id) == 32 and int(resource_id, 16) >= 0
    assert (shelve.ref, shelve.resource_id, shelve.label) == (resource_id,) * 2 + ("an image",)
    assert (shelve.identifier, shelve.task, shelve.pos, shelve.task_step) == (
        "@mikro/image",
        None,
        2,
        None,
    ), "shelved by no task"

    # Synchronously, from a task's code: the reference comes back at once.
    from arkitekt_runtime.task_scope import task_id_scope

    agent.journal.begin_task("42")
    with task_id_scope("42"):
        reference = agent.put_on_shelve("@mikro/image", "raw")
    await asyncio.sleep(0)
    await asyncio.gather(*agent._dispatch_tasks)
    assert reference == {"__identifier": "@mikro/image", "object": reference["object"]}
    second = transport.of_type(messages.Shelve)[1]
    assert (second.resource_id, second.task, second.task_step) == (reference["object"], "42", 1)

    # COLLECT names the drawer by its resource id: dropped, and recorded.
    await agent.process(messages.Collect(drawers=[resource_id]))
    assert resource_id not in agent.shelve
    (unshelve,) = transport.of_type(messages.Unshelve)
    assert (unshelve.ref, unshelve.drawer, unshelve.pos) == (resource_id, resource_id, 4)


async def test_an_unlock_names_the_task_that_held_the_lock() -> None:
    from arkitekt_runtime.agents.lock import TaskLock
    from arkitekt_spec.actions import LockDefinitionInput, LockImplementationInput

    agent, transport = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    agent.journal.begin_task("42")
    lock = TaskLock(
        agent,
        LockImplementationInput(
            key="stage", definition=LockDefinitionInput(key="stage", description="The stage")
        ),
    )
    await lock.acquire("42")
    await agent._adispatch(messages.Completed(task="42"))
    await lock.release()

    (unlock,) = transport.of_type(messages.Unlock)
    assert (unlock.task, unlock.task_step) == ("42", 3), "the holder's, after its end"


async def test_a_value_shelved_from_a_worker_thread_is_queued_in_order() -> None:
    agent, transport = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    agent._event_queue = janus.Queue()
    agent._patch_processor_task = asyncio.create_task(agent.apatch_event_loop())
    try:
        reference = await asyncio.to_thread(agent.put_on_shelve, "@mikro/image", "raw")
        await asyncio.wait_for(agent._event_queue.async_q.join(), 1.0)
        (shelve,) = transport.of_type(messages.Shelve)
        assert shelve.resource_id == reference["object"] and shelve.pos == 2
    finally:
        agent._patch_processor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent._patch_processor_task
        queue, agent._event_queue = agent._event_queue, None
        queue.close()


async def test_a_value_shelved_after_its_tasks_end_is_shelved_by_no_task() -> None:
    agent, transport = _agent()
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))
    agent.journal.begin_task("42")
    agent._task_gates["42"] = TaskGate()
    await agent._adispatch(messages.Completed(task="42"))
    agent.put_on_shelve("@mikro/image", "late", task="42")
    await asyncio.gather(*agent._dispatch_tasks)
    (shelve,) = transport.of_type(messages.Shelve)
    assert (shelve.task, shelve.task_step) == (None, None)


def test_the_client_knows_every_task_event_kind_the_server_records() -> None:
    """A task's history holds EFFECT events (rekuest server 4); a client that cannot parse
    the kind crashes on the first one it is sent."""
    from arkitekt_runtime.types import TaskEventKind

    assert TaskEventKind("EFFECT") is TaskEventKind.EFFECT
