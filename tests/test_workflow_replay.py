"""A resumed workflow replays what it recorded, and only a workflow may call other actions.

An ``Assign`` with ``resume`` hands the task what its earlier run recorded, by key. The task's
``now()``/``random()``/``sleep()``/``record()``/``hold()`` return that instead of taking a new
value, and its reports continue after the earlier run's last step.
"""

import asyncio
import time
from typing import Any


from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.task import StateRef, Task

from .agent_helpers import run_assignment
from .memory_transport import MemoryAgentTransport


def _agent(*workflows: Any, plain: tuple = ()) -> tuple[BaseAgent, MemoryAgentTransport]:  # noqa: ANN401
    registry = AppRegistry()
    for workflow in workflows:
        registry.register_workflow(workflow)
    for action in plain:
        registry.register(action)
    transport = MemoryAgentTransport()
    agent = BaseAgent(transport=transport, app_registry=registry)
    agent.collect_from_registry()
    return agent, transport


def _assign(interface: str, resume: dict | None = None) -> messages.Assign:
    return messages.Assign(
        task="42", interface=interface, args={}, implementation="impl-1", action="action-1",
        reference="ref-1", user="user-1", org="org-1", resume=resume,
    )


async def _ended(agent: BaseAgent, message: messages.Assign) -> MemoryAgentTransport:
    """Feed an assignment that is expected to fail, and wait until it reports so."""
    transport: MemoryAgentTransport = agent.transport  # type: ignore[assignment]
    transport.feed(message)
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def until_ended() -> None:
        while not (transport.of_type(messages.Failed) or transport.of_type(messages.Critical)):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(until_ended(), timeout=5.0)
    return transport


async def _drive(agent: BaseAgent):  # noqa: ANN202
    async for message in agent.transport.areceive():
        await agent.process(message)
        yield


def stamped(task: Task) -> str:
    """Takes the clock, randomness and a recorded value."""
    return f"{task.now()}:{task.random(4)}:{task.record(lambda: 'fresh')}"


async def test_a_resumed_run_gets_its_recorded_values_back() -> None:
    agent, transport = _agent(stamped)
    resume = {"last_step": 5, "effects": [
        {"key": "NOW:1", "effect": "NOW", "value": 111.0},
        {"key": "RANDOM:1", "effect": "RANDOM", "value": "abcd"},
        {"key": "RECORD:1", "effect": "RECORD", "value": "old"},
    ]}

    returns = await run_assignment(agent, _assign("stamped", resume))

    assert returns["return0"] == "111.0:abcd:old"
    assert transport.of_type(messages.Effect) == [], "a replayed value is not recorded again"


async def test_what_the_earlier_run_did_not_record_is_taken_and_recorded() -> None:
    agent, transport = _agent(stamped)
    resume = {"last_step": 3, "effects": [{"key": "NOW:1", "effect": "NOW", "value": 111.0}]}

    returns = await run_assignment(agent, _assign("stamped", resume))

    now, drawn, recorded = returns["return0"].split(":")
    assert now == "111.0" and recorded == "fresh"
    assert [(e.key, e.value) for e in transport.of_type(messages.Effect)] == [("RANDOM:1", drawn), ("RECORD:1", "fresh")]


async def test_reports_continue_after_the_earlier_runs_last_step() -> None:
    agent, transport = _agent(stamped)
    await agent._adispatch(messages.SessionInit(session_id="s", states={}))  # frames are numbered in a session

    await run_assignment(agent, _assign("stamped", {"last_step": 7, "effects": []}))

    (started,) = transport.of_type(messages.Started)
    assert started.task_step == 8


async def test_a_run_that_takes_another_path_is_refused() -> None:
    agent, _ = _agent(stamped)
    # The earlier run recorded randomness where this one asks for the clock.
    resume = {"last_step": 1, "effects": [{"key": "NOW:1", "effect": "RANDOM", "value": "abcd"}]}

    transport = await _ended(agent, _assign("stamped", resume))

    ended = transport.of_type(messages.Failed) + transport.of_type(messages.Critical)
    assert "NOW" in ended[0].error and "RANDOM" in ended[0].error


async def test_a_resumed_sleep_waits_only_until_its_recorded_deadline() -> None:
    def sleeper(task: Task) -> float:
        """Sleeps a long time."""
        task.sleep(3600)
        return time.time()

    agent, _ = _agent(sleeper)
    past = time.time() - 1
    started = time.time()

    await run_assignment(agent, _assign("sleeper", {"last_step": 2, "effects": [{"key": "SLEEP:1", "effect": "SLEEP", "value": past}]}))

    assert time.time() - started < 2, "the deadline had passed: nothing left to wait"


async def test_record_takes_json_only() -> None:
    def records_an_object(task: Task) -> str:
        """Records something that is not JSON."""
        task.record(object)
        return "no"

    agent, _ = _agent(records_an_object)

    transport = await _ended(agent, _assign("records_an_object"))

    ended = transport.of_type(messages.Failed) + transport.of_type(messages.Critical)
    assert "JSON" in ended[0].error


async def test_a_plain_action_may_not_call_another() -> None:
    def plain(task: Task) -> str:
        """Tries to call another action."""
        task.call(None)  # type: ignore[arg-type]
        return "no"

    agent, _ = _agent(plain=(plain,))

    transport = await _ended(agent, _assign("plain"))

    ended = transport.of_type(messages.Failed) + transport.of_type(messages.Critical)
    assert "only a workflow" in ended[0].error


def held(task: Task) -> str:
    """Waits for a person."""
    task.hold("Check the well")
    return "went on"


async def test_a_hold_pauses_with_why_and_goes_on_when_resumed() -> None:
    agent, transport = _agent(held)
    transport.feed(_assign("held"))
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def paused() -> messages.Paused:
        while not transport.of_type(messages.Paused):
            await asyncio.sleep(0.01)
        return transport.of_type(messages.Paused)[0]

    pause = await asyncio.wait_for(paused(), timeout=5.0)
    assert pause.message == "Check the well"

    transport.feed(messages.Resume(task="42"))
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def done() -> None:
        while not transport.of_type(messages.Yield):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(done(), timeout=5.0)
    assert transport.of_type(messages.Yield)[0].returns == {"return0": "went on"}
    assert [(e.key, e.value) for e in transport.of_type(messages.Effect)] == [("HOLD:1", "resumed")]


async def test_a_hold_already_resumed_before_is_not_held_again() -> None:
    agent, transport = _agent(held)

    returns = await run_assignment(agent, _assign("held", {"last_step": 3, "effects": [{"key": "HOLD:1", "effect": "HOLD", "value": "resumed"}]}))

    assert returns["return0"] == "went on"
    assert transport.of_type(messages.Paused) == []


class Guardian:
    """Answers a guard's questions: the revision now, and whether it changed."""

    connected = True

    def __init__(self, changed: bool = False) -> None:
        self.changed = changed
        self.asked: list[dict] = []

    async def astate_revision(self, **kwargs: Any) -> messages.StateRevisionResponse:  # noqa: ANN401
        self.asked.append(kwargs)
        return messages.StateRevisionResponse(
            request="r", revision={"session": "s1", "global_rev": 4},
            changed=self.changed if kwargs.get("since") is not None else None,
            detail="'plate' changed at /barcode (by task 9) since the workflow last saw it.",
        )


def guarded(task: Task) -> str:
    """Works on the plate, guarded."""
    with task.guard(StateRef(dependency="handler", state="plate"), "barcode"):
        return "worked"


async def test_a_guard_records_the_revision_on_entering() -> None:
    agent, transport = _agent(guarded)
    guardian = Guardian()
    agent.use_caller(guardian)

    assert (await run_assignment(agent, _assign("guarded")))["return0"] == "worked"

    assert guardian.asked == [{"parent": "42", "dependency": "handler", "state": "plate", "paths": ("barcode",)}]
    (recorded,) = transport.of_type(messages.Effect)
    assert (recorded.effect, recorded.value) == ("RECORD", {"session": "s1", "global_rev": 4})


async def test_a_resumed_guard_carries_on_when_nothing_else_changed_the_state() -> None:
    agent, _ = _agent(guarded)
    guardian = Guardian(changed=False)
    agent.use_caller(guardian)
    seen = {"session": "s1", "global_rev": 2}

    returns = await run_assignment(agent, _assign("guarded", {"last_step": 2, "effects": [{"key": "RECORD:1", "effect": "RECORD", "value": seen}]}))

    assert returns["return0"] == "worked"
    assert guardian.asked[0]["since"] == seen


async def test_a_resumed_guard_stops_when_something_else_changed_the_state() -> None:
    agent, _ = _agent(guarded)
    agent.use_caller(Guardian(changed=True))

    transport = await _ended(agent, _assign("guarded", {"last_step": 2, "effects": [{"key": "RECORD:1", "effect": "RECORD", "value": {"session": "s1", "global_rev": 2}}]}))

    ended = transport.of_type(messages.Failed) + transport.of_type(messages.Critical)
    assert "/barcode" in ended[0].error and "task 9" in ended[0].error
