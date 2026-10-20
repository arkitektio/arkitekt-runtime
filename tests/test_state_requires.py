"""A call's ``state_requires`` is tested by the agent, with the locks held, before anything runs."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.actors.base import STATE_REQUIRES_KEY
from arkitekt_runtime.actors.functional import FunctionalActor
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.calls import _astream_raw
from arkitekt_runtime.local import _local_transport, acall_local, local_agent
from arkitekt_runtime.types import AssignInput, StateRequirementInput, TaskEventKind
from arkitekt_spec.actions import DescriptorOperator
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.errors import STATE_PRECONDITION, ErrorCallError, StatePreconditionFailed
from arkitekt_spec.declare.task import StateRequirement

ran: list[str] = []


def requirement(path: str, operator: str, value: Any = None, state: str = "Stage") -> StateRequirement:  # noqa: ANN401
    return StateRequirement(state=state, path=path, operator=DescriptorOperator(operator), value=value)


def build_registry() -> AppRegistry:
    registry = AppRegistry()

    @registry.state(required_locks=["stage"])
    class Stage:
        """A stage."""

        busy: bool = False
        mode: str = "idle"

    @registry.startup
    def home() -> Stage:
        """Home."""
        return Stage()

    def move(stage: Stage, mode: str) -> str:
        """Move, holding the stage."""
        ran.append("move")
        stage.mode = mode
        return stage.mode

    def peek() -> str:
        """Run without the stage's lock of its own."""
        ran.append("peek")
        return "peeked"

    def guarded() -> str:
        """A workflow tested against the stage."""
        return "guarded"

    registry.register(move, locks=["stage"])
    registry.register(peek)
    registry.register_workflow(guarded)
    return registry


@pytest.fixture()
async def agent() -> AsyncIterator[BaseAgent]:
    ran.clear()
    agent = local_agent(build_registry())
    await agent.aconnect(timeout=5.0)
    yield agent
    await agent.atear_down()


async def test_a_requirement_that_holds_lets_the_action_run(agent: BaseAgent) -> None:
    result = await acall_local(agent, "move", "fast", state_requires=[requirement("/busy", "EQUALS", False)])
    assert result == "fast"
    assert ran == ["move"]


async def test_a_requirement_that_fails_runs_nothing_and_says_why(agent: BaseAgent) -> None:
    with pytest.raises(StatePreconditionFailed, match=r"Stage/mode EQUALS 'parked' \(actual 'idle'\)"):
        await acall_local(agent, "move", "fast", state_requires=[requirement("/mode", "EQUALS", "parked")])
    assert ran == []
    assert agent.states["Stage"].mode == "idle"


async def test_a_refusal_is_a_failed_call(agent: BaseAgent) -> None:
    """Code that already handles failed calls handles a refusal too."""
    with pytest.raises(ErrorCallError):
        await acall_local(agent, "move", "fast", state_requires=[requirement("/mode", "EQUALS", "parked")])


async def test_a_missing_path_or_state_fails(agent: BaseAgent) -> None:
    with pytest.raises(StatePreconditionFailed, match="missing"):
        await acall_local(agent, "peek", state_requires=[requirement("/gone", "EQUALS", 1)])
    with pytest.raises(StatePreconditionFailed, match="no state 'Laser'"):
        await acall_local(agent, "peek", state_requires=[requirement("/on", "EQUALS", True, state="Laser")])
    assert ran == []


async def test_the_check_sees_a_change_made_just_before(agent: BaseAgent) -> None:
    await acall_local(agent, "move", "parked")
    assert await acall_local(agent, "peek", state_requires=[requirement("/mode", "EQUALS", "parked")]) == "peeked"


async def test_the_state_lock_is_held_from_the_check_through_the_run(agent: BaseAgent) -> None:
    """``peek`` declares no lock; tested against ``Stage``, it holds the stage's lock."""
    lock = agent.locks["stage"]
    holders: list[str | None] = []
    original = FunctionalActor.aiterate_results

    def watching(self: FunctionalActor, **params: Any) -> Any:  # noqa: ANN401
        holders.append(lock.locking_task)
        return original(self, **params)

    FunctionalActor.aiterate_results = watching  # type: ignore[method-assign]
    try:
        await acall_local(agent, "peek")
        await acall_local(agent, "peek", state_requires=[requirement("/busy", "EQUALS", False)])
    finally:
        FunctionalActor.aiterate_results = original  # type: ignore[method-assign]
    assert holders[0] is None
    assert holders[1] is not None
    assert lock.locking_task is None


async def test_a_resumed_workflow_is_tested_again_unless_it_had_passed(agent: BaseAgent) -> None:
    transport = _local_transport(agent)
    watcher = transport.watch()
    failing = [requirement("/mode", "EQUALS", "parked")]
    passed = messages.RecordedEffect(key=STATE_REQUIRES_KEY, effect=messages.EffectKind.RECORD, value=True)

    async def ends(task: str, interface: str, resume: messages.Journal | None, requires: list) -> list:  # noqa: ANN401
        transport.feed(
            messages.Assign(
                task=task, interface=interface, args={}, implementation="i", action="a",
                reference=task, user="u", org="o", state_requires=requires, resume=resume,
            )
        )
        seen = []
        while True:
            message = await asyncio.wait_for(watcher.get(), 5)
            if getattr(message, "task", None) != task:
                continue
            seen.append(message)
            if isinstance(message, (messages.Completed, messages.Failed, messages.Critical)):
                return seen

    try:
        # Its steps (STARTED, the first progress) were taken before the locks: tested again.
        (*_, stepped) = await ends("stepped", "peek", messages.Journal(last_step=2), failing)
        assert isinstance(stepped, messages.Failed) and stepped.code == STATE_PRECONDITION
        (*_, fresh) = await ends("fresh", "peek", messages.Journal(), failing)
        assert isinstance(fresh, messages.Failed) and fresh.code == STATE_PRECONDITION
        # It had passed: not tested again.
        (*_, again) = await ends("again", "peek", messages.Journal(last_step=4, effects=[passed]), failing)
        assert isinstance(again, messages.Completed)

        # A workflow that passes records it; a plain action has no journal to keep it in.
        holding = [requirement("/mode", "EQUALS", "idle")]
        recorded = [
            m for m in await ends("guarded", "guarded", None, holding)
            if isinstance(m, messages.Effect) and m.key == STATE_REQUIRES_KEY
        ]
        assert len(recorded) == 1
        assert not [
            m for m in await ends("plain", "peek", None, holding) if isinstance(m, messages.Effect)
        ]
    finally:
        transport.unwatch(watcher)


async def test_an_actor_that_does_not_test_requirements_refuses_them(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FunctionalActor, "tests_state_requires", False)
    with pytest.raises(StatePreconditionFailed, match="does not test state requirements"):
        await acall_local(agent, "peek", state_requires=[requirement("/busy", "EQUALS", False)])
    assert ran == []
    assert await acall_local(agent, "peek") == "peeked"


async def test_the_agent_says_it_tests_them_on_register(agent: BaseAgent) -> None:
    handshake = await agent.aget_handshake_params()
    assert messages.FEATURE_STATE_REQUIRES in handshake.features


# ------------------------------------------------------------------ the caller side


class Recording:
    """A postman that records what it was asked and answers with one event."""

    def __init__(self, event: Any) -> None:  # noqa: ANN401
        self.event = event
        self.asked: dict[str, Any] = {}

    async def aassign(self, **kwargs: Any) -> AsyncIterator[Any]:  # noqa: ANN401
        self.asked = kwargs
        yield self.event


class Event:
    def __init__(self, kind: TaskEventKind, message: str | None = None, value: Any = None) -> None:  # noqa: ANN401
        self.kind = kind
        self.message = message
        self.value = value
        self.returns = None
        self.task = "t"


async def test_requirements_reach_the_postman_only_when_set() -> None:
    postman = Recording(Event(TaskEventKind.COMPLETED))
    [_ async for _ in _astream_raw(postman, args={})]  # type: ignore[arg-type]
    assert "state_requires" not in postman.asked

    wanted = [requirement("/busy", "EQUALS", False)]
    [_ async for _ in _astream_raw(postman, args={}, state_requires=wanted)]  # type: ignore[arg-type]
    assert postman.asked["state_requires"] == wanted


async def test_a_failed_event_with_the_precondition_code_raises_its_own_error() -> None:
    refused = Recording(Event(TaskEventKind.FAILED, "stage/busy", {"code": STATE_PRECONDITION}))
    with pytest.raises(StatePreconditionFailed, match="stage/busy"):
        [_ async for _ in _astream_raw(refused, args={})]  # type: ignore[arg-type]

    failed = Recording(Event(TaskEventKind.FAILED, "boom"))
    with pytest.raises(ErrorCallError) as raised:
        [_ async for _ in _astream_raw(failed, args={})]  # type: ignore[arg-type]
    assert not isinstance(raised.value, StatePreconditionFailed)


def test_an_assign_input_without_requirements_leaves_them_out() -> None:
    """A server from before them does not know the field."""
    bare = AssignInput(args={}, capture=False)
    assert "stateRequires" not in bare.model_dump(by_alias=True)
    assert "state_requires" not in bare.model_dump()

    wanted = AssignInput(
        args={},
        capture=False,
        state_requires=(StateRequirementInput(state="Stage", path="/busy", operator="EQUALS", value=False),),
    )
    assert wanted.model_dump(mode="json", by_alias=True)["stateRequires"] == [
        {"state": "Stage", "path": "/busy", "operator": "EQUALS", "value": False}
    ]
