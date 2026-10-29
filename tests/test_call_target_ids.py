"""What a call names as its target reaches the postman as an id.

The translation happens once, at the boundary between the two halves of the call
surface: ``rekuest.invoke`` holds the fetched ``Action``/``Implementation`` and
reads ``.id`` off it, ``rekuest.calls`` only ever sees the id. Every other test
that drives these helpers leaves the target unset, so both sides of that
boundary were unguarded.
"""

from collections.abc import AsyncGenerator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from arkitekt_runtime import messages
from arkitekt_spec.actions import Execution
from arkitekt_runtime.types import TaskEventKind
from arkitekt_runtime.task import Task


class RecordingPostman:
    """Records the one ``aassign`` it is given, then completes the task."""

    connected = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def aassign(self, **kwargs: Any) -> AsyncGenerator[Any, None]:  # noqa: ANN401
        self.calls.append(kwargs)
        yield _Event(TaskEventKind.YIELD, returns=("ok",))
        yield _Event(TaskEventKind.COMPLETED)


class _Event:
    """As much of a task event as the stream helper reads."""

    def __init__(self, kind: TaskEventKind, returns: Sequence[Any] = ()) -> None:
        self.kind = kind
        self.returns = returns
        self.message = None


class _Target:
    """An ``Action`` or ``Implementation``, as far as the translation goes."""

    def __init__(self, id: str) -> None:
        self.id = id


# --------------------------------------------------------------------------- #
# The agnostic half: an id goes straight through
# --------------------------------------------------------------------------- #


async def _stream(postman: RecordingPostman, **kwargs: Any) -> list[Any]:  # noqa: ANN401
    from arkitekt_runtime.calls import _astream_raw

    return [item async for item in _astream_raw(postman, **kwargs)]


@pytest.mark.asyncio
async def test_an_action_id_goes_to_the_postman_unchanged() -> None:
    postman = RecordingPostman()

    assert await _stream(postman, action_id="action-1") == [("ok",)]

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == ("action-1", None)


@pytest.mark.asyncio
async def test_an_implementation_id_goes_to_the_postman_unchanged() -> None:
    postman = RecordingPostman()

    await _stream(postman, implementation_id="impl-1")

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == (None, "impl-1")


@pytest.mark.asyncio
async def test_a_call_with_no_target_sends_neither_id() -> None:
    """A dependency or method call names no target at all."""
    postman = RecordingPostman()

    await _stream(postman, method="a_method")

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == (None, None)
    assert call["method"] == "a_method"


# --------------------------------------------------------------------------- #
# The client half: a fetched model is reduced to its id, and only at the surface
# --------------------------------------------------------------------------- #
#
# The reduction moved. `rekuest.invoke` takes ids -- it is the agnostic half and may not
# name a fragment -- so whoever holds the model reads `.id` off it: `Rekuest.acall_raw`
# and `Task.acall_raw`. These test it where it now happens, on the Task surface, because
# that one needs no GraphQL client to stand up.


async def _iterate(postman: RecordingPostman, **kwargs: Any) -> list[Any]:  # noqa: ANN401
    from arkitekt_runtime.invoke import _aiterate_raw as aiterate_raw

    return [item async for item in aiterate_raw(postman=postman, **kwargs)]


class _Helper:
    """The least of an AssignmentHelper that a Task's call path reads."""

    def __init__(self, postman: RecordingPostman) -> None:
        self.agent = SimpleNamespace(caller_postman=postman)
        # Only `.task` is read from it, on its way to `_resolve_parent`.
        self.assignment = messages.Assign.model_construct(task="task-1")
        self.structure_registry = object()
        self.task = "task-1"
        # Calls are what only a workflow makes, and one naming its target declares dynamic calls.
        self.execution = Execution.WORKFLOW
        self.dynamic_calls = True


def _task(postman: RecordingPostman) -> Task:
    return Task(_Helper(postman))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_fetched_action_is_reduced_to_its_id() -> None:
    postman = RecordingPostman()

    result = [
        item
        async for item in _task(postman).aiterate_raw(action=_Target("action-1"))
    ]
    assert result == [("ok",)]

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == ("action-1", None)


@pytest.mark.asyncio
async def test_a_fetched_implementation_is_reduced_to_its_id() -> None:
    postman = RecordingPostman()

    async for _ in _task(postman).aiterate_raw(implementation=_Target("impl-1")):
        pass

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == (None, "impl-1")


@pytest.mark.asyncio
async def test_the_raw_engine_itself_only_ever_sees_ids() -> None:
    """The other side of the same boundary: no model reaches `rekuest.invoke`."""
    postman = RecordingPostman()

    await _iterate(postman, action_id="action-1", implementation_id=None)

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == ("action-1", None)


class PostmanBeforeHolds:
    """A postman from before prompts and policies: it names every parameter it takes."""

    connected = True

    async def aassign(
        self,
        *,
        args: dict[str, Any],
        capture: bool = False,
        reference: str | None = None,
        hooks: Any = None,  # noqa: ANN401
        action: str | None = None,
        implementation: str | None = None,
        parent: str | None = None,
        dependency: str | None = None,
        method: str | None = None,
        escalate_to_interrupt: bool = False,
        cancel_timeout: float | None = None,
        call_key: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        yield _Event(TaskEventKind.YIELD, returns=("ok",))
        yield _Event(TaskEventKind.COMPLETED)


@pytest.mark.asyncio
async def test_a_call_without_prompt_or_policy_still_reaches_an_older_postman() -> None:
    assert await _stream(PostmanBeforeHolds(), args={}, action_id="a1") == [("ok",)]


@pytest.mark.asyncio
async def test_a_prompt_and_a_policy_reach_the_postman_when_given() -> None:
    from arkitekt_runtime.types import TaskPolicy
    from arkitekt_spec.declare.task import AssignPrompt

    postman = RecordingPostman()
    prompt = AssignPrompt(title="Run it")
    await _stream(postman, args={}, action_id="a1", prompt=prompt, policy=TaskPolicy.AUTO)
    await _stream(postman, args={}, action_id="a1")

    assert (postman.calls[0]["prompt"], postman.calls[0]["policy"]) == (prompt, TaskPolicy.AUTO)
    assert "prompt" not in postman.calls[1] and "policy" not in postman.calls[1]


def test_an_assign_input_without_a_policy_dumps_as_it_did_before_policies() -> None:
    """A published rekuest spreads the dump into an assign that takes no ``policy``."""
    from arkitekt_runtime.types import AssignInput, TaskPolicy

    def assign_before_policies(*, args: dict, capture: bool, **rest: Any) -> dict:  # noqa: ANN401
        assert "policy" not in rest
        return rest

    assign_before_policies(**AssignInput(args={}, capture=False).model_dump())
    assert AssignInput(args={}, capture=False, policy=TaskPolicy.AUTO).model_dump()["policy"] == "AUTO"


class _Implementation:
    """An ``Implementation``: its own id, wrapping an action with no ports."""

    def __init__(self, id: str, action_id: str) -> None:
        self.id = id
        self.action = SimpleNamespace(id=action_id, args=(), returns=())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_chosen_implementation_goes_alone_never_with_its_action(stream: bool) -> None:
    """The server takes an action first and runs any free agent of it: naming both would
    drop the implementation chosen (a caller's own agent, found with ``callerOf``)."""
    postman = RecordingPostman()
    task = _task(postman)
    target = _Implementation("impl-1", action_id="action-1")

    if stream:
        async for _ in task.aiterate(target):
            pass
    else:
        await task.acall(target)

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == (None, "impl-1")


@pytest.mark.asyncio
async def test_an_action_is_still_named_when_no_implementation_is() -> None:
    postman = RecordingPostman()

    await _task(postman).acall(SimpleNamespace(id="action-1", args=(), returns=()))

    (call,) = postman.calls
    assert (call["action"], call["implementation"]) == ("action-1", None)
