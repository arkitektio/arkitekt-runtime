"""A call whose task ends LOST raises ``AgentLost``, not a failure.

``_astream_raw`` is where every postman's events turn into a result or an exception; the
agent-socket postman is covered in rekuest (``test_agent_postman``). Here a bare postman
stands in for the GraphQL one, whose events carry the LOST details as ``value``.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pytest

from arkitekt_runtime.calls import _astream_raw
from arkitekt_runtime.types import TaskEventKind
from arkitekt_spec.declare.errors import AgentLost


@dataclass
class Event:
    kind: TaskEventKind
    returns: dict | None = None
    message: str | None = None
    value: dict | None = None


class OnePostman:
    def __init__(self, *events: Event) -> None:
        self.events = events

    async def aassign(self, **_: Any) -> AsyncGenerator[Event, None]:  # noqa: ANN401
        for event in self.events:
            yield event


async def _drain(postman: OnePostman) -> list:
    return [returns async for returns in _astream_raw(postman)]


async def test_a_lost_task_raises_agent_lost_with_what_is_known() -> None:
    postman = OnePostman(
        Event(
            TaskEventKind.LOST,
            message="Its agent died while it ran.",
            value={"started": True, "last_progress": 40, "effects": "NONE", "reason": "Its agent died while it ran."},
        )
    )

    with pytest.raises(AgentLost, match="Its agent died") as lost:
        await _drain(postman)

    assert (lost.value.started, lost.value.last_progress, lost.value.effects) == (True, 40, "NONE")


async def test_a_task_that_never_started_says_so() -> None:
    postman = OnePostman(Event(TaskEventKind.LOST, value={"started": False, "reason": "never picked up"}))

    with pytest.raises(AgentLost) as lost:
        await _drain(postman)

    assert lost.value.started is False and lost.value.effects == "UNKNOWN"


async def test_what_arrives_before_the_loss_is_still_yielded_first() -> None:
    postman = OnePostman(Event(TaskEventKind.YIELD, returns={"x": 1}), Event(TaskEventKind.LOST, value={}))
    received = []

    with pytest.raises(AgentLost):
        async for returns in _astream_raw(postman):
            received.append(returns)

    assert received == [{"x": 1}]
