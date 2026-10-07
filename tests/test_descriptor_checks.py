"""A structure that describes its objects is tested against the ports it crosses.

``Requires`` on an argument is tested after the object is expanded, ``Provides`` on a
return before it is shrunk. A key the structure does not compute is provenance: it is
taken on the producer's word.
"""

from collections.abc import AsyncIterator
from typing import Annotated

import pytest

from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import LocalCallError, acall_local, local_agent
from arkitekt_spec.actions import DescriptorOperator
from arkitekt_spec.declare.annotations import Provides, Requires
from arkitekt_spec.declare.app import AppRegistry

CHANNELS = "@test/n_channels"
KIND = "@test/value_kind"


class Picture:
    """Travels by id; the id is its channel count, so expanding needs no store."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.channels = int(id)


def describe_picture(picture: Picture) -> dict[str, int]:
    return {CHANNELS: picture.channels}


def _both(key: str, operator: str, value: object) -> tuple[Requires, Provides]:
    op = DescriptorOperator(operator)
    return Requires(key=key, operator=op, value=value), Provides(key=key, operator=op, value=value)


SingleChannel = Annotated[Picture, *_both(CHANNELS, "LTE", 1)]
Labels = Annotated[Picture, *_both(KIND, "EQUALS", "categorical")]

ran: list[str] = []


def passthrough(picture: Picture) -> Picture:
    """No constraints either way"""
    return picture


def needs_single(picture: SingleChannel) -> int:
    """Requires one channel"""
    ran.append("needs_single")
    return picture.channels


def promises_single(channels: int) -> SingleChannel:
    """Provides one channel, and returns whatever it is told to"""
    return Picture(str(channels))


def promises_labels(channels: int) -> Labels:
    """Provides a key the structure cannot compute"""
    return Picture(str(channels))


def promises_singles(channels: list[int]) -> list[SingleChannel]:
    """Provides one channel per item"""
    return [Picture(str(c)) for c in channels]


@pytest.fixture()
async def agent() -> AsyncIterator[BaseAgent]:
    registry = AppRegistry()

    @registry.structure("@test/picture", describe=describe_picture)
    async def expand_picture(id: str) -> Picture:
        return Picture(id)

    for function in (passthrough, needs_single, promises_single, promises_labels, promises_singles):
        registry.register(function)
    ran.clear()
    agent = local_agent(registry)
    await agent.aconnect(timeout=5.0)
    yield agent
    await agent.atear_down()


async def test_a_return_that_keeps_its_promise_goes_out(agent: BaseAgent) -> None:
    result = await acall_local(agent, "promises_single", 1)
    assert result.channels == 1


async def test_a_return_that_breaks_its_promise_is_refused_before_shrinking(agent: BaseAgent) -> None:
    with pytest.raises(LocalCallError, match=r"not what the port provides.*@test/n_channels LTE 1 \(actual 3\)"):
        await acall_local(agent, "promises_single", 3)


async def test_an_argument_that_breaks_the_requirement_never_reaches_the_body(agent: BaseAgent) -> None:
    with pytest.raises(LocalCallError, match=r"not what the port requires.*@test/n_channels LTE 1 \(actual 2\)"):
        await acall_local(agent, "needs_single", Picture("2"))
    assert ran == []
    assert await acall_local(agent, "needs_single", Picture("1")) == 1


async def test_a_key_the_structure_does_not_compute_is_taken_on_trust(agent: BaseAgent) -> None:
    assert (await acall_local(agent, "promises_labels", 4)).channels == 4


async def test_each_item_of_a_list_is_tested(agent: BaseAgent) -> None:
    assert [p.channels for p in await acall_local(agent, "promises_singles", [1, 0])] == [1, 0]
    with pytest.raises(LocalCallError, match=r"@test/n_channels LTE 1 \(actual 5\)"):
        await acall_local(agent, "promises_singles", [1, 5])


async def test_a_port_without_constraints_carries_anything(agent: BaseAgent) -> None:
    assert (await acall_local(agent, "passthrough", Picture("9"))).channels == 9
