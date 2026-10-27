"""Named returns through the real agent: a returned (or yielded) NamedTuple lands in the return
ports its fields name, and ``Key`` names a single return on the wire."""

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, NamedTuple

import pytest

from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import acall_local, aiterate_local, aiterate_local_raw, local_agent
from arkitekt_spec.declare.annotations import Key
from arkitekt_spec.declare.app import AppRegistry


class Measured(NamedTuple):
    count: int
    mean: float


def measure(values: list[int]) -> Measured:
    """Count and average"""
    return Measured(count=len(values), mean=sum(values) / len(values))


def measure_each(values: list[int]) -> Iterator[Measured]:
    """Count and average every prefix"""
    for end in range(1, len(values) + 1):
        yield Measured(count=end, mean=sum(values[:end]) / end)


async def ameasure_each(values: list[int]) -> AsyncIterator[Measured]:
    """Count and average every prefix, on the loop"""
    for end in range(1, len(values) + 1):
        yield Measured(count=end, mean=sum(values[:end]) / end)


def total(values: list[int]) -> Annotated[int, Key("total")]:
    """Sum"""
    return sum(values)


@pytest.fixture()
async def agent() -> AsyncIterator[BaseAgent]:
    registry = AppRegistry()
    for function in (measure, measure_each, ameasure_each, total):
        registry.register(function)
    agent = local_agent(registry)
    await agent.aconnect(timeout=5.0)
    yield agent
    await agent.atear_down()


async def test_a_returned_named_tuple_fills_the_ports_its_fields_name(agent: BaseAgent) -> None:
    assert [returns async for returns in aiterate_local_raw(agent, "measure", {"values": [1, 2, 3]})] == [{"count": 3, "mean": 2.0}]
    assert await acall_local(agent, "measure", [1, 2, 3]) == (3, 2.0)


@pytest.mark.parametrize("interface", ["measure_each", "ameasure_each"])
async def test_each_yielded_named_tuple_fills_the_ports_its_fields_name(agent: BaseAgent, interface: str) -> None:
    raw = [returns async for returns in aiterate_local_raw(agent, interface, {"values": [2, 4]})]
    assert raw == [{"count": 1, "mean": 2.0}, {"count": 2, "mean": 3.0}]
    assert [value async for value in aiterate_local(agent, interface, [2, 4])] == [(1, 2.0), (2, 3.0)]


async def test_a_keyed_return_travels_under_its_key(agent: BaseAgent) -> None:
    assert [returns async for returns in aiterate_local_raw(agent, "total", {"values": [1, 2]})] == [{"total": 3}]
    assert await acall_local(agent, "total", [1, 2]) == 3
