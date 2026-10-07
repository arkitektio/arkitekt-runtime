"""An action is called in-process through the real agent: ports, hooks, states and all."""

from collections.abc import AsyncIterator, Iterator

import pytest

from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import (
    LocalCallError,
    MemoryAgentTransport,
    acall_local,
    aiterate_local,
    local_agent,
)
from arkitekt_spec.declare.app import AppRegistry


def add(a: int, b: int = 2) -> int:
    """Add, in a worker thread"""
    return a + b


async def aadd(a: int, b: int = 2) -> int:
    """Add, on the loop"""
    return a + b


def count(to: int) -> Iterator[int]:
    """Count up"""
    yield from range(to)


async def acount(to: int) -> AsyncIterator[int]:
    """Count up, on the loop"""
    for i in range(to):
        yield i


def both(text: str) -> tuple[str, int]:
    """Two results"""
    return text.upper(), len(text)


def nothing() -> None:
    """No result"""


def broken(text: str) -> str:
    """Raises"""
    raise ValueError(f"cannot take {text}")


class Canvas:
    def __init__(self, pixels: list[int]) -> None:
        self.pixels = pixels


def brighten(canvas: Canvas) -> Canvas:
    """Takes and returns an object that never leaves the agent"""
    return Canvas([p + 1 for p in canvas.pixels])


@pytest.fixture()
async def agent() -> AsyncIterator[BaseAgent]:
    registry = AppRegistry()
    for function in (add, aadd, count, acount, both, nothing, broken):
        registry.register(function)
    registry.register_memory_structure(Canvas, "@test/canvas")
    registry.register(brighten)
    agent = local_agent(registry)
    await agent.aconnect(timeout=5.0)
    yield agent
    await agent.atear_down()


@pytest.mark.parametrize("interface", ["add", "aadd"])
async def test_a_function_is_called_with_python_values(agent: BaseAgent, interface: str) -> None:
    assert await acall_local(agent, interface, 1) == 3
    assert await acall_local(agent, interface, a=1, b=5) == 6


@pytest.mark.parametrize("interface", ["count", "acount"])
async def test_a_generator_yields_each_result(agent: BaseAgent, interface: str) -> None:
    assert [value async for value in aiterate_local(agent, interface, to=3)] == [0, 1, 2]
    assert await acall_local(agent, interface, to=3) == 2


async def test_several_results_come_back_as_a_tuple_and_none_as_none(agent: BaseAgent) -> None:
    assert await acall_local(agent, "both", "abc") == ("ABC", 3)
    assert await acall_local(agent, "nothing") is None


async def test_a_raising_action_raises_what_a_server_would_be_told(agent: BaseAgent) -> None:
    with pytest.raises(LocalCallError, match="cannot take x") as raised:
        await acall_local(agent, "broken", "x")
    assert raised.value.interface == "broken"
    assert raised.value.critical


async def test_the_arguments_cross_the_ports(agent: BaseAgent) -> None:
    """Not a plain function call: a value the port cannot carry is refused."""
    with pytest.raises(Exception, match="a"):
        await acall_local(agent, "add", "not a number")


async def test_an_unknown_action_and_an_unknown_argument_say_what_exists(agent: BaseAgent) -> None:
    with pytest.raises(KeyError, match="no action 'missing'.*add"):
        await acall_local(agent, "missing")
    with pytest.raises(TypeError, match="no argument 'c'.*a, b"):
        await acall_local(agent, "add", a=1, c=2)


async def test_a_memory_structure_is_passed_and_returned_as_itself(agent: BaseAgent) -> None:
    result = await acall_local(agent, "brighten", Canvas([1, 2]))
    assert isinstance(result, Canvas)
    assert result.pixels == [2, 3]


async def test_calls_do_not_hear_each_other(agent: BaseAgent) -> None:
    import asyncio

    results = await asyncio.gather(*(acall_local(agent, "aadd", i) for i in range(5)))
    assert results == [2, 3, 4, 5, 6]


async def test_an_agent_on_another_transport_is_refused() -> None:
    agent = BaseAgent(transport=MemoryAgentTransport(), app_registry=AppRegistry())
    agent.transport = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="local_agent"):
        await acall_local(agent, "add", 1)
