"""An agent's lifecycle on the core: the shelve is its own and emptied on teardown,
and a message stream that ends or fails stops the agent.

Runs a bare BaseAgent through the core handshake (``Register`` answered by ``Init``).
"""

import asyncio

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.agents.connection import ConnectionListener, ConnectionState
from arkitekt_runtime.agents.transport.types import HandshakeParams
from arkitekt_spec.declare.agents.errors import AgentException
from arkitekt_spec.declare.app import AppRegistry

from .memory_transport import MemoryAgentTransport


class RecordingTransport(MemoryAgentTransport):
    """A memory transport that asks its host for the handshake, as a socket one does."""

    handshakes: list[HandshakeParams] = []

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self.handshakes = []

    async def aconnect(self) -> None:
        assert self._host is not None, "the agent installs itself before connecting"
        self.handshakes.append(await self._host.aget_handshake_params())
        await super().aconnect()


async def _until(predicate, timeout: float = 2.0) -> None:  # noqa: ANN001
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0)


async def _connected(transport: RecordingTransport) -> BaseAgent:
    """An agent through ``aconnect``: the backend acknowledged what ``Register`` declared."""
    agent = BaseAgent(transport=transport, app_registry=AppRegistry(), name="lifecycle-test")
    connecting = asyncio.create_task(agent.aconnect(timeout=2.0))
    await _until(lambda: transport.handshakes)
    transport.feed(messages.Init(agent="agent-1", hash=transport.handshakes[0].declaration.hash))
    await connecting
    return agent

@pytest.mark.asyncio
async def test_collect_of_an_unknown_drawer_keeps_the_loop_alive() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)
    agent.shelve["drawer-1"] = object()

    # A drawer this agent never held (or already dropped) must not kill the loop.
    transport.feed(messages.Collect(drawers=["unknown"]))
    transport.feed(messages.Collect(drawers=["drawer-1"]))
    await _until(lambda: "drawer-1" not in agent.shelve)
    await agent.atear_down()


@pytest.mark.asyncio
async def test_teardown_empties_the_shelve() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)
    agent.shelve["drawer-1"] = object()

    await agent.atear_down()

    assert agent.shelve == {}


@pytest.mark.asyncio
async def test_teardown_empties_the_shelve_even_when_it_fails() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)
    agent.shelve["drawer-1"] = object()

    async def failing_disconnect() -> None:
        raise RuntimeError("transport broke while disconnecting")

    object.__setattr__(transport, "adisconnect", failing_disconnect)

    with pytest.raises(RuntimeError):
        await agent.atear_down()

    assert agent.shelve == {}


@pytest.mark.asyncio
async def test_missing_drawer_is_a_clear_error() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)

    with pytest.raises(AgentException, match="drawer-x"):
        await agent.aget_from_shelve("drawer-x")
    await agent.atear_down()




@pytest.mark.asyncio
async def test_a_stream_that_ends_tears_the_agent_down() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)
    agent.shelve["drawer-1"] = object()

    looping = asyncio.create_task(agent.aloop())
    await _until(lambda: agent.running)
    transport.close_stream()
    await asyncio.wait_for(looping, timeout=2.0)

    assert not agent.running
    assert not transport.connected
    assert agent.shelve == {}


@pytest.mark.asyncio
async def test_a_failing_stream_stops_the_agent_running() -> None:
    transport = RecordingTransport()
    agent = await _connected(transport)

    looping = asyncio.create_task(agent.aloop())
    await _until(lambda: agent.running)
    transport.fail(RuntimeError("socket died"))
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(looping, timeout=2.0)

    assert not agent.running


# -- the connection listener -------------------------------------------------------


async def _listening(
    transport: RecordingTransport, listener: ConnectionListener
) -> BaseAgent:
    """An agent through ``aconnect`` whose connection ``listener`` hears about."""
    agent = BaseAgent(
        transport=transport,
        app_registry=AppRegistry(),
        name="lifecycle-test",
        connection_listener=listener,
    )
    connecting = asyncio.create_task(agent.aconnect(timeout=2.0))
    await _until(lambda: transport.handshakes)
    transport.feed(messages.Init(agent="agent-1", hash=transport.handshakes[0].declaration.hash))
    await connecting
    return agent


@pytest.mark.asyncio
async def test_the_listener_hears_the_registration_and_every_drop_and_return() -> None:
    heard: list[ConnectionState] = []

    async def listener(state: ConnectionState) -> None:
        heard.append(state)

    transport = RecordingTransport()
    agent = await _listening(transport, listener)
    assert heard == [ConnectionState.REGISTERED]

    await transport.drop_link()
    assert heard[-1] is ConnectionState.DISCONNECTED

    # The socket coming back is not yet a registration: only the backend's Init is.
    await transport.restore_link()
    assert heard == [ConnectionState.REGISTERED, ConnectionState.DISCONNECTED]

    transport.feed(messages.Init(agent="agent-1", hash=transport.handshakes[0].declaration.hash))
    await _until(lambda: len(heard) == 3)
    assert heard[-1] is ConnectionState.REGISTERED
    await agent.atear_down()


@pytest.mark.asyncio
async def test_a_failing_listener_never_reaches_the_agent() -> None:
    async def listener(state: ConnectionState) -> None:
        raise RuntimeError("the reporter broke")

    transport = RecordingTransport()
    agent = await _listening(transport, listener)

    await transport.drop_link()
    agent.shelve["drawer-1"] = object()
    transport.feed(messages.Collect(drawers=["drawer-1"]))
    await _until(lambda: "drawer-1" not in agent.shelve)
    await agent.atear_down()
