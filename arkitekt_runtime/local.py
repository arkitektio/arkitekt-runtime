"""Running an app's actions in-process: an agent with no backend behind it.

An action is not its function. Between a call and the function sit the ports
(arguments expanded, results shrunk), the injected clients, the task, the states
and the startup hooks. Calling the function directly skips all of it, so what a
test or a quick try needs is the real agent, fed an assignment by hand.

:class:`MemoryAgentTransport` is the transport that makes that possible: a queue in,
a list out. :func:`local_agent` builds an agent on it that acknowledges itself, and
:func:`acall_local` / :func:`aiterate_local` call one action through it with Python
values, the way a caller on the other side of a server would.
"""

import asyncio
from types import TracebackType
import uuid
from typing import Any, Optional, Self, TypeVar
from collections.abc import AsyncIterator

from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.targets import SerializablePort
from arkitekt_spec.actions import PortKind

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.agents.transport.base import AgentTransport
from arkitekt_runtime.structures.serialization.expand import aexpand_returns
from arkitekt_runtime.structures.serialization.shrink import ashrink_args

_CLOSED = object()

#: Who a local assignment says it comes from: there is no server to have said.
LOCAL_AGENT_ID = "local"
LOCAL_USER = "local"
LOCAL_ORG = "local"

T = TypeVar("T", bound=messages.FromAgentMessage)


class MemoryAgentTransport(AgentTransport):
    """A queue-backed transport that records what the agent sends."""

    sent: list[messages.FromAgentMessage] = []
    """Every outbound message, in order, including the ``seq`` the agent stamped."""

    acknowledge: bool = False
    """Answer the agent's registration itself, as a backend would with its ``Init``.

    Off, the agent waits for an ``Init`` somebody feeds; on, connecting is all it
    takes to have a running agent."""

    _watchers: list["asyncio.Queue[messages.FromAgentMessage]"] = []

    _in_queue: Optional["asyncio.Queue[object]"] = None
    _connected: bool = False

    def model_post_init(self, __context: object) -> None:
        """Give every instance its own recording list and queue."""
        self.sent = []
        self._watchers = []
        self._in_queue = asyncio.Queue()

    # -- inbound -------------------------------------------------------------------

    def feed(self, message: messages.ToAgentMessage) -> None:
        """Hand the agent a message, as the backend would."""
        self._queue.put_nowait(message)

    def fail(self, error: BaseException) -> None:
        """Make the stream raise, as a terminal connection failure does."""
        self._queue.put_nowait(error)

    def close_stream(self) -> None:
        """End the stream, as a closed connection does."""
        self._queue.put_nowait(_CLOSED)

    @property
    def _queue(self) -> "asyncio.Queue[object]":
        if self._in_queue is None:  # pragma: no cover - guarded by model_post_init
            raise RuntimeError("Transport was not entered")
        return self._in_queue

    async def areceive(self) -> AsyncIterator[messages.ToAgentMessage]:
        """Yield fed messages until the stream is closed."""
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, messages.Message)
            yield item

    # -- outbound ------------------------------------------------------------------

    async def asend(self, message: messages.FromAgentMessage) -> None:
        """Record an outbound message instead of putting it on a wire."""
        self.sent.append(message)
        for watcher in self._watchers:
            watcher.put_nowait(message)

    def watch(self) -> "asyncio.Queue[messages.FromAgentMessage]":
        """A queue of everything sent from now on. Hand it back to :meth:`unwatch`."""
        watcher: asyncio.Queue[messages.FromAgentMessage] = asyncio.Queue()
        self._watchers.append(watcher)
        return watcher

    def unwatch(self, watcher: "asyncio.Queue[messages.FromAgentMessage]") -> None:
        """Stop filling a queue :meth:`watch` returned."""
        self._watchers.remove(watcher)

    def of_type(self, kind: type[T]) -> list[T]:
        """Every recorded message of one type, for readable assertions."""
        return [m for m in self.sent if isinstance(m, kind)]

    # -- lifecycle -----------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """Whether :meth:`aconnect` has run and :meth:`adisconnect` has not."""
        return self._connected

    async def aconnect(self) -> None:
        """Mark the transport connected. There is no socket to open."""
        self._connected = True
        if self.acknowledge and self._host is not None:
            params = await self._host.aget_handshake_params()
            declared = params.declaration.hash if params.declaration is not None else None
            self.feed(messages.Init(agent=LOCAL_AGENT_ID, hash=declared))

    async def adisconnect(self) -> None:
        """Mark the transport disconnected and end the stream."""
        self._connected = False
        self.close_stream()

    async def drop_link(self) -> None:
        """Simulate the socket dropping while the transport keeps retrying.

        The stream deliberately stays open: that is the whole point of the window
        this models. The real websocket transport reconnects transparently, so the
        agent's message loop never ends and the only signal it gets is this
        callback.
        """
        self._connected = False
        await self.anotify_connection_change(False)

    async def restore_link(self) -> None:
        """Simulate the transport getting its socket back."""
        self._connected = True
        await self.anotify_connection_change(True)

    async def __aenter__(self) -> Self:
        """Enter the transport context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Leave the transport context."""
        await self.adisconnect()


class LocalCallError(Exception):
    """An action called in-process failed.

    Attributes:
        interface: The action that was called.
        error: What the actor reported, as a server would have been told.
        critical: Whether the action raised (rather than reported a failure).
    """

    def __init__(self, interface: str, error: str, critical: bool = True) -> None:
        super().__init__(f"'{interface}' failed: {error}")
        self.interface = interface
        self.error = error
        self.critical = critical


class LocalCallCancelledError(LocalCallError):
    """An action called in-process was cancelled before it finished."""


def local_agent(registry: AppRegistry, name: str = "local", **options: Any) -> BaseAgent:  # noqa: ANN401
    """An agent serving ``registry`` with no backend: it registers with itself.

    Connect it (``await agent.aconnect(context)``) and it runs the app's startup
    hooks, states and background work as a provided app does; tear it down
    (``await agent.atear_down()``) to run the shutdown hooks.

    Args:
        registry: What the agent serves: an app's registry, or a run's snapshot of it.
        name: The agent's name.
        **options: Anything else :class:`BaseAgent` takes (``task_scopes``,
            ``task_listener``).
    """
    return BaseAgent(
        name=name,
        transport=MemoryAgentTransport(acknowledge=True),
        app_registry=registry,
        **options,
    )


def _local_transport(agent: BaseAgent) -> MemoryAgentTransport:
    transport = agent.transport
    if not isinstance(transport, MemoryAgentTransport):
        raise TypeError(
            f"A local call needs an agent on a MemoryAgentTransport, and this one is on "
            f"a {type(transport).__name__}. Build it with `local_agent(registry)`."
        )
    return transport


def _definition_of(agent: BaseAgent, interface: str) -> Any:  # noqa: ANN401
    implementations = agent.app_registry.implementations
    if interface not in implementations:
        offered = ", ".join(sorted(implementations)) or "none"
        raise KeyError(f"This app has no action '{interface}'. It has: {offered}.")
    return implementations[interface].definition


def _each_memory_value(port: SerializablePort, value: Any, swap: Any) -> Any:  # noqa: ANN401
    """``value`` with every memory structure in it replaced by ``swap(port, it)``."""
    if value is None:
        return None
    if port.kind == PortKind.MEMORY_STRUCTURE:
        return swap(port, value)
    children = getattr(port, "children", None) or []
    if port.kind == PortKind.LIST and children and isinstance(value, (list, tuple)):
        return [_each_memory_value(children[0], item, swap) for item in value]
    if port.kind == PortKind.DICT and children and isinstance(value, dict):
        return {key: _each_memory_value(children[0], item, swap) for key, item in value.items()}
    return value


async def aiterate_local_raw(
    agent: BaseAgent, interface: str, args: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Assign ``interface`` once, with arguments as they travel, and yield what it yields.

    Args:
        agent: A connected agent on a :class:`MemoryAgentTransport`.
        interface: The action to call.
        args: Its arguments, already shrunk.

    Raises:
        LocalCallError: If the action fails or raises.
        LocalCallCancelledError: If it is cancelled first.
    """
    transport = _local_transport(agent)
    task = uuid.uuid4().hex
    watcher = transport.watch()
    try:
        transport.feed(
            messages.Assign(
                task=task,
                interface=interface,
                args=args,
                implementation=f"local-{interface}",
                action=f"local-{interface}",
                reference=task,
                user=LOCAL_USER,
                org=LOCAL_ORG,
            )
        )
        while True:
            message = await watcher.get()
            if getattr(message, "task", None) != task:
                continue
            if isinstance(message, messages.Yield):
                yield message.returns or {}
            elif isinstance(message, messages.Completed):
                return
            elif isinstance(message, messages.Critical):
                raise LocalCallError(interface, message.error, critical=True)
            elif isinstance(message, messages.Failed):
                raise LocalCallError(interface, message.error, critical=False)
            elif isinstance(message, (messages.Cancelled, messages.Interrupted)):
                raise LocalCallCancelledError(interface, "it was cancelled")
    finally:
        transport.unwatch(watcher)


async def aiterate_local(
    agent: BaseAgent,
    interface: str,
    *args: Any,  # noqa: ANN401 -- the action's own arguments
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> AsyncIterator[Any]:
    """Call ``interface`` in-process with Python values and yield each of its results.

    The arguments are shrunk through the action's ports and the results expanded
    back, so the call crosses the same serialization a remote one does: a value a
    port cannot carry fails here as it would against a server. A memory structure
    is put on the agent's shelve going in and read off it coming out, so it is
    passed and returned as the object itself.

    Raises:
        KeyError: If the app has no such action.
        LocalCallError: If the action fails or raises.
    """
    _local_transport(agent)
    definition = _definition_of(agent, interface)
    registry = agent.app_registry.structure_registry

    def shelve(port: SerializablePort, value: Any) -> Any:  # noqa: ANN401
        drawer = uuid.uuid4().hex
        agent.shelve[drawer] = value
        return drawer

    def unshelve(port: SerializablePort, drawer: Any) -> Any:  # noqa: ANN401
        return agent.shelve[drawer]

    args = tuple(
        _each_memory_value(port, value, shelve) for port, value in zip(definition.args, args)
    )
    by_key = {port.key: port for port in definition.args}
    kwargs = {
        key: _each_memory_value(by_key[key], value, shelve) if key in by_key else value
        for key, value in kwargs.items()
    }
    unknown = sorted(set(kwargs) - set(by_key))
    if unknown:
        raise TypeError(
            f"'{interface}' takes no argument {', '.join(repr(k) for k in unknown)}. "
            f"It takes: {', '.join(by_key) or 'nothing'}."
        )

    shrunk = await ashrink_args(definition, args, kwargs, structure_registry=registry)
    async for raw in aiterate_local_raw(agent, interface, shrunk):
        returns = await aexpand_returns(definition, raw, structure_registry=registry)
        returns = tuple(
            _each_memory_value(port, value, unshelve)
            for port, value in zip(definition.returns, returns)
        )
        if not returns:
            yield None
        elif len(returns) == 1:
            yield returns[0]
        else:
            yield returns


async def acall_local(
    agent: BaseAgent,
    interface: str,
    *args: Any,  # noqa: ANN401 -- the action's own arguments
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> Any:  # noqa: ANN401 -- whatever the action returns
    """Call ``interface`` in-process and return its result (the last, of a generator).

    See :func:`aiterate_local` for what the call goes through.

    Raises:
        KeyError: If the app has no such action.
        LocalCallError: If the action fails or raises.
    """
    result: Any = None
    async for result in aiterate_local(agent, interface, *args, **kwargs):  # noqa: B007
        pass
    return result


__all__ = [
    "LOCAL_AGENT_ID",
    "LocalCallCancelledError",
    "LocalCallError",
    "MemoryAgentTransport",
    "acall_local",
    "aiterate_local",
    "aiterate_local_raw",
    "local_agent",
]
