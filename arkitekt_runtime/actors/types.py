"""Types for the actors module"""

import asyncio
from typing import (
    TYPE_CHECKING,
    Protocol,
    Self,
    runtime_checkable,
    Any,
)
from arkitekt_spec.declare.agents.types import BoundApp
from arkitekt_spec.declare.task import AssignmentHook
from arkitekt_runtime import messages
from arkitekt_spec.actions import Execution
from arkitekt_spec.declare.actors.policy import DisconnectPolicy
from arkitekt_runtime.postmans.types import Postman
from arkitekt_spec.declare.protocol.types import AnyState
from arkitekt_spec.scalars import Identifier
from arkitekt_spec.declare.state.publish import Patch
from collections.abc import Sequence


if TYPE_CHECKING:
    from arkitekt_spec.declare.app import AppRegistry
    from arkitekt_runtime.agents.connection import ConnectionListener
    from arkitekt_runtime.agents.lock import TaskLock


















@runtime_checkable
class Shelver(Protocol):
    """A protocol for mostly fullfield by the agent that is used to store data"""

    async def aput_on_shelve(
        self,
        identifier: Identifier,
        value: Any,  # noqa: ANN401
    ) -> str:  # noqa: ANN401
        """Put a value on the shelve and return the key. This is used to store
        values on the shelve."""
        ...

    async def aget_from_shelve(self, key: str) -> Any:  # noqa: ANN401
        """Get a value from the shelve. This is used to get values from the
        shelve."""
        ...


@runtime_checkable
class Capturable(Protocol):
    """The capture gate an actor's debug tooling coordinates on.

    Only :mod:`rekuest.actors.debug` touches this, which is why it is its own slice
    rather than part of what every actor sees.
    """

    capture_condition: asyncio.Condition
    capture_active: bool


@runtime_checkable
class LockHost(Protocol):
    """Reports lock acquisition, and resolves an actor's declared lock keys.

    Implemented by the agent and used by :class:`~rekuest.agents.lock.TaskLock`.
    """

    async def alock(self, key: str, task: str) -> None:
        """Report that a task has acquired a lock."""
        ...

    async def aunlock(self, key: str, task: str | None = None) -> None:
        """Report that a task (``task``, the holder) has released a lock."""
        ...

    def get_locks_for_keys(self, keys: Sequence[str]) -> list["TaskLock"]:
        """Resolve the agent's task locks for the given lock keys."""
        ...


@runtime_checkable
class ActorContext(Shelver, LockHost, Capturable, Protocol):
    """Everything a running actor needs from the agent above it — and nothing more.

    Deliberately excludes the agent's own lifecycle (``aprovide`` / ``aconnect`` /
    ``aloop``): no actor calls those, and having them in one flat protocol made it read as
    though an actor could drive the agent it runs inside.
    """

    app_registry: "AppRegistry"

    @property
    def caller_postman(self) -> Postman:
        """The agent-as-caller postman: :class:`~arkitekt_runtime.task.Task` calls through it.

        Declared as a property, not an attribute: implementations build it lazily, and a
        mutable protocol attribute is invariant, so a read-only property would not satisfy it.
        """
        ...

    async def asend(self, actor: "Actor", message: messages.FromAgentMessage) -> None:
        """Send a message from an actor up to the agent, which forwards it onward."""
        ...

    async def aget_read_only_proxy(self, key: str) -> AnyState:  # noqa: ANN401
        """Get a state an actor only reads. See the note on the agent implementation:
        read-only is declarative, not enforced."""
        ...

    async def aget_write_proxy(self, key: str, mutation: Any = None) -> AnyState:  # noqa: ANN401
        """Get a state an actor writes to, as a view for ``mutation`` (a task's)."""
        ...

    async def aget_context(self, context: str) -> Any:  # noqa: ANN401
        """Get a context value registered with ``@context``."""
        ...

    async def aget_bound_app(self) -> BoundApp | None:
        """Get the app this agent is bound to, or ``None`` when it runs on its own.

        It answers ``get(cls)`` with the app's clients, which an actor injects
        into parameters annotated with a client class.
        """
        ...

    def publish_patch(self, interface: str, patch: Patch) -> None:
        """Publish a state patch. Satisfies ``state.publish.StateHolder``."""
        ...


@runtime_checkable
class AgentLifecycle(Protocol):
    """Driving the agent itself — what the composition root uses, not what actors use.

    Called only from :class:`~rekuest.client.client.Rekuest` and the FastAPI routes.
    """

    force: bool | None
    """Kick any connection already registered for this agent and take over. ``None``
    defers to the transport's own build-time policy. Settable per run."""
    connection_listener: "ConnectionListener | None"
    """Told when the backend acknowledges the agent and when the link drops.
    Settable per run."""

    async def aprovide(self, context: Any) -> None:  # noqa: ANN401
        """Connect, then process messages until cancelled."""
        ...

    async def aconnect(self, context: Any = None, timeout: float | None = None) -> None:
        """Start the agent and connect, returning once the server acknowledges it."""
        ...

    async def aloop(self) -> None:
        """Process incoming messages after the agent has connected."""
        ...


@runtime_checkable
class Agent(ActorContext, AgentLifecycle, Protocol):
    """The whole agent surface: what actors need plus what drives it.

    Kept as the union of the slices above so every existing annotation and import keeps
    working. Prefer the narrowest slice that fits when writing new code —
    :class:`ActorContext` for anything an actor reaches, :class:`AgentLifecycle` for
    anything that starts or stops the agent.
    """


@runtime_checkable
class Actor(Protocol):
    """An actor is a function that takes a passport and a transport"""

    id: str
    """Stable identifier for this actor, recorded against the tasks it is running."""
    agent: Agent
    policy: DisconnectPolicy
    execution: Execution
    """What happens to this actor's work when the agent loses its control channel."""

    def has_running_tasks(self) -> bool:
        """Whether this actor currently has any assignment in flight."""
        ...

    def install_assignment_hook(self, task_id: str, hook: AssignmentHook) -> None:
        """Install an assignment hook for the current task.

        Args:
            task_id (str): The task to install the hook for.
            hook (AssignmentHook): The hook to install.
        """
        ...

    async def acancel(self) -> None:
        """Stop every assignment this actor is running.

        Called by the agent as it tears down, so in-flight work does not outlive the
        agent that owns it. Each cancelled task is reported to the backend.
        """
        ...

    async def acancel_for_policy(self, reason: str) -> int:
        """Stop every assignment this actor is running, per the disconnect policy.

        Distinct from :meth:`acancel`: it reports the given reason rather than the
        teardown wording, and prunes its task bookkeeping so a later liveness
        inquiry does not claim the killed work is still running. Returns how many
        assignments were stopped.
        """
        ...

    async def ahold(self, task_id: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Pause ``task_id`` by itself until a person resumes it."""
        ...

    async def abreak(self, task_id: str) -> bool:
        """Break the actor. This method will break the actor and return None.
        This is used to break the actor"""
        ...

    async def asend(
        self: Self,
        message: messages.FromAgentMessage,
    ) -> None:
        """Send a message to the actor. This method will send a message to the
        actor and return None.
        """
        ...

    async def apass(
        self: Self,
        message: messages.ToAgentMessage,
    ) -> None:
        """Pass a message to the actor. This method will pass a message to the
        actor and return None.
        """
        ...

    async def acheck_task(
        self: Self,
        task_id: str,
    ) -> bool:
        """Check the task. This method will check the task and
        return None.
        """
        ...


@runtime_checkable
class ActorBuilder(Protocol):
    """An actor builder is a function that takes a passport and a transport
    and returns an actor. This method will create the actor and return it.
    """

    def __call__(
        self,
        agent: Agent,
    ) -> Actor:
        """Create the actor and return it. This method will create the actor and"""

        ...




