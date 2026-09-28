"""A task's view of a declared dependency.

    def capture(camera: Camera, task: Task) -> bytes:
        return camera.snap(exposure_ms=10.0)   # a child of this task

``camera`` is an :class:`AgentDependencyProxy` the actor made for the task the
action runs in, handed the ``Task`` and the agent the way an injected client is
(the task it runs for is ambient, see :mod:`rath.task`). Each attribute is
one of the protocol's declared actions, an :class:`AgentMethodProxy`, and calling
it sends the call over the agent's socket as a child of that task, (de)serialized
with the actor's structure registry.

A dependency's state demands are not readable through the proxy.
"""

from typing import Any

from arkitekt_spec.scalars import coerce_id

from arkitekt_runtime.actors.types import ActorContext
from arkitekt_spec.declare.declare import DeclaredAgentAction, DeclaredAgentProtocol
from arkitekt_runtime.calls import acall_dependency, call_dependency
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.task import Task
from arkitekt_spec.declare.agents.errors import NoCallerError
from arkitekt_spec.declare.task import LocalTask


class AgentMethodProxy:
    """One method of a dependency, called on behalf of one task.

    Made by :class:`AgentDependencyProxy` for the task the action runs in, never by
    user code. The call is parented to the task's assignment, goes out over the
    agent's socket (``agent.caller_postman``, read when the call is made) and is
    (de)serialized with the actor's structure registry: the one the protocol's ports
    were built against. Nothing is looked up from context.
    """

    def __init__(
        self,
        dependency_key: str,
        method: str,
        action: DeclaredAgentAction[Any, Any],
        *,
        task: Task,
        agent: ActorContext,
        structure_registry: StructureRegistry,
    ) -> None:
        self.dependency_key = dependency_key
        self.method = method
        self.action = action
        self.is_async = action.is_async
        self.task = task
        self.agent = agent
        self.structure_registry = structure_registry

    def _call_args(self) -> tuple[Any, ...]:
        return (
            self.action.definition,
            coerce_id(self.dependency_key),
            self.method,
        )

    def _call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("parent", self.task.assignment)
        kwargs.setdefault("postman", self.agent.caller_postman)
        kwargs.setdefault("structure_registry", self.structure_registry)
        return kwargs

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Call the dependency's method and block for its result."""
        return call_dependency(*self._call_args(), *args, **self._call_kwargs(kwargs))

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Call the dependency's method and await its result."""
        return await acall_dependency(
            *self._call_args(), *args, **self._call_kwargs(kwargs)
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call as the protocol declared the method: awaitable if it is ``async``."""
        if self.is_async:
            return self.acall(*args, **kwargs)

        return self.call(*args, **kwargs)


class AgentDependencyProxy:
    """A declared dependency, as the action that asked for it sees it.

    Made by the actor for one task, never by user code: ``def capture(camera: Camera)``
    receives one of these, and ``camera.snap(...)`` calls the resolved agent's action
    as a child of that task. Its attributes are the protocol's actions and nothing
    else; everything of its own is underscore-prefixed, since a declared protocol
    never has such members (see :class:`~rekuest.declare.DeclaredAgentProtocol`),
    so no attribute here can shadow a declared method.
    """

    def __init__(
        self,
        key: str,
        protocol: DeclaredAgentProtocol[Any],
        *,
        task: Task | LocalTask,
        agent: ActorContext,
        structure_registry: StructureRegistry,
    ) -> None:
        """Bind the dependency under ``key`` to ``task`` and ``agent``.

        Args:
            key: The dependency's key on the implementation (the parameter name).
            protocol: The protocol the app declared for the parameter's annotation.
            task: The task the action runs for; its calls become children of it.
            agent: The agent running the actor; calls leave over its socket.
            structure_registry: The actor's registry, which built the protocol's ports.

        Raises:
            NoCallerError: If ``task`` runs for no assignment (a ``Task.local()``): a
                dependency call is never a root, so there is nothing to parent it to.
        """
        if not isinstance(task, Task) or task.assignment is None:
            raise NoCallerError(
                f"Dependency '{key}' can only be called for an assignment; task "
                f"{task.id!r} runs for none (Task.local()?)."
            )
        self._key = key
        self._protocol = protocol
        self._task = task
        self._agent = agent
        self._structure_registry = structure_registry

    def __getattr__(self, name: str) -> AgentMethodProxy:
        """The method proxy for the declared action ``name``.

        Raises:
            AttributeError: If the protocol declares no such action. A state demand
                is named as such: a dependency's state is not readable through the
                proxy.
        """
        action = self._protocol.actions.get(name)
        if action is None:
            if name in self._protocol.states:
                raise AttributeError(
                    f"'{name}' is a state demand of '{self._protocol.interface}'; "
                    "reading a dependency's state through the proxy is not supported."
                )
            raise AttributeError(
                f"'{self._protocol.interface}' (dependency '{self._key}') has no "
                f"action '{name}'. Its actions are: "
                f"{', '.join(sorted(self._protocol.actions)) or 'none'}."
            )
        return AgentMethodProxy(
            self._key,
            name,
            action,
            task=self._task,
            agent=self._agent,
            structure_registry=self._structure_registry,
        )

    def __dir__(self) -> list[str]:
        return sorted(self._protocol.actions)


__all__ = ["AgentDependencyProxy", "AgentMethodProxy"]
