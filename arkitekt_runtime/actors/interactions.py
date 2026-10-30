"""A task's view of a declared interactions protocol.

    def segment(dataset: Image, ui: ClusterUI, task: Task) -> Scene:
        picked = ui.pick_channels(dataset=dataset)   # the task is HELD on the blok
        ui.progress(step=1, total=3)                 # shown; the task goes on

``ui`` is an :class:`InteractionsProxy` the actor made for the task the action runs in. Each
attribute is one of the protocol's asks (an :class:`AskProxy`) or informs (an
:class:`InformProxy`): calling it shrinks the arguments with the actor's structure registry and
names the declared blok by its key, ``<parameter>.<method>``, which the server looks up on the
running action. An ask holds the task until a person answers and expands what they answered as
a call's returns are; a dismissal raises
:class:`~arkitekt_spec.declare.errors.DismissedError`. An inform shows the blok and returns.
"""

from collections.abc import Callable
from typing import Any, get_type_hints

from koil import unkoil

from arkitekt_spec.declare.definition.define import is_named_tuple
from arkitekt_spec.declare.interactions import DeclaredInteraction, DeclaredInteractions
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.task import TaskBlok

from arkitekt_runtime.task import Task
from arkitekt_runtime.structures.serialization.actor import (
    aexpand_actor_returns,
    ashrink_actor_args,
)


class AskProxy:
    """One ask of an interactions protocol, asked on behalf of one task.

    Made by :class:`InteractionsProxy` for the task the action runs in, never by user code.
    """

    def __init__(self, key: str, ask: DeclaredInteraction, *, task: Task, structure_registry: StructureRegistry) -> None:
        self.key = key
        self.ask = ask
        self.task = task
        self.structure_registry = structure_registry
        returns = get_type_hints(ask.method.func).get("return")
        self._named: Callable[..., Any] | None = returns if returns is not None and is_named_tuple(returns) else None

    async def aask(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- the ask's own arguments and answer
        """Ask and await the answer, expanded as the method's return declares it."""
        shrunk = await ashrink_actor_args(self.ask.definition, args, kwargs, structure_registry=self.structure_registry)
        values = await self.task._aask_declared(TaskBlok(key=self.key, args=shrunk), self.ask.description)
        returns = await aexpand_actor_returns(self.ask.definition, values, self.structure_registry)
        if self._named is not None:
            return self._named(*returns)
        if not returns:
            return None
        if len(returns) == 1:
            return returns[0]
        return tuple(returns)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- ditto
        """Ask as the protocol declared it: awaitable if the method is ``async``, else blocking."""
        if self.ask.is_async:
            return self.aask(*args, **kwargs)
        return unkoil(self.aask, *args, **kwargs)


class InformProxy:
    """One inform of an interactions protocol, shown on behalf of one task.

    Made by :class:`InteractionsProxy` for the task the action runs in, never by user code.
    """

    def __init__(self, key: str, inform: DeclaredInteraction, *, task: Task, structure_registry: StructureRegistry) -> None:
        self.key = key
        self.inform = inform
        self.task = task
        self.structure_registry = structure_registry

    async def ainform(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 -- the inform's own arguments
        """Show the blok, opened on the arguments; nothing waits on it."""
        shrunk = await ashrink_actor_args(self.inform.definition, args, kwargs, structure_registry=self.structure_registry)
        await self.task._ainform_declared(TaskBlok(key=self.key, args=shrunk), self.inform.description)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- ditto
        """Show as the protocol declared it: awaitable if the method is ``async``, else blocking."""
        if self.inform.is_async:
            return self.ainform(*args, **kwargs)
        return unkoil(self.ainform, *args, **kwargs)


class InteractionsProxy:
    """An interactions protocol, made for one task: each ask and inform an attribute."""

    def __init__(
        self, parameter: str, interactions: DeclaredInteractions, *, task: Task, structure_registry: StructureRegistry
    ) -> None:
        self._interactions: dict[str, AskProxy | InformProxy] = {
            name: (AskProxy if each.role == "ask" else InformProxy)(
                f"{parameter}.{name}", each, task=task, structure_registry=structure_registry
            )
            for name, each in interactions.interactions.items()
        }

    def __getattr__(self, name: str) -> AskProxy | InformProxy:
        try:
            return self._interactions[name]
        except KeyError:
            raise AttributeError(
                f"The interactions protocol declares no ask or inform {name!r} (it has: {', '.join(self._interactions)})"
            ) from None
