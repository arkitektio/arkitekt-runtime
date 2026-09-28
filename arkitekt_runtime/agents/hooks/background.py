"""Hooks for the agent"""

from typing import (
    TYPE_CHECKING,
    Any,
)

from arkitekt_spec.declare.agents.types import BoundApp
from koil.bridge import run_threaded
from arkitekt_spec.declare.state.publish import StateHolder
from arkitekt_spec.declare.agents.hooks.declare import BackgroundWithVariables

if TYPE_CHECKING:
    from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.protocol.types import (
    ThreadedBackgroundFunction,
    AsyncBackgroundFunction,
)


class WrappedBackgroundTask(BackgroundWithVariables):
    """Background task that runs in the event loop"""

    def __init__(
        self, func: AsyncBackgroundFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the background task
        Args:
            func (Callable): The function to run in the background async
        """
        super().__init__(func, structure_registry)

    async def arun(
        self,
        agent: StateHolder,
        contexts: dict[str, Any],
        states: dict[str, Any],
        app_context: Any = None,  # noqa: ANN401
        bound_app: BoundApp | None = None,
    ) -> None:
        """Run the background task in the event loop"""
        kwargs = self.get_kwargs(contexts, states, app_context, bound_app=bound_app)
        return await self.func(**kwargs)


class WrappedThreadedBackgroundTask(BackgroundWithVariables):
    """Background task that runs in a thread pool"""

    def __init__(
        self, func: ThreadedBackgroundFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the background task
        Args:
            func (Callable): The function to run in the background
        """
        super().__init__(func, structure_registry)

    def run_with_publishing(self, agent: StateHolder, **kwargs: Any) -> None:
        return self.func(**kwargs)

    async def arun(
        self,
        agent: StateHolder,
        contexts: dict[str, Any],
        states: dict[str, Any],
        app_context: Any = None,  # noqa: ANN401
        bound_app: BoundApp | None = None,
    ) -> None:
        """Run the background task in a thread pool"""
        kwargs = self.get_kwargs(contexts, states, app_context, bound_app=bound_app)
        return await run_threaded(
            self.run_with_publishing,
            agent,
            **kwargs,  # type: ignore[arg-type]
        )


# --- Implementation ---
