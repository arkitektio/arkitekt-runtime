"""Hooks for the agent"""

from typing import (
    TYPE_CHECKING,
    Any,
)

from arkitekt_spec.declare.agents.types import BoundApp
from koil.bridge import run_threaded
from arkitekt_spec.declare.state.publish import StateHolder
from arkitekt_spec.declare.agents.hooks.declare import ShutdownWithVariables

if TYPE_CHECKING:
    from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.protocol.types import (
    AsyncShutdownFunction,
    ThreadedShutdownFunction,
)


class WrappedShutdownHook(ShutdownWithVariables):
    """Shutdown hook that runs in the event loop"""

    def __init__(
        self, func: AsyncShutdownFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the shutdown hook

        Args:
            func (Callable): The function to run when the agent tears down
        """
        super().__init__(func, structure_registry)

    async def arun(
        self,
        agent: StateHolder,
        contexts: dict[str, Any],
        states: dict[str, Any],
        app_context: Any,
        bound_app: BoundApp | None = None,
    ) -> None:
        """Run the shutdown hook in the event loop"""
        kwargs = self.get_kwargs(contexts, states, app_context, bound_app=bound_app)
        await self.func(**kwargs)


class ThreadedShutdownHook(ShutdownWithVariables):
    """Shutdown hook that runs in a thread"""

    def __init__(
        self, func: ThreadedShutdownFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the shutdown hook

        Args:
            func (Callable): The function to run when the agent tears down
        """
        super().__init__(func, structure_registry)

    def run_with_publishing(self, agent: StateHolder, **kwargs: Any) -> None:
        self.func(**kwargs)

    async def arun(
        self,
        agent: StateHolder,
        contexts: dict[str, Any],
        states: dict[str, Any],
        app_context: Any,
        bound_app: BoundApp | None = None,
    ) -> None:
        """Run the shutdown hook in a thread"""
        kwargs = self.get_kwargs(contexts, states, app_context, bound_app=bound_app)
        await run_threaded(
            self.run_with_publishing,
            agent,
            **kwargs,  # type: ignore[arg-type]
        )


# --- Implementation ---
