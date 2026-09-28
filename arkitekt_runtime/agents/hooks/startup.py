"""Hooks for the agent"""

from typing import (
    TYPE_CHECKING,
    Any,
)
from arkitekt_spec.declare.agents.types import BoundApp
from koil.bridge import run_threaded
from arkitekt_spec.declare.agents.hooks.errors import StartupHookError
from arkitekt_spec.declare.agents.hooks.registry import (
    StartupHookReturns,
)
from arkitekt_spec.declare.agents.hooks.declare import StartupWithVariables

if TYPE_CHECKING:
    from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.protocol.types import (
    AsyncStartupFunction,
    ThreadedStartupFunction,
)
from arkitekt_runtime.calls import ensure_return_as_tuple


class WrappedStartupHook(StartupWithVariables):
    """Startup hook that runs in the event loop"""

    def __init__(
        self, func: AsyncStartupFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the startup hook

        Args:
            func (Callable[[str, Any], AnyContext]): The function to run in the startup hook
            func (Callable): The function to run in the startup hook
        """
        super().__init__(func, structure_registry)

    async def arun(
        self,
        app_context: Any,  # noqa: ANN401
        bound_app: BoundApp | None = None,
    ) -> StartupHookReturns:
        """Run the startup hook in the event loop
        Args:
            app_context (Any): The context for the startup hook
            bound_app (Any): The app the agent belongs to, if any
        Returns:
            Optional[Dict[str, Any]]: The state variables and contexts
        """
        parsed_returns = await self.func(**self.get_startup_kwargs(app_context, bound_app))

        returns = ensure_return_as_tuple(parsed_returns)

        states: dict[str, Any] = {}
        contexts: dict[str, Any] = {}

        for index, return_value in enumerate(returns):
            if index in self.state_returns.state_returns:
                states[self.state_returns.state_returns[index]] = return_value
            elif index in self.context_returns.context_returns:
                contexts[self.context_returns.context_returns[index]] = return_value
            else:
                raise StartupHookError(
                    f"Startup hook must return state or context variables. Other returns are not allowed {self.context_returns}, {self.state_returns}"
                )

        return StartupHookReturns(states=states, contexts=contexts)


class ThreadedStartupHook(StartupWithVariables):
    """Startup hook that runs in the event loop"""

    def __init__(
        self, func: ThreadedStartupFunction, structure_registry: "StructureRegistry | None" = None
    ) -> None:
        """Initialize the startup hook

        Args:
            func (Callable[[str], AnyContext]): The function to run in the startup hook
            func (Callable): The function to run in the startup hook
        """
        super().__init__(func, structure_registry)

    async def arun(
        self,
        app_context: Any,  # noqa: ANN401
        bound_app: BoundApp | None = None,
    ) -> StartupHookReturns:
        """Run the startup hook in the event loop
        Args:
            app_context (Any): The context for the startup hook
            bound_app (Any): The app the agent belongs to, if any
        Returns:
            Optional[Dict[str, Any]]: The state variables and contexts
        """

        parsed_returns = await run_threaded(
            self.func,
            **self.get_startup_kwargs(app_context, bound_app),  # type: ignore[arg-type]
        )

        returns = ensure_return_as_tuple(parsed_returns)

        states: dict[str, Any] = {}
        contexts: dict[str, Any] = {}

        for index, return_value in enumerate(returns):
            if index in self.state_returns.state_returns:
                states[self.state_returns.state_returns[index]] = return_value
            elif index in self.context_returns.context_returns:
                contexts[self.context_returns.context_returns[index]] = return_value
            else:
                raise StartupHookError(
                    "Startup hook must return state or context variables. Other returns are not allowed"
                )

        return StartupHookReturns(states=states, contexts=contexts)


# --- Implementation ---
