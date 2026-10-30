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
        previous_states: dict[str, Any] | None = None,
    ) -> StartupHookReturns:
        """Run the startup hook in the event loop
        Args:
            app_context (Any): The context for the startup hook
            bound_app (Any): The app the agent belongs to, if any
            previous_states (dict): What earlier runs persisted of the states it asks for
        Returns:
            Optional[Dict[str, Any]]: The state variables and contexts
        """
        parsed_returns = await self.func(
            **self.get_startup_kwargs(app_context, bound_app, previous_states)
        )

        return _split_returns(self, ensure_return_as_tuple(parsed_returns))


def _split_returns(hook: StartupWithVariables, returns: tuple[Any, ...]) -> StartupHookReturns:
    """A startup hook's returns, by the state or context each slot is annotated with.

    A context slot may be ``None``: the hook does not provide that context (a device
    the setup does not have, say). Actions needing it fail when called, and background
    and shutdown hooks needing it are not run. A state is always required.
    """
    states: dict[str, Any] = {}
    contexts: dict[str, Any] = {}

    for index, return_value in enumerate(returns):
        if index in hook.state_returns.state_returns:
            states[hook.state_returns.state_returns[index]] = return_value
        elif index in hook.context_returns.context_returns:
            if return_value is not None:
                contexts[hook.context_returns.context_returns[index]] = return_value
        else:
            raise StartupHookError(
                "Startup hook must return state or context variables. Other returns are not "
                f"allowed {hook.context_returns}, {hook.state_returns}"
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
        previous_states: dict[str, Any] | None = None,
    ) -> StartupHookReturns:
        """Run the startup hook in the event loop
        Args:
            app_context (Any): The context for the startup hook
            bound_app (Any): The app the agent belongs to, if any
            previous_states (dict): What earlier runs persisted of the states it asks for
        Returns:
            Optional[Dict[str, Any]]: The state variables and contexts
        """

        parsed_returns = await run_threaded(
            self.func,
            **self.get_startup_kwargs(app_context, bound_app, previous_states),  # type: ignore[arg-type]
        )

        return _split_returns(self, ensure_return_as_tuple(parsed_returns))


# --- Implementation ---
