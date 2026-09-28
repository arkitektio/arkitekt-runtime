"""Turning an app's hook declarations into the runners an agent drives."""

from typing import Any

from arkitekt_spec.declare.agents.hooks.declare import (
    BackgroundDeclaration,
    ShutdownHookDeclaration,
    StartupHookDeclaration,
)
from arkitekt_runtime.agents.hooks.background import (
    WrappedBackgroundTask,
    WrappedThreadedBackgroundTask,
)
from arkitekt_runtime.agents.hooks.shutdown import ThreadedShutdownHook, WrappedShutdownHook
from arkitekt_runtime.agents.hooks.startup import ThreadedStartupHook, WrappedStartupHook


def materialize_hook(declaration: Any) -> Any:  # noqa: ANN401
    """The runner for a declared hook: in the event loop, or in a worker thread.

    A hook that already runs (a hand-written ``BackgroundTask``/``StartupHook``/
    ``ShutdownHook`` with an ``arun``) is returned as it is.
    """
    if hasattr(declaration, "arun"):
        return declaration
    func, registry = declaration.func, declaration.structure_registry
    if isinstance(declaration, StartupHookDeclaration):
        cls = ThreadedStartupHook if declaration.threaded else WrappedStartupHook
    elif isinstance(declaration, BackgroundDeclaration):
        cls = WrappedThreadedBackgroundTask if declaration.threaded else WrappedBackgroundTask
    elif isinstance(declaration, ShutdownHookDeclaration):
        cls = ThreadedShutdownHook if declaration.threaded else WrappedShutdownHook
    else:
        raise TypeError(f"Not a hook declaration: {declaration!r}")
    return cls(func, registry)
