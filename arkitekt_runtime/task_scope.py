"""What an assignment runs inside: the scopes its runtime asks for.

Service clients attribute their requests to the task they run for. rath's clients read
it from rath's ambient task (``rath.task.task_scope``); other clients may read it from
elsewhere. The core knows none of them: a runtime lists the scope factories its agent
enters around every assignment (``BaseAgent.task_scopes``) -- rekuest's agent enters
rath's by default. Each factory is called with the running :class:`~arkitekt_runtime.task.Task`.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any

#: A context-manager factory entered around an assignment, handed its task.
TaskScope = Callable[[Any], AbstractContextManager[Any]]


@contextmanager
def enter_task_scopes(scopes: Iterable[TaskScope], task: Any) -> Iterator[None]:  # noqa: ANN401
    """Enter every scope for ``task``, innermost last; unwind them all on the way out."""
    with ExitStack() as stack:
        for scope in scopes:
            stack.enter_context(scope(task))
        yield
