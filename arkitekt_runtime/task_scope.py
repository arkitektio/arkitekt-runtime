"""What an assignment runs inside: the scopes its runtime asks for.

Service clients attribute their requests to the task they run for. rath's clients read
it from rath's ambient task (``rath.task.task_scope``); other clients may read it from
elsewhere. The core knows none of them: a runtime lists the scope factories its agent
enters around every assignment (``BaseAgent.task_scopes``) -- rekuest's agent enters
rath's by default. Each factory is called with the running :class:`~arkitekt_runtime.task.Task`.

The core's own ambient is :data:`current_task_id`: the id of the task whose code is
running, which is what names the task that shelved a value (``SHELVE.task``).
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any

#: The id of the task whose code is running (``None`` outside of one).
current_task_id: ContextVar[str | None] = ContextVar("arkitekt_current_task_id", default=None)


@contextmanager
def task_id_scope(task_id: str | None) -> Iterator[None]:
    """Run the block as ``task_id``'s code (see :data:`current_task_id`)."""
    token = current_task_id.set(task_id)
    try:
        yield
    finally:
        current_task_id.reset(token)

#: A context-manager factory entered around an assignment, handed its task.
TaskScope = Callable[[Any], AbstractContextManager[Any]]


@contextmanager
def enter_task_scopes(scopes: Iterable[TaskScope], task: Any) -> Iterator[None]:  # noqa: ANN401
    """Enter every scope for ``task``, innermost last; unwind them all on the way out."""
    with ExitStack() as stack:
        stack.enter_context(task_id_scope(getattr(task, "id", None)))
        for scope in scopes:
            stack.enter_context(scope(task))
        yield
