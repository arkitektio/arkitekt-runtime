"""A task's gate: after a task's end is reported, nothing more is done for it.

Reports and state changes made for a task *enter* its gate for their (synchronous)
duration. Reporting the task's end *closes* the gate first: closing refuses new
entries and waits for the ones in flight, so nothing the task does afterwards -- a
cancelled task still running in a worker thread, a log on its way out -- is recorded
after its end. The check happens before anything is changed, so a refused change
leaves the state as it was.

Entering twice on one thread is allowed (a log made while a change is in progress),
and closing never waits for passes held by the closing thread itself.
"""

import contextlib
import threading
from collections.abc import Iterator

__all__ = ["TaskClosedError", "TaskGate"]


class TaskClosedError(RuntimeError):
    """A report or state change for a task whose end was already reported."""


class TaskGate:
    """Closes a task: see the module docs."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._closed = False
        # Passes in flight, by the thread holding them.
        self._active: dict[int, int] = {}

    @property
    def closed(self) -> bool:
        """Whether the gate is closed (or closing)."""
        with self._condition:
            return self._closed

    def enter(self) -> bool:
        """Take a pass; ``False`` once the gate is closed. Pair with :meth:`leave`."""
        with self._condition:
            if self._closed:
                return False
            thread = threading.get_ident()
            self._active[thread] = self._active.get(thread, 0) + 1
            return True

    def leave(self) -> None:
        """Give back a pass taken by :meth:`enter`."""
        with self._condition:
            thread = threading.get_ident()
            count = self._active.get(thread, 0) - 1
            if count > 0:
                self._active[thread] = count
            else:
                self._active.pop(thread, None)
            self._condition.notify_all()

    @contextlib.contextmanager
    def held(self, what: str = "task") -> Iterator[None]:
        """Hold a pass for the block; raise :class:`TaskClosedError` if the gate is closed."""
        if not self.enter():
            raise TaskClosedError(f"The {what} has already ended; nothing more is recorded for it")
        try:
            yield
        finally:
            self.leave()

    def close(self) -> None:
        """Refuse new passes, then wait until those other threads hold are given back."""
        with self._condition:
            self._closed = True
            me = threading.get_ident()
            while any(count for thread, count in self._active.items() if thread != me):
                self._condition.wait()

    def __repr__(self) -> str:
        return f"TaskGate(closed={self.closed})"
