"""The task an action is running for, handed to it by annotation.

    def segment(image: ArrayDataset, task: Task, mikro: Mikro):
        task.progress(10, "thresholding")
        mask = mikro.threshold(image)   # attributed to this task automatically
        ...

A ``Task`` is about the task only: what it is (id, user, org, token) and what
it reports (logs, progress, pause points, hooks). The app's clients are services
and are injected as their own parameters -- the one shared instance of each, not
a per-task copy. What attributes their requests to this task is
:data:`rath.task.current_task`, which the actor sets around the body. A task
knows no service: it speaks the agent protocol (:mod:`rekuest.messages`) and
nothing of any client, which is what lets ``arkitekt`` export it as its own.

The ``Task`` object itself looks nothing up, so it works the same in the loop, in
a worker thread and in code the action hands it to. The *ambient* task does not
reach a thread the action starts itself; pass ``task=`` to a call, or carry the
context, when you do that (see :mod:`rath.task`).
"""

import logging
from typing import TYPE_CHECKING, Any

from koil import unkoil, unkoil_gen

from arkitekt_runtime.messages import LogLevel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from arkitekt_runtime.actors.helper import AssignmentHelper
    from arkitekt_runtime.actors.types import AssignmentHook
    from arkitekt_runtime.invoke import CallTarget, ImplementationTarget
    from arkitekt_runtime.messages import Assign
    from arkitekt_runtime.postmans.types import Postman
    from arkitekt_runtime.types import HookInput
    from arkitekt_spec.declare.structures.registry import StructureRegistry
    from arkitekt_spec.declare.structures.types import JSONSerializable

logger = logging.getLogger("rekuest.task")

_LOG_LEVELS = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}



class Task:
    """One running assignment, as the action that runs it sees it."""

    __arkitekt_task__ = True  # arkitekt_spec.declare.task.TASK_MARKER: injected, not a port

    def __init__(self, helper: "AssignmentHelper") -> None:
        self._helper = helper

    # -- what the task is ------------------------------------------------- #

    @property
    def assignment(self) -> "Assign":
        """The assignment this task runs."""
        return self._helper.assignment

    @property
    def id(self) -> str:
        """The task id."""
        return self._helper.task

    @property
    def user(self) -> str:
        """The user that caused the task."""
        return self._helper.user

    @property
    def org(self) -> str:
        """The organization the task runs in."""
        return self._helper.org

    @property
    def token(self) -> str | None:
        """The provenance token; ``None`` when the implementation opted out."""
        return self._helper.token

    @property
    def agent(self) -> Any:  # noqa: ANN401 - ActorContext
        """The agent running this task's actor; ``None`` for a :meth:`local` task.

        A client's calls are parented through it -- child calls go over the
        agent's socket -- and a dependency proxy is made for it.
        """
        return self._helper.agent

    # -- reporting -------------------------------------------------------- #

    async def alog(self, message: str, level: LogLevel = LogLevel.DEBUG) -> None:
        """Log to the task."""
        await self._helper.alog(level, str(message))

    def log(self, message: str, level: LogLevel = LogLevel.DEBUG) -> None:
        """Log to the task."""
        unkoil(self.alog, message, level)

    async def aprogress(self, percentage: int, message: str | None = None) -> None:
        """Report progress, 0-100."""
        await self._helper.aprogress(int(percentage), message=message)

    def progress(self, percentage: int, message: str | None = None) -> None:
        """Report progress, 0-100."""
        unkoil(self.aprogress, percentage, message)

    async def apausepoint(self) -> None:
        """Pause here if the task was asked to."""
        await self._helper.abreakpoint()

    def pausepoint(self) -> None:
        """Pause here if the task was asked to."""
        unkoil(self.apausepoint)

    # -- calling ---------------------------------------------------------- #

    def _caller(self) -> "tuple[Postman, StructureRegistry, Assign]":
        """The socket, the registry and the parent for a call made as this task's child.

        Read off the assignment rather than looked up: the postman is the agent's
        caller socket, the parent is this task's own assignment, and the registry is the
        one the actor was built with. Mirrors
        :meth:`rekuest.actors.dependency.AgentMethodProxy._call_kwargs`, whose docstring
        puts it best -- nothing is looked up from context.

        Raises:
            ValueError: If no agent runs this task (a :meth:`local` one): a child call
                needs a socket to leave over and an assignment to hang off.
        """
        agent = self._helper.agent
        assignment = self._helper.assignment
        if agent is None or assignment is None:
            raise ValueError(
                f"Task {self.id!r} runs for no assignment (a Task.local()?), so a call "
                "made through it would have nothing to be a child of. Call through a "
                "Rekuest client instead -- rekuest.call(action, ...) -- which makes a root."
            )
        return agent.caller_postman, self._helper.structure_registry, assignment

    async def acall(
        self,
        target: "CallTarget | ImplementationTarget",
        *args: Any,  # noqa: ANN401 -- the action's own arguments
        reference: str | None = None,
        hooks: "list[HookInput] | None" = None,
        capture: bool = False,
        escalate_to_interrupt: bool = False,
        cancel_timeout: float | None = None,
        **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
    ) -> Any:  # noqa: ANN401 -- whatever the action returns
        """Call an action as a child of this task.

        ``target`` is an already-fetched ``Action`` or ``Implementation``. A task knows
        no client, so it cannot look one up by id or by registered function: take a
        ``rekuest: Rekuest`` parameter, ``await rekuest.aresolve(target)``, and pass the
        result here.
        """
        from arkitekt_runtime.invoke import _acall

        postman, structure_registry, parent = self._caller()
        return await _acall(
            target,
            *args,
            postman=postman,
            structure_registry=structure_registry,
            parent=parent,
            reference=reference,
            hooks=hooks,
            capture=capture,
            escalate_to_interrupt=escalate_to_interrupt,
            cancel_timeout=cancel_timeout,
            **kwargs,
        )

    def call(
        self,
        target: "CallTarget | ImplementationTarget",
        *args: Any,  # noqa: ANN401 -- the action's own arguments
        **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
    ) -> Any:  # noqa: ANN401 -- whatever the action returns
        """Call an action as a child of this task, blocking for its result."""
        self._caller()  # refuse a local task here, not inside koil's loop machinery
        return unkoil(self.acall, target, *args, **kwargs)

    async def aiterate(
        self,
        target: "CallTarget | ImplementationTarget",
        *args: Any,  # noqa: ANN401 -- the action's own arguments
        reference: str | None = None,
        hooks: "list[HookInput] | None" = None,
        capture: bool = False,
        escalate_to_interrupt: bool = False,
        cancel_timeout: float | None = None,
        **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
    ) -> "AsyncGenerator[Any, None]":
        """Stream a generator action's yields as a child of this task."""
        from arkitekt_runtime.invoke import _aiterate

        postman, structure_registry, parent = self._caller()
        async for value in _aiterate(
            target,
            *args,
            postman=postman,
            structure_registry=structure_registry,
            parent=parent,
            reference=reference,
            hooks=hooks,
            capture=capture,
            escalate_to_interrupt=escalate_to_interrupt,
            cancel_timeout=cancel_timeout,
            **kwargs,
        ):
            yield value

    def iterate(
        self,
        target: "CallTarget | ImplementationTarget",
        *args: Any,  # noqa: ANN401 -- the action's own arguments
        **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
    ) -> "Generator[Any, None, None]":
        """Stream a generator action's yields, blocking between them."""
        self._caller()  # refuse a local task here, not inside koil's loop machinery
        return unkoil_gen(self.aiterate, target, *args, **kwargs)

    async def acall_raw(
        self,
        kwargs: "dict[str, JSONSerializable] | None" = None,
        *,
        action: "CallTarget | None" = None,
        implementation: "ImplementationTarget | None" = None,
        reference: str | None = None,
        hooks: "list[HookInput] | None" = None,
        capture: bool = False,
        escalate_to_interrupt: bool = False,
        cancel_timeout: float | None = None,
    ) -> Any:  # noqa: ANN401 -- the raw backend payload
        """Call with already-serialized arguments, as a child of this task."""
        from arkitekt_runtime.invoke import _acall_raw

        postman, _, parent = self._caller()
        return await _acall_raw(
            postman=postman,
            kwargs=kwargs,
            action_id=action.id if action is not None else None,
            implementation_id=implementation.id if implementation is not None else None,
            parent=parent,
            reference=reference,
            hooks=hooks,
            capture=capture,
            escalate_to_interrupt=escalate_to_interrupt,
            cancel_timeout=cancel_timeout,
        )

    async def aiterate_raw(
        self,
        kwargs: "dict[str, JSONSerializable] | None" = None,
        *,
        action: "CallTarget | None" = None,
        implementation: "ImplementationTarget | None" = None,
        reference: str | None = None,
        hooks: "list[HookInput] | None" = None,
        capture: bool = False,
        escalate_to_interrupt: bool = False,
        cancel_timeout: float | None = None,
    ) -> "AsyncGenerator[Any, None]":
        """Stream with already-serialized arguments, as a child of this task."""
        from arkitekt_runtime.invoke import _aiterate_raw

        postman, _, parent = self._caller()
        async for value in _aiterate_raw(
            postman=postman,
            kwargs=kwargs,
            action_id=action.id if action is not None else None,
            implementation_id=implementation.id if implementation is not None else None,
            parent=parent,
            reference=reference,
            hooks=hooks,
            capture=capture,
            escalate_to_interrupt=escalate_to_interrupt,
            cancel_timeout=cancel_timeout,
        ):
            yield value

    def install_hook(self, hook: "AssignmentHook") -> None:
        """Install an assignment hook for this task."""
        self._helper.install_hook(hook)

    @classmethod
    def local(cls, id: str = "local", user: str = "local", org: str = "local") -> "Task":
        """Make a task for calling an action directly, with no agent behind it.

        ``segment(image, task=Task.local(), mikro=mikro)``: logs and progress go
        to the ``rekuest.task`` logger, pause points return at once, and there is
        no assignment and no provenance token, so clients handed out for it
        attribute nothing.

        Args:
            id: The task id its logs are tagged with.
            user: The user the task claims to run for.
            org: The organization the task claims to run in.

        Returns:
            The task.
        """
        return _LocalTask(_LocalHelper(id=id, user=user, org=org))  # type: ignore[arg-type]


class _LocalTask(Task):
    """A :class:`Task` whose sync reporting needs no event loop: there is no agent to reach."""

    def log(self, message: str, level: LogLevel = LogLevel.DEBUG) -> None:
        self._helper.log(level, str(message))

    def progress(self, percentage: int, message: str | None = None) -> None:
        self._helper.progress(int(percentage), message)

    def pausepoint(self) -> None:
        return None


class _LocalHelper:
    """What a :class:`Task` needs of a helper, for a task no agent is running."""

    assignment = None
    token = None
    agent = None
    structure_registry = None

    def __init__(self, id: str, user: str, org: str) -> None:
        self.task = id
        self.user = user
        self.org = org
        self.hooks: list["AssignmentHook"] = []

    def log(self, level: LogLevel, message: str) -> None:
        logger.log(_LOG_LEVELS.get(level, logging.INFO), "[%s] %s", self.task, message)

    def progress(self, percentage: int, message: str | None = None) -> None:
        logger.info("[%s] %d%% %s", self.task, percentage, message or "")

    async def alog(self, level: LogLevel, message: str) -> None:
        self.log(level, message)

    async def aprogress(self, percentage: int, message: str | None = None) -> None:
        self.progress(percentage, message)

    async def abreakpoint(self) -> None:
        return None

    def install_hook(self, hook: "AssignmentHook") -> None:
        self.hooks.append(hook)


__all__ = ["Task"]
