"""The task an action is running for, handed to it by annotation.

    def segment(image: ArrayDataset, task: Task, mikro: Mikro):
        task.progress(10, "thresholding")
        mask = mikro.threshold(image)   # attributed to this task automatically
        ...

A ``Task`` is about the task only: what it is (id, user, org, token) and what
it reports (logs, progress, pause points, hooks, and the effects -- ``now()``,
``random(n)``, ``sleep(seconds)`` -- it takes from outside itself). The app's clients are services
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

import asyncio
import inspect
import json
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any, ClassVar

from koil import unkoil, unkoil_gen

from arkitekt_spec.actions import Execution
from arkitekt_spec.declare.agents.errors import NoCallerError
from arkitekt_spec.declare.errors import AgentLost, NotAWorkflowError
from arkitekt_spec.declare.task import aretry, retry
from arkitekt_spec.declare.task import AssignmentHook, LocalTask, LogLevel
from arkitekt_spec.declare.task import Task as TaskProtocol
from arkitekt_runtime.messages import EffectKind

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

    from arkitekt_runtime.actors.helper import AssignmentHelper
    from arkitekt_spec.declare.targets import CallTarget, ImplementationTarget
    from arkitekt_runtime.messages import Assign
    from arkitekt_runtime.postmans.types import Postman
    from arkitekt_spec.declare.task import HookInput
    from arkitekt_spec.declare.structures.registry import StructureRegistry
    from arkitekt_spec.declare.structures.types import JSONSerializable

logger = logging.getLogger("arkitekt.task")

_LOG_LEVELS = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}



def _json(value: Any) -> Any:  # noqa: ANN401
    """``value``, if a recorded effect can carry it (JSON); else a clear TypeError."""
    try:
        json.dumps(value)
    except TypeError as e:
        raise TypeError(f"task.record(...) records JSON only; got {type(value).__name__}: {e}") from None
    return value


class Task:
    """One running assignment, as the action that runs it sees it."""

    __arkitekt_task__: ClassVar[bool] = True  # TASK_MARKER: injected, not a port

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

    # -- effects ---------------------------------------------------------- #
    # Values the task takes from outside itself, each recorded as an ``EFFECT`` under a
    # key. A resumed workflow gets the recorded value back instead of taking a new one.

    async def anow(self) -> float:
        """The current time, in epoch seconds, recorded as this task's effect."""
        return await self._atake(EffectKind.NOW, time.time)

    def now(self) -> float:
        """The current time, in epoch seconds, recorded as this task's effect."""
        return unkoil(self.anow)

    async def arandom(self, n: int = 16) -> str:
        """``n`` random bytes, as hex, recorded as this task's effect."""
        return await self._atake(EffectKind.RANDOM, lambda: secrets.token_hex(n))

    def random(self, n: int = 16) -> str:
        """``n`` random bytes, as hex, recorded as this task's effect."""
        return unkoil(self.arandom, n)

    async def asleep(self, seconds: float) -> None:
        """Sleep for ``seconds``: the deadline is recorded, then slept until.

        A resumed workflow sleeps until the recorded deadline, so a sleep its agent died
        in only waits for what is left of it.
        """
        deadline = await self._atake(EffectKind.SLEEP, lambda: time.time() + max(0.0, seconds))
        await asyncio.sleep(max(0.0, deadline - time.time()))

    def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``: the deadline is recorded, then slept until."""
        deadline = unkoil(self._atake, EffectKind.SLEEP, lambda: time.time() + max(0.0, seconds))
        time.sleep(max(0.0, deadline - time.time()))

    async def arecord(self, fn: "Callable[[], Any]", key: str | None = None) -> Any:  # noqa: ANN401
        """Take a value from outside the task through ``fn`` (awaited if it is a coroutine
        function), recorded under ``key`` (by default ``RECORD:n``).

        A resumed workflow gets the recorded value back and does not call ``fn``: put
        anything that can differ from one run to the next here (an LLM completion, a
        search, a read of something another app may change). JSON only.
        """
        found, value, key = await self._helper.areplay(EffectKind.RECORD, key)
        if found:
            return value
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        await self._helper.aeffect(EffectKind.RECORD, _json(value), key=key)
        return value

    def record(self, fn: "Callable[[], Any]", key: str | None = None) -> Any:  # noqa: ANN401
        """Take a value from outside the task through ``fn``, recorded (see :meth:`arecord`).

        ``fn`` runs here, in the calling thread, not in the event loop.
        """
        found, value, key = unkoil(self._helper.areplay, EffectKind.RECORD, key)
        if found:
            return value
        value = fn()
        unkoil(self._helper.aeffect, EffectKind.RECORD, _json(value), key)
        return value

    async def _atake(self, effect: EffectKind, take: "Callable[[], Any]") -> Any:  # noqa: ANN401
        found, value, key = await self._helper.areplay(effect)
        if found:
            return value
        value = take()
        await self._helper.aeffect(effect, value, key=key)
        return value

    # -- deciding about a lost step ---------------------------------------- #

    def retry(self, call: "Callable[..., Any]", *args: Any, attempts: int = 3, if_started: bool = False, **kwargs: Any) -> Any:  # noqa: ANN401
        """Call ``call(*args, **kwargs)``, again when its agent is lost and that is safe:
        the lost step never started, or ``if_started=True``. Anything else is re-raised."""
        return retry(call, *args, attempts=attempts, if_started=if_started, **kwargs)

    async def aretry(self, call: "Callable[..., Any]", *args: Any, attempts: int = 3, if_started: bool = False, **kwargs: Any) -> Any:  # noqa: ANN401
        """:meth:`retry`, awaiting ``call``."""
        return await aretry(call, *args, attempts=attempts, if_started=if_started, **kwargs)

    async def ahold(self, message: str, *, lost: AgentLost | None = None) -> None:
        """Wait for a person: the task pauses with ``message`` until someone resumes it
        (it carries on from here) or cancels it.

        ``lost`` puts what is known about a lost step in front of whoever decides: what
        running it again would do (its effects), how far it got. A resumed workflow that
        was already resumed from this hold carries on without holding again.
        """
        found, _, key = await self._helper.areplay(EffectKind.HOLD)
        if found:
            return
        details = (
            {"started": lost.started, "last_progress": lost.last_progress, "effects": lost.effects, "task": lost.task}
            if lost is not None
            else None
        )
        await self._helper.ahold(message, details)
        await self._helper.aeffect(EffectKind.HOLD, "resumed", key=key)

    def hold(self, message: str, *, lost: AgentLost | None = None) -> None:
        """Wait for a person (see :meth:`ahold`)."""
        unkoil(self.ahold, message, lost=lost)

    # -- calling ---------------------------------------------------------- #

    def _caller(self) -> "tuple[Postman, StructureRegistry, Assign]":
        """The socket, the registry and the parent for a call made as this task's child.

        Read off the assignment rather than looked up: the postman is the agent's
        caller socket, the parent is this task's own assignment, and the registry is the
        one the actor was built with. Mirrors
        :meth:`rekuest.actors.dependency.AgentMethodProxy._call_kwargs`, whose docstring
        puts it best -- nothing is looked up from context.

        Raises:
            NoCallerError: If no agent runs this task: a child call needs a socket to
                leave over and an assignment to hang off.
        """
        agent = self._helper.agent
        assignment = self._helper.assignment
        if self._helper.execution != Execution.WORKFLOW:
            raise NotAWorkflowError(
                f"Task {self.id!r} called another action, which only a workflow may do: "
                "register it with @app.workflow. A workflow finds the calls it made again "
                "when it is resumed; a plain action cannot."
            )
        if agent is None or assignment is None:
            raise NoCallerError(
                f"Task {self.id!r} runs for no assignment, so a call made through it "
                "would have nothing to be a child of. Call through a Rekuest client "
                "instead -- rekuest.call(action, ...) -- which makes a root."
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
        call_key: str | None = None,
        **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
    ) -> Any:  # noqa: ANN401 -- whatever the action returns
        """Call an action as a child of this task.

        ``target`` is an already-fetched ``Action`` or ``Implementation``. A task knows
        no client, so it cannot look one up by id or by registered function: take a
        ``rekuest: Rekuest`` parameter, ``await rekuest.aresolve(target)``, and pass the
        result here.

        ``call_key`` is what this task calls the child. A workflow that resumes finds the
        child again by it, and gets its result instead of running it twice. Without one,
        it is derived from the target, the args and how often this task made that call:
        pass your own when concurrent calls to the same target with the same args must
        be told apart (a flow engine names them by node).
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
            call_key=call_key,
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
        call_key: str | None = None,
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
            call_key=call_key,
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
        call_key: str | None = None,
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
            call_key=call_key,
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
        call_key: str | None = None,
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
            call_key=call_key,
        ):
            yield value

    def install_hook(self, hook: "AssignmentHook") -> None:
        """Install an assignment hook for this task."""
        self._helper.install_hook(hook)

    @classmethod
    def local(cls, id: str = "local", user: str = "local", org: str = "local") -> LocalTask:
        """A task for calling an action directly, with no agent behind it.

        The spec's :class:`~arkitekt_spec.declare.task.LocalTask`: logs and progress go
        to the ``arkitekt.task`` logger, pause points return at once, and calls raise
        :class:`~arkitekt_spec.declare.agents.errors.NoCallerError`.
        """
        return LocalTask(id=id, user=user, org=org)


if TYPE_CHECKING:  # the type checker proves the running task is the spec's Task

    def _running_task_is_a_task(task: Task) -> TaskProtocol:
        return task


__all__ = ["Task"]
