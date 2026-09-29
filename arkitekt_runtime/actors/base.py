"""The base class for all actors."""

from arkitekt_spec.declare.agents.dependency import dependency_to_protocol

import asyncio
import contextlib
import logging
from typing import (
    Any,
    Literal,
    Self,
)
from collections.abc import Mapping
import uuid
from functools import partial
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from arkitekt_runtime.actors.dependency import AgentDependencyProxy
from arkitekt_runtime.actors.errors import UnknownMessageError
from arkitekt_spec.declare.actors.policy import KEEP, DisconnectPolicy
from arkitekt_spec.declare.agents.context import PreparedContextReturns, PreparedContextVariables
from arkitekt_spec.declare.agents.errors import StateRequirementsNotMet
from arkitekt_spec.declare.actors.types import (
    PreparedInjectedVariables,
    PreparedDependencyVariables,
)
from arkitekt_runtime.actors.types import Agent
from arkitekt_spec.declare.task import AssignmentHook
from arkitekt_runtime import messages
from arkitekt_spec.actions import DefinitionInput, Execution
from arkitekt_spec.declare.protocol.types import AnyContext, AnyState
from arkitekt_runtime.state.observable import Mutation
from arkitekt_spec.declare.actors.types import (
    PreparedStateReturns,
    PreparedStateVariables,
)
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.task import Task
from arkitekt_runtime.agents.lock import LockGroup

logger = logging.getLogger(__name__)


class Actor(BaseModel):
    """The base class for all actors.

    Actors are the main building blocks of the system and are used to
    perform actions that they receive from the agent. They are responsible for
    processing the actions and sending the results back to the agent.

    Actors are long running processes that are managed by the agent.

    """

    agent: Agent = Field(
        description="The agent that is managing the actor. This is used to send messages to the agent"
    )
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="The id of the actor"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    running_assignments: dict[str, messages.Assign] = Field(default_factory=dict)
    locks: tuple[str, ...] | None = Field(
        default=None,
        description="The lock keys this actor requires. Locks will be acquired before running.",
    )
    concurrency: Literal["parallel", "serial"] = Field(
        default="serial",
        description="Whether assignments to this actor may run concurrently ('parallel') or one at a time ('serial', the default).",
    )
    policy: DisconnectPolicy = Field(
        default=KEEP,
        description="What happens to this actor's in-flight work when the agent loses its control channel. Defaults to keeping it running.",
    )
    execution: Execution = Field(
        default=Execution.PLAIN,
        description="How the implementation runs: only a WORKFLOW may call other actions.",
    )

    _running_asyncio_tasks: dict[str, asyncio.Task[None]] = PrivateAttr(
        default_factory=lambda: {}
    )
    _break_futures: dict[str, asyncio.Future[bool]] = PrivateAttr(
        default_factory=lambda: {}
    )
    _running_assignment_hooks: dict[str, AssignmentHook] = PrivateAttr(
        default_factory=lambda: {},
    )
    _serial_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def has_running_tasks(self) -> bool:
        """Whether this actor currently has any assignment in flight."""
        return bool(self._running_asyncio_tasks)

    def install_assignment_hook(self, task_id: str, hook: AssignmentHook) -> None:
        """Install an assignment hook for the given task ID.

        Args:
            task_id (str): The ID of the task to install the hook for.
            hook (AssignmentHook): The hook to install.
        """
        self._running_assignment_hooks[task_id] = hook

    @contextlib.asynccontextmanager
    async def sync_context(self: Self, task_id: str, interface: str):
        """Context manager that holds the actor's locks while an assignment runs.

        Acquisition order is fixed (see the deadlock invariants documented in
        ``rekuest.agents.lock``): the actor's private serial lock first —
        only when ``concurrency="serial"`` — then the shared lock keys, sorted
        by key. The serial lock is uncontended outside this actor, so shared
        keys are only held while an assignment actually runs, never while it is
        queued behind the actor. None of the locks are reentrant: an assignment
        that re-enters this actor or calls another implementation requiring one
        of its keys will deadlock.

        Args:
            task_id: The ID of the task.
            interface: The interface name for this actor.

        Yields:
            None after all locks are acquired.
        """
        async with contextlib.AsyncExitStack() as stack:
            if self.concurrency == "serial":
                await stack.enter_async_context(self._serial_lock)

            if self.locks:
                lock_group = LockGroup(
                    locks=self.agent.get_locks_for_keys(self.locks),
                    task_id=task_id,
                )
                await stack.enter_async_context(lock_group)

            yield

    async def on_resume(self: Self, resume: messages.Resume) -> None:
        """A function that is called once the actor is resumed from a paused state.
        This can be used to re-initialize the actor after a pause.

        Args:
            resume (Resume): The resume message containing the information about the
                actor that was resumed.
        """
        if resume.task in self._running_assignment_hooks:
            pause_hook = self._running_assignment_hooks[resume.task]
            if pause_hook.kind == "resume":
                logger.info(
                    f"Calling pause hook {pause_hook.id} for task {resume.task}"
                )
                await pause_hook.hook(resume)

        if resume.task in self._break_futures:
            self._break_futures[resume.task].set_result(True)
            if resume.step:
                # Step: resume only until the next breakpoint by re-arming the
                # break future (the equivalent of the former standalone Step).
                self._break_futures[resume.task] = asyncio.Future()
            else:
                del self._break_futures[resume.task]
        else:
            logger.warning(
                f"Actor {self.id} was resumed but no break future was found for {resume.task}"
            )

    async def asend(
        self: Self,
        message: messages.FromAgentMessage,
    ) -> None:
        """A function to send a message to the agent. This is used to send messages
        to the agent from the actor.

        Args:
            message (ToAgentMessage): The message to send
        """
        await self.agent.asend(self, message=message)

    async def on_pause(self: Self, pause: messages.Pause) -> None:
        """A function that is called once the actor is paused. This can be used to
        clean up resources or stop any ongoing tasks.

        Args:
            pause (Pause): The pause message containing the information about the
                actor that was paused.
        """
        if pause.task in self._running_assignment_hooks:
            pause_hook = self._running_assignment_hooks[pause.task]
            if pause_hook.kind == "pause":
                logger.info(f"Calling pause hook {pause_hook.id} for task {pause.task}")
                await pause_hook.hook(pause)

        if pause.task in self._break_futures:
            logger.warning(
                f"Actor {self.id} was paused but a break future was already set for {pause.task}"
            )
            return

        self._break_futures[pause.task] = asyncio.Future()

    async def on_assign(
        self: Self,
        assignment: messages.Assign,
    ) -> None:
        """A function that is called once the actor is assigned a task. This is used to
        process the task and send the results back to the agent.

        Args:
            assignment (messages.Assign): The assignment message containing the information about the
             assignment.
            collector (TaskCollector): A collector that is used to collect the results of the assignment.

        Raises:
            NotImplementedError: Needs to be overwritten in Actor subclass. Never use this class directly
        """
        raise NotImplementedError(
            "Needs to be owerwritten in Actor Subclass. Never use this class directly"
        )

    async def apass(self: Self, message: messages.ToAgentMessage) -> None:
        """A function that is called once the actor is passed a message. This is used to
        process the message and send the results back to the agent.

        Args:
            self (Self): A reference to the actor instance.
            message (messages.FromAgentMessage):   The message to process.
        """
        await self.aprocess(message)

    async def acancel(self: Self) -> None:
        """A function to cancel the actor. This is used to cancel the actor and
        stop listening for messages from the agent.
        """
        logger.info(f"Cancelling Actor {self.id}")
        await self._astop_all(
            "Cancelled trhough application (this is not nice from the application and will be regarded as an error)",
            prune=False,
        )

    async def _astop_all(self: Self, error: str, prune: bool) -> int:
        """Cancel every running assignment, await each, and report ``Critical(error)``.

        The mechanism shared by :meth:`acancel` and :meth:`acancel_for_policy`; the
        two differ in wording and in whether the bookkeeping is pruned (see the
        latter's docstring for why).

        Returns:
            How many assignments were stopped.
        """
        running = list(self._running_asyncio_tasks.items())
        stopped = 0
        for key, task in running:
            task.cancel()
            # Reported before the task unwinds (nothing is awaited in between), so
            # its end is recorded before what the unwinding reports (the release of
            # its locks), and nothing it still does is recorded after it.
            await self.agent.asend(
                self, message=messages.Critical(task=key, error=error)
            )
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {key} was cancelled. Set Critical")
                stopped += 1
            except Exception:  # noqa: BLE001 — the body failed on its way out
                logger.error("Task %s errored while being stopped", key, exc_info=True)
            if prune:
                self._running_asyncio_tasks.pop(key, None)
                self.running_assignments.pop(key, None)
        return stopped

    async def ahold(self: Self, task_id: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Pause ``task_id`` by itself until a person resumes it (``task.hold``).

        The same pause a Pause request makes, started from the task: PAUSED goes out with
        why, and the task waits until a RESUME comes (or a cancel ends it).
        """
        self._break_futures.setdefault(task_id, asyncio.Future())
        await self.agent.asend(self, message=messages.Paused(task=task_id, message=message, details=details))
        await self._break_futures[task_id]
        await self.agent.asend(self, message=messages.Resumed(task=task_id))

    async def abreak(self: Self, task_id: str) -> bool:
        """A function to pause the actor. This is used to instruct the actor to
        stop processing the assignment at the current time
        """
        if task_id in self._break_futures:
            logger.debug(f"Breaking on task_id {task_id}")
            await self.agent.asend(
                self,
                message=messages.Paused(
                    task=task_id,
                ),
            )
            await self._break_futures[task_id]
            await self.agent.asend(
                self,
                message=messages.Resumed(
                    task=task_id,
                ),
            )
            return True
        else:
            logger.debug(
                f"Currently no break future for {task_id} was found. Wasn't paused"
            )
            return False

    def assign_task_done(self: Self, task_id: str, task: asyncio.Task[None]) -> None:
        """Called once an assignment task finishes, however it finished.

        This prunes ``_running_asyncio_tasks``. Nothing used to: the map was only
        ever cleared on the explicit Cancel/Interrupt paths, so a task that simply
        ran to completion stayed in it forever. That made :meth:`acheck_task` answer
        "still running" for work that had finished hours earlier — which is the
        answer the agent gives the backend when it inquires about task liveness
        after a reconnect — and leaked an entry per assignment besides.

        Args:
            task_id: The assignment this task was running.
            task: The task that finished.
        """
        logger.info(f"Assign task is done: {task}")
        # ``pop`` rather than ``del``: done callbacks run via ``call_soon``, so the
        # Cancel/Interrupt paths can and do race this one, and either may get there
        # first.
        self._running_asyncio_tasks.pop(task_id, None)
        self.running_assignments.pop(task_id, None)
        # These two are keyed per task as well and used to outlive it: a task that
        # was cancelled while paused kept its break future, and every task kept
        # its hook.
        self._running_assignment_hooks.pop(task_id, None)
        self._break_futures.pop(task_id, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Assign task {task} failed with exception {e}", exc_info=True)

    async def acancel_for_policy(self: Self, reason: str) -> int:
        """Stop this actor's in-flight work because the disconnect policy says so.

        Deliberately *not* :meth:`acancel`. That one reports ``Critical`` with "this
        is not nice from the application", which is the right thing to say about a
        teardown that interrupts work but the wrong thing to say about a deliberate
        safety stop — and its wording is pinned by the teardown tests. This one also
        prunes ``_running_asyncio_tasks``, which :meth:`acancel` does not: a policy
        kill is followed by a reconnect, and a stale entry there would make the
        agent answer "still running" to the backend's liveness inquiry about a task
        it had just killed.

        Args:
            reason: Human-readable cause, reported to the backend.

        Note the whole call is bounded by the agent's ``actor_cancel_timeout``, and
        each task's terminal report is awaited in turn — so an actor holding many
        assignments while the socket is mid-reconnect can exhaust that budget and
        have the remainder abandoned.

        Returns:
            How many assignments were stopped.
        """
        if not self._running_asyncio_tasks:
            return 0

        logger.warning(
            "Actor %s stopping %d assignment(s): %s",
            self.id,
            len(self._running_asyncio_tasks),
            reason,
        )
        return await self._astop_all(reason, prune=True)

    async def acheck_task(self: Self, task_id: str) -> bool:
        """A function to check if the assignment is still running. This is used to
        check if the assignment is still running and if it is still valid.

        Args:
            id (str): The id of the assignment to check.
        Returns:
            bool: True if the assignment is still running, False otherwise.
        """
        if task_id in self._running_asyncio_tasks:
            return True
        return False

    async def _astop_task(
        self: Self,
        task_id: str,
        terminal: type[messages.Cancelled] | type[messages.Interrupted],
        verb: str,
    ) -> None:
        """Stop one running assignment on the backend's request and report ``terminal``.

        ``Cancel`` and ``Interrupt`` differ only in the terminal message they owe the
        backend; the mechanism — cancel the task, await it, report — is the same.
        """
        if task_id not in self._running_asyncio_tasks:
            logger.error(
                f"Actor for {self}: Received {verb} for unknown task {task_id}"
            )
            return

        task = self._running_asyncio_tasks[task_id]
        if task.done():
            logger.warning(f"Race Condition: Task was already done before {verb}")
            await self.agent.asend(self, message=terminal(task=task_id))
            return

        task.cancel()
        # Reported before the task unwinds (nothing is awaited in between): the end
        # closes the task's gate, so nothing the task still does -- a state change
        # from a worker thread, a log on its way out -- is recorded after it, and the
        # release of its locks (``UNLOCK``) follows it.
        await self.agent.asend(self, message=terminal(task=task_id))
        try:
            await task
        except asyncio.CancelledError:
            logger.info(
                f"Task {task_id} was {verb} through arkitekt. Reported {terminal.__name__}"
            )
        except Exception:  # noqa: BLE001 — the body failed on its way out
            logger.error("Task %s errored while being %s", task_id, verb, exc_info=True)
        self._running_asyncio_tasks.pop(task_id, None)

    async def _aon_assign_reporting(self: Self, assignment: messages.Assign) -> None:
        """Run :meth:`on_assign`, guaranteeing the backend hears how it ended.

        ``on_assign`` reports the failures it anticipates (bad inputs, the function
        raising). Anything it does *not* anticipate — a missing state or context when
        resolving the function's locals, no bound app, a lock or publisher that fails
        on entry — used to escape into the task, where :meth:`assign_task_done` only
        logged it. The backend was never told, and the task sat there forever with a
        perfectly healthy-looking agent. This is the net under every ``on_assign``,
        whichever actor class implements it.

        It also reports ``STARTED`` first, for every actor class: it is the task's step 1,
        and until it arrives the server still counts the task as QUEUED, so an idempotent
        task whose agent dies is taken for one already re-queued and never re-dispatched.
        """
        try:
            await self.asend(message=messages.Started(task=assignment.task))
            await self.on_assign(assignment)
        except asyncio.CancelledError:
            raise  # Cancel/Interrupt/teardown report their own terminal message
        except Exception as e:
            logger.critical(
                f"Assignment {assignment.task} on {assignment.interface} failed "
                "outside the actor's own error handling",
                exc_info=True,
            )
            await self.asend(
                message=messages.Critical(
                    task=assignment.task,
                    error=f"{type(e).__name__}: {e}",
                )
            )

    async def aprocess(self: Self, message: messages.ToAgentMessage) -> None:
        """A function to process the message. This is used to process the message
        and send the results back to the agent.


        Args:
            message (messages.ToAgentMessage): The message to process.
        """

        logger.info(f"Actor for {self.id}: Received {message}")

        if isinstance(message, messages.Assign):
            if message.step:
                # We are creating a break future already
                logger.debug(f"Creating break future for task {message.task} in step")
                self._break_futures[message.task] = asyncio.Future()

            task = asyncio.create_task(self._aon_assign_reporting(message))

            task.add_done_callback(partial(self.assign_task_done, message.task))
            self._running_asyncio_tasks[message.task] = task

        elif isinstance(message, messages.Cancel):
            await self._astop_task(message.task, messages.Cancelled, "cancelled")

        elif isinstance(message, messages.Interrupt):
            await self._astop_task(message.task, messages.Interrupted, "interrupted")

        elif isinstance(message, messages.Pause):
            await self.on_pause(message)

        elif isinstance(message, messages.Resume):
            await self.on_resume(message)

        else:
            raise UnknownMessageError(f"{message}")


class SerializingActor(Actor):
    """A serializing actor is an actor that will
    serialize and deserialize the arguments and return values
    of the assignments it receives.

    """

    definition: DefinitionInput = Field(
        description="The definition of the actor, describing what arguents and return values it provides"
    )
    state_returns: PreparedStateReturns = Field(
        description="The state returns of the actor"
    )
    state_variables: PreparedStateVariables = Field(
        description="The state variables of the actor"
    )
    dependency_variables: PreparedDependencyVariables = Field(
        description="The dependency variables of the actor"
    )
    context_variables: PreparedContextVariables = Field(
        description="The context variables of the actor"
    )
    context_returns: PreparedContextReturns = Field(
        description="The context returns of the actor"
    )
    injected_variables: PreparedInjectedVariables = Field(
        default_factory=PreparedInjectedVariables,
        description="The parameters that receive the app the agent is bound to",
    )

    structure_registry: StructureRegistry = Field(
        default_factory=StructureRegistry,
        description="The structure regsistry to use for this actor",
    )
    expand_inputs: bool = Field(
        default=True,
        description="Whether to expand the inputs of the actor. Can overwrite the default behaviour of the actor to expand the inputs with the structure registry.",
    )
    shrink_outputs: bool = Field(
        default=True,
        description="Whether to shrink the outputs of the actor. Can overwrite the default behaviour of the actor to shrink the outputs with the structure registry.",
    )

    async def aget_locals(
        self: Self,
        assignment: Any = None,  # noqa: ANN401 - messages.Assign
    ) -> tuple[Mapping[str, AnyContext], Mapping[str, AnyState]]:
        """The contexts and states the function asked for, keyed by parameter."""

        state_kwargs: Mapping[str, AnyContext | AnyState] = {}
        context_kwargs: Mapping[str, AnyContext] = {}

        for key, interface in self.context_variables.context_variables.items():
            try:
                context_kwargs[key] = await self.agent.aget_context(interface)
            except KeyError as e:
                raise StateRequirementsNotMet(f"State requirements not met: {e}") from e

        for key, interface in self.state_variables.write_state_variables.items():
            try:
                state_kwargs[key] = await self.agent.aget_write_proxy(
                    interface,
                    Mutation(
                        correlation_id=assignment.task if assignment else None,
                        locks=frozenset(self.locks or ()),
                        gate=self._task_gate(assignment),
                    ),
                )
            except KeyError as e:
                raise StateRequirementsNotMet(f"State requirements not met: {e}") from e

        for key, interface in self.state_variables.read_only_variables.items():
            try:
                state_kwargs[key] = await self.agent.aget_read_only_proxy(interface)
            except KeyError as e:
                raise StateRequirementsNotMet(f"State requirements not met: {e}") from e

        return context_kwargs, state_kwargs

    def _task_gate(self, assignment: Any) -> Any:  # noqa: ANN401 - TaskGate | None
        """The gate of the assignment's task: its state changes are refused after its end."""
        task_gate = getattr(self.agent, "task_gate", None)
        if assignment is None or task_gate is None:
            return None
        return task_gate(assignment.task)

    async def aget_dependency_locals(
        self: Self, task: Task
    ) -> Mapping[str, AgentDependencyProxy]:
        """A proxy for every parameter annotated with a declared protocol, made
        for ``task``: its calls are children of that task and leave over this
        actor's agent.

        Raises:
            StateRequirementsNotMet: If a parameter's protocol is not declared on
                this actor's structure registry.
        """
        proxies: dict[str, AgentDependencyProxy] = {}
        for key, cls in self.dependency_variables.dependency_variables.items():
            try:
                protocol = dependency_to_protocol(cls, self.structure_registry)
            except KeyError as e:
                raise StateRequirementsNotMet(f"State requirements not met: {e}") from e
            proxies[key] = AgentDependencyProxy(
                key,
                protocol,
                task=task,
                agent=self.agent,
                structure_registry=self.structure_registry,
            )
        return proxies

    async def aget_injected_locals(
        self: Self, task: Task | None = None
    ) -> Mapping[str, Any]:
        """The Task, and the app's clients, keyed by every parameter that asked
        for them by annotation. ``task`` is the one the actor made for this
        assignment; the dependency proxies are made for the same object.

        A client is handed out as itself -- one shared instance per app:
        one client is shared by every concurrent task, so the view is what lets
        what it does be attributed to this one (mikro stamps the task's token on
        its requests, rekuest parents its calls to the task).

        Raises:
            StateRequirementsNotMet: If the function asks for a client the agent's
                app does not have, or for its Task outside an assignment. Passing
                ``None`` instead would only move the failure into the function
                body, away from its cause.
        """
        wanted = self.injected_variables
        if wanted.count == 0:
            return {}

        if task is None and wanted.task_variables:
            raise StateRequirementsNotMet(
                f"'{self.definition.name}' asks for its Task, but it is not being "
                "run for an assignment."
            )

        kwargs: dict[str, Any] = {key: task for key in wanted.task_variables}
        # The agent refused to start with anything but the declared class, so the
        # instance is what the annotation promised.
        kwargs.update({key: self.agent.app_context for key in wanted.app_context_variables})
        if not wanted.service_client_variables:
            return kwargs

        from arkitekt_spec.declare.agents.types import resolve_service_clients

        app = await self.agent.aget_bound_app()
        kwargs.update(
            resolve_service_clients(
                wanted.service_client_variables,
                app,
                whose=f"'{self.definition.name}'",
                task=task,
            )
        )
        return kwargs


Actor.model_rebuild()
SerializingActor.model_rebuild()
