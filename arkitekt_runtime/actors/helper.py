"""The AssignmentHelper is a helper class that is used to manage the assignment"""

from typing import Any, Protocol, Self, runtime_checkable
from pydantic import BaseModel, ConfigDict
from enum import Enum

from arkitekt_spec.declare.task import LogLevel
from koil import unkoil
from arkitekt_runtime import messages
from arkitekt_runtime.actors.types import Actor
from arkitekt_spec.declare.task import AssignmentHook


@runtime_checkable
class _Serializes(Protocol):
    """An actor that carries a structure registry, i.e. one with ports."""

    @property
    def structure_registry(self) -> Any:  # noqa: ANN401 - StructureRegistry
        """What this actor's calls (de)serialize with."""
        ...


class AssignmentHelper(BaseModel):
    """Helper class to manage an assignment during its lifetime.

    Can be used to send logs, progress and to inspect for breakpoints.
    """

    assignment: messages.Assign
    actor: Actor
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def agent(self) -> Any:  # noqa: ANN401 - ActorContext
        """The agent running the actor this assignment belongs to."""
        return self.actor.agent

    @property
    def structure_registry(self) -> Any:  # noqa: ANN401 - StructureRegistry
        """What a call made as this assignment's child (de)serializes with.

        Read off the concrete actor rather than declared on the ``Actor`` protocol: only
        a ``SerializingActor`` carries a registry -- an actor with no ports never shrinks
        anything -- and a mutable protocol attribute would be invariant besides.
        """
        actor = self.actor
        if not isinstance(actor, _Serializes):
            raise ValueError(
                f"The actor running {self.assignment.task!r} does not serialize, so it "
                "has no structure registry for a call to (de)serialize with."
            )
        return actor.structure_registry

    async def alog(
        self: Self, level: LogLevel | messages.LogLevelLiteral, message: str
    ) -> None:
        """Send a log message to the actor.

        Args:
            level (LogLevel): The log level.
            message (str): The log message.
        """
        await self.actor.asend(
            message=messages.Log(
                task=self.assignment.task,
                level=level.value if isinstance(level, Enum) else level,
                message=message,
            )
        )

    def install_hook(self, hook: "AssignmentHook") -> None:
        """Install an assignment hook for the current task.

        Args:
            hook (AssignmentHook): The hook to install.
        """
        self.actor.install_assignment_hook(self.assignment.task, hook)

    async def aprogress(self, progress: int, message: str | None = None) -> None:
        """Send a progress message to the actor.

        Args:
            progress (int): The progress percentage.
            message (Optional[str]): The progress message.
        """
        if progress < 0 or progress > 100:
            raise ValueError("Progress must be between 0 and 100")

        await self.actor.asend(
            message=messages.Progress(
                task=self.assignment.task,
                progress=progress,
                message=message,
            )
        )

    async def aeffect(self, effect: messages.EffectKind, value: float | str) -> None:
        """Record a value the task took from outside itself (an ``EFFECT``).

        Args:
            effect (EffectKind): What was taken (the clock, randomness, a deadline).
            value (float | str): The value taken.
        """
        await self.actor.asend(
            message=messages.Effect(task=self.assignment.task, effect=effect, value=value)
        )

    async def abreakpoint(self) -> bool:
        """Check if the actor needs to break"""
        return await self.actor.abreak(self.assignment.task)
        # await self.actor.acheck_needs_break()

    def breakpoint(self) -> bool:
        """Check if the actor needs to break

        This is a blocking call, and should be
        only called from a seperath thread (i.e
        from the actor thread
        )

        """
        return unkoil(self.abreakpoint)

    def progress(self, progress: int, message: str | None = None) -> None:
        """Send a progress message to the agent.

        Args:
            progress (int): The progress percentage.
            message (Optional[str]): The progress message.
        """

        return unkoil(self.aprogress, progress, message=message)

    def log(self, level: LogLevel, message: str) -> None:
        """Send a log message to the agent.

        Args:
            level (LogLevel): The log level.
            message (str): The log message.
        """
        return unkoil(self.alog, level, message)

    @property
    def user(self) -> str:
        """Returns the user that caused the task"""
        return self.assignment.user

    @property
    def task(self) -> str:
        """Returns the governing task that cause the chained that lead to this execution"""
        return self.assignment.task

    @property
    def org(self) -> str:
        """Returns the organization that caused the task"""
        return self.assignment.org

    @property
    def action(self) -> str:
        """Returns the node that caused the task"""
        return self.assignment.action

    @property
    def args(self) -> dict[str, Any]:
        """Returns the args that caused the task"""
        return self.assignment.args

    @property
    def token(self) -> str | None:
        """Returns the opaque provenance token of the task, if any.

        The token is forwarded untouched to downstream services; it is None
        when the implementation opted out of provenance (needs_token=False).
        """
        return self.assignment.token
