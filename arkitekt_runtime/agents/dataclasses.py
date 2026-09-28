"""Plain data containers the agent moves around.

Kept apart from :mod:`rekuest.agents.types` (protocols and type variables)
and from the agent's behaviour so that each can be imported without the other.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, UTC

from arkitekt_runtime import messages
from arkitekt_spec.declare.state.publish import Patch
from arkitekt_spec.declare.structures.types import JSONSerializable


@dataclass
class QueuedPatchEvent:
    """A state patch waiting in the agent's patch queue."""

    interface: str
    patch: Patch
    event_time: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class QueuedMessage:
    """A report waiting in the agent's ordered queue, behind the patches made before it."""

    message: messages.FromAgentMessage
    waiter: "asyncio.Future[None] | None" = None
    """Resolved once the message was handed to the transport, for the few callers
    that wait for it (the session baseline)."""


@dataclass
class QueuedAssign:
    """An assignment to record as its task's first journal entry (served agents)."""

    assign: messages.Assign
    action_key: str


@dataclass
class RevisedState:
    """Current agent-owned shrunk state together with its local revision."""

    revision: int
    data: JSONSerializable


__all__ = ["QueuedAssign", "QueuedMessage", "QueuedPatchEvent", "RevisedState"]
