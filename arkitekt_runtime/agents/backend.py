"""The agent's backend: its id, its sessions, and the shelve.

Registration is not here: it is the transport handshake. ``Register`` carries the agent's
declaration (:meth:`~rekuest.agents.base.BaseAgent.aget_handshake_params`) and ``Init``
answers it, so an agent registers by connecting and nothing in its path calls the rekuest
GraphQL API. What a deployment still varies is where sessions are minted and where shelved
values are recorded: against a Rekuest server that is the agent's own socket
(rekuest's ``SocketAgentBackend``), against the in-process FastAPI agent it is a local sink, and
an agent under test has neither (:class:`LocalAgentBackend`).

That difference used to be expressed by *subclassing the agent*, which is why swapping
deployments meant overriding a handful of unrelated ``BaseAgent`` methods. Here it is a
collaborator instead, so the agent has one implementation and the deployment picks a
backend.
"""

import logging
import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from arkitekt_spec.scalars import Identifier

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentBackend(Protocol):
    """Where an agent's sessions are minted and its shelved values recorded."""

    @property
    def registered_agent_id(self) -> str | None:
        """The id the backend assigned this agent, once registered.

        ``None`` before registration, and for backends that assign none.
        """
        ...

    async def acreate_session(self) -> str:
        """Mint the identifier for this run of the agent."""
        ...

    async def ashelve(
        self,
        identifier: Identifier,
        resource_id: str,
        label: str | None = None,
        description: str | None = None,
    ) -> str:
        """Put a value on the shelve and return its drawer id."""
        ...

    async def acollect(self, key: str) -> None:
        """Release a drawer."""
        ...


class LocalAgentBackend(BaseModel):
    """A backend that keeps nothing anywhere: no id, in-memory sessions, no shelve."""

    @property
    def registered_agent_id(self) -> str | None:
        """Nobody assigned an id."""
        return None

    async def acreate_session(self) -> str:
        """A fresh identifier per process."""
        return str(uuid.uuid4())

    async def ashelve(
        self,
        identifier: Identifier,
        resource_id: str,
        label: str | None = None,
        description: str | None = None,
    ) -> str:
        """Not supported: there is no shelve to put anything on."""
        raise NotImplementedError(
            "This agent has no shelve. Give it a backend that provides one "
            "(e.g. SocketAgentBackend) to shelve values."
        )

    async def acollect(self, key: str) -> None:
        """Not supported: nothing was ever shelved remotely."""
        raise NotImplementedError(
            "This agent has no shelve, so there is nothing to collect."
        )




__all__ = ["AgentBackend", "LocalAgentBackend"]
