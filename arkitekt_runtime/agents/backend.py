"""The agent's backend: its id, its sessions, and the shelve.

Registration is not here: it is the transport handshake. ``Register`` carries the agent's
declaration (:meth:`~rekuest.agents.base.BaseAgent.aget_handshake_params`) and ``Init``
answers it, so an agent registers by connecting and nothing in its path calls the rekuest
GraphQL API. What a deployment still varies is where sessions are minted: against a Rekuest
server locally (rekuest's ``SocketAgentBackend``), against the in-process FastAPI agent a
local sink, and an agent under test has neither (:class:`LocalAgentBackend`). Shelved values
are the agent's own, recorded as numbered frames, in every deployment.

That difference used to be expressed by *subclassing the agent*, which is why swapping
deployments meant overriding a handful of unrelated ``BaseAgent`` methods. Here it is a
collaborator instead, so the agent has one implementation and the deployment picks a
backend.
"""

import logging
import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


logger = logging.getLogger(__name__)


@runtime_checkable
class AgentBackend(Protocol):
    """Where an agent's sessions are minted.

    Shelving is not here: the agent keeps shelved values itself and records them as
    numbered ``SHELVE``/``UNSHELVE`` frames, whatever its backend.
    """

    @property
    def registered_agent_id(self) -> str | None:
        """The id the backend assigned this agent, once registered.

        ``None`` before registration, and for backends that assign none.
        """
        ...

    async def acreate_session(self) -> str:
        """Mint the identifier for this run of the agent."""
        ...


class LocalAgentBackend(BaseModel):
    """A backend that keeps nothing anywhere: no id, in-memory sessions."""

    @property
    def registered_agent_id(self) -> str | None:
        """Nobody assigned an id."""
        return None

    async def acreate_session(self) -> str:
        """A fresh identifier per process."""
        return str(uuid.uuid4())


__all__ = ["AgentBackend", "LocalAgentBackend"]
