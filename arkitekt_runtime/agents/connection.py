"""What an agent tells whoever runs it about its connection to the backend."""

from collections.abc import Awaitable, Callable
from enum import Enum


class ConnectionState(str, Enum):
    """Where the agent's connection to the backend stands."""

    REGISTERED = "registered"
    """The backend acknowledged the agent's registration (an ``Init``). Reported on
    the first connection and again whenever a dropped one is back."""
    DISCONNECTED = "disconnected"
    """The link dropped and the transport is trying to get it back."""


ConnectionListener = Callable[[ConnectionState], Awaitable[None]]
"""Called with each change of the agent's connection. It only reports: what it
raises is logged and dropped, never the agent's problem."""
