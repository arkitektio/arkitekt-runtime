"""The exceptions for the Postman module."""

from arkitekt_spec.declare.errors import RekuestError


class PostmanException(RekuestError):
    """
    Base class for all exceptions raised by the Agent.
    """


class AssignException(PostmanException):
    """
    Raised when an error occurs during the assignment of a task to an agent.
    """


#: How the server begins its refusal of a stateful action assigned by action alone.
STATEFUL_NEEDS_AGENT = "A stateful action is assigned to a named agent"


class StatefulNeedsAgent(AssignException):
    """Raised when a stateful action was assigned without saying on which agent.

    A stateful action keeps state on the agent that runs it, so "whichever agent is
    free" is not an answer: the server refuses an assign that names such an action
    alone. The message keeps what the server said and adds what to do about it.
    """

    def __init__(self, error: str) -> None:
        """Keep the server's text, and say how the call names an agent."""
        said = error.strip()
        if not said.endswith((".", "!", "?")):
            said += "."
        super().__init__(
            f"{said} Pass on_agent= to the call (or call an Implementation, which names its agent)."
        )


class RootOnlyAssignError(PostmanException):
    """Raised when a non-root call is handed to a transport that only creates roots.

    A GraphQL ``assign`` creates a *root* task by definition: the backend dropped
    ``parent``/``dependency``/``method`` from ``AssignInput``, so a child task or a
    dependency method call can only be originated over the agent socket. Silently
    dropping those fields would detach a child into an orphan root, so we raise.
    """
