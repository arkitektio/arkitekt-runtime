"""Calling something, described by ids rather than by fetched models.

This is the half of the call surface that needs no GraphQL: a target is an id,
the postman is handed in, and nothing here knows what an ``Action`` or an
``Implementation`` is. The agent runtime calls through here -- dependency method
calls in particular, which never have a fetched target to begin with.

The other half, which resolves a fetched ``Action``/``Implementation`` into
those ids, is :mod:`rekuest.invoke`. Keeping them apart is what lets
``agents.hooks.startup`` and ``actors.dependency`` import a call helper without
dragging the generated client surface into the agnostic runtime.
"""

import uuid
from typing import Any
from collections.abc import AsyncGenerator

from koil import unkoil, unkoil_gen
from arkitekt_spec.scalars import ID, coerce_id

from arkitekt_spec.actions import DefinitionInput
from arkitekt_runtime.types import (
    HookInput,
    TaskEventKind,
)
from arkitekt_spec.declare.errors import AgentLost, CriticalCallError, ErrorCallError
from arkitekt_runtime.messages import Assign, JSONSerializable
from arkitekt_runtime.postmans.types import Postman
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.actor import (
    aexpand_actor_returns,
    ashrink_actor_args,
)

__all__: list[str] = []


def ensure_return_as_tuple(value: Any) -> tuple[Any]:  # noqa: ANN401
    """Ensure that the value is a list."""
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value  # type: ignore
    return tuple([value])


def _resolve_parent(parent: Assign | None) -> ID | None:
    """The parent task id to attach this call to, as the socket wants it.

    Only what the caller passed: a :class:`~arkitekt_runtime.task.Task` passes its own assignment.
    """
    return coerce_id(parent.task) if parent is not None else None


async def _astream_raw(  # noqa: PLR0913 - the call description, mirrored from the protocol
    postman: Postman,
    *,
    args: dict[str, Any] | None = None,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    capture: bool = False,
    action_id: ID | None = None,
    implementation_id: ID | None = None,
    parent: ID | None = None,
    dependency: ID | None = None,
    method: str | None = None,
    escalate_to_interrupt: bool = False,
    cancel_timeout: float | None = None,
    call_key: str | None = None,
) -> AsyncGenerator[Any, None]:
    """Stream the YIELD payloads of a task, returning on DONE.

    The call is described by its arguments rather than by a pre-built payload: the
    GraphQL postman and the agent postman no longer share one (see
    :meth:`rekuest.postmans.types.Postman.aassign`), so each builds its own.

    Raises:
        ErrorCallError: If the backend reports a task error.
        CriticalCallError: If the backend reports a critical task error.
        AgentLost: If the agent running the task died while it ran.
        RootOnlyAssignError: If a ``parent``/``dependency``/``method`` call is routed
            to a postman that can only create root tasks.
    """
    async for i in postman.aassign(
        args=args or {},
        capture=capture,
        reference=reference or str(uuid.uuid4()),
        hooks=tuple(hooks or []),
        action=action_id,
        implementation=implementation_id,
        parent=parent,
        dependency=dependency,
        method=method,
        escalate_to_interrupt=escalate_to_interrupt,
        cancel_timeout=cancel_timeout,
        call_key=call_key,
    ):
        if i.kind == TaskEventKind.YIELD:
            yield i.returns

        if i.kind == TaskEventKind.COMPLETED:
            return

        if i.kind == TaskEventKind.FAILED:
            raise ErrorCallError(i.message)

        if i.kind == TaskEventKind.CRITICAL:
            raise CriticalCallError(i.message)

        # Not a failure: the agent running it died, and how it ended is unknown. Whoever
        # called decides, with what is known (``AgentLost.started`` and friends).
        if i.kind == TaskEventKind.LOST:
            raise AgentLost.from_details(getattr(i, "value", None), message=i.message)

        # CANCELLED and INTERRUPTED are terminal too. Somebody else (a user in the UI,
        # an interrupt cascading down a tree) can end a task this call is waiting on;
        # without these arms the stream simply never ended and the caller hung forever.
        # The agent-side postman surfaces them the same way (``agents.caller._adapt``).
        if i.kind in (TaskEventKind.CANCELLED, TaskEventKind.INTERRUPTED):
            raise CriticalCallError(
                i.message
                or f"The task was {i.kind.value.lower()} before it completed."
            )


async def acall_dependency_raw(
    dependency_key: ID,
    method: str,
    kwargs: dict[str, JSONSerializable],
    *,
    postman: Postman,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    parent: Assign | None = None,
    capture: bool = False,
) -> Any:  # noqa: ANN401 -- the raw backend payload
    """Call a method on a dependency with already serialized arguments.

    A dependency method call is never a root, so it can only be originated over the
    agent socket — which is the postman bound while an actor body runs. Calling it from
    outside a task raises ``RootOnlyAssignError``.

    ``cached`` and ``log`` are gone rather than accepted-and-ignored: the backend dropped
    both fields from ``AssignInput`` (``cached`` had already been documented there as
    having no effect — replay is decided caller-side via ``reusableTaskFor``), and a
    parameter that is deleted on the first line of the body only lies to its callers.
    """
    returns = tuple()

    async for r in _astream_raw(
        postman,
        args=kwargs,
        reference=reference,
        hooks=hooks,
        capture=capture,
        parent=_resolve_parent(parent),
        dependency=dependency_key,
        method=method,
    ):
        returns = r

    return returns


async def acall_dependency(
    definition: DefinitionInput,
    dependency_key: ID,
    method: str,
    *args: Any,  # noqa: ANN401 -- the method's own arguments
    postman: Postman,
    structure_registry: StructureRegistry,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    parent: Assign | None = None,
    capture: bool = False,
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> Any:  # noqa: ANN401 -- whatever the method returns, expanded
    """Call a method on a dependency and return expanded Python values."""

    shrinked_args = await ashrink_actor_args(
        definition, args, kwargs, structure_registry=structure_registry
    )

    raw_returns = await acall_dependency_raw(
        kwargs=shrinked_args,
        dependency_key=dependency_key,
        method=method,
        reference=reference,
        hooks=hooks,
        parent=parent,
        capture=capture,
        postman=postman,
    )

    returns = await aexpand_actor_returns(
        definition, raw_returns, structure_registry
    )
    if len(returns) == 1:
        return returns[0]
    return returns


def call_dependency(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Synchronously call a method on a dependency.

    Blocking counterpart to :func:`acall_dependency` (see there for
    parameters).
    """
    return unkoil(acall_dependency, *args, **kwargs)


async def aiterate_dependency(
    definition: DefinitionInput,
    dependency_key: ID,
    method: str,
    *args: Any,  # noqa: ANN401 -- the method's own arguments
    postman: Postman,
    structure_registry: StructureRegistry,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    parent: Assign | None = None,
    capture: bool = False,
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> AsyncGenerator[Any, None]:
    """Stream a generator method of a dependency, each yield expanded."""
    shrinked_args = await ashrink_actor_args(
        definition, args, kwargs, structure_registry=structure_registry
    )
    async for raw_returns in _astream_raw(
        postman,
        args=shrinked_args,
        reference=reference,
        hooks=hooks,
        capture=capture,
        parent=_resolve_parent(parent),
        dependency=dependency_key,
        method=method,
    ):
        returns = await aexpand_actor_returns(
            definition, raw_returns, structure_registry
        )
        yield returns[0] if len(returns) == 1 else returns


def iterate_dependency(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Blocking counterpart to :func:`aiterate_dependency` (see there for parameters)."""
    return unkoil_gen(aiterate_dependency, *args, **kwargs)
