"""Calling a fetched ``Action`` or ``Implementation``.

The half of the call surface that holds a fetched model and reads ``.id`` off it;
:mod:`rekuest.calls` is the half that only ever sees ids. The split is load-bearing:
``rekuest.calls`` is importable without pulling in the generated client surface, and the
agnostic runtime (:mod:`rekuest.actors.dependency`, :mod:`rekuest.agents.hooks.startup`)
relies on that. This module names neither: ``tests/test_layering.py`` reserves ``rekuest.api.schema``
for ``rekuest/client/``, so the target is described structurally and the raw entry points
take plain ids.

**Neither half knows a client.** A :class:`~rekuest.client.client.Rekuest` calls in here
with its own postman and registry, a :class:`~arkitekt_runtime.task.Task` with its agent's --
which is what lets :mod:`arkitekt_runtime.task` use this while still knowing no service.

``postman`` and ``structure_registry`` are required keyword arguments, not defaulted
options. They were once ``None``-defaulted and passed to a resolver that fell back to a
contextvar; the contextvar is gone, so the only thing an omitted argument could produce
was a deferred error. Every entry point is private: what is public is the two classes.
"""

from collections.abc import AsyncGenerator
from typing import Any, Protocol, cast, runtime_checkable

from arkitekt_runtime.calls import _astream_raw, _resolve_parent
from arkitekt_runtime.messages import Assign
from arkitekt_runtime.postmans.types import Postman
from arkitekt_spec.declare.task import HookInput
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.postman import aexpand_returns, ashrink_args
from arkitekt_spec.declare.targets import CallTarget, ImplementationTarget
from arkitekt_spec.declare.structures.types import JSONSerializable

__all__: list[str] = []


@runtime_checkable
class _Identified(Protocol):
    """The least a target must have: a name the socket knows it by."""

    @property
    def id(self) -> Any:  # noqa: ANN401 -- an ID scalar
        """What the socket calls this."""
        ...


def _resolve_target(
    target: "CallTarget | ImplementationTarget",
) -> tuple[CallTarget, Any]:
    """Resolve a target into ``(action, implementation_id)``.

    Gated on ``id`` rather than on the full :class:`CallTarget`: a partly built model --
    ``Action.model_construct(id=...)``, which the tests use -- carries no ports, and
    refusing it here would report "not an action" about something that is one. Missing
    ports are the serializer's complaint to make, against the real port list.
    """
    if isinstance(target, ImplementationTarget):
        return target.action, target.id
    if isinstance(target, _Identified):
        return cast("CallTarget", target), None
    raise ValueError(
        "A call target is an Action or an Implementation (something the socket can name "
        f"by id); got {type(target).__name__}"
    )


async def _aiterate_raw(
    *,
    postman: Postman,
    kwargs: dict[str, JSONSerializable] | None = None,
    action_id: Any = None,  # noqa: ANN401 -- an ID scalar
    implementation_id: Any = None,  # noqa: ANN401 -- an ID scalar
    parent: Assign | None = None,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    capture: bool = False,
    escalate_to_interrupt: bool = False,
    cancel_timeout: float | None = None,
    call_key: str | None = None,
) -> AsyncGenerator[Any, None]:
    """Stream the raw YIELD payloads of a remote call.

    Operates on already-serialized arguments and yields transport-level payloads;
    prefer :func:`_aiterate` unless you are deliberately working at that level.
    """
    async for returns in _astream_raw(
        postman,
        args=kwargs,
        reference=reference,
        hooks=hooks,
        capture=capture,
        action_id=action_id,
        implementation_id=implementation_id,
        parent=_resolve_parent(parent),
        escalate_to_interrupt=escalate_to_interrupt,
        cancel_timeout=cancel_timeout,
        call_key=call_key,
    ):
        yield returns


async def _acall_raw(
    *,
    postman: Postman,
    kwargs: dict[str, JSONSerializable] | None = None,
    action_id: Any = None,  # noqa: ANN401 -- an ID scalar
    implementation_id: Any = None,  # noqa: ANN401 -- an ID scalar
    parent: Assign | None = None,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    capture: bool = False,
    escalate_to_interrupt: bool = False,
    cancel_timeout: float | None = None,
    call_key: str | None = None,
) -> Any:  # noqa: ANN401 -- the raw backend payload, whatever the action returned
    """Run a call with already-serialized arguments and return the final YIELD payload.

    Raises:
        ErrorCallError: If the backend reports a recoverable task error.
        CriticalCallError: If the backend reports a critical task error.
    """
    returns = tuple()
    async for r in _aiterate_raw(
        postman=postman,
        kwargs=kwargs,
        action_id=action_id,
        implementation_id=implementation_id,
        parent=parent,
        reference=reference,
        hooks=hooks,
        capture=capture,
        escalate_to_interrupt=escalate_to_interrupt,
        cancel_timeout=cancel_timeout,
        call_key=call_key,
    ):
        returns = r
    return returns


async def _acall(
    target: "CallTarget | ImplementationTarget",
    *args: Any,  # noqa: ANN401 -- the action's own arguments, shrunk by its registry
    postman: Postman,
    structure_registry: StructureRegistry,
    parent: Assign | None = None,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    capture: bool = False,
    escalate_to_interrupt: bool = False,
    cancel_timeout: float | None = None,
    call_key: str | None = None,
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> Any:  # noqa: ANN401 -- whatever the action returns, expanded
    """Call an action and return its expanded result.

    Raises:
        ValueError: If the target is neither an action nor an implementation.
        ErrorCallError: If the backend reports a task error.
        CriticalCallError: If the backend reports a critical task error.
    """
    action, implementation_id = _resolve_target(target)

    shrinked_args = await ashrink_args(
        action, args, kwargs, structure_registry=structure_registry
    )
    raw_returns = await _acall_raw(
        postman=postman,
        kwargs=shrinked_args,
        action_id=action.id,
        implementation_id=implementation_id,
        parent=parent,
        reference=reference,
        hooks=hooks,
        capture=capture,
        escalate_to_interrupt=escalate_to_interrupt,
        cancel_timeout=cancel_timeout,
        call_key=call_key,
    )
    returns = await aexpand_returns(
        action, raw_returns, structure_registry=structure_registry
    )
    if len(returns) == 1:
        return returns[0]
    return returns


async def _aiterate(
    target: "CallTarget | ImplementationTarget",
    *args: Any,  # noqa: ANN401 -- the action's own arguments
    postman: Postman,
    structure_registry: StructureRegistry,
    parent: Assign | None = None,
    reference: str | None = None,
    hooks: list[HookInput] | None = None,
    capture: bool = False,
    escalate_to_interrupt: bool = False,
    cancel_timeout: float | None = None,
    call_key: str | None = None,
    **kwargs: Any,  # noqa: ANN401 -- ditto, by keyword
) -> AsyncGenerator[Any, None]:
    """Stream a generator action's yields, each one expanded."""
    action, implementation_id = _resolve_target(target)

    shrinked_args = await ashrink_args(
        action, args, kwargs, structure_registry=structure_registry
    )
    async for raw_returns in _aiterate_raw(
        postman=postman,
        kwargs=shrinked_args,
        action_id=action.id,
        implementation_id=implementation_id,
        parent=parent,
        reference=reference,
        hooks=hooks,
        capture=capture,
        escalate_to_interrupt=escalate_to_interrupt,
        cancel_timeout=cancel_timeout,
        call_key=call_key,
    ):
        returns = await aexpand_returns(
            action, raw_returns, structure_registry=structure_registry
        )
        if len(returns) == 1:
            yield returns[0]
        else:
            yield returns
