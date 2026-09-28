"""Just a protocol for the serialization of ports."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from arkitekt_spec.actions import PortKind


@runtime_checkable
class SerializablePort(Protocol):
    """Structural type for the ports the serialization layer walks over.

    The generated GraphQL port models (``ArgPort``, ``ReturnPort`` and their
    ``ChildPort``/``Nested`` variants) form a depth-limited hierarchy, where the
    deepest leaf nodes drop ``children``/``choices``/``default``. The
    serialization routines recurse over ``children`` uniformly, so this protocol
    captures the attributes they actually read. ``children``/``choices`` are
    typed ``Any`` so the recursion type-checks regardless of the concrete
    nesting depth.

    ``nullable`` is widened to ``bool | None`` because the *Input* spelling of a
    port has a GraphQL default and is therefore optional; every reader treats it
    as a truth value.

    ``default`` is deliberately *not* a member: only arg ports carry one, so
    requiring it here made every ``ReturnPort`` fail the protocol. The one place
    that reads it (``shrink.ashrink_args``) asks for it defensively.

    Every member is a read-only property: a mutable protocol attribute is
    invariant, and the output spelling of a port (``ArgPort.nullable: bool``) and
    the input spelling (``ArgPortInput.nullable: bool | None``) could then never
    both satisfy it.
    """

    @property
    def kind(self) -> PortKind:
        """What the port carries."""
        ...

    @property
    def key(self) -> str:
        """The port's name in the args/returns mapping."""
        ...

    @property
    def nullable(self) -> bool | None:
        """Whether a missing value is allowed. Read as a truth value."""
        ...

    @property
    def identifier(self) -> Any:  # noqa: ANN401
        """The structure identifier, for STRUCTURE ports."""
        ...

    @property
    def children(self) -> Any:  # noqa: ANN401
        """Nested ports, one level down. ``Any``: the depth is bounded by the schema."""
        ...

    @property
    def choices(self) -> Any:  # noqa: ANN401
        """The allowed values, for ports that enumerate them."""
        ...

    @property
    def reference_unit(self) -> str | None:
        """The unit a QUANTITY port is carried in."""
        ...

    @property
    def dimension(self) -> str | None:
        """The physical dimension a QUANTITY port belongs to."""
        ...


@runtime_checkable
class SerializableDefinition(Protocol):
    """Structural type for the things ``ashrink_args``/``aexpand_returns`` walk.

    Both a ``DefinitionInput`` on its way to the server and an ``Action`` fetched
    back from it carry the same two port lists, and the serialization routines
    read nothing else. Saying so structurally is what keeps this layer from
    importing the generated client surface: ``Action`` is a fragment, and a
    fragment belongs to ``rekuest.client``.

    Read-only properties rather than attributes: a mutable protocol attribute is
    invariant, so ``tuple[ArgPort, ...]`` would not satisfy
    ``Sequence[SerializablePort]``.
    """

    @property
    def args(self) -> Sequence[SerializablePort] | None:
        """The input ports, in call order."""
        ...

    @property
    def returns(self) -> Sequence[SerializablePort] | None:
        """The output ports, in return order."""
        ...
