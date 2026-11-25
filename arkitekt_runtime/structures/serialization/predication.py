"""Predication: decide whether a Python value fits a port.

Used to pick the matching branch of a ``UNION`` port before shrinking.
"""

import datetime as dt
from enum import Enum
from typing import Any

from arkitekt_spec.actions import PortKind
from arkitekt_spec.declare.structures.quantities import matches_dimension
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.targets import SerializablePort


def _single_child(port: SerializablePort) -> SerializablePort:
    if not port.children or len(port.children) != 1:
        raise ValueError(f"Port {port.identifier} must have exactly one child")
    return port.children[0]


def _require_identifier(port: SerializablePort) -> str:
    if not port.identifier:
        raise ValueError(f"Port {port} has no identifier")
    return port.identifier


def predicate_port(
    port: SerializablePort,
    value: Any,  # noqa: ANN401
    structure_registry: StructureRegistry | None,
) -> bool:
    """Check whether ``value`` is of the type described by ``port``.

    Without a registry (an untyped call) a structure is told by its reference or its
    class's mark, a model by being a dict, an enum by being one of the choices.

    Container kinds recurse into their children; registered kinds
    (structure / model / memory structure / enum) defer to the predicate the
    registry recorded for the identifier.

    Raises:
        ValueError: If the port is malformed (missing identifier / children) or
            of an unknown kind.
    """
    kind = port.kind
    if kind == PortKind.DICT:
        if not isinstance(value, dict):
            return False
        child = _single_child(port)
        return all(
            predicate_port(child, item, structure_registry)
            for item in value.values()  # type: ignore[union-attr]
        )
    if kind == PortKind.LIST:
        if not isinstance(value, list):
            return False
        child = _single_child(port)
        return all(
            predicate_port(child, item, structure_registry)
            for item in value  # type: ignore[union-attr]
        )
    if kind == PortKind.DATE:
        return isinstance(value, dt.datetime)
    if kind == PortKind.INT:
        # bool is an int subclass, but True belongs to a BOOL variant.
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == PortKind.FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == PortKind.BOOL:
        return isinstance(value, bool)
    if kind == PortKind.STRING:
        return isinstance(value, str)
    if kind == PortKind.QUANTITY:
        return matches_dimension(value, port.dimension)
    if structure_registry is None:
        return _by_reference(port, value)
    if kind == PortKind.STRUCTURE and _is_ref(value):
        return value.identifier == port.identifier
    if kind == PortKind.STRUCTURE:
        return structure_registry.get_fullfilled_structure(
            _require_identifier(port)
        ).predicate(value)
    if kind == PortKind.MODEL:
        identifier = _require_identifier(port)
        try:
            fmodel = structure_registry.get_fullfilled_model(identifier)
        except KeyError:
            # Unregistered model: fall back to a structural check over children.
            if not port.children:
                raise ValueError(f"Port {identifier} has no children") from None
            return all(
                hasattr(value, child.key)
                and predicate_port(child, getattr(value, child.key), structure_registry)
                for child in port.children
            )
        return fmodel.predicate(value)
    if kind == PortKind.MEMORY_STRUCTURE:
        return structure_registry.get_fullfilled_memory_structure(
            _require_identifier(port)
        ).predicate(value)
    if kind == PortKind.ENUM:
        return structure_registry.get_fullfilled_enum(
            _require_identifier(port)
        ).predicate(value)

    raise ValueError(f"Unknown port kind: {port.kind} to predicate")


def _is_ref(value: object) -> bool:
    from arkitekt_runtime.ref import Ref

    return isinstance(value, Ref)


def _by_reference(port: SerializablePort, value: Any) -> bool:  # noqa: ANN401
    from arkitekt_spec.declare.structures.declare import structure_of

    from arkitekt_runtime.ref import Ref

    if port.kind in (PortKind.STRUCTURE, PortKind.MEMORY_STRUCTURE):
        if isinstance(value, str):
            return True
        if isinstance(value, Ref):
            return value.identifier == port.identifier
        declared = structure_of(type(value))
        return declared is not None and declared.identifier == port.identifier
    if port.kind == PortKind.MODEL:
        keys = [child.key for child in port.children or ()]
        if isinstance(value, dict):
            return all(key in value for key in keys)
        return all(hasattr(value, key) for key in keys)
    if port.kind == PortKind.ENUM:
        said = value.name if isinstance(value, Enum) else str(value)
        return any(said == choice.value for choice in port.choices or ())
    raise ValueError(f"Unknown port kind: {port.kind} to predicate")


# Backwards-compatible names; both used to be separate, drifting copies.
predicate_port_input = predicate_port
predicate_serializable_port = predicate_port
