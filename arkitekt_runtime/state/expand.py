"""Expand a shrunk state using a schema and a structure registry"""

from typing import Any

from arkitekt_spec.actions import StateDefinitionInput
from arkitekt_spec.declare.protocol.types import AnyState
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.expand import aexpand_return


async def aexpand_state(
    cls: Any,  # noqa: ANN401
    shrunk: dict[str, Any],
    schema: StateDefinitionInput,
    structure_reg: StructureRegistry,
) -> AnyState:
    """Expand a shrunk state back into an instance of its class: the mirror of
    :func:`~arkitekt_runtime.state.shrink.ashrink_state`.

    Args:
        cls (type): The state class
        shrunk (Dict[str, Any]): The shrunk state
        schema (StateDefinitionInput): The schema to use (defines the ports)
        structure_reg (StructureRegistry): The structure registry to use

    Returns:
        Any: The state, a plain instance (an agent makes it evented when it adopts one)

    """
    values: dict[str, Any] = {}
    for port in schema.ports:
        values[port.key] = await aexpand_return(port, shrunk[port.key], structure_reg)

    return cls(**values)
