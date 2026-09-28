"""Shrink a state using a schema and a structure registry"""

from typing import Any
from arkitekt_runtime.actors.types import Shelver
from arkitekt_spec.actions import StateDefinitionInput
from arkitekt_runtime.messages import JSONSerializable
from arkitekt_spec.declare.protocol.types import AnyState
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.actor import ashrink_return


async def ashrink_state(
    state: AnyState,  # noqa: ANN401
    schema: StateDefinitionInput,
    structure_reg: StructureRegistry,  # noqa: ANN401
    shelver: Shelver,
) -> dict[str, Any]:
    """Shrink a state  using a schema and a structure registry

    Args:
        state (Any): The state to shrink
        schema (StateDefinitionInput): The schema to use (defines the ports)
        structure_reg (StructureRegistry): The structure registry to use

    Returns:
        Dict[str, Any]: The shrunk state

    """

    shrinked: dict[str, JSONSerializable] = {}
    for port in schema.ports:
        shrinked[port.key] = await ashrink_return(
            port, getattr(state, port.key), structure_reg, shelver=shelver
        )

    return shrinked
