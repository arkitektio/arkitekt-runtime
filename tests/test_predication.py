"""Tests for union-branch predication over ports."""


import pytest

from arkitekt_spec.actions import (
    ArgPortInput,
    PortKind,
)
from arkitekt_spec.declare.definition.define import prepare_definition
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.postman import ashrink_args
from arkitekt_runtime.structures.serialization.predication import predicate_port


def test_list_port_predicates_lists_not_dicts() -> None:
    """A LIST port matches a list of matching items and rejects a dict."""
    port = ArgPortInput(
        key="xs",
        kind=PortKind.LIST,
        nullable=False,
        children=(ArgPortInput(key="item", kind=PortKind.INT, nullable=False),),
    )
    registry = StructureRegistry()
    assert predicate_port(port, [1, 2, 3], registry) is True
    assert predicate_port(port, {"a": 1}, registry) is False
    assert predicate_port(port, ["a"], registry) is False


class Point:
    """A model, declared on the registry each test builds."""

    x: int
    y: int


def point_or_int(item: Point | int) -> int:
    """Accepts a Point or an int."""
    return 1


def list_or_str(item: list[int] | str) -> int:
    """Accepts a list of ints or a str."""
    return 1


@pytest.mark.asyncio
async def test_union_shrink_selects_model_branch() -> None:
    """A model item passed to a ``Model | int`` union picks the model branch."""
    registry = StructureRegistry()
    registry.model(Point)
    definition = prepare_definition(point_or_int, structure_registry=registry)

    shrunk = await ashrink_args(
        definition, (Point(x=1, y=2),), {}, structure_registry=registry
    )
    assert shrunk["item"]["__use"] == 0
    assert shrunk["item"]["__value"]["x"] == 1

    shrunk = await ashrink_args(definition, (7,), {}, structure_registry=registry)
    assert shrunk["item"] == {"__use": 1, "__value": 7}


@pytest.mark.asyncio
async def test_union_shrink_selects_list_branch() -> None:
    """A list item passed to a ``List[int] | str`` union picks the list branch."""
    registry = StructureRegistry()
    registry.model(Point)
    definition = prepare_definition(list_or_str, structure_registry=registry)

    shrunk = await ashrink_args(definition, ([1, 2],), {}, structure_registry=registry)
    assert shrunk["item"] == {"__use": 0, "__value": [1, 2]}

    shrunk = await ashrink_args(definition, ("s",), {}, structure_registry=registry)
    assert shrunk["item"] == {"__use": 1, "__value": "s"}
