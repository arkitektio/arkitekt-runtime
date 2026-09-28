"""Test that typing.Literal annotations are autoconverted to enum ports."""

from enum import Enum
from typing import Literal

import pytest

from arkitekt_runtime.actors.types import Shelver
from arkitekt_spec.actions import PortKind
from arkitekt_spec.declare.definition.define import prepare_definition
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_runtime.structures.serialization.actor import expand_inputs, shrink_outputs


def literal_arg_function(x: Literal["a", "b", "c"]) -> str:
    """A function with a literal argument."""
    return x


def literal_default_function(x: Literal["a", "b", "c"] = "b") -> str:
    """A function with a literal argument that has a default."""
    return x


def literal_int_function(x: Literal[1, 2, 3]) -> str:
    """A function with an int literal argument."""
    return str(x)


def literal_return_function(x: int) -> Literal["a", "b", "c"]:
    """A function returning a literal."""
    return "a"


def test_literal_arg_becomes_enum_port(simple_registry: StructureRegistry) -> None:
    """A bare Literal arg should produce an ENUM port with the right choices."""
    definition = prepare_definition(
        literal_arg_function, structure_registry=simple_registry
    )
    port = definition.args[0]
    assert port.kind == PortKind.ENUM
    assert [choice.value for choice in port.choices] == ["a", "b", "c"]


def test_literal_default_becomes_enum_port(simple_registry: StructureRegistry) -> None:
    """A Literal with a string default must not be mistaken for a STRING port."""
    definition = prepare_definition(
        literal_default_function, structure_registry=simple_registry
    )
    port = definition.args[0]
    assert port.kind == PortKind.ENUM
    assert port.default == "b"


def test_literal_int_becomes_enum_port(simple_registry: StructureRegistry) -> None:
    """Int literals are autoconverted too (and not treated as plain INT)."""
    definition = prepare_definition(
        literal_int_function, structure_registry=simple_registry
    )
    port = definition.args[0]
    assert port.kind == PortKind.ENUM
    assert [choice.value for choice in port.choices] == ["1", "2", "3"]


def test_literal_return_becomes_enum_port(simple_registry: StructureRegistry) -> None:
    """Literals in the return annotation are autoconverted too."""
    definition = prepare_definition(
        literal_return_function, structure_registry=simple_registry
    )
    port = definition.returns[0]
    assert port.kind == PortKind.ENUM
    assert [choice.value for choice in port.choices] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_literal_roundtrip(
    simple_registry: StructureRegistry, mock_shelver: Shelver
) -> None:
    """Expand a wire value into a literal arg, then shrink it back."""
    definition = prepare_definition(
        literal_return_function, structure_registry=simple_registry
    )

    args = await expand_inputs(
        definition,
        {"x": 5},
        structure_registry=simple_registry,
        shelver=mock_shelver,
    )
    assert args["x"] == 5

    shrunk = await shrink_outputs(
        definition,
        literal_return_function(5),
        structure_registry=simple_registry,
        shelver=mock_shelver,
    )
    assert shrunk["return0"] == "a"


@pytest.mark.asyncio
async def test_literal_arg_expand_roundtrip(
    simple_registry: StructureRegistry, mock_shelver: Shelver
) -> None:
    """A wire choice for a literal arg expands to a value equal to that choice."""
    definition = prepare_definition(
        literal_arg_function, structure_registry=simple_registry
    )

    args = await expand_inputs(
        definition,
        {"x": "c"},
        structure_registry=simple_registry,
        shelver=mock_shelver,
    )
    # str-based enum member compares and stringifies as the bare literal value.
    assert args["x"] == "c"
    assert str(args["x"]) == "c"


def test_same_literal_shares_identifier(simple_registry: StructureRegistry) -> None:
    """The same Literal used twice resolves to the same enum identifier, and the
    identifier has the server's ``@package/key`` form."""
    definition_a = prepare_definition(
        literal_arg_function, structure_registry=simple_registry
    )
    definition_b = prepare_definition(
        literal_default_function, structure_registry=simple_registry
    )
    assert definition_a.args[0].identifier == "@literal/a_b_c"
    assert definition_a.args[0].identifier == definition_b.args[0].identifier


def test_literal_is_its_own_port_class(simple_registry: StructureRegistry) -> None:
    """A Literal stays a Literal: no enum class is minted, the annotation is the key."""
    annotation = Literal["a", "b", "c"]
    prepare_definition(literal_arg_function, structure_registry=simple_registry)

    fenum = simple_registry.get_fullfilled_type_for_cls(annotation)
    assert fenum.cls is annotation
    assert fenum.members == {"a": "a", "b": "b", "c": "c"}
    keys = [
        key
        for key, value in simple_registry.cls_fullfilled_type_map.items()
        if value is fenum
    ]
    assert keys == [annotation]


def test_literal_registry_survives_copy(simple_registry: StructureRegistry) -> None:
    """Copying a registry that derived a Literal must not trip pydantic.

    This is the path ``App.snapshot`` takes at ``run(app)``; it used to fail
    with ``Input should be a type`` on the Literal key.
    """
    prepare_definition(literal_arg_function, structure_registry=simple_registry)
    prepare_definition(literal_int_function, structure_registry=simple_registry)

    copied = simple_registry.copy_maps()
    assert copied.find_for_cls(Literal["a", "b", "c"]) is simple_registry.find_for_cls(
        Literal["a", "b", "c"]
    )
    assert copied.bound({}).find_for_cls(Literal[1, 2, 3]) is not None


@pytest.mark.asyncio
async def test_int_literal_expands_to_bare_int(
    simple_registry: StructureRegistry, mock_shelver: Shelver
) -> None:
    """A wire choice for an int literal expands to the int itself, from str or int."""
    definition = prepare_definition(
        literal_int_function, structure_registry=simple_registry
    )
    for wire in ("2", 2):
        args = await expand_inputs(
            definition,
            {"x": wire},
            structure_registry=simple_registry,
            shelver=mock_shelver,
        )
        assert args["x"] == 2
        assert type(args["x"]) is int


def test_literal_predicate_is_typed(simple_registry: StructureRegistry) -> None:
    """``True`` is not a member of ``Literal[1, 2, 3]`` even though ``True == 1``."""
    prepare_definition(literal_int_function, structure_registry=simple_registry)
    fenum = simple_registry.get_fullfilled_type_for_cls(Literal[1, 2, 3])
    assert fenum.predicate(1)
    assert not fenum.predicate(True)
    assert not fenum.predicate("1")


class Colour(Enum):
    """A colour."""

    RED = "red"
    BLUE = "blue"


def colour_function(colour: Colour) -> str:
    """A function with an enum argument."""
    return colour.value


def test_enum_port_names_its_enum(simple_registry: StructureRegistry) -> None:
    """An Enum class port carries its choices and names the enum they came from."""
    definition = prepare_definition(colour_function, structure_registry=simple_registry)
    port = definition.args[0]
    assert port.kind == PortKind.ENUM
    assert port.identifier == "@tests.test_literal_enums/colour"
    assert [choice.value for choice in port.choices] == ["RED", "BLUE"]
    assert simple_registry.get_fullfilled_enum(port.identifier).cls is Colour


@pytest.mark.asyncio
async def test_enum_port_expands_to_member(
    simple_registry: StructureRegistry, mock_shelver: Shelver
) -> None:
    """The actor hands the function the member its annotation names."""
    definition = prepare_definition(colour_function, structure_registry=simple_registry)
    args = await expand_inputs(
        definition,
        {"colour": "BLUE"},
        structure_registry=simple_registry,
        shelver=mock_shelver,
    )
    assert args["colour"] is Colour.BLUE
