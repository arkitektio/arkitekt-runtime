"""An untyped call carries structures by reference, and expands them on request."""

from types import SimpleNamespace
from typing import Any

from arkitekt_spec.actions import PortKind
from arkitekt_spec.declare.client import client
from arkitekt_spec.declare.structures.declare import structure

from arkitekt_runtime.ref import Ref, ato
from arkitekt_runtime.structures.serialization.postman import (
    aexpand_returns,
    ashrink_args,
)

IDENTIFIER = "@test/ref-picture"


class Picture:
    def __init__(self, id: str) -> None:
        self.id = id


@client(requires={})
class Pictures:
    async def get(self, id: str) -> Picture:
        return Picture(id)


@structure(IDENTIFIER)
async def expand_picture(id: str, pictures: Pictures) -> Picture:
    return await pictures.get(id)


class Run:
    """The least of a run: it hands out its clients."""

    def __init__(self) -> None:
        self.pictures = Pictures()

    def get(self, cls: type) -> object | None:
        return self.pictures if isinstance(self.pictures, cls) else None


def _port(key: str, kind: PortKind, **kwargs: object) -> SimpleNamespace:
    fields = {"identifier": None, "children": None, "nullable": False, "choices": None}
    return SimpleNamespace(key=key, kind=kind, default=None, **{**fields, **kwargs})


async def test_a_structure_travels_as_a_ref_and_expands_through_its_run() -> None:
    picture = _port("p", PortKind.STRUCTURE, identifier=IDENTIFIER)
    pictures = _port("ps", PortKind.LIST, children=[picture])
    definition: Any = SimpleNamespace(args=[picture], returns=[picture, pictures])
    run = Run()  # held: a Ref keeps only a weak reference to its run

    one, many = await aexpand_returns(
        definition,
        {
            "p": {"__identifier": IDENTIFIER, "object": "1"},
            "ps": [{"__identifier": IDENTIFIER, "object": "2"}],
        },
        structure_registry=None,
        origin=run,
    )

    assert one == Ref(identifier=IDENTIFIER, id="1")
    assert (await one.ato(Picture)).id == "1"
    assert [p.id for p in await ato(many, list[Picture])] == ["2"]

    wire = {"p": {"__identifier": IDENTIFIER, "object": "1"}}
    assert await ashrink_args(definition, [one], {}, structure_registry=None) == wire
    assert (
        await ashrink_args(definition, [Picture("1")], {}, structure_registry=None)
        == wire
    )
