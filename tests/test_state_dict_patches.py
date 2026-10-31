"""A change inside a map is a patch like any other.

A map's port has one child that describes every value, as a list's does. A key of a map is no field
of it, so a change below one has to find that child, or the patch has no port to shrink its value by
and is lost.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from arkitekt_runtime.state.observable import adopt, evented
from arkitekt_runtime.structures.serialization.actor import ashrink_return
from arkitekt_spec.declare.app import AppRegistry
from tests.conftest import MockShelver


@dataclass
class Move:
    value: float = 0.0
    unit: str = "um"


@dataclass
class Moves:
    by_name: dict[str, Move] = field(default_factory=dict)
    by_cube: dict[str, dict[str, Move]] = field(default_factory=dict)
    in_lists: list[dict[str, Move]] = field(default_factory=list)
    of_lists: dict[str, list[Move]] = field(default_factory=dict)
    numbers: dict[str, float] = field(default_factory=dict)


class Recorder:
    def __init__(self) -> None:
        self.patches: list[Any] = []

    def publish_patch(self, interface: str, patch: Any) -> None:  # noqa: ANN401
        self.patches.append(patch)


def adopted(state: Moves) -> tuple[Moves, Recorder, Any]:
    registry = AppRegistry()
    registry.model(Move)
    registry.state(Moves)
    structures = registry.snapshot().structure_registry
    made, recorder = evented(state, structures.state_for(Moves)), Recorder()
    adopt(made, recorder)
    return made, recorder, structures


def shrunk(patch: Any, structures: Any) -> Any:  # noqa: ANN401
    """What the agent would send: it shrinks a patch's value by the patch's port."""
    assert patch.port is not None, f"no port for {patch.path}"
    return asyncio.run(ashrink_return(patch.port, patch.value, structures, MockShelver()))


def test_a_field_of_a_value_in_a_map() -> None:
    state, recorder, structures = adopted(Moves(by_name={"x": Move()}))
    state.by_name["x"].value = 3.0
    (patch,) = recorder.patches
    assert (patch.op, patch.path) == ("replace", "/by_name/x/value")
    assert shrunk(patch, structures) == 3.0


def test_a_field_two_maps_down() -> None:
    state, recorder, structures = adopted(Moves(by_cube={"stage": {"z": Move()}}))
    state.by_cube["stage"]["z"].unit = "mm"
    (patch,) = recorder.patches
    assert patch.path == "/by_cube/stage/z/unit"
    assert shrunk(patch, structures) == "mm"


def test_a_key_that_is_also_the_name_of_a_field_means_nothing() -> None:
    """Keys are data: `value` as a key is a `Move`, not the float the field `value` is."""
    state, recorder, structures = adopted(Moves(by_name={"value": Move()}))
    state.by_name["value"].value = 3.0
    (patch,) = recorder.patches
    assert patch.path == "/by_name/value/value"
    assert shrunk(patch, structures) == 3.0


def test_a_new_entry_of_a_map_is_shrunk_as_a_value_of_it() -> None:
    state, recorder, structures = adopted(Moves())
    state.by_name["x"] = Move(value=2.0)
    (patch,) = recorder.patches
    assert (patch.op, patch.path) == ("add", "/by_name/x")
    assert shrunk(patch, structures)["value"] == 2.0


def test_a_new_entry_goes_on_publishing() -> None:
    """A value put into a map later changes like one that was there from the start."""
    state, recorder, structures = adopted(Moves())
    state.by_name["x"] = Move()
    state.by_name["x"].value = 4.0
    assert recorder.patches[-1].path == "/by_name/x/value"
    assert shrunk(recorder.patches[-1], structures) == 4.0


def test_a_replaced_map_of_a_map() -> None:
    state, recorder, structures = adopted(Moves(by_cube={"stage": {}}))
    state.by_cube["stage"] = {"z": Move(value=1.0)}
    (patch,) = recorder.patches
    assert patch.path == "/by_cube/stage"
    assert shrunk(patch, structures)["z"]["value"] == 1.0


@pytest.mark.parametrize(
    ("start", "change", "path", "expected"),
    [
        (
            Moves(in_lists=[{"x": Move()}]),
            lambda s: setattr(s.in_lists[0]["x"], "value", 5.0),
            "/in_lists/0/x/value",
            5.0,
        ),
        (
            Moves(of_lists={"x": [Move()]}),
            lambda s: setattr(s.of_lists["x"][0], "value", 5.0),
            "/of_lists/x/0/value",
            5.0,
        ),
        (
            Moves(numbers={"x": 1.0}),
            lambda s: s.numbers.__setitem__("x", 5.0),
            "/numbers/x",
            5.0,
        ),
    ],
    ids=["a map in a list", "a list in a map", "a number in a map"],
)
def test_maps_and_lists_nest(start: Moves, change: Any, path: str, expected: float) -> None:
    state, recorder, structures = adopted(start)
    change(state)
    (patch,) = recorder.patches
    assert patch.path == path
    assert shrunk(patch, structures) == expected
