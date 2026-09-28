"""A task changes a state through its own view: its changes carry its id and its locks.

No context variable says which task is changing a state; the view the task was
handed does.
"""

import dataclasses
import gc
import threading

import pytest

from arkitekt_spec.actions import StateDefinitionInput
from arkitekt_runtime.state.observable import Mutation, StateConfig, adopt, make_evented
from arkitekt_runtime.state.write import write_view


@dataclasses.dataclass
class Board:
    count: int = 0
    items: list = dataclasses.field(default_factory=list)
    meta: dict = dataclasses.field(default_factory=dict)


class Recorder:
    def __init__(self) -> None:
        self.patches: list[tuple[str, str, str | None]] = []

    def publish_patch(self, interface: str, patch: object) -> None:
        self.patches.append((patch.op, patch.path, patch.correlation_id))  # type: ignore[attr-defined]


def board(required_locks: list[str] | None = None) -> Board:
    config = StateConfig(
        state_name="Board",
        definition=StateDefinitionInput(ports=(), name="Board"),
        required_locks=required_locks or [],
    )
    return make_evented(Board(), config, "")


def test_changes_through_a_view_carry_the_task_and_stay_the_state_type() -> None:
    state, recorder = board(), Recorder()
    adopt(state, recorder)

    view = write_view(state, Mutation(correlation_id="t1"))
    view.count = 1
    view.items.append("a")
    view.meta["k"] = "v"

    assert isinstance(view, Board)
    assert (state.count, list(state.items), dict(state.meta)) == (1, ["a"], {"k": "v"})
    assert recorder.patches == [
        ("replace", "/count", "t1"),
        ("add", "/items/0", "t1"),
        ("add", "/meta/k", "t1"),
    ]


def test_two_tasks_views_of_one_state_stay_apart() -> None:
    state, recorder = board(), Recorder()
    adopt(state, recorder)
    one, two = (write_view(state, Mutation(correlation_id=t)) for t in ("t1", "t2"))

    def write(view: Board, n: int) -> None:
        for _ in range(50):
            view.items.append(n)

    threads = [threading.Thread(target=write, args=(v, n)) for v, n in ((one, 1), (two, 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    by_task = {c: [p for p in recorder.patches if p[2] == c] for c in ("t1", "t2")}
    assert len(by_task["t1"]) == len(by_task["t2"]) == 50
    assert len(state.items) == 100


def test_the_locks_a_view_holds_are_the_ones_checked() -> None:
    state = board(required_locks=["stage"])
    adopt(state, Recorder())

    write_view(state, Mutation(correlation_id="t", locks=frozenset({"stage"}))).count = 1
    with pytest.raises(RuntimeError, match="without required locks"):
        write_view(state, Mutation(correlation_id="t", locks=frozenset())).count = 2
    with pytest.raises(RuntimeError, match="without required locks"):
        state.count = 3  # adopted: outside a task no locks are held
    assert state.count == 1


def test_a_state_is_built_freely_before_it_is_adopted() -> None:
    state, recorder = board(required_locks=["stage"]), Recorder()
    state.count = 5  # the startup hook building it
    adopt(state, recorder)
    assert state.count == 5 and recorder.patches == []


def test_a_view_leaves_nothing_behind() -> None:
    state = board()
    view = write_view(state, Mutation(correlation_id="t"))
    view.count = 1
    del view
    gc.collect()
    assert state.__dict__["__rekuest_mutations__"] == {}
