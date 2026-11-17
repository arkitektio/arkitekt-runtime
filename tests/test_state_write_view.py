"""A task changes a state through its own view: its changes carry its id.

No context variable says which task is changing a state; the view the task was
handed does.
"""

import dataclasses
import gc
import threading


from arkitekt_spec.actions import StateDefinitionInput
from arkitekt_runtime.state.observable import (
    EventedDict,
    Mutation,
    StateConfig,
    adopt,
    make_evented,
)
from arkitekt_runtime.state.write import write_view


@dataclasses.dataclass
class Board:
    count: int = 0
    items: list = dataclasses.field(default_factory=list)
    meta: dict = dataclasses.field(default_factory=dict)
    other: dict = dataclasses.field(default_factory=dict)


class Recorder:
    def __init__(self) -> None:
        self.patches: list[tuple[str, str, str | None]] = []

    def publish_patch(self, interface: str, patch: object) -> None:
        self.patches.append((patch.op, patch.path, patch.correlation_id))  # type: ignore[attr-defined]


def board() -> Board:
    config = StateConfig(
        state_name="Board",
        definition=StateDefinitionInput(ports=(), name="Board"),
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


def test_one_field_assigned_to_another_stores_a_copy_not_the_view() -> None:
    """``state.a = state.b`` reads ``b`` as the view's wrapper: what is stored is a dict
    of ``a``'s own, which the next read and the next patch find where they expect it."""
    state, recorder = board(), Recorder()
    adopt(state, recorder)
    view = write_view(state, Mutation(correlation_id="t1"))
    view.meta["k"] = "v"

    view.other = view.meta
    view.other["later"] = 1

    assert type(state.other) is EventedDict and state.other is not state.meta
    assert (dict(state.other), dict(state.meta)) == ({"k": "v", "later": 1}, {"k": "v"})
    assert recorder.patches[1:] == [
        ("replace", "/other", "t1"),
        ("add", "/other/later", "t1"),
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


def test_locks_do_not_guard_writes() -> None:
    """A lock keeps two tasks from commanding a thing at once; it does not say who may
    write the state. A manager or a background hook writes it directly, uncorrelated."""
    state, recorder = board(), Recorder()
    adopt(state, recorder)

    write_view(state, Mutation(correlation_id="t")).count = 1
    state.count = 2  # outside any task
    assert state.count == 2
    assert recorder.patches == [("replace", "/count", "t"), ("replace", "/count", None)]


def test_a_state_is_built_before_it_is_adopted_without_patches() -> None:
    state, recorder = board(), Recorder()
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


def test_the_instance_a_startup_hook_returned_is_the_published_state() -> None:
    """A host keeps what its startup hook returned and writes it from its own threads.

    What a device reports by itself -- a joystick, a drifting stage -- has no task and
    no action, so the instance itself has to be writable from wherever it is noticed.
    """
    returned = Board()
    state = make_evented(returned, config_of(), "")
    recorder = Recorder()
    adopt(state, recorder)
    assert state is returned, "adopted in place: the host's reference stays the live one"

    def from_the_hosts_thread() -> None:
        returned.count = 5
        returned.meta["pump"] = 2.5
        returned.count = 5  # unchanged: publishes nothing

    thread = threading.Thread(target=from_the_hosts_thread)
    thread.start()
    thread.join()

    assert [(op, path) for op, path, _ in recorder.patches] == [
        ("replace", "/count"),
        ("add", "/meta/pump"),
    ]


def config_of() -> StateConfig:
    return StateConfig(
        state_name="Board",
        definition=StateDefinitionInput(ports=(), name="Board"),
    )
