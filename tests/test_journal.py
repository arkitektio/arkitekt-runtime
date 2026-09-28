"""The agent journal, piece by piece: numbering and folding, the task gate and the wire
stamp.

Retention until ``JOURNAL_ACK`` is the distributed runtime's (rekuest); the served
journal and its storage are arkitekt-fastapi's.
"""

import asyncio
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.agents.journal import (
    Journal,
    JournalEntry,
    TaskClosedError,
    TaskGate,
    Watermark,
    entry_matches,
    iso_from_ms,
    world_from_entries,
)
from arkitekt_runtime.state.observable import Mutation, StateConfig, make_evented
from arkitekt_runtime.state.write import write_view


def _session_init(session: str) -> messages.SessionInit:
    return messages.SessionInit(session_id=session, states={"Camera": {"exposure": 1, "tags": []}})


def _patch(rev: int, task: str | None, op: str = "replace", path: str = "/exposure", value: object = None) -> messages.StatePatch:
    return messages.StatePatch(
        session_id="s",
        global_rev=rev,
        state_name="Camera",
        ts=0.0,
        op=op,
        path=path,
        value=rev + 1 if value is None else value,
        old_value=None,
        task_id=task,
    )


def _assign(task: str, token: str | None = "secret") -> messages.Assign:
    return messages.Assign(
        interface="set",
        task=task,
        reference="r",
        args={},
        user="u",
        org="o",
        action="a",
        implementation="i",
        token=token,
    )


# ------------------------------------------------------------------ journal --


def test_numbers_and_folds() -> None:
    seen: list[tuple[str, int]] = []
    journal = Journal()
    journal.add_listener(lambda entry, message: seen.append((entry.kind, entry.pos)))

    assert journal.append(messages.Progress(task="early")) is None, (
        "nothing is recorded before the session"
    )
    journal.append(_session_init("s"))
    journal.record_assign(_assign("t"), "set")
    journal.append(messages.Lock(key="cam", task="t"))
    journal.append(_patch(1, "t"))
    journal.append(messages.Yield(task="t", returns={"return0": 2}), "set")
    journal.append(messages.Completed(task="t"), "set")
    journal.append(messages.Unlock(key="cam"))
    assert journal.append(messages.HeartbeatEvent()) is None
    assert journal.append(messages.Register(token="x")) is None

    assert seen == [
        (kind, pos + 1)
        for pos, kind in enumerate(
            ["SESSION_INIT", "ASSIGN", "LOCK", "STATE_PATCH", "YIELD", "COMPLETED", "UNLOCK"]
        )
    ]

    with journal.locked() as view:
        assert view.watermark == Watermark("s", 7, 1)
        fold = view.fold
        assert fold.states["Camera"] == {"exposure": 2, "tags": []}
        task = fold.tasks["t"]
        assert (task.status, task.done, task.yields) == ("COMPLETED", True, 1)
        assert task.reference == "r"
        assert task.last_returns == {"return0": 2}
        assert fold.locks == {}
        recent = view.recent(0, 7)
        assert recent is not None and len(recent) == 7
        assert "token" not in recent[1].payload, "secrets are not recorded"
        assert recent[1].payload["type"] == "ASSIGN"
        assert recent[6].task_id == "t", "UNLOCK names the holder"
        assert recent[6].action_key == "set", "an UNLOCK carries its holder's action key"
        assert recent[3].global_rev == 1
        assert [e.pos for e in view.recent(2, 4) or []] == [3, 4]
        assert all("seq" not in e.payload for e in recent)

    # A new session starts over.
    journal.append(_session_init("s2"))
    assert journal.watermark() == Watermark("s2", 1, 0)


def test_the_entry_json_matches_the_contract() -> None:
    journal = Journal(session="s")
    entry = journal.append(messages.Log(task="t", message="hi"), "act")
    assert entry is not None
    assert list(entry.to_json()) == [
        "session_id",
        "pos",
        "global_rev",
        "timepoint",
        "kind",
        "task_id",
        "action_key",
        "subject",
        "message_id",
        "payload",
    ]
    assert entry.payload["id"] == entry.message_id
    assert entry.frame()["pos"] == 1 and entry.frame()["journal_session"] == "s"
    assert entry.route() == ("action", "act")


def test_iso_format() -> None:
    assert iso_from_ms(1_700_000_000_123) == "2023-11-14T22:13:20.123000Z"
    assert iso_from_ms(1_700_000_000_000) == "2023-11-14T22:13:20Z"


def test_a_snapshot_is_not_applied_twice() -> None:
    """A snapshot at N already contains the patch reaching N, which follows it."""
    journal = Journal(session="s")
    journal.append(_session_init("s"))
    journal.append(_patch(1, "t", op="add", path="/tags/-", value="a"))
    journal.append(
        messages.StateSnapshot(
            session_id="s", global_rev=2, snapshots={"Camera": {"exposure": 1, "tags": ["a", "b"]}}
        )
    )
    journal.append(_patch(2, "t", op="add", path="/tags/-", value="b"))
    journal.append(_patch(3, "t", op="add", path="/tags/-", value="c"))
    with journal.locked() as view:
        assert view.fold.states["Camera"]["tags"] == ["a", "b", "c"]
        entries = view.recent(0, 5) or []
    for pos in range(1, 6):
        world = world_from_entries(entries, pos)
        assert world is not None
    assert world_from_entries(entries, 4)[1].states["Camera"]["tags"] == ["a", "b"]  # type: ignore[index]
    assert world_from_entries(entries, 5)[1].states["Camera"]["tags"] == ["a", "b", "c"]  # type: ignore[index]
    assert world_from_entries(entries, 2)[1].states["Camera"]["tags"] == ["a"]  # type: ignore[index]


def test_key_filters_route_like_the_websocket() -> None:
    def entry(kind: str, subject: str | None = None, action_key: str | None = None) -> JournalEntry:
        return JournalEntry("s", 1, 0, 0, kind, "t", action_key, subject, "m", {})

    patch = entry("STATE_PATCH", subject="Camera")
    lock = entry("LOCK", subject="cam")
    snapshot = entry("STATE_SNAPSHOT")
    log = entry("LOG", action_key="act")
    anonymous = entry("CRITICAL")
    only_other = {"action_keys": ["other"], "state_keys": ["Other"], "lock_keys": ["other"]}
    assert [entry_matches(e, **only_other) for e in (patch, lock, snapshot, log, anonymous)] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert entry_matches(patch, action_keys=["x"]), "an unset key filter admits every key"
    assert not entry_matches(log, kinds=["YIELD"])
    assert entry_matches(log, task_id="t") and not entry_matches(log, task_id="u")


def test_finished_tasks_are_forgotten_by_the_live_fold() -> None:
    journal = Journal(session="s", finished_keep=2)
    for task in ("a", "b", "c"):
        journal.append(messages.Completed(task=task))
    with journal.locked() as view:
        assert list(view.fold.tasks) == ["b", "c"]


def test_recent_falls_back_once_the_ring_overflowed() -> None:
    journal = Journal(session="s", ring_capacity=3)
    for i in range(5):
        journal.append(messages.Log(task="t", message=str(i)))
    with journal.locked() as view:
        assert view.recent(0, 5) is None, "entries 1 and 2 are no longer in memory"
        assert [e.pos for e in view.recent(2, 5) or []] == [3, 4, 5]
        assert view.recent(5, 5) == []


@pytest.mark.asyncio
async def test_persists_in_order_in_batches() -> None:
    class Collect:
        def __init__(self) -> None:
            self.positions: list[int] = []
            self.batches = 0

        async def awrite_journal(self, entries: Sequence[JournalEntry]) -> None:
            self.batches += 1
            assert len({e.session_id for e in entries}) == 1, "a batch never spans sessions"
            self.positions.extend(e.pos for e in entries)
            await asyncio.sleep(0)

    sink = Collect()
    journal = Journal(sink=sink, session="s")

    async def report(task: int) -> None:
        for i in range(50):
            journal.append(messages.Log(task=f"t{task}", message=str(i)))
            await asyncio.sleep(0)

    await asyncio.gather(*(report(t) for t in range(8)))
    assert await journal.aflush(2.0)
    assert sink.positions == list(range(1, 401))
    assert sink.batches < 400, "entries are written in batches"
    await journal.aclose()


# --------------------------------------------------------------------- gate --


def test_gate_refuses_after_close_and_allows_nesting() -> None:
    gate = TaskGate()
    assert gate.enter() and gate.enter(), "entering twice on one thread is fine"
    gate.leave()
    gate.leave()
    gate.close()
    assert not gate.enter()
    assert gate.closed
    with pytest.raises(TaskClosedError):
        with gate.held():
            pass


def test_gate_close_waits_for_in_flight() -> None:
    gate = TaskGate()
    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        assert gate.enter()
        entered.set()
        release.wait()
        time.sleep(0.05)
        gate.leave()

    thread = threading.Thread(target=worker)
    thread.start()
    entered.wait()
    elapsed: list[float] = []

    def closer() -> None:
        start = time.monotonic()
        gate.close()
        elapsed.append(time.monotonic() - start)

    closing = threading.Thread(target=closer)
    closing.start()
    time.sleep(0.02)
    release.set()
    thread.join()
    closing.join()
    assert elapsed[0] >= 0.05


def test_gate_close_does_not_wait_for_its_own_thread() -> None:
    gate = TaskGate()
    assert gate.enter()
    gate.close()  # would deadlock if it waited for this thread's own pass
    gate.leave()
    assert gate.closed


@dataclass
class Camera:
    exposure: float = 1.0
    tags: list[str] = field(default_factory=list)


def _evented_camera() -> Camera:
    from arkitekt_spec.actions import StateDefinitionInput

    config = StateConfig(state_name="Camera", definition=StateDefinitionInput(name="Camera", ports=[]))
    return make_evented(Camera(), config, "")


def test_a_closed_task_changes_no_state() -> None:
    camera = _evented_camera()
    gate = TaskGate()
    view = write_view(camera, Mutation(correlation_id="t", gate=gate))
    view.exposure = 2.0
    view.tags.append("a")
    gate.close()
    with pytest.raises(TaskClosedError):
        view.exposure = 3.0
    with pytest.raises(TaskClosedError):
        view.tags.extend(["b", "c"])
    assert camera.exposure == 2.0 and camera.tags == ["a"], "a refused change changes nothing"


def test_the_gate_does_not_change_what_a_mutation_equals() -> None:
    assert Mutation(correlation_id="t", gate=TaskGate()) == Mutation(correlation_id="t")


# ----------------------------------------------------------------- messages --


def test_unstamped_frames_are_unchanged() -> None:
    progress = messages.Progress(task="t", progress=None, message=None, seq=3)
    assert json.loads(progress.model_dump_json()) == {
        "id": progress.id,
        "seq": 3,
        "type": "PROGRESS",
        "task": "t",
        "progress": None,
        "message": None,
    }
    patch = _patch(1, None)
    assert set(json.loads(patch.model_dump_json())) == {
        "id",
        "type",
        "session_id",
        "global_rev",
        "state_name",
        "ts",
        "op",
        "path",
        "value",
        "old_value",
        "task_id",
    }
    stamped = progress.model_copy(update={"pos": 4, "journal_session": "s", "agent_ts": 1.5})
    dumped = json.loads(stamped.model_dump_json())
    assert (dumped["pos"], dumped["journal_session"], dumped["agent_ts"]) == (4, "s", 1.5)
    assert "pos" not in messages.Register.model_fields


def test_init_and_journal_ack_parse() -> None:
    init = messages.Init(agent="a")
    assert init.journal is False
    from pydantic import TypeAdapter

    adapter: TypeAdapter[messages.ToAgentMessage] = TypeAdapter(messages.ToAgentMessage)
    ack = adapter.validate_python({"type": "JOURNAL_ACK", "journal_session": "s", "pos": 3})
    assert isinstance(ack, messages.JournalAck) and ack.pos == 3
    assert adapter.validate_python({"type": "INIT", "agent": "a", "journal": True}).journal  # type: ignore[union-attr]
