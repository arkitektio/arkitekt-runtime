"""One ordered record of everything an agent reports.

Task events (``PROGRESS``, ``YIELD``, ``COMPLETED``, ...), lock changes, state
patches, snapshots, the session baseline and (when served) the ``ASSIGN`` that
started a task each get a position ``pos`` in their session. ``pos`` starts at 1
with ``SESSION_INIT`` and has no gaps; ``(session_id, pos)`` is the durable key.
Every entry also carries the state revision ``global_rev`` as of that entry, so
"the world at ``pos``" is the state at ``global_rev`` plus the tasks and locks
folded from the entries up to ``pos``.

The agent hands every message it reports to :meth:`Journal.append` from one ordered
step (its patch processor), and the transport hand-off follows in that same step,
so the order entries are delivered in is the order they are numbered in, and the
order they were made in: a task's patches come before its ``YIELD``, its
``COMPLETED`` before its ``UNLOCK``.

This module is the Python side of the contract in ``docs/journal.md``; the Rust
agent (``rekuest`` crate, ``journal.rs``) implements the same.
"""

import asyncio
import contextlib
import copy
import dataclasses
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import jsonpatch  # type: ignore[import-untyped]

from arkitekt_runtime import messages
from arkitekt_runtime.state.gate import TaskClosedError, TaskGate

logger = logging.getLogger(__name__)

__all__ = [
    "FLUSH_TIMEOUT",
    "RING_CAPACITY",
    "Fold",
    "Journal",
    "JournalEntry",
    "JournalReader",
    "JournalSink",
    "JournalView",
    "TaskClosedError",
    "TaskFold",
    "TaskGate",
    "Watermark",
    "entry_matches",
    "iso_from_ms",
    "is_terminal_kind",
    "world_from_entries",
]

#: Entries kept in memory for resuming subscribers.
RING_CAPACITY = 4096
#: Finished tasks kept in the live fold.
FINISHED_KEEP = 1024
#: Entries persisted per write.
WRITE_BATCH = 256
#: How long a reader waits for the journal to be persisted up to what it reads.
FLUSH_TIMEOUT = 5.0

TERMINAL_KINDS = frozenset({"COMPLETED", "FAILED", "CRITICAL", "CANCELLED", "INTERRUPTED"})
#: Kinds that belong to the whole session rather than to a task, state or lock.
SESSION_KINDS = ("SESSION_INIT", "STATE_SNAPSHOT")
#: Kinds that carry state values.
STATE_KINDS = ("SESSION_INIT", "STATE_SNAPSHOT", "STATE_PATCH")
#: The agent messages that are journaled (``ASSIGN`` is recorded separately).
JOURNALED_KINDS = frozenset(
    {
        "PROGRESS",
        "LOG",
        "YIELD",
        "STARTED",
        "PAUSED",
        "RESUMED",
        "COMPLETED",
        "FAILED",
        "CRITICAL",
        "CANCELLED",
        "INTERRUPTED",
        "LOCK",
        "UNLOCK",
        "STATE_PATCH",
        "STATE_SNAPSHOT",
        "SESSION_INIT",
    }
)


def now_ms() -> int:
    """Milliseconds since the epoch."""
    return time.time_ns() // 1_000_000


def iso_from_ms(ms: int) -> str:
    """ISO 8601 with ``Z``, microseconds only when non-zero (pydantic's format)."""
    moment = datetime.fromtimestamp(ms // 1000, UTC) + timedelta(milliseconds=ms % 1000)
    if moment.microsecond == 0:
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def is_terminal_kind(kind: str) -> bool:
    """Whether an entry of this kind ends its task."""
    return kind in TERMINAL_KINDS


def kind_of(message: messages.Message) -> str:
    """The wire ``type`` of a message."""
    kind = getattr(message, "type", "")
    return str(getattr(kind, "value", kind))


# ------------------------------------------------------------------ entry --

#: How an entry is routed to subscribers: ``("state", name)``, ``("lock", key)``,
#: ``("action", key)`` or ``("everyone", None)`` (session-wide entries, reports
#: about tasks without an action key).
Route = tuple[str, str | None]


@dataclasses.dataclass
class JournalEntry:
    """One recorded fact."""

    session_id: str
    pos: int
    global_rev: int
    """The state revision after this entry."""
    event_time: int
    """When the agent recorded it, in epoch milliseconds."""
    kind: str
    """The frame ``type``: ``ASSIGN``, ``PROGRESS``, ``STATE_PATCH``, ..."""
    task_id: str | None
    """The task this entry belongs to (for ``UNLOCK``: the task that held the lock)."""
    action_key: str | None
    subject: str | None
    """The state (``STATE_PATCH``) or lock key (``LOCK``/``UNLOCK``) it is about."""
    message_id: str
    """The frame's ``id``, as sent."""
    payload: dict[str, Any]
    """The frame, without the stream-level ``seq``."""

    @property
    def timepoint(self) -> str:
        return iso_from_ms(self.event_time)

    @property
    def is_terminal(self) -> bool:
        return is_terminal_kind(self.kind)

    def to_json(self) -> dict[str, Any]:
        """The entry as the journal routes answer it."""
        return {
            "session_id": self.session_id,
            "pos": self.pos,
            "global_rev": self.global_rev,
            "timepoint": self.timepoint,
            "kind": self.kind,
            "task_id": self.task_id,
            "action_key": self.action_key,
            "subject": self.subject,
            "message_id": self.message_id,
            "payload": self.payload,
        }

    def frame(self) -> dict[str, Any]:
        """The payload with ``pos`` and ``journal_session``, as journal subscribers get it."""
        return stamp(dict(self.payload), self)

    def route(self) -> Route:
        """Who the entry is for, as subscribers filter."""
        if self.kind == "STATE_PATCH" and self.subject is not None:
            return ("state", self.subject)
        if self.kind in ("LOCK", "UNLOCK") and self.subject is not None:
            return ("lock", self.subject)
        if self.kind in SESSION_KINDS:
            return ("everyone", None)
        if self.action_key is not None:
            return ("action", self.action_key)
        return ("everyone", None)


def stamp(frame: dict[str, Any], entry: JournalEntry) -> dict[str, Any]:
    """Add ``pos`` and ``journal_session`` to a frame. (Not ``session_id``: state
    frames already have one, and it would clash.)"""
    frame["pos"] = entry.pos
    frame["journal_session"] = entry.session_id
    return frame


def payload_of(message: messages.Message) -> dict[str, Any]:
    """A message as the journal records it: without the stream-level fields."""
    payload = message.model_dump(mode="json")
    payload.pop("seq", None)
    for key in messages.JOURNAL_STAMP_FIELDS:
        payload.pop(key, None)
    return payload


def entry_matches(
    entry: JournalEntry,
    *,
    kinds: Sequence[str] | None = None,
    task_id: str | None = None,
    action_keys: Sequence[str] | None = None,
    state_keys: Sequence[str] | None = None,
    lock_keys: Sequence[str] | None = None,
) -> bool:
    """Whether an entry passes the journal route's filters.

    Key filters route as the websocket does: patches by state, locks by key,
    session-wide entries always, the rest by action key (entries without one always).
    The same rules as the SQL of the sqlite retriever.
    """
    if kinds and entry.kind not in kinds:
        return False
    if task_id is not None and entry.task_id != task_id:
        return False
    if action_keys is None and state_keys is None and lock_keys is None:
        return True
    if entry.kind == "STATE_PATCH":
        return not state_keys or entry.subject in state_keys
    if entry.kind in ("LOCK", "UNLOCK"):
        return not lock_keys or entry.subject in lock_keys
    if entry.kind in SESSION_KINDS:
        return True
    return entry.action_key is None or not action_keys or entry.action_key in action_keys


# ------------------------------------------------------------------- fold --


def apply_op(doc: Any, op: str, path: str, value: Any) -> Any:  # noqa: ANN401
    """Apply one JSON patch operation to ``doc``; returns the (possibly new) document."""
    operation: dict[str, Any] = {"op": op, "path": path}
    if op != "remove":
        operation["value"] = value
    try:
        return jsonpatch.apply_patch(doc, [operation], in_place=True)
    except Exception as e:  # noqa: BLE001 — a bad patch must not stop a replay
        logger.warning("could not apply %s %s: %s", op, path, e)
        return doc


@dataclasses.dataclass
class TaskFold:
    """A task as the journal knows it at some position."""

    task: str
    action_key: str | None
    interface: str | None = None
    reference: str | None = None
    status: str = "ASSIGNED"
    """``ASSIGNED``, ``RUNNING``, ``PAUSED``, or the terminal kind."""
    done: bool = False
    progress: int | None = None
    message: str | None = None
    error: str | None = None
    yields: int = 0
    last_returns: Any = None
    first_pos: int = 0
    last_pos: int = 0

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


@dataclasses.dataclass
class Fold:
    """States, tasks and locks, folded from entries in ``pos`` order."""

    states: dict[str, Any] = dataclasses.field(default_factory=dict)
    tasks: dict[str, TaskFold] = dataclasses.field(default_factory=dict)
    locks: dict[str, str] = dataclasses.field(default_factory=dict)
    """Lock key → the task holding it."""
    baseline_rev: int = 0
    """The revision of the last ``SESSION_INIT``/``STATE_SNAPSHOT`` applied. A
    snapshot at revision N is sent before the patch that reaches N and already
    contains it, so patches up to it are not applied a second time."""

    @classmethod
    def from_entries(cls, entries: Iterable[JournalEntry]) -> "Fold":
        fold = cls()
        for entry in entries:
            fold.apply(entry)
        return fold

    def copy(self) -> "Fold":
        return copy.deepcopy(self)

    def apply(self, entry: JournalEntry) -> None:
        payload = entry.payload
        kind = entry.kind
        if kind == "SESSION_INIT":
            self._replace_states(payload.get("states"))
            self.baseline_rev = 0
        elif kind == "STATE_SNAPSHOT":
            self._replace_states(payload.get("snapshots"))
            self.baseline_rev = int(payload.get("global_rev") or entry.global_rev)
        elif kind == "STATE_PATCH":
            revision = payload.get("global_rev", entry.global_rev)
            subject = entry.subject
            if (
                subject is not None
                and subject in self.states
                and not (isinstance(revision, int) and revision <= self.baseline_rev)
            ):
                self.states[subject] = apply_op(
                    self.states[subject],
                    str(payload.get("op") or ""),
                    str(payload.get("path") or ""),
                    payload.get("value"),
                )
        elif kind == "LOCK":
            if entry.subject is not None and entry.task_id is not None:
                self.locks[entry.subject] = entry.task_id
        elif kind == "UNLOCK":
            if entry.subject is not None:
                self.locks.pop(entry.subject, None)

        task_id = entry.task_id
        if task_id is None:
            return
        if kind in ("STATE_PATCH", "LOCK", "UNLOCK"):
            known = self.tasks.get(task_id)
            if known is not None:
                known.last_pos = max(known.last_pos, entry.pos)
            return

        task = self.tasks.get(task_id)
        if task is None:
            task = TaskFold(
                task=task_id,
                action_key=entry.action_key,
                first_pos=entry.pos,
                last_pos=entry.pos,
            )
            self.tasks[task_id] = task
        task.last_pos = entry.pos
        if task.action_key is None:
            task.action_key = entry.action_key
        if task.done:
            return
        if kind == "ASSIGN":
            task.interface = _text(payload, "interface")
            task.reference = _text(payload, "reference")
        elif kind == "PROGRESS":
            progress = payload.get("progress")
            if isinstance(progress, int) and not isinstance(progress, bool):
                task.progress = progress
            message = _text(payload, "message")
            if message is not None:
                task.message = message
            task.status = "RUNNING"
        elif kind == "LOG":
            task.message = _text(payload, "message")
            task.status = "RUNNING"
        elif kind == "YIELD":
            task.yields += 1
            task.last_returns = payload.get("returns")
            task.status = "RUNNING"
        elif kind in ("STARTED", "RESUMED"):
            task.status = "RUNNING"
        elif kind == "PAUSED":
            task.status = "PAUSED"
        elif is_terminal_kind(kind):
            task.status = kind
            task.done = True
            task.error = _text(payload, "error")

    def _replace_states(self, states: Any) -> None:  # noqa: ANN401
        if isinstance(states, dict):
            for name, value in states.items():
                self.states[name] = copy.deepcopy(value)

    def world(
        self,
        action_keys: Iterable[str] | None = None,
        state_keys: Iterable[str] | None = None,
        lock_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """States, tasks and locks, optionally only some keys."""
        actions = set(action_keys) if action_keys is not None else None
        states = set(state_keys) if state_keys is not None else None
        locks = set(lock_keys) if lock_keys is not None else None
        return {
            "states": {
                name: copy.deepcopy(value)
                for name, value in self.states.items()
                if states is None or name in states
            },
            "tasks": {
                task_id: task.to_json()
                for task_id, task in self.tasks.items()
                if actions is None or (task.action_key is not None and task.action_key in actions)
            },
            "locks": {
                key: task for key, task in self.locks.items() if locks is None or key in locks
            },
        }


def world_from_entries(entries: Sequence[JournalEntry], pos: int) -> tuple[JournalEntry, Fold] | None:
    """The entry at ``pos`` and the world as of it, from one session's entries in order.

    States are replayed from the last ``SESSION_INIT``/``STATE_SNAPSHOT`` at or before
    ``pos``; tasks and locks are folded from every entry up to it.
    """
    at = next((entry for entry in entries if entry.pos == pos), None)
    if at is None:
        return None
    anchor = max(
        (entry.pos for entry in entries if entry.pos <= pos and entry.kind in SESSION_KINDS),
        default=0,
    )
    fold = Fold()
    for entry in entries:
        if entry.pos > pos:
            break
        if entry.kind in STATE_KINDS and entry.pos < anchor:
            continue
        fold.apply(entry)
    return at, fold


# ---------------------------------------------------------------- storage --


@runtime_checkable
class JournalSink(Protocol):
    """Where journal entries are persisted (a sink that keeps the journal)."""

    async def awrite_journal(self, entries: Sequence[JournalEntry]) -> None:
        """Persist entries (in ``pos`` order, never spanning sessions); re-sent
        entries are ignored (``INSERT OR IGNORE``)."""
        ...


@runtime_checkable
class JournalReader(Protocol):
    """Reads the persisted journal (a retriever that keeps the journal)."""

    async def aget_journal_entries(
        self,
        session_id: str,
        after: int = 0,
        until: int | None = None,
        limit: int | None = None,
        kinds: Sequence[str] | None = None,
        task_id: str | None = None,
        action_keys: Sequence[str] | None = None,
        state_keys: Sequence[str] | None = None,
        lock_keys: Sequence[str] | None = None,
    ) -> list[JournalEntry]:
        """Entries of a session after ``after`` (up to ``until``), in order."""
        ...

    async def aget_journal_task_entries(self, task_id: str) -> list[JournalEntry]:
        """Every entry of one task, in order."""
        ...

    async def aget_journal_pos_at_time(self, session_id: str, ms: int) -> int | None:
        """The last position of a session at or before ``ms`` (epoch milliseconds)."""
        ...

    async def aget_journal_world(
        self, session_id: str, pos: int
    ) -> tuple[JournalEntry, Fold] | None:
        """The entry at ``pos`` and the states, tasks and locks as of it."""
        ...


# ---------------------------------------------------------------- journal --


@dataclasses.dataclass(frozen=True)
class Watermark:
    """The last position of a session."""

    session_id: str
    pos: int
    global_rev: int

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class JournalView:
    """What :meth:`Journal.locked` hands out: a consistent view at the watermark."""

    def __init__(self, journal: "Journal") -> None:
        self._journal = journal

    @property
    def watermark(self) -> Watermark | None:
        """``None`` before the first session."""
        journal = self._journal
        if journal._session is None:
            return None
        return Watermark(journal._session, journal._pos, journal._global_rev)

    @property
    def fold(self) -> Fold:
        """States, tasks and locks at the watermark (finished tasks are only kept for a while)."""
        return self._journal._fold

    def recent(self, after: int, until: int) -> list[JournalEntry] | None:
        """Entries in ``(after, until]`` if they are still in memory."""
        if after >= until:
            return []
        ring = self._journal._ring
        if not ring:
            return None
        if ring[0].pos > after + 1:
            return None
        return [entry for entry in ring if after < entry.pos <= until]


#: Called under the journal's lock with each new entry and the message it records
#: (``None`` for ``ASSIGN``), in ``pos`` order.
JournalListener = Callable[[JournalEntry, messages.Message | None], None]


class Journal:
    """Numbers, records and keeps every message the agent reports. See the module docs.

    Without a ``session``, entries start with the first ``SESSION_INIT``; messages
    before it are not recorded.
    """

    def __init__(
        self,
        sink: JournalSink | None = None,
        session: str | None = None,
        ring_capacity: int = RING_CAPACITY,
        finished_keep: int = FINISHED_KEEP,
    ) -> None:
        self._lock = threading.RLock()
        self._session = session
        self._pos = 0
        self._global_rev = 0
        self._ring: deque[JournalEntry] = deque(maxlen=ring_capacity)
        self._fold = Fold()
        self._finished: deque[str] = deque()
        self._finished_keep = finished_keep
        self._listeners: list[JournalListener] = []
        self.sink = sink
        # Persistence: a writer task per event loop, fed in pos order.
        self._write_queue: asyncio.Queue[JournalEntry] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._writer_loop: asyncio.AbstractEventLoop | None = None
        self._durable: tuple[str, int] = ("", 0)
        self._durable_changed: asyncio.Condition | None = None

    def __repr__(self) -> str:
        return f"Journal(session={self._session!r}, pos={self._pos})"

    # -- reading --------------------------------------------------------------

    @property
    def session(self) -> str | None:
        return self._session

    def watermark(self) -> Watermark | None:
        """The last position, ``None`` before the first session."""
        with self.locked() as view:
            return view.watermark

    @contextlib.contextmanager
    def locked(self) -> Iterator[JournalView]:
        """Hold the journal: nothing is recorded meanwhile, so what is read is
        consistent with the watermark, and a subscription made inside gets exactly
        the entries after it."""
        with self._lock:
            yield JournalView(self)

    def set_ring_capacity(self, capacity: int) -> None:
        """How many recent entries are kept in memory for resuming subscribers."""
        with self._lock:
            self._ring = deque(self._ring, maxlen=capacity)

    def add_listener(self, listener: JournalListener) -> None:
        """Be told about every new entry, under the lock, in ``pos`` order."""
        self._listeners.append(listener)

    # -- recording ------------------------------------------------------------

    def append(
        self, message: messages.Message, action_key: str | None = None
    ) -> JournalEntry | None:
        """Record a message the agent reports. ``None`` if it is not recorded
        (not a journaled kind, or before the first session)."""
        kind = kind_of(message)
        if kind not in JOURNALED_KINDS:
            return None
        with self._lock:
            if isinstance(message, messages.SessionInit):
                if self._session != message.session_id:
                    self._session = message.session_id
                    self._pos = 0
                    self._ring.clear()
                    self._fold = Fold()
                    self._finished.clear()
                self._global_rev = 0
            elif isinstance(message, (messages.StatePatch, messages.StateSnapshot)):
                self._global_rev = message.global_rev

            task_id: str | None
            subject: str | None = None
            if isinstance(message, messages.StatePatch):
                task_id, subject = message.task_id, message.state_name
            elif isinstance(message, messages.Lock):
                task_id, subject = message.task, message.key
            elif isinstance(message, messages.Unlock):
                task_id, subject = self._fold.locks.get(message.key), message.key
            else:
                task_id = getattr(message, "task", None)
            return self._append(
                kind, task_id, action_key, subject, payload_of(message), message.id, message
            )

    def record_assign(self, assign: messages.Assign, action_key: str) -> JournalEntry | None:
        """Record the assignment that starts a task (secrets removed)."""
        payload = assign.model_dump(mode="json")
        payload.pop("token", None)
        payload["type"] = "ASSIGN"
        with self._lock:
            return self._append("ASSIGN", assign.task, action_key, None, payload, assign.id, None)

    def _append(
        self,
        kind: str,
        task_id: str | None,
        action_key: str | None,
        subject: str | None,
        payload: dict[str, Any],
        message_id: str,
        message: messages.Message | None,
    ) -> JournalEntry | None:
        session = self._session
        if session is None:
            return None
        self._pos += 1
        payload["id"] = message_id
        if action_key is None and task_id is not None:
            known = self._fold.tasks.get(task_id)
            action_key = known.action_key if known is not None else None
        entry = JournalEntry(
            session_id=session,
            pos=self._pos,
            global_rev=self._global_rev,
            event_time=now_ms(),
            kind=kind,
            task_id=task_id,
            action_key=action_key,
            subject=subject,
            message_id=message_id,
            payload=payload,
        )
        self._fold.apply(entry)
        if entry.is_terminal and task_id is not None:
            self._finished.append(task_id)
            if len(self._finished) > self._finished_keep:
                self._fold.tasks.pop(self._finished.popleft(), None)
        self._ring.append(entry)
        self._persist(entry)
        for listener in self._listeners:
            try:
                listener(entry, message)
            except Exception:  # noqa: BLE001 — a listener must not break numbering
                logger.error("A journal listener failed", exc_info=True)
        return entry

    # -- persistence ----------------------------------------------------------

    def _persist(self, entry: JournalEntry) -> None:
        if self.sink is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop; the journal entry %s is not persisted", entry.pos)
            return
        if self._writer_loop is not loop or self._writer is None or self._writer.done():
            self._write_queue = asyncio.Queue()
            self._durable_changed = asyncio.Condition()
            self._writer_loop = loop
            self._writer = loop.create_task(self._awrite_loop(self._write_queue))
        assert self._write_queue is not None
        self._write_queue.put_nowait(entry)

    async def _awrite_loop(self, queue: "asyncio.Queue[JournalEntry]") -> None:
        carry: JournalEntry | None = None
        while True:
            first = carry if carry is not None else await queue.get()
            carry = None
            batch = [first]
            while len(batch) < WRITE_BATCH and not queue.empty():
                entry = queue.get_nowait()
                if entry.session_id != first.session_id:
                    carry = entry
                    break
                batch.append(entry)
            try:
                assert self.sink is not None
                await self.sink.awrite_journal(batch)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — keep numbering; report the loss
                logger.warning("Could not persist the journal", exc_info=True)
            last = batch[-1]
            self._durable = (last.session_id, last.pos)
            condition = self._durable_changed
            if condition is not None:
                async with condition:
                    condition.notify_all()

    async def aflush_to(self, pos: int, timeout: float = FLUSH_TIMEOUT) -> bool:
        """Wait (bounded) until everything up to ``pos`` of the current session is persisted."""
        session = self._session
        if session is None or self.sink is None or pos <= 0:
            return True

        def reached() -> bool:
            return self._durable[0] == session and self._durable[1] >= pos

        condition = self._durable_changed
        if reached() or condition is None:
            return True
        if self._writer_loop is not asyncio.get_running_loop():
            # Read from another event loop (a test client's): poll instead.
            deadline = time.monotonic() + timeout
            while not reached():
                if time.monotonic() > deadline:
                    logger.warning("The journal was not persisted up to %s within %ss", pos, timeout)
                    return False
                await asyncio.sleep(0.005)
            return True
        try:
            async with condition:
                await asyncio.wait_for(condition.wait_for(reached), timeout)
            return True
        except TimeoutError:
            logger.warning("The journal was not persisted up to %s within %ss", pos, timeout)
            return False

    async def aflush(self, timeout: float = FLUSH_TIMEOUT) -> bool:
        """Wait (bounded) until every entry so far is persisted."""
        return await self.aflush_to(self._pos, timeout)

    async def aclose(self, timeout: float = FLUSH_TIMEOUT) -> None:
        """Persist what is left (bounded), then stop the writer."""
        await self.aflush(timeout)
        writer = self._writer
        self._writer = None
        self._writer_loop = None
        if writer is not None and not writer.done():
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await writer
