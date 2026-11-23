import contextlib
import dataclasses
import functools
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem, SupportsRichComparison

from arkitekt_spec.actions import (
    PortKind,
    ReturnPortInput,
    StateDefinitionInput,
)
from arkitekt_spec.declare.state.publish import Patch, StateHolder
from arkitekt_spec.declare.structures.types import StateDeclaration

from arkitekt_runtime.state.gate import TaskClosedError, TaskGate

# --- JSON Pointer Utilities (RFC 6901) ---


def _escape_json_pointer(key: str) -> str:
    """Escape special characters for JSON Pointer (RFC 6901).

    Per RFC 6901:
    - '~' is escaped as '~0'
    - '/' is escaped as '~1'
    """
    return key.replace("~", "~0").replace("/", "~1")


def _make_path(base: str, key: str | int) -> str:
    """Create a JSON Pointer path by appending a key to a base path."""
    escaped_key = _escape_json_pointer(str(key)) if isinstance(key, str) else str(key)
    return f"{base}/{escaped_key}"


# --- Configuration ---


@dataclasses.dataclass
class Marks:
    """What is said of a task's changes while it runs, and changes as it goes on."""

    lifecycle: str | None = None
    """Set by the runtime while the task is not in its ordinary course (``CLEANUP`` once
    it is being stopped)."""
    qualifiers: tuple[str, ...] = ()
    """What the app says its changes are for right now (``task.qualified``)."""


@dataclasses.dataclass(frozen=True)
class Mutation:
    """Who is changing a state: the task, for the patches' correlation.

    With a ``gate`` (the task's), a change is refused -- before anything is
    changed -- once the task's end has been reported.

    Locks play no part here. A state's ``required_locks`` are the locks an action
    taking it holds (inferred when the action is declared), so two tasks never
    command the same thing at once; who may *write* the state is not restricted.
    A change made outside a task (a manager's, a background hook's) is simply
    not correlated with one.
    """

    correlation_id: str | None = None
    gate: TaskGate | None = dataclasses.field(default=None, compare=False)
    marks: Marks | None = dataclasses.field(default=None, compare=False)
    """The task's marks, which its patches carry as they are when each is made."""


OUTSIDE_TASK = Mutation()
"""What a change made on the state object itself (not through a task's view) counts as."""


@dataclasses.dataclass
class StateConfig:
    """Configuration for evented objects, storing the state interface name and schema.

    One per state *instance*, shared by all of its evented children, so it is also
    where that instance's publisher and the mutation in progress live.
    """

    state_name: str
    definition: StateDefinitionInput
    publisher: StateHolder | None = dataclasses.field(default=None, compare=False)
    """Where patches go: the agent that adopted the state. None until then."""
    active: Mutation | None = dataclasses.field(default=None, init=False, compare=False)
    """The mutation a task's write view is making right now, for one operation."""
    pending: "list[Patch] | None" = dataclasses.field(
        default=None, init=False, compare=False
    )
    """The update being put together (see :meth:`updating`); None while there is none."""
    lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock, init=False, compare=False, repr=False
    )
    """Serializes changes: tasks in worker threads write the same instance."""

    @classmethod
    def for_declaration(cls, declaration: StateDeclaration) -> "StateConfig":
        """A fresh config for one instance of a declared state."""
        return cls(
            state_name=declaration.interface,
            definition=declaration.definition,
        )

    @property
    def mutation(self) -> Mutation:
        return self.active or OUTSIDE_TASK

    @contextlib.contextmanager
    def mutating(self, mutation: Mutation) -> Iterator[None]:
        """Make one synchronous operation count as ``mutation`` (a task's view calls this).

        The task's gate is entered for the whole operation, and before it changes
        anything: a task that has ended changes nothing, and one whose end is being
        reported is waited for.
        """
        with self.lock:
            gate = mutation.gate
            if gate is not None and not gate.enter():
                raise TaskClosedError(
                    f"Cannot modify state '{self.state_name}': task "
                    f"{mutation.correlation_id} has already ended"
                )
            previous = self.active
            self.active = mutation
            try:
                # One operation is one update, whatever it takes apart into.
                with self.updating():
                    yield
            finally:
                self.active = previous
                if gate is not None:
                    gate.leave()

    @contextlib.contextmanager
    def updating(self) -> Iterator[None]:
        """Make every change in the block one update of the state.

        An update is the unit a state changes by: it takes the state from one revision to
        the next, and whoever follows the state sees all of it or none of it. One operation
        on a state (``moves.update({...})``, ``items.extend([...])``) is one update however
        many JSON Patch operations describe it, and so is everything a task does inside
        ``task.updating(state)``.

        The state's lock is held for the block, so no other thread's change lands in the
        middle of it. The update is handed on when the outermost block is left, also when
        it is left by an exception: what was changed until then was changed.
        """
        with self.lock:
            if self.pending is not None:
                yield  # inside an update already: its changes join it
                return
            self.pending = []
            try:
                yield
            finally:
                patches, self.pending = self.pending, None
                self._hand_on(patches)

    def _hand_on(self, patches: "list[Patch]") -> None:
        if patches and self.publisher is not None:
            self.publisher.publish_update(self.state_name, patches)

    def publish(self, patch: Patch) -> None:
        """Add ``patch`` to the update being put together, or hand it on as one of its own.

        An update is one task's, said one way: a change that is another task's, or marked
        otherwise, than what the update holds so far (two coroutines of one thread inside
        the same block) closes that update and starts the next.
        """
        with self.lock:
            if self.pending is None:
                self._hand_on([patch])
                return
            if self.pending and _said_of(self.pending[0]) != _said_of(patch):
                patches, self.pending = self.pending, []
                self._hand_on(patches)
            self.pending.append(patch)


def _said_of(patch: Patch) -> tuple[str | None, str | None, tuple[str, ...]]:
    """What an update says of every change in it: who made it, and what is said of it."""
    return (patch.correlation_id, patch.lifecycle, tuple(patch.qualifiers))


def config_of(obj: Any) -> StateConfig | None:  # noqa: ANN401
    """The config of an evented state object or container, if it is one."""
    return getattr(obj, "_event_config", None) or getattr(obj, "_config", None)


class ContainerView:
    """What a state's view hands out in place of a list or a dict it holds.

    A view is how a container is *reached*, never what a state holds: wherever one is
    stored or sent, it is the container it shows that is (see :func:`unwrapped`).
    """

    __slots__ = ("_target",)


def unwrapped(value: Any) -> Any:  # noqa: ANN401
    """The container a view shows, or ``value`` itself when it is no view."""
    while isinstance(value, ContainerView):
        value = object.__getattribute__(value, "_target")
    return value


def evented(obj: Any, declaration: StateDeclaration) -> Any:  # noqa: ANN401
    """Make ``obj`` an evented instance of the declared state, with a config of its own.

    What an agent does to a startup hook's return before adopting it. A plain
    instance of the class is nothing but a dataclass until then: the class is
    not changed by being registered, so this is where the rules of the app that
    adopts it are attached.
    """
    return make_evented(obj, StateConfig.for_declaration(declaration), "")


def adopt(obj: Any, publisher: StateHolder) -> None:  # noqa: ANN401
    """An agent takes a state over: its patches go to ``publisher`` from now on."""
    config = config_of(obj)
    if config is None:
        return
    config.publisher = publisher


def _publish_patch(config: StateConfig, patch: Patch) -> None:
    """Publish a patch to the state's publisher, if an agent has adopted it.

    It is stamped with the task making the change, which is the mutation in
    progress (a task's write view), not something looked up.
    """
    mutation = config.mutation
    patch.correlation_id = mutation.correlation_id
    if mutation.marks is not None:
        patch.lifecycle = mutation.marks.lifecycle
        patch.qualifiers = mutation.marks.qualifiers
    config.publish(patch)


def _resolve_child_port(
    config: StateConfig,
    parent_port: ReturnPortInput | None,
    key: str | int,
) -> ReturnPortInput | None:
    """Resolve the schema port for a direct child without walking a JSON path.

    A list's and a map's one child describes every item: an index or a key names no port.
    """
    if isinstance(key, int) or (parent_port and parent_port.kind == PortKind.DICT):
        return parent_port.children[0] if parent_port and parent_port.children else None

    search_scope = (
        config.definition.ports if parent_port is None else (parent_port.children or [])
    )
    return next((port for port in search_scope if port.key == key), None)


def _make_patch(
    *,
    op: str,
    path: str,
    value: object,
    old_value: object,
    port: ReturnPortInput | None,
) -> Patch:
    """Create a patch carrying the already resolved schema port."""
    return Patch(
        op=op,
        path=path,
        value=value,
        old_value=old_value,
        port=port,
    )


def _one_update(method: Callable[..., Any]) -> Callable[..., Any]:
    """Make everything a container's method changes one update of its state."""

    @functools.wraps(method)
    def as_one(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        config = self._config
        if config is None:
            return method(self, *args, **kwargs)
        with config.updating():
            return method(self, *args, **kwargs)

    return as_one


# --- JSON Patch Compliant EventedDict (RFC 6902) ---

#: Bound to what a JSON Pointer segment can be: every key is handed to `_make_path`, so a
#: mapping keyed by anything else cannot be described as a patch path in the first place.
K = TypeVar("K", bound="str | int")
V = TypeVar("V")
_D = TypeVar("_D")
#: The fallback a caller hands to `pop`, which need not be one of the mapping's own values.
D = TypeVar("D")


class _EventedInstance(Protocol):
    """A dataclass instance make_evented swizzled: where its patches go."""

    _event_config: StateConfig
    _event_path: str
    _event_port: ReturnPortInput | None


class EventedDict(dict[K, V], Generic[K, V]):
    """A dictionary wrapper that emits JSON Patch operations on modification.

    Supports all standard dict mutation methods with proper JSON Patch semantics:
    - add: When a new key is added
    - remove: When a key is deleted
    - replace: When an existing key's value is changed
    """

    def __init__(
        self,
        data: dict[K, V],
        config: StateConfig,
        path: str,
        port: ReturnPortInput | None,
    ) -> None:
        super().__init__(data)
        self._config = config
        self._path = path
        self._port = port

    def __setitem__(self, key: K, value: V) -> None:
        full_path = _make_path(self._path, key)
        exists = key in self
        old_value = self.get(key) if exists else None
        child_port = _resolve_child_port(self._config, self._port, key)

        # Wrap new complex objects so future changes are caught
        value = make_evented(value, self._config, full_path, port=child_port)

        super().__setitem__(key, value)

        # JSON Patch: "add" for new keys, "replace" for existing
        op = "replace" if exists else "add"
        _publish_patch(
            self._config,
            _make_patch(
                op=op, path=full_path, value=value, old_value=old_value, port=child_port
            ),
        )

    def __delitem__(self, key: K) -> None:
        full_path = _make_path(self._path, key)
        old_value = self[key]
        child_port = _resolve_child_port(self._config, self._port, key)

        super().__delitem__(key)
        _publish_patch(
            self._config,
            _make_patch(
                op="remove",
                path=full_path,
                value=None,
                old_value=old_value,
                port=child_port,
            ),
        )

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop(self, key: K, default: D, /) -> V | D: ...

    def pop(self, key: K, /, *default: D) -> V | D:
        """Remove specified key and return the corresponding value.

        Emits a 'remove' patch if the key exists.
        """
        if key in self:
            full_path = _make_path(self._path, key)
            old_value = self[key]
            child_port = _resolve_child_port(self._config, self._port, key)
            result = super().pop(key)
            _publish_patch(
                self._config,
                _make_patch(
                    op="remove",
                    path=full_path,
                    value=None,
                    old_value=old_value,
                    port=child_port,
                ),
            )
            return result
        elif default:
            return default[0]
        else:
            raise KeyError(key)

    def popitem(self) -> tuple[K, V]:
        """Remove and return a (key, value) pair as a 2-tuple.

        Emits a 'remove' patch for the removed item.
        """
        key, value = super().popitem()
        full_path = _make_path(self._path, key)
        child_port = _resolve_child_port(self._config, self._port, key)
        _publish_patch(
            self._config,
            _make_patch(
                op="remove",
                path=full_path,
                value=None,
                old_value=value,
                port=child_port,
            ),
        )
        return key, value

    @_one_update
    def clear(self) -> None:
        """Remove all items from the dictionary.

        Emits a 'remove' patch for each key.
        """
        keys = list(self.keys())
        for key in keys:
            del self[key]

    @overload
    def setdefault(
        self: "EventedDict[K, _D | None]", key: K, default: None = None
    ) -> _D | None: ...

    @overload
    def setdefault(self, key: K, default: V) -> V: ...

    def setdefault(self, key: K, default: object = None) -> object:
        """Insert key with a value of default if key is not in the dictionary.

        Emits an 'add' patch only if the key was not present.
        """
        if key not in self:
            self[key] = cast(V, default)  # dict.setdefault stores None too
        return self[key]

    @_one_update
    def update(
        self,
        other: "SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] | None" = None,
        **kwargs: V,
    ) -> None:
        """Update the dictionary with key/value pairs.

        Emits 'add' or 'replace' patches for each key.
        """
        if other is not None:
            if hasattr(other, "keys"):
                mapping = cast("SupportsKeysAndGetItem[K, V]", other)
                for k in mapping.keys():
                    self[k] = mapping[k]
            else:
                for k, v in cast("Iterable[tuple[K, V]]", other):
                    self[k] = v
        for name, value in kwargs.items():
            # Keyword arguments are str by construction, which only satisfies K when K is str --
            # the same unsoundness typeshed confines to its `dict[str, V]` overload.
            self[cast("K", name)] = value


# --- JSON Patch Compliant EventedList (RFC 6902) ---

T = TypeVar("T")
S = TypeVar("S")


class EventedList(list[T], Generic[T]):
    """A list wrapper that emits JSON Patch operations on modification.

    Supports all standard list mutation methods with proper JSON Patch semantics:
    - add: When an item is inserted (uses index or '/-' for append)
    - remove: When an item is deleted
    - replace: When an existing item is changed

    Per RFC 6902, the special path element '-' refers to the end of the array
    for 'add' operations.
    """

    def __init__(
        self,
        iterable: Iterable[T],
        config: StateConfig,
        path: str,
        port: ReturnPortInput | None,
    ) -> None:
        super().__init__(iterable)
        self._config = config
        self._path = path
        self._port = port

    def _reindex_items(self, start_index: int) -> None:
        """Update the internal path references for items after a shift.

        This is necessary after insert/remove operations that shift indices.
        """
        for i in range(start_index, len(self)):
            item = super().__getitem__(i)
            if hasattr(item, "_event_path"):
                new_path = _make_path(self._path, i)
                object.__setattr__(item, "_event_path", new_path)
            elif isinstance(item, (EventedDict, EventedList)):
                item._path = _make_path(self._path, i)

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    @_one_update
    def __setitem__(self, index: SupportsIndex | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            # Expand slice assignment into the single-index primitives so the
            # emitted patches are valid RFC-6902 operations: remove the old
            # slice (highest index first) then insert the new values in order.
            indices = list(range(*index.indices(len(self))))
            new_values = list(cast("Iterable[T]", value))
            if index.step not in (None, 1):
                # Extended slices cannot change length: replace slot by slot.
                if len(indices) != len(new_values):
                    raise ValueError(
                        f"attempt to assign sequence of size {len(new_values)} "
                        f"to extended slice of size {len(indices)}"
                    )
                for idx, item in zip(indices, new_values, strict=True):
                    self[idx] = item
                return
            for idx in reversed(indices):
                del self[idx]
            insert_at = indices[0] if indices else index.indices(len(self))[0]
            for offset, item in enumerate(new_values):
                self.insert(insert_at + offset, item)
        else:
            idx = index.__index__()
            full_path = _make_path(self._path, idx)
            old_value = self[idx]
            child_port = _resolve_child_port(self._config, self._port, idx)

            # Wrap new value
            item = make_evented(
                cast("T", value), self._config, full_path, port=child_port
            )
            super().__setitem__(idx, item)
            _publish_patch(
                self._config,
                _make_patch(
                    op="replace",
                    path=full_path,
                    value=value,
                    old_value=old_value,
                    port=child_port,
                ),
            )

    @_one_update
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            # Handle slice deletion
            indices = sorted(range(*index.indices(len(self))), reverse=True)
            for idx in indices:
                del self[idx]
        else:
            idx = index.__index__()
            full_path = _make_path(self._path, idx)
            old_value = self[idx]
            child_port = _resolve_child_port(self._config, self._port, idx)

            super().__delitem__(idx)
            _publish_patch(
                self._config,
                _make_patch(
                    op="remove",
                    path=full_path,
                    value=None,
                    old_value=old_value,
                    port=child_port,
                ),
            )
            # Reindex items after the deleted position
            self._reindex_items(idx)

    def append(self, item: T) -> None:
        """Append item to end of list.

        Per RFC 6902, uses '/-' to indicate appending to end of array.
        """
        # Use the special '-' path element for array append per RFC 6902
        append_path = f"{self._path}/-"

        # The actual index for the evented item's path reference
        actual_index = len(self)
        actual_path = _make_path(self._path, actual_index)
        child_port = _resolve_child_port(self._config, self._port, actual_index)

        item = make_evented(item, self._config, actual_path, port=child_port)
        super().append(item)
        _publish_patch(
            self._config,
            _make_patch(
                op="add", path=append_path, value=item, old_value=None, port=child_port
            ),
        )

    def insert(self, index: SupportsIndex, item: T) -> None:
        """Insert item before index.

        Emits an 'add' patch at the specified index.
        """
        # Normalize negative index
        idx = index.__index__()
        if idx < 0:
            idx = max(0, len(self) + idx)
        idx = min(idx, len(self))

        full_path = _make_path(self._path, idx)
        child_port = _resolve_child_port(self._config, self._port, idx)

        item = make_evented(item, self._config, full_path, port=child_port)
        super().insert(idx, item)
        _publish_patch(
            self._config,
            _make_patch(
                op="add", path=full_path, value=item, old_value=None, port=child_port
            ),
        )
        # Reindex items after the inserted position
        self._reindex_items(idx + 1)

    @_one_update
    def extend(self, items: Iterable[T]) -> None:
        """Extend list by appending elements from the iterable.

        Emits an 'add' patch for each item.
        """
        for item in items:
            self.append(item)

    def pop(self, index: SupportsIndex = -1) -> T:  # type: ignore[override]
        """Remove and return item at index (default last).

        Emits a 'remove' patch.
        """
        # Convert to int
        idx: int = index.__index__() if hasattr(index, "__index__") else int(index)  # type: ignore
        # Normalize negative index
        if idx < 0:
            idx = len(self) + idx

        full_path = _make_path(self._path, idx)
        old_value = self[idx]
        child_port = _resolve_child_port(self._config, self._port, idx)

        result = super().pop(idx)
        _publish_patch(
            self._config,
            _make_patch(
                op="remove",
                path=full_path,
                value=None,
                old_value=old_value,
                port=child_port,
            ),
        )
        # Reindex items after the removed position
        self._reindex_items(idx)
        return result

    def remove(self, item: T) -> None:
        """Remove first occurrence of item.

        Emits a 'remove' patch at the item's index.
        """
        index = self.index(item)
        del self[index]

    @_one_update
    def clear(self) -> None:
        """Remove all items from list.

        Emits a 'remove' patch for each item, from end to start.
        """
        while len(self) > 0:
            self.pop()

    @_one_update
    def _reorder(self, operation: Callable[[], None]) -> None:
        """Run an in-place reordering and emit 'replace' patches for moved slots."""
        old_items = list(self)
        operation()

        for i in range(len(self)):
            if old_items[i] != self[i]:
                full_path = _make_path(self._path, i)
                _publish_patch(
                    self._config,
                    _make_patch(
                        op="replace",
                        path=full_path,
                        value=self[i],
                        old_value=old_items[i],
                        port=_resolve_child_port(self._config, self._port, i),
                    ),
                )
        self._reindex_items(0)

    def reverse(self) -> None:
        """Reverse list in place.

        Emits 'replace' patches for each changed position.
        """
        self._reorder(super().reverse)

    @overload
    def sort(
        self: "EventedList[SupportsRichComparison]",
        *,
        key: None = None,
        reverse: bool = False,
    ) -> None: ...
    @overload
    def sort(
        self, *, key: "Callable[[T], SupportsRichComparison]", reverse: bool = False
    ) -> None: ...

    def sort(
        self,
        *,
        key: "Callable[[T], SupportsRichComparison] | None" = None,
        reverse: bool = False,
    ) -> None:
        """Sort list in place.

        Emits 'replace' patches for each changed position.
        """

        def _sort() -> None:
            # `key=None` and `key=<callable>` are separate overloads on `list.sort`, so the two
            # cases are spelled out rather than passing an optional through one of them.
            if key is None:
                list.sort(cast("list[SupportsRichComparison]", self), reverse=reverse)
            else:
                super(EventedList, self).sort(key=key, reverse=reverse)

        self._reorder(_sort)

    @_one_update
    def __iadd__(self, other: Iterable[T]) -> "EventedList[T]":
        """Implement += operator."""
        self.extend(other)
        return self

    @_one_update
    def __imul__(self, n: SupportsIndex) -> "EventedList[T]":  # type: ignore[override]
        """Implement *= operator."""
        count = n.__index__() if hasattr(n, "__index__") else int(n)  # type: ignore
        if count <= 0:
            self.clear()
        else:
            original = list(self)
            for _ in range(count - 1):
                self.extend(original)
        return self


# --- The Recursive Factory ---


def make_evented(
    obj: S,
    config: StateConfig,
    path: str = "",
    port: ReturnPortInput | None = None,
) -> S:
    """
    Recursively converts Dataclasses, Dicts, and Lists into event-emitting objects.
    """
    # A container read through a view (``state.a = state.b``) is stored as a copy of
    # the container it shows, at its own path: never the view, and never an alias.
    obj = unwrapped(obj)

    # CASE A: Dictionary
    if isinstance(obj, dict):
        # Recursively wrap items inside the dict first
        wrapped_data = {}
        for k, v in obj.items():
            child_path = _make_path(path, k)
            wrapped_data[k] = make_evented(
                v,
                config,
                child_path,
                port=_resolve_child_port(config, port, k),
            )

        # An evented dict is a dict of the same items: still what came in.
        return cast(S, EventedDict(wrapped_data, config, path, port))

    # CASE B: List
    if isinstance(obj, list):
        wrapped_items = [
            make_evented(
                item,
                config,
                path=_make_path(path, i),
                port=_resolve_child_port(config, port, i),
            )
            for i, item in enumerate(obj)
        ]
        # An evented list is a list of the same items: still what came in.
        return cast(S, EventedList(wrapped_items, config, path, port))

    # CASE C: Dataclass
    if dataclasses.is_dataclass(obj):
        # We modify the object IN-PLACE by changing its class to a dynamic subclass

        # 1. Recursively wrap children fields
        for field in dataclasses.fields(obj):
            val = getattr(obj, field.name)
            child_path = _make_path(path, field.name)

            wrapped_val = make_evented(
                val,
                config,
                path=child_path,
                port=_resolve_child_port(config, port, field.name),
            )
            # Use object.__setattr__ to bypass any existing hooks
            object.__setattr__(obj, field.name, wrapped_val)

        # 2. Check if already patched (optimization)
        if hasattr(obj, "_is_evented_wrapper"):
            return obj

        # 3. Create the interceptor hook
        def setattr_hook(self: _EventedInstance, name: str, value: object) -> None:
            if name.startswith("_"):
                super(EventedClass, self).__setattr__(name, value)  # pyright: ignore[reportArgumentType] -- self is an instance of the class type() makes below
                return

            old_value = getattr(self, name, None)

            if old_value != value:
                # Calculate path using JSON Pointer format
                base = self._event_path
                current_path = _make_path(base, name)
                current_port = _resolve_child_port(
                    self._event_config, self._event_port, name
                )

                # Wrap the new value immediately!
                value = make_evented(
                    value,
                    self._event_config,
                    path=current_path,
                    port=current_port,
                )

                super(EventedClass, self).__setattr__(name, value)  # pyright: ignore[reportArgumentType] -- self is an instance of the class type() makes below

                # Publish the patch
                _publish_patch(
                    self._event_config,
                    _make_patch(
                        op="replace",
                        path=current_path,
                        value=value,
                        old_value=old_value,
                        port=current_port,
                    ),
                )

        # 4. Create dynamic subclass
        original_cls = obj.__class__
        EventedClass = type(
            f"Evented{original_cls.__name__}",
            (original_cls,),
            {
                "__setattr__": setattr_hook,
                "_is_evented_wrapper": True,
                "__rekuest__config__": config,
            },
        )

        # 5. Swizzle
        obj.__class__ = EventedClass
        object.__setattr__(obj, "_event_config", config)
        object.__setattr__(obj, "_event_path", path)
        object.__setattr__(obj, "_event_port", port)

        return obj

    # CASE D: Primitive
    return obj
