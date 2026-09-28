import dataclasses
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload, SupportsIndex
from collections.abc import Callable, Iterable

import contextlib
import threading
from collections.abc import Iterator

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem, SupportsRichComparison

from arkitekt_spec.actions import (
    ReturnPortInput,
    StateDefinitionInput,
)
from arkitekt_runtime.state.gate import TaskClosedError, TaskGate
from arkitekt_spec.declare.state.publish import Patch, StateHolder
from arkitekt_spec.declare.structures.types import StateDeclaration

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


@dataclasses.dataclass(frozen=True)
class Mutation:
    """Who is changing a state: the task (for the patches' correlation) and the
    locks it holds. ``locks=None`` means unrestricted: a state being constructed.

    With a ``gate`` (the task's), a change is refused -- before anything is
    changed -- once the task's end has been reported.
    """

    correlation_id: str | None = None
    locks: frozenset[str] | None = frozenset()
    gate: TaskGate | None = dataclasses.field(default=None, compare=False)


UNRESTRICTED = Mutation(locks=None)
"""What a state is changed as before an agent adopts it (its startup hook builds it)."""

UNLOCKED = Mutation()
"""What an adopted state is changed as outside a task: no task, no locks held."""


@dataclasses.dataclass
class StateConfig:
    """Configuration for evented objects, storing the state interface name and schema.

    One per state *instance*, shared by all of its evented children, so it is also
    where that instance's publisher and the mutation in progress live.
    """

    state_name: str
    definition: StateDefinitionInput
    publish_interval: float = (
        0.1  # Optional: Minimum interval between patches to prevent flooding
    )
    required_locks: list[str] = dataclasses.field(default_factory=list)
    publisher: StateHolder | None = dataclasses.field(default=None, compare=False)
    """Where patches go: the agent that adopted the state. None until then."""
    default_mutation: Mutation = dataclasses.field(default=UNRESTRICTED, compare=False)
    """What a change made on the object itself (not through a task's view) counts as."""
    active: Mutation | None = dataclasses.field(default=None, init=False, compare=False)
    """The mutation a task's write view is making right now, for one operation."""
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
            publish_interval=declaration.publish_interval,
            required_locks=list(declaration.required_locks),
        )

    @property
    def mutation(self) -> Mutation:
        return self.active or self.default_mutation

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
                yield
            finally:
                self.active = previous
                if gate is not None:
                    gate.leave()


def config_of(obj: Any) -> StateConfig | None:  # noqa: ANN401
    """The config of an evented state object or container, if it is one."""
    return getattr(obj, "_event_config", None) or getattr(obj, "_config", None)


def evented(obj: Any, declaration: StateDeclaration) -> Any:  # noqa: ANN401
    """Make ``obj`` an evented instance of the declared state, with a config of its own.

    What an agent does to a startup hook's return before adopting it. A plain
    instance of the class is nothing but a dataclass until then: the class is
    not changed by being registered, so this is where the rules of the app that
    adopts it are attached.
    """
    return make_evented(obj, StateConfig.for_declaration(declaration), "")


def adopt(obj: Any, publisher: StateHolder) -> None:  # noqa: ANN401
    """An agent takes a state over: its patches go to ``publisher`` from now on, and a
    change made outside a task holds no locks."""
    config = config_of(obj)
    if config is None:
        return
    config.publisher = publisher
    config.default_mutation = UNLOCKED


def _publish_patch(config: StateConfig, patch: Patch) -> None:
    """Publish a patch to the state's publisher, if an agent has adopted it.

    It is stamped with the task making the change, which is the mutation in
    progress (a task's write view), not something looked up.
    """
    patch.correlation_id = config.mutation.correlation_id
    if config.publisher is not None:
        config.publisher.publish_patch(config.state_name, patch)


def _resolve_child_port(
    config: StateConfig,
    parent_port: ReturnPortInput | None,
    key: str | int,
) -> ReturnPortInput | None:
    """Resolve the schema port for a direct child without walking a JSON path."""
    if isinstance(key, int):
        if parent_port and parent_port.children:
            return parent_port.children[0]
        return None

    search_scope = (
        config.definition.ports if parent_port is None else (parent_port.children or [])
    )
    return next((port for port in search_scope if port.key == key), None)


def _make_patch(
    *,
    op: str,
    path: str,
    value: Any,
    old_value: Any,
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


# --- JSON Patch Compliant EventedDict (RFC 6902) ---

#: Bound to what a JSON Pointer segment can be: every key is handed to `_make_path`, so a
#: mapping keyed by anything else cannot be described as a patch path in the first place.
K = TypeVar("K", bound="str | int")
V = TypeVar("V")
#: The fallback a caller hands to `pop`, which need not be one of the mapping's own values.
D = TypeVar("D")


def _require_locks(config: StateConfig, path: str) -> None:
    """Raise unless the mutation in progress holds every lock the state requires."""
    acquired_locks = config.mutation.locks
    if acquired_locks is None:
        return
    missing_locks = [
        lock for lock in config.required_locks if lock not in acquired_locks
    ]
    if missing_locks:
        raise RuntimeError(
            f"Cannot modify state '{config.state_name}' at path '{path}' without required locks: {missing_locks}"
        )


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
    ):
        super().__init__(data)
        self._config = config
        self._path = path
        self._port = port

    def __setitem__(self, key: K, value: V) -> None:
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
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

    def clear(self) -> None:
        """Remove all items from the dictionary.

        Emits a 'remove' patch for each key.
        """
        _require_locks(self._config, self._path)
        keys = list(self.keys())
        for key in keys:
            del self[key]

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Insert key with a value of default if key is not in the dictionary.

        Emits an 'add' patch only if the key was not present.
        """
        _require_locks(self._config, self._path)
        if key not in self:
            self[key] = default
        return self[key]

    def update(
        self,
        other: "SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] | None" = None,
        **kwargs: V,
    ) -> None:
        """Update the dictionary with key/value pairs.

        Emits 'add' or 'replace' patches for each key.
        """
        _require_locks(self._config, self._path)
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
    ):
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

    def __setitem__(self, index: SupportsIndex | slice, value: T | Iterable[T]) -> None:
        _require_locks(self._config, self._path)
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
                for idx, item in zip(indices, new_values):
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
            item = make_evented(cast("T", value), self._config, full_path, port=child_port)
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

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
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

    def extend(self, items: Iterable[T]) -> None:
        """Extend list by appending elements from the iterable.

        Emits an 'add' patch for each item.
        """
        _require_locks(self._config, self._path)
        for item in items:
            self.append(item)

    def pop(self, index: SupportsIndex = -1) -> T:  # type: ignore[override]
        """Remove and return item at index (default last).

        Emits a 'remove' patch.
        """
        _require_locks(self._config, self._path)
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
        _require_locks(self._config, self._path)
        index = self.index(item)
        del self[index]

    def clear(self) -> None:
        """Remove all items from list.

        Emits a 'remove' patch for each item, from end to start.
        """
        _require_locks(self._config, self._path)
        while len(self) > 0:
            self.pop()

    def _reorder(self, operation: Callable[[], None]) -> None:
        """Run an in-place reordering and emit 'replace' patches for moved slots."""
        _require_locks(self._config, self._path)
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

    def __iadd__(self, other: Iterable[T]) -> "EventedList[T]":
        """Implement += operator."""
        _require_locks(self._config, self._path)
        self.extend(other)
        return self

    def __imul__(self, n: SupportsIndex) -> "EventedList[T]":  # type: ignore[override]
        """Implement *= operator."""
        _require_locks(self._config, self._path)
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
    obj: Any,
    config: StateConfig,
    path: str = "",
    port: ReturnPortInput | None = None,
) -> Any:
    """
    Recursively converts Dataclasses, Dicts, and Lists into event-emitting objects.
    """

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

        return EventedDict(wrapped_data, config, path, port)

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
        return EventedList(wrapped_items, config, path, port)

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
        def setattr_hook(self, name, value):
            _require_locks(self._event_config, self._event_path)
            if name.startswith("_"):
                super(EventedClass, self).__setattr__(name, value)
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

                super(EventedClass, self).__setattr__(name, value)

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
