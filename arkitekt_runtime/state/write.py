"""A task's writeable view of a state.

Many tasks write the same state instance, and every change has to say which task
made it (the patch's correlation) and which locks it holds (a state can require
some). A view carries that, explicitly: each task gets its own view, made for its
assignment, and a change made through it counts as that task's.

Like :mod:`rekuest.state.readonly`, a view is a *subclass of the state's own class
sharing its instance dict*: ``isinstance(view, MyState)`` holds and reads are live.
Assignments go through the evented ``__setattr__`` while the view's mutation is in
progress; nested lists, dicts and dataclasses are wrapped on the way out, so
``state.items.append(...)`` counts as the task's too.

Known limit, as for read-only views: a *method defined on the state class* that
mutates ``self`` writes through the underlying object, as a change made outside a
task (no correlation, no locks held).
"""

import weakref
from collections.abc import Iterator, MutableMapping, MutableSequence
from typing import Any, TypeVar

from arkitekt_runtime.state.observable import Mutation, StateConfig, config_of

__all__ = ["write_view"]

T = TypeVar("T")

_LIST_MUTATORS = frozenset(
    {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse"}
)
_DICT_MUTATORS = frozenset({"update", "setdefault", "pop", "popitem", "clear"})


class _WriteContainer:
    """A live container view whose changes count as one task's mutation."""

    __slots__ = ("_target", "_mutation", "_config")

    _mutators: frozenset[str] = frozenset()

    def __init__(self, target: Any, mutation: Mutation, config: StateConfig) -> None:  # noqa: ANN401
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_mutation", mutation)
        object.__setattr__(self, "_config", config)

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        return _wrap(self._target[key], self._mutation)

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        with self._config.mutating(self._mutation):
            self._target[key] = value

    def __delitem__(self, key: Any) -> None:  # noqa: ANN401
        with self._config.mutating(self._mutation):
            del self._target[key]

    def __len__(self) -> int:
        return len(self._target)

    def __contains__(self, item: object) -> bool:
        return item in self._target

    def __repr__(self) -> str:
        return repr(self._target)

    def __eq__(self, other: object) -> bool:
        target = other._target if isinstance(other, _WriteContainer) else other
        return self._target == target

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        attribute = getattr(object.__getattribute__(self, "_target"), name)
        if name in type(self)._mutators and callable(attribute):
            config, mutation = self._config, self._mutation

            def mutate(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                with config.mutating(mutation):
                    return attribute(*args, **kwargs)

            return mutate
        return attribute


class _WriteList(_WriteContainer, MutableSequence[Any]):
    __slots__ = ()
    _mutators = _LIST_MUTATORS

    def __iter__(self) -> Iterator[Any]:
        return (_wrap(item, self._mutation) for item in self._target)

    def insert(self, index: int, value: Any) -> None:  # noqa: ANN401
        with self._config.mutating(self._mutation):
            self._target.insert(index, value)

    def __iadd__(self, other: Any) -> "_WriteList":  # noqa: ANN401
        with self._config.mutating(self._mutation):
            self._target.extend(other)
        return self


class _WriteDict(_WriteContainer, MutableMapping[Any, Any]):
    __slots__ = ()
    _mutators = _DICT_MUTATORS

    def __iter__(self) -> Iterator[Any]:
        return iter(self._target)


def _wrap(value: Any, mutation: Mutation) -> Any:  # noqa: ANN401
    """Wrap the evented parts of a state, so changes to them count as the task's."""
    config = config_of(value)
    if config is None:
        return value
    if isinstance(value, dict):
        return _WriteDict(value, mutation, config)
    if isinstance(value, list):
        return _WriteList(value, mutation, config)
    if hasattr(value, "__dict__"):
        return write_view(value, mutation)
    return value


_view_classes: dict[type, type] = {}


def _write_class(cls: type) -> type:
    cached = _view_classes.get(cls)
    if cached is not None:
        return cached

    def __setattr__(self: Any, name: str, value: Any) -> None:  # noqa: ANN401, N807
        config = object.__getattribute__(self, "_event_config")
        mutation = object.__getattribute__(self, "__dict__")["__rekuest_mutations__"][id(self)]
        with config.mutating(mutation):
            cls.__setattr__(self, name, value)

    def __getattribute__(self: Any, name: str) -> Any:  # noqa: ANN401, N807
        value = object.__getattribute__(self, name)
        if name.startswith("_"):
            return value
        mutations = object.__getattribute__(self, "__dict__")["__rekuest_mutations__"]
        return _wrap(value, mutations[id(self)])

    view_cls = type(
        f"Writing{cls.__name__}",
        (cls,),
        {
            "__setattr__": __setattr__,
            "__getattribute__": __getattribute__,
            "__hash__": None,
            "__doc__": f"A task's writeable view of {cls.__name__}.",
        },
    )
    _view_classes[cls] = view_cls
    return view_cls


def write_view(state: T, mutation: Mutation) -> T:
    """``state``, with every change made through the result counting as ``mutation``.

    Falls back to the state itself when it is not evented or has no instance dict to
    share (a ``slots=True`` dataclass): changes then count as made outside a task.
    """
    if config_of(state) is None or not hasattr(state, "__dict__"):
        return state
    view_cls = _write_class(type(state))
    view = object.__new__(view_cls)
    object.__setattr__(view, "__dict__", state.__dict__)
    # The view shares the instance dict, so its mutation is kept there, per view,
    # and dropped again when the view is (the assignment ends).
    mutations = state.__dict__.setdefault("__rekuest_mutations__", {})
    mutations[id(view)] = mutation
    weakref.finalize(view, mutations.pop, id(view), None)
    return view  # type: ignore[return-value]
