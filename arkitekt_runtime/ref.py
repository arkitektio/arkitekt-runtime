"""What an untyped call returns: references, turned into objects on request.

``rekuest.call(blur, image)`` and a workflow's ``task.call(...)`` call an action found
at run time, so nothing says what comes back. A structure comes back as a :class:`Ref`
(its identifier and id), a model as its plain value; nothing is fetched. The caller
says what it expects::

    blurred = rekuest.call(blur, image).to(Image)

``to`` expands through the class's own ``@structure`` mark, with the clients of the
run the reference came from -- or the ones passed to it.
"""

import types
import weakref
from typing import Any, Self, TypeVar, Union, get_args, get_origin, get_type_hints

from arkitekt_spec.declare.agents.types import BoundApp
from arkitekt_spec.declare.markers import MODEL, marker_of
from arkitekt_spec.declare.structures.declare import structure_of
from arkitekt_spec.declare.structures.registry import injected_clients_of
from koil import unkoil
from pydantic import BaseModel, ConfigDict, PrivateAttr

T = TypeVar("T")


class Ref(BaseModel):
    """A structure by reference: what it travels as, and its id.

    Equal by identifier and id alone; the run it came from is never sent or compared.
    """

    identifier: str
    id: str

    model_config = ConfigDict(frozen=True)

    _app: "weakref.ref[BoundApp] | None" = PrivateAttr(default=None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and (self.identifier, self.id) == (
            other.identifier,
            other.id,
        )

    def __hash__(self) -> int:
        return hash((self.identifier, self.id))

    def __getstate__(self) -> dict[Any, Any]:
        # A pickled reference leaves its run behind: the run is this process's.
        state = super().__getstate__()
        state["__pydantic_private__"] = {"_app": None}
        return state

    def from_app(self, app: BoundApp | None) -> Self:
        """This reference, expanding through ``app``'s clients."""
        if app is not None:
            self._app = weakref.ref(app)
        return self

    @classmethod
    def of(cls, obj: Any) -> "Ref":  # noqa: ANN401 -- any @structure instance
        """The reference to ``obj``, an instance of a ``@structure`` class, by its ``.id``.

        Raises:
            TypeError: If its class is not one.
        """
        if isinstance(obj, Ref):
            return obj
        declared = structure_of(type(obj))
        if declared is None:
            raise TypeError(
                f"A {type(obj).__name__} is not a @structure: it has no reference."
            )
        return cls(identifier=declared.identifier, id=str(obj.id))

    @classmethod
    async def aof(cls, obj: object) -> "Ref":
        """The reference to ``obj``, its id made by the declared ``shrink`` when that needs
        no client (else :meth:`of`)."""
        declared = None if isinstance(obj, Ref) else structure_of(type(obj))
        shrink = declared.ashrink if declared is not None else None
        if (
            declared is not None
            and shrink is not None
            and not injected_clients_of(shrink, declared.identifier, "shrink")
        ):
            return cls(identifier=declared.identifier, id=str(await shrink(obj)))
        return cls.of(obj)

    async def ato(self, cls: type[T], *clients: object) -> T:
        """Fetch what this refers to, as ``cls``.

        Args:
            cls: The ``@structure`` class it is expected to be.
            clients: Clients to expand with; by default the run's it came from.

        Raises:
            TypeError: If ``cls`` is not a structure, or not the one referred to.
            LookupError: If a client the expander needs is neither given nor in the run.
        """
        declared = structure_of(cls)
        if declared is None:
            raise TypeError(f"{cls.__name__} is not a @structure.")
        if declared.identifier != self.identifier:
            raise TypeError(
                f"This is a {self.identifier}, not a {declared.identifier} ({cls.__name__})."
            )
        app = self._app() if self._app is not None else None
        kwargs: dict[str, object] = {}
        for name, wanted in injected_clients_of(
            declared.aexpand, declared.identifier, "expand"
        ).items():
            client = next((c for c in clients if isinstance(c, wanted)), None)
            if client is None and app is not None:
                client = app.get(wanted)
            if client is None:
                raise LookupError(
                    f"Expanding {self.identifier} needs a {wanted.__name__}: this reference "
                    f"has no run that has one. Pass it: ref.to({cls.__name__}, {name})."
                )
            kwargs[name] = client
        return await declared.aexpand(self.id, **kwargs)  # type: ignore[call-arg]

    def to(self, cls: type[T], *clients: object) -> T:
        """Fetch what this refers to, as ``cls`` (see :meth:`ato`)."""
        return unkoil(self.ato, cls, *clients)


async def ato(value: Any, annotation: Any, *clients: object) -> Any:  # noqa: ANN401 -- shaped by the annotation
    """Turn an untyped call's result into what ``annotation`` says it is.

    Follows lists, tuples, dicts and ``X | None``: a :class:`Ref` becomes the
    ``@structure`` class it is annotated as, a dict the ``@model`` class.
    """
    origin = get_origin(annotation)
    if value is None:
        return None
    if origin in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            return await ato(value, options[0], *clients)
        return value
    if origin is list:
        (item,) = get_args(annotation)
        return [await ato(v, item, *clients) for v in value]
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple([await ato(v, args[0], *clients) for v in value])
        return tuple(
            [await ato(v, a, *clients) for v, a in zip(value, args, strict=True)]
        )
    if origin is dict:
        _, item = get_args(annotation)
        return {k: await ato(v, item, *clients) for k, v in value.items()}
    if isinstance(value, Ref) and isinstance(annotation, type):
        return await value.ato(annotation, *clients)
    if isinstance(value, dict) and marker_of(annotation, MODEL) is not None:
        hints = get_type_hints(annotation)
        return annotation(
            **{k: await ato(v, hints.get(k, Any), *clients) for k, v in value.items()}
        )
    return value


def to(value: Any, annotation: Any, *clients: object) -> Any:  # noqa: ANN401 -- shaped by the annotation
    """Turn an untyped call's result into what ``annotation`` says it is (see :func:`ato`)."""
    return unkoil(ato, value, annotation, *clients)


__all__ = ["Ref", "ato", "to"]
