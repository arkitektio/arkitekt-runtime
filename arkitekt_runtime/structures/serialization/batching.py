"""One request for the ids of a structure that are expanded together.

A list port of N structures expands as N concurrent ``_expand`` calls, which used
to mean N requests. A structure that can fetch several ids at once
(``aexpand_many``) gets them in one: every id asked for in the same turn of the
event loop is collected, and fetched when that turn is over. Since expansion
already fans out with ``asyncio.gather`` at every level, this needs no special
case for lists: the elements of a list, the values of a dict, the fields of a
model and sibling arguments of the same structure all end up in one batch, and
nesting does not split it -- every leaf asks before the flush runs.

What does split it is leaves at *different* depths: the flush is scheduled when
the first id arrives, and a leaf one container deeper is still descending when it
runs. So it is one request per depth, not strictly one. Doing better means
knowing when the whole walk is quiescent, which the loop does not tell us; a
single list port, which is what this is for, is one request either way.

A structure without ``aexpand_many`` is expanded directly, exactly as before -- and
whether it has one is settled when the registry is bound, not here.
"""

import asyncio
from typing import Any

from arkitekt_spec.scalars import ID

from arkitekt_spec.declare.structures.types import FullFilledStructure


class StructureNotFound(LookupError):
    """The service has no object for an id that was sent to be expanded."""


class _Batch:
    def __init__(self, structure: FullFilledStructure) -> None:
        self.structure = structure
        # Keyed by id: the same object named twice is fetched once.
        self.waiting: dict[ID, asyncio.Future[Any]] = {}


class ExpandBatcher:
    """Collects the expansions of one (de)serialization call."""

    def __init__(self) -> None:
        self._open: dict[str, _Batch] = {}
        # The loop only holds weak references to tasks: without this, a flush whose
        # waiters were all cancelled could be collected mid-request.
        self._flushing: set[asyncio.Task[None]] = set()

    async def load(self, structure: FullFilledStructure, id: ID) -> Any:  # noqa: ANN401
        """Expand ``id``, together with whatever else is asked for this turn."""
        if structure.aexpand_many is None:
            return await structure.expand(id)

        batch = self._open.get(structure.identifier)
        if batch is None:
            batch = self._open[structure.identifier] = _Batch(structure)
            # Runs after everything already scheduled, which includes the sibling
            # expansions `gather` has just started. The task is created from this
            # context, so it runs under the same app as a direct expansion would.
            asyncio.get_running_loop().call_soon(self._flush, structure.identifier)

        future = batch.waiting.get(id)
        if future is None:
            future = batch.waiting[id] = asyncio.get_running_loop().create_future()
        return await future

    def _flush(self, identifier: str) -> None:
        task = asyncio.ensure_future(self._fetch(identifier))
        self._flushing.add(task)
        task.add_done_callback(self._flushing.discard)

    async def _fetch(self, identifier: str) -> None:
        batch = self._open.pop(identifier)
        ids = list(batch.waiting)
        try:
            found = await batch.structure.expand_many(ids)
        except BaseException as error:  # noqa: BLE001 -- handed to every waiter
            for future in batch.waiting.values():
                if not future.done():
                    future.set_exception(error)
                    future.exception()  # retrieved: a cancelled waiter is no leak
            if not isinstance(error, Exception):
                raise
            return

        for id, value in zip(ids, found):
            future = batch.waiting[id]
            if future.done():
                continue
            if value is None:
                future.set_exception(
                    StructureNotFound(f"There is no {identifier} with id {id!r}")
                )
                future.exception()
            else:
                future.set_result(value)


__all__ = ["ExpandBatcher", "StructureNotFound"]
