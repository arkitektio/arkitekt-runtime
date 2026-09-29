"""A protocol method annotated as a generator streams every yield of the remote action.

``acall`` on a streaming action keeps only its last yield; that is why the proxy's
``__call__`` picks the stream for a generator method.
"""

from collections.abc import AsyncGenerator, AsyncIterator, Generator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from arkitekt_runtime.actors.dependency import AgentMethodProxy
from arkitekt_runtime.types import TaskEventKind
from arkitekt_spec.declare.declare import DeclaredAgentAction
from arkitekt_spec.declare.structures.registry import StructureRegistry


@dataclass
class Event:
    kind: TaskEventKind
    returns: dict | None = None
    message: str | None = None


class WordsPostman:
    """Answers every call with three yields, and remembers what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def aassign(self, **kwargs: Any) -> AsyncGenerator[Event, None]:  # noqa: ANN401
        self.calls.append(kwargs)
        for word in ("hello", "from", "testo"):
            yield Event(TaskEventKind.YIELD, returns={"return0": word})
        yield Event(TaskEventKind.COMPLETED)


class Streams(Protocol):
    def stream_words(self, text: str) -> Generator[str, None, None]:
        """Stream Words"""
        ...

    async def count(self, to: int) -> AsyncIterator[str]:
        """Count"""
        ...


def _proxy(method: Any, postman: WordsPostman) -> AgentMethodProxy:  # noqa: ANN401
    registry = StructureRegistry()
    action = DeclaredAgentAction(method, "testo", method.__name__, registry, app="testo")
    return AgentMethodProxy(
        "testo",
        method.__name__,
        action,
        task=SimpleNamespace(assignment=None),  # type: ignore[arg-type]
        agent=SimpleNamespace(caller_postman=postman),  # type: ignore[arg-type]
        structure_registry=registry,
    )


async def test_an_async_generator_method_streams_every_yield() -> None:
    postman = WordsPostman()
    count = _proxy(Streams.count, postman)

    assert [word async for word in count(to=3)] == ["hello", "from", "testo"]
    assert postman.calls[0]["dependency"] == "testo" and postman.calls[0]["method"] == "count"
    assert postman.calls[0]["args"] == {"to": 3}


async def test_aiterate_streams_a_sync_declared_generator_too() -> None:
    words = _proxy(Streams.stream_words, WordsPostman())

    assert [word async for word in words.aiterate("a b c")] == ["hello", "from", "testo"]


async def test_acall_on_a_stream_keeps_only_the_last_yield() -> None:
    words = _proxy(Streams.stream_words, WordsPostman())

    assert await words.acall("a b c") == "testo"
