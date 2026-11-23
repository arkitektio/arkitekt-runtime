"""A task shows a person a blok: an ask holds on it until they answer through it, an inform
shows it while the task goes on, and a blok the server cannot make is raised at the ask.

An ask or inform is declared (an interactions protocol the action is handed: typed, named by
key) or dynamic (a blok made at run time, which only an action declaring ``dynamic_asks`` or
``dynamic_informs`` may send).
"""

import asyncio
from typing import Annotated, Any, NamedTuple, Protocol

from arkitekt_spec.actions import (
    BlokDemoInput,
    PortKind,
    ReturnPortInput,
)
from arkitekt_spec.declare.annotations import Key
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.errors import BlokRefused, DismissedError, NotDeclaredError
from arkitekt_spec.declare.interactions import ask, inform
from arkitekt_spec.declare.task import Task, TaskBlok

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import MemoryAgentTransport
from tests.agent_helpers import sent_until

from .test_workflow_replay import _agent, _assign, _drive, _returned

PICKER = """<Blok kind="task" context="panel">
  <Card>
    <foreach items={args.images} let="img" key={img.object}>
      <Button onClick={task.answer(values: {image: img})}>Pick</Button>
    </foreach>
  </Card>
</Blok>"""

IMAGES = [
    {"__identifier": "@mikro/image", "object": "7"},
    {"__identifier": "@mikro/image", "object": "8"},
]
IMAGE = [
    ReturnPortInput(
        key="image", kind=PortKind.STRUCTURE, identifier="@mikro/image", nullable=False
    )
]


def picks(task: Task) -> str:
    """Asks which image."""
    try:
        values = task.dynamic_ask(
            TaskBlok(source=PICKER, name="Which image?"),
            args={"images": IMAGES},
            returns=IMAGE,
        )
    except BlokRefused as refused:
        return f"refused:{refused.reason}"
    except DismissedError:
        return "dismissed"
    except NotDeclaredError:
        return "not declared"
    return f"answered:{values.get('image')}"


def _dynamic_agent(*workflows: Any) -> tuple[BaseAgent, MemoryAgentTransport]:
    """An agent whose workflows declare dynamic_asks."""
    registry = AppRegistry()
    for workflow in workflows:
        registry.register_workflow(workflow, dynamic_asks=True)
    transport = MemoryAgentTransport()
    agent = BaseAgent(transport=transport, app_registry=registry)
    agent.collect_from_registry()
    return agent, transport


class Picked(NamedTuple):
    name: str
    count: int


CONFIRM = """<Blok kind="task" context="panel"><Card><CardTitle text={args.question} />
  <Button label="Go" onClick={task.answer(values: {name: args.question, count: 2})} />
</Card></Blok>"""


class Asks(Protocol):
    @ask(CONFIRM)
    def pick(self, question: str, error: str = "") -> Picked:
        """Pick one."""
        ...

    @ask(
        '<Blok kind="task" context="panel"><Button label="Ok" onClick={task.answer(values: {ok: true})} /></Blok>'
    )
    def ok(self) -> Annotated[bool, Key("ok")]:
        """Go on?"""
        ...

    @inform(
        '<Blok kind="task" context="panel"><Card><CardTitle text={args.done} /></Card></Blok>'
    )
    def progress(self, done: int) -> None:
        """Halfway."""
        ...


def _declared_agent() -> tuple[BaseAgent, MemoryAgentTransport]:
    registry = AppRegistry()
    registry.register_interactions(Asks)

    def confirms(asks: Asks) -> str:
        """Asks through its declared asks."""
        try:
            picked = asks.pick(question="Which?")
        except DismissedError:
            return "dismissed"
        return f"{picked.name}:{picked.count}:{asks.ok()}"

    registry.register_workflow(confirms)
    transport = MemoryAgentTransport()
    agent = BaseAgent(transport=transport, app_registry=registry)
    agent.collect_from_registry()
    return agent, transport


def shows_dynamically(task: Task) -> str:
    """Shows a blok made now and goes on."""
    try:
        task.dynamic_inform(
            TaskBlok(
                source='<Blok kind="task" context="panel"><Text text={args.done} /></Blok>',
                name="Halfway",
            ),
            args={"done": 5},
        )
    except NotDeclaredError:
        return "not declared"
    return "went on"


def _informing_agent(dynamic: bool) -> tuple[BaseAgent, MemoryAgentTransport]:
    registry = AppRegistry()
    registry.register_interactions(Asks)

    def shows(asks: Asks) -> str:
        """Shows progress through its declared inform and goes on."""
        asks.progress(done=5)
        return "went on"

    registry.register_workflow(shows)
    registry.register_workflow(shows_dynamically, dynamic_informs=dynamic)
    transport = MemoryAgentTransport()
    agent = BaseAgent(transport=transport, app_registry=registry)
    agent.collect_from_registry()
    return agent, transport


async def _until(
    agent: BaseAgent, transport: MemoryAgentTransport, interface: str, kind: type
) -> tuple:
    transport.feed(_assign(interface))
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def sent() -> object:
        await sent_until(transport, lambda: bool(transport.of_type(kind)))
        return transport.of_type(kind)[0]

    return loop, await asyncio.wait_for(sent(), timeout=5.0)


async def test_an_ask_holds_on_its_blok_and_returns_the_answer() -> None:
    agent, transport = _dynamic_agent(picks)
    loop, held = await _until(agent, transport, "picks", messages.Held)
    assert held.message == "Which image?"
    assert held.blok == TaskBlok(
        source=PICKER, name="Which image?", args={"images": IMAGES}, returns=IMAGE
    )

    transport.feed(
        messages.Decide(task="42", approved=True, values={"image": IMAGES[1]})
    )
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == f"answered:{IMAGES[1]}"
    # Recorded like a hold, with what was answered: a resumed run gets it back.
    assert [(e.key, e.value) for e in transport.of_type(messages.Effect)] == [
        (
            "HOLD:1",
            {
                "approved": True,
                "answer": None,
                "auto": False,
                "values": {"image": IMAGES[1]},
            },
        )
    ]


async def test_a_dismissed_ask_raises() -> None:
    agent, transport = _dynamic_agent(picks)
    loop, _ = await _until(agent, transport, "picks", messages.Held)

    transport.feed(messages.Decide(task="42", approved=False))
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "dismissed"


async def test_a_dynamic_ask_needs_dynamic_asks() -> None:
    """Refused before anything is sent: the server would refuse it too."""
    agent, transport = _agent(picks)
    transport.feed(_assign("picks"))
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "not declared"
    assert not transport.of_type(messages.Held)


async def test_a_declared_ask_holds_by_key_and_answers_typed() -> None:
    agent, transport = _declared_agent()
    loop, held = await _until(agent, transport, "confirms", messages.Held)
    # Named by its key on the action, its args shrunk; the docstring is the message.
    assert (held.message, held.blok.key, held.blok.source) == (
        "Pick one.",
        "asks.pick",
        None,
    )
    assert (held.blok.args or {})["question"] == "Which?"

    transport.feed(
        messages.Decide(task="42", approved=True, values={"name": "this", "count": 3})
    )
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def second() -> object:
        await sent_until(transport, lambda: len(transport.of_type(messages.Held)) >= 2)
        return transport.of_type(messages.Held)[1]

    ok = await asyncio.wait_for(second(), timeout=5.0)
    assert (ok.message, ok.blok.key) == ("Go on?", "asks.ok")
    transport.feed(messages.Decide(task="42", approved=True, values={"ok": True}))
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "this:3:True"


def test_a_declared_ask_registers_with_its_action() -> None:
    registry = AppRegistry()
    registry.register_interactions(Asks)

    def confirms(asks: Asks) -> str:
        """Asks."""
        return ""

    registry.register_workflow(confirms)
    implementation = registry.implementations["confirms"]

    assert implementation.definition.args == ()
    assert [(ask.key, ask.description) for ask in implementation.asks or ()] == [
        ("asks.ok", "Go on?"),
        ("asks.pick", "Pick one."),
    ]
    assert [
        (inform.key, inform.description) for inform in implementation.informs or ()
    ] == [("asks.progress", "Halfway.")]
    pick = next(ask for ask in implementation.asks or () if ask.key == "asks.pick")
    assert [port.key for port in pick.args] == ["question", "error"]
    assert [(port.key, port.kind) for port in pick.returns] == [
        ("name", "STRING"),
        ("count", "INT"),
    ]


async def test_a_refused_blok_is_raised_at_the_ask() -> None:
    agent, transport = _dynamic_agent(picks)
    loop, _ = await _until(agent, transport, "picks", messages.Held)

    transport.feed(
        messages.BlokRefused(task="42", reason="component 'Knob' is not registered")
    )
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "refused:component 'Knob' is not registered"


async def test_a_declared_inform_shows_by_key_and_the_task_goes_on() -> None:
    agent, transport = _informing_agent(dynamic=False)
    _, inform = await _until(agent, transport, "shows", messages.Inform)
    assert (inform.message, inform.blok) == (
        "Halfway.",
        TaskBlok(key="asks.progress", args={"done": 5}),
    )
    assert await _returned(transport) == "went on"


async def test_a_dynamic_inform_shows_its_blok() -> None:
    agent, transport = _informing_agent(dynamic=True)
    _, inform = await _until(agent, transport, "shows_dynamically", messages.Inform)
    assert (inform.message, inform.blok.source, inform.blok.args) == (
        "Halfway",
        '<Blok kind="task" context="panel"><Text text={args.done} /></Blok>',
        {"done": 5},
    )
    assert await _returned(transport) == "went on"


async def test_a_dynamic_inform_needs_dynamic_informs() -> None:
    agent, transport = _informing_agent(dynamic=False)
    transport.feed(_assign("shows_dynamically"))
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "not declared"
    assert not transport.of_type(messages.Inform)


def test_a_task_blok_travels_under_its_own_names() -> None:
    """The blok goes on the wire as the spec names its fields; its locals, queries and
    subscriptions are nodes of its source, so they travel in it, and it comes back the same."""
    source = (
        '<Blok kind="task" context="panel"><local name="note" default="" persist />'
        '<query name="positions" service="mikro" document="query Positions { positions { id } }" more="positions" />'
        "<Card><Input bind={note} /></Card></Blok>"
    )
    blok = TaskBlok(
        source=source,
        vocab=2,
        catalog="https://arkitekt.live/catalogs/orkestrator/v1",
        demo=BlokDemoInput(args={"images": IMAGES}),
    )
    inform = messages.Inform(task="42", message="Pick one", blok=blok)

    wire = inform.model_dump(mode="json")["blok"]

    assert (wire["vocab"], wire["catalog"], wire["source"]) == (
        2,
        "https://arkitekt.live/catalogs/orkestrator/v1",
        source,
    )
    assert wire["demo"]["args"] == {"images": IMAGES}
    assert not {"locals", "queries", "subscriptions"} & set(wire)
    assert messages.Inform.model_validate(inform.model_dump(mode="json")).blok == blok
    assert (
        messages.Held.model_validate(
            messages.Held(task="42", blok=blok).model_dump(mode="json")
        ).blok
        == blok
    )
