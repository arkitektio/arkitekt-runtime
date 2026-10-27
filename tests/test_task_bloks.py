"""A task shows a person a blok: an ask holds on it until they answer through it, an inform
shows it while the task goes on, and a blok the server cannot make is raised at the ask.
"""

import asyncio
import json

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import MemoryAgentTransport
from arkitekt_spec.actions import (
    BlokDemoInput,
    BlokLocalInput,
    BlokQueryInput,
    BlokSubscriptionInput,
    PortKind,
    ReturnPortInput,
)
from arkitekt_spec.declare.errors import BlokRefused
from arkitekt_spec.declare.task import Task, TaskBlok

from .test_workflow_replay import _agent, _assign, _drive, _returned

PICKER = """<Html>
  <Card>
    <foreach items={args.images} let="img" key={img.object}>
      <Button onClick={task.answer(values: {image: img})}>Pick</Button>
    </foreach>
  </Card>
</Html>"""

IMAGES = [{"__identifier": "@mikro/image", "object": "7"}, {"__identifier": "@mikro/image", "object": "8"}]
IMAGE = [ReturnPortInput(key="image", kind=PortKind.STRUCTURE, identifier="@mikro/image", nullable=False)]


def picks(task: Task) -> str:
    """Asks which image."""
    try:
        decision = task.ask(PICKER, "Which image?", args={"images": IMAGES}, returns=IMAGE)
    except BlokRefused as refused:
        return f"refused:{refused.reason}"
    return f"{decision.approved}:{(decision.values or {}).get('image')}"


def shows(task: Task) -> str:
    """Shows progress and goes on."""
    task.inform(TaskBlok(key="progress_panel"), "Halfway", args={"done": 5})
    return "went on"


async def _until(agent: BaseAgent, transport: MemoryAgentTransport, interface: str, kind: type) -> tuple:
    transport.feed(_assign(interface))
    loop = _drive(agent)
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    async def sent() -> object:
        while not transport.of_type(kind):
            await asyncio.sleep(0.01)
        return transport.of_type(kind)[0]

    return loop, await asyncio.wait_for(sent(), timeout=5.0)


async def test_an_ask_holds_on_its_blok_and_returns_the_answer() -> None:
    agent, transport = _agent(picks)
    loop, held = await _until(agent, transport, "picks", messages.Held)
    assert held.message == "Which image?"
    assert held.blok == TaskBlok(source=PICKER, args={"images": IMAGES}, returns=IMAGE)

    transport.feed(messages.Decide(task="42", approved=True, values={"image": IMAGES[1]}))
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == f"True:{IMAGES[1]}"
    # Recorded like a hold, with what was answered: a resumed run gets it back.
    assert [(e.key, e.value) for e in transport.of_type(messages.Effect)] == [
        ("HOLD:1", {"approved": True, "answer": None, "auto": False, "values": {"image": IMAGES[1]}})
    ]


async def test_a_refused_blok_is_raised_at_the_ask() -> None:
    agent, transport = _agent(picks)
    loop, _ = await _until(agent, transport, "picks", messages.Held)

    transport.feed(messages.BlokRefused(task="42", reason="component 'Knob' is not registered"))
    await asyncio.wait_for(loop.__anext__(), timeout=2.0)

    assert await _returned(transport) == "refused:component 'Knob' is not registered"


async def test_an_inform_shows_its_blok_and_the_task_goes_on() -> None:
    agent, transport = _agent(shows)
    _, inform = await _until(agent, transport, "shows", messages.Inform)
    assert (inform.message, inform.blok) == ("Halfway", TaskBlok(key="progress_panel", args={"done": 5}))
    assert await _returned(transport) == "went on"


def test_a_task_blok_travels_under_its_own_names() -> None:
    """The blok goes on the wire as the spec names its fields (``catalogs``, ``module``,
    ``locals``, ``demo``, ``queries``, ``subscriptions``) and comes back the same."""
    blok = TaskBlok(
        source=PICKER,
        vocab=2,
        catalogs=["ui@1"],
        module="orkestrator",
        demo=BlokDemoInput(args={"images": IMAGES}),
        locals=[BlokLocalInput(key="note", kind=PortKind.STRING, default="", persist=True)],
        queries=[BlokQueryInput(key="positions", service="mikro", document="query Positions { positions { id } }", more="positions")],
        subscriptions=[BlokSubscriptionInput(key="live", service="mikro", document="subscription Live { live { id } }", keep=3, history="positions")],
    )
    inform = messages.Inform(task="42", message="Pick one", blok=blok)

    wire = inform.model_dump(mode="json")["blok"]

    assert (wire["vocab"], wire["catalogs"], wire["module"]) == (2, ["ui@1"], "orkestrator")
    assert wire["demo"]["args"] == {"images": IMAGES}
    assert "catalog" not in wire and "demo_state" not in wire
    assert [(local["key"], local["kind"], local["persist"]) for local in wire["locals"]] == [("note", "STRING", True)]
    assert wire["queries"][0]["more"] == "positions"
    assert (wire["subscriptions"][0]["keep"], wire["subscriptions"][0]["history"]) == (3, "positions")
    assert messages.Inform.model_validate(inform.model_dump(mode="json")).blok == blok
    held = messages.Held(task="42", blok=blok).model_dump(mode="json")["blok"]
    assert held["module"] == "orkestrator"
    assert messages.Held.model_validate(messages.Held(task="42", blok=blok).model_dump(mode="json")).blok == blok


def test_a_local_without_kind_goes_without_one() -> None:
    """The server infers a left-out kind and keeps a given one, ANY included: the spec's ANY
    default must not reach the wire, while a kind given stays."""
    blok = TaskBlok(
        source="<Html><Card><Input bind={note} /><Switch bind={on} /></Card></Html>",
        vocab=2,
        locals=[BlokLocalInput(key="note"), BlokLocalInput(key="on", kind="ANY")],
    )

    for message in (messages.Inform(task="42", blok=blok), messages.Held(task="42", blok=blok)):
        wire = json.loads(message.model_dump_json())["blok"]
        assert wire["locals"][0]["key"] == "note" and "kind" not in wire["locals"][0]
        assert wire["locals"][1]["kind"] == "ANY"
    assert messages.Inform.model_validate(messages.Inform(task="42", blok=blok).model_dump(mode="json")).blok == blok
