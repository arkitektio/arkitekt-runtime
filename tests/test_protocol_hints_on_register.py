"""Protocol hints on the way to the server: off the wire unless given, and in the hash when given.

takt refuses a definition field it does not know (``deny_unknown_fields``), so an
app that hints nothing must register without the field, or an older takt would
refuse every agent. And takt skips the whole reconciliation when the agent's hash
is unchanged, so a change to the hints alone has to change that hash, or it would
never reach the server.
"""

import pytest

from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_runtime.local import MemoryAgentTransport


def attach(scene: str) -> str:
    """Attach a scene."""
    return scene


def agent_with(protocols: list[str] | None) -> BaseAgent:
    agent = BaseAgent(transport=MemoryAgentTransport(), name="Test")
    agent.app_registry.register_action(attach, protocols=protocols)
    return agent


async def declared_definition(agent: BaseAgent) -> dict:
    params = await agent.aget_handshake_params()
    (implementation,) = params.declaration.implementations
    return implementation["definition"]


@pytest.mark.asyncio
async def test_without_hints_the_definition_carries_no_protocols_field() -> None:
    assert "protocols" not in await declared_definition(agent_with(None))


@pytest.mark.asyncio
async def test_with_hints_the_definition_carries_them() -> None:
    definition = await declared_definition(agent_with(["on_scene_init"]))
    assert definition["protocols"] == ["on_scene_init"]


@pytest.mark.asyncio
async def test_changing_only_the_hints_changes_the_agent_hash() -> None:
    plain = await agent_with(None).aget_hash()
    hinted = await agent_with(["on_scene_init"]).aget_hash()
    other = await agent_with(["on_scene_leave"]).aget_hash()
    assert len({plain, hinted, other}) == 3
    assert hinted == await agent_with(["on_scene_init"]).aget_hash()
