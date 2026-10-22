"""A startup hook may leave a context out by returning ``None`` for it.

The setup then simply has no such thing: actions needing it fail when called, and
background and shutdown hooks needing it are not run. States stay required.
"""

import asyncio

import pytest

from arkitekt_runtime.local import acall_local, local_agent
from arkitekt_spec.declare.app import AppRegistry


def build_registry(ran: dict[str, asyncio.Event]) -> AppRegistry:
    registry = AppRegistry()

    @registry.state
    class Counter:
        """A counter."""

        value: int = 0

    @registry.context
    class Camera:
        """A camera this setup has."""

        def snap(self) -> int:
            return 1

    @registry.context
    class Laser:
        """A laser this setup does not have."""

        def fire(self) -> int:
            return 2

    @registry.startup
    def setup() -> tuple[Counter, Camera, Laser]:
        """Only a camera."""
        return Counter(), Camera(), None  # type: ignore[return-value]

    @registry.background
    async def watch_camera(camera: Camera) -> None:
        """Runs: the camera is there."""
        ran["camera"].set()

    @registry.background
    async def watch_laser(laser: Laser) -> None:
        """Not started: there is no laser."""
        ran["laser"].set()

    @registry.shutdown
    def cool_laser(laser: Laser) -> None:
        """Not run: there is no laser."""
        ran["cooled"].set()

    def snap(camera: Camera) -> int:
        """Snap."""
        return camera.snap()

    def fire(laser: Laser) -> int:
        """Fire."""
        return laser.fire()

    registry.register(snap)
    registry.register(fire)
    return registry


async def test_a_context_returned_as_none_is_not_provided() -> None:
    ran = {key: asyncio.Event() for key in ("camera", "laser", "cooled")}
    agent = local_agent(build_registry(ran))
    await agent.aconnect(timeout=5.0)
    try:
        await asyncio.wait_for(ran["camera"].wait(), 5.0)
        assert "camera" in agent.contexts and "laser" not in agent.contexts
        assert await acall_local(agent, "snap") == 1
        with pytest.raises(Exception, match="laser"):
            await acall_local(agent, "fire")
        await asyncio.sleep(0.05)
        assert not ran["laser"].is_set()
    finally:
        await agent.atear_down()
    assert not ran["cooled"].is_set()
