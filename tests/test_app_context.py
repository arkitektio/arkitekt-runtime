"""The app context: one declared class, required of every run, injected by annotation.

An app declares at most one class as its app context. Whatever starts the agent
must hand it an instance of that class (and nothing at all to an app that declares
none); the agent refuses to start otherwise, so no hook or action ever sees the
wrong thing. Hooks and actions take it by annotating a parameter with the class.
"""

from typing import Any

import pytest

from arkitekt_runtime import messages
from arkitekt_runtime.agents.base import BaseAgent
from arkitekt_spec.declare.agents.errors import StateRequirementsNotMet
from arkitekt_spec.declare.agents.hooks.errors import StartupHookError
from arkitekt_spec.declare.app import AppRegistry
from arkitekt_spec.declare.errors import AppContextError
from arkitekt_runtime.task import Task

from .agent_helpers import run_assignment
from .memory_transport import MemoryAgentTransport


class Config:
    def __init__(self, label: str = "cfg") -> None:
        self.label = label


class OtherConfig:
    pass


def _agent(registry: AppRegistry | None = None) -> BaseAgent:
    return BaseAgent(name="ctx", transport=MemoryAgentTransport(), app_registry=registry or AppRegistry())


def _assign(interface: str, **args: Any) -> messages.Assign:
    return messages.Assign(
        task=f"task-{interface}", interface=interface, args=args, implementation="impl-1",
        action="action-1", reference="ref-1", user="user-1", org="org-1",
    )


# ---------------------------------------------------------------------- agent


@pytest.mark.asyncio
async def test_the_agent_refuses_to_start_without_its_declared_context() -> None:
    registry = AppRegistry()
    registry.app_context(Config)
    agent = _agent(registry)
    before = agent.current_session

    with pytest.raises(AppContextError, match="none was given"):
        await agent.astart(app_context=None)
    with pytest.raises(AppContextError, match="OtherConfig"):
        await agent.astart(app_context=OtherConfig())
    assert agent.current_session == before  # nothing was minted for a doomed start

    await agent.astart(app_context=Config("ok"))
    assert agent.app_context.label == "ok"


@pytest.mark.asyncio
async def test_an_agent_declaring_no_context_refuses_one() -> None:
    agent = _agent()
    with pytest.raises(AppContextError, match="declares no app context"):
        await agent.astart(app_context=Config())


@pytest.mark.asyncio
async def test_a_startup_hook_gets_the_declared_kind_or_nothing_runs() -> None:
    registry = AppRegistry()
    registry.app_context(Config)
    seen: list[str] = []

    @registry.startup
    async def boot(config: Config):  # noqa: ANN202
        """Boot."""
        seen.append(config.label)

    agent = _agent(registry)
    agent.collect_from_registry()
    with pytest.raises(StartupHookError) as failed:
        await agent.arun_startup_hooks(app_context=None)
    assert isinstance(failed.value.__cause__, StateRequirementsNotMet)
    assert "Config" in str(failed.value.__cause__)
    assert seen == []

    await agent.arun_startup_hooks(app_context=Config("booted"))
    assert seen == ["booted"]


# -------------------------------------------------------------------- actions


def test_an_action_parameter_of_the_context_class_is_not_a_port() -> None:
    registry = AppRegistry()
    registry.app_context(Config)

    @registry.register
    def scan(task: Task, config: Config, exposure: float) -> str:
        """Scan."""
        return config.label

    definition = registry.implementations["scan"].definition
    assert [arg.key for arg in definition.args] == ["exposure"]


@pytest.mark.asyncio
async def test_an_action_is_handed_the_context_the_agent_started_with() -> None:
    registry = AppRegistry()
    registry.app_context(Config)

    @registry.register
    def scan(config: Config, exposure: float) -> str:
        """Scan."""
        return f"{config.label}@{exposure}"

    @registry.register
    def in_thread(config: Config) -> str:
        """The same, from a worker thread."""
        return config.label

    agent = _agent(registry)
    await agent.astart(app_context=Config("live"))
    assert await run_assignment(agent, _assign("scan", exposure=0.5)) == {"return0": "live@0.5"}

    threaded = _agent(registry)
    await threaded.astart(app_context=Config("thread"))
    assert await run_assignment(threaded, _assign("in_thread")) == {"return0": "thread"}
