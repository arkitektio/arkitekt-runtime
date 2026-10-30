"""Building the actors a declaration's implementations run as.

A declaration records each registered function and how it was registered
(:class:`~arkitekt_spec.declare.actors.types.DeclaredImplementation`); it holds no
actors, because an actor needs an agent. The agent builds them here, against the
structure registry of the registry it serves -- a run's snapshot, whose clients are
bound -- with the declared actifier, or :func:`~rekuest.actors.actify.reactify`.
"""

from typing import TYPE_CHECKING

from arkitekt_runtime.actors.actify import reactify
from arkitekt_runtime.actors.types import Actor, ActorBuilder

if TYPE_CHECKING:
    from arkitekt_runtime.actors.types import Agent
    from arkitekt_spec.declare.actors.types import RegisterConfig
    from arkitekt_spec.declare.app import AppRegistry


class DeclaredActorBuilder:
    """The actifier's builder, with what the registration says the action may do set on
    every actor it builds (call, ask or inform at run time), whichever actifier it was: an
    actor that missed it would refuse what was declared."""

    def __init__(self, builder: ActorBuilder, config: "RegisterConfig") -> None:
        self.builder = builder
        self.config = config

    def __call__(self, agent: "Agent") -> "Actor":
        actor = self.builder(agent=agent)
        actor.execution = self.config.execution
        actor.dynamic_asks = self.config.dynamic_asks
        actor.dynamic_calls = self.config.dynamic_calls
        actor.dynamic_informs = self.config.dynamic_informs
        return actor


def actor_builder_for(registry: "AppRegistry", interface: str) -> DeclaredActorBuilder:
    """The actor builder for the implementation at ``interface``.

    Raises:
        KeyError: If nothing is registered at ``interface``.
    """
    declared = registry.get_declared_implementation(interface)
    actifier = declared.actifier or reactify
    _, _, builder = actifier(declared.function, registry.structure_registry, declared.config)
    return DeclaredActorBuilder(builder, declared.config)
