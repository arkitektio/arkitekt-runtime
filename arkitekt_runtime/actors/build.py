"""Building the actors a declaration's implementations run as.

A declaration records each registered function and how it was registered
(:class:`~arkitekt_spec.declare.actors.types.DeclaredImplementation`); it holds no
actors, because an actor needs an agent. The agent builds them here, against the
structure registry of the registry it serves -- a run's snapshot, whose clients are
bound -- with the declared actifier, or :func:`~rekuest.actors.actify.reactify`.
"""

from typing import TYPE_CHECKING

from arkitekt_runtime.actors.actify import reactify
from arkitekt_runtime.actors.types import ActorBuilder

if TYPE_CHECKING:
    from arkitekt_spec.declare.app import AppRegistry


def actor_builder_for(registry: "AppRegistry", interface: str) -> ActorBuilder:
    """The actor builder for the implementation at ``interface``.

    Raises:
        KeyError: If nothing is registered at ``interface``.
    """
    declared = registry.get_declared_implementation(interface)
    actifier = declared.actifier or reactify
    _, _, builder = actifier(declared.function, registry.structure_registry, declared.config)
    return builder
