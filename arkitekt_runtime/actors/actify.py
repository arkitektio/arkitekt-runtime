"""Actifier

This module contains the actify function, which is used to convert a function
into an actor.
"""

import inspect
from functools import partial
from typing import Any

from arkitekt_runtime.actors.functional import (
    FUNC,
    GEN,
    THREADED_FUNC,
    THREADED_GEN,
    FunctionalActor,
)
from arkitekt_spec.declare.protocol.types import AnyFunction
from arkitekt_spec.declare.actors.types import (
    ImplementationDetails,
    RegisterConfig,
)
from arkitekt_runtime.actors.types import ActorBuilder
from arkitekt_spec.actions import DefinitionInput
from arkitekt_spec.declare.structures.registry import StructureRegistry
from arkitekt_spec.declare.actors.actify import (
    derive_implementation_details,
    prepare_definition_from_config,
)






def reactify(
    function: AnyFunction,
    structure_registry: StructureRegistry,
    config: RegisterConfig | None = None,
) -> tuple[DefinitionInput, ImplementationDetails, ActorBuilder]:
    """Reactify a function

    This function takes a callable (of type async or sync function or generator) and
    returns a builder function that creates an actor that makes the function callable
    from the rekuest server.

    All registration options are read from the bundled ``config``
    (:class:`~rekuest.actors.types.RegisterConfig`); ``config`` defaults to an
    empty config so ``reactify(func, registry)`` keeps working.
    """
    config = config or RegisterConfig()

    implementation_details = derive_implementation_details(function, config, structure_registry)
    definition = prepare_definition_from_config(
        function, structure_registry, config, implementation_details
    )

    is_coroutine = inspect.iscoroutinefunction(function)
    is_asyncgen = inspect.isasyncgenfunction(function)
    is_method = inspect.ismethod(function)

    is_generatorfunction = inspect.isgeneratorfunction(function)
    is_function = inspect.isfunction(function)

    actor_attributes: dict[str, Any] = {
        "assign": function,
        "expand_inputs": not config.bypass_expand,
        "shrink_outputs": not config.bypass_shrink,
        "structure_registry": structure_registry,
        "definition": definition,
        **implementation_details.actor_kwargs(),
        "concurrency": config.concurrency,
        "policy": config.policy,
    }

    if is_coroutine:
        iterator = FUNC
    elif is_asyncgen:
        iterator = GEN
    elif is_generatorfunction and not config.in_process:
        iterator = THREADED_GEN
    elif (is_function or is_method) and not config.in_process:
        iterator = THREADED_FUNC
    else:
        raise NotImplementedError("No way of converting this to a function")

    return (
        definition,
        implementation_details,
        partial(FunctionalActor, iterator=iterator, **actor_attributes),
    )
