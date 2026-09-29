"""Builders for Qt actors.

This allow the async patterns of actors to extend to the Qt world.

"""

import inspect
from functools import partial
from typing import (
    Any,
    get_args,
    get_origin,
)
from collections.abc import AsyncGenerator, Callable
from qtpy import QtCore, QtWidgets
from koil.qt import QtGenerator, QtFuture, qt_to_async, qt_gen_to_async_gen
from arkitekt_runtime.actors.functional import FUNC, GEN, FunctionalActor

from arkitekt_spec.declare.actors.actify import (
    derive_implementation_details,
    prepare_definition_from_config,
)
from arkitekt_spec.actions import DefinitionInput
from arkitekt_spec.declare.actors.types import (
    ImplementationDetails,
    RegisterConfig,
)
from arkitekt_runtime.actors.types import (
    ActorBuilder,
    Agent,
)
from arkitekt_spec.declare.protocol.types import AnyFunction
from arkitekt_spec.declare.structures.registry import StructureRegistry


class QtInLoopBuilder(QtCore.QObject):
    """A function that takes a provision and an actor transport and returns an actor.

    The actor produces by this builder will be running in the same thread as the
    koil instance (aka, the thread that called the builder).

    Args:
        QtCore (_type_): _description_
    """

    def __init__(
        self,
        assign: Callable = None,
        *args,  # noqa: ANN002
        parent: QtWidgets.QWidget | None = None,
        definition: DefinitionInput = None,
        **actor_kwargs: Any,  # noqa: ANN401 -- forwarded to the actor as is
    ) -> None:
        """Initialize the builder."""
        super().__init__(*args, parent=parent)
        self.wrapped_function = assign
        self.coro = qt_to_async(self.qt_assign)
        self.provisions = {}
        self.actor_kwargs = actor_kwargs
        self.definition = definition

    def qt_assign(self, future: QtFuture[Any], *args, **kwargs) -> None:
        """Assigns the future to the coroutine."""
        try:
            result = self.wrapped_function(*args, **kwargs)
        except Exception as e:
            # Unresolved, the awaiting assignment would hang forever.
            future.reject(e)
            return
        future.resolve(result)

    async def on_assign(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Runs in the same thread as the koil instance."""

        return await self.coro.acall(*args, **kwargs)

    def build(self, agent: Agent, structure_registry: StructureRegistry) -> "FunctionalActor":
        """Build the actor against ``structure_registry``.

        The registry is not held here: the actifier returns a ``partial`` over this
        method carrying it, so a run's snapshot can re-point it at its own copy.
        """
        try:
            ac = FunctionalActor(
                agent=agent,
                structure_registry=structure_registry,
                assign=self.on_assign,
                iterator=FUNC,
                definition=self.definition,
                **self.actor_kwargs,
            )
            return ac
        except Exception as e:
            raise e


class QtFutureBuilder(QtCore.QObject):
    """A function that takes a provision and an actor transport and returns an actor.

    The actor produces by this builder will be running in the same thread as the
    koil instance (aka, the thread that called the builder).

    Args:
        QtCore (_type_): _description_
    """

    def __init__(
        self,
        assign: Callable = None,
        *args,  # noqa: ANN002
        parent: QtWidgets.QWidget | None = None,
        definition: DefinitionInput = None,
        **actor_kwargs: Any,  # noqa: ANN401 -- forwarded to the actor as is
    ) -> None:
        """Initialize the builder."""
        super().__init__(*args, parent=parent)
        self.coro = qt_to_async(lambda *args, **kwargs: assign(*args, **kwargs))
        self.provisions = {}
        self.actor_kwargs = actor_kwargs
        self.definition = definition

    async def on_assign(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Runs in the same thread as the koil instance."""
        x = await self.coro.acall(*args, **kwargs)
        return x

    def build(self, agent: Agent, structure_registry: StructureRegistry) -> "FunctionalActor":
        """Build the actor against ``structure_registry``.

        The registry is not held here: the actifier returns a ``partial`` over this
        method carrying it, so a run's snapshot can re-point it at its own copy.
        """
        try:
            ac = FunctionalActor(
                agent=agent,
                structure_registry=structure_registry,
                assign=self.on_assign,
                iterator=FUNC,
                definition=self.definition,
                **self.actor_kwargs,
            )
            return ac
        except Exception as e:
            raise e


class QtGeneratorBuilder(QtCore.QObject):
    """A function that takes a provision and an actor transport and returns an actor.

    The actor produces by this builder will be running in the same thread as the
    koil instance (aka, the thread that called the builder).

    Args:
        QtCore (_type_): _description_
    """

    def __init__(
        self,
        assign: Callable = None,
        *args,  # noqa: ANN002
        parent: QtWidgets.QWidget | None = None,
        structure_registry: StructureRegistry | None = None,
        definition: DefinitionInput | None = None,
        **actor_kwargs: Any,  # noqa: ANN401 -- forwarded to the actor as is
    ) -> None:
        """Initialize the builder."""
        super().__init__(*args, parent=parent)
        self.generator = qt_gen_to_async_gen(
            lambda *args, **kwargs: assign(*args, **kwargs)
        )
        self.provisions = {}
        self.actor_kwargs = actor_kwargs
        self.definition = definition

    async def on_assign(self, *args, **kwargs) -> AsyncGenerator[Any, None]:  # noqa: ANN002, ANN003
        """Runs in the same thread as the koil instance."""
        async for i in self.generator.acall(*args, **kwargs):
            yield i

    def build(self, agent: Agent, structure_registry: StructureRegistry) -> "FunctionalActor":
        """Build the actor against ``structure_registry``.

        The registry is not held here: the actifier returns a ``partial`` over this
        method carrying it, so a run's snapshot can re-point it at its own copy.
        """
        try:
            ac = FunctionalActor(
                agent=agent,
                structure_registry=structure_registry,
                assign=self.on_assign,
                iterator=GEN,
                definition=self.definition,
                **self.actor_kwargs,
            )
            return ac
        except Exception as e:
            raise e


def qtinloopactifier(
    function: AnyFunction,
    structure_registry: StructureRegistry,
    config: RegisterConfig | None = None,
    *,
    parent: QtWidgets.QWidget | None = None,
) -> tuple[DefinitionInput, ImplementationDetails, ActorBuilder]:
    """Qt Actifier

    The qt actifier wraps a function and returns a builder that will create an actor
    that runs in the same thread as the Qt instance, enabling the use of Qt widgets
    and signals.
    """
    config = config or RegisterConfig()

    implementation_details = derive_implementation_details(function, config, structure_registry)
    definition = prepare_definition_from_config(
        function, structure_registry, config, implementation_details
    )

    actor_attributes: dict[str, Any] = {
        "expand_inputs": not config.bypass_expand,
        "shrink_outputs": not config.bypass_shrink,
        **implementation_details.actor_kwargs(),
    }

    in_loop_instance = QtInLoopBuilder(
        parent=parent,
        assign=function,
        definition=definition,
        **actor_attributes,
    )

    return (
        definition,
        implementation_details,
        partial(in_loop_instance.build, structure_registry=structure_registry),
    )


def qtwithfutureactifier(
    function: Callable,
    structure_registry: StructureRegistry,
    config: RegisterConfig | None = None,
    *,
    parent: QtWidgets.QWidget | None = None,
) -> tuple[DefinitionInput, ImplementationDetails, ActorBuilder]:
    """Qt Actifier

    The qt actifier wraps a function and returns a build that calls the function with
    its first parameter being a future that can be resolved within the qt loop
    """
    config = config or RegisterConfig()

    sig = inspect.signature(function)

    if len(sig.parameters) == 0:
        raise ValueError(
            f"The function {function} you are trying to register with a future actifier must have at least one parameter, the QtFuture"
        )

    first = sig.parameters[list(sig.parameters.keys())[0]].annotation

    if not get_origin(first) == QtFuture:
        raise ValueError(
            f"The function {function} you are trying to register needs to have a QtFuture as its first parameter"
        )

    return_params = get_args(first)

    if len(return_params) == 0:
        raise ValueError(
            "If you are using a QtFuture as the first parameter, you need to provide the return type of the future as a type hint. E.g `QtFuture[int]`"
        )

    implementation_details = derive_implementation_details(function, config, structure_registry)
    definition = prepare_definition_from_config(
        function,
        structure_registry,
        config,
        implementation_details,
        omitfirst=1,
        return_annotations=return_params,
    )

    in_loop_instance = QtFutureBuilder(
        parent=parent,
        assign=function,
        definition=definition,
    )

    return (
        definition,
        implementation_details,
        partial(in_loop_instance.build, structure_registry=structure_registry),
    )


def qtwithgeneratoractifier(
    function: Callable,
    structure_registry: StructureRegistry,
    config: RegisterConfig | None = None,
    *,
    parent: QtWidgets.QWidget | None = None,
) -> tuple[DefinitionInput, ImplementationDetails, ActorBuilder]:
    """Qt Actifier

    The qt actifier wraps a function and returns a build that calls the function with
    its first parameter being a generator that can be yielded to within the qt loop
    """
    config = config or RegisterConfig()

    sig = inspect.signature(function)

    if len(sig.parameters) == 0:
        raise ValueError(
            "The function you are trying to register with a generator actifier must have at least one parameter, the Generator"
        )

    first = sig.parameters[list(sig.parameters.keys())[0]].annotation

    if not get_origin(first) == QtGenerator:
        raise ValueError(
            "The function needs to have a QtGenerator as its first parameter"
        )

    return_params = get_args(first)

    if len(return_params) == 0:
        raise ValueError(
            "If you are using a QtGenerator as the first parameter, you need to provide the return type of the generator as a type hint. E.g `QtGenerator[int]`"
        )

    implementation_details = derive_implementation_details(function, config, structure_registry)
    definition = prepare_definition_from_config(
        function,
        structure_registry,
        config,
        implementation_details,
        omitfirst=1,
        return_annotations=return_params,
    )

    in_loop_instance = QtGeneratorBuilder(
        parent=parent,
        assign=function,
        definition=definition,
    )

    return (
        definition,
        implementation_details,
        partial(in_loop_instance.build, structure_registry=structure_registry),
    )
