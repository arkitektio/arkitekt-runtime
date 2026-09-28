"""This file contains the structures used in the tests."""

from typing import TYPE_CHECKING

from pydantic.main import BaseModel

if TYPE_CHECKING:
    from arkitekt_spec.declare.structures.registry import StructureRegistry


class SerializableObject(BaseModel):
    """SerializableObject is a simple object should be serializeable and deserailblie globalyy."""

    number: int

    @classmethod
    def get_identifier(cls) -> str:
        """Get the identifier of the object."""
        return "@mock/serializable"

    async def ashrink(self) -> str:
        """Shrink the object to its id."""
        return self.number

    @classmethod
    async def aexpand(cls, value: str) -> "SerializableObject":
        """Expand the object from its id."""
        return cls(number=value)


class GlobalObject(BaseModel):
    """GlobalObject is a simple object that can't be serialized and deserialized. It should get put on a shelve"""

    number: int


class SecondSerializableObject:
    """Another SerializableObject that has a id"""

    def __init__(self, id: str) -> None:
        """Initialize the object with an id."""
        self.id = id

    @classmethod
    def get_identifier(cls) -> str:
        """Get the identifier of the object."""
        return "@mock/secondserializable"

    async def ashrink(self) -> str:
        """Shrink the object to its id."""
        return self.id

    @classmethod
    async def aexpand(cls, value: str) -> "SecondSerializableObject":
        """Expand the object from its id."""
        return cls(id=value)


class IdentifiableSerializableObject(BaseModel):
    """IdentifiableSerializableObject is a simple object that can be serialized and deserialized.
    it should get a global scope"""

    number: int

    @classmethod
    def get_identifier(cls) -> str:
        """Get the identifier of the object."""
        return "@mock/identifiable"

    async def ashrink(self) -> str:
        """Shrink the object to its id."""
        return self.number

    @classmethod
    async def aexpand(cls, shrinked_value: str) -> "IdentifiableSerializableObject":
        """Expand the object from its id."""
        return cls(number=shrinked_value)


class SecondObject:
    """Another SerializableObject that has a id"""

    pass

    def __init__(self, id: str) -> None:  # noqa: D107
        """Initialize the object with an id.

        Args:
            id (str): The id of the object.
        """
        self.id = id

    @classmethod
    def get_identifier(cls) -> str:
        """Get the identifier of the object."""
        return "@mock/secondobject"

    async def ashrink(self) -> str:
        """Shrink the object to its id."""
        return self.id

    @classmethod
    async def aexpand(cls, value: str) -> "SecondObject":
        """Expand the object from its id."""
        return cls(id=value)


def test_registry(**kwargs: object) -> "StructureRegistry":
    """A structure registry holding the structures in this module.

    Nothing registers itself any more, so a test that builds a bare
    ``StructureRegistry()`` and then names one of these in a signature gets a
    refusal. This is the preregistration those tests need, in one place.
    """
    from arkitekt_spec.declare.structures.registry import StructureRegistry

    registry = StructureRegistry(**kwargs)
    register_test_structures(registry)
    return registry


def register_test_structures(registry: "StructureRegistry") -> None:
    """Register every structure in this module into ``registry``."""
    for cls in (
        SerializableObject,
        SecondSerializableObject,
        IdentifiableSerializableObject,
        SecondObject,
    ):
        registry.register_from_protocol(cls)
    # The one that cannot fetch itself, and so lives on the agent's shelve.
    registry.register_as_memory_structure(GlobalObject)
