import pytest
from arkitekt_spec.declare.structures.registry import StructureRegistry


class MockShelver:
    """A mock shelver that stores values in memory. This is used to test the
    shelver functionality without using a real shelver."""

    def __init__(self) -> None:
        """Initialize the mock shelver."""
        self.shelve: dict[str, object] = {}

    async def aput_on_shelve(self, identifier: str, value: object) -> str:
        """Put a value on the shelve and return the key. This is used to store
        values on the shelve."""
        key = str(id(value))
        self.shelve[key] = value
        return key

    async def aget_from_shelve(self, key: str) -> object:
        """Get a value from the shelve. This is used to get values from the
        shelve."""
        return self.shelve[key]


@pytest.fixture()
def simple_registry() -> StructureRegistry:
    """A registry preloaded with the structures the test functions use.

    Nothing registers itself any more, so the structures a signature names have
    to be in here before a definition can be built against it.
    """
    from .funcs import Karl, LocalizedStructure
    from .structures import test_registry

    registry = test_registry()
    # Lives in funcs.py beside the function that returns it, and is the one thing
    # here that is kept on the shelve rather than fetched by id.
    registry.register_as_memory_structure(LocalizedStructure)
    # The one model the test functions name; a model is declared on an app.
    registry.model(Karl)
    return registry


@pytest.fixture()
def mock_shelver() -> MockShelver:
    """Fixture for a mock shelver"""
    return MockShelver()
