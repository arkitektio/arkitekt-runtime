"""Being a state is a declaration on an app, not a property of the class.

Registering a class as a state writes nothing on it: the app's registry records
the interface, schema and locks, and an instance only becomes evented when an
agent adopts it. So one class can be a state of two apps under different rules.
"""

from dataclasses import dataclass

from arkitekt_spec.declare.app import AppRegistry
from arkitekt_runtime.state.observable import config_of, evented


@dataclass
class Counter:
    count: int = 0


def test_registering_a_state_leaves_the_class_alone() -> None:
    registry = AppRegistry()
    registry.state(Counter, required_locks=["stage"])

    assert not any(name.startswith("__rekuest") for name in vars(Counter))
    assert Counter.__init__ is Counter.__dict__["__init__"], "no wrapped __init__"
    assert config_of(Counter()) is None, "a plain instance is not evented"


def test_one_class_is_a_state_of_two_apps_under_each_apps_rules() -> None:
    a, b = AppRegistry(), AppRegistry()
    a.state(Counter, name="CounterA", required_locks=["a"])
    b.state(Counter, name="CounterB", required_locks=["b"])

    declared_a = a.snapshot().structure_registry.state_for(Counter)
    declared_b = b.snapshot().structure_registry.state_for(Counter)

    assert (declared_a.interface, declared_a.required_locks) == ("CounterA", ("a",))
    assert (declared_b.interface, declared_b.required_locks) == ("CounterB", ("b",))


def test_an_action_of_each_app_sees_its_own_locks() -> None:
    a, b = AppRegistry(), AppRegistry()
    a.state(Counter, name="CounterA", required_locks=["a"])
    b.state(Counter, name="CounterB", required_locks=["b"])

    def bump(counter: Counter) -> None:
        """Bump."""
        counter.count += 1

    a.register(bump)
    b.register(bump)

    assert a.implementations["bump"].locks == ("a",)
    assert b.implementations["bump"].locks == ("b",)


def test_adoption_makes_an_instance_evented_with_the_adopting_apps_rules() -> None:
    a, b = AppRegistry(), AppRegistry()
    a.state(Counter, name="CounterA", required_locks=["a"])
    b.state(Counter, name="CounterB", required_locks=["b"])

    one = evented(Counter(), a.structure_registry.state_for(Counter))
    two = evented(Counter(), b.structure_registry.state_for(Counter))

    assert isinstance(one, Counter) and isinstance(two, Counter)
    assert config_of(one).state_name == "CounterA" and config_of(one).required_locks == ["a"]
    assert config_of(two).state_name == "CounterB" and config_of(two).required_locks == ["b"]
    assert config_of(one) is not config_of(two)
