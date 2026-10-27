"""Who a task works for: ``task.assigner`` is the human at the root of its tree, on every task."""

from arkitekt_runtime import messages
from arkitekt_runtime.actors.helper import AssignmentHelper
from arkitekt_runtime.task import Task


def _task(**overrides: object) -> Task:
    assign = messages.Assign(
        interface="i", task="t", args={}, user="agent-user", org="lok#7", action="a", implementation="m", **overrides
    )
    return Task(AssignmentHelper.model_construct(assignment=assign, actor=None))


def test_a_delegated_task_names_the_human_who_assigned_its_tree() -> None:
    task = _task(parent="p", root="r", assigner="human")
    assert (task.user, task.assigner, task.org) == ("agent-user", "human", "lok#7")


def test_without_an_assigner_from_the_server_the_causer_is_the_assigner() -> None:
    assert _task().assigner == "agent-user"


def test_the_org_is_the_issuer_qualified_identifier_on_the_wire() -> None:
    wire = messages.Assign.model_validate(
        {"interface": "i", "task": "t", "args": {}, "user": "u", "assigner": "h", "org": "lok#7", "action": "a", "implementation": "m"}
    )
    assert (wire.org, wire.assigner) == ("lok#7", "h")
