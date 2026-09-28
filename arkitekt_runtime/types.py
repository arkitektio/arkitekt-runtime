"""The runtime's own slice of the task protocol: task event kinds, and the requests
that assign, cancel, pause and resume a task (hooks are the spec's).

Seeded from rekuest's generated protocol inputs and owned here from then on (rekuest's
generated modules import these names from this module): every runtime -- the socket
agent and the HTTP server alike -- speaks them.
"""

from enum import Enum
from typing import Annotated

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from arkitekt_spec.declare.task import HookInput
from arkitekt_spec.scalars import ID, ActionHash, Args


class GraphQLDefault:
    """Records a GraphQL field schema default value. The client omits the field so the server applies its own default; this preserves the value for introspection."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'GraphQLDefault(' + repr(self.value) + ')'


class TaskEventKind(str, Enum):
    """The event kind of the taskevent"""
    BOUND = 'BOUND'
    QUEUED = 'QUEUED'
    STARTED = 'STARTED'
    PROGRESS = 'PROGRESS'
    DELEGATE = 'DELEGATE'
    UNASSIGN = 'UNASSIGN'
    DISCONNECTED = 'DISCONNECTED'
    YIELD = 'YIELD'
    COMPLETED = 'COMPLETED'
    LOG = 'LOG'
    CANCELLING = 'CANCELLING'
    CANCELLED = 'CANCELLED'
    INTERRUPTING = 'INTERRUPTING'
    INTERRUPTED = 'INTERRUPTED'
    PAUSING = 'PAUSING'
    PAUSED = 'PAUSED'
    RESUMING = 'RESUMING'
    RESUMED = 'RESUMED'
    FAILED = 'FAILED'
    CRITICAL = 'CRITICAL'
    __str__ = str.__str__


class MappedAgentInput(BaseModel):
    """The input for mapping actions to implementations in a agent."""
    key: str = Field(description='The key of the agent to map. This is used to identify the agent in the system.')
    agent: ID = Field(description='The agent ID to map the actions to. This is used to identify the agent in the system.')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


class ResolvedDependencyInput(BaseModel):
    """The input for mapping dependencies to implementations in a agent."""
    key: str = Field(description='The key of the dependency to map. This is used to identify the dependency in the system.')
    mapped_agents: tuple[MappedAgentInput, ...] = Field(validation_alias=AliasChoices('mapped_agents', 'mappedAgents'), serialization_alias='mappedAgents', description='The list of mapped agents to map to implementations in agents. This is used to identify the mapped agents in the system.')
    auto_resolve: Annotated[bool | None, GraphQLDefault('False')] = Field(validation_alias=AliasChoices('auto_resolve', 'autoResolve'), serialization_alias='autoResolve', default=None, description='Whether this dependency should be automatically resolved by the system. If true, the system will attempt to find a agent that can resolve this dependency and assign it to the action when the action is assigned. This is used to enable automatic resolution of dependencies without requiring the user to specify a specific agent for the dependency.')
    'Whether this dependency should be automatically resolved by the system. If true, the system will attempt to find a agent that can resolve this dependency and assign it to the action when the action is assigned. This is used to enable automatic resolution of dependencies without requiring the user to specify a specific agent for the dependency.\nDefault: False'
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


class AssignInput(BaseModel):
    """The input for assigning args to a action. A GraphQL assign is a ROOT by definition — children are created only over the agent socket (AssignRequest, where parent is mandatory) and by server-internal paths like init hooks, so parent/dependency/method are deliberately absent here."""
    action: ID | None = Field(default=None, description='The action ID to assign to')
    resolution: ID | None = Field(default=None, description='The resolution ID to assign to when assining to a implementation with dependencies')
    implementation: ID | None = Field(default=None, description='The implementation ID to assign to when directly assingint to a implementation')
    agent: ID | None = Field(default=None, description='The agent ID to assign to when directly assingint to a implementation')
    action_hash: ActionHash | None = Field(validation_alias=AliasChoices('action_hash', 'actionHash'), serialization_alias='actionHash', default=None, description='The hash of the action. This is used to identify the action in the system.')
    interface: str | None = Field(default=None, description='The interface of the implementation. Only ussable if you also set agent')
    hooks: tuple['HookInput', ...] | None = Field(default=None, description='The hooks of the task. This is used to identify the task in the system.')
    args: Args = Field(description='The args of the task. Its a dictionary of ports and values')
    reference: str | None = Field(default=None, description='The reference of the task. This is used to identify the task in the system.')
    capture: bool = Field(description='Whether to capture the task.')
    dependencies: tuple['ResolvedDependencyInput', ...] | None = Field(default=None, description='The dependencies of the task. This maps dependency keys to implementation IDs.')
    step: bool | None = Field(default=None, description='Whether the task should step. Ie. go to the next breakpoint')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


class CancelInput(BaseModel):
    """The input for canceling a task."""
    task: ID = Field(description='The task ID to cancel')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


class PauseInput(BaseModel):
    """The input for pausing a task."""
    task: ID = Field(description='The task ID to pause')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


class ResumeInput(BaseModel):
    """The input for resuming a task."""
    task: ID = Field(description='The task ID to resume')
    step: Annotated[bool | None, GraphQLDefault('False')] = Field(default=None, description='Resume only until the next breakpoint instead of running on freely.')
    'Resume only until the next breakpoint instead of running on freely.\nDefault: False'
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)


MappedAgentInput.model_rebuild()
ResolvedDependencyInput.model_rebuild()
AssignInput.model_rebuild()
CancelInput.model_rebuild()
PauseInput.model_rebuild()
ResumeInput.model_rebuild()
