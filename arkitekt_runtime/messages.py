"""Messages that are used to communicate between the rekuest backend and the agent.

Naming convention — the role lives in the affix and the direction is unambiguous:

- **Backend → Agent commands**: bare imperative (``Assign``, ``Cancel``, ``Pause``).
- **Agent → Backend reports**: bare past/noun (``Completed``, ``Failed``, ``Progress``).
- **Caller → Backend requests**: ``…Request`` suffix (``AssignRequest``, ``CancelRequest``).
- **Backend → Caller acks**: ``…Response`` suffix (``AssignResponse``, ``ControlResponse``).
- **Backend → Caller event stream**: ``…Event`` suffix (``CompletedEvent``, ``ProgressEvent``).

No names collide because commands are imperative, agent reports are bare past-tense, and
the caller stream carries the ``…Event`` suffix (``Pause`` cmd vs ``Paused`` report vs
``PausedEvent`` stream).
"""

from typing import Any, ClassVar, Literal, Union, get_args
from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer
from enum import Enum
from pydantic import Field
import uuid
from arkitekt_spec.declare.task import AssignPrompt, StateRequirement, TaskBlok


JSONSerializable = (
    str | int | float | bool | None | dict[str, "JSONSerializable"] | list["JSONSerializable"]
)


ShallowJSONSerializable = str | int | float | bool | None | dict[str, Any] | list[Any]

LogLevelLiteral = Literal[
    "DEBUG",
    "INFO",
    "ERROR",
    "WARN",
    "CRITICAL",
]


class ToAgentMessageType(str, Enum):
    """The message types that can be sent to the agent from the rekuest backend"""

    ASSIGN = "ASSIGN"
    CANCEL = "CANCEL"
    COLLECT = "COLLECT"
    RESUME = "RESUME"
    PAUSE = "PAUSE"
    INTERRUPT = "INTERRUPT"
    # The answer to a task's own hold (HELD).
    DECIDE = "DECIDE"
    # The blok a HELD or an INFORM carried could not be made: nothing of it was kept.
    BLOK_REFUSED = "BLOK_REFUSED"
    INIT = "INIT"
    HEARTBEAT = "HEARTBEAT"
    BOUNCE = "BOUNCE"
    KICK = "KICK"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    # Cumulative: everything of a journal session through a position is persisted.
    JOURNAL_ACK = "JOURNAL_ACK"
    # A frame arrived ahead of the position expected: send again from there.
    JOURNAL_NACK = "JOURNAL_NACK"
    # What of this task? (to the agent it was given to), and the answer to a caller who asked.
    INQUIRE = "INQUIRE"
    INQUIRE_RESPONSE = "INQUIRE_RESPONSE"
    ASSIGN_RESPONSE = "ASSIGN_RESPONSE"
    PROBE_RESPONSE = "PROBE_RESPONSE"
    STATE_REVISION_RESPONSE = "STATE_REVISION_RESPONSE"
    # Replies to the agent's shelving requests.
    # Caller-bound event-stream mirrors — one per TaskEventKind — streamed back to the
    # participant that originated the task (see ``ExecutionEvent`` and subclasses).
    BOUND_EVENT = "BOUND_EVENT"
    QUEUED_EVENT = "QUEUED_EVENT"
    STARTED_EVENT = "STARTED_EVENT"
    PROGRESS_EVENT = "PROGRESS_EVENT"
    DELEGATE_EVENT = "DELEGATE_EVENT"
    YIELD_EVENT = "YIELD_EVENT"
    COMPLETED_EVENT = "COMPLETED_EVENT"
    LOG_EVENT = "LOG_EVENT"
    CANCELLING_EVENT = "CANCELLING_EVENT"
    CANCELLED_EVENT = "CANCELLED_EVENT"
    INTERRUPTING_EVENT = "INTERRUPTING_EVENT"
    INTERRUPTED_EVENT = "INTERRUPTED_EVENT"
    PAUSING_EVENT = "PAUSING_EVENT"
    PAUSED_EVENT = "PAUSED_EVENT"
    HELD_EVENT = "HELD_EVENT"
    DENIED_EVENT = "DENIED_EVENT"
    RESUMING_EVENT = "RESUMING_EVENT"
    RESUMED_EVENT = "RESUMED_EVENT"
    FAILED_EVENT = "FAILED_EVENT"
    CRITICAL_EVENT = "CRITICAL_EVENT"
    LOST_EVENT = "LOST_EVENT"
    # Ack for a caller's lifecycle-control request (cancel/interrupt/pause/resume).
    CONTROL_RESPONSE = "CONTROL_RESPONSE"


class FromAgentMessageType(str, Enum):
    """The message types that can be sent from the agent to the rekuest backend"""

    REGISTER = "REGISTER"
    LOG = "LOG"
    PROGRESS = "PROGRESS"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    YIELD = "YIELD"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    # The task waits for a person's decision (task.hold); answered by DECIDE.
    HELD = "HELD"
    # The task shows a person a blok while it goes on (task.inform).
    INFORM = "INFORM"
    CRITICAL = "CRITICAL"
    RESUMED = "RESUMED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    HEARTBEAT_ANSWER = "HEARTBEAT_ANSWER"
    # Cumulative: every command through a position is handled.
    COMMAND_ACK = "COMMAND_ACK"
    # What this agent knows of a task it was asked about, and a caller asking about one.
    INQUIRY_ANSWER = "INQUIRY_ANSWER"
    INQUIRE_REQUEST = "INQUIRE_REQUEST"
    # Reports the agent no longer holds: the only way a position is passed over.
    JOURNAL_SKIP = "JOURNAL_SKIP"
    STATE_PATCH = "STATE_PATCH"
    LOCK = "LOCK"
    UNLOCK = "UNLOCK"
    STATE_UPDATE = "STATE_UPDATE"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    SESSION_INIT = "SESSION_INIT"
    ASSIGN_REQUEST = "ASSIGN_REQUEST"
    PROBE_REQUEST = "PROBE_REQUEST"
    STATE_REVISION_REQUEST = "STATE_REVISION_REQUEST"
    # Shelving: what the agent holds in memory.
    SHELVE = "SHELVE"
    UNSHELVE = "UNSHELVE"
    # A value a task took from outside itself (the clock, randomness, a deadline).
    EFFECT = "EFFECT"
    # Caller-issued lifecycle control requests over the socket (mirroring ASSIGN_REQUEST).
    CANCEL_REQUEST = "CANCEL_REQUEST"
    INTERRUPT_REQUEST = "INTERRUPT_REQUEST"
    PAUSE_REQUEST = "PAUSE_REQUEST"
    RESUME_REQUEST = "RESUME_REQUEST"


class Message(BaseModel):
    """A base message class"""

    # This is the local mapping of the message, reply messages should have the same id
    model_config = ConfigDict(use_enum_values=True, frozen=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


#: The journal fields a numbered frame carries next to its ``id``.
JOURNAL_STAMP_FIELDS = ("pos", "journal_session", "agent_ts", "task_step")


class JournaledMessage(Message):
    """Base for the agent reports the agent's journal numbers (see ``docs/journal.md``).

    Task events (effects included), lock changes, state patches, snapshots, the session
    baseline and shelving. A numbered frame sent to a server carries its position in
    the session (``pos``) and, when it belongs to a task, its step in that task
    (``task_step``). The fields
    are left out of the frame entirely while unset, so a frame that is not stamped
    (every frame a FastAPI websocket client that did not opt in receives) is exactly
    what it was before the journal existed. ``Register`` never has them: the server
    forbids extras there.
    """

    pos: int | None = Field(
        default=None,
        description="The frame's position in its journal session (1, 2, 3, ... with no gaps). Unset for frames that are not journaled.",
    )
    journal_session: str | None = Field(
        default=None,
        description="The journal session ``pos`` counts in (the agent's session).",
    )
    agent_ts: float | None = Field(
        default=None,
        description="When the agent recorded the frame, in seconds since the epoch.",
    )
    task_step: int | None = Field(
        default=None,
        description="The frame's step in its task (1, 2, 3, ... per task, shared with the task's child calls). Only on frames of a task. Not ``step``: ``ASSIGN`` has a ``step`` flag.",
    )

    left_out_unset: ClassVar[tuple[str, ...]] = JOURNAL_STAMP_FIELDS
    """The fields a frame leaves out while they are unset (the stamp, and whatever a
    message added later that an older reader never saw)."""

    @model_serializer(mode="wrap")
    def _drop_unset_journal_stamp(self, handler: SerializerFunctionWrapHandler) -> Any:  # noqa: ANN401
        data = handler(self)
        if isinstance(data, dict):
            for key in type(self).left_out_unset:
                if key in data and data[key] is None:
                    del data[key]
        return data


class FromAgentEvent(JournaledMessage):
    """Base for a task's reports: what a task says about itself, each at its position."""


class Command(Message):
    """Base for what the backend sends an agent and must not lose: work for it and what
    steers it, the answers to its numbered requests, and what happened to the tasks it
    assigned.

    A command carries ``pos``, its position in the backend's stream to this agent (1, 2,
    3, ...). The backend keeps a command until a ``CommandAck`` covers it and sends it
    again after a reconnect; the agent handles a position once. A probe's frames are
    commands by kind and travel without a position: a probe is ephemeral.
    """

    pos: int | None = Field(
        default=None,
        description="The command's position in the backend's stream to this agent. Unset on a probe's frame, and where no backend numbers (a served agent, a local run).",
    )

    @model_serializer(mode="wrap")
    def _drop_unset_position(self, handler: SerializerFunctionWrapHandler) -> Any:  # noqa: ANN401
        data = handler(self)
        if isinstance(data, dict) and data.get("pos") is None:
            data.pop("pos", None)
        return data


FEATURE_PROTOCOL_2 = "protocol_2"
"""What an agent speaking this protocol says on ``Register`` (``features``): positions in both
directions, acknowledged commands, refused gaps. A server refuses an agent that does not."""

FEATURE_STATE_REQUIRES = "state_requires"
"""What an agent says on ``Register`` (``features``) when it tests an Assign's
``state_requires``. A server must not send requirements to an agent that does not: it
would drop them and run the task unchecked."""


class Assign(Command):
    """An assign call

    And assign call is the initial request to start a specific
    functionality and will have a task id, that will stand
    as a reference for all sub calls (Pause, Interrupt, Reumse, Collect...).
    as well should be passed to all events within the task (
        Progress, Logs, Done, Error, etc)
    )
    """

    type: Literal[ToAgentMessageType.ASSIGN] = ToAgentMessageType.ASSIGN
    interface: str = Field(
        description="The registered interface, that the agent should use to run this task"
    )
    step: bool | None = Field(
        default=None,
        description="Whether to step the task or not (i.e. stop at the first breakpoint and wait for a step message from the rekuest backend to continue). If None don't step.",
    )
    task: str = Field(description="The task id")
    root: str | None = Field(
        default=None,
        description="The root of all cascaded tasks (user triggered task), None if this is the mother",
    )
    """ The mother task (root)"""
    parent: str | None = Field(
        default=None,
        description="The direct parent of this task, None if this is this is the mother",
    )
    """ The parent s"""
    resolution: str | None = Field(
        default=None, description="The resolution id this task has dependencies"
    )
    probe: bool = Field(
        default=False,
        description="Whether this is a probe (p-… id): an ephemeral invocation with no server-side history, no replay/recovery; sub-assignment and locks are unavailable. Agents may adapt (e.g. skip audit side effects).",
    )
    capture: bool | None = Field(
        default=None, description="Whether to run in debug mode, false by default"
    )
    reference: str | None = Field(
        default=None, description="A reference that the assinger provided"
    )
    args: dict[str, ShallowJSONSerializable] = Field(
        description="The arguments that was sendend"
    )
    message: str | None = None
    user: str = Field(..., description="Who caused this hop: the human for a top-level task, the delegating agent's user for a child")
    assigner: str | None = Field(
        default=None,
        description="The human at the root of the task tree, who assigned it (the provenance `rcb`): the same for every task it delegated to. None from a server that does not send it.",
    )
    org: str = Field(description="The organization the task runs in, `<iss>#<org>`: unique across issuers")
    action: str = Field(description="The action of this task")
    implementation: str = Field(
        description="The implementation of this task (the id of the implementation)"
    )
    token: str | None = Field(
        default=None,
        description="An opaque, signed provenance token attesting who caused this task and with which inputs. The agent forwards it untouched to downstream services; it does not validate it. None when the implementation opts out of provenance (needs_token=False).",
    )
    traceparent: str | None = Field(
        default=None,
        description="The trace the task is part of, as a W3C `traceparent`: what the agent does for the task, and what it asks of others on its behalf, belongs to that trace. None where nobody traces.",
    )
    resume: "Journal | None" = Field(
        default=None,
        description="Set when a workflow's agent died and the task is sent again to be resumed: what it recorded so far.",
    )
    state_requires: list[StateRequirement] = Field(
        default_factory=list,
        description="What this agent's state must hold for the task to run. Tested after the task's locks are taken; on a mismatch the task ends FAILED with code STATE_PRECONDITION and nothing of it runs.",
    )


class Bounce(Message):
    """A bounce call

    Tells the agent to disconnect and reconnect (a soft restart of the connection).
    Exposed as the ``bounce`` GraphQL mutation. ``duration`` optionally hints how long
    to wait before reconnecting.
    """

    type: Literal[ToAgentMessageType.BOUNCE] = ToAgentMessageType.BOUNCE
    duration: int | None = None


class Kick(Message):
    """A kick call

    Tells the agent to force-disconnect. Unlike ``Bounce`` it will fail and NOT reconnect.
    Exposed as the ``kick`` GraphQL mutation. ``reason`` is an optional human-readable cause.
    """

    type: Literal[ToAgentMessageType.KICK] = ToAgentMessageType.KICK
    reason: str | None = None


class Heartbeat(Message):
    """A heartbeat call
    A heartbeat call tells the agent to send a heartbeat
    and all its children task until a resume is received
    Its on the actor to decide what to do with the children tasks
    """

    type: Literal[ToAgentMessageType.HEARTBEAT] = ToAgentMessageType.HEARTBEAT


class Pause(Command):
    """A pause call

    A pause call tells the agent to pause the task
    and all its children task until a resume is received

    Its on the actor to decide what to do with the children tasks
    (i.e. pause them, cancel them, etc) or to raise an error if the
    state of the assignaiton wouldn't allow this.

    """

    type: Literal[ToAgentMessageType.PAUSE] = ToAgentMessageType.PAUSE
    task: str


class Resume(Command):
    """A resume call

    A resume call unpauses the pause. With ``step=True`` the agent resumes only until the
    next breakpoint (the equivalent of the old standalone step instruction); with
    ``step=False`` it runs on freely."""

    type: Literal[ToAgentMessageType.RESUME] = ToAgentMessageType.RESUME
    task: str
    step: bool = False


class Decide(Command):
    """The answer to a task's own hold (``Held``).

    A person approved or denied it, or the run is in auto-mode (``auto``) and the server
    approved it without asking. A denial does not end the task.
    """

    type: Literal[ToAgentMessageType.DECIDE] = ToAgentMessageType.DECIDE
    task: str
    approved: bool
    answer: str | None = None
    auto: bool = False
    values: dict[str, Any] | None = None
    """The answer through the hold's blok (``task.ask``), checked against its returns."""


class BlokRefused(Command):
    """The blok a ``Held`` or an ``Inform`` carried could not be made, and why: nothing of
    the frame was kept (the task is not held, nothing is shown)."""

    type: Literal[ToAgentMessageType.BLOK_REFUSED] = ToAgentMessageType.BLOK_REFUSED
    task: str
    reason: str


class Cancel(Command):
    """A cancel call

    A cancellation call is a request from the user to
    cancel a task nicely (i.e by also nicely
    cancelling all the children tasks).
    Cancel represent a "nice alternative" to the interrupt call.
    While a cancellation of a mother task is only send to
    the mother to kill the children nicely (what the fuck is
    this metaphor), a interrupt will be send to all children
    automatically without the mother.


    Find more information on this in the arkitekt.live
    """

    type: Literal[ToAgentMessageType.CANCEL] = ToAgentMessageType.CANCEL
    task: str


class Collect(Command):
    """A collect call

    A collect call tells the agent to collect data LOCALLY,
    by deleting data on the "shelves" that live in memory.

    Find more information on this in the arkitekt.live
    documentation on local workflwos


    """

    type: Literal[ToAgentMessageType.COLLECT] = ToAgentMessageType.COLLECT
    drawers: list[str]


class Interrupt(Command):
    """A interrupt"""

    type: Literal[ToAgentMessageType.INTERRUPT] = ToAgentMessageType.INTERRUPT
    task: str


class Cancelled(FromAgentEvent):
    """A cancelled report

    Sent when the task was successfully cancelled by the actor.
    """

    type: Literal[FromAgentMessageType.CANCELLED] = FromAgentMessageType.CANCELLED
    task: str


class Interrupted(FromAgentEvent):
    """An interrupted report

    Sent when the task was successfully interrupted by the actor.
    """

    type: Literal[FromAgentMessageType.INTERRUPTED] = FromAgentMessageType.INTERRUPTED
    task: str


class Paused(FromAgentEvent):
    """A paused report

    Sent when the task was successfully paused by the actor.
    """

    type: Literal[FromAgentMessageType.PAUSED] = FromAgentMessageType.PAUSED
    task: str
    message: str | None = None
    """Why it paused, when the task paused itself (``task.hold``)."""
    details: dict[str, Any] | None = None
    """What a person deciding may want to know (``task.hold(lost=...)``: the lost step)."""


class Held(FromAgentEvent):
    """The task waits for a person's decision (``task.hold``); the server answers with
    ``Decide``. Unlike ``Paused``, the answer comes back to the task."""

    type: Literal[FromAgentMessageType.HELD] = FromAgentMessageType.HELD
    task: str
    message: str | None = None
    prompt: AssignPrompt | None = None
    details: dict[str, Any] | None = None
    """What a person deciding may want to know (``task.hold(lost=...)``: the lost step)."""
    blok: TaskBlok | None = None
    """Ask through a blok (``task.ask``): the person answers in it, and ``Decide`` carries its values."""


class Inform(FromAgentEvent):
    """The task shows a person a blok while it goes on (``task.inform``)."""

    type: Literal[FromAgentMessageType.INFORM] = FromAgentMessageType.INFORM
    task: str
    message: str | None = None
    blok: TaskBlok


class Resumed(FromAgentEvent):
    """A resumed report

    Sent when the task was successfully resumed by the actor.
    """

    type: Literal[FromAgentMessageType.RESUMED] = FromAgentMessageType.RESUMED
    task: str


class Started(FromAgentEvent):
    """A started report

    Sent when the actor has accepted a task and begun executing it.
    Mirrored to the caller as ``StartedEvent``.
    """

    type: Literal[FromAgentMessageType.STARTED] = FromAgentMessageType.STARTED
    task: str


class Log(FromAgentEvent):
    """A log report

    Sent when the agent wants to send a log message to the rekuest backend.
    """

    type: Literal[FromAgentMessageType.LOG] = FromAgentMessageType.LOG
    task: str
    message: str
    level: LogLevelLiteral = "INFO"
    """The log level of the message"""


class Progress(FromAgentEvent):
    """A progress report

    Sent when the agent wants to report progress on a task.
    """

    type: Literal[FromAgentMessageType.PROGRESS] = FromAgentMessageType.PROGRESS
    task: str
    progress: int | None = None
    message: str | None = None


class Yield(FromAgentEvent):
    """A yield report

    Sent when the agent wants to yield an intermediate result to the rekuest backend.
    """

    type: Literal[FromAgentMessageType.YIELD] = FromAgentMessageType.YIELD
    task: str
    returns: dict[str, Any] | None = None


class Completed(FromAgentEvent):
    """A completed report

    Sent when the actor has finished the task and all its children tasks.
    """

    type: Literal[FromAgentMessageType.COMPLETED] = FromAgentMessageType.COMPLETED
    task: str


class Failed(FromAgentEvent):
    """A failed report

    Sent when the agent reports a (potentially recoverable) error from execution.
    A ``Critical`` report signals an unrecoverable one instead.
    """

    type: Literal[FromAgentMessageType.FAILED] = FromAgentMessageType.FAILED
    task: str
    error: str
    code: str | None = Field(
        default=None,
        description="Why, when it is one the protocol names: STATE_PRECONDITION when the task's state_requires did not hold.",
    )


class Critical(FromAgentEvent):
    """A critical report

    Sent when the agent reports an unrecoverable error from execution.
    """

    type: Literal[FromAgentMessageType.CRITICAL] = FromAgentMessageType.CRITICAL
    task: str
    error: str


class HeartbeatEvent(Message):
    """A heartbeat event

    A heartbeat event is sent when the agent replies to a heartbeat
    message from the rekuest backend. Agents should never send
    heartbeat events, but only reply to them.
    """

    type: Literal[FromAgentMessageType.HEARTBEAT_ANSWER] = (
        FromAgentMessageType.HEARTBEAT_ANSWER
    )


class SessionInit(JournaledMessage):
    """A session init message

    A session init message is sent when the agent starts and wants to
    initialize the session with the rekuest backend. This is used to
    send session initialization information from the agent to the rekuest backend
    """

    type: Literal[FromAgentMessageType.SESSION_INIT] = FromAgentMessageType.SESSION_INIT
    session_id: str = Field(
        description="The session id of the agent (generated on a restart of the agent)"
    )
    states: dict[str, Any] = Field(
        description="A dictionary containing the initial state snapshots, where the key is the state name and the value is the state snapshot"
    )


class StatePatch(JournaledMessage):
    """One operation on a state, as agents said it before :class:`StateUpdate`.

    This runtime reports a :class:`StateUpdate`; a backend still reads this frame from
    agents that do not, as an update of one operation.
    """

    type: Literal[FromAgentMessageType.STATE_PATCH] = FromAgentMessageType.STATE_PATCH
    # A patch that says nothing of itself is the frame it always was.
    left_out_unset: ClassVar[tuple[str, ...]] = (*JOURNAL_STAMP_FIELDS, "lifecycle", "qualifiers")
    session_id: str = Field(
        description="The session id of the agent (generated on a restart of the agent)"
    )
    global_rev: int = Field(description="The global revision of the state")
    state_name: str = Field(description="The name of the state that is being patched")
    ts: float
    op: str
    path: str
    value: Any
    task_id: str | None = Field(
        default=None,
        description="An optional correlation id to correlate this patch with a specific task_id or event in the agent's execution, which can be used for debugging and tracing purposes",
    )
    lifecycle: str | None = Field(
        default=None,
        description="Where in a task's or the agent's life the change was made, when not in its ordinary course: 'cleanup' (a task being stopped, on its way out) or 'shutdown' (the agent's shutdown hooks). Said by the runtime.",
    )
    qualifiers: list[str] | None = Field(
        default=None,
        description="What the app says the change was for (task.qualified): its own names.",
    )


class StateOp(BaseModel):
    """One RFC 6902 operation of a state update."""

    op: str
    path: str
    value: Any = None


class StateUpdate(JournaledMessage):
    """One update of a state: what takes it from one revision to the next.

    An update is the unit a state changes by. However many operations describe it
    (``moves.update({...})`` is one per key, ``task.updating(state)`` whatever the block
    did), they are applied in order and seen together or not at all: nobody who follows
    the state is shown it between two of them. ``global_rev`` is the revision the state
    is at once the update is applied.

    The session it belongs to and when it was made are the frame's own
    (``journal_session``, ``agent_ts``): it says neither a second time.
    """

    type: Literal[FromAgentMessageType.STATE_UPDATE] = FromAgentMessageType.STATE_UPDATE
    left_out_unset: ClassVar[tuple[str, ...]] = (*JOURNAL_STAMP_FIELDS, "task", "lifecycle", "qualifiers")
    state: str = Field(description="The state that changed, by the name the agent declares it under.")
    global_rev: int = Field(description="The agent's state revision once this update is applied.")
    ops: list[StateOp] = Field(description="What the update does, as RFC 6902 operations, applied in order.")
    task: str | None = Field(
        default=None,
        description="The task that made the update, if one did.",
    )
    lifecycle: str | None = Field(
        default=None,
        description="Where in a task's or the agent's life the update was made, when not in its ordinary course: 'cleanup' (a task being stopped, on its way out) or 'shutdown' (the agent's shutdown hooks). Said by the runtime.",
    )
    qualifiers: list[str] | None = Field(
        default=None,
        description="What the app says the update was for (task.qualified): its own names.",
    )


class StateSnapshot(JournaledMessage):
    """A state snapshot message

    A state snapshot is sent when the agent wants to send a full state snapshot
    to the rekuest backend.
    """

    type: Literal[FromAgentMessageType.STATE_SNAPSHOT] = (
        FromAgentMessageType.STATE_SNAPSHOT
    )
    session_id: str = Field(
        description="The session id of the agent (generated on a restart of the agent)"
    )
    global_rev: int = Field(description="The global revision of the state")
    snapshots: dict[str, Any] = Field(
        description="A dictionary containing the state snapshots, where the key is the state name and the value is the state snapshot"
    )


class Lock(JournaledMessage):
    """A lock message

    Sent when the agent wants to acquire a distributed lock on the rekuest backend.
    """

    type: Literal[FromAgentMessageType.LOCK] = FromAgentMessageType.LOCK
    key: str
    task: str


class Unlock(JournaledMessage):
    """An unlock message

    Sent when the agent wants to release a distributed lock on the rekuest backend.
    ``task`` names the task that held it.
    """

    type: Literal[FromAgentMessageType.UNLOCK] = FromAgentMessageType.UNLOCK
    key: str
    task: str | None = Field(default=None, description="The task that held the lock.")


class EffectKind(str, Enum):
    """What a task took from outside itself (see ``Effect``)."""

    NOW = "NOW"
    """The clock: ``value`` is epoch seconds (float)."""
    RANDOM = "RANDOM"
    """Random bytes: ``value`` is their hex."""
    SLEEP = "SLEEP"
    """A sleep: ``value`` is its deadline, in epoch seconds (float)."""
    RECORD = "RECORD"
    """Any value the task took through ``task.record(fn)``: JSON."""
    HOLD = "HOLD"
    """A hold's decision: ``value`` is the ``Decision`` (``approved``, ``answer``, ``auto``)."""


class RecordedEffect(BaseModel):
    """A value a workflow recorded, which its resumed run gets back by ``key``."""

    key: str
    effect: EffectKind
    value: Any = None


class Journal(BaseModel):
    """What a workflow recorded before its agent died: handed back so the resume can replay it.

    Calls it made are not listed: re-issued with the same call key, each finds its child
    again (the server answers with the child's events so far).
    """

    last_step: int = Field(default=0, description="The last step the task took; new reports continue after it.")
    effects: list[RecordedEffect] = Field(default_factory=list)


class Effect(FromAgentEvent):
    """A value a task took from outside itself, recorded so that a replay can return
    the same value instead of taking a new one. Record-only for now."""

    type: Literal[FromAgentMessageType.EFFECT] = FromAgentMessageType.EFFECT
    task: str
    effect: EffectKind
    value: Any
    key: str | None = None
    """What the task calls this value (by default the kind and occurrence, ``NOW:1``)."""


class TaskStatus(str, Enum):
    """What an agent says about a task it is asked about (``Inquire``), from memory and at
    once: asking never touches the task."""

    UNKNOWN = "UNKNOWN"
    """The agent has no such task. From the agent the task was given to, that is final: the
    backend ends the task as lost."""
    WAITING = "WAITING"
    """Taken, not started yet (behind a lock, or while the app starts)."""
    RUNNING = "RUNNING"
    """Started, and at work."""
    PAUSED = "PAUSED"
    """Stopped at a pause point, until it is resumed."""
    HELD = "HELD"
    """Waiting for a person's decision (a hold it asked for)."""
    ENDED = "ENDED"
    """The agent reported its end."""


class OnLost(str, Enum):
    """What an assign asks for should its task's agent be lost while it runs.

    Not asked, the implementation decides: a workflow goes on from what it recorded (nothing
    is done twice: what it did is read back, not repeated), anything else ends lost. Running
    something again from the start is never done unasked.
    """

    FAIL = "FAIL"
    """It ends lost, whatever it is: the caller wants to decide what comes next."""
    RESUME = "RESUME"
    """A workflow is sent again with what it recorded, and goes on. Refused for anything else."""
    RESTART = "RESTART"
    """It is run again from the start. Refused unless the action is declared idempotent or pure."""


class Inquire(Message):
    """Backend → agent: what of this task? Answered at once with an ``InquiryAnswer``.

    A question, not a command: it has no position, is not kept, and changes nothing. The
    backend asks when whoever waits for the task asks it to, and about what an agent had
    under way when it comes back from a dropped connection.
    """

    type: Literal[ToAgentMessageType.INQUIRE] = ToAgentMessageType.INQUIRE
    task: str
    request: str | None = Field(default=None, description="Says which question an answer belongs to.")


class InquiryAnswer(Message):
    """Agent → backend: what this agent knows of the task it was asked about."""

    type: Literal[FromAgentMessageType.INQUIRY_ANSWER] = FromAgentMessageType.INQUIRY_ANSWER
    task: str
    status: TaskStatus
    request: str | None = None
    step: int | None = Field(default=None, description="The last step the task reported.")
    reported_at: float | None = Field(
        default=None, description="When the task last reported anything, in seconds since the epoch."
    )
    progress: int | None = Field(default=None, description="What the task last said of its progress.")
    message: str | None = Field(default=None, description="The message of its last progress or log.")

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler: SerializerFunctionWrapHandler) -> Any:  # noqa: ANN401
        data = handler(self)
        if isinstance(data, dict):
            for key in ("request", "step", "reported_at", "progress", "message"):
                if key in data and data[key] is None:
                    del data[key]
        return data


class InquireRequest(Message):
    """Caller → backend: is the agent of this task, which I assigned, still on it?

    A read: not numbered, not kept, and nothing is done to the task. Answered with an
    ``InquireResponse``. How long to wait for a task, and what to do when its agent no
    longer knows it, stays with whoever asks.
    """

    type: Literal[FromAgentMessageType.INQUIRE_REQUEST] = FromAgentMessageType.INQUIRE_REQUEST
    task: str


class InquireResponse(Message):
    """Backend → caller: what the task's agent says of it, or, where the agent was not asked
    (it is away, or has not acknowledged the assign yet), what the backend knows."""

    type: Literal[ToAgentMessageType.INQUIRE_RESPONSE] = ToAgentMessageType.INQUIRE_RESPONSE
    request: str
    task: str
    status: TaskStatus | None = Field(
        default=None,
        description="What the agent said; ENDED for a task the backend already closed. None where the agent was not asked or did not answer.",
    )
    asked: bool = Field(default=False, description="Whether the agent was asked.")
    detail: str | None = Field(default=None, description="Why the agent was not asked, or why there is no answer.")
    step: int | None = Field(default=None, description="The last step the task reported, as its agent has it.")
    reported_at: float | None = Field(
        default=None, description="When the task last reported anything, as its agent has it (seconds since the epoch)."
    )
    progress: int | None = Field(default=None, description="What the task last said of its progress.")
    message: str | None = Field(default=None, description="The message of its last progress or log.")
    error: str | None = None


class AgentDeclaration(BaseModel):
    """What an agent offers, as it goes out in ``Register``: registering *is* implementing.

    The same shapes as ``ImplementAgentInput``, already dumped to JSON (this module cannot
    import the schema). The backend reconciles it in one transaction before it answers
    ``Init``, skips the reconciliation when ``hash`` is the one it already holds, and refuses
    the connection when the declaration does not fit (``ProtocolError`` before ``Init``).
    """

    name: str | None = None
    description: str | None = None
    """What the agent is, in a sentence. Omitting it leaves whatever the agent already
    has -- unlike ``name``, which the backend falls back to the client id for."""
    hash: str | None = None
    implementations: list[dict[str, Any]] | None = None
    states: list[dict[str, Any]] | None = None
    locks: list[dict[str, Any]] | None = None
    bloks: list[dict[str, Any]] | None = None
    assets: list[dict[str, Any]] | None = None
    """The files the agent brings, by name. It uploads what the backend does not hold."""


class AgentGpu(BaseModel):
    """One GPU of an agent's machine."""

    name: str
    memory_mib: int | None = None


class AgentHardware(BaseModel):
    """What an agent's machine has, as a runtime formula reads it. Unknown stays ``None``."""

    cpu_count: int | None = None
    memory_mib: int | None = None
    gpus: list[AgentGpu] = Field(default_factory=list)


class Register(AgentDeclaration, Message):
    """A register message

    A register message is sent from the agent to the rekuest backend
    to register the agent with the rekuest backend: its token, its session and its
    :class:`AgentDeclaration`. Registering is implementing; a ``Register`` without a
    declaration only makes sure the agent exists.
    """

    type: Literal[FromAgentMessageType.REGISTER] = FromAgentMessageType.REGISTER
    token: str
    spec_version: str = Field(
        description="The arkitekt-spec this agent runs, which a server refuses to go without. It says what the agent was written against: what a server may send it is what `features` says. Not part of the declaration or its hash, so an upgrade that declares nothing new still says it.",
    )
    force: bool = False
    """If another connection is already registered for this agent, kick it and take over.

    Only honoured for participants that ``executes_work`` (the executor singleton). A
    non-executor (frontend/observer) never force-displaces — its other connections coexist."""
    session_id: str | None = Field(
        default=None,
        description="Per-process identifier minted in-memory by the executor at start-up (never persisted). Its volatility is the reclaim signal: a reconnect with the SAME session_id means the process survived (reclaim in-flight work); a DIFFERENT session_id means a fresh process (fail-and-cascade). Omitted by non-executors.",
    )
    supervisor: str | None = Field(
        default=None,
        description="The app that says it controls this agent's lifecycle (e.g. 'plugin_engine'): a self-declared hint, stored on the agent on every Register. Not part of the definition hash.",
    )
    hardware: AgentHardware | None = None
    """What the agent's machine has, written on every ``Register``: not part of the declaration
    or its hash, so a hash-equal reconnect still says it, and moving machines declares nothing."""
    features: list[str] = Field(
        default_factory=list,
        description="What this agent implements beyond the base protocol. 'state_requires': it tests an Assign's state_requires; a server sends none to an agent that does not say so. 'protocol_2': it speaks this protocol, without which a server refuses it.",
    )
    nonce: str | None = Field(
        default=None,
        description="New for every connection attempt; the Init that answers this Register says it again.",
    )
    commands_seen: int = Field(
        default=0,
        description="The highest command position this process has handled; 0 for a fresh process. The server sends its commands from the one after.",
    )
    sessions: list[str] = Field(
        default_factory=list,
        description="The journal sessions this agent still holds unacknowledged frames of.",
    )
    tasks: list[str] | None = Field(
        default=None,
        description="Every task this process holds, in whatever state: waiting, under way, or ended with its report still to come. A server that had one of its tasks under way and does not find it here ends it as lost, at once. None from an agent that does not say: the server asks about each instead.",
    )


class ProtocolError(Message):
    type: Literal[ToAgentMessageType.PROTOCOL_ERROR] = ToAgentMessageType.PROTOCOL_ERROR
    error: str
    """ The error message that was raised by the agent"""


class Init(Message):
    """An init message

    An init message is sent from the rekuest backend to the agent
    as a response to the register message. It contains the
    information about the agent and the rekuest backend.
    """

    type: Literal[ToAgentMessageType.INIT] = ToAgentMessageType.INIT
    agent: str
    hash: str | None = Field(
        default=None,
        description="The definition hash the backend now holds for this agent: what the Register declared, or what an earlier one did. None when it was never implemented.",
    )
    diagnostics: list["RegistrationDiagnostic"] = Field(
        default_factory=list,
        description="Non-fatal findings of the registration the Register carried; the agent surfaces them as CatalogWarnings.",
    )
    nonce: str | None = Field(
        default=None,
        description="The nonce of the Register this answers. An Init that is not the current connection's is ignored.",
    )
    positions: dict[str, int] = Field(
        default_factory=dict,
        description="How far the backend has projected each journal session the Register listed. The agent drops what is at or below it and sends the rest.",
    )


class RegistrationDiagnostic(BaseModel):
    """A non-fatal finding of a registration, as the backend reports it on ``Init``."""

    level: str = "WARNING"
    code: str
    message: str
    path: str | None = None


# --------------------------------------------------------------------------- #
# Shelving: the agent mints the drawer's id (``resource_id``) itself, and nothing
# is replied (``JOURNAL_ACK`` covers numbered frames).
# --------------------------------------------------------------------------- #


class Shelve(JournaledMessage):
    """Record that the agent holds a value in memory: a drawer on its shelve, named by
    the ``resource_id`` the agent minted."""

    type: Literal[FromAgentMessageType.SHELVE] = FromAgentMessageType.SHELVE
    identifier: str
    resource_id: str
    label: str | None = None
    description: str | None = None
    task: str | None = Field(
        default=None, description="The task that shelved the value, if one did."
    )


class Unshelve(JournaledMessage):
    """The agent dropped a drawer (answering ``Collect``, or on its own); ``drawer`` is
    its resource id."""

    type: Literal[FromAgentMessageType.UNSHELVE] = FromAgentMessageType.UNSHELVE
    drawer: str


class AssignRequest(JournaledMessage):
    """A caller's request to originate (or resolve) a task over the agent socket.

    This is the WebSocket equivalent of the GraphQL ``assign`` mutation: a participant
    holding ``can_assign_root`` (for a parentless root) or running inside a resolved
    task (for a dependent) asks the backend to create work and dispatch it to an
    executing agent. The fields mirror ``facade.inputs.AssignInputModel``.

    Creation safety is by **idempotency, not transport**: ``reference`` must be stable for
    a logical request, so a resend after reconnect returns the same task (see
    ``AssignResponse``) rather than creating a duplicate. A child call of a numbering
    agent also carries the parent's step it took (``parent_step``), and is idempotent on
    ``(parent, parent_step)`` as well.
    """

    type: Literal[FromAgentMessageType.ASSIGN_REQUEST] = (
        FromAgentMessageType.ASSIGN_REQUEST
    )
    reference: str | None = Field(
        default=None,
        description="Caller-supplied idempotency key, stable across resends of the same logical request; idempotent on (caller, reference). The caller's own: the server never derives it. Minted by the server when omitted.",
    )
    args: dict[str, ShallowJSONSerializable] = Field(
        default_factory=dict, description="The args of the task (ports → values)."
    )
    action: str | None = Field(
        default=None, description="The action ID to assign to."
    )
    action_hash: str | None = Field(
        default=None, description="The action hash to assign to."
    )
    implementation: str | None = Field(
        default=None, description="A direct implementation ID to assign to."
    )
    agent: str | None = Field(
        default=None, description="A direct agent ID to assign to (with interface)."
    )
    interface: str | None = Field(
        default=None, description="The implementation interface (only with agent)."
    )
    parent: str | None = Field(
        default=None,
        description="The parent task ID. None for a root task (requires can_assign_root).",
    )
    parent_step: int | None = Field(
        default=None,
        description="The parent's step this call takes (numbering agents). With it, the request is idempotent on (parent, parent_step): a call re-issued after a restart returns the same child, whatever its reference. Stored as the child's parent_step. Not ``task_step``: that is the envelope's journal stamp.",
    )
    call_key: str | None = Field(
        default=None,
        description="What the parent calls this child: its own key, or one derived from the target, args and occurrence. Idempotent on (parent, call_key), checked before the step: concurrent calls reserve steps in no fixed order.",
    )
    dependency: str | None = Field(
        default=None,
        description="The dependency key to resolve when running inside a resolved task.",
    )
    method: str | None = Field(
        default=None, description="The dependency method to assign."
    )
    resolution: str | None = Field(
        default=None,
        description="The resolution ID for an implementation with dependencies.",
    )
    hooks: list[dict[str, Any]] | None = Field(
        default=None, description="Lifecycle hooks for the task."
    )
    capture: bool | None = Field(
        default=None, description="Whether to run in debug capture mode."
    )
    step: bool | None = Field(
        default=None, description="Whether to step to breakpoints."
    )
    prompt: AssignPrompt | None = Field(
        default=None,
        description="Why the caller wants the child run. Unless the run is in auto-mode the server holds the child until a person approves it, and shows this next to what it knows.",
    )
    state_requires: list[StateRequirement] = Field(
        default_factory=list,
        description="What the child's agent's state must hold for it to run (see Assign.state_requires).",
    )
    on_lost: OnLost | None = Field(
        default=None,
        description="What to do with the child should its agent be lost while it runs. Left out, a workflow resumes and anything else ends lost.",
    )
    max_resumes: int | None = Field(
        default=None,
        description="How often the child may be sent again (resumed or restarted) before it ends lost after all. Left out, the server's own limit applies.",
    )
    traceparent: str | None = Field(
        default=None,
        description="Where in the parent's trace the call was made, as a W3C `traceparent`: the child continues from there. Left out, the child continues the parent task's own.",
    )


class StateRevisionRequest(Message):
    """A workflow's guard asking about a state it depends on (``task.guard``).

    Without ``since``: the state's revision now, which the guard records. With ``since``
    (the recorded one, on a resumed run): whether anything other than the workflow's own
    calls changed the guarded ``paths`` since, or the state was set up again.
    """

    type: Literal[FromAgentMessageType.STATE_REVISION_REQUEST] = FromAgentMessageType.STATE_REVISION_REQUEST
    parent: str
    dependency: str
    state: str
    since: dict[str, Any] | None = None
    paths: list[str] = Field(default_factory=list)


class StateRevisionResponse(Message):
    """The answer to a ``StateRevisionRequest``."""

    type: Literal[ToAgentMessageType.STATE_REVISION_RESPONSE] = ToAgentMessageType.STATE_REVISION_RESPONSE
    request: str
    revision: dict[str, Any] | None = None
    changed: bool | None = None
    detail: str | None = None
    error: str | None = None


class AssignResponse(Command):
    """The backend's authoritative ack that an ``AssignRequest`` was persisted.

    Echoes the originating request id (``request``) and carries the durable task
    id, so the caller can map ``reference``/``request`` → ``task`` BEFORE any
    subsequent task events arrive (those are keyed only by task id). A
    resend of the same ``reference`` yields the same ``task`` with ``created=False``.
    """

    type: Literal[ToAgentMessageType.ASSIGN_RESPONSE] = (
        ToAgentMessageType.ASSIGN_RESPONSE
    )
    request: str = Field(description="The id of the AssignRequest this result answers.")
    reference: str = Field(
        description="The task's reference: the request's, or the one the server minted when the request had none."
    )
    task: str | None = Field(
        default=None, description="The durable task id, or None when error is set."
    )
    created: bool = Field(
        default=True,
        description="False when an existing task was returned (a duplicate reference, or a (parent, parent_step) already taken).",
    )
    error: str | None = Field(
        default=None,
        description="A human-readable error if the assign was rejected (e.g. missing can_assign_root).",
    )


class ProbeRequest(Message):
    """An agent's request to fire a probe over the socket.

    The socket twin of the GraphQL ``probe`` mutation: an ephemeral, zero-persistence
    invocation under the requesting agent's own identity. Unlike ``AssignRequest`` no
    ``parent`` exists — probes are always provenance roots and cannot join a task tree.
    Best-effort: there is no durable row to dedupe resends against, so a resend after a
    lost ``ProbeResponse`` fires a NEW probe (probes are cheap and TTL-bounded).
    """

    type: Literal[FromAgentMessageType.PROBE_REQUEST] = (
        FromAgentMessageType.PROBE_REQUEST
    )
    reference: str | None = Field(
        default=None,
        description="An optional requester-side reference echoed to the executor.",
    )
    args: dict[str, ShallowJSONSerializable] = Field(
        default_factory=dict, description="The args of the probe (ports → values)."
    )
    action: str | None = Field(default=None, description="The action ID to probe.")
    action_hash: str | None = Field(
        default=None, description="The action hash to probe."
    )
    implementation: str | None = Field(
        default=None, description="A direct implementation ID to probe."
    )


class ProbeResponse(Message):
    """The backend's answer to a ``ProbeRequest``: the probe id, or a refusal.

    Events of the probe then stream to the requester as ``…Event`` mirrors whose
    ``task`` is the probe id (``p-…``) and whose ``seq`` is the per-probe counter.
    """

    type: Literal[ToAgentMessageType.PROBE_RESPONSE] = ToAgentMessageType.PROBE_RESPONSE
    request: str = Field(description="The id of the ProbeRequest this answers.")
    probe: str | None = Field(
        default=None, description="The probe id (p-…), or None when error is set."
    )
    error: str | None = Field(
        default=None,
        description="A human-readable error if the probe was refused (allow_probe not declared, cap exceeded, …).",
    )


class ControlRequest(JournaledMessage):
    """Base for a caller's lifecycle-control request over the socket (cancel/interrupt/…).

    The WebSocket equivalent of the GraphQL postman lifecycle mutations: the caller that
    originated a task drives its lifecycle. The request is two-phase — it broadcasts a
    ToAgent control message and is acked with a ``ControlResponse``; the *outcome*
    (CANCELLED/PAUSED/…) is observed via the ``…Event`` mirror stream, not this ack.
    """

    task: str = Field(description="The task to control (must be owned by this caller).")


class CancelRequest(ControlRequest):
    """Request a graceful cancel of a task."""

    type: Literal[FromAgentMessageType.CANCEL_REQUEST] = (
        FromAgentMessageType.CANCEL_REQUEST
    )
    auto_interrupt: float | None = Field(
        default=None,
        description="Seconds. If the cancel is not confirmed within this window, auto-escalate to an interrupt. None disables escalation (the cancel stays pending until the agent confirms or the caller escalates manually).",
    )


class InterruptRequest(ControlRequest):
    """Request a forceful interrupt (propagates to all children)."""

    type: Literal[FromAgentMessageType.INTERRUPT_REQUEST] = (
        FromAgentMessageType.INTERRUPT_REQUEST
    )


class PauseRequest(ControlRequest):
    """Request the agent to suspend the task."""

    type: Literal[FromAgentMessageType.PAUSE_REQUEST] = (
        FromAgentMessageType.PAUSE_REQUEST
    )


class ResumeRequest(ControlRequest):
    """Request the agent to resume a suspended task.

    ``step=True`` resumes only to the next breakpoint (the equivalent of the old step
    instruction); ``step=False`` runs on freely."""

    type: Literal[FromAgentMessageType.RESUME_REQUEST] = (
        FromAgentMessageType.RESUME_REQUEST
    )
    step: bool = False


class ControlResponse(Command):
    """The backend's ack that a caller lifecycle-control *request* was accepted (or rejected).

    ``accepted`` is True once the request was persisted (an ``-ING`` event) and broadcast to the
    executing agent; False (with ``error``) when rejected — e.g. the task is not owned by
    this caller, is unknown, or is already terminal. The resolved outcome arrives later as a
    ``…Event`` mirror.
    """

    type: Literal[ToAgentMessageType.CONTROL_RESPONSE] = (
        ToAgentMessageType.CONTROL_RESPONSE
    )
    request: str = Field(
        description="The id of the CancelRequest/InterruptRequest/PauseRequest/ResumeRequest this answers."
    )
    task: str | None = Field(default=None, description="The controlled task id.")
    accepted: bool = Field(
        description="True when the request was accepted (broadcast + -ING persisted); False when rejected."
    )
    error: str | None = Field(
        default=None,
        description="A human-readable reason when the request was rejected.",
    )


class JournalAck(Message):
    """Backend → agent: everything of ``journal_session`` through ``through`` is projected.

    Cumulative, and the only thing that retires a numbered frame the agent retains
    (terminal reports included).
    """

    type: Literal[ToAgentMessageType.JOURNAL_ACK] = ToAgentMessageType.JOURNAL_ACK
    journal_session: str = Field(description="The journal session the position counts in.")
    through: int = Field(description="Every position up to and including this one is persisted.")


class JournalNack(Message):
    """Backend → agent: a frame of ``journal_session`` arrived ahead of ``expected`` and was
    not taken. The agent sends again from ``expected``, in order; if it no longer holds that
    frame it says so with a ``JournalSkip``."""

    type: Literal[ToAgentMessageType.JOURNAL_NACK] = ToAgentMessageType.JOURNAL_NACK
    journal_session: str = Field(description="The journal session the position counts in.")
    expected: int = Field(description="The position the backend takes next.")


class CommandAck(Message):
    """Agent → backend: every command through ``through`` is handled. Cumulative; the
    backend forgets them."""

    type: Literal[FromAgentMessageType.COMMAND_ACK] = FromAgentMessageType.COMMAND_ACK
    through: int = Field(description="Every command up to and including this position is handled.")


class JournalSkip(Message):
    """Agent → backend: the reports ``first`` to ``last`` of ``session`` are gone (the agent
    no longer holds them). The answer to a ``JournalNack`` it cannot satisfy, and the only
    way a position is ever passed over: the backend records the loss and goes on."""

    type: Literal[FromAgentMessageType.JOURNAL_SKIP] = FromAgentMessageType.JOURNAL_SKIP
    session: str = Field(description="The journal session the positions count in.")
    first: int
    last: int


class ExecutionEvent(Command):
    """Base for backend→caller task-event mirrors.

    When a participant originates work (``AssignRequest``), each resulting task
    event is streamed back to it over its own socket as one of the ``…Event`` subclasses
    below — a minimal mirror of the persisted ``TaskEvent``, so the caller never
    needs GraphQL to read results. A mirror is a command: kept by the backend until the
    caller acknowledged it, and sent again after a reconnect.

    Correlation: ``task`` is the key the caller already learned from
    ``AssignResponse``. ``event`` is the originating ``TaskEvent`` id: a mirror seen twice (a
    replay overlapping what arrived live) is the same event.
    """

    task: str = Field(
        description="The task this event belongs to (the caller's correlation key)."
    )
    event: str = Field(
        description="The originating TaskEvent id: what a repeated mirror is recognised by."
    )


class BoundEvent(ExecutionEvent):
    """The task was bound to an agent."""

    type: Literal[ToAgentMessageType.BOUND_EVENT] = ToAgentMessageType.BOUND_EVENT


class QueuedEvent(ExecutionEvent):
    """The task was queued."""

    type: Literal[ToAgentMessageType.QUEUED_EVENT] = ToAgentMessageType.QUEUED_EVENT


class StartedEvent(ExecutionEvent):
    """The agent accepted the task and began executing it."""

    type: Literal[ToAgentMessageType.STARTED_EVENT] = ToAgentMessageType.STARTED_EVENT


class ProgressEvent(ExecutionEvent):
    """The executing agent reported progress."""

    type: Literal[ToAgentMessageType.PROGRESS_EVENT] = ToAgentMessageType.PROGRESS_EVENT
    progress: int | None = None
    message: str | None = None


class DelegateEvent(ExecutionEvent):
    """The task was delegated to another task."""

    type: Literal[ToAgentMessageType.DELEGATE_EVENT] = ToAgentMessageType.DELEGATE_EVENT


class YieldEvent(ExecutionEvent):
    """The executing agent yielded a result."""

    type: Literal[ToAgentMessageType.YIELD_EVENT] = ToAgentMessageType.YIELD_EVENT
    returns: dict[str, Any] | None = None


class CompletedEvent(ExecutionEvent):
    """The task finished successfully."""

    type: Literal[ToAgentMessageType.COMPLETED_EVENT] = (
        ToAgentMessageType.COMPLETED_EVENT
    )


class LogEvent(ExecutionEvent):
    """A log line from the executing agent."""

    type: Literal[ToAgentMessageType.LOG_EVENT] = ToAgentMessageType.LOG_EVENT
    message: str | None = None
    level: LogLevelLiteral = "INFO"


class CancellingEvent(ExecutionEvent):
    """The task is being cancelled."""

    type: Literal[ToAgentMessageType.CANCELLING_EVENT] = (
        ToAgentMessageType.CANCELLING_EVENT
    )


class CancelledEvent(ExecutionEvent):
    """The task was cancelled."""

    type: Literal[ToAgentMessageType.CANCELLED_EVENT] = (
        ToAgentMessageType.CANCELLED_EVENT
    )


class InterruptingEvent(ExecutionEvent):
    """The task is being interrupted."""

    type: Literal[ToAgentMessageType.INTERRUPTING_EVENT] = (
        ToAgentMessageType.INTERRUPTING_EVENT
    )


class InterruptedEvent(ExecutionEvent):
    """The task was interrupted."""

    type: Literal[ToAgentMessageType.INTERRUPTED_EVENT] = (
        ToAgentMessageType.INTERRUPTED_EVENT
    )


class PausingEvent(ExecutionEvent):
    """The task is being paused."""

    type: Literal[ToAgentMessageType.PAUSING_EVENT] = ToAgentMessageType.PAUSING_EVENT


class PausedEvent(ExecutionEvent):
    """The task was paused (suspended)."""

    type: Literal[ToAgentMessageType.PAUSED_EVENT] = ToAgentMessageType.PAUSED_EVENT


class HeldEvent(ExecutionEvent):
    """The child waits for a person to approve it: nothing runs until someone does."""

    type: Literal[ToAgentMessageType.HELD_EVENT] = ToAgentMessageType.HELD_EVENT
    prompt: AssignPrompt | None = None


class DeniedEvent(ExecutionEvent):
    """A person refused to let the child run: it ended without ever starting."""

    type: Literal[ToAgentMessageType.DENIED_EVENT] = ToAgentMessageType.DENIED_EVENT
    answer: str | None = None


class ResumingEvent(ExecutionEvent):
    """The task is being resumed."""

    type: Literal[ToAgentMessageType.RESUMING_EVENT] = ToAgentMessageType.RESUMING_EVENT


class ResumedEvent(ExecutionEvent):
    """The task was resumed (running again)."""

    type: Literal[ToAgentMessageType.RESUMED_EVENT] = ToAgentMessageType.RESUMED_EVENT


class FailedEvent(ExecutionEvent):
    """The task errored (potentially recoverable)."""

    type: Literal[ToAgentMessageType.FAILED_EVENT] = ToAgentMessageType.FAILED_EVENT
    error: str | None = None
    code: str | None = None
    """The FAILED report's code, passed on (STATE_PRECONDITION)."""


class CriticalEvent(ExecutionEvent):
    """The task hit an unrecoverable error."""

    type: Literal[ToAgentMessageType.CRITICAL_EVENT] = ToAgentMessageType.CRITICAL_EVENT
    error: str | None = None


class LostEvent(ExecutionEvent):
    """The agent running the task died while it ran: not failed, how it ended is unknown.

    Carries what is known, for whoever decides what to do: whether it was ever picked up
    (if not, nothing ran), the last progress it reported, and its implementation's effects.
    """

    type: Literal[ToAgentMessageType.LOST_EVENT] = ToAgentMessageType.LOST_EVENT
    started: bool = True
    last_progress: int | None = None
    effects: str | None = None
    reason: str | None = None


# Every backend→caller mirror, in TaskEventKind order. Imported by ``facade.caller_events``.
ExecutionEventMessage = (
    BoundEvent
    | QueuedEvent
    | StartedEvent
    | ProgressEvent
    | DelegateEvent
    | YieldEvent
    | CompletedEvent
    | LogEvent
    | CancellingEvent
    | CancelledEvent
    | InterruptingEvent
    | InterruptedEvent
    | PausingEvent
    | PausedEvent
    | HeldEvent
    | DeniedEvent
    | ResumingEvent
    | ResumedEvent
    | FailedEvent
    | CriticalEvent
    | LostEvent
)


#: Terminal agent→backend reports: they close their task's gate.
TERMINAL_REPORTS = (
    Completed,
    Failed,
    Critical,
    Cancelled,
    Interrupted,
)

#: The backend→caller mirrors of exactly those reports, in the same order. Kept adjacent
#: to ``TERMINAL_REPORTS`` so the pairing is visible and the two cannot drift apart: the
#: caller side ends a delegated task's stream on these.
TERMINAL_EVENT_MIRRORS = (
    CompletedEvent,
    FailedEvent,
    CriticalEvent,
    CancelledEvent,
    InterruptedEvent,
    LostEvent,
    DeniedEvent,
)


ToAgentMessage = Union[
    Init,
    Assign,
    Cancel,
    Interrupt,
    Heartbeat,
    Pause,
    Resume,
    Decide,
    BlokRefused,
    Collect,
    ProtocolError,
    Bounce,
    Kick,
    JournalAck,
    JournalNack,
    Inquire,
    InquireResponse,
    AssignResponse,
    ProbeResponse,
    StateRevisionResponse,
    ControlResponse,
    # ...plus every backend→caller mirror (kept in one place above).
    *get_args(ExecutionEventMessage),
]
FromAgentMessage = (
    Critical
    | Log
    | Progress
    | Started
    | Completed
    | Failed
    | Yield
    | Register
    | HeartbeatEvent
    | CommandAck
    | JournalSkip
    | InquiryAnswer
    | InquireRequest
    | Resumed
    | Paused
    | Held
    | Inform
    | Cancelled
    | Interrupted
    | StatePatch
    | StateUpdate
    | StateSnapshot
    | Lock
    | Unlock
    | SessionInit
    | AssignRequest
    | ProbeRequest
    | StateRevisionRequest
    | CancelRequest
    | InterruptRequest
    | PauseRequest
    | ResumeRequest
    | Shelve
    | Unshelve
    | Effect
)


Assign.model_rebuild()
