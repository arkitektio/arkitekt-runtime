# arkitekt-runtime

The execution core of an Arkitekt app: what every runtime needs to *run* a declared app.

An app is declared with [arkitekt-spec](../arkitekt-spec). A runtime executes it:
[rekuest](../rekuest) in distributed mode (a socket agent registered with a rekuest server),
[arkitekt-fastapi](../arkitekt-fastapi) in server mode (an HTTP service). Both build on this
package: the agent's assignment handling, actors, `Task`, the journal, state, locks, hook
runners and structure serialization.

It carries no transport and no client -- no websockets, rath or graphql-core. A runtime plugs
in its transport (`agents.transport.base.AgentTransport`), its caller (to call other actions)
and its task scope (e.g. rath's, so service clients attribute requests to the running task).

It is also where a workflow's recovery runs: the replay of recorded values (`task.now()`,
`task.random()`, `task.sleep()`, `task.record(fn)`) from the journal an `Assign` resumes with,
`task.retry` / `task.hold` / `task.guard`, and the rule that only a `WORKFLOW` implementation
may call other actions (`NotAWorkflowError`). A dependency method annotated as a generator
streams the remote action's yields (`iterate` / `aiterate`). The design is described in
[rekuest's workflows guide](../rekuest/docs/workflows.md).
