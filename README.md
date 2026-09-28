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
