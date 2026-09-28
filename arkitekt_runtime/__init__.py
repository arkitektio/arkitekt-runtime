"""arkitekt-runtime: the execution core every runtime of an Arkitekt app shares.

A declared app (:mod:`arkitekt_spec.declare`) is run by a runtime: rekuest in
distributed mode, arkitekt-fastapi in server mode. Both are built on this package --
the agent (:mod:`arkitekt_runtime.agents.base`), actors, :class:`~arkitekt_runtime.task.Task`,
the journal, state, locks, hook runners and structure serialization. A runtime plugs
in its transport, its caller and its task scope; this package has none of its own.
"""
