## Pack execution isolation

- All supported platforms can run packs in separate worker processes and
  virtual environments for dependency and crash isolation. That alone is not
  a host security boundary. Each manifest directory is on its worker's import
  path regardless of the server's working directory; entry modules may live
  there or in the selected pack interpreter's installed environment. A relative
  `--pack` or `--legacy-pack` path resolves once from the server launch directory;
  only its absolute path reaches workers. Doctor reports the absolute path of the
  imported nodes entry and the interpreter that imported it.
- Linux supports opt-in OS sandboxing for local isolated packs through
  `dinkster-serve --sandbox-packs`. Requested sandboxing fails closed unless full
  user, PID, mount, and network namespace isolation is available. Pack
  manifests request GPU, writable-mount, and network authority; each request
  takes effect only when the host grants it. Network grants name exact HTTPS
  origins and never restore host or loopback network access.
- `dinkster-doctor --sandbox` provides a Linux-only, network-isolated import
  probe jail. It refuses instead of running a requested probe unjailed when
  bubblewrap is unavailable.
- Windows and macOS do not currently provide an OS security sandbox for pack
  workers or doctor probes. Their isolated workers retain the operating
  user's filesystem and network authority.
- In-process packs and remote worker daemons are outside the local serving
  sandbox on every platform.
- Worker conversations carry typed runtime-expansion results. Present
  values retain ordinary resource ownership, source adoption, saved-artifact
  validation and shared-memory/result acknowledgements. Carrying a fragment
  across a worker boundary does not bypass graph admission by the engine.
  Cache stores preserve these declarations with their present values across
  memory retention, disk restart, deferred spill and peer promotion. Incomplete
  persisted declarations are whole-entry misses; resident resources remain
  memory-only and invalidate the entire entry when released.
- Runtime-expanding nodes can return nested child graphs. The engine validates
  each fragment before execution, runs demanded children and generated output
  nodes, and attributes child events to their declared parent and display node.
  Cached parents replay their declarations without invoking the parent again;
  generated output nodes still run. Expansion also works inside existing graph
  regions without changing their port bindings or iteration isolation.
- Packs can author expansions through `dinkster_api.v1.ExpansionBuilder`: add
  local nodes, connect outputs by name or index, and finalize named outputs
  containing ordinary Python values or child references. Present values use
  the same wrapping and ownership rules as ordinary node returns. Output
  indices resolve against the engine's elaborated interface, including when
  the referenced node belongs to another pack.
- Expansion fragments can explicitly reference existing nodes in the same
  run for inputs, returned outputs and display attribution. Missing references,
  incompatible output types and dynamic demand cycles fail at the expanding
  parent before its fragment runs. Completed values stay available for later
  fragments until that run ends.
- In-process nodes can declare `execution_context` to read an immutable live
  graph snapshot and await or publish typed run values. Expanding nodes can
  also inhibit or override nodes that have not started. Changes stay within
  the current run. Waiting releases execution admission and governed memory
  reservations so a publisher can run at concurrency one. Contexts expire
  when their invocation ends; these live-state nodes bypass result caches.
  These control calls are not yet available in isolated or remote workers.
- Model-backed vision provider packs can implement stable owner node schemas.
  Ordinary node cards hide implementation pack ids. A missing selection is
  resolved deterministically from the requested semantic model and compatible
  live workers; stored legacy provider ids remain pinned. The standard vision
  components run in isolated workers and are available without separate pack
  installation. Their declared software dependencies are provisioned before
  workers announce, while exact digest-pinned model artifacts are preflighted
  before queueing. Model acquisition requires explicit digest consent;
  execution never installs software or downloads model files. Remote workers
  with known-incompatible capability evidence are refused; older workers with
  incomplete evidence are tried only when they expose an exact compatible
  execution path, and receipts report what actually ran.
