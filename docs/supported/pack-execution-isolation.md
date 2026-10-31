## Pack execution isolation

- Packs can use image-document construction, editing, rendering, import/export
  and blend helpers through `dinkster_api.v1`, alongside VIDEO values and
  streaming helpers. Remote clients can read host invocation cancellation
  through that API without importing worker internals.
- `dinkster-serve --pack packages/dinkster-nodes-dev` composes the development
  scaffold pack as an ordinary isolated pack. It stays outside the default pack
  set. Its `dev` namespace is not reserved; overlapping pack claims are refused.
- Value type ownership is discoverable for live and cold-catalog packs without
  activating cold workers. Removing or replacing a pack retracts its old type
  identities; a rejected replacement preserves the active owners.
  Built-in identities remain owned by foundation independently of the
  official pack that provides their codec and survive pack removal.
- Typed asset references can target a type declared by an isolated pack without
  loading its codec in the engine. Envelope-aware nodes can forward values
  without decoding payloads, including foundation list selection of images.
- Dinkster-specific built-in socket types use bare uppercase ids, including
  ASSET, AUDIO_RECORD, CURVE, REGION, DETECTION, RESOURCE and VIDEO_DOCUMENT. Schemas,
  codecs and generators share their canonical spelling module. The internal
  core.absent execution envelope is not a socket type.
- Packs can use the canonical `INTERP_MODEL` socket for process-resident
  interpolation models. Values retain their holding-process identity across
  ordinary worker transport; other processes cannot resolve model contents.
  This shared type does not itself provide a model loader or interpolation node.
- Non-dinkster publishers' node schemas and type claims require the pack-name
  prefix at loading and catalog preparation. Declared type claims remain
  exclusive even when the pack has no codec callback. Dependent packs can
  reference an owner's type without registering or claiming it themselves.
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
- Boundary workers with configured pack scratch keep produced asset bytes
  separate for each job attempt. Returning a result still requires verified
  engine adoption before acknowledgement; this does not delete temporary files.
- Saved-file previews preserve scratch handoff intent and verified engine
  adoption state. Host-generated previews also retain adoption inventory;
  existing untracked vault bytes are not reclassified. Incomplete staging
  remains pending and does not authorize deletion.
- Operators can persist a produced-asset size policy, initially 20 GiB or
  explicitly unlimited. Clear reports blocked while global reference ownership
  is unknown; neither changing the policy nor clearing deletes files.
- Completed, failed and cancelled invocation scopes refuse further publication
  into their job-owned scratch, including from detached tasks and held writers.
- Produced scratch claims refuse live invocations and open writers, including
  in another process. Crashed-process locks release without age-based expiry;
  this does not itself collect or discard produced data.
- Serving jobs retain scratch ownership between nodes until run settlement,
  including failure or cancellation. Finishing one job does not release another
  live job's ownership, and missing host ownership state refuses reconciliation.
- Open verified produced-asset readers retain their bytes after the producing
  invocation ends, including readers in another process. Closing the last
  descriptor or exiting the reader process releases its ownership.
- Verified managed paths returned during an invocation retain ownership until
  that invocation ends, even after the verification descriptor closes. Nested
  calls cannot release their parent's paths; expired invocation contexts refuse
  new path retention. Paths retained outside that lifetime need a separate
  owner, and no deletion is enabled.
- Job scratch retains publication ownership and handoff intent across process
  exit. Missing ownership state refuses reuse of nonempty scratch; merely
  inspecting a source does not mark it as offered or adopted.
- Copying produced sources into the engine vault records pending and verified
  adoption across restarts. An interrupted inventory commit can be completed
  from verified destination bytes without the producer. Existing untracked
  inputs and mounted sources are not reclassified, and no deletion is enabled.
- Direct in-process results also adopt produced sources before returning when
  pack scratch is configured. Returned AUDIO and VIDEO sources remain readable
  from the engine vault after scratch removal; failed adoption preserves scratch
  bytes.
- Direct in-process run-value publications use the same durable adoption before
  another node can receive the value. Failed adoption does not publish a value,
  and completed invocation contexts cannot publish again.
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
- Nodes can declare `execution_context` to read an immutable live
  graph snapshot and await or publish typed run values. Expanding nodes can
  also inhibit or override nodes that have not started. Changes stay within
  the current run. Waiting releases execution admission and governed memory
  reservations so a publisher can run at concurrency one. Contexts expire
  when their invocation ends; these live-state nodes bypass result caches.
  Control calls also cross isolated and remote worker conversations. Typed
  publications use ordinary source adoption, resource handoff pins and
  shared-memory acknowledgements; waiting suspends the child reservation.
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
