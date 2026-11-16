## Pack execution isolation

- Pack authors can test a node or Python-declared graph from synchronous
  pytest through `dinkster_engine.testing`. The separately installable
  `dinkster-engine` wheel needs no Torch. Tests share an isolated worker by
  default, can opt into in-process execution, and receive decoded outputs plus
  executed and cached node sets. Doctor and registry validation do not run nodes.
  Doctor permits only the helper's `load_pack` import from test sources;
  production modules and other engine/worker imports remain errors.
- Doctor permits public `dinkster_nodes_<name>` helpers when the consumer
  declares the corresponding `dinkster-<name>` runtime pack dependency.
  Hyphenated pack names use underscores in Python modules. Private symbols,
  private modules and undeclared provider imports remain errors; wheel
  dependencies must still supply the imported code in the worker environment.
- Packs can use image-document construction, editing, rendering, import/export
  and blend helpers through `dinkster_api.v1`, alongside VIDEO values and
  streaming helpers. Remote clients can read host invocation cancellation
  through that API without importing worker internals.
- Packs can register latent-based transport codecs through `dinkster_api.v1`
  using the canonical encoder, decoder, fingerprint and encoded-value validator.
- Packs can register mesh, voxel, shape-subdivision, Load3D camera and model-info
  codecs through the same API without importing Torch or the inference runtime.
- Synchronous ComfyUI-origin node bodies run in inference mode in each of the
  35 first-party packs. Returning or raising restores the caller's mode; async
  orchestration and ordinary gradient-capable nodes remain outside that boundary.
- `dinkster-serve --pack packages/dinkster-nodes-dev` composes the development
  scaffold pack as an ordinary isolated pack. It stays outside the default pack
  set. Its `dev` namespace is not reserved; overlapping pack claims are refused.
- Submit refuses links carrying resident types between packs in different
  processes before queueing the job. The reason names both packs and offers
  admission to the shared inference process or engine-run nodes on references.
  Resident type declarations include MODEL, CLIP, VAE, ControlNet and custom
  registered resident codecs. CONDITIONING, SAMPLER and NOISE can contain
  resident payloads and require the same process when linked between packs.
  Links through region inputs, exports and state feedback follow the original
  producers. A connected resident chain must have one common process.
  The node catalog publishes process affinity for pre-run diagnostics.
  Host-declared unavailable nodes retain their exact refusal in the catalog
  and are refused at submit before any job is queued.
- Value type ownership is discoverable for live and cold-catalog packs without
  activating cold workers. Removing or replacing a pack retracts its old type
  identities; a rejected replacement preserves the active owners.
  Built-in identities remain owned by the engine (`core`) independently of the
  official pack that provides their codec and survive pack removal.
- With sandboxing enabled, a shared group excludes members requesting GPU
  access that the host has not granted. Non-GPU packs can still run in that
  sandboxed group; each excluded pack refuses with its own GPU-grant reason.
  Reload preserves this separation. Unsandboxed placement is unchanged.
- Managed installed and development generations share one worker by default
  for first-party packs. Newly prepared shared environments resolve their
  requirement ranges together without changing the generation's exact torch,
  dinkster-inference or comfy-aimdo versions. Different compatible publication
  pins do not require separate workers. These packs can pass resident MODEL,
  CLIP and VAE values without putting them in the engine process. Third-party
  packs retain separate workers, even when they depend on inference. Explicit
  in-process selections remain unchanged. Existing generations keep their
  recorded shared environment and membership. The remote pack
  remains separate because it requires an asset-vault write grant. Packs requesting
  network or writable-folder grants stay outside the shared worker. Setup, update
  and develop accept `--inference-pack PACK=isolated` to persist a per-pack exclusion;
  `--inference-pack PACK=auto` restores automatic placement. Repair, relocation and
  software snapshots preserve these exclusions. Newly prepared generations retain
  their selected shared-process membership with the software snapshot; serving
  reuses that selection rather than choosing a different environment.
- First-party runtime packs install shared inference, asset and audio codecs in
  their workers. Pack dependencies order composition; isolated workers do not
  share a registry. Family-specific codecs retain their producer's identity,
  including SAM3 tracking values consumed by WAN.
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
- Generic worker startup discovers optional legacy-checkpoint conversion
  components without loading Torch. An advertised conversion command means the
  components are installed, not that their native runtime has been loaded.
  Conversion validates runtime health on first use; an unavailable converter
  returns an explicit error and serving can select another capable worker.
- Linux supports opt-in OS sandboxing for local isolated packs through
  `dinkster-serve --sandbox-packs`. Requested sandboxing fails closed unless full
  user, PID, mount, and network namespace isolation, seccomp kernel-interface
  filtering, and prevention of nested user namespaces are available. Linux
  needs bubblewrap with `--disable-userns` and `--seccomp`, kernel seccomp
  support, and libseccomp2/libseccomp. Missing support is reported before
  pack code starts. The filter denies privileged kernel management, process
  inspection, keyrings, new namespaces, and io_uring; ordinary threads,
  subprocesses, shared memory, and granted GPU access remain available. Pack
  manifests request GPU, writable-mount, and network authority; each request
  takes effect only when the host grants it. Network grants name exact HTTPS
  origins and never restore host or loopback network access.
- macOS supports opt-in OS sandboxing for local isolated packs through
  `dinkster-serve --sandbox-packs`. Workers can access only granted folders
  and their private Unix sockets. Direct networking, including loopback,
  and inspection or signalling of outside processes are denied. GPU access
  requires a host grant and supports physical Apple Silicon GPUs. Virtual
  GPUs requiring AppleParavirtDeviceUserClient are not granted access; a
  GPU-authorized Torch worker reports unavailable sandboxed MPS explicitly
  and can still run CPU operations. Sandboxed packs use separate Metal compile
  caches; when a private cache is unavailable, they compile operations at startup
  without accessing the user's shared cache.
- WSL2 Linux pack sandboxes support GPU grants through the WSL GPU device
  and read-only driver libraries. Windows drive mounts and WSL interop paths
  cannot be granted to pack workers; copy models and packs to the WSL disk.
- `dinkster-env start --sandbox-packs` requests the same isolation for a managed
  local engine. Changing the selection requires stopping the engine first;
  environment updates, recovery and moves preserve the selection. Installed
  wheel packs can import their selected package without exposing sibling
  generation settings or unrelated packs.
- `dinkster-doctor --sandbox-readiness --json` reports whether full pack
  sandboxing is available and explains missing prerequisites without importing
  pack code. `dinkster-env doctor --sandbox --json` asks the selected engine
  for that report. On native Windows, availability also requires the selected
  engine interpreter to exchange wakeup bytes inside a real no-network container.
  A refused wakeup reports the operation, selected socket path and byte length,
  contained temp/profile values and policy profile entry before managed startup.
  The API prerequisite alone does not establish availability; unavailable
  containment never silently falls back to an unsandboxed start.
  An engine without the probe cannot enable sandboxing.
- `dinkster-doctor --sandbox` provides a network-isolated import probe jail
  on Linux and macOS. It refuses instead of running a requested probe
  unjailed when the platform sandbox is unavailable.
- Windows 11 supports opt-in native pack sandboxing when its process-isolation
  API and Desktop's source-built MXC executor are available. Windows 10 and
  unsupported Windows 11 builds refuse a requested native sandbox and explain
  the WSL2 alternative. Worker values travel over inherited pipes, not named
  shared memory. The worker gets an explicit environment and no host loopback
  access without an origin grant. Granted HTTPS origins use an authenticated
  proxy; DNS lookups remain possible in that mode. Windows system paths and
  GPU drivers retain the platform's baseline app-container access; the native
  sandbox does not enforce Linux's GPU-denial or process/file-size limits.
  Windows doctor import-probe jails remain unavailable; readiness reporting
  is separate from executing a pack probe.
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
