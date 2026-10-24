## Pack execution isolation

- Value type ownership is discoverable for live and cold-catalog packs without
  activating cold workers. Removing or replacing a pack retracts its old type
  identities; a rejected replacement preserves the active owners.
  Built-in identities remain owned by foundation independently of the
  official pack that provides their codec and survive pack removal.
- Dinkster-specific built-in socket types use bare uppercase ids, including
  ASSET, CURVE, REGION, DETECTION, RESOURCE and VIDEO_DOCUMENT. Schemas,
  codecs and generators share their canonical spelling module. The internal
  core.absent execution envelope is not a socket type.
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
