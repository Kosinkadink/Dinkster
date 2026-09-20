## Pack execution isolation

- All supported platforms can run packs in separate worker processes and
  virtual environments for dependency and crash isolation. That alone is not
  a host security boundary. A pack's entry modules resolve from its manifest
  directory regardless of the server's working directory.
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
