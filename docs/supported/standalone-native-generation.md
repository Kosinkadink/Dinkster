## Standalone native generation

- Native nodes can publish encoded preview artifacts into the host asset vault
  without an output mount. Successful job results expose digest-backed artifacts
  through the existing verified asset-serving route.
- Source checkouts can prepare and launch the browser UI without Desktop using
  `run.sh` on Linux or Apple Silicon macOS 14+, or `run.ps1` on Windows.
  Intel macOS is unsupported. The scripts select the native
  execution environment and preserve local model mounts and outputs on reruns.
- Without a ComfyUI root, the server advertises the native core node pack's
  executable class IDs directly. Workflows require every referenced node to be
  present; missing native nodes are not replaced with translated carriers.
  Set `DINKSTER_EXECUTION_PYTHON` to the Python environment containing PyTorch
  and `dinkster_inference` for tensor operations.
- Safetensors weights load from extensionless content-addressed vault paths
  as well as named model mounts. Weight format does not depend on the asset name.
- Consented public Hugging Face model acquisition verifies the provider's pinned
  revision, SHA-256, byte size and declared content identity before publishing
  the provider filename in a shared ComfyUI-layout model folder. Provider/source
  and reported-license metadata must be saved before a new file becomes visible.
  Existing bytes are never overwritten. Model locations are shared across project
  libraries; the default is `~/Dinkster/Models`. Existing read-only locations stay read-only.
  The host declares the SD 1.5 quickstart checkpoint as an available source,
  not as a required download for every checkpoint loader.
- Shared VOXEL values preserve dense occupancy or sparse coordinate tensors,
  all voxel color channels, optional resolution and supported metadata without
  converting geometry layouts.
- The native catalog does not advertise schema-only generation nodes or
  translated import aliases. Unsupported imports report missing node IDs.
- Fleet ingress exposes native job submission, not compatibility prompt routes,
  even when an engine member offers a legacy endpoint.
- Same-host NVIDIA single-job execution supports guidance-lane splitting and
  explicit two- or four-rank H3 Ulysses sequence sharding. Automatic sequence
  selection remains two-rank.
- Unmodified legacy custom packs still require a ComfyUI installation and
  run in the compatibility quarantine.
