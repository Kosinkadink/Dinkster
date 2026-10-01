## Standalone native generation

- The default SD 1.5 workflow runs natively without a ComfyUI checkout. For the
  bare `dinkster` launcher, set `DINKSTER_EXECUTION_PYTHON` to the Python
  environment containing PyTorch and `dinkster_inference`. The graph
  uses native checkpoint, text-encode, empty-latent, sampler, decode, and
  image-save nodes.
- Pinned core ComfyUI schemas are advertised as import metadata, including
  KSampler and CLIPTextEncode. They are not additional executable node IDs.
  Unsupported or ambiguous imports refuse with node-specific diagnostics.
- Native diffusion loaders accept supported GGUF models, including SDXL Q8_0
  and K-quants. GGUF residency can minimize VRAM, cache decoded weights within
  the available VRAM budget, or eagerly decode weights while retaining normal
  model offloading.
- Same-host NVIDIA single-job execution supports guidance-lane splitting,
  canonical window scattering, and explicit two- or four-rank H3 Ulysses
  sequence sharding. Automatic sequence selection remains two-rank. H3 Sol-Attn
  and SLA sparse attention are routed through the worker-owned fork attention
  registry and remain incompatible with sequence mode.
- Unmodified legacy custom packs still require a ComfyUI installation and
  run in the compatibility quarantine.
- Native 3D operations execute without a ComfyUI checkout: geometry estimation,
  background removal, Crop Image to Mask, voxel-to-mesh conversion, remeshing,
  decimation, normal smoothing, UV unwrapping, voxel-color painting,
  texture/normal/ambient-occlusion baking, UV-atlas rendering, texture
  application, Get Mesh Info, and Mesh to Model3D. They support variable-size
  mesh batches and textured GLB output.
