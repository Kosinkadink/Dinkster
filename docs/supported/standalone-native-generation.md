## Standalone native generation

- The default SD 1.5 workflow runs natively without a ComfyUI checkout. For the
  bare `dinkster` launcher, set `DINKSTER_EXECUTION_PYTHON` to the Python
  environment containing PyTorch and `dinkster_comfy`. The graph
  uses native checkpoint, text-encode, empty-latent, sampler, decode, and
  image-save nodes.
- Pinned core ComfyUI schemas are advertised as import metadata, including
  KSampler and CLIPTextEncode. They are not additional executable node IDs.
  Unsupported or ambiguous imports refuse with node-specific diagnostics.
- Two-rank same-host NVIDIA single-job execution is supported for SD1.5 FP16
  with the fork's SDPA route under torch 2.14.0+cu130. It preserves the stock
  ComfyUI decoded output but does not claim a latency improvement. Whole-job
  replicas remain the multi-GPU throughput option.
- Unmodified legacy custom packs still require a ComfyUI installation and
  run in the compatibility quarantine.
- Native 3D operations execute without a ComfyUI checkout: geometry estimation,
  background removal, Crop Image to Mask, voxel-to-mesh conversion, remeshing,
  decimation, normal smoothing, UV unwrapping, voxel-color painting,
  texture/normal/ambient-occlusion baking, UV-atlas rendering, texture
  application, Get Mesh Info, and Mesh to Model3D. They support variable-size
  mesh batches and textured GLB output.
