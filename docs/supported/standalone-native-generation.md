## Standalone native generation

- The working native SD 1.5 path today is a SamplerCustomAdvanced graph served
  with `--comfy-root`, as recorded in
  [maintainer issue #114](https://github.com/Kosinkadink/comfy-vibe-station/issues/114).
  Native-only generation without a ComfyUI checkout returns when that issue
  lands. The graph uses native checkpoint, text-encode, empty-latent, sampler,
  decode, and image-save nodes.
- Pinned core ComfyUI schemas are advertised as import metadata, including
  KSampler and CLIPTextEncode. They are not additional executable node IDs.
  Unsupported or ambiguous imports refuse with node-specific diagnostics.
- Single-job multi-GPU device selection does not require ComfyUI.
- Unmodified legacy custom packs still require a ComfyUI installation and
  run in the compatibility quarantine.
- Native 3D operations execute without a ComfyUI checkout: geometry estimation,
  background removal, Crop Image to Mask, voxel-to-mesh conversion, remeshing,
  decimation, normal smoothing, UV unwrapping, voxel-color painting,
  texture/normal/ambient-occlusion baking, UV-atlas rendering, texture
  application, Get Mesh Info, and Mesh to Model3D. They support variable-size
  mesh batches and textured GLB output.
