## Standalone native generation

- Native generation requires Dinkster's torch runtime, not a ComfyUI checkout.
  SD1.5 text-to-image uses native checkpoint, text-encode, empty-latent,
  sampler, decode, and image-save nodes. ComfyUI EmptyLatentImage imports
  lower to the native empty-latent node with dimensions and batch size preserved.
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
