## Dtypes and quantization

- Weight storage dtypes handled by the loaders: float32, float16, bfloat16,
  fp8 (E4M3, E5M2, and FNUZ variants)
- Quantized formats recognized and executable: fp8 E4M3/E5M2 and NVFP4 for
  registered components, plus INT8 ConvRot for registered MiniMax H3, SeedVR2,
  MiniMax Music 3, Wan 2.1 Animate2, and Ideogram 4 components
- Recognized but not yet executable: mxfp8, other int8-tensorwise models,
  ConvRot W4A4, asymmetric W4A8-int8
- fp8-stored checkpoints run on Apple Silicon (MPS) with the affected weights
  held at the compute dtype (roughly double that component's memory);
  unpatched outputs are identical to running the same fp8 checkpoint
  cast-at-use. Memory pressure produces a diagnostic and a memory-budgeted
  loading attempt; actual allocation failures remain explicit.
  The official Qwen Image Base split is verified at BF16 compute on Apple M4.
  MPS enrollment of FP8 hardware-matmul layers uses CPU dequantization and
  portable linear execution at the bound compute dtype, with a diagnostic.
- INT8-stored checkpoints run on Apple Silicon (MPS) with weights kept packed
  and dequantized to the compute dtype at each use; the native INT8 matrix
  multiply is unavailable there
- On macOS 14.5 and later, float16 attention runs in float32 (outputs stay
  float16), matching ComfyUI's workaround for the macOS black-image bug
- Eligible large SDPA calls retain FLASH, cuDNN, efficient, then math priority
  from the first CUDA call onward; cold execution does not discard an attention
  result to initialize backend selection
- When no fused attention backend serves an invocation, the SDPA math
  fallback accumulates float16/bfloat16 in reduced precision, matching
  ComfyUI's setting
- Server-exposed precision settings: independent diffusion, text-encoder, and
  VAE compute dtype policy (`auto`, `float16`, `bfloat16`, or `float32`),
  `fp8-matmul` (fp8 hardware matmul, E4M3), and `aimdo-policy` (offload
  residency: auto/on/off). On supported NVIDIA CUDA workers on Linux and
  Windows, `auto` uses the maintained `Kosinkadink/dinkster-aimdo` fork for
  partial weight offload; `on` requests the same mechanism wherever its
  capability chain passes. When Aimdo is unavailable, selection reports the
  failed capability and explicitly falls back to eager residency. Once Aimdo
  is selected, a component construction failure fails the load instead of
  silently switching that component or later components to eager residency.
  Temporary per-operation weight materialization still uses Aimdo.
  MiniMax H3 enrolls its
  independently loaded diffusion, conditioner, video VAE, and audio VAE
  components through that mechanism. `auto` resolves diffusion to the family
  reference dtype, text
  encoders to bfloat16 except the float32 SD-era CLIP, Anima, Krea 2, Lumina2,
  Wan, and Z-Image towers, and VAE to the first device-supported dtype from
  ComfyUI's family working list. SD-era, Flux, Z-Image, Flux2, and LTX VAEs use
  bfloat16 or fall back to float32;
  Wan-family VAEs can fall back from bfloat16 to float16; MiniMax H3 uses
  float16 video and float32 audio; SeedVR2 and TripoSplat prefer float16
- Native Load Diffusion Model and Load Diffusion Components honor the public
  `weight_dtype` choices for FP8 E4M3, fast FP8 E4M3, and FP8 E5M2 storage.
  The override applies only to diffusion weights; text encoders and VAEs retain
  their independently selected storage and compute dtypes.
- `auto` resolution is constrained by the compute dtypes of the NVIDIA
  devices that actually execute native jobs (ComfyUI's per-device gates:
  capability tiers plus the 10-series and 16-series device lists),
  intersected across multi-GPU replica lanes; hosts without nvidia-smi are
  assumed fully capable
- Cast-at-operation fallback when checkpoint storage differs from the selected
  supported compute dtype
