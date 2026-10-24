## ComfyUI compatibility

- Four native audio latent constructors preserve Stable Audio, ACE-Step 1.0,
  ACE-Step 1.5 and YuE2 duration rounding, tensor layouts and audio metadata.
- Native EmptyMochiLatentVideo creates video latents with the core interface.
  Empty image and Mochi latents honor CPU-only process policy without requiring
  attention discovery or command-line initialization first.
- Seven native dataset text nodes lowercase, uppercase, truncate, prefix,
  suffix, replace, and strip text under exact core IDs, retaining the fixed
  reference's one-element list payloads on scalar STRING outputs.
- Save Text preserves its core text, filename prefix, and format inputs while
  writing UTF-8 output through the configured mounted output target. Filename
  variables use 1-by-1 dimensions and local time; unknown variables stay literal.
- The native core pack exposes eleven latent creation, batch, transform, resize,
  blend, composite, crop, and noise-mask nodes under their exact ComfyUI IDs.
  IMAGE, MASK, and LATENT values cross ordinary pack boundaries without host type
  overrides.
- Empty SD3, Flux 2, Chroma Radiance, HiDream-O1, Qwen layered, Hunyuan 3D v2,
  and Trellis2 latents preserve each model's dimensions, channel count, and tags.
- Six native latent operation nodes blend, tonemap, sharpen, apply operations
  directly or within a denoising schedule interval, and replace video frames.
  LATENT_OPERATION uses the same resident-value boundary as model hooks.
- Nine native advanced latent nodes add, subtract, multiply, interpolate,
  concatenate, cut, split to batches, join batches, and set batch seed behavior.
- Native CLIP text encoding preserves scheduled conditioning; layer selection
  clones the encoder without changing the source under exact core interfaces.
- FreeU and FreeU_V2 clone models and patch UNet output blocks under their
  exact core interfaces, including CPU Fourier fallback.
- Ten native conditioning nodes combine, blend, concatenate, multiply, mask,
  zero, and set area or timestep metadata under their exact ComfyUI interfaces.
- Native VAE inpainting encoding and model conditioning preserve crop, mask,
  latent, and conditioning metadata behavior under exact ComfyUI interfaces.
- Fifteen native mask and composite nodes convert, create, invert, crop, combine,
  feather, grow, threshold, Porter-Duff composite, and split or join image alpha
  under exact core IDs.
- Six native core image nodes create, resize, invert, batch, and pad images for
  outpainting, preserving source color, alpha, resize, and feathering behavior.
- Five native image filters blend, blur, quantize, sharpen, and scale to a total
  pixel count under their exact ComfyUI interfaces.
- Nine native image transforms crop, repeat and select batches, add seeded noise,
  stitch, resize with padding, rotate, flip, and scale to a maximum dimension.
- Native morphology applies erosion, dilation, opening, closing, gradients and
  top/bottom hats; RGB/YUV conversion splits and joins three image channels.
- Seven native video latent nodes create Hunyuan Video, Hunyuan Video 1.5, LTXV,
  Cosmos, and autoregressive video samples, trim video frames, and freeze LTXV
  video or audio samples under their exact ComfyUI IDs.
- Native Hunyuan image latent construction and refiner conditioning retain the
  source latent metadata while preparing refiner noise latents.
- Portable masks become Torch tensors at native tensor-consuming nodes, without
  moving already-live tensors or loading Torch in the host decoder.
- `dinkster-serve --comfy-root ... --legacy-pack PATH` runs unmodified v1 and
  pure V3 custom-node packs in quarantined isolated workers under namespaced
  node ids.
- Comfy image, mask, audio, and video values cross the compatibility boundary
  through explicit codecs. Invocation-scoped media staging supports source-file
  inputs on POSIX; source-filename custom nodes are unsupported on Windows.
- SD1.5 and MiniMax H3 execute through the pinned `dinkster_inference` package.
  Dinkster owns the graph schemas and worker integration, not a second model or
  sampling implementation.
- AnimateDiff-Evolved runs unmodified through the compatibility worker for SD1.5
  motion models, context and keyframe schedules, and scheduled LoRA, including
  INT8 checkpoints.
- Maintained aliases in the foundation, image, and media packs remain active.
  The retired generation and compatibility aliases are preserved as
  `comfy-aliases.inactive.json` evidence and are unavailable until their
  capability is re-expressed against the fork and its acceptance receipt passes.
- The 602-workflow translation census, source-parity baseline, registry source
  snapshot, and pre-retirement capability evidence remain committed as
  historical evidence. The old capability selectors are inactive and make no
  current support claim.
- Pack HTTP routes, web assets, and executor hooks are diagnosed rather than
  emulated.
