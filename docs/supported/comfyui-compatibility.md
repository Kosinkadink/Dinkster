## ComfyUI compatibility

- Native POSE_KEYPOINT values preserve frame canvas dimensions, people and flat
  body, foot, face and hand keypoints across isolated hosts, including absent
  and empty optional fields. This is distinct from the existing POSE format.
- The `--comfy-root` and `--execution-python` composition supports an installed
  inference package alongside source-only ComfyUI modules from the checkout,
  while sharing one inference module state and leaving source files unchanged.
- Core schema comparison permits the optional output mount for output nodes
  with a scalar STRING filename prefix, including CLIPSave, VAESave,
  CheckpointSave, and ModelSave, without relying on their class-name spelling.
- The native pack declares bound-file checkpoint, diffusion-model, text
  encoder, vision-encoder, ControlNet, style-model and GLIGEN loader schemas.
  Model construction uses the pinned inference package; file inputs grant
  access only to the selected asset.
- Image-only checkpoint loading returns separate model, vision encoder and VAE
  outputs. LTXV audio VAE loading retains audio and vocoder weights and checkpoint
  metadata; LTXAV text encoding loads its text encoder before its checkpoint.
  Loader CPU checks use tiny files and controlled model constructors, not full models.
- Native Load Audio decodes bound audio files with preserved channel order and PCM
  scaling. Preview Audio returns the audio without writing a preview file.
- Native audio saves write FLAC, MP3 and Opus through the selected output mount,
  preserving audio outputs and batch-specific filename prefixes.
- Native YuE2 ABC and music conditioning preserve generation controls, automatic
  off mode for empty scores, guidance defaults and generated duration. CPU checks
  use controlled encoders, not full-model music generation.
- Native LoRA file loaders apply model and text-encoder strengths separately,
  retain weight metadata, and support ordered stacks and bypass application.
  Disabled stack entries and zero strengths do not resolve their file inputs.
- Native Anima LLLite applies image and optional mask guidance to cloned models
  through scheduled input, self-attention, cross-attention and MLP patches.
- Native SUPIR applies scheduled control strength and optional restoration
  guidance, using checkpoint denoise-encoder weights when present.
- Native KSampler and KSamplerAdvanced use the inference sampling engine with
  seeded batch noise, masks, denoise strength and advanced step boundaries.
- Native standard and custom samplers retain ordered audio/video stream roles
  and masks through sampling and denoised outputs. AddNoise retains those roles
  through empty-latent handling and replaces nonfinite output values with zero.
- Native Wan Uni3C applies render-video guidance with temporal padding, spatial
  alignment, scheduled residuals and shared conditional-batch computation.
- Native Qwen DiffSynth and Z-Image Fun ControlNet apply scheduled image and
  inpaint guidance; USO prepends projected vision features to text tokens.
- Native LoRA hook loaders use bound model/lora assets, preserve prior hooks
  and apply separate model and CLIP strengths while retaining loaded weights.
- Native VAELoader accepts granted VAE files, video approximation files, six
  same-root image TAESD encoder/decoder pairs and the file-free pixel_space
  option on one input. Native DiffusersLoader accepts a granted model subtree.
  Selection names are rechecked on each execution, not reused from result caches.
- Native WanDancer keyframe padding places reference frames and masks into
  video segments, with matching audio slices and scalar or list outputs.
- Native LTXV duration prediction uses caption connectors and the duration head,
  returning raw seconds and a frame count snapped to the causal VAE grid.
- Native ComfyUI AUDIO sockets retain the first registered audio authority
  and work across isolated pack processes without a companion media pack.
- Literal VIDEO_EDIT mappings retain their crop and trim fields across ordinary
  in-process and isolated pack hosts without host type overrides.
- Native LTXVCropGuides removes appended guide frames and their conditioning
  metadata while preserving the remaining video latent and noise mask.
- Native LTXV latent-guide and generated-keyframe nodes attach, separate and
  convert guide frames while preserving conditioning histories and source tensors.
- Native LTXV image guides accept canonical IC-LoRA parameters extracted from
  LoRA model metadata, including sparse spatial reference downscaling.
- LTXV reference-audio, spatio-temporal and modality guidance preserve the source
  model and apply guidance within the requested sampling window. The dual-CFG
  guider applies separate video/audio scales to native multi-stream latents.
  Controlled CPU predictions cover nonempty Euler sampling, masks, seeded noise
  and denoised output restoration; real-model AV sampling is not validated.
- MiniMax H3 reference conditioning pairs image, video and soundtrack inputs;
  Fun ControlNet applies scheduled control while preserving the source model. CPU tests
  use controlled encoders and predictions, not real-model or GPU execution.
- Native SeedVR2 output post-processing aligns video frames, restores alpha
  and applies LAB, wavelet or AdaIN color matching with bounded chunks.
- Native ARVideoI2V encodes the first start-image frame into a cloned model's
  autoregressive configuration and creates the requested empty video latent.
- Native Sampler AR Video constructs the autoregressive video sampler with
  the workflow's frame-block size under its exact core interface.
- Native LTXV sampling patches cloned models and constructs shifted sigma
  schedules from latent token counts, with optional terminal stretching.
- Four native audio latent constructors preserve Stable Audio, ACE-Step 1.0,
  ACE-Step 1.5 and YuE2 duration rounding, tensor layouts and audio metadata.
- Native EmptyMochiLatentVideo creates video latents with the core interface.
  Empty image and Mochi latents honor CPU-only process policy without requiring
  attention discovery or command-line initialization first.
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
- LatentUpscaleModelLoader loads Hunyuan Video 1.5 and LTXV latent-upscale checkpoints
  into LATENT_UPSCALE_MODEL resident values across ordinary native pack hosts without
  loading Torch during registration.
- LATENT_UPSCALE_MODEL preserves resident identity, ownership, and cache lifetime
  across ordinary native pack hosts without loading Torch during registration.
- UPSCALE_MODEL preserves the live image-upscale descriptor and its patcher as a
  resident value across ordinary pack hosts; model contents are not serialized.
- Hunyuan Video 1.5 and LTXV latent upscale consumers preserve core resize,
  normalization, dtype, device, and latent metadata behavior.
- Nine native advanced latent nodes add, subtract, multiply, interpolate,
  concatenate, cut, split to batches, join batches, and set batch seed behavior.
- Native CLIP text encoding preserves scheduled conditioning; layer selection
  clones the encoder without changing the source under exact core interfaces.
- FreeU and FreeU_V2 clone models and patch UNet output blocks under their
  exact core interfaces, including CPU Fourier fallback.
- Video Linear CFG Guidance and Video Triangle CFG Guidance clone model
  patchers and install their native frame-wise CFG callbacks under exact core IDs.
- Native Context Windows nodes clone model patchers and configure manual, Wan,
  and LTXV context-window sampling under exact core IDs.
- Video latent frame replacement preserves the core destination/source/index
  input order, including the optional source input.
- MiniMax H3 sigma shifts clone model patchers, preserve noise scale, and set
  independent video/audio flow shifts without modifying the source model.
- MiniMax H3 image-to-video conditioning encodes optional first and last
  keyframes and scheduled text while creating paired zero audio/video latents.
- MiniMax H3 guides anchor image clips and audio at positive or negative frame
  indices, retaining existing keyframes and cropping audio to the remaining duration.
- Ten native conditioning nodes combine, blend, concatenate, multiply, mask,
  zero, and set area or timestep metadata under their exact ComfyUI interfaces.
- Native hook conditioning preserves paired ordering, default records, mask and
  timestep metadata, and CLIP hook clone scheduling under exact core interfaces.
- Hook keyframes from floats preserve the forced FLOATS socket and its numeric
  default, minimum, and step under the exact core interface.
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
- Eight native video latent nodes create Hunyuan Video, Hunyuan Video 1.5, LTXV,
  Cosmos, and autoregressive video samples, trim video frames, freeze LTXV video
  or audio samples, and normalize video latent starts under their exact ComfyUI IDs.
- Native Wan image and first/last-frame conditioning encodes image starts and
  ends, applies temporal masks and optional CLIP vision states, and supports
  reference padding and inpaint conditioning.
- Native Wan 2.2 image-to-video latents encode image starts, mask encoded temporal
  tokens, and normalize only the unfilled tokens before repeating image batches.
- Native Wan control conditioning builds Fun Control and Wan 2.2 control latents,
  VACE masked control streams and reference frames, and separate Phantom subject
  image/text guidance.
- Native Wan SCAIL conditioning combines per-identity reference masks, timed pose
  conditioning, and continuation frames with their latent noise masks.
- Native Wan track conditioning applies JSON point trajectories to image-start
  latents, preserving visible-point weights and per-batch image conditioning.
- Native Wan Animate conditioning combines reference, pose, face and continuation
  inputs. Animate 2 supports timed pose guidance and cleanup-scoped pose caching.
- Native Wan camera trajectories and image-start conditioning preserve camera
  embeddings and optional vision states. Wan Move generates and combines visible
  trajectories, conditions image starts, and renders colored tracks.
- Native Wan Sound, HuMo and InfiniteTalk conditioning use audio-encoder layers
  with image starts, continuation motion and single- or two-speaker controls.
- Native WanDancer audio encoding extracts onset, MFCC, chroma, peak and beat
  features with frame-rate metadata and matching prompt text. This CPU feature
  extraction does not establish full-model generation quality.
- Native WanDancer video conditioning combines optional image starts and masks,
  first-frame and reference vision states, and encoded audio features with an
  empty video latent. CPU tensor coverage does not certify model quality or GPU parity.
- Native SCAIL-2 colored masks render packed SAM3 tracks with identity selection,
  appearance/area ordering, palette controls and background masks. Source
  experimental presentation badges are not preserved.
- Native LTXV preprocessing applies the model's optional per-frame H.264 image
  compression before encoding video starts.
- Native LTXV image-to-video and frame-rate conditioning nodes encode image starts,
  replace existing latent starts, and attach frame-rate metadata.
- Native SeedVR2 preprocessing drops alpha, clamps image values, and pads video
  dimensions and frame counts for encoding.
- Native SeedVR2 conditioning builds positive and negative conditioning from a
  16-channel VAE latent, preserving its tensor dtype and device.
- Native Kandinsky 5 image-to-video conditioning encodes optional start images,
  preserving clean start latents separately from empty sampler latents.
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
- Reference file-input metadata includes manually enumerated file choices whose
  declared execution method directly resolves the input as an annotated path.
  Empty file lists retain this classification; unrelated empty combos do not.
- Native Save SVG writes ordered UTF-8 SVG streams to the selected output mount,
  retaining duplicate and empty streams, source batch counters, and submitted
  prompt/workflow metadata in source-compatible CDATA blocks. Invalid UTF-8 fails
  rather than being replaced. Saved files appear as host-validated run artifacts;
  the unchanged SVG value remains available downstream.
- Native animated PNG and WebP outputs retain frame order, alpha, timing,
  compression controls and source images while writing to the selected mount.
  Each saved animation is attached as a validated run artifact on every execution.
  SaveImage, SaveAnimatedPNG and SaveAnimatedWEBP retain submitted prompt and
  workflow metadata unless metadata is disabled. Animated PNG omits empty prompts.
- Native CLIP and VAE exports preserve tensor keys and dtypes in safetensors
  on the selected output mount. CLIP weights split by encoder prefix.
  Filenames retain the source trailing underscore before `.safetensors`.
  Submitted prompt and workflow metadata are retained unless metadata is disabled.
  Exported files are attached as validated run artifacts within the shared result limits.
- Native latent files load verified safetensors, convert samples to float32,
  and apply legacy scaling only when the format marker is absent. Saving
  writes contiguous samples and the format marker to the selected output mount.
  SaveLatent retains submitted prompt and workflow metadata unless disabled;
  an absent prompt remains an empty string, distinct from an empty prompt object.
  Saved latent filenames retain the source trailing underscore before `.latent`.
  Each saved latent is attached as a validated run artifact on every execution.
- Native image file schemas expose bound assets and adjacent output mounts;
  PNG output truncates channel values and preview outputs retain source images.
  PNG batches retain source counter progression and the trailing filename underscore.
  Every saved PNG in a SaveImage batch is attached as a validated run artifact,
  including batches larger than 64 images, within the shared result limits.
  File operations reject oversized image batches and encoded outputs.
  Image loading retains decoded frame precision, straight alpha and frame order;
  animated WebP uses a bounded Pillow fallback because FFmpeg cannot decode it.
- Native CheckpointSave and ModelSave export model weights through the selected
  output mount, retaining architecture and prediction metadata, auxiliary sampling
  tensors, and the reserved filename in supported model-spec titles. Submitted prompt
  and workflow fields are included unless disabled; architecture metadata remains.
  Submitted extra fields override matching metadata keys. Tiny CPU
  checks use controlled model state, not full-model or GPU execution.
  Saved checkpoints and models are attached as validated run artifacts within the
  shared result limits.
- Native SaveImageAdvanced writes 8/16-bit PNG and half/full-float EXR through
  the selected output mount, preserving source counters and straight alpha.
  EXR exports retain extended range and convert sRGB or HLG input to scene light.
  PNG, EXR and AVIF exports retain submitted prompt and workflow metadata unless
  metadata is disabled.
  Every published file, including each still image or complete AVIF animation,
  is attached as a validated run artifact within the shared result limits.
