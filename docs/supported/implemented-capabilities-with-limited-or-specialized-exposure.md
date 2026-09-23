## Implemented capabilities with limited or specialized exposure

These tested capabilities vary in exposure. Some have end-to-end loading and
sampling workflows, while others are limited to training APIs, specific node
compositions, optional workers, or compatibility boundaries described below.
Training capabilities are delivered by separately versioned packages.

- Experimental on-demand execution supports static, data-only nodes with scalars,
  images, masks, latents and recursive lists through a caller-supplied dispatch
  and object store. Only local execution is validated; no cloud provider or
  server routing is included. Resident handles, dynamic/lazy interfaces, events,
  artifacts and continuations are unsupported. Failed or cancelled dispatch
  does not prove remote execution stopped and is never retried automatically.
- Load Dual CLIP loads and encodes SDXL CLIP-L/CLIP-G and Flux CLIP-L/T5-XXL
  pairs, preserving both ordered assets, the requested recipe, and CPU device
  overrides through reconstruction. Unsupported or incompatible pairs are
  rejected rather than reduced to one encoder. Standalone recipe conditioning
  is not supported by the sampler nodes; other dual-encoder recipes are not
  supported natively.
- SD1.5 UNet LoRA training through the optional torch training worker backend:
  bfloat16 or float32 frozen-base storage, float32 adapter masters, gradients,
  and optimizer state, exports from the float32 masters, bfloat16
  forward/backward, AdamW or factored AdamW, accumulation, activation
  checkpointing, prepared batches or
  deterministic image folders with same-basename UTF-8 caption files and
  one-time native VAE/CLIP-L precomputation followed by encoder release,
  with an optional digest-verified disk cache for the encoded dataset,
  content-addressed optimizer/RNG/cursor checkpoints, safe-point cancellation,
  exact interrupted-run resume, and deterministic fp16, bf16, or fp32
  kohya-layout safetensors export through `training.export_lora`.
- SDXL base UNet LoRA training through the optional torch training worker
  backend: frozen bfloat16 or float32 base with float32 adapter and optimizer
  state, AdamW or factored AdamW, accumulation, activation checkpointing,
  fixed-resolution image/caption folders with sequential VAE, CLIP-L, and
  CLIP-G precomputation and release and the same optional encoded-dataset disk
  cache, content-addressed exact resume, safe-point cancellation, and
  deterministic kohya-layout safetensors export. Refiners,
  v-prediction variants, text-encoder training, and aspect-ratio buckets are
  refused.
- MiniMax H3 FL2VA or REF2VA DiT LoRA training through the optional torch
  training worker backend: frozen bfloat16, float32, or explicitly selected
  INT8 base with deterministic per-projection bfloat16 dequantization or an
  opt-in CUDA W8A8 forward with bounded bfloat16 input-gradient dequantization,
  float32 adapter and optimizer state, optional pinned-host paging for a
  configurable fraction of frozen transformer layers, fixed-schedule
  multistream velocity flow matching, AdamW or factored AdamW, accumulation,
  activation checkpointing,
  cursor-keyed prepared video/audio latent and conditioner batches, or FL2VA
  fixed-resolution media datasets with UTF-8 captions. Media items can use
  pre-decoded frame folders with PCM WAV audio or PyAV-decoded containers such
  as MP4 with embedded audio or a same-basename WAV override. Media training
  sequentially precomputes with the pinned video VAE, audio VAE, and Qwen3-VL
  conditioner and supports a digest-verified encoded cache. Training has
  content-addressed exact resume and safe-point
  cancellation. Committed checkpoints retain the adapter masters and support
  deterministic fp16, bf16, or fp32
  PEFT-layout safetensors export for native model-only application.
- Community-derived MiniMax Music 3 DiT LoRA training through the optional
  torch training worker backend: frozen native FP16 or FP32 bases, or the
  native INT8 ConvRot base with bfloat16 compute; float32 adapter, gradient,
  optimizer, objective, and loss state; logistic-normal flow matching; and
  activation checkpointing. Fixed-duration 44.1 kHz PCM audio datasets require
  same-basename UTF-8 caption and lyrics files and sequentially precompute with
  digest-pinned community DAV and RVQ encoders plus a native autoregressive text
  encoder. The encoded dataset cache is digest-verified, checkpoints resume
  exactly, and fp16, bf16, or fp32 PEFT-layout exports load through native
  MiniMax Music 3 inference. Dataset items are limited to 5.12 seconds, and
  text, DAV, and RVQ training are not supported.
- Wan 2.1 T2V and Wan 2.2 TI2V 5B, T2V 14B, or I2V 14B DiT LoRA training
  through the optional torch training worker backend. Video/caption datasets
  precompute and release the digest-pinned UMT5-XXL and matching Wan VAE before
  loading the DiT, with verified cache contracts that distinguish incompatible
  VAE latents and I2V first-frame conditioning. Wan 2.2 14B high-noise and
  low-noise experts train in separate sessions restricted to their configured
  timestep windows. Checkpoints support exact resume and deterministic Wan Fun
  kohya-layout fp16, bf16, or fp32 export; Wan 2.2 14B exports identify their
  expert for routing. Dataset bucketing and joint expert training are not
  supported.
- FLUX.1 dev and schnell DiT LoRA training through the optional torch training
  worker backend. Fixed-resolution image/caption datasets precompute and release
  the digest-pinned VAE, CLIP-L, and T5-XXL before loading the DiT, with a
  digest-verified encoded cache. Training uses each variant's flow schedule,
  includes distilled guidance for dev only, supports exact checkpoint resume,
  and exports deterministic PEFT-layout fp16, bf16, or fp32 adapters tagged with
  the FLUX.1 variant. Quantized bases, Kontext, Fill, Redux, and dataset bucketing
  are not supported.
- Flux2 dev and Klein 4B/9B DiT LoRA training through the optional torch training
  worker backend. Fixed-resolution image/caption datasets precompute and release
  the packed batch-norm KL VAE and the variant's Mistral3-Small or Qwen3 text
  encoder before loading the DiT, with a digest-verified encoded cache. Training
  uses the Flux2 flow schedule, includes distilled guidance for dev only, supports
  exact checkpoint resume, and exports deterministic PEFT-layout fp16, bf16, or
  fp32 adapters tagged with the Flux2 variant. Quantized bases, reference-image
  conditioning, inpaint, text-encoder training, and dataset bucketing are not
  supported.
- Base Qwen-Image DiT LoRA training through the optional torch training worker
  backend. Fixed-resolution image/caption datasets use Wan 2.1 single-frame
  latents and Qwen2.5-VL-7B prompt conditioning. Training uses the shift-1.15
  flow objective, supports exact checkpoint resume, and exports deterministic
  PEFT-layout fp16, bf16, or fp32 adapters. Edit, Layered, control training,
  text-encoder training, dataset bucketing, guidance, and quantized bases are
  not supported.
- Conditional and unconditional Ideogram 4 DiT LoRA training through the
  optional torch training worker backend. Fixed-resolution image datasets use
  the native Flux2 VAE; conditional training also uses the native Qwen3-VL-8B
  text encoder. FP8 and INT8 ConvRot bases retain their storage, training uses
  a rectified-flow objective, and role-bound adapters support exact resume and
  deterministic native fp16, bf16, or fp32 export. Text-encoder training,
  dataset bucketing, and custom diffusion checkpoints are not supported.
- Native SDXL diffusion loading from strict GGUF v3 Q8_0 checkpoints through
  the Python inference API, with CPU reference decode and sampling. Three
  residency modes, selected by default with `auto` (resolves to `balanced`
  when every quantized tensor uses an encoded-resident layout - Q4_0, Q8_0,
  Q4_K, Q5_K, or Q6_K - else `speed`): `speed` (tensors dequantize to
  float32 once at load), `memory` (opt-in; eligible linear weights stay
  encoded as raw quantized blocks in the checkpoint's layout at a fraction
  of the weight memory and dequantize on each forward with bit-identical
  outputs), and `balanced`
  (encoded residency plus a
  budgeted sticky cache of decoded weights, so layers within the byte budget
  skip the per-forward decode; the budget is an explicit byte count or auto,
  which admits cached weights only while the execution device keeps the
  inference working reserve free - concurrent auto caches across components
  and models share the device without overcommit; outputs stay bit-identical
  to `speed`). A module's decoded caches can enroll in the residency
  manager (`GgufDecodedCacheResidency`), so memory pressure evicts cached
  decoded weights and later forwards re-decode them. Encoded-resident
  modules also enroll in module residency (`enroll_component`/
  `enroll_assembled`) like any other layer: an offloaded unit leases its
  raw quantized blocks to the load device (asynchronously over the
  transfer stream) and decodes there, bit-identical to the fully loaded
  forward. Eager module-residency mechanisms also serve the block-loop
  prefetch queue, so models that walk their blocks through it (UNet,
  DiT, and text-encoder block loops) stream each offloaded unit's
  weights - GGUF encoded layers included - one block ahead of the
  consuming forward on the transfer stream, bit-identical to the
  unprefetched forward. GGUF encoded layers also run under aimdo-managed
  residency backends: raw quantized blocks stream through the aimdo
  transport (get, prefetch, and stored patch materialization) exactly
  like packed storage, bit-identical to the eager route. Partial-residency
  timing receipts (`collect_partial_residency_timing`) report transfer,
  dequantization, compute, and exposed stall time for eager and
  aimdo-managed module-residency leases, proving whether copy and decode
  work overlapped compute or stalled it, and split transferred bytes and
  move counts between mechanism-prefetched and lease-started copies.
- Native GGUF text-encoder loading through the Python inference API: T5-XXL
  (Flux, LTXV) and UMT5-XXL (Wan 2.1/2.2) from llama.cpp `t5`/`t5encoder`
  exports such as the city96 encoder GGUFs, in F32/F16/BF16 and
  Q4_0/Q4_K/Q5_K/Q6_K/Q8_0. All three residency modes apply to every one of
  those quantized layouts, selected by default with `auto` (`balanced` when
  every quantized tensor uses an encoded-resident layout, else `speed`):
  `speed` dequantizes to float32 at load and casts to the text dtype;
  `memory` (opt-in) keeps the encoder's projection Linears encoded as raw
  quantized blocks in the checkpoint's layout at a fraction of the weight
  memory and decodes on forward (bit-identical conditioning);
  `balanced` adds the budgeted decoded-weight cache over the same
  encoded residency. GGUF text sources carry
  no `spiece_model` tensor, so Wan assembly substitutes the vendored
  hash-verified UMT5 SentencePiece model (byte-identical to the tokenizer
  safetensors checkpoints embed). The Load CLIP node route stays
  safetensors-only, matching diffusion GGUF.
- Extended SD1.5 ControlNet runtime controls not exposed by the compat nodes:
  keyframed gains, independent residual-site and guidance-lane gains, and
  ordered effect masks
- TencentARC SD1.5 v2 full Canny T2I Adapter checkpoints, with ordered control
  chains, gains, windows, residual-site/lane gains, and masks.
- Stability AI rank-128 SDXL Canny Control-LoRA checkpoints assembled against
  their exact asset-identified SDXL base, with ordered chains, gains, windows,
  all nine input-skip and middle residual-site/lane gains, and effect masks.
- Classic SDXL ControlNet checkpoints in native, `control_model.`-prefixed,
  and Diffusers layouts, with detected hint channels, ordered chains, gains,
  windows, all nine input-skip and middle residual-site/lane gains, and effect
  masks.
- xinsir SDXL ControlNet Union 1.0 6-mode and ProMax 8-mode checkpoints, with
  an explicit semantic control mode, ordered chains, gains, windows, all nine
  input-skip and middle residual-site/lane gains, effect masks, and the
  authenticated `unet` attention route for every attention site.

  | Samplers | Constant gain | Non-uniform scheduled gain |
  | --- | --- | --- |
  | All except `dpm_fast` and `dpm_adaptive` | Supported | Supported |
  | `dpm_fast`, `dpm_adaptive` | Supported with a full application window | Refused |

  The constant-only samplers use internal evaluation timelines that do not map
  one-to-one to executed sigma rows.
<!-- capability:dinkster.anima -->
- Anima 2B (Cosmos Predict2 DiT with Qwen3-0.6B LLM adapter): strict header
  detection and per-component Python runtimes (Qwen3-0.6B text encoding with
  T5 token weighting, flow sampling at shift 3.0). Exact split diffusion,
  Qwen3-0.6B, and shared Wan VAE files load through Load Diffusion Model,
  Load CLIP (type `anima`), and Load VAE; CLIP Text Encode and KSampler
  compose the diffusion and text components, and VAE decode and encode ride
  the canonical Wan 2.1 VAE handle. Custom sampling (SamplerCustom,
  SamplerCustomAdvanced with BasicGuider or CFGGuider) also executes the
  split components over exact caller-supplied sigmas, with the
  model-dependent sigma nodes served from Anima's shift-3.0 flow
  space. `load_runtime` and Load Checkpoint also load combined diffusion
  and Qwen3-0.6B weights, retaining T5 token weighting through text encoding.
  The VAE remains a separate shared Wan component. KSampler normalizes empty
  image latents to the active family's channel count and rank, including
  Anima's 16-channel, single-frame latent shape. The official workflow's
  `stable_diffusion` CLIP type routes to Anima when the text asset has the exact
  Qwen3-0.6B layout. Qwen runs at the reference float32 compute dtype with
  native grouped-query attention. The official 1024x1024, 30-step ER-SDE/simple
  workflow executes end to end on CUDA with a bit-exact normalized sampled
  latent against pinned ComfyUI; cheap and quality previews decode that latent.
<!-- capability:dinkster.krea2 -->
- Krea 2 (SingleStreamDiT with Qwen3-VL-4B text tower): strict header
  detection and per-component Python runtimes (Qwen3-VL-4B stacked tap-layer
  text encoding, flow sampling at shift 1.15). Exact split diffusion and
  Qwen3-VL-4B files load through Load Diffusion Model and Load CLIP (type
  `krea2`); CLIP Text Encode and KSampler compose the diffusion and text
  components, and VAE decode and encode ride the canonical shared Wan 2.1
  VAE handle through Load VAE. Custom sampling (SamplerCustom,
  SamplerCustomAdvanced with BasicGuider or CFGGuider) also executes the
  split components over exact caller-supplied sigmas, with the
  model-dependent sigma nodes served from Krea 2's shift-1.15 flow
  space. `load_runtime` and Load Checkpoint also load combined diffusion
  and Qwen3-VL-4B weights; the VAE remains a separate shared Wan component.
  KSampler normalizes empty
  image latents to Krea 2's 16-channel, single-frame latent shape. The
  `stable_diffusion` CLIP type also routes to Krea 2 when the text asset has
  the exact Qwen3-VL-4B layout, and previews use the shared Wan latent
  providers (cheap latent2rgb and quality `lighttaew2_1`). The diffusion
  component defaults to bfloat16 compute; the text tower defaults to float32,
  matching the reference's executed text arithmetic. The official 1024x1024
  euler/simple workflows execute end to end on CUDA for both checkpoints -
  Turbo (8 steps, CFG disabled) and RAW (52 steps, CFG 3.5) - with bit-exact
  denormalized sampled latents against pinned ComfyUI; cheap and quality
  previews decode the sampled latent.
<!-- capability:dinkster.flux2_dev -->
<!-- capability:dinkster.flux2_klein_4b -->
<!-- capability:dinkster.flux2_klein_9b -->
- Flux2 dev and Klein 4B/9B: strict header detection, native assembly, and a
  Python runtime (Mistral3-Small 24B or Qwen3 stacked text encoding
  with the reference 512-row context floor, flow sampling at shift 2.02 with a
  per-run shift override for empirical-mu schedules, packed batch-norm KL VAE
  decode and encode) are wired through `load_runtime`. Set Reference Latent
  adds one or more ordered Flux2 image-edit latents to conditioning; KSampler
  and custom sampling consume them through the same engine. Inpaint is
  refused. The split diffusion,
  text-encoder, and VAE components load independently through Load Diffusion
  Model, Load CLIP (type `flux2`), and Load VAE; CLIP Text Encode, KSampler,
  and VAE Decode/Encode execute those independently loaded components.
  Custom sampling (SamplerCustom, SamplerCustomAdvanced with BasicGuider or
  CFGGuider) also executes them, alongside EmptyFlux2LatentImage,
  Flux2Scheduler (the reference empirical-mu schedule), and FluxGuidance
  (accepted by guidance-distilled Flux and Flux2 models; refused by models
  without a guidance input). Flux Disable Guidance explicitly omits the
  distilled-guidance embed. Both official ComfyUI Flux2 text-to-image workflows
  (dev and Klein 9B) port over.
- MODEL guidance transforms are available through CFGZeroStar, CFGNorm,
  Tangential Damping CFG, FreSca, Adaptive Projected Guidance,
  Positive-Biased Guidance, Epsilon Scaling, RescaleCFG, RenormCFG, and
  TSR - Temporal Score Rescaling.
- Normalized Attention Guidance (NAGuidance) attaches as a MODEL guidance
  transform and rewrites self-attention outputs inside the fused
  conditional/unconditional forward. It executes on SD1.5 and SDXL through
  KSampler and custom sampling alike; other families, distributed sampling,
  and runs whose lanes cannot share one fused forward refuse it.
- Structurally compatible MiniMax H3 FL2VA and REF2VA DiTs, including
  finetunes, can be loaded independently as MODEL handles through Load
  Diffusion Components with an explicit role. Compatible conditioner and
  video/audio VAE finetunes can likewise be loaded through Load CLIP and Load
  VAE. Generic H3 composition nodes execute these independently loaded
  components. Setting `DINKSTER_ATTENTION_POLICY=dinkster_kitchen_int8`
  on the server makes dinkster-kitchen INT8 the default for each job on capable
  workers. Job submissions can override that default globally or for individual
  model roles. Built-in SDPA serves causal and grouped-query invocations the INT8
  kernel cannot execute. Dinkster-kitchen INT8 remains explicit-only.
- Named attention policies use SDPA with a diagnostic when their provider,
  kernel, or hardware support is unavailable. Authenticated routes preserve
  the requested policy and record the actual SDPA execution, including
  per-role overrides and isolated or remote workers.
- SageAttention 2 INT8 attention executes through the `sage` attention policy
  (server default `DINKSTER_ATTENTION_POLICY=sage` or per-job/per-role override)
  when the managed `dinkster-kitchen` distribution is installed on a CUDA worker
  whose SM has an upstream kernel arm (80, 86, 89, 90, or 120). Prebuilt
  Python 3.12 wheels support CUDA 13.0 on Windows and Linux x86_64 without a
  compiler; private-repository users can fetch and install the matching pinned
  wheel automatically with GitHub authentication. Built-in SDPA serves masked,
  non-fp16/bf16, wider-than-128-head-dim, and causal cross-length invocations
  the quantized kernels cannot execute. Automatic routing selects SageAttention
  when this support is authenticated.
- Approximate, training-free Sol sparse attention executes through the opt-in
  `sol` policy (server default `DINKSTER_ATTENTION_POLICY=sol` or per-job/per-role
  override) with dinkster-kitchen 0.2.35.post1 on NVIDIA SM80+ workers. It serves
  unmasked, noncausal BF16 self-attention with equal q/k/v shapes and 128-wide
  heads, trading output similarity and temporary workspace memory for speed;
  built-in SDPA preserves exact behavior for all other calls. Automatic routing
  never selects Sol.
- Automatic VAE attention uses built-in SDPA with a bounded-memory fallback on
  recognized out-of-memory failures. ROCm workers select bounded-memory VAE
  attention directly. Other exceptions propagate without fallback, and explicit
  attention policies retain their selected behavior.
- Attention Schedule switches between the model's authenticated Sol, Sage, or
  dinkster-kitchen INT8 provider and SDPA on one discrete sampling-step window.
  An optional native `dinkster.curve` drives Sol tau on that same realized
  timeline. MiniMax H3 Sol schedules may keep the packed conditioning prefix
  exact as KV.
  KSampler and decomposed custom sampling install one immutable row before each
  outer step, shared by conditioning splits, context-window tiles, control gain
  callbacks, and conditioning/LoRA percent ranges. Distributed execution refuses
  scheduled timelines until every rank can authenticate the same timeline digest
  and row.
- Wan 2.2 Animate 14B: strict checkpoint detection and assembly, plus Python
  runtime execution with reference latents, pose-video latents, face pixels,
  optional CLIP vision conditioning, and classifier-free guidance batching.
  The native model pack executes its conditioning node through distinct public
  model and codec resources, including optional reference, pose, face,
  background, mask, and continuation media. Its rich canonical conditioning
  executes through the universal KSampler and KSampler Advanced providers. No
  bundled workflow is published.
- Wan 2.1 Animate2 14B: strict checkpoint detection and assembly, lockstep pose
  branch execution, reference and pose CLIP vision, independent pose text,
  pose-window and pose/reference strengths, one-frame continuation, and
  execution-scoped CPU or GPU pose-block caching with default, INT8, or INT4
  storage. The native model pack executes the canonical two-record
  conditioning provider; its rich canonical conditioning executes through the
  universal KSampler and KSampler Advanced providers.
- Flux packed-grid spatial windows execute through the KSampler facade and
  decomposed custom-sampling runtime; workflow and server carriage are not
  exposed. On a single-job
  multi-GPU group, the registered Flux Dev BF16 plan scatters each step's
  window evaluations across 2 or 4 Blackwell ranks with bit-exact merges
  and delivers the serial path's per-step progress events; sampling state
  callbacks stay refused in distributed modes (circumstantial acceleration
  of windowed plans; unproven dtype, world size, or capability combinations
  refuse)
- Regional/grouped conditioning channels beyond text/pooled - not reachable
  from nodes
- Collaboration service (sessions, operations, snapshots, session
  WebSockets) - mountable but not mounted by `create_app`
- Ovis/Qwen3-2B text encoder path for Flux Schnell - server source roles
  do not expose it
<!-- capability:dinkster.ltxv -->
<!-- capability:dinkster.ltxav -->
- LTX-Video 2B v0.9 and v0.9.5 text-to-video through the Python inference
  API: independent native loading of diffusion, T5-XXL, and causal-VAE
  components, classic-T5 text encoding masked through EOS
  without zeroing padding, flow sampling with the fixed or an explicit
  sampling shift, classifier-free guidance, and chunked causal VAE
  encode/decode (the VAE streams its own chunks; tiled execution is
  refused). Custom sampling over exact caller-supplied sigmas is
  available through the Python inference API, with the shared scheduler,
  beta, sd-turbo, and percent-to-sigma surfaces served from the LTX flow
  space and single or dual classifier-free guidance; KSampler-style
  sampling produces bit-identical results. LTX-Video Image to Video, LTX-Video
  Image to Video (In-place), LTX-Video Add Guide, and LTX-Video Crop Guides
  expose initial-frame replacement, ordered initial/final guides, per-guide
  strength and spatial attention masks, guide cropping, and sampler masks that
  drive per-token model timesteps through the same custom-sampling engine. The
  matching ComfyUI LTXV node IDs are accepted as aliases.
  The components load through Load Diffusion Model, Load CLIP type `ltxv`, and
  Load VAE. Load Checkpoint also loads the detected diffusion, T5, and VAE
  components from a combined checkpoint, without requiring unused companions.
  Reconstruction preserves the component key maps and compute dtypes. CLIP Text
  Encode, LTX-Video Conditioning (sets the frame rate the sampler consumes,
  default 25), Empty LTX-Video Latent, KSampler, and VAE Decode execute;
  the decomposed custom-sampling nodes sample the same video
  latent stream. ModelSamplingLTXV optionally derives the reference's dynamic
  sampling shift from that latent's video token count; workflows without it
  retain the family's fixed sampling shift. The
  LTX-2 19B, LTX-2.3 22B, and LTX-2.5 22B audio-video profiles run
  text-to-audio-video through the Python inference API: native loading
  of independent diffusion, Gemma 3 or Gemma 4 12B, text-projection, video-VAE, and
  audio-codec components with strict tensor
  routing and fail-closed checkpoint-metadata gates, joint flow sampling
  of the video and audio latent streams with classifier-free guidance and
  a fixed or explicit sampling shift, chunked causal video decode, and
  audio decode through the audio VAE and vocoder (causal audio
  autoencoder, per-channel latent statistics, a plain-torch mel/resample
  front end bit-equal to the reference's torchaudio path, and the
  HiFi-GAN/BigVGAN vocoder including bandwidth extension, all pinned to
  executed-reference goldens). Python media conditioning supports sampler
  masks for both latent streams and one content-owned reference-audio input.
  Standalone artifacts containing the paired audio VAE and vocoder load
  through Load LTX-2 Audio VAE in fixed float32, and LTX-2 Reference Audio
  encodes a ComfyUI AUDIO value and attaches it to positive and negative conditioning.
  LTXV Reference Audio (ID-LoRA) exposes the compatible model-patching surface:
  within its selected sampling range, it adds one no-reference conditional
  evaluation and applies identity guidance through the same custom-sampling
  engine. It composes with standard CFG and post-CFG transforms; pre-CFG
  transforms and transform-bypassing guidance strategies refuse.
  All text-to-audio-video profiles execute through nodes: Load Diffusion Model
  loads the diffusion component, Load LTX-2 Text Encoder composes the selected
  Gemma asset with the checkpoint's text components, Load VAE loads the video
  VAE, and Load LTX-2 Audio VAE loads the audio codec. CLIP Text Encode, LTX-2 AV
  Conditioning (sets the frame rate the sampler consumes, default 25),
  Empty LTX-2 AV Latent, KSampler, Concat/Separate AV Latent, VAE
  Decode (video stream), and LTX-2 Audio VAE Decode execute through the
  corresponding loaded component. Load Latent Upscale Model and LTXV Latent
  Upsampler provide the official x2 spatial latent upscaler used by the
  two-stage LTX-2.3 and LTX-2.5 workflows, including video-VAE
  unnormalization and renormalization. VAE Decode (Tiled) executes the
  official LTX video-VAE tile geometry used by those workflows. Custom sampling (SamplerCustom,
  SamplerCustomAdvanced with BasicGuider or CFGGuider, and the model-dependent sigma nodes
  served from the LTX flow space) executes the same joint audio-video
  streams over exact caller-supplied sigmas; dual classifier-free
  guidance is available through the Python inference API and nodes. LTX-2.5
  also supports the neighborhood-attention diffusion video VAE, prompt duration
  prediction, spatio-temporal guidance, and audio-video modality guidance.
  Load Checkpoint composes detected LTX-2 diffusion, text-projection,
  connector, and video-VAE components from an all-in-one checkpoint.
  Load CLIP type `ltxv` accepts ordered standalone Gemma and projection
  assets for LTX-2: Gemma 3 with a single projection and checkpoint
  connectors, Gemma 3 with a dual projection, or Gemma 4 with its matching
  dual projection. A Gemma encoder alone refuses and points at Load LTX-2
  Text Encoder; image-to-video is not exposed.
  The LTX-2 text encoders execute through the Python inference API:
  checkpoint SentencePiece or tokenizer-JSON tokenization with the LTX prompt
  policy, the text-only Gemma tower returning the all-layer hidden-state stack, and
  both the single (3840) and dual video+audio (6144) text-embedding
  projections; 19B checkpoints that carry the video and audio
  text-embedding connector towers (the ltx-2-19b-dev format) refine the
  single projection through both towers into concatenated video+audio
  embeddings (7680). LTX-2.3 uses the dual video+audio projection (6144)
  with its gated connector towers and bandwidth-extended 48 kHz audio
  output. LTX-2.5 uses Gemma 4 12B with the official dual projection;
  text only (no image tokens)
<!-- capability:dinkster.triposplat -->
- TripoSplat image-to-3D gaussian splats: detection, catalog registration,
  and source modules pinned to executed-reference goldens. Torch-free
  exact-header detection of the published DiT (bare or
  `model.diffusion_model.`-prefixed), the octree gaussian decoder, and the
  DINOv3 ViT-H/16+ vision conditioner, plus flow sampling constants at
  shift 3.0 and the latent+camera token-stream geometry. The torch modules
  cover the flow denoiser over explicit latent and camera streams, the
  octree probability decoder with seeded systematic-resampling descent, the
  elastic gaussian decoder activating into render-ready splat tensors, and
  the DINOv3 encoder with its reference preprocessing; every random draw
  goes through the caller's generator. Component planning and loading run
  through the shared assembly pipeline (per-role plans for the DiT, the
  DINOv3 vision conditioner, and the gaussian decoder, with asset identity
  facts and per-role dtype identity slots), and the sampling runtime
  performs joint FLOW denoising of the latent and camera streams with
  single-guidance CFG, carrier-encoded DINOv3 features plus the optional
  reference latent, and reference-parity nested-stream noise. Loader wiring
  runs through the native and generation KSampler arms, the decomposed
  custom-sampling seam (noise/guider/sampler/sigmas nodes) executes the
  family with optional negative conditioning and the shared scheduler
  surfaces, and the
  dinkster-model-triposplat pack exposes the vision-encoder and gaussian-decoder
  loaders, the reference image preprocessor, the DINOv3+reference-latent
  conditioning node, and the splat decode node
