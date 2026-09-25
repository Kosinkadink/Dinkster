## Model families (native execution)

- Diffusion, text-encoder, and VAE loaders recognize registered component
  architectures from their assets. The text loader's type hint resolves shared
  geometry but does not reject a recognized component of another family.
  Shared text encoders retain their existing execution provider when the
  selected type does not resolve the native profile ambiguity.
  Component loaders can read their named tensor namespace from a combined
  checkpoint, including LTX-Video diffusion and VAE weights.
  Loading a component does not supply missing conditioning companions; LTX-2
  text conditioning still needs the checkpoint projections and connectors.
- Load Model exposes metadata-probed MODEL, CLIP, and VAE outputs for combined
  checkpoints, or MODEL for diffusion-only assets. Saved output profiles bind
  the asset identity and detector revision and are revalidated before reuse.
  Unrecognized metadata retains a MODEL default with a diagnostic; it does
  not claim that the selected artifact has an executable implementation.
- Native standalone CLIP loading and text encoding support explicit SD1.5
  and SDXL recipes, including both SDXL towers in one file. Reconstruction
  retains source identities, encoding options, and textual-inversion bindings.
  Standalone recipe conditioning is not supported by the sampler nodes.

<!-- capability:dinkster.sd15 -->
- Stable Diffusion 1.5, including the official standard four-token IP-Adapter
  with one reference image, scalar strength, inclusive sampling window, and
  optional output-token mask. Multiple standard adapters compose in workflow
  order through ordinary KSampler and decomposed custom sampling. Light, Plus,
  face, ViT-G, SDXL, multiple-reference, and per-layer weighting variants are
  not supported.
<!-- capability:dinkster.sdxl -->
<!-- capability:dinkster.sdxl_refiner -->
- SDXL base and refiner, including Illustrious-XL and Pony Diffusion V6 XL
  fine-tunes through generic checkpoint loading, CLIP hidden-layer selection,
  embedded or standalone AutoencoderKL decoding, and sampling
<!-- capability:dinkster.flux_dev -->
<!-- capability:dinkster.flux_schnell -->
- Flux (dev and schnell), including bounded unequal-length classifier-free
  guidance batching
<!-- capability:dinkster.chroma -->
<!-- capability:dinkster.chroma_radiance -->
- Chroma and Chroma Radiance, with strict independent diffusion, PixArt T5-XXL,
  and latent-component loading, combined checkpoint construction with optional
  text and codec companions, distilled guidance, and flow sampling through
  KSampler or custom sampling. Chroma uses the 16-channel Flux latent and KL
  VAE. Radiance uses a 3-channel RGB pixel latent with no VAE and preserves its
  convolutional patch embed, tiled linear or convolutional NeRF head, x0
  prediction marker, sequential text-position marker, and sigma-windowed runtime
  options.
<!-- capability:dinkster.wan21 -->
- Wan 2.1 1.3B and 14B text-to-video, 1.3B CausalAR blockwise text-to-video
  and first-frame image-to-video, 1.3B FlowRVS video-mask prediction,
  14B image-to-video and FLF first/last-frame checkpoints, 1.3B/14B Fun camera,
  1.3B Fun control/inpaint, 17B HuMo audio/reference image-to-video,
  14B InfiniteTalk single- or two-speaker audio-driven image-to-video and
  continuation,
  1.3B/14B Phantom subject-reference, 14B SCAIL/SCAIL2 character animation and
  replacement, and 1.3B/14B VACE checkpoints. Base 14B T2V and I2V support the
  Uni3C model patch with an RGB render video, strength, and sampling window.
  HuMo uses Whisper Large v3 audio features and optional reference-image
  conditioning. VACE supports T2V, V2V,
  reference-to-video, FLF, inpaint, outpaint, and restyle through control
  video, mask, and optional reference-image conditioning. Non-CausalAR profiles
  use positive/negative text conditioning, classifier-free guidance, and one
  plain 5D video latent through KSampler or decomposed custom sampling.
  CausalAR uses CFG 1, the AR Video sampler, and an invocation-owned temporal
  cache. Exact 14B T2V, 1.3B CausalAR, or 14B S2V diffusion, UMT5-XXL, and
  causal video VAE files can also load independently.
  This split path supports model-only LoRA application and decoding one sampled
  latent through separate RGB and alpha VAEs for the official Wan 2.1 alpha
  workflow
<!-- capability:dinkster.wan22 -->
- Wan 2.2 TI2V 5B text-to-video and first-frame image-to-video, 5B and 14B Fun
  control/inpaint, 14B Fun camera, 14B S2V sound/image-to-video and latent
  continuation, plus 14B high/low-noise text-to-video and image-to-video
  composition, 14B Bernini context-conditioned text-to-video, and 14B
  WanDancer music-driven image-to-video with local/global FPS paths. TI2V and
  5B Fun use 48-channel video latents; 14B I2V, Fun, S2V, Bernini, and
  WanDancer use Wan 2.1 latents. KSampler and decomposed custom sampling share
  the same flow engine.
<!-- capability:dinkster.z_image -->
<!-- capability:dinkster.z_image_pixel_space -->
- Z-Image Base and Turbo (latent, non-Omni), Zeta-Chroma pixel-space Z-Image,
  and the official Turbo Fun ControlNet Union model patch with image
  conditioning, typed timeline/site/guidance-lane gains at its six injection
  blocks, immutable control-latent binding, asset-proven model identity, and
  KSampler or custom sampling through the same shift-3 flow engine. Fun
  control chains and effect/source masks are refused.
<!-- capability:dinkster.lumina2 -->
- Lumina Image 2.0, including Neta Lumina, NetaYume, and compatible Lumina2
  fine-tunes. All-in-one checkpoints support Load Checkpoint with shared
  model, CLIP, and VAE outputs. Split files and all-in-one checkpoints also
  load through independent diffusion model, Lumina2 text encoder, and VAE loaders. Lumina2
  uses Gemma 2 2B prompt encoding, Flux AutoencoderKL latents, classifier-free
  guidance, and custom sampling with its default shift 6 or an explicit
  AuraFlow sampling shift. Its strict model identity remains distinct from
  Z-Image even though the families share transformer ancestry.
<!-- capability:dinkster.ideogram4 -->
- Ideogram 4 text-to-image with the official split FP8-scaled or INT8 ConvRot
  conditional and unconditional diffusion artifacts, Qwen3-VL-8B text encoder,
  and Flux2 VAE. Load Diffusion Model and Load CLIP compose through the shared
  custom-sampling engine and KSampler sugar. The official DualModelGuider
  image-only unconditional lane, CFGOverride, Euler sampling, and
  Ideogram4Scheduler are supported with bfloat16 or float32 diffusion compute.
  Both maintained ComfyUI workflows, `image_ideogram4_t2i.json` and
  `image_ideogram4_t2i_int8.json`, translate natively.
<!-- capability:dinkster.minimax_h3 -->
- MiniMax H3 audio/video generation: text-to-video+audio, first/last-frame,
  reference conditioning, arbitrary positive or end-relative image, clip, and
  audio timeline guides, continuation from a sampled AV latent with selectable
  overlap, ordinary Empty Latent Image adaptation to H3 video/audio streams,
  generic frame-range and fractional video/audio denoise masks, role-preserving
  latent mask composition, mask previews and diagnostics,
  ordinary KSampler sampling with classifier-free guidance through separate
  one-prompt conditioning nodes for positive and negative lanes, CFG++ sampling
  with real or synthetic unconditional prediction, custom sampling (SamplerCustom,
  SamplerCustomAdvanced, and the model-dependent sigma nodes) over the same
  video/audio streams, AV encode/decode, and MiniMax H3 Fun ControlNet Union
  v1/v2 control-video or masked-source conditioning with strength and sampling
  windows through the same sampling engine.
<!-- capability:dinkster.minimax_music3 -->
- MiniMax Music 3 text-to-music generation through the official split-component
  workflow. Diffusion supports the FP16, FP32, and INT8 ConvRot artifacts; the
  autoregressive text encoder supports full BF16, pruned BF16, and pruned INT8
  ConvRot artifacts. KSampler and decomposed custom sampling share the Euler
  flow path, and the FP32 DAV supports direct or tiled stereo audio decode.
<!-- capability:dinkster.qwen_image -->
- Qwen Image Base, Edit 2509/2511, Edit Plus, and Layered with independently
  loaded diffusion, Qwen2.5-VL, and Wan VAE components. The first-party pack
  exposes Edit, Layered, and typed maintained InstantX, Qwen Fun, and DiffSynth
  controls through universal sampling and codec nodes. Custom sampling
  (SamplerCustom, SamplerCustomAdvanced with BasicGuider or CFGGuider) also
  executes the composed components over exact caller-supplied sigmas, with the
  model-dependent sigma nodes served from Qwen Image's shift-1.15 flow space
  and control and DiffSynth applications materializing into the same
  decomposed path. ModelSamplingAuraFlow applies its explicit shift and
  unit-timestep flow space to complete and diffusion-only Qwen models, shared
  by schedule construction, KSampler, and custom sampling without changing
  the default space. CPU latent inputs are accepted by complete and component
  Qwen models running on CUDA through both KSampler and custom sampling.
<!-- capability:dinkster.trellis2 -->
- TRELLIS.2 and Pixal3D image-to-3D meshes with DINOv3 image conditioning,
  Pixal3D projected features, direct 512/1024 generation, 1024/1536 cascades,
  structure/shape/texture sampling, subdivision-guided decoding, and textured
  GLB/PLY output. Fused ComfyUI artifacts and Microsoft's five split flow and
  three split decoder artifacts use generic strict loaders, ordinary residency,
  and the shared KSampler or decomposed custom-sampling engine.
<!-- capability:dinkster.seedvr2 -->
- SeedVR2 3B and both 7B architectures for image and video restoration, with
  independently loaded diffusion and causal VAE components or combined
  checkpoints containing both. Native preprocess,
  conditioning, temporal chunk/merge, VAE tiling, LAB/wavelet/AdaIN color
  correction, KSampler sugar, and decomposed custom sampling preserve the
  official 4n+1 video and 16-channel latent semantics.

Not yet supported natively: SD2.x.
