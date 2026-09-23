## Conditioning

- Positive/negative text conditioning for all wired image families
- Native conditioning strength and sampling-range scheduling
- Native conditioning utility nodes: merge (ordered combine, weighted
  average, and token concat modes; combine accepts up to eight inputs),
  scalar multiply, rectangular area restriction in pixels or image fractions,
  and ComfyUI spatiotemporal video-area metadata preservation,
  mask restriction with strength and mask-bounds option, inclusive
  timestep-range windows, and zero-out. The legacy ComfyUI class names
  (ConditioningCombine, ConditioningAverage, ConditioningConcat,
  ConditioningMultiply, ConditioningSetArea, ConditioningSetAreaPercentage,
  ConditioningSetAreaPercentageVideo, ConditioningSetMask,
  ConditioningSetTimestepRange, ConditioningZeroOut)
  translate to them. Grouped aliases also translate KJNodes Conditioning Multi
  Combine (2-20 inputs and count output), Conditioning Set Mask And Combine
  (2-5 positive/negative lanes), ComfyUI Essentials Conditioning Combine
  Multiple, and SD3 Negative Conditioning workflows
- CLIP hidden-layer selection for SD1.5, SDXL, and Flux text encoding, and
  T5/UMT5 minimum-padding and minimum-length overrides for Flux, LTX-Video,
  and Wan text encoding. ControlNet-specific text encoding stamps separate
  cross-attention and pooled payload references without replacing the target
  conditioning; native ControlNet sampling does not yet consume those stamps.
- Qwen, Mistral3, Qwen Image, Krea 2, Ideogram 4, and MiniMax H3 text towers refuse
  sequences beyond their model position limit; Qwen Image counts visual rows
  after placeholder expansion. CLIP encodes independent 77-token chunks.
  T5/UMT5 relative positions and Gemma's local sliding window are not treated
  as fixed total prompt limits.
- Wan 2.1 I2V CLIP-vision and masked first-frame or FLF first/last-frame
  conditioning
- Wan 2.1 and 2.2 camera trajectories with optional first-frame conditioning;
  Wan 2.1 camera also supports optional CLIP-vision conditioning
- Wan 2.1 Phantom single- and multi-subject temporal reference conditioning
  with regular or nested three-lane Dual CFG sampling
- Wan 2.1 ATI point-track motion with first-frame and optional CLIP-vision
  conditioning
- Wan 2.1 WanMove track construction, concatenation, visualization, and
  first-frame motion conditioning with optional CLIP vision
- Wan 2.1 Uni3C RGB-render control for exact base 14B T2V and I2V models,
  with target-sized VAE encoding and strength/window controls
- Wan 2.1 HuMo audio and optional reference-image conditioning, including
  exact 25 Hz grouped Whisper windows and zero-audio/reference operation
- Wan 2.1 InfiniteTalk audio-driven image-to-video and continuation with
  motion overlap, one- or two-speaker audio, and optional speaker masks
- Wan 2.2 Bernini source-video and independently sized reference-video/image
  context streams, including ordered multi-image inputs
- Wan 2.2 S2V audio, optional reference image, control video, and 19-latent
  motion context, plus continuation from an existing video latent
- Wan 2.2 WanDancer start/reference CLIP vision, optional music features,
  masked start frames, FPS selection, and audio injection strength
- Wan 2.1 SCAIL/SCAIL2 reference and pose-video conditioning, including
  multi-reference identity masks, animation or replacement layout, pose
  strength/window controls, and SCAIL2 continuation anchors. Colored mask
  images are accepted and can be generated from native segmentation masks.
- Wan Fun control video, optional start/reference image, and first/last-frame
  inpaint conditioning for the official Wan 2.1 and 2.2 profiles
- Wan 2.1 VACE control video, mask, and optional reference-image conditioning
- Wan 2.2 TI2V and 14B I2V masked first-frame or first/last-frame conditioning
- MiniMax H3 conditioning: T2VA, first/last-frame, and reference requests,
  each with an optional negative prompt that prepares an unconditional lane;
  image, clip, and audio guides can be chained at arbitrary target frames, and
  sampled AV clips can continue with a selectable aligned overlap
- YuE2 style and lyrics conditioning with generated or edited full-score or
  melody-only ABC notation, optional autoregressive guidance, and duration-
  matched acoustic prefix conditioning
- SeedVR2 positive and negative conditioning from the encoded restoration
  source, bound to the exact independently loaded diffusion component
- Typed absence: `core.absent` values, optional outputs, per-input absence
  policies (skip/accept/fail/omit), engine-level skip propagation
