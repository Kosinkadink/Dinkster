## Conditioning

- SD1.5 positive and negative text conditioning executes through the pinned
  `dinkster_inference` runtime.
- `CLIPTextEncodeSDXL` encodes separate global and local prompts with size,
  crop and target-size conditioning. Unequal token streams are padded with
  empty-prompt tokens. `CLIPTextEncodeSDXLRefiner` adds aesthetic-score and
  size conditioning. Both preserve scheduled CLIP outputs.
- MiniMax H3 supports text-to-video conditioning through the pinned
  `dinkster_inference` runtime. First/last-frame, reference image, clip, and audio
  guide conditioning are unsupported.
- Stable Zero123 and Stable Video 3D image conditioning execute through the pinned
  `dinkster_inference` runtime.
- Stable Video image conditioning attaches motion, frame rate, augmentation, and
  concatenated image latents to positive and negative prompt streams.
- Cosmos image conditioning encodes optional first and last images into latent
  video frames for Cosmos and Cosmos Predict2 workflows.
- Hunyuan Image text conditioning combines independent BERT and MT5 prompts.
  Hunyuan Video image-to-video text conditioning uses projected vision embeddings
  with the video prompt template and configurable image interleave.
- Hunyuan Image and Hunyuan Video 1.5 image-to-video conditioning prepare native
  image latents, masks, and optional vision metadata for their model variants.
- Native Stable Audio conditioning attaches start and total seconds to both prompt
  streams. Video percentage-area conditioning preserves temporal and spatial bounds.
- Typed absence remains a Dinkster engine contract: `core.absent` values,
  optional outputs, per-input absence policies, and skip propagation.
- Native ControlNet application supports both legacy and scheduled positive/negative
  conditioning, prior-control chaining, union type selection, and AliMama masked hints.
- Native `InstructPixToPixConditioning` crops and encodes the input image,
  attaches image conditioning to both prompt streams, and returns a zero latent.
- Native `ReferenceLatent` appends a guiding latent to conditioning while
  preserving existing references; an unconnected latent leaves conditioning unchanged.
- Native `ReferenceMultiLatent` encodes ordered reference images, appends them
  to positive and optional negative conditioning, and returns a matching empty latent.
- `MiniMaxMusic3TextEncode` encodes captions and lyrics with duration, seed, and
  guidance controls, returning acoustic conditioning and its duration.
- `ReferenceTimbreAudio` appends audio timbre latents to conditioning; an
  unconnected latent leaves conditioning unchanged.
- `TextEncodeAceStepAudio` and `TextEncodeAceStepAudio1.5` encode ACE Step
  audio tags and lyrics with their native generation controls.
- `CLIPTextEncodeKandinsky5` combines its CLIP-L and Qwen prompt streams for
  Kandinsky 5 conditioning.
- Hook-bearing native conditioning retains its runtime hook groups across node
  links. Releasing a referenced hook invalidates dependent conditioning and caches.
- Nine native hook conditioning nodes set and combine single or paired prompt
  properties, default prompt records, hook timesteps, and CLIP hook schedules.
- Other model-family conditioning surfaces are unsupported.
