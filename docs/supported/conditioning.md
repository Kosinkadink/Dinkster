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
- Typed absence remains a Dinkster engine contract: `core.absent` values,
  optional outputs, per-input absence policies, and skip propagation.
- Native ControlNet application supports both legacy and scheduled positive/negative
  conditioning, prior-control chaining, union type selection, and AliMama masked hints.
- Native `InstructPixToPixConditioning` crops and encodes the input image,
  attaches image conditioning to both prompt streams, and returns a zero latent.
- Native `ReferenceLatent` appends a guiding latent to conditioning while
  preserving existing references; an unconnected latent leaves conditioning unchanged.
- Hunyuan 3D v2 uses single-view vision features or ordered front/left/back/right
  positional features for positive conditioning and matching zero negative conditioning.
- Other model-family conditioning surfaces are unsupported.
