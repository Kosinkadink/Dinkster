## Conditioning

- SD1.5 positive and negative text conditioning executes through the pinned
  `dinkster_inference` runtime.
- MiniMax H3 supports text-to-video conditioning through the pinned
  `dinkster_inference` runtime. First/last-frame, reference image, clip, and audio
  guide conditioning are unsupported.
- Typed absence remains a Dinkster engine contract: `core.absent` values,
  optional outputs, per-input absence policies, and skip propagation.
- Native ControlNet application supports both legacy and scheduled positive/negative
  conditioning, prior-control chaining, union type selection, and AliMama masked hints.
- Other model-family conditioning surfaces are unsupported.
