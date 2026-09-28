## Conditioning

- SD1.5 positive and negative text conditioning executes through the pinned
  `dinkster_comfy` runtime.
- MiniMax H3 supports text-to-video conditioning through the pinned
  `dinkster_comfy` runtime. First/last-frame, reference image, clip, and audio
  guide conditioning are unsupported.
- MiniMax H3 Fun ControlNet model patches support control video and masked
  source-video inpainting through the same runtime.
- Typed absence remains a Dinkster engine contract: `core.absent` values,
  optional outputs, per-input absence policies, and skip propagation.
- Other model-family conditioning surfaces are unsupported.
