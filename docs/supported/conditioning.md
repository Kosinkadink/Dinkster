## Conditioning

- SD1.5 positive and negative text conditioning executes through the pinned
  `dinkster_comfy` runtime.
- MiniMax H3 supports T2VA, first/last-frame, and reference conditioning through
  the pinned `dinkster_comfy` runtime, including image, clip, and audio guides.
- Typed absence remains a Dinkster engine contract: `core.absent` values,
  optional outputs, per-input absence policies, and skip propagation.
- Other model-family conditioning surfaces are unsupported.
