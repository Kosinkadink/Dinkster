## Text encoders and VAEs

- SD1.5 CLIP-L and image VAE components execute through the pinned
  `dinkster_inference` runtime.
- Wan UMT5 text encoders support native Q8_0 GGUF loading with minimal-VRAM,
  budgeted decoded-weight, and eager residency choices.
- MiniMax H3's Qwen3-VL conditioner, video VAE, and audio VAE execute through
  the pinned `dinkster_inference` runtime.
- Other model-family text encoders, audio encoders, and VAEs are unsupported.
