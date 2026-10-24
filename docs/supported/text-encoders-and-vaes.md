## Text encoders and VAEs

- Native VAEEncode, VAEDecode and their tiled variants preserve the core
  interfaces, including image/video batches and spatial/temporal tile settings.
- SD1.5 CLIP-L and image VAE components execute through the pinned
  `dinkster_inference` runtime.
- Flux conditioning supports CLIP-L and T5 text-encoder pairs.
- MiniMax H3's Qwen3-VL conditioner, video VAE, and audio VAE execute through
  the pinned `dinkster_inference` runtime.
- Other model-family text encoders, audio encoders, and VAEs are unsupported.
