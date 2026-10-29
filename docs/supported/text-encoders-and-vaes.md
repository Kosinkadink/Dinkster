## Text encoders and VAEs

- Text encoders: CLIP-L (SD1.5/SDXL/Flux), CLIP-G (SDXL/refiner),
  T5-XXL (Flux, PixArt/Chroma, and LTX-Video), UMT5-XXL (Wan 2.1/2.2),
  Llama3 with raw CLIP-L pooled output (original Hunyuan Video), CLIP ViT-H
  vision (Wan 2.1), Qwen3-4B (Z-Image), Gemma 2 2B (Lumina2), MiniMax Music
  3's Qwen autoregressive music encoder, Mistral3-Small 24B full or pruned
  (Flux2 dev), Qwen3 4B/8B (Flux2 Klein), and Qwen3-VL-8B (Ideogram 4)
  stacked-layer encoders, including the official mixed FP8/NVFP4 and
  FP8/NVFP4 and FP4/NVFP4 files; and Hunyuan Image's Qwen2.5-VL-7B
  hidden-layer and optional ByT5-small glyph conditioning; NewBie's Gemma 3
  4B sequence conditioning with Jina CLIP v2 masked pooled output; and
  ACE-Step 1.5's Qwen3 0.6B structured conditioner with a Qwen3 2B or 4B
  audio-code language model
- Audio encoders: Wav2Vec2-large with 16 kHz resampling and all 25 hidden-layer
  outputs for Wan 2.2 S2V; Wav2Vec2 Chinese base with all 13 hidden-layer
  outputs for Wan 2.1 InfiniteTalk; Whisper Large v3 with exact 30-second
  feature extraction and all 33 hidden-layer outputs for Wan 2.1 HuMo
- VAEs and pixel codecs: AutoencoderKL for latent image families, including
  Chroma's Flux VAE, Lumina2's Flux-family 16-channel latent format, and the
  Flux2 packed-latent (batch-norm) variant with 128 external latent channels at
  16x downscale; RGB image `[0, 1]` to pixel-latent `[-1, 1]` conversion for
  Chroma Radiance, and an identity pixel codec for Zeta-Chroma;
  TAESD/TAESDXL preview decoders for the SD family; MiniMax H3 FP16 or INT8
  ConvRot video VAE and FP32 audio VAE; MiniMax Music 3's FP32 decode-only
  stereo DAV; causal Wan 2.1 RGB and FlowRVS mask VAEs; the 16x-spatial Wan 2.2
  video VAE; the classic LTX-Video causal VAE; and the SeedVR2 8x-spatial,
  4x-temporal causal VAE
