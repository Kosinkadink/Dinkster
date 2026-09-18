## LoRA and model patching

- Native ModelPatchLoader for the official Z-Image Turbo Fun ControlNet Union
  and Wan 2.1 InfiniteTalk/MultiTalk patches, with ZImageFunControlnet workflow
  compatibility through KSampler, SamplerCustom, and SamplerCustomAdvanced for
  Z-Image
- Native Load LoRA and Load LoRA (Model Only) generation nodes use
  precalculated weights. `auto` selects precalculation; explicit `attach` is
  refused because canonical conditioning carries encoded values rather than
  source prompts.
- SD-era UNet LoRAs accept native, Kohya, Diffusers processor, and SimpleTuner
  LyCORIS key layouts.
- Native Load Checkpoint Stack and model-and-CLIP or model-only Apply LoRA
  Stack nodes apply up to 50 ordered LoRAs in one runtime materialization.
- Independently loaded Flux2 diffusion and text components support native
  model-and-text or model-only LoRA application through the same precalculated
  stack path.
- LazyCache provides opt-in approximate acceleration by reusing
  complete guided denoiser results during a configurable sampling range. It
  applies to KSampler and decomposed custom sampling, including classifier-free
  guidance and model-owned context windows. In a distributed job, rank 0 runs
  the cached sampling stage and broadcasts its final result to peers.
- EasyCache provides opt-in approximate acceleration by reusing
  model residuals for stable conditioning lanes during a configurable sampling
  range. It shares the KSampler and decomposed custom sampling path, works after
  split conditioning and context-window fusion, and uses rank-0 execution with
  final-result broadcast in distributed jobs.
- Context Windows (Manual), WAN Context Windows (Manual), and LTXV Context Windows
  model-patch nodes
  attach a context-windows configuration that samples the latent in
  overlapping fused windows with optional FreeNoise noise shuffling, keeping
  the matching ComfyUI node IDs and inputs. WAN and LTXV lengths use their
  matching real-frame units. LTXV can retain its first latent frame in every
  window while keeping text conditioning unretained.
  Wan 2.1 and Wan 2.2 base text-to-video profiles and LTX-Video text-to-video
  execute windowed sampling through both KSampler and the custom-sampling
  path; every other family and profile refuses an attached configuration.
  Latent retention is supported; the relative fuse method and window
  conditioning retention are not supported.
- Comfy workflow replacement covers ordinary loader subsets from KJNodes,
  rgthree, Easy Use, WAS Node Suite, pysssss Custom Scripts, and Efficiency
  Nodes. Custom dtype/config/VAE paths, prior stacks, overrides, populated
  prompts, and unsupported dynamic or opaque stacks are refused.
- Diffusers-format Z-Image LoRAs, including split Q/K/V adapters over native
  fused attention weights; auto mode precalculates adapters with offset targets
- PEFT-layout MiniMax H3 FL2VA and REF2VA DiT LoRAs through Load LoRA (Model
  Only). Native runtimes without scheduled patch resolution, including MiniMax
  H3, Wan, and split LTX-Video, warn and precalculate explicit attach requests.
- Native per-step strength curves for simple LoRA patches on owned linear and
  two-dimensional convolution weights through hook keyframes
- Conditioning-scoped regional LoRA execution with float32 masks, optional
  mask-derived bounds, strength and percent-range scheduling, overlap
  normalization, and default-region coverage
- FP8, NVFP4, and registered INT8 ConvRot model storage supports ordinary LoRA
  application. Per-step scheduling does not target FP8 hardware-matmul layers
  or packed INT8/NVFP4 modules.
- Adapter formats: LoRA, LoHa, LoKr, GLoRA, OFT, BOFT
- Patch algebra: adapters, additive diffs, set/replace, model-as-LoRA,
  nested patches, model-strength scaling, offset windows

LoRA payload/model dtype pairings for ordinary application and per-step
scheduling (`ordinary + scheduled` means both paths are supported):

| Model weight dtype | LoRA float16 | LoRA bfloat16 | LoRA float32 |
| --- | --- | --- | --- |
| float16 | ordinary + scheduled | ordinary + scheduled | ordinary + scheduled |
| bfloat16 | ordinary + scheduled | ordinary + scheduled | ordinary + scheduled |
| float32 | ordinary + scheduled | ordinary + scheduled | ordinary + scheduled |

LoRA factors are multiplied in float32. Plain and storage-converted weights
also accumulate there before deterministic storage-dtype writeback.
Cast-at-operation residency casts the float32 LoRA result into the selected
compute dtype, adds it to a transient compute-dtype weight, and does not write
back to checkpoint storage. Ordinary and scheduled application use the same
policy in each case.
