# dinkster-inference

`dinkster-inference` is the torch-free half of the native inference
program (docs/native-inference-plan.md): the typed contracts that
later stages implement and that nodes/engine code consumes (stage 1),
native checkpoint inspection and evidence-based family detection
behind those contracts (stage 2), the sampling math - sigma spaces,
the nine sigma schedules, prediction parameterizations, CFG - as pure
functions (stage 3a), the solver/scheduler catalog: 13 k_diffusion
solver ports with typed option schemas behind `SamplerDescriptor`,
plus `SchedulerDescriptor`s for the nine schedules, both in
namespaced registries with legacy-ComfyUI-name aliases (stage 3b),
and the torch-free parts of stage 5's native components: tile/codec
planning, AutoencoderKL detection, SD1/SDXL CLIP tokenization +
prompt weighting, CLIP text-model configuration/detection, and the
diffusion-core configuration/layout/detection for the SD1/SDXL UNet
the classic Flux dev/schnell transformer, and Chroma/Chroma Radiance.
It contains
protocols, frozen data models, header-only file
inspection, and pure float/tensor-arithmetic math - no torch, no
execution, no global state. The structural demands placed on tensors
are `SizedTensor` (has a `.shape`) and `ArithTensor` (supports
elementwise `+ - * /`), which `torch.Tensor` satisfies without this
package importing torch.

Every contract that reimplements or replaces a ComfyUI mechanism cites
its source module and the audited baseline commit (`b78cec87`) in its
docstring, per the upstream-tracking promise.

## Scheduling activation

Graph compiler declarations have deterministic ordering and identity across
isolated worker generations. Native runtime call sites consume scheduled text,
conditioning ranges, regional metadata, and scheduled patch stacks through the
canonical conditioning carrier. Compat packs can expose authoring nodes while
execution remains on the native arm; patch digests are resolved to immutable
worker-local patch sets before sampling.

## Setup

This package is a uv workspace member. From the repository root:

```sh
uv sync --all-packages
```

Dependencies: `dinkster-schema` only (for the closed name grammar).
Deliberately NOT a dependency: torch, numpy, comfy. Mapping a `DType`
to a framework dtype is the executing backend's job (stage 4+).

## Modules

| Module | Contracts | Replaces (ComfyUI @ b78cec87) |
|--------|-----------|-------------------------------|
| `devices` | `DType`, `DeviceRef`, `DeviceCapabilities`, `PrecisionRequest`, `PrecisionPlan` | comfy/model_management.py global probing + ad hoc dtype policy functions |
| `weights` | `TensorGeometry`, `WeightEntry`, `WeightSource`, `filter_prefix`/`replace_prefix`/`count_prefix` | comfy/utils.py state-dict helpers; whole-dict loading in comfy/sd.py |
| `patches` | `DiffPatch`/`SetPatch`/`ModelAsLoraPatch`/`AdapterPatch`/`NestedPatch`, `PatchEntry`, `PatchOffset`, `PatchSet`, `calculate_shape`, `WeightAdapter` | comfy/model_patcher.py positional-tuple patches; comfy/lora.py calculate_weight/calculate_shape |
| `latents` | `LatentDescriptor` | comfy/latent_formats.py constant classes |
| `registry` | `Registry`, `Registrable`, `RegistryError`, `validate_registry_id` | hardcoded name lists (KSAMPLER_NAMES, supported_models.models) extended by monkey-patching |
| `application` | `ComponentApplication`, `ApplicationChain` | Ordered, identity-bound model applications that keep standalone components opaque until a host stages them with their runtime |
| `sampling` | `Denoiser`, `SolverFn`, `SamplerInfo`, `StepEvent`, `SigmaScheduleFn`, `ArithTensor`, `SamplingDescriptor`, `SamplerDescriptor`/`SchedulerDescriptor`, `Parameterization` | comfy/samplers.py + comfy/k_diffusion/sampling.py solver boundary and wrapper probing |
| `spaces` | `SigmaSpace`, `DiscreteSigmas`, `FlowSigmas`, `FluxFlowSigmas`, `ContinuousEDMSigmas`, `linear_beta_sigmas`, `time_snr_shift`/`flux_time_shift` | comfy/model_sampling.py schedule classes (the sigma<->timestep half), untangled from prediction mixins |
| `schedules` | `normal`/`sgm_uniform`/`simple`/`ddim_uniform`/`karras`/`exponential`/`beta`/`linear_quadratic`/`kl_optimal` `_schedule` fns; `SchedulerDescriptor` catalog + `builtin_scheduler_registry`; `offset_first_sigma_for_snr` | comfy/samplers.py SCHEDULER_HANDLERS + k_diffusion get_sigmas_*, as pure (steps, space) functions; the beta quantile is implemented privately (schedule-domain accuracy only) so scipy stays out |
| `solvers` | 13 `sample_*` ports (euler through lcm, RF/flow variants included) as `SolverFn` factories; `OptionSpec`/`resolve_options` typed option schemas; `SamplerDescriptor` catalog + `builtin_sampler_registry` (legacy names as aliases) | comfy/k_diffusion/sampling.py solver bodies (injected `NoiseSampler` instead of default-constructed noise, `SamplerInfo` instead of model probing); comfy/samplers.py KSAMPLER extra_options untyped dict |
| `steps` | `sampling_sigmas` (denoise-fraction tail trim, discard-penultimate correction, 0.9999 full-denoise threshold), `max_denoise` | comfy/samplers.py KSampler.set_steps/calculate_sigmas + Sampler.max_denoise, as pure functions over scheduler descriptors; the hardcoded DISCARD_PENULTIMATE_SIGMA_SAMPLERS name set becomes `SamplerDescriptor.discard_penultimate` data |
| `parameterizations` | `calculate_input`, `calculate_denoised`, `noise_scaling`, `inverse_noise_scaling` over `Parameterization` | comfy/model_sampling.py EPS/V_PREDICTION/EDM/CONST/X0 prediction mixins, as explicit dispatch instead of dynamic multiple inheritance |
| `lora` | `normalize_lora_keys`/`NormalizedLora` (dialect detection), `decode_lora`/`LoraDecodeResult` (adapter classification into `LoRASpec`/`LoHaSpec`/`LoKrSpec`/`GLoRASpec`/`OFTSpec`/`BOFTSpec`/`DiffPatchRef`/`SetPatchRef` against `PatchTarget`), `native_unet_key_map`/`clip_lora_key_map`/`flux_linear1_qkv_key_map` | comfy/lora_convert.py convert_lora, comfy/lora.py load_lora + model_lora_keys_clip, comfy/weight_adapter/*.load - key-string decisions only; every spec field references a SOURCE key, tensor reads deferred to the stage-4 torch layer |
| `cfg` | `cfg_combine`, `cfg_needs_uncond` | comfy/samplers.py cfg_function core + the cfg==1 uncond skip |
| `families` | `ModelFamily`, `FamilyDetector`, `DetectionEvidence`, `DetectionResult`, `FamilyRegistry`, `ComponentWiring` | comfy/model_detection.py ordered if-chain; comfy/supported_models.py ordered class list |
| `sources` | `SafetensorsSource`, `load_safetensors_header`, `MalformedSafetensors` | whole-dict loading via the safetensors library (comfy/utils.py load_torch_file) - here the header is validated strictly up front against the reference format rules (full dtype table, sub-byte alignment, exact payload tiling), payload bytes untouched |
| `signatures` | `KeySignature`, `ShapeIs`, `RankIs`, `DimField` | comfy/model_detection.py detect_unet_config's imperative key/shape walking, lifted into declarative signature data |
| `catalog` | `SD15`, `SDXL`, `SDXL_REFINER`, `FLUX_DEV`, `FLUX_SCHNELL`, `CHROMA`, `CHROMA_RADIANCE`, `MINIMAX_H3`, `builtin_families`, `builtin_family_registry` | grounded SD15/SDXL/Flux/Chroma and MiniMax H3 registrations, including distinct latent and structural multistream contracts. Unsupported variants are explicitly excluded, never misdetected |
| `codecs` | `CodecDescriptor`, `CodecEncoder`, `CodecDecoder`, `CodecMemoryEstimator`, `CodecTiling`, `latent_scales`, `plan_codec_decode`/`plan_codec_encode` | comfy/sd.py VAE signature-switch constructor + the decode_tiled_*/encode_tiled_* default geometry, planned from descriptor data instead of per-family methods |
| `tiling` | `LinearScale`/`CausalScale`, `TileSlice`, `PlannedTile`, `TilePlan`, `plan_tiles` | comfy/utils.py tiled_scale_multidim's index math (positions, edge clamping, output placement, feather widths) as a pure torch-free plan; the causal-video lambdas become typed `CausalScale`; step-zero/negative-step tile geometry refuses with `TilePlanError` instead of raising bare or planning nothing |
| `autoencoder_kl` | `KLConfig`, `KLDetectError`, `detect_kl_config`, `normalize_kl_keys`, `kl_descriptor`, `KLMemoryEstimator`, `KL_STANDARD_CH_MULT`/`KL_X4_CH_MULT`/`KL_PREFIX_RENAMES`, `KL_BATCH_NORM_EPS`/`KL_LATENT_PATCH` | comfy/sd.py VAE.__init__'s default SD1.x/SD2.x branch, run over `TensorGeometry` headers instead of a loaded state dict; canonical and Diffusers AutoencoderKL keys normalize to one strict layout, including rank-2 attention weights reshaped to 1x1 convs; ch_mult is inferred and required to match standard x8 or x4-upscaler; three accepted forms: the classic quant-conv layout, the regularizer-only layout (no quant convs, embed_dim == z_channels, the classic Flux ae.safetensors), and the Flux2 packed-latent (batch-norm) variant (bn running stats over the packed 2x2 latent, quant convs and embed_dim == z_channels required); every other variant (video/conv3d, TAESD, decoder-only, double_z=False, incoherent bn pairings) refuses with `KLDetectError` naming what was found - deferrals in ROADMAP "AutoencoderKL variants" |
| `taesd` | `TAESDConfig`, `TAESDDetectError`, `detect_taesd_config`, `taesd_layout`, `taesd_descriptor`, `TAESDMemoryEstimator` | comfy/taesd/taesd.py and the TAESD branch of comfy/sd.py @ f4b99bc: strict independent encoder/decoder geometry, SD1.5 versus SDXL role selection, latent scale/shift, descriptor, and memory policy, kept distinct from AutoencoderKL |
| `text_encoders` | `Tokenizer`, `WeightedSpan`, `EmbeddingRef`, `Conditioning` | comfy/sd1_clip.py braided tokenizer/weighting/embedding/forward hierarchy |
| `generation` | Immutable prompt/chat requests, ordered sampler chains, provider-bound session handles, provider capabilities, token/terminal events, results, and timing/token statistics | comfy_extras/nodes_textgen.py loose generation call and comfy/text_encoders/llama.py model-owned KV state @ 0a33ed6c |
| `openai_generation` | Stateless OpenAI-compatible completion/chat forwarding with strict standard and llama.cpp sampler mapping, SSE/JSON validation, pull backpressure, cancellation, timeout, usage normalization, and secret-safe failures | OpenAI-compatible `/v1/completions` and `/v1/chat/completions` transport without importing a model runtime |
| `scheduled` | Frozen normalized scheduled prompts/routes, layout-bound post-encode transform descriptors, ordered text/diffusion overlay states, closed-range intersection, finite `dinkster.inference/*` metadata identity, and partitioned execution-scoped variant ownership behind sibling `ScheduledFamilyRuntime` | Worker-local scheduled encoding contracts over the existing ConditioningSet/DMFC IR; no prompt syntax, torch, native handle, or load authority |
| `clip_bpe` | `ClipBpe`, `load_clip_bpe`, `normalize_text`, `CLIP_BOS`/`CLIP_EOS`/`CLIP_VOCAB_SIZE` | the Hugging Face `CLIPTokenizer` that comfy/sd1_clip.py delegates BPE to (transformers 4.57.3 over comfy/sd1_tokenizer/ data), reimplemented on the stdlib: non-ftfy normalization pinned, explicit scanner for the CLIP splitting regex, byte-level BPE over gzipped vendored vocab/merges (`data/clip_*.gz`, load-time sha256 provenance checks) |
| `prompt_tokens` | `parse_prompt_weights`, `tokenize_prompt`/`TokenizedPrompt`, `pack_spans`/`empty_chunk`/`TokenizerProfile`/`PackedToken`/`EmbeddingSlot`, `CLIP_L_PROFILE`/`CLIP_G_PROFILE`, `PromptTokenizer`, `WordEncoder`/`EmbeddingResolver` | comfy/sd1_clip.py tokenize_with_weights: the `(text:1.2)` emphasis grammar, `embedding:` directives (resolver returns vector counts - no weight files at tokenization), and chunk packing (BOS/EOS/pad family, large-word split, word ids); a bare `embedding:` is reported, not crashed (docs/comfyui-issues/sd1-clip-bare-embedding-directive-crash.md) |
| `unet` | `UNetConfig`, `SD15_UNET_CONFIG`/`SDXL_UNET_CONFIG`/`SDXL_REFINER_UNET_CONFIG`/`KNOWN_UNET_CONFIGS`, `UNET_HEAD_PROFILES`, `unet_layout`, `detect_unet_config`, `UNetDetectError` | comfy/model_detection.py detect_unet_config's standard-UNet leg (the `input_blocks.0.0.weight` scan) made strict over `TensorGeometry` headers, with attention-head counts resolved through supported_models.py unet_extra_config facts (q/k/v shapes cannot reveal head policy) and the surviving candidate required to reproduce the ENTIRE `unet_layout` listing; MMDiT/Cascade/audio/Flux, temporal/video (time_stack), SD2.x (unported fp32-attention pin), and pruned SDXL distillates refuse with `UNetDetectError` naming what was found (deferrals in ROADMAP "SD-era UNet variants") |
| `clip_text` | `ClipTextConfig`, `CLIP_L_TEXT_CONFIG`/`CLIP_G_TEXT_CONFIG`/`KNOWN_CLIP_TEXT_CONFIGS`, `clip_text_layout`, `detect_clip_text_config`, `ClipTextDetectError` | comfy/clip_model.py CLIPTextConfig-shaped kwargs dicts + comfy/sd1_clip.py/sdxl_clip.py per-model json configs, as one frozen config with an exact state-dict layout; detection matches checkpoint geometry against the KNOWN configs only - head count and activation are not inferable from geometry, so unknown/ambiguous layouts refuse with `ClipTextDetectError` (deferrals in ROADMAP "CLIP text-encoder variants") instead of guessing |
| `vendored` | `read_vendored` | shared loader for the gzipped `data/` resources: decompress + SHA-256 provenance check, loud `ValueError` on mismatch (consumed by `clip_bpe`, `graphemes`, `t5_spm`) |
| `graphemes` | `grapheme_clusters`, `grapheme_boundaries` | Unicode 16 UAX #29 extended grapheme cluster segmentation over a gzipped vendored break-property table (`data/grapheme_break.json.gz`, generated by `tools/gen_grapheme_tables.py`); needed because the SentencePiece Precompiled normalizer applies its charsmap per grapheme cluster |
| `t5_spm` | `T5SpmTokenizer`, `load_t5_spm`, `T5_XXL_FLUX_PROFILE`, `T5_XXL_PIXART_PROFILE`, `T5_PAD`/`T5_EOS`/`T5_UNK`/`T5_VOCAB_SIZE` | the Hugging Face `T5TokenizerFast` pipeline ComfyUI loads from comfy/text_encoders/t5_tokenizer/ (tokenizers 0.22.2 semantics), reimplemented on the stdlib: leftmost-longest added-token extraction, SentencePiece Precompiled-charsmap normalization per grapheme cluster, metaspace pre-tokenization, unigram Viterbi with unknown-fusion, decode - over the gzipped vendored tokenizer.json (`data/t5_tokenizer.json.gz`, sha256-guarded, byte-identical to the reference's); the profiles pin Flux and PixArt/Chroma packing |
| `t5_text` | `T5Config`, `T5_XXL_CONFIG`/`UMT5_XXL_CONFIG`/`BYT5_SMALL_GLYPH_CONFIG`/`KNOWN_T5_CONFIGS`, `T5_TEXT_ACTIVATIONS`/`T5_TEXT_OPTIONAL_KEYS`, `t5_layout`, `detect_t5_config`, `T5TextDetectError` | Exact T5-XXL, UMT5-XXL, and Hunyuan Image ByT5-small state-dict layouts from ComfyUI's T5 encoders, including bias-free Linears, RMS layer norms, model-specific relative-attention bias placement, gated activation, and optional duplicate embeddings. Detection accepts only a complete registered geometry; head count and activation are not inferred from projection shapes. |
| `jina_clip_text` | `JinaClipTextConfig`, `JINA_CLIP_V2_CONFIG`, `jina_clip_text_layout`, `detect_jina_clip_text_config` | Exact Jina CLIP v2 XLM-RoBERTa text layout for NewBie, including token-type embeddings, rotary attention, post-norm blocks, and masked mean pooling. Detection requires the complete 24-layer geometry. |
| `newbie_text` | `NEWBIE_TEXT_RECIPE`, `plan_newbie_component`, `bind_newbie_text` | Production dual-source NewBie binding over exact Gemma 3 4B and Jina CLIP v2 components, preserving source order, tokenizer provenance, reconstruction identity, and role composition. |
| `ace15_text` | `ACE15_TEXT_RECIPE`, `detect_ace15_components`, `bind_ace15_text_recipe` | ACE-Step 1.5 binding over an exact Qwen3 0.6B conditioner and Qwen3 2B or 4B audio-code language model, preserving structured prompt and ordered-source identity. |
| `quantization` | `LayerQuant`, `QuantSplit`, `split_quantization`, `KNOWN_QUANT_FORMATS`, `SUPPORTED_QUANT_FORMATS`, `UNSUPPORTED_QUANT_FORMATS`, `QuantizationError` | Classifies metadata, bounded payload `.comfy_quant`, and legacy quantization spellings before strict architecture detection. The exact FP8, NVFP4, MXFP8, INT8 tensorwise/ConvRot, ConvRot W4A4, and asymmetric W4A8 payload contracts are torch-free and identity-significant. Per-tensor FP8, NVFP4, and INT8 tensorwise/ConvRot are executable; MXFP8, ConvRot W4A4, and asymmetric W4A8 refuse before runtime/provider selection. |
| `assembly` / `chroma_component` | `FluxAssemblyPlan`, `SDAssemblyPlan`, `TAESDCodecPlan`, `ComponentPlan`, family and Chroma per-component planners, component prefixes, `AssemblyError` | comfy/sd.py load_checkpoint_guess_config / load_state_dict_guess_config @ b78cec87 plus current Chroma loading, planned from headers plus explicitly modeled scalar configuration tensors instead of a loaded state dict. Generic family assembly records exactly which source keys feed which model keys; Chroma plans diffusion, PixArt T5-XXL, and Flux KL VAE independently with strict storage dtypes, quantization, key renames, and immutable component identity. Chroma Radiance uses the shared T5 component and a synthetic pixel-space codec instead of a VAE. |
| `flux` | `FluxConfig`, `FLUX_DEV_CONFIG`/`FLUX_SCHNELL_CONFIG`/`KNOWN_FLUX_CONFIGS`, `flux_layout`, `detect_flux_config`, `normalize_flux_keys`, `FluxDetectError` | comfy/model_detection.py's flux branch (the `double_blocks.0.img_attn.norm.key_norm` leg) made strict over `TensorGeometry` headers: classic-Flux constants pinned as the reference pins them (axes_dim/theta/patch_size/mlp_ratio/qkv_bias/in_channels), derived facts scanned from the header, and the surviving candidate required to reproduce the ENTIRE bare-BFL `flux_layout` listing (dev 780 keys / schnell 776); optional `txt_norm` admits the row-28 context-normalized variant with its required RMSNorm weight while preserving the existing Flux family; Flux2 (double_stream_modulation_img), Chroma/Chroma Radiance (distilled_guidance_layer prefix), Ovis (yak MLP), LongCat-Image (vector-free at context width 3584; classic-width headers missing vector_in report as truncated), and FluxInpaint (widened img_in) refuse with `FluxDetectError` naming what was found (deferrals in ROADMAP "Classic-Flux variant and conditioning surface"); `normalize_flux_keys` is supported_models.py's scale->weight RMSNorm rename as a pure mapping transform |
| `chroma` | `ChromaConfig`, `ChromaRadianceConfig`, exact layout and detection functions | ComfyUI Chroma and Chroma Radiance detection/configuration: the shared 19+38 Flux-lineage transformer and distilled modulation approximator remain distinct from classic Flux; Radiance retains RGB pixel latents, convolutional patching, linear/conv NeRF heads, x0 and sequential-token markers |

## Design rules

- **Ids are namespaced and grammar-bound.** Registry ids follow
  `dinkster_schema.names` and must contain a `.` (`dinkster.euler`,
  `res4lyf.res_2m`). Collisions raise; aliases resolve on lookup but are
  never identity.
- **Detection is evidence, not order.** Family detectors return
  `DetectionEvidence` or `None`; the registry ranks by explicit
  `specificity`, and a tie at the top is a reported ambiguity, never a
  silent first-match win.
- **Policy is a value, not a probe.** Precision decisions produce a
  `PrecisionPlan`; solvers receive `SamplerInfo` instead of unwrapping
  model internals.
- **Everything is frozen.** Contracts are immutable dataclasses;
  mapping fields (`PatchSet.patches`, `DetectionEvidence.fields`) are
  snapshotted at construction, and `PatchSet` identity/equality is its
  `revision`, never structure. `PatchSet.merge` returns a new set with
  a new revision.
- **Protocols are static contracts.** Deliberately not
  `runtime_checkable` - `isinstance` on protocols with property members
  is shallow and misleading; plugin admission validates explicitly when
  it needs to.

## Learn more

See docs/native-inference-plan.md for the stage map, the upstream watch
map, and what each later stage adds. Proving tests:
`tests/test_inference_contracts.py` (stage-1 contracts),
`tests/test_inference_inspection.py` (stage-2 inspection/detection),
`tests/test_inference_sampling_math.py` (stage-3a sampling math), and
`tests/test_inference_solvers.py` (stage-3b solvers/registries) - the
stage-3 tests are pinned against goldens generated by RUNNING the
ComfyUI reference implementations (`tests/goldens/sampling_goldens.json`,
regenerated with `tools/gen_sampling_goldens.py` and a torch+scipy
interpreter). The stage-5 torch-free pieces are proven by
`tests/test_inference_tiling.py`, `tests/test_inference_kl.py`,
`tests/test_inference_clip_tokenize.py`,
`tests/test_inference_clip_text.py`,
`tests/test_inference_unet.py`,
`tests/test_inference_flux.py`,
`tests/test_inference_graphemes.py` (the full UAX #29 conformance
file), `tests/test_inference_t5_tokenize.py`, and
`tests/test_inference_t5_text.py` - the KL and tokenizer suites
likewise pinned against executed-reference goldens
(`tools/gen_kl_goldens.py`, `tools/gen_clip_tokenizer_goldens.py`,
`tools/gen_t5_tokenizer_goldens.py`;
the clip_text, unet, flux, and t5_text layouts are proven against
`tools/gen_clip_text_goldens.py` / `tools/gen_unet_goldens.py` /
`tools/gen_flux_goldens.py` / `tools/gen_t5_text_goldens.py`
output shared with the torch package).
