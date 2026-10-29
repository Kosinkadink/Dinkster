# Native inference plan (historical stage 0 architecture baseline)

Status: historical investigation and architecture baseline completed in
2026-07 against ComfyUI `947c2749`. This document is not a current-tip coverage
census or proof. Current source and accepted-parity status are reconciled in
[inference parity research](https://github.com/Kosinkadink/comfy-vibe-station/blob/main/notes/research/inference-parity-conclusions.md)
and `ROADMAP.md`. Direction is
pinned in DESIGN.md (mission + section 4): Dinkster is the successor to
ComfyUI. The compat pack remains the transitional execution backend and the
conformance oracle; the native stack below replaces it slice by slice. No
timeline map - stages land like every other slice.

This document answers three questions:

1. What does ComfyUI's inference stack actually know (module by module)?
2. Which parts are portable algorithms/data vs architecture to reject?
3. What is the staged Dinkster program, with dependency boundaries and order?

Terminology: "reuse" means reimplement the algorithm/knowledge faithfully in
typed Dinkster code with attribution; "reject" means the structure is not
copied, even when the behavior it produces is.

---

## 1. Map of the ComfyUI inference stack

Eight clusters, from the reference checkout (`../ComfyUI/comfy/`):

```diagram
+------------------------------ nodes / sd.py orchestration ----------------------------+
|                                                                                        |
|  +---------------+   +------------------+   +---------------+   +------------------+  |
|  | detection +   |   | loading (sd.py)  |   | sampling      |   | text encoders    |  |
|  | families      |-->| CLIP / VAE /     |-->| (samplers.py, |   | (sd1_clip.py,    |  |
|  | (model_detec- |   | checkpoint split |   | k_diffusion)  |   | text_encoders/)  |  |
|  | tion, suppor- |   +--------+---------+   +-------+-------+   +--------+---------+  |
|  | ted_models,   |            |                     |                    |            |
|  | model_base,   |            v                     v                    v            |
|  | latent_for-   |   +-------------------------------------------------------------+  |
|  | mats, model_  |   | model_patcher (patches, residency, hooks, clones, saving)   |  |
|  | sampling)     |   +------------------------------+------------------------------+  |
|  +---------------+                                  v                                  |
|                     +---------------------------------------------------------------+  |
|                     | model_management (device policy, dtype policy, VRAM registry, |  |
|                     | streams, pinning) + memory_management (file slices, gathered  |  |
|                     | buffers) + ops.py (cast-at-use layers, quant, lowvram)        |  |
|                     +---------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------+
```

Per-cluster verdicts are historical architecture findings (full investigation
notes preserved in the thread; key line references are against `947c2749`),
not current-tip source claims:

### 1.1 model_management.py (+ memory_management.py, pinned_memory.py, model_prefetch.py)

- Knows: device/backend detection and capability tables (fp16/bf16/fp8/nvfp4
  support per backend), the VRAM state machine (DISABLED..HIGH_VRAM/SHARED),
  effective-free-memory math (physical free + reclaimable allocator reserve),
  the loaded-model MRU registry with partial-unload-before-detach eviction,
  inference-memory reserve (~0.8 GiB + platform reserve), dtype selection for
  UNet/text-encoder/VAE, offload streams and cast buffers, pin-budget
  hysteresis. memory_management.py adds file-slice direct reads, geometry-only
  size planning, 1024-byte-aligned gathered buffers, quantized-tensor
  flatten/unflatten.
- Reuse: all of the math and policy criteria above; partial-unload ordering;
  stable dedup + dependency inclusion in load requests; deferred pin
  registration with transactional rollback; one-step prefetch lookahead.
- Reject: import-time hardware probing and global torch mutation; the
  process-global registry/stream/pin/interrupt state; metadata hidden on
  tensor storage (`_comfy_tensor_file_slice`, `_model_dtype`) and on modules
  (`comfy_cast_weights`, `_pin`, `_v`, ...); weakref/finalizer-driven
  correctness; one module owning device inventory + policy + orchestration.
- Dinkster shape: explicit services - DeviceInventory, PrecisionPolicy,
  MemoryPlanner, ResidencyManager, TransferPool, HostPinAllocator - all
  constructed, injected, request-scoped where possible. Much of the *policy*
  seam already exists as dinkster-memory's MemoryGovernor (reservations, leases,
  placement); the native program adds the mechanism layer under it.

### 1.2 model_patcher.py (+ lora.py, patcher_extension.py)

- Knows: the weight-patch algebra. `patches: dict[key, list[entry]]` with
  entries `(strength_patch, value, strength_model, offset, function)`; value
  is an adapter object or `("diff"|"set"|"model_as_lora", ...)` or a nested
  list. Application: backup-before-mutation, compute in intermediate dtype,
  deterministic stochastic rounding back to storage dtype, exact restoration
  on unpatch. Low-VRAM mode defers patching into per-use weight functions.
  Hook patches are the same algebra grouped by hook ref with keyframe
  scheduling. patcher_extension is an ordered middleware chain keyed by
  string phases.
- Reuse: patch composition order and strength semantics; offset/narrow
  patches; shape prediction before allocation; stochastic rounding; backup/
  restore; patch-set identity (uuid); lowvram deferred patching; the
  middleware-chain idea.
- Reject: ModelPatcher as one mutable god object (patch set + residency +
  hooks + extension registry + save adapter + clone manager); tuple-length
  dispatch; clones sharing mutable backup dicts; nested untyped
  model_options/transformer_options as the extension protocol; arbitrary
  object injection as the normal extension mechanism; `__del__` cleanup.
- Dinkster shape: immutable typed PatchSet (a small algebra: Diff, Set,
  ModelAsLora, Adapter, Offset, Nested), separate ResidencyState and
  HookPlan, application as a pure-ish function (weights in, weights out,
  with an explicit backup store). LoRA format normalization (lora_convert
  dialect detection) becomes typed format decoders producing PatchSet.

### 1.3 model_detection.py + supported_models(_base).py + model_base.py + latent_formats.py + model_sampling.py

- Knows: the detection decision tree (key signatures + tensor shapes +
  block counting -> untyped config dict), first-match-wins against the
  ordered 96-class `supported_models.models` list; per-family metadata
  (unet_config signature, latent format, sampling settings, component
  prefixes, dtype capabilities, memory_usage_factor) plus executable factory
  and conversion methods; BaseModel as the architecture adapter hub (input
  preconditioning via model_sampling, concat conds, timestep conversion,
  extra_conds wrapping, memory estimation) with ~all `ldm/*` architectures
  imported eagerly; 35 latent-format classes (mostly declarative constants +
  some packing logic); sampling parameterizations as mixins (EPS,
  V_PREDICTION, EDM, CONST/flow, ...) dynamically multiple-inherited with
  schedule classes (Discrete, ContinuousEDM, DiscreteFlow, Flux shift, ...).
- Reuse AS DATA: detection signatures (required keys, shape-derived fields,
  block-count rules) with explicit priority/specificity; family match
  predicates; latent descriptors (channels, dimensions, scale/shift,
  preview factors, downscale ratios); sampling parameterization enum +
  schedule parameters; component prefixes; dtype capabilities; memory
  coefficients. Reuse as algorithms: sigma/timestep conversion math,
  prediction formulas (eps/v/EDM/flow), the diffusers key-mapping tables.
- Reject: the central if-chain + central ordered list (silent order-
  dependence; adding a model means editing two central files); detection
  that mutates its input state dict; config as untyped dict; dynamic
  multiple inheritance for sampling; BaseModel importing every architecture;
  family classes owning device/dtype/ops policy.
- Dinkster shape: THE model-family plugin registry. A family registration is
  (detector with evidence + explicit specificity, typed config schema,
  model factory, conditioning adapter, latent descriptor, sampling
  descriptor, component wiring). Detection returns typed evidence and
  diagnostics; ambiguous matches are explicit and testable. This is the
  centerpiece that makes "support a new model" a pack-registerable act
  instead of a core edit - directly serving the mission.

### 1.4 sd.py (loading orchestration) + ops.py + quant_ops.py

- Knows: checkpoint assembly end to end (safetensors/ckpt read -> prefix
  detection -> component split via family prefixes -> dtype/device policy ->
  construction -> patcher wrapping); the VAE facade (a giant ordered
  signature switch over ~30 codec families, each mutating layout/memory/
  tiling fields) with portable N-dimensional overlap/feather tiling math;
  ops.py's cast-at-use engine (storage dtype vs compute dtype vs bias dtype,
  layer classes with disabled init, manual_cast, fp8 paths, lowvram weight
  functions, quant dispatch via comfy-kitchen layouts) injected as an
  `operations` class namespace into every model constructor.
- Reuse: safetensors-safe loading semantics + metadata; pure prefix/key
  transforms; the storage/compute/offload dtype-and-device four-way
  distinction; cast-at-use ordering (gather -> dequant -> lowvram patch ->
  patch fn -> requant); tiling index math and feather blending as typed
  TilePlan; weighted batch-size-from-free-memory as an explicit policy.
- Reject: sd.py as one 2200-line universal factory; VAE as descriptor +
  factory + scheduler + memory manager + codec in one class; exception-
  driven OOM fallback as the normal strategy; operations as an untyped
  class namespace with patcher-installed attributes; aimdo/kitchen coupling
  smeared across loader, storage, ops, and policy instead of behind one
  backend adapter.
- Dinkster shape: a WeightSource abstraction (safetensors header inspection,
  file-slice reads - no full-materialization requirement), typed codec
  descriptors (each VAE family is a plugin implementing encode/decode with
  a declared layout + memory formula; TAESD-style codecs are just small
  plugins), and an Ops protocol (typed interface, not a class namespace)
  with kitchen/aimdo as optional backends behind it.

### 1.5 samplers.py + k_diffusion/ + conds.py + sample.py

- Knows: the full sampling path (CFGGuider -> sampling_function ->
  calc_cond_batch -> BaseModel.apply_model, k-diffusion solver on top);
  CFG math with pre/post-CFG callback ordering; conditioning as
  `[cross_attn, metadata]` pairs upgraded into wrapper values (CONDRegular/
  NoiseShape/CrossAttn/Constant/List) with area/mask/timestep-range/hook
  semantics; batching by compatibility (same shape, control, patches,
  concat-able conds) chunked by free memory; 45 `sample_*` solver functions
  sharing the signature `(model, x, sigmas, extra_args, callback, disable,
  **opts)`; 9 registered sigma schedules, mostly pure functions; sampler/
  scheduler registration as hardcoded module-level name lists resolved via
  `getattr(k_diffusion.sampling, "sample_"+name)` - the exact monkey-patch
  seam res4lyf-style packs are forced through.
- Reuse: solver equations wholesale; schedule formulas (karras, exponential,
  beta, kl_optimal, ... are pure); CFG semantics and callback ordering;
  area/mask weighted composition; the cond-value concatenation protocol
  (typed); the callback event shape; the custom-sampler/SIGMAS object model.
- Reject: global name tables and getattr-as-registry; special-name branches
  (uni_pc, ddim); solvers probing `model.inner_model.inner_model.
  model_sampling` (capability probing through wrapper guts); untyped
  cond/option envelopes; calc_cond_batch braiding semantics + memory policy
  + hooks + controlnet + multigpu + execution in one function.
- Dinkster shape: a typed Denoiser protocol as the solver boundary (solver
  capabilities like RF/noise-scaling are explicit context fields, never
  introspection); a first-class sampler/scheduler REGISTRY where entries
  are descriptors (stable id, display name, factory, typed options schema,
  required model capabilities, sigma endpoint policy, provenance/owner).
  This registry is what the deferred "first-class cross-worker sampler/
  scheduler registry" promise plugs into, and what the remote-COMBO route
  already serves on the wire.

### 1.6 sd1_clip.py + text_encoders/

- Knows: prompt tokenization with emphasis syntax (`(text:1.2)`, nesting at
  1.1x, escapes), textual-inversion embedding splicing, section splitting
  for long prompts, the empty-baseline weighting interpolation
  `empty + w * (enc - empty)`, per-family multi-encoder composition (SDXL
  L+G channel concat, SD3 L/G pad-to-T5 + sequence concat, etc.). 47 encoder
  modules, integration via central branch edits in sd.py (no registry).
- Reuse: the parser, sectioning rules, splice algorithm, weighting
  interpolation, and each family's composition rule as typed conditioning
  schemas.
- Reject: tokenization braided with filesystem lookup, device policy, and
  execution; central branchy registration; tuple/dict outputs whose meaning
  varies by family.
- Dinkster shape: text encoders register through the same family-plugin
  registry as diffusion models (a family's component wiring declares its
  encoders); tokenizer/encoder/composer are separate typed pieces.

---

## 2. Cross-cutting decisions

1. **Contracts before code.** Every stage starts by landing typed protocols
   in `dinkster-inference`; implementations follow. Pyright-strict; contracts
   ARE the deliverable of stage 1. This directly fixes the "zero type hints
   in the intricate parts" failure.
2. **Registry over central edits.** Model families, codecs, samplers,
   schedulers, and text encoders are all registrations (in-core registrations
   for the basics, pack-provided for extensions), never edits to a central
   if-chain or list. Ordering is explicit (specificity/priority values),
   ambiguity is a diagnostic, never silent first-match.
3. **No hidden state.** Nothing gets attached to torch modules, tensors, or
   storage. Residency, pins, prefetch state, and patch state live in
   explicit typed owners keyed by stable ids.
4. **Comfy as oracle.** Conformance tests run the same weights + seed
   through the compat worker and the native path and compare within
   tolerance. The compat pack is the regression net that lets native pieces
   land incrementally.
5. **Effects at boundaries.** IO (file reads), device transfers, and
   allocator calls happen in named services (WeightSource, TransferPool,
   ResidencyManager); model/solver code is pure with respect to them.
6. **Governor owns policy.** dinkster-memory's MemoryGovernor remains the
   single policy seat. The native stack supplies mechanisms (placement
   plans, transfer execution, pin budget enforcement) as governor-driven
   services - never a second free-and-hope pathway.
7. **Attribution doubles as a tracking anchor.** Every reimplemented
   algorithm cites its ComfyUI source module AND the reference commit it
   was ported at in the docstring (e.g. "ported from comfy/samplers.py
   @ 947c2749"). This is not just credit - it is what makes upstream
   diffing mechanical (section 5).

## 3. Staged program (order = dependency order)

Each stage is independently shippable and testable; later stages consume
earlier contracts. Stages produce ROADMAP slices when picked up.

- **Stage 1 - contracts (`dinkster-inference` package).** Typed protocols and
  data models: Device/DTypePolicy, WeightSource (safetensors header +
  slice reads), StateDictView (prefix/key transforms as pure functions),
  PatchSet algebra, LatentDescriptor, SamplingDescriptor (parameterization
  enum + schedule params), ModelFamily registration record, Denoiser
  protocol, SamplerDescriptor/SchedulerDescriptor, CodecDescriptor
  (VAE encode/decode), TextEncoder contracts. No torch execution required
  to test the contracts; unit tests pin shapes and invariants.
- **Stage 2 - inspection and detection.** Native safetensors header
  inspection (no weight materialization), detection-as-data: family
  signatures with evidence + specificity, the registry, diagnostics for
  ambiguity. Oracle test: every checkpoint the compat worker can classify,
  the native detector classifies identically. (Builds on the existing
  probe_file work in assets.)
- **Stage 3 - sampling math.** Split into two slices:
  - **3a (shipped 2026-07):** sigma spaces (`DiscreteSigmas`, `FlowSigmas`,
    `FluxFlowSigmas`, `ContinuousEDMSigmas` behind the `SigmaSpace`
    protocol, replacing stage 1's placeholder `ScheduleContext`), all nine
    ComfyUI schedules (normal, sgm_uniform, simple, ddim_uniform, karras,
    exponential, beta, linear_quadratic, kl_optimal - beta via a pure-
    Python quantile, no scipy), prediction parameterizations
    (eps/v/EDM/flow-CONST/x0: calculate_input, calculate_denoised,
    noise_scaling, inverse_noise_scaling), and the pure CFG combination
    with the exact cfg==1 skip predicate. All torch-free against a
    structural `ArithTensor` protocol. Goldens are NOT re-derived: they
    are generated by executing ComfyUI's own implementations at the
    audited reference commit 947c2749 (`tools/gen_sampling_goldens.py`,
    regeneration command in its docstring), pinned in
    `tests/goldens/sampling_goldens.json` and proven by
    `tests/test_inference_sampling_math.py`.
  - **3b (shipped 2026-07):** 13 solver ports from k_diffusion behind
    the Denoiser protocol (euler, euler_ancestral, heun, dpm_2,
    dpm_2_ancestral, dpmpp_2s_ancestral, dpmpp_sde, dpmpp_2m,
    dpmpp_2m_sde, dpmpp_2m_sde_heun, dpmpp_3m_sde, ddpm, lcm - each
    with the reference's RF/flow variant where one exists), torch-free
    with injected NoiseSampler and declared NoiseKind
    (gaussian/brownian), typed per-sampler option schemas
    (OptionSpec/resolve_options: FLOAT bounds, CHOICE sets, NaN/bool
    rejection), SchedulerDescriptor catalog for the nine schedules,
    and both registries served as core choices (dinkster.samplers /
    dinkster.schedulers on every composition, /api/choices/{id}; legacy
    ComfyUI names resolve as aliases). Golden-pinned like 3a by
    executing the reference at 947c2749 (40 solver cases incl.
    flow-at-1.0 SDE offset, the 2S-ancestral sigma==1 branch, heun
    flow path, eta=0, scaled-flow ancestral, churn saturation).
    Proven by `tests/test_inference_solvers.py` and
    `tests/test_choices.py`. Unported solvers (cfg_pp family, ddim,
    lcm extras, exotic tail) are ledgered in ROADMAP "Unported solver
    tail" with per-item blockers; cross-worker pack registration of
    descriptors stays a separate deferred item.
- **Stage 4 - weights, patches, residency mechanism.** WeightSource-backed
  loading (file-slice -> gathered buffer -> device), typed PatchSet
  application with backup/restore + stochastic rounding, LoRA dialect
  decoders, cast-at-use Ops protocol, governor-driven ResidencyManager
  (partial-unload ordering, pin budget) with aimdo as an optional backend.
  Oracle test: patched-weight equality against compat within rounding
  tolerance. Sub-slices: 4a (SHIPPED 2026-07) torch-free LoRA dialect
  decoding in `dinkster_inference.lora` - convert_lora dialect
  normalization (BFL Flux Control, Wan Fun, USO), all six
  weight-adapter probes (LoRA x7 naming variants, LoHa, LoKr, GLoRA,
  OFT/BOFT split by block rank) as key-layout classification whose
  specs reference SOURCE keys with all tensor reads deferred,
  diff/set/norm patches, universal unet + CLIP + Flux fused-qkv key
  maps; golden-pinned against executed ComfyUI @ 947c2749
  (`tests/test_inference_lora.py`, goldens from
  `tools/gen_lora_goldens.py`). 4b: the `dinkster-inference-torch`
  package - weight materialization, adapter math, patch application,
  oracle equality (torch code ports stay near-transcription per the
  torch-free boundary rules). 4b slice 1 (SHIPPED 2026-07): adapter
  tensor math - LoRA (LoCon mid, reshape/pad, DoRA both axes, fp16),
  LoHa (Tucker, DoRA), LoKr (full/decomposed, DoRA), GLoRA (both
  orientations incl. conv, DoRA), OFT, BOFT (rescale, fp16) as
  near-transcriptions of comfy/weight_adapter @ 947c2749,
  golden-pinned against the executed reference (26 cases,
  `tools/gen_adapter_goldens.py`); the reference's log-and-return-
  unpatched error path becomes a loud `AdapterMathError`, and two
  reference paths that ALWAYS fail at the pin (LoKr Tucker+conv kron
  contiguity, BOFT fp16 at strength != 1) are FIXED in Dinkster with
  independent verification, upstream analyses in docs/comfyui-issues/
  (ROADMAP "Upstream-broken adapter paths"). 4b
  slice 2 (SHIPPED 2026-07): weight materialization from the stage-4a
  source-key specs (`materialize.py` - reference scalar-read
  semantics, loud MaterializeError on missing source keys) + PatchSet
  application (`apply.py` - sequential entries, strength_model,
  offsets, nested, diff/set/model-as-lora, exact-original
  backup/restore with rollback) + storage-dtype writeback
  (`rounding.py` - manual stochastic rounding with per-key CRC
  seeds, LOAD-BEARING for fp8/quantized storage); golden-pinned
  against executed comfy/lora.py calculate_weight, weight_adapter
  loads, comfy/float.py and comfy/utils.py string_to_seed @ 947c2749
  (`tools/gen_patch_goldens.py`,
  `packages/dinkster-inference-torch/tests/test_patches.py`). fp8
  rounding goldens replay bitwise only on the generation torch build
  (torch.rand fp16 CPU streams changed between torch 2.9 and 2.13)
  and via the exact adjacent-fp8-grid contract elsewhere; the port
  was verified bit-equal on the reference interpreter. 4c slice 1
  (SHIPPED 2026-07): cast-at-use + native fp8 storage - the
  per-entry LowVramPatch `function` hook and nested-patch
  `convert_func` restored end to end (PatchEntry.function /
  NestedPatch.convert, threaded through every adapter delta site and
  weight_decompose exactly where the reference calls `function(...)`),
  native scaled-fp8 weight storage (`quant.py` Fp8ScaledWeight +
  quantize/dequantize/requantize transcribed from comfy/quant_ops.py
  _TensorCoreFP8LayoutBase + kitchen's eager kernels, bit-exact),
  patch_weights routing fp8 stores through
  dequantize -> patch -> seeded requantize, comfy-kitchen accelerated
  fp8 stochastic rounding behind a capability probe with the manual
  path as fallback (accelerated path ships WITH fp8 handling per the
  performance-parity discipline, and its seeded uint8 randint stream
  replays bit-exact across torch builds, unlike the manual fp16
  stream), and the cast-at-use core (`ops.py` cast_weight: move at
  storage dtype -> cast/dequantize to compute dtype -> deferred
  functions on an owned buffer; DeferredPatch = LowVramPatch with the
  intermediate pinned to the weight's own dtype); dynamo fullgraph
  parity for the compute that consumes cast outputs, with patching
  and storage mutation kept outside compiled regions; CUDA/Triton
  validated 2026-07-24 on 2x RTX 4090 (tests/test_gpu.py: kitchen
  CUDA backend dispatch per GPU, inductor/triton fullgraph compile
  from main + worker threads, cross-device moves; one multi-GPU
  defect found and fixed in rounding.py - device-context pin around
  the kitchen kernel). 4c slice 2 (SHIPPED 2026-07): residency
  mechanism + aimdo seam - device-memory introspection (`memory.py`:
  the reference's effective-free-memory math - driver-free plus the
  torch allocator's reclaimable reserve - with reserve constants as
  an injected `MemoryPolicy` value instead of import-time globals),
  per-model placement (`residency.py` ResidentWeights: key-group
  units settle largest-offload-estimate-first under strict-< byte
  budgets, resident keys hold patched storage with exact-original
  backups, offloaded keys stay pristine and patch at cast time via
  DeferredPatch - the reference's patch_weight_to_device/LowVramPatch
  split; `apply.py` gained the pure one-key `patch_stored_weight`),
  and fleet policy (`ResidencyManager`: MRU registry, the 1.1-inflated
  free-ahead-of-load pass, the verbatim low-VRAM budget formula,
  eviction ordered most-offloaded/smallest/newest with
  partial-unload-before-detach - load_models_gpu/free_memory
  @ 947c2749 over explicit injected state, int byte budgets replacing
  the 0.1/1e32 float sentinels, `None` = unlimited). The manager sees
  models only through the `ResidencyMechanism` protocol - that seam
  is where an aimdo VBAR-backed mechanism drops in (`aimdo.py` ships
  the capability probe and pins the two integration constraints:
  control.init() before torch in worker bootstrap, probe via
  comfy_aimdo.control never the package root); the full dynamic
  mechanism needs stage 5's per-op fault brackets (ROADMAP). Pin
  budgets, offload streams, prefetch, force_patch_weights, and the
  VRAMState host modes are explicitly deferred (ROADMAP). Policy math
  pinned on CPU (test_residency.py), real cross-device movement,
  fp8 residency, manager eviction on live free-memory measurements,
  and two-device isolation CUDA-validated 2026-07-24 on 2x RTX 4090
  (test_gpu.py residency section).

  **Per-unit native offloading floor (SHIPPED 2026-07).** Assembly remains
  unchanged: enrollment derives `ModuleStateStore`, a live mutable view of
  each assembled component's module state, after strict state-dict loading.
  Direct state-owning modules become residency units; units that share
  tensor storage are merged so tied aliases cannot split across devices.
  Enrolled layers keep their existing fast path while resident and switch,
  in the shape of ComfyUI's `comfy_cast_weights`, to
  `ResidentWeights.use()` cast-at-use only while offloaded. `Fp8Linear`
  exposes qdata plus weight scale as one folded `Fp8ScaledWeight`; its
  unpatched hardware path moves that folded value without dequantizing,
  while an offloaded patched fp8 weight uses the dequantized route because
  `DeferredPatch` dequantizes it. `T5LayerNorm` is the one architecture
  weight owner not built by an Operations factory and is explicitly
  enrolled; its resident input-dtype variance path remains unchanged.
  CPU and CUDA parity tests pin resident, mixed, fully offloaded, fp8, and
  patched routes without changing state-dict layouts or pipeline goldens.
- **Stage 5 - native components.** In order: VAE (smallest surface; codec
  plugins + typed tiler), text encoders (tokenizer/weighting/composition),
  then diffusion cores per family (start with one modern DiT family and
  SDXL-era UNet; port `ldm/` architectures as needed with `operations`
  replaced by the Ops protocol). Oracle test per component: output
  closeness on fixed seeds/weights. Slice 1 (VAE groundwork) shipped
  2026-07: `dinkster_inference.tiling` plans tiled_scale_multidim's index
  math torch-free (typed LinearScale/CausalScale replace the reference's
  number-or-lambda scale entries; `LatentDescriptor.temporal_causal`
  marks causal video time axes), `plan_codec_decode`/`plan_codec_encode`
  plan from `CodecTiling` descriptor defaults, `tiled_apply` executes
  the plan golden-pinned against executed reference outputs (no
  inference_mode - training program, 3.1), and `CodecPlugin` binds a
  descriptor to encoder/decoder implementations with the reference's 2D
  three-aspect seam sweep. Slice 2 (SHIPPED 2026-07): the first real
  codec family, native SD/SDXL AutoencoderKL. Torch-free half
  (`dinkster_inference.autoencoder_kl`): geometry-based detection over
  safetensors headers replacing sd.py VAE.__init__'s
  loaded-state-dict shape reads - accepts the standard x8 (1,2,4,4)
  and x4-upscaler (1,2,4) layouts, infers ch_mult from block widths
  instead of trusting key-presence heuristics, refuses every other
  variant loudly (`KLDetectError`; deferrals ledgered in ROADMAP
  "AutoencoderKL variants"). Torch half
  (dinkster_inference_torch.autoencoder_kl): full
  Encoder/Decoder/ResnetBlock/AttnBlock port from
  ldm/modules/diffusionmodules/model.py under the typed `Operations`
  factory seam (operations.py, replacing comfy/ops.py's class-swap
  mechanism; attention via SDPA), Comfy-compatible state-dict names,
  deterministic posterior-mode encode (the reference default,
  regularizer sample=False; stochastic encodes take an explicit
  generator, never hidden global RNG),
  `kl_codec_plugin` wiring it into the slice-1 CodecPlugin seam with
  whole-content transforms (`content_crop` center-crops encode input
  to the downscale grid, the reference's vae_encode_crop_pixels
  spatial branch, before tile planning ever sees the shape; then
  x*2-1 in, (x+1)/2 clamp out - each applied once around direct OR
  tiled traversal, never per tile).
  sources.py reads safetensors payload ranges into
  owned torch tensors straight from the stage-2 header abstractions
  (no numpy, no safetensors package). Goldens executed from the
  pinned reference (tools/gen_kl_goldens.py, deterministic
  name-seeded weights); CUDA-validated on 2x RTX 4090 including
  strict-fp32 golden closeness, fp16/bf16, autograd, tiled, fullgraph
  compile, worker threads, and two-GPU isolation (test_gpu.py KL
  section). Codec-owned tiling, extra_1d_channel audio, and OOM
  fallback stay ledgered (ROADMAP "Codec plugin gaps").
  Slice 3 (SHIPPED 2026-07): SD1/SDXL CLIP tokenization and prompt
  weighting, entirely torch-free (`dinkster_inference.clip_bpe` +
  `prompt_tokens`). `clip_bpe` reimplements the exact Hugging Face
  CLIPTokenizer pipeline ComfyUI delegates to (comfy/sd1_clip.py
  SDTokenizer -> transformers 4.57.3 CLIPTokenizer over
  comfy/sd1_tokenizer/ data) on the stdlib: the non-ftfy
  normalization path (the audited reference environment does not
  install ftfy; one deterministic normalization is pinned instead of
  varying with an optional import), an explicit scanner for the CLIP
  splitting regex (stdlib `re` has no `\p` classes), raw
  special-token literal extraction before normalization, and
  byte-level BPE over gzipped copies of the vendored vocab/merges
  (provenance sha256 of the uncompressed bytes checked at load).
  `prompt_tokens` ports `tokenize_with_weights` split along the
  stage-1 text_encoders contracts: `parse_prompt_weights` (the
  `(text:1.2)` emphasis grammar verbatim, replace-not-multiply
  explicit weights, 1.1x nesting, escapes), `tokenize_prompt`
  (WeightedSpans per reference token group; embedding directives
  resolve through an injected `EmbeddingResolver` returning vector
  counts, keeping tokenization free of weight files), `pack_spans`
  (chunking, BOS/EOS/pad family, large-word split rule, word ids)
  with `TokenizerProfile` capturing the reference SDTokenizer
  constructor knobs (CLIP_L_PROFILE pads with EOS, CLIP_G_PROFILE
  with 0), and the `PromptTokenizer` facade satisfying the stage-1
  `Tokenizer` contract. Goldens executed from the pinned reference
  (tools/gen_clip_tokenizer_goldens.py: raw HF BPE ids, SD1 + SDXL-G
  tokenize_with_weights with real safetensors embedding fixtures,
  packing variants). Deliberate divergence: a bare `embedding:`
  directive is reported, not crashed
  (docs/comfyui-issues/sd1-clip-bare-embedding-directive-crash.md).
  Slice 4 (SHIPPED 2026-07): the SD1/SDXL CLIP text model forward
  and conditioning composition. Torch-free half
  (`dinkster_inference.clip_text`): `ClipTextConfig` carries exactly
  the construction-relevant subset of the reference config JSONs
  (vocab_size 49408 is hardcoded in the reference CLIPEmbeddings and
  pinned explicitly; text_projection is always hidden->hidden, the
  JSON's projection_dim ignored), `clip_text_layout` generates the
  reference's exact state-dict key/shape listing, and
  `detect_clip_text_config` classifies text-model headers by FULL
  layout comparison - only CLIP-L (49408x768) and CLIP-G
  (49408x1280) are accepted because attention-head count and
  activation are not derivable from tensor shapes; everything else
  (OpenCLIP resblocks format, long-context position tables, unknown
  geometries) refuses with `ClipTextDetectError` (deferrals ledgered
  in ROADMAP "CLIP text-encoder variants"). Torch half
  (dinkster_inference_torch.clip_text): faithful clip_model.py port
  under the typed `Operations` seam (linear/layer_norm/embedding
  initless factories added), causal attention via SDPA
  `is_causal=True`, bind-time activation selection (quick_gelu for
  L, gelu for G), hidden-layer selection with negative indices,
  EOS-position pooling with raw + projected outputs
  (`ClipTextOutput`), reference-identical state-dict keys.
  `ClipTextEncoder` ports SDClipModel.encode_token_weights over the
  slice-3 packed chunks: batched sections plus one empty-prompt
  chunk when any weight deviates, textual-inversion row substitution
  into the embedding sequence, `(z - z_empty) * weight + z_empty`
  interpolation with weight-1.0 positions bit-exact, first-chunk
  pooled, float32 outputs; `ClipEncodePolicy` captures the family
  knobs (SD1_CLIP_L_POLICY: final layer + raw pooled;
  SDXL_CLIP_POLICY: un-normed penultimate layer + projected pooled);
  `compose_sdxl_conditioning` is SDXLClipModel's two-tower
  concatenation (cut to shorter, 768+1280 features, CLIP-G pooled).
  Deliberate divergence: width-mismatched textual-inversion vectors
  refuse with `ClipEncodeError` instead of the reference's
  warn-and-drop (silent conditioning drift is worse). Goldens
  executed from the pinned reference
  (tools/gen_clip_text_goldens.py: full-size CLIP-L/G layouts on the
  meta device, tiny hash-filled architectures through
  encode_token_weights and the SDXL composition; generation refuses
  off the audited commit).
  Slice 5 (SHIPPED 2026-07): the first diffusion core, the native
  SD1/SDXL 2D image UNet. Torch-free half (`dinkster_inference.unet`):
  `UNetConfig` carries exactly the construction-relevant subset of
  the dicts the reference's detection emits, with the three
  supported profiles pinned as constants (SD15/SDXL-base/
  SDXL-refiner); `unet_layout` generates the reference's exact
  state-dict key/shape listing; `detect_unet_config` is
  comfy/model_detection.py detect_unet_config's standard-UNet leg
  made strict - the input/output transformer-depth scan over
  geometry headers, attention-head counts resolved through
  `UNET_HEAD_PROFILES` (supported_models.py unet_extra_config facts;
  q/k/v shapes cannot reveal head policy), and the surviving
  candidate must reproduce the ENTIRE layout. Everything else -
  MMDiT/Cascade/audio/Flux (no input_blocks.0.0.weight),
  temporal/video UNets (time_stack), SD2.x (unported
  fp32-attention pin), pruned distillates (negative
  transformer_depth_middle) - refuses with `UNetDetectError` naming
  what was found (deferrals ledgered in ROADMAP "SD-era UNet
  variants"). Torch half (dinkster_inference_torch.unet): faithful
  openaimodel.py + attention.py port under the typed `Operations`
  seam - timestep_embedding, ResBlock, Upsample/Downsample with the
  odd-extent output_shape leg, CrossAttention through SDPA with the
  reference head reshape, GEGLU/FeedForward, SpatialTransformer with
  conv (SD1) or linear (SDXL) projections, optional SDXL ADM
  conditioning (y required iff adm_in_channels set), state-dict keys
  IDENTICAL to the reference (time_embed.*, label_emb.0.*,
  input_blocks.N.M.*, middle_block.M.*, output_blocks.N.M.*, out.*).
  No inference_mode/no_grad anywhere and out-of-place residual adds
  (training program, 3.1). Goldens executed from the pinned
  reference (tools/gen_unet_goldens.py: full-size layouts on the
  meta device for all three profiles, tiny hash-filled executed
  forwards incl. odd-spatial and ADM cases; generation refuses off
  the audited commit). CUDA-validated on 2x RTX 4090 (strict-fp32
  golden closeness, bf16/fp16, autograd, fullgraph inductor compile,
  worker threads).
  Slice 6 (SHIPPED 2026-07): the first modern DiT core, native
  classic Flux dev/schnell. Torch-free half (`dinkster_inference.flux`):
  `FluxConfig` carries exactly the construction-relevant FluxParams
  subset (most classic-Flux facts are pinned constants in the
  reference's detection: axes_dim (16,56,56), theta 10000,
  patch_size 2, mlp_ratio 4.0, qkv_bias, in_channels 16);
  `flux_layout` generates the reference's exact bare-BFL key/shape
  listing (dev 780 keys with guidance_in, schnell 776 without);
  `detect_flux_config` is comfy/model_detection.py's flux branch
  made strict - derives hidden_size/context_in_dim/vec_in_dim/
  depths/guidance_embed from the header, requires the surviving
  candidate to reproduce the ENTIRE layout, and refuses every
  variant lineage loudly: Flux2 (double_stream_modulation_img),
  Chroma/Chroma Radiance (distilled_guidance_layer prefix), Ovis
  (yak gated MLP), txt_norm variants, LongCat-Image (vector-free at
  the 3584 Qwen2.5-VL context width; classic-width headers missing
  vector_in report as truncated), FluxInpaint (widened img_in);
  `normalize_flux_keys` ports the supported_models.py
  scale->weight RMSNorm rename as a pure mapping transform. Torch
  half (dinkster_inference_torch.flux): faithful model.py + layers.py +
  math.py port under the `Operations` seam (LayerNorm affine +
  RMSNorm factories added) - timestep embedding, ND RoPE (float64
  frequencies), QKNorm, Modulation, double/single stream blocks,
  guidance embedding, LastLayer, patchify/unpatchify with the
  circular pad_to_patch_size leg, SDPA attention with
  SDP_BATCH_LIMIT chunking, state-dict keys IDENTICAL to bare BFL.
  RoPE prefers the comfy-kitchen kernel behind a capability probe,
  gated on autograd facts and torch.compile (pure-torch math under
  compile - inductor fuses it; the kernel's Python launcher would
  graph-break) with a device-context pin fixing an upstream
  multi-GPU Triton defect
  (docs/comfyui-issues/comfy-kitchen-triton-apply-rope-wrong-device.md).
  Deliberate strictness: y is required with exactly vec_in_dim
  columns and guidance required iff guidance_embed (the reference
  zero-fills/slices/skips silently). No inference_mode/no_grad,
  out-of-place residuals (training program, 3.1). Goldens executed
  from the pinned reference (tools/gen_flux_goldens.py: full-size
  dev/schnell layouts on the meta device, tiny hash-filled executed
  forwards incl. guidance, guidance-free, and odd-spatial cases;
  forces the pytorch SDPA backend by patching the BOUND global in
  comfy/ldm/flux/math.py and the pure-torch RoPE path via
  in_training; refuses off the audited commit). CUDA-validated on
  2x RTX 4090 (fp32 golden closeness, fp16/bf16, autograd,
  fullgraph inductor compile, worker threads, kitchen-vs-eager RoPE
  parity).
  Slice 7 (SHIPPED 2026-07): the Flux text-encoding stack, native
  classic T5-XXL. Torch-free half: `dinkster_inference.graphemes`
  (Unicode 16 UAX #29 extended grapheme segmentation over a gzipped
  vendored break-property table, proven on the full 1093-case
  conformance file), `dinkster_inference.t5_spm` (pure-stdlib port of
  the T5 SentencePiece unigram pipeline the reference reaches
  through T5TokenizerFast: leftmost-longest added-token extraction,
  the SentencePiece Precompiled-charsmap normalization applied per
  grapheme cluster, metaspace pretokenization, unigram Viterbi with
  unknown-fusion, decode - all over the gzipped vendored
  tokenizer.json byte-identical to the reference's;
  pinned against tokenizers 0.22.2 with curated + 4000-case fuzz
  goldens; `T5_XXL_FLUX_PROFILE` = the reference T5XXLTokenizer
  facts: no BOS, EOS 1, pad 0, pad_to_max_length=False, min_length
  256, max_length effectively unbounded -> one packed chunk),
  `dinkster_inference.t5_text` (classic-T5 config/layout/detection:
  `T5Config`, `t5_layout` exact 219-key state-dict listing plus the
  checkpoints' duplicate encoder.embed_tokens.weight,
  `detect_t5_config` accepting only T5-XXL by full layout
  comparison, refusing UMT5's per-block-bias layout and unknown
  geometries loudly), and `empty_chunk` promoted from clip_text
  into prompt_tokens (shared empty-prompt baseline; vendored-data
  loading unified in `vendored.py` with SHA-256 provenance guards).
  Torch half (dinkster_inference_torch.t5_text): faithful
  comfy/text_encoders/t5.py port under the Operations seam -
  T5LayerNorm with the reference's exact RMS math (variance in the
  INPUT dtype, weight cast to input dtype; deliberately not
  F.rms_norm, whose accumulation dtype diverges in reduced
  precision), Mesh-TF relative-position buckets, UNSCALED SDPA
  (scale=1.0; bit-identical to the reference's k*sqrt(head_dim)
  fold since sqrt(64)=8 is exact), block-0-only bias ownership
  threaded to later blocks as past_bias, gated GELU-tanh / relu
  feed-forward, reference-identical state-dict keys;
  `T5TextEncoder` ports the ClipTokenWeightEncoder policy for
  Flux's T5 (batched chunks + empty baseline when weighted, no
  attention mask, float32 conditioning, pooled=None);
  `compose_flux_conditioning` = FluxClipModel.encode_token_weights
  (T5 sequence + CLIP-L RAW pooled, return_projected_pooled=False).
  New in operations.py: `CastOperations`, the reference manual_cast
  counterpart - parameters keep checkpoint storage dtype, every
  forward casts through the stage-4c cast_weight pipeline to a
  bind-time compute dtype. This is LOAD-BEARING for T5-XXL: fp16
  activations overflow the fp16 RMS variance (rsqrt(inf) zeros the
  output), and the reference runs fp16-stored text encoders at
  fp32 compute for exactly this reason. Goldens executed from the
  pinned reference (tools/gen_t5_text_goldens.py: full-size T5-XXL
  layout on the meta device, tiny hash-filled executed forwards
  incl. relu/non-gated, weighted/unweighted encodes, Flux
  composition; tools/gen_t5_tokenizer_goldens.py for the tokenizer;
  both refuse off the audited commit). CUDA-validated on 2x RTX
  4090 (strict-fp32 goldens both GPUs, fp16/bf16, autograd,
  fullgraph inductor compile, Triton, worker threads, and the REAL
  t5xxl_fp16.safetensors loaded and encoded under
  CastOperations(float32)). Deferred variants (UMT5, SD3's masked
  T5 leg, intermediate-layer capture, other T5 sizes, fp8 T5)
  ledgered in ROADMAP "T5 surface outside the stage-5 slice 7
  port".
  Slice 8 (SHIPPED): classic Flux (dev/schnell) runtime assembly
  with mixed per-parameter dtypes and fp8. Torch-free half:
  `dinkster_inference.quantization` (`split_quantization` scopes a
  weight-source header - optionally under a combined-checkpoint
  prefix - into architecture keys vs quantization keys, decoding
  all three reference spellings: the `_quantization_metadata`
  header JSON, per-layer `.comfy_quant` payload configs, and the
  legacy `scaled_fp8` marker with `scale_weight.*`/`scale_input.*`
  keys; per-layer results are `LayerQuant` records carrying the
  ORIGINAL source scale-key names; per-tensor float8_e4m3fn/e5m2
  are the ported formats, block-scaled/packed formats - mxfp8,
  nvfp4, int8, w4a4 - refuse loudly by name; plain fp8 storage
  WITHOUT scales is deliberately not quantization, it is storage
  dtype handled by cast-at-use) and `dinkster_inference.assembly`
  (`plan_flux_assembly` turns source headers into a typed
  `FluxAssemblyPlan` for the four components - Flux DiT, CLIP-L,
  classic T5-XXL, Flux KL VAE - from a combined checkpoint, split
  per-component files, or any mix with split-over-combined
  precedence; per component: source-key -> model-key map,
  per-parameter storage-dtype inventory (mixed dtypes within one
  model are normal, never a model-global dtype), per-layer
  quantization, and contextual `AssemblyError` failures naming the
  component + source; the regularizer-only real-ae.safetensors KL
  layout - no quant_conv/post_quant_conv - became first-class via
  `KLConfig.quant_convs` with paired-presence validation). Torch
  half: `quant_linear.py` (`Fp8Linear` - scaled-fp8 weights as
  ORDINARY module state: fp8 qdata as a non-trainable Parameter
  plus float32 `weight_scale`/`input_scale` buffers matching the
  comfy_quant layout, so stock state_dict/.to()/_apply()/assign
  loading and compile/autograd/threading all work with no tensor
  subclass or wrapper; always-available dequant compute route plus
  an opt-in `torch._scaled_mm` hardware route
  (`supports_fp8_matmul` capability probe), plain vs scaled input
  handling, and `set_weight`/`stored`/`load_stored` bridging to
  the stage-4c patch/requantization flow) and `assemble.py`
  (`assemble_flux` executes a plan: reads only planned tensor
  slices, per-component INITLESS vs CastOperations selection,
  swaps planned Linears to Fp8Linear BEFORE strict assign loading
  so checkpoint storage dtypes survive per parameter, resolves
  payload-borne quant configs, fills split CLIP-L's absent
  text_projection with identity, refuses unsupported quantized
  non-Linear layers loudly). The fp8-T5 NaN-scrub gate (reference
  t5.py "Fix for fp8 T5 base") rides `T5TextModel.embed_tokens`
  only when the embedding storage is fp8. CUDA-validated on 2x
  RTX 4090 (94 GPU tests; each GPU independently; _scaled_mm vs
  exact quantized-reference per layer for e4m3fn/e5m2 with
  independent input/weight scales; fullgraph inductor compile over
  the _scaled_mm route; worker threads; autograd; REAL
  checkpoints end to end: split scaled-fp8 flux1-dev-fp8-new
  assembled with 266 Fp8Linear swaps + bounded DiT forward through
  BOTH routes (cosine ~0.95 dequant-vs-hardware on random input),
  real ae.safetensors VAE decode, identity-filled CLIP-L encode,
  and combined plain-fp8 flux1-dev-fp8 assembled with fp8
  per-parameter storage preserved + finite T5 encode). Kontext and
  Flux 2 remain outside classic Flux and refuse loudly; NVFP4/
  packed formats, fp8-matmul enablement policy, and kitchen
  scaled_mm_v2 parity are ledgered in ROADMAP.
  Slice 9 (SHIPPED): the classic-Flux denoise/runtime bridge - the
  typed layer between an assembled model and the torch-free
  sampling substrate. NOT a user-facing workflow yet: no node/server
  integration, no prompt->decode->save surface; that is the family
  plugin wiring + stage-6 flip. Torch-free half:
  `dinkster_inference.steps` (`sampling_sigmas` - the KSampler
  set_steps/calculate_sigmas port: denoise-fraction tail trim,
  discard-penultimate correction, the 0.9999 full-denoise
  threshold; `max_denoise` - the pure-noise predicate feeding
  noise_scaling), with the reference's hardcoded
  DISCARD_PENULTIMATE_SIGMA_SAMPLERS name set becoming descriptor
  data (`SamplerDescriptor.discard_penultimate`, set on dpm_2 /
  dpm_2_ancestral; uni_pc variants carry it upstream but are not
  ported). Goldens executed from the pinned reference
  (tools/gen_steps_goldens.py: KSampler.set_steps over
  discrete-SD + Flux-flow spaces x normal/simple/karras x
  euler/dpm_2 x 7 denoise fractions, Sampler.max_denoise probes;
  refuses off the audited commit). Torch half:
  `dinkster_inference_torch.denoise` - `FluxDenoiser` (the executing
  `Denoiser[torch.Tensor]`: BaseModel._apply_model for the FLOW
  path - identity calculate_input, sigma-as-timestep lift kept
  float32, per-call conditioning device/dtype casts, distilled
  guidance embedding (dev default 3.5, refused on schnell) kept
  separate from external CFG, cond/uncond batched into ONE forward
  when token counts match with two-forward fallback otherwise,
  cfg_needs_uncond/cfg_combine reuse, singleton-conditioning batch
  broadcast), `prepare_noise` (reference CPU-float32 draw + cast,
  private generator instead of global manual_seed),
  `GaussianNoise` (default_noise_sampler incl. the CPU seed+1
  bump), `latent_process_in/out`, and `run_denoise` (KSAMPLER.
  sample + CFGGuider.inner_sample: float32 sampling state, the
  empty-latent process-in guard, noise_scaling at sigmas[0] with
  max_denoise, solver drive, inverse scaling, process-out;
  brownian NoiseKind refused loudly at ship time - lifted by the
  brownian slice below). No hidden no_grad/inference_
  mode - autograd flows (training seams stay open). Residency/
  placement stays caller-owned. CUDA-validated on 2x RTX 4090
  (each GPU independently + both visible): real split scaled-fp8
  flux1-dev assembled and driven through sampling_sigmas +
  FluxDenoiser + euler for real steps on device. Deferrals
  (noise_inds batch-skip, CONDCrossAttn repeat-to-lcm concat,
  node/server integration) ledgered in ROADMAP
  "Denoise bridge scope outside the stage-5 slice 9 port"; the
  brownian-tree deferral shipped as the next slice.
  Brownian slice (SHIPPED): brownian-tree step noise -
  `dinkster_inference_torch.brownian` (`BrownianTreeNoise`), the
  native BrownianTreeNoiseSampler/BatchedBrownianTree (cpu=True)
  port over full dependency-free transcriptions of numpy's
  SeedSequence entropy mixing and torchsde 0.2.6 BrownianInterval
  in the exact configuration ComfyUI's BrownianTree pins
  (halfway_tree, tol=1e-6, pool_size=24, levy area "none").
  Bit-identical noise streams with neither numpy nor torchsde in
  the validation environments, golden-pinned against the EXECUTED
  reference stack (tools/gen_brownian_goldens.py: SeedSequence,
  torchsde.BrownianTree, ComfyUI BatchedBrownianTree +
  BrownianTreeNoiseSampler @ 947c2749). run_denoise now
  default-constructs it for NoiseKind.BROWNIAN with the reference
  SDE solvers' bounds (min positive sigma, max sigma) and grows
  the reference's `noise_sampler` override (needed for bit-exact
  SNR-offset flow schedules, whose pre-offset tree bounds the
  offset schedule can no longer express) - the already-ported
  dpmpp_sde/dpmpp_2m_sde/dpmpp_2m_sde_heun/dpmpp_3m_sde
  descriptors now drive end to end. Within-slice deferrals
  (per-batch-item seed lists, the GPU-resident cpu=False tree)
  carry revival triggers in brownian.py and the ROADMAP ledger.
  Family wiring slice (SHIPPED 2026-07-26): the stage-6 runtime
  seam - torch-free `dinkster_inference.runtime` (`probe_native`
  capability probe: family detection remains header-only; assembly
  planning may read explicitly modeled scalar configuration tensors,
  never model weights, and is deterministic for stable source contents;
  `NativeCapability`,
  `NATIVE_WIRED_FAMILY_IDS`, the `FamilyRuntime` protocol:
  encode_text / sample / decode_latent / encode_content + family +
  runtime_identity) and its torch realization
  `dinkster_inference_torch.wiring` (`FluxRuntime`, `load_runtime`
  behind the probe gate, the loader table test-pinned to the
  torch-free claim, `runtime_identity` cache-rotation string over
  canonical plan/knob lines). A native prompt->encode->sample->
  decode workflow now runs end to end behind the seam. The
  repeat-to-lcm cond-batching deferral was investigated at its
  trigger and re-deferred: the reference's Flux uses CONDRegular
  (equal shapes only) for cross_attn - repeat-padding is not
  output-neutral under Flux's joint attention; the optimization
  has no native consumer until a CONDCrossAttn family (SD1/SDXL
  UNet) wires (ROADMAP "Denoise bridge scope", item (c)).
  Next stage-5 slice: SD1/SDXL family runtimes behind the same
  seam, or further codec families as stage 6 needs them; stage-6
  phase 2 (execution host + dispatch) is backend-owned.
- **Stage 6 - flip the nodes (SD arm shipped 2026-07-27).** The compat
  pack's same-session `native` arm now switches `dinkster.load_checkpoint`,
  `clip_text_encode`, `ksampler`, and `vae_decode/encode` from
  worker-resident Comfy calls to `dinkster_inference_torch` for SD1.5, SDXL,
  and SDXL refiner; every refusal stays on the unchanged compat body. The
  arm asserts the host-computed runtime identity, uses Comfy-compatible
  conditioning/latent/image wire shapes, and temporarily places every
  assembled module in full on cuda:0. Flux remains compat-routed until
  native residency/offload can fit its diffusion model plus T5-XXL on a
  24 GiB target. This first flip intentionally does not enroll
  ResidencyManager: three resident handles share one worker-lifetime
  runtime, carry no native VRAM cost/reservation, and cannot be unloaded by
  the Comfy pool hook. The arm-blind planner still applies the conservative
  `CHECKPOINT_RAM_FACTOR = 2.6` host-RAM reservation. ROADMAP "Native
  runtime residency and arm-aware planning" owns metadata, admission,
  unloading, ResidencyManager enrollment, and arm-aware planning in the
  next memory slice. The native basics porting program continues in
  parallel throughout - new `dinkster.*` schema ports remain worthwhile
  because they own the graph contract even when a body remains
  compat-backed.

Priority rationale: stages 1-3 need no GPU-heavy work and unlock the
highest-leverage external win (registerable samplers/schedulers). Stage 4
is the deepest engineering; stage 5 rides on 4; stage 6 is mechanical per
family once 5 exists.

### 3.1 Training (program goal riding the same substrate)

Dinkster training support is a first-class program goal, not an appendix:
state-of-the-art training for every family Dinkster supports natively, at
parity with the reference trainers - kohya-ss/sd-scripts and
ostris/ai-toolkit - in effectiveness, performance, and multi-GPU
support. ComfyUI's training nodes are severely undercooked; this is a
place Dinkster leapfrogs rather than ports. (Source: user directive,
2026-07.)

Training is deliberately noted HERE, before stages 2-5 ship, because it
consumes the same substrate and the contracts must not bake in
inference-only assumptions:

- A trained LoRA IS a PatchSet; LoRA/LoKr/LoHa/DoRA trainers produce
  WeightAdapter-family artifacts. The patch algebra is shared, not
  duplicated.
- Family detection, ComponentWiring, WeightSource selective loading,
  PrecisionPolicy, and the memory governor all serve training runs the
  same way they serve inference.
- Seams stages 2-5 must keep open: the Ops protocol (stage 4-5) must
  not assume no-grad/eval-only execution; the residency manager must
  admit optimizer state and gradients as governed consumers (not just
  weights); the job model needs long-running, checkpointable,
  resumable jobs; the workers layer must not assume one device per job
  (DDP/FSDP-style multi-worker collectives).
- Parity checklist to scope the eventual stages: adapter breadth +
  full fine-tune, EMA, latent/text-embed caching, aspect-ratio
  bucketing, modern optimizers (incl. 8-bit and Prodigy-style),
  quantized-base + block-swap low-VRAM training, timestep/flow-matching
  sampling strategies per family, masked loss, min-SNR-style weighting,
  in-run sample generation, and multi-GPU scaling. Family registry
  makes per-family training support data + adapters, not forked
  scripts.

Training stages are NOT numbered yet: they slot in after stage 5 gives
native forward passes (a training-contracts slice can land alongside
stage 4's weights/patches work). The ROADMAP "Native inference" ledger
carries the revival trigger. sd-scripts and ai-toolkit join the watch
map below as external reference repos.

## 4. What this does NOT change

- The compat pack's contract and isolation model are untouched; it gains a
  second role (oracle) and eventually loses its "only execution path" role.
- No schema wire changes are implied by stages 1-4. Stage 3's registry
  feeds the existing choices/remote-COMBO machinery. Any wire-visible
  change coordinates with the frontend thread as usual.
- dinkster-memory remains the policy seat; nothing here duplicates it.

## 5. Upstream tracking (continuous)

ComfyUI keeps improving: new model families, sampler/scheduler additions,
memory-management refinements, quant formats, dtype heuristics for new
hardware. Replacing ComfyUI does not mean freezing at the audit commit -
relevant upstream improvements get ported into the native stack. This is
a standing obligation, not a stage.

**Reference checkout.** The workspace's `../ComfyUI` clone is the
reference source; pr-tracker watches the repo for PR/commit activity.
This plan's audit baseline is 947c2749; each ported piece records its own
baseline per cross-cutting decision 7, so different areas may track
different upstream commits without confusion.

**Review procedure.** When picking up any native-inference slice, first
diff the watched modules for that area since their recorded baseline
(`git -C ../ComfyUI log --oneline <baseline>.. -- <paths>`) and fold in
relevant changes as part of the slice. Additionally, upstream additions
of whole model families (new `comfy/ldm/<family>/` + `supported_models.py`
entry) are candidates for family-registry registrations once stage 2
exists - they arrive as data + one adapter, not core edits.

**Watch map** (Dinkster area -> ComfyUI reference paths):

| Dinkster area (stage) | Watch in ComfyUI |
| --- | --- |
| Memory/residency mechanism (4) | `comfy/model_management.py`, `comfy/memory_management.py`, `comfy/pinned_memory.py`, `comfy/model_prefetch.py`; comfy-aimdo repo |
| Patches/LoRA (4) | `comfy/model_patcher.py`, `comfy/lora.py`, `comfy/lora_convert.py`, `comfy/weight_adapter/`, `comfy/float.py`, `comfy/hooks.py` |
| Detection + families (2) | `comfy/model_detection.py`, `comfy/supported_models.py`, `comfy/supported_models_base.py`, `comfy/model_base.py`, `comfy/latent_formats.py`, `comfy/model_sampling.py` |
| Loading/ops/quant (4-5) | `comfy/sd.py`, `comfy/ops.py`, `comfy/quant_ops.py`, `comfy/utils.py`; comfy-kitchen repo |
| Sampling (3) | `comfy/samplers.py`, `comfy/k_diffusion/sampling.py`, `comfy/extra_samplers/`, `comfy/sample.py`, `comfy/sampler_helpers.py`, `comfy/conds.py`, `comfy/context_windows.py` |
| Text encoders (5) | `comfy/sd1_clip.py`, `comfy/text_encoders/`, `comfy/clip_model.py` |
| VAE/codecs (5) | `comfy/sd.py` (VAE class), `comfy/taesd/`, `comfy/ldm/` VAE modules |
| Diffusion cores (5) | `comfy/ldm/modules/diffusionmodules/openaimodel.py`, `comfy/ldm/modules/attention.py`, `comfy/ldm/modules/diffusionmodules/util.py` |
| New architectures (2/5) | `comfy/ldm/<family>/` additions paired with `supported_models.py` / `latent_formats.py` entries |
| Training (post-5, section 3.1) | ComfyUI training nodes (`comfy_extras/nodes_train.py`); external reference repos: kohya-ss/sd-scripts, ostris/ai-toolkit (technique + parity source, not ported code) |

**comfy-aimdo and comfy-kitchen are not exempt from the mission.** Today
they ride as optional backends behind Dinkster interfaces (aimdo behind the
governor/residency seam, kitchen behind the Ops/quant protocol). Those
interfaces are deliberately backend-shaped so that if we find
opportunities to improve on them specifically for Dinkster - e.g. a
residency/offload engine designed around the governor's cross-process
reservations and leases rather than aimdo's process-local VBAR policy, or
kernels tuned to Dinkster's typed dtype/layout contracts - a native
equivalent can replace them without touching callers. If that happens,
the replaced library moves to the same status as the `comfy` library:
watched reference and idea source (their rows above stay in the watch
map), no longer a dependency.

Not every upstream change is worth porting: bug-for-bug compatibility is
explicitly not pursued (DESIGN section on compat), and upstream
*architecture* changes (new globals, new hidden module state) are ported
as their underlying algorithm/policy change only. When an upstream change
is deliberately skipped, note it in the relevant module docstring next to
the baseline commit so the skip is a decision, not an oversight.
