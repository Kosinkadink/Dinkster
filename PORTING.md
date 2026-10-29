# Porting a model family into Dinkster

The playbook for bringing a ComfyUI-supported model family to native Dinkster
execution. It names the seams a port touches and the order that works. What
each seam does lives in the code and package READMEs; the execution-identity
rules live in docs/typed-execution-contracts.md. AGENTS.md gates and
disciplines apply to every commit and are not repeated here.

## Before writing code

- File the canonical issue: scope, non-goals (variants excluded from the
  first slice), acceptance runs, and which user-visible capability the port
  advances.
- Work from a fresh clone on a same-repo branch.
- Pin the upstream reference: goldens are generated only on the pinned
  ComfyUI commit (see AGENTS.md numerical parity discipline). Record the
  upstream source file and commit in each ported module's docstring, as
  `dinkster_inference_torch/wan21_vae.py` does.
- Source model artifacts per the standing artifact-sourcing authorization
  and log provenance (URL, size, sha256) where the artifact is pinned.

## The seams, in working order

1. **Detection and admission** (`dinkster_inference/catalog.py`,
   `families.py`, `sources.py`, `weights.py`): a `FamilyDetector` with
   exact tensor-name/shape/dtype evidence, wired into
   `builtin_families()`. Detection must distinguish the new family from
   every family already registered, including lookalike checkpoints that
   share a backbone.
2. **Native model modules** (`dinkster_inference_torch/<family>_*.py`): the
   architecture, built on `operations.py` (INITLESS) and
   `attention.py` `select_attention`, never raw `torch.nn` layer
   construction. One module set per family; do not grow `wiring.py` -
   follow `qwen_image_runtime.py` / `minimax_h3_runtime.py` and give the
   family its own runtime module, touching shared registries only at
   insertion points.
3. **Text encoders** (`dinkster_inference/text_encoders.py`, tokenizer
   modules, `dinkster_inference_torch/*_text.py`): reuse an existing encoder
   port when the family shares one; a new encoder is its own parity
   surface with its own goldens.
4. **Conditioning contract** (`dinkster_inference/conditioning.py`,
   `token_layout.py`, family `*_layout.py`): a `ConditioningEvaluation`
   with declared batchability, and a declared `ModelTokenLayout` plus
   transforms and tensor validator wherever the family's semantic token
   geometry is known. Layout-absent is legal only while the geometry is
   genuinely undeclared, and that choice needs a spec ruling, not a
   default.
5. **Latents and codec** (`dinkster_inference/spaces.py`, `latents.py`,
   `codecs.py`; VAE port in `dinkster_inference_torch`): latent space
   declaration, codec registration, encode/decode parity including any
   temporal or tiled paths the family actually uses.
6. **Runtime integration** (`dinkster_inference/runtime.py`, `scheduled.py`,
   `schedules.py`, `solvers.py`; parameterizations in
   `dinkster_inference_torch`): the generic
   `FamilyRuntime` / `MultiStreamFamilyRuntime` / `ScheduledFamilyRuntime`
   protocols, the family's sigma parameterization, shared CFG and
   sampling. Validate the sampler subset the family actually supports
   against the solver registry; record the matrix in the matching
   `docs/supported/` area rather
   than assuming all solvers work.
7. **Exposure**: package `__init__` exports, loader/admission public
   entries, serve/workflow exposure as applicable.
8. **Evidence**: CPU contract tests; goldens per the numerical parity
   discipline; GPU parity for representative prompts, seeds, and shapes on
   a CUDA machine; codec roundtrip; the cross-implementation parity run
   below; the performance comparison below; the matching `docs/supported/`
   area updated in the

## Cross-implementation parity (user directive, 2026-08-18)

A port is not accepted on Dinkster-internal evidence alone. The same
checkpoint, seed, sampler, schedule, and settings must be run in ComfyUI
native (at the pinned commit) and Dinkster native, and the results must be
identical or every difference understood and recorded. "Looks the same"
is not a verdict. The standard, per seam:

- **Bit-exact, required**: tokenization, schedule sigmas ported onto
  reference kernels, noise stream construction and per-draw values,
  latent packing, and weight/tensor-name mapping. These are discrete or
  kernel-for-kernel ports; any mismatch is a defect, not a tolerance
  case.
- **Value-close, pinned per case**: denoiser forwards, VAE encode/decode,
  and end-to-end sample outputs, where float accumulation order and
  kernel choice differ legitimately. Each case pins its tolerance at the
  override site with the observed drift, the headroom, and the
  amplification mechanism, per the numerical parity discipline in
  AGENTS.md - never a blanket tolerance.
- **Understood-differences register**: each family's parity test module
  carries the register - every known cross-implementation difference
  (RNG stream construction, op ordering, scheduler discretization,
  kernel fusion) named with its cause and its measured effect. A
  difference that cannot be named and bounded is unexplained drift, which
  is an escalation, not a register entry.

Control mechanisms are held to the same standard: strength and schedule
application must match ComfyUI's semantics on the shared surface, and any
deliberate deviation (for example exact realized gain tables versus
upstream per-step float computation) is a register entry stating where
the outputs diverge and why that is correct.

Comparison scope includes custom nodes (user directive, 2026-08-18):
what users expect from ComfyUI is the custom-nodes experience, so when
pinned core ComfyUI lacks a comparable surface but a widely used custom
node provides one, that custom-node implementation is the reference for
the comparison. Pin the exact custom-node commit alongside the pinned
ComfyUI commit and record both in the receipts, as the GGUF comparison
did with ComfyUI-GGUF. First standing assignment: within-step spatial
windowed generation has no core surface, so shiimizu/ComfyUI-TiledDiffusion
(MultiDiffusion / Mixture of Diffusers, with Flux support) is the
ecosystem reference for windowed Flux execution. License caution: a
custom-node reference may carry a non-permissive license
(ComfyUI-TiledDiffusion's core algorithm code is CC BY-NC-SA);
comparing and benchmarking against it is fine, but never port code from
such a reference - reimplement from the declared contract and cite the
comparison evidence only.

## Performance comparison (user directive, 2026-08-18)

Users care most about sampling and end-to-end speed on repeated runs, and
the performance parity discipline in AGENTS.md already makes
correct-but-slower-than-ComfyUI a defect. Every port therefore compares
Dinkster with the latest ComfyUI master available at execution time and records
its exact commit. Numerical goldens remain on the separately pinned ComfyUI
commit. Run both arms serially on the same physical GPU with an identical
workload (checkpoint, resolution, steps, sampler, and batch), and measure:

- **Warm repeated-run steady state**: sampling it/s (or per-step ms) and
  end-to-end wall time per image/video. Report the median and spread over
  several runs after the cold run.
- **Cold first run**: end-to-end time including load, compile, and cache build.
- **Peak memory**: board memory plus torch peak allocated and reserved memory.
- **Numerical stability**: repeated output fingerprints and their within-arm
  stability.
- **Receipts**: exact source commits, GPU model and identity, driver, torch
  version, and attention backend. Measurements without receipts are not
  evidence.

Acceptance requires parity or better, within stated measurement noise, for
warm steady state, cold start, and peak memory. Every measured deficit must be
attributed to the responsible component. It must be fixed before merge or, if
it remains, recorded in a canonical defect issue before or with the port's
merge.

## Model identity and hashing (user directive, 2026-08-18)

- Model content identity is established exactly once, by the asset system,
  at ingest/download. The canonical digest is `blake3:<64 hex>`
  (`dinkster_assets.identity`); it is the only digest type that names asset
  content anywhere in the codebase.
- Loaders and assembly consume that identity. Re-reading or re-hashing a
  model file at load time is a defect, not a safety feature: it is a cost
  ComfyUI does not pay, and on network or slow drives with 30-60GB
  checkpoints it makes Dinkster painful to use. Byte-size and header/structure
  checks stay; full-content hashing does not.
- The one permitted runtime hash of model content is lazy first-time
  ingest: a user-supplied file the asset system has not yet cataloged,
  hashed when its identity is actually needed (for example
  exported-workflow provenance). That path must stream, be cancellable,
  and cache its result through the asset system so it never runs twice.
- Identity digests composed from content identity (resource digests,
  intervention facts, receipts) bind the asset-system blake3 digest plus
  declared load knobs, never a fresh file hash.
- sha256 is legal only outside content identity: publisher-issued
  provenance digests verified at most once at download time, and small
  canonical metadata/structure digests (safetensors headers, mount binding
  state, patch overlays, wheel pins). None of these may be presented as
  asset identity, and none may require reading full model bytes at load.

## When to stop and get a spec ruling

Any of these means the port has left the mechanical path and needs a ruling
on the typed-execution-contracts before implementation:

- sequence/packing semantics or token roles not expressible in the current
  `ModelTokenLayout` vocabulary
- multi-stream or cross-modal ownership the runtime protocols do not
  represent
- unusual CFG or conditioning-fusion semantics
- a new sigma/noise parameterization kind
- new conditioning payload kinds (image embeddings, masks, reference
  frames)
- public API additions of any kind
- novel codec/latent geometry or offload lifecycle behavior

## Merge hygiene

The hot shared files are `wiring.py`, package `__init__.py` export lists,
the family/codec registries, and `docs/supported/model-families-native-execution.md`.
Keep the port's touches to
those files minimal and additive; everything else should live in
family-owned modules so parallel ports do not collide.

Every new or changed maintained Comfy mapping must include a source/native
confidence receipt in the maintainer evidence corpus. CI verifies every
receipt and rejects an increase in the unreceipted-mapping debt recorded in
`docs/comfy-source-parity-baseline.json`; duplicate cases for one mapping do
not increase the parity count. `tools/gen_comfy_source_parity_receipts.py`
owns the receipt tree and CI regenerates it from pinned sources and committed
raw execution evidence. Structurally fail-closed mappings are reported as
refusals and excluded from parity counts; receipts for them are forbidden.
Direct native aliases outside the maintained mapping registries need a pinned
source-generated golden and replay test, but operation evidence must not be
presented as model-family parity.

Dinkster-only development scaffolding, remote partner API nodes, and schema-only
training session nodes have no meaningful local ComfyUI execution counterpart.
Their explicit baseline dispositions do not claim equivalence.
