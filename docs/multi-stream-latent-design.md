# Multi-stream latent architecture

## Goal

One latent contract and one sampling surface must cover ordinary tensors and
joint latent streams. MiniMax H3 is the first migration target. LTXV-class AV
models must be able to use the same contract without adding another sampler or
another family-specific latent type.

The reference is ComfyUI at `b323a345bbbfb2f3a95b5b73b68eb7919a26515e`.
It places H3 video and audio tensors in one ordinary `LATENT` value, prepares
noise for each stream, packs the streams for stock samplers, restores them for
callbacks and output, and keeps H3 audio schedule conversion in the model. The
relevant source is `comfy/nested_tensor.py`, `comfy/sample.py`,
`comfy/samplers.py`, `comfy_extras/nodes_minimax_h3.py`,
`comfy_extras/nodes_lt.py`, `comfy/latent_formats.py`, and
`comfy/model_base.py`.

## Decisions requiring approval

1. `comfy.LATENT` remains the only graph type for sampled latent data. A
   multi-stream value is an ordered structural payload inside that type, not a
   new graph type and not a family ID disguised as a type.
2. Solvers continue to receive one flat tensor and one scalar sigma schedule.
   The runtime packs before the solver and unpacks at the model, mask, callback,
   and result boundaries.
3. Stream-specific schedule conversion is model policy. H3 owns audio scaling,
   audio sigma conversion, and audio velocity conversion. The sampler engine,
   scheduler registry, and solver registry remain family-neutral.
4. `dinkster.ksampler` is the only new-workflow sampling node. The dedicated H3
   sampling implementation and node are deleted. Unreleased Dinkster H3 workflows
   are not migrated or kept executable.
5. Family-specific validation moves from graph type unification to the
   per-component execution boundaries. This deliberately relaxes graph proofs
   for H3 latent inputs and outputs, H3 conditioning family, conditioner/DiT
   compatibility, and whether a generic MODEL consumer supports an H3 model
   handle. Generic `LATENT`, `CONDITIONING`, and `MODEL` links can therefore
   reach the wrong combination, but the selected component or consumer must
   reject it before drawing noise, entering a solver, or decoding.
6. The implementation is one issue, one implementation branch, and one review
   unit. The ordered work below is an integration sequence, not independently
   accepted feature slices.
7. H3 initial noise changes to ComfyUI's per-stream draw order before packing.
   Current Dinkster draws once over the already packed shape, so existing H3 seeds
   will produce different outputs. This is a deliberate parity correction and
   requires the runtime identity and cache rotation described below.

## Canonical latent value

### Graph envelope

The graph value remains the ordinary latent mapping:

```text
{
  "samples": Tensor | MultiStreamLatent[Tensor],
  "noise_mask": Tensor | MultiStreamLatent[Tensor] | absent,
  ...opaque compatible LATENT metadata
}
```

Existing single-tensor latent mappings are unchanged. `MultiStreamLatent` is a
small torch-free structural value in `dinkster-inference`, parameterized by the
backend tensor type. It contains a nonempty ordered tuple of `LatentStream`
records. Each stream has only:

- a stable semantic role such as `video`, `audio`, or `camera`;
- its tensor payload.

Roles are unique within a value. Order is significant and canonical for pack,
noise, fingerprint, and callback behavior. Concat nodes establish order;
family runtimes validate it. The container carries neither model family ID nor
latent format descriptor. Those facts belong to the selected model. Their
omission is intentional: a joint AV latent can be split, have one stream
replaced, and be consumed by another compatible AV family without lying about
its origin or copying model policy into data.

The container exposes only structural operations needed by sampling and compat
adapters: enumerate/unbind, map, topology comparison, and replacement by role.
Torch packing, allocation, interpolation, and device movement remain in
`dinkster-inference-torch`.

### Pack layout

The torch adapter validates all streams before packing:

- exact dense strided floating tensors;
- positive dimensions and a common batch size;
- one device and one dtype for a single solver invocation;
- family-declared channels and content rank matching each tensor;
- unique roles and the family-required role order.

It records an immutable `LatentPackLayout` containing each role, original
shape, flattened length, and ordered offset. Each tensor is
reshaped to `[batch, 1, elements]` and concatenated on the final axis. Unpack
requires exact batch, middle width, and total length before restoring shapes.
This generalizes the current H3 `[B, 1, total]` representation without making
H3's two shapes part of the generic type.

The layout is invocation state, not mutable value metadata. Stable topology
facts are included in the latent codec and fingerprint; temporary offsets can
always be derived from shapes.

### Codec and compatibility representation

`comfy.LATENT` receives a declared structural codec instead of relying on a
pickle of an arbitrary Python object. Its frame has a fixed magic value, codec
version, and `single` or `multi` kind. The payload preserves the latent mapping,
ordered stream roles, tensor dtype and shape, tensor bytes, and matching mask
structure. Mapping keys are sorted UTF-8 strings; metadata is recursively
limited to canonical primitives, lists, tuples, mappings, and registered value
forms. Unsupported metadata fails at the boundary rather than falling back to
pickle. Compatible mapping keys round-trip and do not affect sampling unless
their existing contract says they do.

Host, native worker, and compatibility worker register the identical codec and
`validate_encoded` function. Validation checks framing, version, kind, tensor
records, metadata forms, and multi-stream topology before cached bytes are
attached to a value. A canonical descriptor covers the type ID, frame magic and
version, metadata grammar, tensor record grammar, validator version, and role
sidecar schema. Its SHA-256 digest is advertised in each worker capability
manifest and pinned in the execution plan; host/worker negotiation rejects a
mismatch before dispatch.

Every encoded or in-memory boundary value carries the frame version. The
implementation runtime-identity change invalidates old LATENT and
`MiniMaxH3AVValue` cache entries; no legacy decoder or cache migration path is
provided. New single-stream execution remains numerically identical. Encoded
fingerprint rotation is accepted.

The codec preserves the raw structural mask. Normalization is sampler
invocation state and does not rewrite the cached value. Fingerprints hash the
canonical encoded raw mask, so differently expressed but behaviorally
equivalent masks may retain different cache identities.

The ComfyUI worker boundary converts between `MultiStreamLatent` and
`comfy.nested_tensor.NestedTensor`. Conversion occurs at LATENT ingress and
egress, not in solvers and not in model implementations. This lets translated
ComfyUI nodes continue to see their native object while native Dinkster nodes see
the canonical structural value.

Because `NestedTensor` carries only position, the compatibility mapping also
contains a reserved `_dinkster_multistream_roles` sidecar with the form
`{"version": 1, "roles": [...]}`. It is added and validated by the worker
boundary and removed when rebuilding the canonical native value. The sidecar
count must match the nested tensor and its roles must be unique. It is excluded
from opaque metadata merge rules. Stock KSampler preserves the outer mapping
and thus preserves the sidecar even when it replaces `samples`.

Translated nodes declare one of four output capabilities: preserve topology
from one named LATENT input, produce a fixed role tuple, combine named inputs by
an explicit role rule, or do not support multi-stream values. A preserve
declaration selects exactly one source sidecar even when the node has multiple
LATENT inputs; planning rejects an absent or ambiguous source declaration. The
boundary restores that sidecar if the translated node rebuilds only the outer
mapping, then verifies that output stream count is unchanged. A combine
declaration names the input slots, accepted role tuples, output roles, and
metadata precedence.

Every named LATENT input separately declares `single_only`, `any_structural`,
an exact set of accepted role tuples, or `same_topology_as` another named input.
Dispatch validates all present sidecars and relational constraints before node
execution. A node with multiple LATENT inputs cannot use an unqualified
node-wide capability. Output preserve/combine declarations may reference only
inputs whose declarations accept the observed topology; fixed-role producers
declare their output roles even when they have no LATENT input.

The boundary updates the sidecar only from those declarations, never from
count, order, or shape. An undeclared producer that returns a `NestedTensor`, a
transformer whose output contradicts its source sidecar, or a role-changing
node without a fixed or combine declaration fails at egress before any
destination node runs. Dispatch rejects a multi-stream input before executing a
node that does not declare a compatible input capability. Stock KSampler
declares topology preservation from its latent input.

## Standard sampling flow

`FamilyRuntime.sample` is generalized from `Tensor` to
`Tensor | MultiStreamLatent[Tensor]`. Its implementation follows one path:

1. The selected runtime validates the latent topology and conditioning against
   its family contract. Validation happens before noise generation.
2. The runtime applies per-stream latent `process_in` transforms, packs the
   result, and applies the empty-latent guard once to the complete packed value,
   matching ComfyUI. A zero-valued stream is not independently short-circuited.
3. Noise preparation walks streams in canonical order with one CPU generator.
   Each stream draw uses its native shape, matching ComfyUI's sequential draw
   behavior. Noise is represented with the same topology, then packed with the
   latent layout.
4. A family sampling adapter returns the primary sigma space,
   `Parameterization`, and a denoiser over the flat pack. Registry lookup,
   denoise truncation, SNR offset, Brownian bounds, and solver construction stay
   in the ordinary engine.
5. The generic engine applies parameterization noise scaling to the flat pack
   and invokes the registered solver. Solvers remain unaware of stream roles.
6. For every model evaluation, the family adapter unpacks the current state,
   applies stream-specific model input policy, invokes the model, converts the
   model result to the primary parameterization, and packs the denoised result.
7. At each solver's existing callback point, the engine presents callbacks with
   an unpacked private snapshot of the current state and most recent denoised
   prediction. The solver state remains packed and unshared with callbacks.
8. The engine applies inverse noise scaling, unpacks, applies per-stream
   `process_out`, preserves ordinary latent mapping metadata, and returns
   `comfy.LATENT`.

Single-stream sampling uses the same control flow with an identity pack adapter
and must remain bit-identical. It must not gain an extra random draw, reshape,
copy, dtype conversion, or arithmetic operation.

### Noise and stochastic samplers

Initial noise is drawn per stream before packing because differently shaped
streams cannot be reconstructed from one random draw over the flat shape while
preserving ComfyUI's generator sequence. Step noise is generated over the
already packed solver state, as it is in the current H3 private path.

`batch_index` and `noise_inds` are one invocation-level index sequence applied
to every stream in canonical order. Initial noise uses one shared CPU generator;
each native-shape stream draw consumes that generator and applies the same index
selection semantics before packing. A descriptor-owned alternate inpaint noise
stream is separate from initial nested noise. ComfyUI's random inpaint path,
including DDIM's seed-plus-one stream, draws once over the already packed noise
shape and applies packed indexing semantics. Dinkster preserves that packed draw
exactly. Family adapters cannot introduce private noise indexing or seed
streams.

Brownian tree bounds remain based on the primary pre-offset schedule. The tree
sees the flat pack shape and needs no stream knowledge. A future family that
needs independent per-stream stochastic schedules is not represented by this
contract and must fail rather than silently reuse the primary tree.

### Masks

A multi-stream noise mask follows ComfyUI normalization on both arms:

| Input mask | Per-stream interpretation |
| --- | --- |
| absent | no packed mask and no masking |
| singular tensor | role 0 uses the tensor; later roles use all ones |
| structural mask with missing roles | present roles are used; missing roles use all ones |
| structural mask with extra trailing roles | extras are discarded |

Duplicate semantic roles are invalid. Each selected mask may have lower rank
than its latent. It is reshaped and resized with linear, bilinear, or trilinear
interpolation according to latent content rank, cyclically repeated or
truncated to the latent batch, expanded across channels as required, and then
flattened with its stream's pack layout. The final mask is `[B, 1, total]`.
These rules are shared implementation, not stricter native-arm policy.

The generic inpaint wrapper then operates on the packed mask, packed source
latent, and packed noise. Family-specific inpaint conditioning remains a
separate capability. H3 initially supports structural noise masks but continues
to reject unsupported generic control or inpaint-conditioning payloads.

### Sampling state callbacks and previews

The public progress callback remains source-compatible with `StepEvent`.
`ProgressScope` adds `on_state: SamplingStateCallback | None` beside `on_step`,
and `SamplingExecutionContext.progress` remains the only registration and
lifetime owner. `SamplingStateCallback` receives `SamplingStateEvent`; no global
callback registry is introduced.

The solver-to-engine seam replaces its scalar-only internal report with a
`SolverStateEvent` at each solver's existing callback point. It contains the
step index, total steps, sigma, callback phase (`pre_update` or `post_update`),
packed current state, and optional most recent packed denoised prediction. Each
solver pins callback count, phase, and tensor meaning against the corresponding
ComfyUI solver. Multi-evaluation solvers report only at their existing public
callback point, not after every denoiser evaluation.

The engine unpacks those tensors with the invocation's immutable
`LatentPackLayout` and emits an additive `SamplingStateEvent` to preview-aware
consumers. Its `current` and optional `denoised` fields have the same structural
topology as the input. When `on_state` is present, the engine clones and detaches
the packed current and denoised tensors before unpacking them; the event owns
those private snapshots. A callback may retain or mutate them without sharing
solver storage. No snapshot allocation occurs when only ordinary progress is
registered. The existing progress adapter derives the unchanged `StepEvent`
from the same solver event. `ProgressScope` checks cancellation before
callbacks, invokes `on_state` then `on_step`, and checks cancellation again. A
callback exception propagates through ordinary sampler cleanup; no later
callback is emitted after cancellation, failure, or a callback exception.

The compatibility worker maps ComfyUI's callback dictionary into the same
`SamplingStateEvent` at ComfyUI's callback point, using the role sidecar and
invocation layout to unpack `x` and `denoised`. Native and compatibility
conformance fixtures pin event count, phase, tensor lifetime, cancellation
ordering, and progress adaptation for every registered solver.

### Parameterization and family facts

`SamplingDescriptor` continues to own the primary prediction
parameterization. The scheduler registry continues to build one primary sigma
schedule. `SamplerInfo.parameterization` remains a scalar fact because the
solver updates one packed state.

The family sampling adapter owns facts that map streams onto that state:

- required roles and their descriptors;
- canonical pack order;
- per-stream latent processing;
- conversion from the primary sigma to model-native stream sigmas;
- conversion of model outputs back to the primary denoised prediction;
- supported mask, conditioning, guidance, and scheduled-patch capabilities;
- the primary stream used by default previews.

For MiniMax H3, the primary stream is video and the primary parameterization is
FLOW. The video flow schedule drives the solver. Before packing, audio is
carried at `video_shift / audio_shift`. At each model evaluation, the H3
adapter derives audio sigma and audio state/velocity factors from the video
sigma, exactly where the current H3 DiT already does so. The adapter reverses
the carry after sampling. None of these facts enter `dinkster.ksampler`, a
scheduler, or a solver.

## `dinkster.ksampler` and execution arms

The node schema remains the ordinary model, positive, negative, latent, seed,
steps, CFG, sampler, scheduler, and denoise surface. It does not add stream,
audio, or H3 inputs.

The native arm accepts a canonical multi-stream LATENT, resolves the selected
model's family adapter, and calls the generalized `FamilyRuntime.sample` or
scheduled equivalent. The compatibility arm converts the canonical structure
to `NestedTensor` and invokes stock ComfyUI KSampler. Both arms convert their
result back to the same LATENT representation.

H3 conditioning becomes ordinary `comfy.CONDITIONING` carrying a typed H3
payload and runtime identity in metadata. Each H3 conditioning node's final
schema outputs ordinary positive and neutral negative CONDITIONING values for
KSampler. The selected role-specific DiT accepts only its compatible positive
payload, matching target layout and runtime identity. H3 remains positive-only
at first: `cfg` must be 1, and negative conditioning must be neutral. CFG++ and
a model-evaluated unconditional lane fail before sampling. Future proven H3
guidance can widen that runtime capability without another node.

Dispatch selection remains based on resident model provenance and policy, not
on latent shape or family guesses. A native H3 DiT handle selects the native
`dinkster.ksampler` arm; a ComfyUI model selects the compatibility arm.

## Generic AV nodes and previews

The generic node surface uses only `comfy.LATENT`:

- `dinkster.concat_av_latent(video_latent, audio_latent) -> latent` creates the
  canonical `(video, audio)` topology. The first input may be single video or
  exactly `(video, audio)`; any other role or extra stream fails. Existing AV
  input replaces audio. It normalizes masks by the table above before fitting,
  then may trim or zero-pad exactly one differing audio content axis to the
  existing audio shape; channel, batch, or rank differences fail. A padded mask
  tail is one so it is generated. Output metadata starts with the video mapping
  and overlays the complete audio mapping, so audio wins on every nonreserved
  collision; samples, mask, and role-sidecar keys are rebuilt structurally.
- `dinkster.separate_av_latent(latent) -> (video_latent, audio_latent)` requires
  exactly those roles and returns ordinary single-stream LATENT mappings. It
  duplicates unrelated mapping metadata into both outputs and assigns each its
  matching samples and normalized mask.
- `dinkster.preview_latent_visual(model, latent, role) -> comfy.IMAGE` resolves a
  selected visual role through the model's stream descriptor and registered
  latent RGB preview provider. Its graph output is always IMAGE and it rejects
  a nonvisual role or a family without that renderer.
- `dinkster.preview_latent_audio(model, latent, role) -> comfy.AUDIO` resolves an
  audio role through a registered family codec preview decoder. Its graph
  output is always AUDIO and it rejects a role without an audio renderer. Raw
  latent bytes are never labeled as waveform samples.

Sampling previews use the same visual resolver from `SamplingStateEvent` and
emit the existing UI preview event; they do not invoke a graph preview node or
an audio decoder during sampling.

The canonical names are not LTXV- or H3-prefixed. Compatibility aliases for
ComfyUI's `LTXVConcatAVLatent` and `LTXVSeparateAVLatent` remain registered for
saved workflows. The concat alias selects positions 0 and 1 from an existing
nested first input and discards later positions, overlays the complete audio
mapping over video, and reproduces ComfyUI fitting, trim, pad, and mask behavior
exactly. This includes preserving ComfyUI's failure when already-nested samples
are paired with a singular mask that cannot be unbound. The separate alias
selects positions 0 and 1 and duplicates unrelated mapping metadata into both
outputs. These legacy adapters then attach canonical `(video, audio)` roles.
The family-neutral nodes instead use the normalized mask table and retain the
stricter role and extra-stream checks above.

## H3 node surface

These dedicated nodes remain because they own H3 facts rather than generic
sampling mechanics:

- H3 empty latent creation and temporal/canvas alignment;
- H3 image, video, and audio reference construction;
- T2VA, FL2VA, and REF2VA conditioning;
- H3 video/audio encode and decode codecs;
- an optional H3 model-sampling settings node if video/audio shifts remain
  user-adjustable.

Generic Load Diffusion Model exposes each role-specific DiT as an ordinary
`MODEL`; Load CLIP and Load VAE expose the conditioner and codecs independently.
Dedicated H3 conditioning and codec nodes consume the required component
handles and validate their H3 capabilities before work. The separate license
authorization input and node are assumed absent after issue #94 and are not
part of this design.

The ordinary model output does not transfer weight residency policy to the
latent or sampler. Each component handle owns the residency needed for its
operation. Moving H3 behind KSampler must not pin unrelated components or
retain packed solver temporaries across conditioning, diffusion, and decode
boundaries.
ComfyUI is the performance floor: like-for-like peak allocated and reserved
VRAM, stage residency, and stage time must meet its behavior. Issue #97 verifies
that target after the rework lands; a measured regression is an implementation
defect rather than a reason to weaken the shared latent contract.

H3's declared FP32 AdaLN storage islands remain FP32 for numerical parity, but
they follow the diffusion component's load, lease, prefetch, and offload
policy. The implementation must not create a second permanent GPU copy or
materialize unrelated weights in FP32 while converting storage to compute
dtype. Component storage dtype, compute dtype, prefetch dtype, and offload
state remain explicit resident-model policy and runtime identity facts rather
than properties of `MultiStreamLatent`.

These H3-only mechanisms dissolve:

- `dinkster.minimax_h3.av` as a graph type;
- the family-private AV pack layout as a public contract;
- the family-private AV sampling facade and its duplicate sampler loop;
- the `dinkster.minimax_h3_sample` node, schema, executor, and registration.

No dedicated H3 or AV latent graph type survives. H3 and LTXV-class models use
the same role-labeled `MultiStreamLatent` inside `comfy.LATENT`.

## Type safety and fail-closed behavior

The following guarantees are retained or strengthened:

- graph edges still carry a registered, closed `comfy.LATENT` type;
- stream containers are immutable, ordered, nonempty, uniquely named, and
  structurally encoded;
- pack and unpack validate exact topology, shape, length, dtype, layout, batch,
  and device before model execution;
- masks are topology-checked and prepared per stream;
- family runtimes validate required roles, latent formats, conditioning kind,
  runtime identity, and feature capabilities before noise or solver work;
- unknown family features, schedules, masks, controls, guidance modes, or patch
  combinations fail explicitly;
- runtime identity and cache identity include every behavior-affecting adapter,
  registry, extension, patch, and topology fact.

Five graph guarantees are deliberately relaxed: type solving no longer proves
that a sampler input latent belongs to H3, that the sampled result is H3 AV,
that conditioning belongs to H3, that the selected H3 conditioner and DiT are
compatible, or that every generic MODEL consumer supports an H3 DiT handle.
Family-specific graph types make the ordinary sampler and generic latent tools
impossible and push every shared feature into duplicate family wiring.

The replacement is one fail-before-noise runtime check against the selected
components. For H3 it proves the exact `(video, audio)` role order, stream ranks,
channels, batch relation, pack layout, conditioning task and target layout,
positive payload type, matching resident runtime identity, neutral negative
conditioning, CFG 1, and support for the requested masks, controls, patches,
guidance, scheduler, and sampler. This is the same boundary at which channel
count, rank, inpaint capability, and conditioning compatibility already become
knowable for ordinary LATENT values.

The resident MODEL advertises an immutable sampling-capability descriptor;
KSampler requires its family adapter before noise generation, while any other
MODEL consumer must validate its own required capability before execution. H3
decode consumers receive the resident handle separately and validate result
roles, ranks, channels, batch, task layout, and codec capability before work;
conditioning consumers also validate their payload's resident runtime identity.
Generic sampled output never bypasses those consumer checks merely because its
graph type is LATENT.

The generic structure does not accept arbitrary sequences as streams, infer
roles from shapes, cast mismatched dtypes, move mismatched devices, or guess a
schedule. Generalization is structural, not permissive.

## Identity and workflow break

### Runtime identity

Implementation changes H3's executed route, initial-noise ownership, callback
once from the value on the rebased implementation base. Same-seed H3 output,
cache identity, and encoded fingerprints may change without migration because
the replaced Dinkster behavior was unreleased. This work must rebase after issue
#92 and regenerate the resulting runtime identities rather than reserve values
in advance. This design-only change does not alter runtime identities.

### Workflows

Unreleased Dinkster workflows containing `dinkster.minimax_h3_sample` or the closed
H3 AV type are intentionally unsupported. No schema alias, hidden executor,
graph rewrite, or prompt migration is implemented. New H3 workflows use the
loader's ordinary MODEL output, generic LATENT/CONDITIONING links,
`dinkster.ksampler` with CFG 1, and the neutral negative conditioning output.

ComfyUI workflow compatibility remains a product contract. The LTXV aliases
and compatibility-arm behavior specified above are retained.

### Cache identity

The versioned LATENT frame fingerprints codec version, mapping keys, ordered
stream roles, shapes, dtypes, bytes, raw structural masks, and compatible
metadata. Reordering streams or replacing one stream changes the fingerprint.
The codec digest and role-sidecar capability registry digest are structural
worker identity facts and must match across every host and worker before
dispatch.

Runtime identity changes rotate native execution cache identity. The H3 family
adapter version is also a structural runtime identity fact. Physical device
IDs, pack buffer addresses, and derived offsets remain outside identity. Old
cache data is discarded. Tests cover new single- and multi-stream frames,
corrupted frames, and cross-worker encode/validate/decode agreement.

## Interaction with concurrent work

### Single-job multi-GPU issue #80

The generalized structure becomes the only latent boundary for these
multi-GPU modes:

- guidance parallelism scatters the same packed state used by a single-device
  solver and gathers packed predictions before rank 0 advances the solver;
- batch sharding applies one ordered batch-index set to every stream, mask,
  conditioning payload, and full-batch noise draw, then restores stream order
  at final gather;
- H3 sequence parallelism keeps pack/unpack and the solver on rank 0 while the
  H3 model adapter distributes attention sequence rows and heads.

Workgroup payload and shard recipe identity include the ordered topology and
layout digest. No multi-GPU mode calls a private H3 sampler. Brownian batch
sharding remains disabled until its full-batch noise authority is proven, as
specified by the multi-GPU design.

### Scheduled convolution LoRA PR #92

Scheduled conditioning and patch carriers enter through the same
`dinkster.ksampler` detection path for single- and multi-stream latents. The H3
family adapter must implement the scheduled runtime seam or explicitly reject
unsupported H3 patch providers before the first sigma. It must not bypass
scheduled execution through a private sample method.

Execution-scoped patch state surrounds the whole H3 denoiser trajectory and is
restored transactionally on success, cancellation, or failure. Stream packing
does not change patch identity. The implementation rebases over #92 so its
convolution support and runtime identity changes are preserved.

### Execution-arm visibility issue #90

H3 sampling becomes an ordinary dual-arm `dinkster.ksampler` execution. Planned
and actual arm receipts therefore come from the same policy decision and cache
entry as every other model. The latent container never selects or labels an
arm.

### Residency issue #79 and license issue #94

The H3 component handles retain staged ownership and per-component memory
accounting. Generic latent values own data, not resident hardware. The design
assumes #94 removes the license gate; no replacement gate or authorization
field is introduced here.

### Performance parity benchmark issue #97

Issue #97 validates the completed rework against ComfyUI on RipperPC with
like-for-like inputs, per-stage timing, and
`torch.cuda.memory._record_memory_history()` timelines. The implementation
defines one torch-free execution-observer protocol in `dinkster-inference`. The
backend attaches an observer to one execution invocation and passes it through
the resident handle and sampler call; it is never process-global. Every event
contains an invocation ID, span ID, optional parent span ID, begin/end phase,
coarse stage (`load`, `condition`, `sample`, `encode`, or `decode`), operation,
host monotonic timestamp, and optional component role, device, storage dtype,
compute dtype, and resident-byte fields. The execution attachment allocates
the invocation ID and concurrency-safe monotonically increasing span IDs.
Emitters receive and pass parent span context explicitly; no thread-local or
ambient parent stack is inferred across concurrent work.

The shared conformance contract fixes the outer stage/operation pairs as
`load/load`, `condition/condition`, `sample/sample`, `encode/encode_video`,
`encode/encode_audio`, `decode/decode_video`, and `decode/decode_audio`. The
resident assembly boundary owns the invocation-level load span; the native arm
owns the remaining outer spans. Within `sample`, the ordinary sampler emits
`prepare`, `model_evaluation`, `step_noise`, and `finalize` operations. The
residency coordinator emits nested `load`, `prefetch`, `lease`, `offload`, and
`release` operations with its existing `text`, `diffusion`, or `vae` component
role; each inherits the enclosing coarse stage. Separating operation from
component role preserves the current shared VAE stage while still
distinguishing video from audio and encode from decode.

Span nesting and names are stable diagnostic API. One shared conformance
fixture defines the enum values, required fields, stage inheritance, nesting,
terminal events, and failure behavior; both inference and backend tests run
against it.

The benchmark harness enables CUDA memory history and consumes those events to
align host-timestamped allocator snapshots with stages; it does not patch H3
nodes, the family adapter, or the solver. When GPU timing is requested, the
CUDA observer records events on the active stream at span boundaries and
synchronizes only once while materializing the report after the entire attached
execution invocation ends, never after an individual operation or between model
evaluations. CPU-only observers use monotonic time. The attachment API owns
observer lifetime and guarantees end events on success, cancellation, or
failure.

The observer is diagnostic only. Disabled execution takes a direct no-observer
branch with no event allocation, CUDA event, or synchronization. Enabled
observer state and measurements do not enter graph, runtime, or cache identity.
Step progress remains the public sampling callback; allocator tracing is not
added to saved workflows or the latent contract.

Issue #97 records one immutable benchmark manifest for both systems: model
artifacts, inputs, seed, sampler, scheduler, step count, dimensions, component
storage and compute dtypes, offload mode, software revisions, and allocator
settings. It includes a small diagnostic workload and a realistic workload
matching the high-memory H3 case. Dinkster runs with Aimdo disabled and enabled;
each is compared only with the equivalent ComfyUI normal or managed-offload
mode.

Each process starts clean, performs one warmup, and then records five measured
runs. Acceptance compares median and median absolute deviation for end-to-end
and per-stage CUDA time, peak active/allocated/reserved bytes, and
component-resident bytes at stage boundaries. Dinkster passes when time is within
5% and each memory metric is no more than `max(2%, 256 MiB)` above ComfyUI. Both
systems retain raw allocator snapshots and rendered timelines. Stack traces,
stage spans, and component roles attribute allocation owners; issue #97 records
material regressions and links dedicated fixes rather than accepting an
unattributed delta.

## Implementation sequence and estimate

Implementation proceeds continuously at agent speed across roughly 28-40
production files and their focused tests, followed by full repository gates and
GPU parity. The largest risks are preserving exact random-number order,
scheduled-patch composition, the cross-worker codec and role sidecar, callback
state lifetime, residency instrumentation, and compat conversion rather than
writing the structural container itself.

The work proceeds on one implementation branch and one PR in this order:

1. Add the torch-free structural latent, versioned LATENT codec with
   `validate_encoded`, translated-node capability declarations, versioned role
   sidecar, topology validation, and ComfyUI boundary conversion. Lock
   single-stream numerical behavior and cross-worker frames with regression
   fixtures.
2. Generalize torch pack/unpack, initial noise, step noise, masks, inpaint
   wrapping, solver state events, callbacks, visual/audio preview schemas, and
   parameterization plumbing. Prove the single-stream path bit-identical and
   mask/callback behavior cross-arm before enabling a family.
3. Move H3 audio carry and sigma/velocity conversion behind a family sampling
   adapter, make the H3 handle implement ordinary sampling, and remove the
   private sampler loop. Prove ComfyUI parity at schedule, initial noise,
   packed state, every denoiser evaluation, and final unpack seams. Preserve
   staged residency, including FP32 AdaLN islands and component dtype/offload
   policy, without duplicate persistent GPU copies.
4. Change H3 producers and conditioning to ordinary LATENT/CONDITIONING,
   expose the model output, route authoring through `dinkster.ksampler`, and delete
   the old H3 sampler node and closed H3 AV type completely.
5. Add generic concat, separate, visual preview, and audio preview nodes plus
   exact LTXV compatibility aliases. Exercise H3 and an LTXV-shaped synthetic
   family so the abstraction is not accidentally H3-only.
6. Integrate scheduled carriers, arm receipts, cache identity, multi-GPU
   topology seams, and the default-off stage/residency observer used by issue
   #97. Add arm-receipt goldens. Regenerate runtime identities and update
   support documentation in the same commit.
7. Run root, torch, CUDA, compatibility, numerical
   parity, cancellation, cache, instrumentation-contract, and full repository
   gates. Prove the issue #97 harness can attach without patches and capture a
   complete synthetic trace. Review the complete implementation as one behavior
   change; issue #97 runs the post-landing like-for-like ComfyUI benchmark.

This design is approved with the amendments above; implementation starts after
the design PR lands.
