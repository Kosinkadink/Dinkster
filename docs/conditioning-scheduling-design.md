# Conditioning and scheduling design

This document defines the boundaries that keep conditioning declarations,
transport, compilation, encoding, and sampling independent.

## Conditioning records

`ConditioningRecord` is the immutable unit of conditioning. A record contains
channel payload descriptors and optional area, mask, percent range, condition
scale, token layout, and namespaced extension metadata. `ConditioningSet` is
an immutable ordered collection with clone, combine, and regional composition.

The existing encoded-text `Conditioning` type remains distinct. Inpaint
conditioning also remains a separate runtime concept. The record model is
torch-free; tensor conversion and model-family interpretation belong to the
runtime that executes a record.

### Channels

The carrier recognizes these channel identities:

- `text`
- `pooled`
- `concat_latent`
- `control_hint`
- `reference_latent`
- `camera`

Recognizing a channel does not claim execution support. Runtimes reject
carried-only channels that they cannot execute.

### Percent ranges

Ranges contain finite normalized percentages satisfying
`0.0 <= start <= end <= 1.0`. Percent-to-sigma conversion occurs only when a
runtime with an explicit sigma space materializes the record. The interval is
closed at both ends, so a zero-width range is active at its exact converted
sigma. Disjoint intersection produces the explicit `EMPTY_RANGE` value rather
than a synthetic percentage pair.

### Composition

Combining sets concatenates their records in order. Cloning preserves
immutable payload descriptors and copies metadata. Regional composition
replaces area and mask metadata while retaining range, condition scale, token
layout, and extension metadata.

Extension keys use `pack_id/key` identities. Their values use the wire-clean
value vocabulary plus declared payload references. Unknown extension metadata
is preserve-only and participates in behavior identity.

Area descriptors use named spatial extents and offsets in either latent-cell or
percentage units. Percentage descriptors may also carry temporal extent and
offset fields for video regions. Mask descriptors bind a payload, strength,
and set-area-to-bounds behavior. Records describe these operations but do not
interpolate or crop tensors.

Condition scale uses the shared `ConditionScaleVector` contract. Token layout
descriptors are versioned model-family facts that define streams and named
segments. Unsupported family and layout combinations fail explicitly.

## Conditioning carrier

The `dinkster.conditioning` value type uses the
`dinkster-conditioning-carrier-v1` format. It transports records and payloads
across node output, worker, cache, and RPC boundaries without importing torch.
Non-decoding processes may relay the encoded payload opaquely.

### Payload binding

Every reachable payload reference must have one `PayloadBinding` containing
its shape, dtype, space, and bytes. Carrier construction rejects missing
bindings, unknown bindings, conflicting duplicate bindings, descriptor
mismatches, and byte lengths that do not match shape and dtype.

Producer-selected reference names are transient. Canonicalization replaces
them with `pcid:<40 hex>` identities derived from shape, dtype, space, and raw
bytes using the repository's length-framed BLAKE2b-160 stable hash. Identical
content deduplicates to one payload segment. Canonicalization is idempotent and
happens before cache fingerprinting.

### Binary format

Canonical bytes contain, in order:

1. The four-byte `DMFC` magic and version byte `0x01`.
2. A little-endian u64 JSON-header length.
3. A canonical UTF-8 JSON header with the format tag, conditioning structure,
   and content-sorted payload manifest.
4. Payload segments in manifest order without padding or trailing bytes.

The permitted byte-aligned dtypes are F64, F32, F16, BF16, I64, I32, I16,
I8, U64, U32, U16, U8, and BOOL. Multi-byte payloads are little-endian.

Decoders reject incorrect magic or version, oversized or malformed headers,
non-canonical JSON, unknown formats or dtypes, unsorted or duplicate content
identities, invalid content hashes, segment-count and size violations,
truncation, and trailing bytes. Decoding reconstructs validated record objects
and preserves `EMPTY_RANGE` singleton identity.

The header limit is 64 MiB and the manifest limit is 4096 payload segments.
Canonical JSON normalizes signed zero. Envelope metadata mirrors the carrier
format only; disagreement with the authoritative framed header refuses.

The type fingerprint is the stable hash of canonical carrier bytes. Cache
rehydration validates encoded bytes before trusting their fingerprint;
validation failure is a conservative cache miss.

Framework adapters convert supported dense tensors to and from payload bytes.
They normalize detached CPU-contiguous storage, preserve canonical BOOL bytes,
and reject scalar, sparse, quantized, complex, sub-byte, unsupported unsigned,
and unknown dtypes rather than casting them.

## Graph compiler boundary

Pack declarations register deterministic graph compilers. The worker freezes
their ordered identities into its generation and executes them to an
idempotent fixpoint before elaboration, validation, signature calculation, and
execution. Every later stage therefore sees the same compiled graph.

Compiler output is additive node emission and input rewiring with deterministic
generated identities, bounded expansion, origin mapping, and parent-verifiable
accounting. A compiler is pure prompt preprocessing: it cannot load models,
create runtime patch state, or mutate execution state.

The canonical scheduling intermediate representation is the existing
`ConditioningSet` and `ConditioningRecord` model with `PercentRange`,
`EMPTY_RANGE`, token-layout and condition-scale descriptors, and the DMFC
carrier. Compilers do not define a parallel scheduling vocabulary.

## Scheduled text encoding

Scheduled encoding accepts explicit encoder routes, prompt schedules, text
patch stacks, diffusion patch stacks, and post-encode transforms. It emits
canonical conditioning records whose metadata is finite and serializable.

Text and pooled payloads bind to the selected model-family token layout.
Effective ranges are intersections of declared ranges. Structural text-encoder
variants and their patch groups are owned by one execution, cached only under
their complete behavior identity, and restored transactionally.

An empty scheduling registry preserves ordinary encode behavior and identity.
Unsupported routes, layouts, transforms, or patch combinations refuse rather
than silently dropping declarations.

## Regional conditioning and condition scale

Runtime materialization converts percentage areas, resizes masks, and groups
compatible regional records once per sample execution. Evaluation follows
closed range semantics and accumulates each role independently with count
normalization. The unconditional lane may be omitted only when no active
record requires it at the current sigma.

Condition-scale vectors bind to grouped patch evaluation. Prepared patch
payloads stage once per execution and are shared across sigma evaluations.
Patch application is transactional and restores the exact baseline even after
cancellation or failure.

## Scheduled sampling

Scheduled sampling consumes canonical carriers and resolves diffusion patch
stack digests through the current worker generation's provider authority. A
digest identifies a request; it never grants loading authority. Resolution
materializes one patch set per stack identity and reuses it across positive and
negative conditioning.

The sampler prepares grouped patches once, evaluates active conditioning and
LoRA strength at each sigma, preserves ordinary schedule and noise order, and
always closes execution-scoped patch state. Unsupported latent batch-index
metadata, model-family combinations, providers, or patch forms fail explicitly.

The unscheduled path remains the ordinary runtime path. Scheduling changes
behavior identity whenever conditioning or patch curves can affect execution.

## Native product activation

Compatibility authoring nodes map hook keyframes, LoRA declarations,
conditioning ranges, and conditioning properties into the canonical records at
native node call sites. Native text encoding and native sampling consume those
records directly. Runtime patch state is created only at the native execution
seam, not by graph compilers.

This keeps a graph loaded with the ordinary checkpoint node on the native arm.
Scheduling does not require a compatibility checkpoint loader, a compatibility
runtime, or test-only injection.

## Native authoring surface

Native scheduling accepts SD 1.5, SDXL, SDXL Refiner, Flux Dev, and Flux
Schnell workflows through these compatibility-shaped authoring nodes:

- Create Hook LoRA sets model and CLIP base strengths from -20.0 through 20.0
  and may append to an ordered hook chain.
- Create Hook Keyframe adds a unique start percentage from 0.0 through 1.0
  with a strength multiplier from -20.0 through 20.0. Set Hook Keyframes binds
  the ordered points to every LoRA in a hook chain. A multiplier holds until
  the next point; points do not interpolate.
- Timesteps Range defines a closed conditioning interval from 0.0 through 1.0
  and also emits the intervals before and after it.
- Cond Set Props Combine and Cond Pair Set Props set conditioning strength from
  0.0 through 10.0, an optional timestep range, and an optional LoRA hook chain.
  An optional float32 mask applies the strength as its regional multiplier;
  `mask bounds` derives the execution crop from the resized mask.

Scheduled patch evaluation accepts simple LoRA up/down tensors without LoCon
mid weights, DoRA scales, or reshape directives. Targets may be uniquely owned,
unaliased Linear weights, or ungrouped Conv2d weights using zero padding mode
whose down tensor matches the target kernel and whose up tensor is 1x1.
Linear targets also accept shape-identical full-diff patches. LoHa, LoKr,
GLoRA, OFT, BOFT,
nested, set, model-as-LoRA, offset, transformed, grouped, shared, pre-hooked,
and fp8-matmul targets are rejected.
The ordinary unscheduled patch path continues to support the broader patch
algebra listed in
[`docs/supported/lora-and-model-patching.md`](supported/lora-and-model-patching.md).

## Contract proofs

The implementation maintains focused proofs for:

- immutable record algebra, range endpoints, and canonical serialization;
- payload binding, deduplication, framing refusal, and byte-stable round trips;
- graph-compiler ordering, generated identities, fixpoint behavior, expansion
  limits, origins, and parent accounting;
- scheduled encoding route, transform, and patch-group refusal behavior;
- regional grouping, masks, condition scale, dynamic lanes, cancellation, and
  exact patch restoration;
- native call-site conversion and one materialized patch set per stack;
- unchanged ordinary behavior when scheduling declarations are absent; and
- real native-arm generation through the public queue API.
