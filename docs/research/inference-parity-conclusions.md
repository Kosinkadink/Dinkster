# ComfyUI inference parity - accepted conclusions and current-tip reconciliation

Historical scan date: 2026-07-29. Scope: ComfyUI model-family coverage, engine
features, and ModelPatcher successor semantics compared with Dinkster's native
inference and accepted extension contracts. The full scan specification,
reports, 70-row family appendix, and synthesis live in the private
[`Kosinkadink/dinkster-research`](https://github.com/Kosinkadink/dinkster-research)
repository.

## Historical headline coverage

| Scan | Rows | Result |
|------|-----:|--------|
| P1 model zoo | 70 | 2 NATIVE / 3 PARTIAL / 65 COMPAT / 0 ABSENT |
| P2 engine features | 52 | 17 NATIVE / 29 PARTIAL / 6 COMPAT / 0 ABSENT |
| P3 ModelPatcher audit | 119 | Five UNOWNED semantic groups, all now dispositioned |

These are explicitly dated scan results, not current coverage totals. No later
upstream families have been silently classified into the 70-row appendix. At
the time of the scan, the principal gaps were composition semantics and
model-family runtimes, not sampler catalog membership. P3's five formerly
UNOWNED groups map to a new auxiliary-model graph slice, the GGUF/provider
program, the S2 Parameter identity contract, and unified typed dependency
discovery.

## Current source census (2026-08-03)

At ComfyUI commit
[`14b05228cef127ce529bc0c08660770d4af3e9a8`](https://github.com/Comfy-Org/ComfyUI/blob/14b05228cef127ce529bc0c08660770d4af3e9a8/comfy/supported_models.py),
`comfy/supported_models.py` contains 98 entries in its `models` list. The
working-tree pin `611f2a4e` contains 97; `MiniMaxH3` is the one remote
addition. This is a census of upstream registrations, not 98 independently
classified Dinkster parity rows.

At Dinkster baseline `5ab6acd6c87c459ff8caed37f22eb2d9d52ff992`, the public
wired-family set in `packages/dinkster-inference/src/dinkster_inference/runtime.py`
and `packages/dinkster-inference-torch/src/dinkster_inference_torch/wiring.py` is
exactly:

- `dinkster.sd15`
- `dinkster.sdxl`
- `dinkster.sdxl_refiner`
- `dinkster.flux_dev`
- `dinkster.flux_schnell`

`packages/dinkster-inference/src/dinkster_inference/identity.py` pins
public family.

## Committed canonical verdicts

The canonical records committed in
[`dinkster-evidence`](https://github.com/Kosinkadink/dinkster-evidence/tree/main/inference-parity/records)
all have `overall_pass: true`. These are committed-record facts from their
named candidate commits, not claims that the gates were rerun at current HEAD.

| Workload | Candidate | Acceptance digest | Verdict |
|----------|-----------|-------------------|---------|
| `W0-HARNESS-SD15-TXT2IMG` | `c1e882e4a9fe25fc8a1237718ee5de92b21d36ae` | `sha256:df8852236fdd13e4c3ac56cfa9af5eff530fb81c153562a27d09c9dbef722d8a` | PASS |
| `W0-SD15-INPAINT` | `c415236b91589f9e3f2040419b28dd8cd9bf36e8` | `sha256:371c0c3aef1769bf517e003da5f19be06b4551137719e58bf546ef228dbc68cf` | PASS |
| `W0-SDXL-VPRED` | `61b926db8b5c5c0b70986ab6ba10dcd6d916d028` | `sha256:8d21fafb5320504c5ba3c18aa4069ede4e6ae75f4b680f5a94a4bf9156338f39` | PASS |
| `W0-SDXL-EDM-VPRED` | `073f8dc5d870eb47b0c4212c504691733f6e3c0e` | `sha256:b043f4e1e640197f6e023bff7312b21ad8395ba1a994b68f00882c360f22d740` | PASS |
| `W0-FLUX-GATED-OVIS-TXT2IMG` | `f893a964b34a9fa635c45dac2e05e5247a2bcd85` | `sha256:8763f0d08c3543ec14b478c35db631f35eac8c18aef703ccfdd1104d0c29120c` | PASS |

## Frozen closure order

This P0 current-tip reconciliation is complete. The remaining active authorized
closure program is graph compiler execution, scheduled encoding and patch
groups, and regional, masked, and per-condition scaling execution.

After that conditioning work, three blocking Wave-0 residual groups remain before
Wave 1:

1. native textual-inversion file resolution behind the shipped
   `EmbeddingLookup`;
2. `W0-SDXL-INPAINT`, with a digest-pinned artifact, template-derived workload,
   and one-warmup plus one-real gate; and
3. patched/LoRA storage-dtype conversion, with byte-exact patch-kind proof and
   digest-gated production GPU confirmation.

The permitted codec-only Diffusers/TAESD benchmark deferrals and LongCat
component-layout prework are distinct from those blocking residuals. Their
historical W0 names neither authorize execution nor add a fourth blocker. Wave
1 remains Qwen Image -> Wan -> LTXV/LTXAV -> Flux2 and is not authorized by
this reconciliation.

## Accepted waves and owners

1. **Wave 0:** establish the reusable benchmark harness and close every native
   partial for SD1.5, SDXL, and Flux dev. Native textual-inversion resolution
   is included because it blocks SD prompt-loader parity. Wave 0 must complete
   100 percent before Wave 1 begins; no partial closures are deferred.
2. **Wave 1:** Qwen Image -> Wan -> LTXV/LTXAV -> Flux2. These programs consume
   shared family components and the separately tracked native GGUF provider
   program rather than creating private storage or auxiliary-graph paths.
3. **Wave 2:** remaining evidenced-demand families and explicit modality
   runtimes. Learned upscaling has its own row; audio lands with ACE-Step
   demand; 3D/volume work lands with Hunyuan3D demand. One
   Lumina2/NextDiT program covers Lumina2, NetaYume, and ZImage.
4. **Wave 3:** the long tail, still revivable by explicit user demand. A zero
   bundled-template hit is not proof of zero users. Anima rides the Wan+Qwen
   component stack when demanded.

Demand ranking aggregates by family line, excludes API-only templates, and
retains all 70 historically scanned rows as the audit appendix. ComfyUI-supported
families without template hits remain eligible when user demand supplies the
evidence.

## Shared graph and dependency contract

S2 reconstruction recipes gain an ordered child-handle/residency-group field,
owned by a new auxiliary-model graph slice. It covers both the general graph
contract and conditioning-scoped auxiliary models; the first video family must
not privately reconstruct either behavior.

One typed `DependencyRef` declaration replaces `additional_models`, duck-typed
`.models()` discovery, conditioning-carried models, and
`AdditionalModelsHook`. A dependency declares its child handle or recipe
reference, scope (`model`, `contribution`, `invocation`, or `conditional`),
clone mode (`with-parent` or `shared`), and explicit accounting owner. One
pre-load discovery walk builds one residency graph. S2 recipe children and S6
contribution dependencies are this same concept at different scopes. The
contract deliberately closes ComfyUI's unaccounted `inference_memory` TODO and
its uncloned shared `.models()` dependents.

S2 additionally preserves torch `Parameter` object identity during explicitly
in-place patch/restore. Restoring only the exact value does not satisfy compile
or training consumers that retain the original object.

## Scheduled and per-condition adapters

Dinkster replaces a direct port of ComfyUI's hook system with three measured
application tiers:

1. **Static bake:** existing S2 overlays, structural-digest identity, and
   aimdo-owned residency.
2. **Runtime application:** low-rank `y = Wx + s(t,cond)*B(Ax)` or full-diff
   `y = Wx + s(t,cond)*Dx`, using per-batch-element scales so mixed conditions
   execute in one forward.
3. **Segment bake:** piecewise-constant heavy schedules become explicit S2
   overlay identities visible to residency rather than unaccounted full-weight
   hook caches.

The harness chooses the tier from measurements. Tier 2 is shared by scheduled
LoRA, per-condition LoRA, and quantized (GGUF) plus LoRA, and must beat
ComfyUI's serial hook path on a per-condition LoRA workload. Transformer option
semantics become per-condition S6 typed contributions; additional hook models
become conditional `DependencyRef`s. Object-patch and injection hooks remain
rejected because those surfaces are `NotImplemented` upstream.

A typed cast-time weight-function contribution is required and lands with the
native GGUF/storage-provider program. Native GGUF remains a separately tracked
program implemented through S9 contracts, integrated with residency/offload
and LoRA, and gated to beat the ComfyUI-GGUF custom node.

## Benchmark and memory policy

- Use ComfyUI's bundled workflow templates or exact Dinkster equivalents.
- Record exactly one warmup run and one real run; measure and gate both.
- Compare every port directly against pinned ComfyUI on matched workloads.
- Accept no performance regression without explicit user approval.
- Where Dinkster replaces a custom pack, add and beat that pack's baseline.
- For P2 M05, require VRAM outcome/policy equivalence only. Dynamic memory
  management supersedes the legacy low/normal/high/no-VRAM mode identities;
  Dinkster does not reproduce those identities.
