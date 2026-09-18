# Dinkster AGENTS

## Docs discipline

Keep docs concise. No delegation ledgers, promise ledgers, handoff records,
or coordination narratives in this repo: GitHub issues and PRs are the record
of work, decisions, and deferred items. SUPPORTED.md is the human-readable
index of what works today. Capability claims live in
`docs/supported/<area>.md`; a commit that changes what is supported edits the
matching area file in the same commit. SUPPORTED.md is the index only, and
capability documents never describe implementation. DESIGN.md holds architecture
rationale only - no milestones, status, or history. ROADMAP.md holds only a
short list of what is next (move completed and deferred history to issues).
The documentation for how code works is the code and package READMEs.
When work uncovers a bug in ComfyUI itself, add a dedicated .md under
`docs/comfyui-issues/` in the same commit and list it in that folder's
README index.

## Comment policy

Comments explain only difficult behavior, invariants, safety or compatibility
constraints, or non-obvious reasons. They never carry plans, task status,
implementation history, or future-work promises (track those in GitHub
issues). Never use invented plan jargon anywhere a human reads - milestone
tags ("M0", "M6 slice"), slice/commission codes ("S2C1"), track or generation
labels, or any code word that only means something with access to a transient
plan. This ban covers comments, commit messages, PR titles/descriptions,
issue text, and docs: name work by what it does, in plain words a reviewer
can understand standalone. Durable real identifiers are fine (issue/PR
numbers, version numbers, WIRE/protocol versions, commit SHAs, API names).
Prefer clear code and names over comments. In-code records required
by a repository contract remain authoritative and must be preserved.

## Review discipline

Every slice gets a real code review after implementation and tests pass,
before the commit - a pass over the new code, not a re-read of the diff
summary: API cohesion (exported-but-unconsumed surface is a cut candidate),
invariants (frozen/immutability claims hold, validators fire), layering (no
reaching into later-stage mechanisms, no hidden global state), and doc drift.
Fix findings before committing or file an issue for deliberate deferrals.
Record the outcome in the commit body ("Reviewed: ..." or "no findings").

## One sampling engine (user directives, 2026-08)

There is exactly one sampling execution engine: the decomposed
custom-sampling seam (noise / guider / sampler / sigmas / latent, the
CustomSamplingRuntime contract in dinkster_inference/runtime.py). Every model
family implements ONLY that seam. KSampler and every other sampler node are
sugar - thin compositions of the seam (see run_ksampler_as_custom in
dinkster_inference_torch/sampling_execution.py) - never a second execution
path. Do not add bespoke per-family assembly branches, KSampler-only
sampling code, or family gates that make the decomposed path reject what the
KSampler path accepts. Cross-cutting sampling behavior (distributed
admission, previews, cancellation, masks) is a property of the engine, so it
applies identically to every sampler node; wiring it into one node or one
path is a defect. Refusing a capability on the decomposed path that the
KSampler path supports is acceptable only as a brief migration intermediate
with an open issue, never as an end state. Reviews gate on this.

## Performance parity discipline

ComfyUI's optimizations are the FLOOR, not a wish list (user directive,
2026-07). Never defer porting an upstream optimization on "wait until
profiling shows a bottleneck": Dinkster being correct-but-slower than ComfyUI on
the same operation is a defect. The trigger for porting an upstream
optimization is when the surface it optimizes lands in Dinkster. Profiling is
for finding opportunities to exceed upstream, not for deciding whether to
match it.

## Numerical parity discipline

Hard-won rules from the flip parity harness and scheduler port slices
(2026-07-26). The failure mode they guard against is FALSE CONFIDENCE: a
green suite whose tolerances quietly absorbed a real defect.

- Never widen a sampling tolerance to make a test pass. A widening is
  legitimate only AFTER instrumentation proves the upstream seams bit-exact
  (sigmas, noise draws, brownian bounds) so the residual is pinned to
  denoiser float32 accumulation. Then it must be per-case, never blanket,
  and documented at the override site with the observed drift, the headroom,
  and the amplification mechanism (see SAMPLE_ATOL_OVERRIDES in
  test_sd_pipeline.py / test_pipeline.py for the required shape).
- Bit-exact and value-close are different contracts. Pure float64 ports are
  value-close (~1e-6 relative) to the reference's float32 tensor kernels -
  fine for value math, NOT for brownian-tree noise streams, which
  decorrelate on one-ulp sigma differences (the harness measured 3.4e-2
  end-to-end from a ~1e-7 karras delta). Any schedule or SigmaSpace kind
  that can feed an SDE sampler must be ported onto reference kernels in
  dinkster_inference_torch/schedules.py WITH executed SDE golden coverage
  before that pairing ships. No evidence, no port, no wiring.
- Goldens are generated ONLY on the generating tool's own pinned ComfyUI
  commit (its BASELINE/REFERENCE_COMMIT constant, which the tool enforces)
  with the generator's documented interpreter, and must be proven bit-stable
  across regeneration (same sha256 twice). When regenerating to add cases,
  diff and confirm every pre-existing case's data is unchanged.
- Unexplained drift is an escalation, not a tuning knob. If outputs disagree
  and the cause is not yet proven, stop and instrument the seams in order
  (schedule sigmas -> initial noise -> per-draw noise -> per-step latents)
  until the divergence point is found. Do not commit anything whose mismatch
  you cannot explain.

## Validation gate (before any commit)

```
.venv/bin/ruff check .
.venv/bin/pyright
.venv/bin/python -m pytest -q
```

All three must be clean. When `packages/dinkster-inference-torch` (or anything
it consumes) changes, two extra gates apply - the root venv is deliberately
torch-free, so that package has its own environment (`.venv-torch`, see its
README for setup) and pyright project:

```
.venv/bin/pyright -p packages/dinkster-inference-torch
CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \
  .venv-torch/bin/python -m pytest -q packages/dinkster-inference-torch/tests
```

When the machine has CUDA GPUs, the capability-gated GPU suite
(`packages/dinkster-inference-torch/tests/test_gpu.py`) must also run - under
`.venv-torch` it silently skips, which proves nothing. Use the dedicated
CUDA venv per the package README "GPU validation" section (on this host:
`.venv-gpu`, torch 2.13.0+cu130, with the CPATH headers from
`.venv-gpu-extras/pyheaders`). Do NOT defer GPU validation on a GPU machine
(user directive, 2026-07).

When `packages/dinkster-kernels` changes, its gates run on GPU machines only
(the kernels need CUDA plus triton; elsewhere its tests skip and prove
nothing):

```
.venv/bin/pyright -p packages/dinkster-kernels
CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \
  .venv-gpu/bin/python -m pytest -q packages/dinkster-kernels/tests
```

Torch testing policy (user directive, 2026-07-26): test/validation
environments run torch >= 2.10. The `torch>=2.5` package floor is a
backwards-compatibility promise for consumers, pinned to upstream ComfyUI's
published minimum - track upstream if it rises, and do not validate against
pre-2.10 torch as if it were the primary target.

Stale wheel cache after rebase: when a pull or rebase changes a node pack or
its committed lockfile, a plain `uv sync --all-packages --frozen` can re-link
stale cached wheels whose artifact digests no longer match the committed
`dinkster.lock`, producing spurious `dinkster.compose`
PackageNotFoundError/CompositionError failures in the local root suite (clean
CI installs are unaffected). Run `uv sync --reinstall` (or
`uv sync --reinstall-package <name>` for the affected pack) before treating
such failures as real regressions.
