# Testing policy

ComfyUI's test coverage was scattered because testing was optional. In Dinkster
it is not: the suite is the enforcement half of every contract DESIGN.md
makes, and these rules keep it that way as the codebase grows.

## Ground rules

- **Every bug fix ships with the regression test that would have caught it.**
  No exceptions - if it broke once without a test noticing, the suite has a
  hole exactly there.
- **Every wire and API contract has golden tests.** Schema wire, engine
  events, job wires, the boundary protocol, and `dinkster_api.v1` all pin their
  exact shapes (`tests/test_graph_wire.py`, `tests/test_server.py`,
  `tests/test_api_v1.py`). Goldens are generated through the real encoders,
  never hand-written, and a golden diff is a deliberate, reviewed act.
  Frontend fixtures are generated the same way - if the encoder changes, the
  fixtures change in the same commit.
- **`dinkster_api.v1` only grows.** The golden-surface test enforces it: a
  removed or renamed name fails CI; additions are a visible diff to
  `GOLDEN_V1_SURFACE`.
- **Concurrency tests assert semantics, not timing.** Wait for the event or
  state that proves the property (with a generous deadline), never a bare
  `sleep` as the assertion. A test that only passes on a fast machine is a
  bug (see `test_remote.py`'s slot-release poll for the pattern).
- **Behavior over lines.** Coverage is a floor detector, not a target: tests
  state the contract in their docstring and would still make sense if the
  implementation were rewritten.
- **Accepted extension entries resolve.** Doctor imports every declared
  extension scope in its disposable probe. The parameterized doctor test is
  keyed by the shared scope vocabulary, so accepting another scope without
  entry-resolution coverage fails the suite.
- **Builtin inference vocabularies have explicit consumers.** The fast CI
  scanner pins every zero-argument registry factory, assembly-registry builder,
  and descriptor-catalog read to an exact source location and separate
  non-increasing ceilings. The committed ceilings are 9 registry factories,
  1 assembly builder, and 17 descriptor-catalog reads. New and stale sites fail
  the check. The shared-engine family-gate scanner also runs in fast CI with its
  current zero-site ceiling.

## Platform golden evidence

Portable golden baselines remain in Dinkster. Platform-specific variants live
only in the maintainer evidence checkout under
`platform-goldens/files/<Dinkster-relative-path>`; its manifest records each
path, platform key, and SHA-256 digest. `DINKSTER_EVIDENCE_ROOT` selects that
checkout and defaults to the expected sibling location.

Golden generators write Linux baselines to Dinkster. On other platforms they
route variant output to the evidence layout instead of creating a sidecar in
Dinkster. Linux runtime-specific baselines without a platform key remain beside
their portable baseline in Dinkster. Generators that deliberately produce a
platform-keyed variant on Linux route it to evidence. After generating a
variant, add or update its unique path/platform manifest entry and SHA-256 in
the evidence repository. Generation fails if the evidence checkout or its
`platform-goldens/files` layout is absent.

## Pull requests and full validation

`.github/workflows/ci.yml` runs one fast job with a ten-minute limit. After
installing the locked workspace and fetching pinned evidence for type
resolution (not model weights or coverage inputs), it runs
`bash scripts/ci-fast.sh`: Ruff format, Ruff lint, Pyright, and this fixed
path-based unit subset:

- `tests/test_extension_contract_pack.py`: a real CPU server composed with only
  the ordinary extension fixture pack, proving two custom nodes linked through
  a pack value type, a typed route and event, the frontend snapshot module, and
  execution end to end.
- `tests/test_extension_factory_guard.py`: exact registry-factory,
  assembly-builder, and descriptor-catalog sites with separate non-increasing
  ceilings.
- `tests/test_family_registration_gates.py`: zero literal family gates in
  shared engine code and registered-property coverage.
- `tests/test_schema.py`: schema construction and type validation.
- `tests/test_values.py`: codecs, fingerprints, inline values and renditions.
- `tests/test_graph.py`: graph validation and execution planning.
- `tests/test_graph_wire.py`: wire round trips and malformed-input rejection.

These tests use synthetic in-process data. Selection does not depend on the
changed files, network access, model availability or hardware. Full suites
remain required locally before landing; this subset is fast PR feedback,
not a replacement for full validation. Reproduce the job after
`uv sync --locked --all-packages` with `bash scripts/ci-fast.sh`.

The required `CI_RUNNERS` repository variable controls every job. Its
`linux`, `windows`, `macos`, and `forkLinux` values select GitHub-hosted
images. CPU Torch jobs pin AVX2 dispatch in their environment rather than
depending on machine labels.
When required private inputs are unavailable, the PR job records an explicit
not-run reason and fails instead of reporting unexecuted validation as green.

```json
{"linux":["ubuntu-latest"],"windows":["windows-latest"],"macos":["macos-latest"],"forkLinux":["ubuntu-latest"]}
```

`.github/workflows/full-validation.yml` runs on every main push, every two
hours from 06:00 through 22:00 Pacific, daily at 10:23 UTC, on manual
dispatch, and when called by the release workflow. The `on.schedule` cron
list in that file is the single schedule definition; change its first cron
line to change the two-hour cadence. A scheduled run skips the heavy jobs
when the latest successful main run already validated the same commit.
Main pushes use one non-cancelling concurrency group. GitHub keeps one active
push run and only the newest pending push run, replacing older pending runs as
new commits arrive. A merge whose pending run is replaced is covered by the
next completed run at a descendant head. Find candidate runs in the Actions
`full-validation` history, then confirm coverage from a local clone with
`git merge-base --is-ancestor <merge-sha> <run-head-sha>`. Scheduled,
dispatched and release-called runs use a separate non-cancelling durable group,
so push traffic neither queues nor replaces them. Full validation runs the
Python 3.12 Linux suite as four whole-file shards, both Python 3.12 Windows
shards, branch coverage as four shards combined before enforcing the unchanged
80% floor, model tests, one torch CPU job, translation coverage, artifact
smoke checks and the macOS descriptor test. Every complete lane has a
20-minute or lower job timeout. `main-status` uploads one JSON artifact with
every aggregate result and fails unless every selected lane passed. Select a branch in Actions'
"Run workflow" menu, or pass the dispatch ref explicitly:

```bash
gh workflow run full-validation.yml --repo Kosinkadink/Dinkster --ref <branch>
```

The dispatch ref selects both the workflow and checked-out code, so owners
can obtain Windows and full-suite evidence for an unmerged branch. The
selected branch must contain the workflow. The daily audit uses main.

`release.yml` runs only for version tags. It calls full validation for the
exact tagged commit before building the complete wheel set, then installs and
launch-checks those exact artifacts on Linux, Windows and macOS before
publishing the GitHub Release. Its hosted-eligible jobs use the same
`CI_RUNNERS` variable.

Both workflows use Python 3.12 only. Package requirements and the dependency
lock continue to support Python 3.13. Heavy jobs run independently; the
torch CPU job pins AVX2 dispatch without retry jobs.

Linux ARM64 artifact smoke and release installation are not part of the
supported matrix. Linux x64, Windows x64, and macOS use GitHub-hosted images.

## Coverage

Measured with branch coverage across all packages:

```bash
uv run pytest -q --cov=dinkster_api --cov=dinkster_assets --cov=dinkster_caches \
  --cov=dinkster_compat_comfy --cov=dinkster_engine --cov=dinkster_graph \
  --cov=dinkster_memory --cov=dinkster_nodes_foundation --cov=dinkster_nodes_media_io \
  --cov=dinkster_schema \
  --cov=dinkster_server --cov=dinkster_values --cov=dinkster_workers \
  --cov-branch --cov-report=term-missing
```

CI enforces a ratcheting floor (see `.github/workflows/full-validation.yml`); raise it
when real coverage rises, never lower it to make a PR pass. Known
measurement caveat: code that runs in worker subprocesses (`host.py`,
`service.py`, `provision.py`, `_doctor_probe.py`) is exercised by the suite
but under-reported, because coverage does not follow spawned processes.
Their coverage numbers are floors, not truth - their *behavior* is covered
through the parent-side boundary tests.

## Conditional and hardware tests

Some contracts only hold on specific hardware or OSes. These tests exist,
are clearly labeled, and skip loudly rather than silently not existing:

- **Live ComfyUI compat** (`test_compat_live.py`): needs
  `DINKSTER_COMFYUI_ROOT`; run locally against a real checkout.
- **Isolation/sandbox** (`test_isolated.py`, `test_sandbox.py`): venv
  provisioning behind `DINKSTER_SLOW_TESTS`; bubblewrap/AppArmor paths skip
  where the kernel says no.
- **GPU/multi-GPU** (`test_device_map.py`, placement): device-conditional;
  the two-4090 box is the reference machine.
- **Headless ANGLE mirror parity** (`test_image_mirror_angle.py`): install
  the optional runtime with `uv sync --group angle`, then run the test to
  execute every declared GLSL image mirror against its CPU parity corpus.
  It skips when ANGLE or a float-renderable EGL/GLES 3 context is unavailable.
- **CI matrix**: Linux + Windows, Python 3.12 - Windows is the
  dominant user OS and keeps the tcp-transport and shm-lifetime rules
  honest.

Gaps we know about (roadmap, not accidents): multi-GPU CI runners, network
fault injection for the remote boundary, and property/fuzz tests for the
wire decoders, type-id grammar, and graph nesting - the highest-value fuzz
targets because their inputs come from outside.

## CPU checks and dedicated model tests

The single `torch-cpu` job never acquires or executes model weights. It
passes `run-model-tests: "false"` to
`.github/actions/torch-cpu-suite`. The action defaults to false as well.
Environment setup, pinned source downloads, source-parity receipts and their
tests, and all eleven Torch/vision pyright projects remain in this job. The
nine vision package suites also run without their artifact
environment variables, so their weight-dependent cases skip while synthetic
input validation, preprocessing, batching, cache, fallback, tiling, and
architecture tests still run. The `dinkster-nodes-vision` distribution contains
the HED, upscale, Depth Anything V2, DETR, RT-DETR, EfficientSAM, BiRefNet,
Depth Anything V3, and SAM 3.1 model packs.

The CPU job excludes all pinned model-weight acquisitions, the combined
`dinkster-inference-torch` and `dinkster-model-ipadapter` test lane, each vision
suite's second real-artifact run, and
`tests/test_benchmark_inference.py::test_minimax_h3_identities_are_accepted_by_the_production_dit_loader`.
No tests, assertions, goldens, hashes, or deadlines are removed or relaxed.
The default local gates remain full, including the model lanes:

```bash
./scripts/setup_envs.sh
.venv/bin/python -m pytest -q
unset MKL_CBWR
export ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv-torch/bin/python -m pytest -q packages/dinkster-inference-torch/tests packages/dinkster-model-ipadapter/tests
```

On Windows, `scripts\setup_envs.ps1` creates the equivalent root, CPU Torch,
and NVIDIA CUDA environments. Run the gates it prints with native Windows
paths; rerunning the script reasserts the environments, while `-Force` rebuilds
only the environments it owns.

Run each vision package's full `tests` directory and the benchmark selector
above with `.venv-torch/bin/python -m pytest -q` too. Supply the real-artifact
environment variables documented in the provider READMEs and composite
action; an absent artifact's skip is not real-model validation. Training and
GPU gates in `scripts/setup_envs.sh` and the Torch README also remain required
where applicable. No CI input or repository variable changes local pytest
selection.

The `model-tests` matrix runs in full validation in `Kosinkadink/Dinkster` on
main pushes, the daily schedule and manual dispatch. Pull requests run only
the fast tier. Eight whole-file inference and IPAdapter shards run beside
fixed groups for acceptance sampling and the benchmark loader; HED, upscale
and EfficientSAM; Depth Anything V2, DETR and RT-DETR; BiRefNet and Depth
Anything V3; and SAM 3.1. Every vision test is selected from the single
`packages/dinkster-nodes-vision/tests` tree. All groups run on GitHub-hosted
Linux with CPU Torch and pinned AVX2 dispatch. `model-tests-gate` requires
every group to pass. The contract test asserts the complete suite manifest so
a suite cannot be silently omitted or assigned twice. Each group uses the
same composite action with `run-model-tests: "true"` and the existing
read-only identity and evidence deploy keys. Checkouts do not persist
credentials, downloads use `RUNNER_TEMP`, and full suites execute directly.

## Tooling enforcement

`uv run ruff check .` and `uv run pyright` are part of validation, not
optional extras. `dinkster doctor` is the same idea pointed at packs: run it in
pack CI (the pack template wires it in) so pack regressions surface before
users hit them.

## Capability evidence census

`tools/comfy_coverage.py` owns both the official-template translation census
and the capability evidence ledger. Its pinned workflow_templates and ComfyUI
revisions are constants in that file. With clean read-only checkouts at those
revisions, regenerate all four checked outputs with:

```bash
uv run --locked python tools/comfy_coverage.py \
  --templates /path/to/workflow_templates/templates \
  --comfyui /path/to/ComfyUI
```

Add `--check` to perform the CI staleness and drift gate without writing. The
maintained evidence source is `tools/data/comfy_capability_evidence.json`.
Every `builtin_families()` registration must be present there, and every
`docs/supported/` family claim must carry the matching capability marker. Maintained
aliases enter the generated ledger automatically at T1 after their test
selectors are validated. T0 and T1 never count as support; missing template
capabilities remain explicit `absent`, `refused`, or `unverified` records.
