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

## Platform golden evidence

Portable golden baselines remain in Dinkster. Platform-specific variants live
only in the sibling `dinkster-evidence` checkout under
`platform-goldens/files/<Dinkster-relative-path>`; its manifest records each
path, platform key, and SHA-256 digest. `DINKSTER_EVIDENCE_ROOT` selects that
checkout and defaults to the sibling `dinkster-evidence` directory.

Golden generators write Linux baselines to Dinkster. On other platforms they
route variant output to the evidence layout instead of creating a sidecar in
Dinkster. Generators that deliberately produce a platform variant on Linux do
the same. After generating a variant, add or update its unique path/platform
manifest entry and SHA-256 in the evidence repository. Generation fails if the
evidence checkout or its `platform-goldens/files` layout is absent.

## Pull requests and full validation

`.github/workflows/ci.yml` runs one job with a five-minute limit. After
installing the locked workspace and fetching pinned evidence for type
resolution (not model weights or coverage inputs), it runs
`bash scripts/ci-fast.sh`: Ruff format, Ruff lint, Pyright, and this fixed
path-based unit subset:

- `tests/test_schema.py`: schema construction and type validation.
- `tests/test_values.py`: codecs, fingerprints, inline values and renditions.
- `tests/test_graph.py`: graph validation and execution planning.
- `tests/test_graph_wire.py`: wire round trips and malformed-input rejection.

These tests use synthetic in-process data. Selection does not depend on the
changed files, network access, model availability or hardware. Full suites
remain required locally before landing; this subset is fast PR feedback,
not a replacement for full validation. Reproduce the job after
`uv sync --locked --all-packages` with `bash scripts/ci-fast.sh`.

The PR job defaults to `[self-hosted, linux, x64]`. To use hosted Linux
without editing the job, set repository variable `DINKSTER_PR_RUNNER` to
the JSON string `"ubuntu-latest"`. No local wrapper or machine path is used
by the fast checks.

`.github/workflows/full-validation.yml` runs on every main push, daily at
10:23 UTC (03:23 Pacific daylight time / 02:23 Pacific standard time), and
manual dispatch. It retains the Python 3.12 Linux suite, both Python 3.12
Windows shards, branch coverage with the 80% floor, model tests, all five
torch CPU attempts and their final gate, translation coverage, artifact
smoke checks and the macOS descriptor test. Select a branch in Actions'
"Run workflow" menu, or pass the dispatch ref explicitly:

```bash
gh workflow run full-validation.yml --repo Kosinkadink/Dinkster --ref <branch>
```

The dispatch ref selects both the workflow and checked-out code, so owners
can obtain Windows and full-suite evidence for an unmerged branch. The
selected branch must contain the workflow. The daily audit uses main.

Both workflows use Python 3.12 only. Package requirements and the dependency
lock continue to support Python 3.13. Heavy jobs run independently; the
torch CPU attempt chain selects a CPU matching its golden-data contract,
not a serialization limit.

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

The CPU retry jobs never acquire or execute model weights. All five
`torch-cpu-try*` callers pass `run-model-tests: "false"` to
`.github/actions/torch-cpu-suite`. The action defaults to false as well.
Environment setup, pinned source downloads, source-parity receipts and their
tests, and all eleven Torch/vision pyright projects remain in these jobs. The
nine vision package suites also run without their artifact
environment variables, so their weight-dependent cases skip while synthetic
input validation, preprocessing, batching, cache, fallback, tiling, and
architecture tests still run. The packages are `dinkster-vision-hed`,
`dinkster-vision-upscale`, `dinkster-vision-depth-anything-v2`,
`dinkster-vision-detr`, `dinkster-vision-rtdetr`, `dinkster-vision-efficient-sam`,
`dinkster-vision-birefnet`, `dinkster-vision-depth-anything-v3`, and
`dinkster-vision-sam31`.

The CPU retry jobs exclude all pinned model-weight acquisitions, the combined
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

The `model-tests` job runs in full validation in `Kosinkadink/Dinkster` on
main pushes, the daily schedule and manual dispatch. PR labels never enable
heavy jobs; dispatch full validation against the branch when model evidence
is needed before landing. The job uses `[self-hosted, linux, x64]` and the
same complete composite action with `run-model-tests: "true"` and the
existing read-only identity and evidence deploy keys. No other job depends
on it.

The runner must have an AuthenticAMD CPU with AVX2 and **without AVX-512**,
`MKL_CBWR` unset, and sufficient disk/RAM for the pinned CPU workloads.
A GPU-equipped machine still runs CPU parity with Torch 2.13.0+cpu and the
existing AVX2 dispatch pins; this does not switch to GPU goldens. Checkouts
clean the workspace and do not persist credentials. Deploy keys live in
per-step SSH agents with post-job cleanup, and downloads use `RUNNER_TEMP`.
Linux full suites use the host's counted-suite launcher. The disk-reclaim
step that removes hosted image tooling is guarded by
`runner.environment == 'github-hosted'` and
never runs on a self-hosted machine.

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
