# Testing policy

ComfyUI's test coverage was scattered because testing was optional. In Dinkster
it is not: the suite is the enforcement half of every contract DESIGN.md
makes, and these rules keep it that way as the codebase grows.

## Ground rules

- **New single-pack execution tests use `dinkster_engine.testing` when enough.**
  Use its real worker-process default and share startup with a module- or
  session-scoped fixture. Keep composer and HTTP tests for behavior that needs
  those layers. See the [engine testing guide](../packages/dinkster-engine/README.md#tests-from-pytest).
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
- **Extension factories have explicit consumers.** The fast CI scanner checks
  extension contribution factories against the committed source baseline.

## Test deadlines

`tests.async_scenarios.run_scenario(coroutine, timeout=120)` runs async test
scenarios under `asyncio.timeout()`. Its default deadline is 120 seconds;
expiry fails with the current test name and allows the next test to run.
Use `run_scenario(scenario(), timeout=240)` only when measured work needs
longer, with a one-line reason at that call site. Cancellation must remain
cooperative, including scenario cleanup.

pytest-timeout applies a 300-second backstop to each whole test, including
setup and teardown, using `timeout_method = "thread"` on every platform.
Blocking synchronous code or cancellation-resistant cleanup can prevent the
cooperative deadline from firing; the backstop dumps thread stacks naming
the active test and exits the process, so the rest of that shard cannot run.
For a measured longer whole-test requirement, use
`@pytest.mark.timeout(seconds)` with a reason, keeping the helper deadline
below that backstop and the backstop below the CI job budget. Job limits are
the final outer bound; neither timeout skips a test. Worker crashes fail the
run without restarting the worker or retrying its tests.

The defaults allow over four times the 28.25-second slowest call reported
by the hosted Windows shard 1 sample in
[full-validation run 42071734785](https://github.com/Kosinkadink/Dinkster/actions/runs/42071734785/job/127401249972).

## Parallel execution

Install the locked development group before testing. Root pytest invocations
use pytest-xdist with at most eight logical-CPU workers and work stealing.
Smaller hosts use their detected CPU count. Tests must own their ports and
temporary directories; generated binary parameters need stable, explicit IDs
so every worker collects the same cases. Assertions and coverage sources are
unchanged by parallel execution.

Use `-n 0` to debug a test in one process, or `-n 2` to select two workers.
Do not use retries to hide a parallel-only failure. Package suites launched
from their own project directory retain that project's pytest settings.
Selected-archive admission with multiple test files splits whole files between
two serial pytest processes after shared catalog preparation. Each process
tracks unavailable exports and executed tests with its inline plugin. Any test
or collection failure fails admission, and at least one selected test must pass.

Full-validation jobs run serially inside each matrix shard with
`PYTEST_ADDOPTS=-n0`: catalog caches and process-state fixtures share state
within a job. The matrix jobs themselves run in parallel. Linux and Windows
root shards print every setup, call and teardown duration with
`--durations=0 --durations-min=0` and retain per-test totals in
`test-timings-linux-N` and `test-timings-windows-N` JUnit artifacts, including
when pytest fails. Use each platform's timings to balance its shards instead
of counting files.
The eight Windows runtime shards rotate each case-split file's sorted cases
from its file-assigned shard, so expensive prefixes do not all start in shard one.

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

`.github/workflows/ci.yml` always reports `format, lint, types` and `tests`.
Docs-only changes skip environment provisioning and test execution inside
those jobs. Other changes run Ruff format, Ruff lint and both platform
Pyright checks with `bash scripts/ci-fast.sh checks`. Draft pull requests run
only these checks. Ready pull requests run the whole configured kept Linux
suite with CPU torch through `bash scripts/ci-fast.sh tests`, including
pack-only changes. Frontend decoding reads the live catalog of an isolated
CPU generation acquired by supervisor setup from the published main channel.
Setup prepares its catalogs before sealing. After stopping its assigned backend,
the current checkout engine serves the generation's admitted packs and runtimes;
it does not substitute the published engine catalog or checkout defaults.
An unavailable default pack or an empty default catalog fails the export.
There is no hand-picked file list, network-marker exclusion
or changed-path runtime deselection. The four audited environment suites retain
their existing nonblocking policy under #867.

`tools/ci_selection.py` additionally selects changed packs' behavioral tests;
shared library changes also select their declared dependents. Registry default
pack admission gates only scanner or harness changes. Publication preflight runs
for publisher or manifest changes, not ordinary pack source edits. Reproduce the
checks and whole configured suite after `uv sync --locked --all-packages` with
`bash scripts/ci-fast.sh`.

Landing requires the selected hosted jobs green and a fresh LOW verifier PASS at
the offered head. Running the counted full suite or its shards locally is not a
landing gate. During development, run the tests the change touches. The whole
suite runs in hosted full validation on main after landing; a red main follows
the integrator's repair route. Issue-specific acceptance evidence still applies.

### Counted pack runtime matrix

Full validation covers the following runtime environments, not only the root
and workspace-member tests. This matrix is not a local pre-freeze requirement.
The selections and setup commands in
`.github/workflows/full-validation.yml` and
`.github/actions/torch-cpu-suite/action.yml` are authoritative.

| Lane | Required execution |
| --- | --- |
| Linux CPython 3.12 root | All twelve serial `test` shards, balanced by native Linux timings with history and generations split by sorted test ID, other files whole, and the workflow's branch-coverage sources and exclusions |
| Windows CPython 3.14.10 root | All ten serial `test` file shards, balanced by native Windows timings with module fixtures kept together, excluding the separate runtime selection and using the patched SQLite runtime |
| Windows CPython 3.14.10 runtime | All eight serial `windows-runtime` jobs: environment, installer, release-install and supervisor contracts; artifacts, CLI, collection, generations, history and transactions split by sorted test ID, other files stay whole |
| CPU node packs | All four `node pack tests (CPU, shard N)` selections in the CPU bundle's `.venv` |
| CPU vision packs | The `vision pack tests (CPU)` root contracts, evidence-tooling tests, vision typecheck and non-model vision tests |
| Probe jail | Live jail, jailed-doctor and sandbox model-selection host contracts in the configured Linux isolation environment |
| Coverage report | The selected platform's coverage artifacts combined, without a percentage floor |

Build the CPU runtime with
`uv run --no-project python scripts/setup_envs.py --variant linux-cpu` on Linux
or `--variant win-cpu` on Windows. The same Python implementation serves the
launcher and CPU action; its receipt is `.run/environment-base.json`.
Contributor setup acceptance runs the root and pack checks with the bundle's
Python version. The Linux CPython 3.12 row describes its separate full-validation
root jobs; it does not override the selected bundle interpreter.
Keep isolated pack-install tests: workspace editables are not evidence that a
published pack declares its complete dependencies. Pin CPU dispatch with
`ATEN_CPU_CAPABILITY=avx2` and `ONEDNN_MAX_CPU_ISA=AVX2` as the workflow does.
Prepare the pinned validation inputs before running evidence-tooling tests.

On X570, launch each complete suite through the wrapper workspace's
`run_counted_suite.sh` with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`.
From a Dinkster root checkout with its locked development dependencies and
pinned validation inputs prepared, the counted CPU root command is:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  DINKSTER_ACCELERATOR=cpu DINKSTER_SLOW_TESTS=1 \
  /home/kosin/comfy-vibe-station/run_counted_suite.sh \
  .venv/bin/python -m pytest -q --durations=0 --durations-min=0 \
  --junitxml=test-results/root.xml \
  --deselect=tests/test_sandbox.py::TestLiveJail \
  --deselect=tests/test_doctor.py::TestLiveJailedProbe \
  --cov=dinkster_api --cov=dinkster_assets --cov=dinkster_caches \
  --cov=dinkster_engine --cov=dinkster_graph --cov=dinkster_memory \
  --cov=dinkster_image --cov=dinkster_schema --cov=dinkster_server \
  --cov=dinkster_values --cov=dinkster_workers --cov-branch \
  --cov-report=term-missing
```

Adjust only the wrapper workspace path for another installation. Root pytest
defaults select up to eight workers with work stealing and no worker restart;
no explicit `-n` is needed. The wrapper holds one host-wide slot for the whole
command, including all workers. Record host load as measurement context, not
as another admission gate. Focused tests and collection-only checks do not use
the counted wrapper. This root run is not a replacement for the CPU-pack,
native Windows or live-jail environments in the matrix.

Native Windows evidence runs on a Windows machine; a Linux collection or
simulated platform is not a replacement. Preserve the existing assertions,
platform exclusions and coverage sources. Model/GPU comparisons are separate
from this non-model runtime matrix.

When reporting this matrix, name the executed lanes, exact head, commands,
results and logs, including each shard and the combined coverage percentage.
Missing inputs or an unexecuted lane are incomplete, never a pass. Hosted fast
feedback alone does not certify the matrix, and matrix evidence does not gate
landing. Do not dispatch or rerun hosted full validation to obtain the record.

The required `CI_RUNNERS` repository variable controls every job. Its
`linux`, `windows`, `macos`, and `forkLinux` values select GitHub-hosted
images. CPU Torch jobs pin AVX2 dispatch in their environment rather than
depending on machine labels.
When required private inputs are unavailable, the PR job records an explicit
not-run reason and fails instead of reporting unexecuted validation as green.

```json
{"linux":["ubuntu-latest"],"windows":["windows-latest"],"macos":["macos-latest"],"forkLinux":["ubuntu-latest"]}
```

`.github/workflows/full-validation.yml` runs Linux validation on batched
main pushes. Started runs finish; newer pending pushes replace older pending
pushes in the same concurrency group. It has no Linux schedule. Confirm a
landing is covered by a natural main run with
`git merge-base --is-ancestor <merge-sha> <run-head-sha>`.

`.github/workflows/windows-validation.yml` calls the Windows lanes once a day
at 10:23 UTC on main, never on a push or pull request. Its admission step
checks completed runs of that daily workflow for the exact commit, regardless
of conclusion. A completed run is not repeated, including cancelled and
skipped runs. One admission decision gates both Windows root/runtime tests
and Windows dependency smoke checks. The daily call does not run the Linux,
CPU-pack or probe-jail lanes. Windows runtime shards retain
the disposable ReFS clone-test environment formerly used on pull requests.

Coverage is collected inside the selected platform's test jobs rather than
by repeating the root suite. Linux root shards retain phase durations and
JUnit totals; Windows runtime jobs print all durations. Root jobs have
30-minute limits except Windows root shard 1 at 45 minutes; Windows runtime
jobs have 45-minute limits, CPU-pack jobs 20 minutes and probe jail 15 minutes.
These are failure backstops, not target completion times. `main-status`
uploads every aggregate result and requires every selected lane to pass.

Owners do not dispatch, rerun or cancel full validation to obtain evidence for
a pull request. Use naturally completed main runs; cancelled or replaced runs
are incomplete evidence. The separate release workflow defines its own build,
installation and publication checks and does not replace main full validation.

The root matrix uses Linux Python 3.12 and Windows Python 3.14.10. CPU-pack
jobs use the interpreter selected by their prepared bundle. Heavy jobs run
independently; CPU Torch jobs pin AVX2 dispatch without retry jobs.

Linux ARM64 artifact smoke and release installation are not part of the
supported matrix. Linux x64, Windows x64, and macOS use GitHub-hosted images.

## Coverage

The counted root command above uses the same branch-coverage source list as
the hosted Linux root and Windows runtime jobs. Each platform's run combines
its own artifacts and runs `coverage report` without a percentage floor. A
missing selected artifact or failed shard still fails the report job. Linux
and daily Windows percentages are separate reports, not combined coverage.

Code that runs in worker subprocesses (`host.py`,
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
- **Headless ANGLE mirror parity** (`test_image_mirror_angle.py`): workspace
  synchronization installs ANGLE through the image pack dependency on supported
  platforms. A core-only checkout can use `uv sync --group angle`. Run the test to
  execute every declared GLSL image mirror against its CPU parity corpus.
  It skips when ANGLE or a float-renderable EGL/GLES 3 context is unavailable.
- **CI root matrix**: Linux Python 3.12 and Windows Python 3.14.10 - Windows is the
  dominant user OS and keeps the tcp-transport and shm-lifetime rules
  honest.

Gaps we know about (roadmap, not accidents): multi-GPU CI runners, network
fault injection for the remote boundary, and property/fuzz tests for the
wire decoders, type-id grammar, and graph nesting - the highest-value fuzz
targets because their inputs come from outside.

## CPU checks

The `torch-cpu` matrix has four `node pack tests (CPU, shard N)` jobs and one
`vision pack tests (CPU)` job, using
`.github/actions/torch-cpu-suite`. It never acquires or executes model weights.
Environment setup runs in each job; vision Pyright runs in the vision job. The
vision package suites also run without their artifact
environment variables, so their weight-dependent cases skip while synthetic
input validation, preprocessing, batching, cache, fallback, tiling, and
architecture tests still run. The `dinkster-nodes-vision` distribution contains
the HED, upscale, Depth Anything V2, DETR, RT-DETR, EfficientSAM, BiRefNet,
Depth Anything V3, and SAM 3.1 model packs.

The CPU job excludes pinned model-weight acquisitions and each vision suite's
second real-artifact run. Sampling-runtime validation lives in
`dinkster-inference`; Dinkster's local gates validate the host and worker
integration:

```bash
uv run --no-sync python scripts/prepare_validation_inputs.py
uv run --no-project python scripts/setup_envs.py --variant linux-cpu
.venv/bin/python -m pytest -q
```

The preparation command checks out the revision in
`tools/evidence-revision.txt` beside the Dinkster checkout under
`.dinkster-evidence-source`, then materializes the core-owned benchmark shim and
workflow helpers into that checkout. Pytest refuses a missing or different
evidence revision, so local and hosted full validation execute the same benchmark
and acceptance sources. Access to the private evidence repository is required
only for the full maintainer suite, not to install or run Dinkster.

On Windows, run the same Python entry point with `--variant win-cpu` for CPU
checks or omit the variant to select the compatible NVIDIA bundle. Use native
`.venv\Scripts\python.exe` paths. Rerunning setup reasserts the selected
environment; `--force` rebuilds `.venv` without deleting the verified base store.
Run tools with `uv run --no-sync` to retain the manifest's binary wheels.

MediaRecorder WebM upload tests generate local samples only for the writable
`libopus` and `libvorbis` encoders present in the installed PyAV wheel. The
fixture gate always requires `libopus`; decode-side admission separately keeps
Vorbis WebM coverage when the wheel cannot generate it through `libvorbis`.

Run each vision package's full `tests` directory and the benchmark selector
above with `.venv/bin/python -m pytest -q` too. Supply the real-artifact
environment variables documented in the provider READMEs and composite
action; an absent artifact's skip is not real-model validation. Training and
GPU checks in the Torch README also remain required
where applicable. No CI input or repository variable changes local pytest
selection.

For physical workflow runs, use the tracked transcript harness and the exact
interpreter, preflight, catalog preparation, and evidence recipe in
[Workflow benchmark invocation](workflow-benchmark.md).

Full validation has no separate `model-tests` or `model-tests-gate` jobs.
Real-artifact checks use their documented provider inputs and are not implied
by a passing non-model CPU matrix.

## Tooling enforcement

`uv run ruff check .` and `uv run pyright` are part of validation, not
optional extras. `dinkster doctor` is the same idea pointed at packs: run it in
pack CI (the pack template wires it in) so pack regressions surface before
users hit them.
