# Pack dependency locks

A published pack carries one dependency lock per engine variant. The publisher
calls `dinkster_workers.pack_lock.resolve_pack_lock` with the variant's interpreter
and engine `constraints.txt`; a local directory or git install calls it once at
plan time. Resolution is binary-only. Installing a published lock never resolves.
The registry's pack artifact lock remains separate: it identifies pack source,
whereas this lock identifies the Python dependency closure.

The publisher/local-checkout CLI is `python -m dinkster_workers.pack_lock
<pack-root> --variant <variant> --generation-python <interpreter> --constraints
<constraints.txt> --output <lock.json>`. Repeat `--index <URL>` to declare indexes
or `--find-links <directory>` for an offline wheel source. `--requires-python`
records the pack's Python constraint; `--accelerator` selects its manifest extras.

`PackLock.write` writes JSON with these fields:

| Field | Value |
|---|---|
| `format` | `dinkster.pack-lock/1` |
| `pack` | Pack id, lowercase letters, digits, underscores and hyphens |
| `variant` | Base variant, such as `linux-cpu` |
| `requiresPython` | PEP 440 Python version specifier |
| `requirements` | PEP 508 requirements after alias normalization |
| `indexes` | Publisher-declared package index URLs, in priority order |
| `packages` | Complete resolved closure, including generation-provided packages |

Each package has a canonical distribution `name`, exact `version` and `wheels`.
Each wheel has its original `filename`, byte `size`, lowercase hex `sha256`, and
concrete source `url`. Only wheel archives are accepted, and filename name/version
must agree with their package entry. Candidate wheels allow the planner to select
a compatible Python/ABI/platform tag; a missing compatible candidate is a conflict,
not permission to build from source or resolve again.

## Alias table

`pack_lock.OPENCV_ALIASES` is the engine's alias data:

| Declared name | Resolved distribution |
|---|---|
| `opencv-python` | `opencv-contrib-python-headless` |
| `opencv-python-headless` | `opencv-contrib-python-headless` |
| `opencv-contrib-python` | `opencv-contrib-python-headless` |
| `opencv-contrib-python-headless` | `opencv-contrib-python-headless` |

The `ffmpeg` extra is removed for these aliases. Version constraints and markers
are preserved; other extras are not removed. The base distribution must satisfy
the resulting version constraint. This prevents two distributions owning `cv2`.

## Provisioning

`GenerationPackages.inspect(generation_python)` reads the actual interpreter's
installed versions, Python version, markers and compatible wheel tags.
`plan_pack(lock, facts, variant=..., own_environment_size=...)` checks shared
versions and Python compatibility before subtracting every generation-provided
package. A conflicting plan names the package and offers a separate environment
with the caller's measured base size. The caller presents this plan before making
changes, including before a generation swap.

`pack_environment.provision_pack` rechecks the plan and creates
`generations/<n>/packs/<pack>/` with `uv venv --python <generation interpreter>
--system-site-packages`. It reads verified stored wheels first, otherwise fetches
the lock's concrete source URLs. Only the remainder of the lock is installed,
using direct wheel paths with `--offline --no-index --no-deps`. No dependency
resolver or package index participates in this install. Downloads are checked for
both size and SHA-256 before admission to `<data>/wheels/<sha256>/<filename>`;
preexisting flat store entries are also accepted after verification. All uv calls
share `<data>/uv-cache`. Failed venv creation removes the incomplete venv.

The exact lock is saved as `<pack venv>/pack-lock.json`. A new engine generation
must plan and build fresh pack venvs from those locks, never reuse the old venvs.
The generation manager owns generation metadata and activation; this package
defines no alternative `generation.json`, `current`, `store` or journal format.

`pack_drift` compares venv-local distributions with the expected lock remainder
and reports missing, extra and differing packages with a rebuild-from-lock offer.
`scratch_venv` supplies a terminal interpreter over the generation without opening
the immutable generation itself to package installation.

## Generation subprocess adapter

The generation manager runs its target interpreter with
`-m dinkster_workers.pack_environment <preflight|rebuild|doctor|terminal>`.
All operations accept these explicit options:

```
--generation <root>/generations/<n>
--generation-python <generation>/python/bin/python3
--constraints <generation>/constraints.txt
--variant linux-cpu
--worker-protocol <integer>
--wheel-store <data>/wheels
--uv-cache <data>/uv-cache
--own-environment-size <base bundle bytes>
--uv <uv executable>
```

On Windows the generation interpreter is `python/python.exe`. Generation
metadata, current/store newline files, and journals belong to `dinkster-env`.
This adapter never changes them. `preflight` and `rebuild` also accept
`--source-generation <old generation>` and repeatable `--lock <path>` for new
installs or explicitly selected target-variant locks. Source pack locks are
mandatory; a supplied replacement may replace one pack's lock, but missing or
incompatible packs are never silently omitted. A variant switch without a matching
lock fails rather than resolving again. Setup may omit the source generation
only when the project has no `current` record; existing projects must supply it.

All operations also accept repeatable `--in-process-pack <id>` from the generation
manager's `inProcessPacks` field. Those packs retain their dependency lock under
`packs/<id>/` but have no venv. Preflight plans their shared dependencies first,
so conflicting versions across packs fail before downloads. Rebuild stages their
extra wheels into the target's own `python/` interpreter before isolated venvs.
The target must be inactive and have no final `generation.json` yet; the manager
writes that record and its immutable inventory after staging. On staging failure
the manager discards the whole target, including partially installed dependencies.
Doctor compares in-process dependencies against that interpreter, and terminal
selection for an in-process pack returns a scratch venv instead.

Stdout is exactly one JSON object, including on operational failures:

```json
{"packs": [{"pack": "example", "conflicts": [], "downloadBytes": 0, "ownEnvironmentBytes": 1000}], "changes": []}
```

`downloadBytes` counts selected wheels not already hash-verified in the store.
Exit 0 means no blocking conflict or drift; nonzero means the caller must not
swap. `preflight` plans every pack without downloads or venv mutation. `rebuild`
rechecks plans, installs into a fresh target, and removes packs it created if a
later pack fails. It finishes before the caller starts the target supervisor.
`doctor` is read-only and returns drift/rebuild offers in `changes`; confirmed
repair builds a fresh generation rather than modifying a running pack venv.
`terminal --pack <id>` returns the installed pack interpreter as the sole
`changes` entry. Without `--pack` it creates/returns `<root>/scratch/<n>/`.
Neither choice returns the generation interpreter.
Venv launchers are relocatable and Unix interpreter symlinks are relative to the
project's generation, so the manager's same-volume `pyvenv.cfg` home rewrite
does not leave an interpreter link pointing to the old project path.

Python callers use `GenerationContext(generation, generation_python, constraints,
variant, worker_protocol, wheel_store, uv_cache, own_environment_size, uv="uv",
in_process_packs=())`,
`preflight_generation(context, *, source_generation=None, locks=())`,
`rebuild_generation(context, plans)` and `doctor_generation(context)`.

An own environment uses a separately installed base of its required variant,
created by the generation manager rather than layered on the conflicting engine.
That base includes its own `dinkster-workers` distribution. The same adapter
consumes the lock over that base. Both local and remote worker sessions verify
the `protocol` integer in the worker hello before adopting its declarations, so
separately shipped workers cannot assume their engine's `workerProtocol`.
