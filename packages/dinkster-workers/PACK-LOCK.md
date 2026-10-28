# Pack dependency locks

A published pack carries one dependency lock per engine variant. The publisher
calls `dinkster_workers.pack_lock.resolve_pack_lock` with the variant's interpreter
and engine `constraints.txt`; a local directory or git install calls it once at
plan time. Resolution is binary-only. Installing a published lock never resolves.
The registry's pack artifact lock remains separate: it identifies pack source,
whereas this lock identifies the Python dependency closure.

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
