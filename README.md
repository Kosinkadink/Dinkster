# Dinkster

Dinkster is a pre-release local engine and browser editor for image, video,
audio, and model-training workflows. Workflows are node graphs executed by
isolated packs, with typed values, content-addressed caching, managed memory,
and optional remote workers. Dinkster runs native model families and can import
existing ComfyUI workflows; an optional ComfyUI checkout adds compatibility
for node packs that have not been ported.

To run Dinkster from source, clone Dinkster and Dinkster-Frontend beside each
other, build the frontend, install the locked Python workspace, and launch:

```sh
cd Dinkster-Frontend
pnpm install --frozen-lockfile
pnpm --filter @dinkster/app build
cd ../Dinkster
uv sync --python 3.12 --all-packages --frozen
uv run dinkster setup
uv run dinkster
```

The editor opens at `http://127.0.0.1:3639`. See the
[browser editor quickstart](docs/quickstart.md) for model folders, execution
environments, and the first-image walkthrough.

Packs are independently installable Python distributions with a
`dinkster-pack.toml` manifest. A pack declares its schemas, runtime entry
point, dependencies, and optional frontend module; the server composes those
contracts and starts a pack worker only when execution needs it. Start with
the [pack authoring guide](docs/pack-authoring.md) and
[pack template](templates/pack/).

- Launch the browser editor: [docs/quickstart.md](docs/quickstart.md)
- End-user backend and Desktop installation: [docs/install.md](docs/install.md)
- What is supported today: [SUPPORTED.md index](SUPPORTED.md)
- Architecture and design rationale: [DESIGN.md](DESIGN.md)
- Load-bearing invariants: [docs/hazards.md](docs/hazards.md)
- Multi-GPU host setup and verification: [docs/multigpu-hosting.md](docs/multigpu-hosting.md)
- Inbound authentication and capability policy: [docs/auth.md](docs/auth.md)
- Pack sandbox platform support and security limits:
  [packages/dinkster-workers/README.md](packages/dinkster-workers/README.md#serving-sandbox)
- Gateway remote nodes: the default-installed `dinkster-nodes-remote` pack
  discovers `dinkster.remote.*` schemas from an explicitly configured
  dinkster-gateway catalog and executes authenticated jobs with verified asset
  downloads; see [remote nodes](docs/supported/remote-nodes.md).
- Writing a pack: [docs/pack-authoring.md](docs/pack-authoring.md), starting
  from the template in [templates/pack/](templates/pack/) - a complete
  doctor-clean pack (manifest, nodes, custom type, icon, tests, CI) kept
  healthy by this repo's own suite. `uv run dinkster-doctor <pack-dir>` is the
  pack linter and publish gate.
- Companion frontend: [Dinkster-Frontend](https://github.com/Kosinkadink/Dinkster-Frontend)

Status: in progress. No stable package or Desktop release exists yet. See
[installation](docs/install.md) for prerequisites and platform limits. Native
and ComfyUI-compatibility workflows run through the same typed graph, server,
worker, and sampling boundaries. The exact model, node, dtype, training, and
compatibility coverage is linked from the
[SUPPORTED.md index](SUPPORTED.md). `uv sync --all-packages` (or
`scripts/setup_envs.sh` on Linux/macOS or `scripts/setup_envs.ps1` on Windows,
which also build the torch/GPU test venvs - see
`packages/dinkster-inference-torch/README.md`), then:

On Linux and Windows, the setup script installs `dinkster-kitchen==0.2.35.post1`
and `dinkster-aimdo==0.5.5.post2` from PyPI. macOS installs the pure-Python
kitchen wheel and does not install Aimdo because the mechanism does not support
that OS.

Both setup scripts pin root synchronization to this checkout's torch-free
`.venv`, regardless of `UV_PROJECT` or `UV_PROJECT_ENVIRONMENT`. They manage
`.venv-torch` and `.venv-gpu` separately with `uv pip`. Do not target those
Torch environments with project `uv sync`: exact sync can remove their
platform-specific torch and kitchen wheels.

Maintainer parity validation requires the maintained evidence checkout as a
sibling, or an absolute `DINKSTER_EVIDENCE_ROOT` override. The
`tools.inference_parity` imports, parity
manifest paths in harness tests, `scripts/benchmark_*` loaders, `benchmarks/`
fixtures, and `packages/dinkster-acceptance` install/test paths refer to that
checkout, not to files in core. Harness tests run with the evidence checkout as
their working directory. Capability evidence selectors under
`tools/inference_parity/` use the same external root. The receipt generator
writes its `docs/comfy-confidence-receipts/` there; `--check` verifies those bytes.

Coverage tests generate JSON into temporary directories from pinned input
checkouts. Set `WORKFLOW_TEMPLATES_ROOT` to workflow_templates at
`d3b4a9e89573162b005961865164c18c8ae2206b` and `COMFYUI_ROOT` to ComfyUI at
`15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`; defaults are sibling checkouts with
those names. The historical research comparison reads the immutable report in
Dinkster commit `fd02ae351d2ba3eb84f5a68cb89fe208a7365ae3` with `git show`; shallow
clones must fetch that commit. Current research lives only in the workspace.
`docs/comfy-source-parity-baseline.json` is a maintained debt baseline, not a
generated report, and remains tracked. Generated translation/capability JSON
is ignored; CI regenerates it and verifies the tracked Markdown separately.
CI's input action requires the read-only `DINKSTER_EVIDENCE_READ_KEY` secret.

For cloud acceptance archives, pass both `--commit` and `--evidence-commit` to
`scripts/prepare_cloud_acceptance.py`. It assembles the pinned external acceptance
package with core in a disposable workspace, updates that workspace's lockfile,
performs a locked install and import check, and records both Git identities.

- `uv run pytest` - test suite (incl. the one-way dependency rule, hazard H6)
- `uv run pyright` - static type checking (strict for `packages/`, standard
  for `src/` and `tests/`)
- `uv run ruff check .` - lint
- `uv run dinkster demo` - demo: toy graph through the real engine; shows caching,
  partial re-execution, non-idempotent nodes, and value interrogation
- `uv run dinkster-pack prepare-catalogs --defaults` - prepare installed
  default-pack dependencies and persist their schemas before serving; run it
  after installation and rerun it after checkout or dependency changes
- `uv run dinkster-serve` - native protocol server (binds a zero-node diagnostic
  host, then announces installed catalogs with exact provenance without
  starting pack workers; workers activate on demand. Each repeatable
  `--pack path/to/dinkster-pack.toml` runs isolated
  in its own process and attributes on `/api/nodes`; `--comfy-root <ComfyUI install>`
  adds the translated core surface as the `comfy` pack on that install's
  interpreter, and each `--legacy-pack <custom nodes dir>` loads unmodified
  in the quarantine worker, attributed as `comfy.<pack>` with a shared
  default legacy badge - a `dinkster-pack.toml` carrying just
  `[pack.presentation]` in the pack directory overrides it):
  `GET /api/nodes`, `GET /api/workers`, `POST /api/jobs`,
  `GET/DELETE /api/jobs/{clientId}/{jobId}`,
  `GET/DELETE /api/jobs/by-ref/{jobRef}`,
  `GET /api/jobs/by-ref/{jobRef}/events?after=N` (cursor replay),
  `GET /api/events` (additive WebSocket delivery), `GET /memory/status`

  `--remote-workers remotes.toml` composes `dinkster_workers.service` daemons
  on other machines into the same surface, executing their node types
  remotely; see [docs/remote-workers.md](docs/remote-workers.md).

  Execution results use layered memory and persistent disk caching by default
  under `<library-root>/execution-cache`, so persistable idempotent nodes can
  hit after a server restart. Pass `--execution-cache-mode memory` to opt out;
  cache budgets and paths are documented in [docs/serve-cli.md](docs/serve-cli.md).

  Memory budgets may be declared as repeatable command-line entries
  (`--memory-budget ram=24G --memory-budget vram:cuda:0=20G`) or persisted
  as defaults in `<library-root>/memory.toml`; see
  [docs/serve-cli.md](docs/serve-cli.md) for the full `memory.toml` format
  and a complete reference of every `dinkster-serve` argument. Runtime-editable
  controls and their granular permission gate are documented in
  [docs/settings-api.md](docs/settings-api.md). Native fp8 matrix
  multiplication is an explicit `--fp8-matmul` opt-in (or persisted
  `fp8-matmul` setting); unsupported workers refuse it before model load.

  Mount catalogs support query-first keyset paging with
  `GET /api/mounts/{mountId}/entries`. Its optional query parameters are
  `q` (case-insensitive full virtual-path substring), `path` (folder path
  below the mount root, default `""`), `recursive` (`true` by default,
  or `false` for immediate children only), `kind` (a namespaced semantic
  asset-kind filter), `cursor`, and `limit` (default 100, range 1-500).
  Rows from semantic mounts carry the mount-wide `kind`; recognized image,
  audio, and video rows on ordinary mounts carry their media kind. A cursor
  is opaque and bound to the mount and all normalized filters. A first-page
  immediate-child request such as
  `GET /api/mounts/models/entries?path=checkpoints&recursive=false&limit=2`
  returns the unpaged immediate folder names alongside the paged files:

  ```json
  {
    "entries": [
      {
        "virtualPath": "mounts/models/checkpoints/model.safetensors",
        "name": "model.safetensors",
        "digest": "sha256:<digest>",
        "size": 123,
        "mediaType": "application/octet-stream"
      }
    ],
    "folders": ["archive", "sdxl"],
    "cursor": "<opaque>"
  }
  ```

  Cursor pages omit `folders`. Invalid/traversing paths and malformed or
  filter-mismatched cursors return 400; unknown mounts return 404.
- `uv run dinkster-isolated` - isolation demo: the image pack runs out-of-process
  behind IsolatedWorker; the parent never imports it (schemas arrive over
  the wire), image payloads cross via shared memory, and per-edge boundary
  diagnostics print live (`DINKSTER_SLOW_TESTS=1 uv run pytest` also covers
  provisioning a real per-pack venv with uv)
- `uv run dinkster-port <legacy-pack> --name my-pack` - generate a doctor-clean
  native pack skeleton from a ComfyUI pack (v1 `NODE_CLASS_MAPPINGS`, V3
  `comfy_entrypoint`, or a mixed pack shipping both): real translated schemas
  (source names kept as aliases), loud `execute()` stubs carrying the source
  as reference, tests and README; the pack imports only in a disposable probe
  subprocess under the ComfyUI install's interpreter (`--comfyui-root` or
  `DINKSTER_COMFYUI_ROOT`). See docs/pack-authoring.md.

# Fleet ingress

`dinkster-station` can bind an optional stateless ingress port configured by
`[ingress]` in `installs.toml`. It uses durable SQLite ownership to route job
keys and job references, fans out honest fleet aggregates, and merges event
WebSockets. Slice 2a intentionally executes on the configured primary only;
multi-engine spreading remains gated on fleet homogeneity and shared-library
proofs. See `packages/dinkster-supervisor/README.md`.
