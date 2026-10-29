## Installed pack startup

- First-party pack inference dependencies install from a pinned source archive
  without requiring a Git executable.
- Pack dependency provisioning can consume per-variant hashed locks, reject
  shared-package conflicts and layer only extra wheels over an engine
  interpreter without copying torch or opencv into the pack venv.
- In-process pack dependencies stage only into a fresh, inactive generation;
  their terminals use scratch venvs and doctor reports dependency drift.
- The official pack artifact builder produces deterministic source archives and
  an existing registry artifact lock from committed sources, preserving the
  fifteen pack identities and node entry points without importing node code.
- Packs may own exact node IDs with case, spaces and dots, without renaming
  them. Namespace and exact-node ownership reject overlapping owners through
  install, composition and reload. Pack and publisher identities remain
  lowercase with equivalent separator spellings.
- Locale catalogs preserve exact node and port IDs, including spaces, case,
  wildcard outputs and trailing dots. Node translations must belong to the pack.
- The native text, primitive, eager boolean, number conversion, seed and math
  expression nodes load through their ordinary pack manifest, including isolated
  execution, reload and exact-ID ownership checks.
- Native model and CLIP merging preserve source models and use the inference
  patcher for weighted changes. Model-specific merges use the longest matching
  block prefix and the first coefficient for unmatched keys. Their MODEL and
  CLIP types register through the ordinary pack manifest.
- Installed pack schemas and static choices are available before pack workers
  start. Workers activate on demand, including in-process packs and explicitly
  configured shared-process groups.
- Pack install, update, and doctor refresh the persisted schema catalog.
  Serving fails before binding when an installed catalog is missing or stale.
- An explicit generation serves only its locked pack sources, exact versions
  and recorded in-process or isolated hosting. Retained generations keep their
  own pack selection for rollback. Missing interpreters or changed source
  artifacts fail startup rather than falling back to installed distributions.
- Environment-managed engine starts pass the selected generation explicitly,
  including candidate startup before activation and retained rollback starts.
- Generation construction admits verified pack archives before sealing sources.
  A rejected source discards the incomplete target without changing the active
  generation or its running engine.
- The environment CLI can acquire an explicit mirror selection, build an offline
  generation and health-check it before activation, with confirmation and retained
  rollback. Registered workflow/addition execution and automatic release source
  acquisition are not connected yet; channel or variant migration is refused.
- The default local launcher prepares missing or stale catalogs before binding,
  then serves the browser application and engine API from one loopback origin.
- Released engines discover the installed frontend wheel without a frontend
  checkout. Release installation checks require the browser application document
  at `/` as well as the engine API.
- Health reports composed and failed counts and is not ready when composition
  is empty or any pack failed.
- Local operators can print a read-only requirement plan with
  `dinkster workflow workflow.json --variant linux-cpu` (optional `--server`
  and `--project`). Multiple files produce their union without rewriting them.
  `/api/workflows/plan` accepts `workflows`, `variant`, optional `project`, and
  optional digest-keyed `assetSources` URL lists. Saved attribution/version and
  semantic asset digests determine the missing items; unknown versions and
  unavailable saved-version manifests are explicit diagnostics. Presence checks
  verify vault bytes or selected project mounts without fetching. Resolver leads
  are advisory, global host metadata, not scoped grants. Remote scoped callers
  are refused; the route installs and downloads nothing.
- A server launched with `--engine-variant` supplies that declared target to
  workflow-open plans that omit `variant`. Explicit request variants remain
  available for read-only comparisons; no target is inferred from hardware.
- Local environment operators can run `python -m dinkster.workflow_requirements
  project-plan --project-root /absolute/project --source-generation /absolute/source
  --catalog-lock /absolute/release/official-packs.lock --artifacts /absolute/cache
  --variant linux-cpu --json`.
  This reads registered scoped workflow records and explicit additions from
  project.json, the source generation's dinkster.lock, and an admitted target
  registry lock. Cached `<artifact-digest-hex>.zip` archives are hash-verified
  before their manifests are read in disposable scratch. It returns requirements,
  a missing-item plan and input revision/digest snapshots without creating a
  generation, fetching content or changing workflow/library bytes. Only verified
  project-vault assets count as materialized; resolver leads remain advisory.
  Exact saved versions must be present in the selected release, even for packs
  already installed. Blocking errors produce nonzero status. First setup omits
  source generation; empty inputs need no catalog or archives. The optional
  `--generation` context does not require or create a target directory. Artifact
  admission, dependency preflight, provisioning and activation remain
  environment-owned operations.
