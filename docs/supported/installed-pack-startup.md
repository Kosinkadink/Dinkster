## Installed pack startup

- First-party pack inference dependencies install from a pinned source archive
  without requiring a Git executable.
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
