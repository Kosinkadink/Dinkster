## Installed pack startup

- First-party pack inference dependencies install from a pinned source archive
  without requiring a Git executable.
- Windows isolated packs and catalog probes support native extensions beneath
  ordinary user data roots, including Depth Anything v2/v3 environments at
  roots of at least 59 UTF-16 code units. A shorter data directory is not required.
- Pack dependency provisioning can consume per-variant hashed locks, reject
  shared-package conflicts and layer only extra wheels over an engine
  interpreter without copying torch or opencv into the pack venv.
- The BiRefNet, Depth Anything v2/v3, DETR, EfficientSAM, HED, RT-DETR,
  SAM 3.1 and upscale packs declare base-carried dependencies without exact pins.
  The generation's effective constraints select those versions; incompatible
  requirements still fail rather than shadowing the engine's packages.
- In-process pack dependencies stage only into a fresh, inactive generation;
  their terminals use scratch venvs and doctor reports dependency drift.
- The official pack artifact builder produces deterministic source archives and
  an existing registry artifact lock from committed sources, preserving the
  fourteen pack identities and node entry points without importing node code.
  Generation schemas remain engine-carried rather than separate official packs.
- Official source artifacts retain their committed manifest dependency
  declarations without adding workspace project requirements at publication.
  Publishing can emit separate per-variant hashed dependency locks against the
  target interpreter and engine constraints, without resolving during setup.
- Prepared official source catalogs preserve dependency-owned codecs without
  claiming them as pack-owned declarations. Existing comfy and image node IDs
  execute from these archives without rewriting saved workflows or pack bytes.
- Core official pack manifests declare their existing third-party runtime
  requirements directly, including platform markers. Pack-to-pack dependencies
  and the host API contract remain separate from Python requirements.
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
- Stopped engine bindings retain their assigned ports across update and rollback;
  restarting an existing named binding defaults to its saved port.
- Generation construction admits verified pack archives before sealing sources.
  A rejected source discards the incomplete target without changing the active
  generation or its running engine.
- Admitted generation packs prepare their runtime catalogs in their selected
  interpreters before activation. Missing interpreters, changed sources or failed
  imports refuse the candidate; sealed generations cannot be prepared in place.
- The environment CLI can acquire an explicit mirror selection, build an offline
  generation and health-check it before activation, with confirmation and retained
  rollback. Setup plans show the selected engine binding, port and ordered GPU
  assignment before confirmation. Releases carrying an official source lock feed
  their verified source ZIPs into target-interpreter admission automatically;
  missing admission support refuses activation. Selected-variant dependency locks
  are acquired from release artifacts unless explicitly overridden. Every selected
  source pack needs a dependency lock, including empty closures. Confirmed setup
  and updates acquire missing pack wheels from their locked URLs and verify size
  and SHA256 before offline construction, including wheels absent from release
  assets. Download failures identify the pack, URL and verification outcome.
  Offline repair and move still require intact retained wheels. Releases without
  pack acquisition support can still build from complete cached wheels. Confirmed
  updates can select another channel or variant, retaining the old selection until candidate health
  and requirement revalidation succeed. Failure recovers the prior selection and
  serving engine.
- Base acquisition verifies ordered tar+zstd parts before streaming extraction.
  No external decoder executable or runtime decoder download is needed. Unsafe
  members refuse extraction before promotion. Internal symlinks are preserved
  where supported by the host filesystem; unsupported or dangling link graphs
  refuse extraction before promotion.
- Linux base archives preserve distinct case-sensitive interpreter paths, including
  terminfo entries. Windows targets and case-insensitive destinations reject
  case collisions. Archive permission bits are preserved. A missing interpreter
  execute bit is repaired with a warning, and its isolated startup must pass
  before the base becomes visible.
- Update plans include explicit-addition needs through the
  active generation's read-only planner. Selected release sources must already be
  verified in the local cache, or supplied through explicit catalog/cache inputs.
  Missing or corrupt cache and unavailable saved pack versions refuse planning;
  plans never fetch payloads, repair caches or activate a generation. Registered
  workflows refuse planning without a side-effect-free library snapshot reader.
- Failed initial setup releases its own metadata and store registrations so a
  retry can select another data directory. Cached content, pre-existing settings,
  other projects' references and the diagnostic journal remain intact.
- Explicit `dinkster-env setup --from-scratch` builds from the checkout lock
  for development or when a published bundle does not fit. Contributor setup
  accepts the same opt-in, with an explicit host interpreter override when its
  platform row is missing. Local generations report their origin and recorded
  drift; channel updates refuse them and offer a fresh published installation.
  Default setup and Desktop remain published-bundle consumers.
- Confirmed environment garbage collection keeps the current generation and its
  successful predecessor, and preserves cache references from other registered
  projects. Interrupted collection recovers before another confirmed mutation;
  unreadable project references refuse shared-cache deletion. Libraries remain
  untouched.
- `dinkster-env doctor --repair` plans an offline rebuild from the active release's
  local wheels and retained base. Confirmation creates and health-checks a new
  generation without editing the drifted tree. Missing or corrupt local inputs
  refuse repair; failures keep the prior generation selected and serving.
- `dinkster-env move` plans before confirmation, preserves project assets and
  named engine assignments, and checks destination health before committing.
  Same-volume relocation keeps shared packages; cross-volume relocation rebuilds
  retained generations offline in the destination store. Old roots retain
  diagnostic journals and refuse stale startup. Interrupted preparation recovers
  the original project before another confirmed mutation.
- A target-interpreter command stages sources from an existing registry lock
  and digest-addressed local archives, without network access or dependency
  resolution. JSON results report admission failures before activation.
- Managed workflow plans default to the selected generation's recorded variant.
  A conflicting launch variant refuses startup; an explicit request variant
  remains a read-only compatibility comparison, not an environment switch.
- The default local launcher prepares missing or stale catalogs before binding,
  then serves the browser application and engine API from one loopback origin.
- A pip-installed engine prepares the bundled default pack catalogs before
  binding; the core worker is selected once with its configured runtime options.
- Checkout defaults retain their managed source lock without requiring an
  installed standard-suite distribution.
- Development scaffolding is workspace-only and is excluded from engine and
  official-pack releases.
- Engine release assembly records variant-specific dependency wheel identities
  without uploading third-party wheels. Finalization rejects unexpected payloads
  and enforces a 100 MB budget excluding native tools ZIPs.
- Confirmed published setup/update acquires non-base wheels by exact locked
  filename, size and SHA256 from their recorded origins or `--package-index`.
  The package index is saved for updates, independently of channel `--mirror`.
  Selected bases must match their variant install lock; intact shared-store
  wheels need no index access, and generation construction remains offline.
- Setup/update refuses non-base dependencies whose version requirements are
  not satisfied by the selected base, naming the requiring distribution and
  base version before constructing or activating a generation.
- Released engines discover the frontend bundled inside the engine wheel without a frontend
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
  Plans support packs supplied with either a directory or an explicit manifest
  path through `--pack`.
- Local environment operators can run `python -m dinkster.workflow_requirements
  project-plan --project-root /absolute/project --source-generation /absolute/source
  --catalog-lock /absolute/release/official-packs.lock --artifacts /absolute/cache
  --variant linux-cpu --json`.
  This reads registered scoped workflow records and explicit additions from
  project.json, the source generation's dinkster.lock, and an admitted target
  registry lock. Cached `<artifact-digest-hex>.zip` archives are hash-verified
  before their manifests are read in disposable scratch. It returns requirements,
  a missing-item plan and input revision/digest snapshots without creating a
  generation, fetching content or changing project/vault bytes or logical library
  records/revisions. SQLite may create or touch its ordinary WAL/shared-memory
  coordination files. A concurrent committed workflow edit either refuses the
  in-flight snapshot or is visible in the next plan. Readable committed data is
  accepted even when a writer exited abruptly; uncommitted changes stay invisible.
  SQLite open/read errors are reported by name without a rollback, checkpoint or
  unsafe fallback from the planner. Only verified project-vault
  assets count as materialized; resolver leads remain advisory.
  Exact saved versions must be present in the selected release, even for packs
  already installed. Blocking errors produce nonzero status. First setup omits
  source generation; empty inputs need no catalog or archives. The optional
  `--generation` context does not require or create a target directory. Artifact
  admission, dependency preflight, provisioning and activation remain
  environment-owned operations.
- `project-apply --yes --generation /absolute/target` consumes that exact approved
  project-plan document on stdin. It revalidates workflow revisions and source
  leads, verifies the target's admitted pack sources and variant, and materializes
  only approved assets. Concurrent workflow edits abort publication; workflow
  records are never rewritten. This core adapter does not activate a generation;
  `dinkster-env update --yes` owns candidate construction and activation, rechecking
  registered workflow inputs and target requirements after candidate health checks.
  Failed acquisition or revalidation retains the previous serving generation and
  diagnostic journal. First setup with requirements still needs an existing
  generation interpreter. Channel/variant selection replans requirements for the
  selected target; workflow records and other project fields are preserved.
  Rollback restores the retained selection without rebuilding, refusing when
  settings were edited after selection rather than discarding user changes.
- A local channel builder accepts completed engine release files and selected
  base manifests, verifies engine payload hashes and base/engine compatibility,
  and atomically writes the existing stable/main channel format. It preserves
  publisher inputs and requires no network. Hosted channel publication and
  published-base selection are not provided by this local command.
- Hosted publication retries transient mirror upload failures with bounded
  backoff and verifies all mirrored payload bytes before exposing the channel.
  Prerelease housekeeping reports exhausted transport retries independently;
  its failure does not prevent a successfully published release from advancing
  the verified channel. Stable and draft releases are not pruned.
- Confirmed gc collects obsolete generation trees while retaining current and
  the actual successful predecessor. Interrupted quarantine is restored; only
  committed trash is deleted. Generation collection leaves shared-store and
  project-library contents unchanged and refuses live obsolete engines.
