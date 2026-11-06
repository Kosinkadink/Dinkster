## Installed pack startup

- Published software history can rebuild a pruned generation without restoring
  workflows, models or outputs. Missing engine inputs expose read-only alternatives
  and all substitutions before a proposal-specific approval; `--yes` alone cannot
  approve an approximate restore. Shared tools are recorded, never replaced.
- Restoring recorded registry pack sources reacquires exact digest-pinned archives,
  not newer releases. Interrupted software restores retain a failed-attempt receipt
  while leaving successful history unchanged.
- Software history records the generation interpreter's wheel compatibility tags.
  Restore planning reports missing pack dependency wheels before activation,
  honors inherited base/engine and in-process dependencies, and verifies retained
  wheel bytes without creating project or cache files.
- Local-development software snapshots rebuild from their recorded manifests,
  retained interpreter and hash-pinned wheels, without resolving a checkout again.
  Missing local interpreter files or wheels are disclosed before activation.
- For snapshots without packs, a missing base can be replaced by an indexed base
  with the same native platform, Python and worker protocol when its locked engine
  requirements agree. Compatible engine/base pairs are offered when replacing
  either component alone cannot work. Proposals prefer fewer changed identities,
  preserve an obtainable compatible engine, and disclose every substitution for
  explicit approval rather than silently upgrading it.
- Standalone pack-only roots can be adopted with `setup --adopt-from` into a
  new, separate supervisor root using explicitly chosen published engine/base
  identities and target dependency locks. The plan displays exact pack sources,
  locks and in-process placement; the old root, shared store and user data stay
  untouched. Shared-venv groups currently refuse rather than change topology.
- Published base selection includes `linux-nvidia` for Linux x86_64 with
  glibc 2.28+, NVIDIA driver 580+ and compute capability 7.5+. Its Python
  3.14.10 environment carries torch 2.14.1+cu132 and the full CUDA closure.
  Explicit CPU selections remain available; an absent published variant is
  refused with the supported variants rather than resolved automatically.
- Catalog-driven packs explicitly declare their remote gateway node source.
  Offline doctor reports that source without treating zero static schemas as
  a missing-node warning; broken imports and invalid declarations still fail.
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
  Depth Anything v3 keeps its Pillow 12 minimum without rejecting a newer
  Pillow selected by a user-defined base.
- In-process pack dependencies stage only into a fresh, inactive generation;
  their terminals use scratch venvs and doctor reports dependency drift.
- The official pack artifact builder produces deterministic source archives and
  an existing registry artifact lock from committed sources, preserving the
  fourteen pack identities and node entry points without importing node code.
  Generation schemas remain engine-carried rather than separate official packs.
- Official publication uses each selected pack's declared version and engine
  range, with complete archive-bound locks for all six supported bases.
  Engine-only pushes select no pack. PR, fork and tag events build and
  preflight without submission; only trusted main may submit to the registry.
  A queued submission is not an admission verdict.
- Official source artifacts retain their committed manifest dependency
  declarations without adding workspace project requirements at publication.
  Publishing can emit separate per-variant hashed dependency locks against the
  target interpreter and engine constraints, without resolving during setup.
- Pack publication can submit publisher-produced per-base locks together with
  the exact source archive. The registry validates their archive and base
  bindings; submitting locks never resolves dependencies on the client.
- Prepared official source catalogs preserve dependency-owned codecs without
  claiming them as pack-owned declarations. Existing comfy and image node IDs
  execute from these archives without rewriting saved workflows or pack bytes.
- Core official pack manifests declare their existing third-party runtime
  requirements directly, including platform markers. Pack-to-pack dependencies
  and the host API contract remain separate from Python requirements.
- Pack manifests can declare an independent public pack version and a PEP 440
  engine compatibility range. Invalid or incomplete declarations are rejected
  without importing pack code. Every first-party node pack declares both,
  including development and engine-carried generation packs.
- The registry client can select the newest active pack version meeting a
  minimum version and target engine range across paginated results. Active
  prereleases inside those bounds remain eligible. Historical releases without
  an engine declaration are ineligible without blocking declared versions;
  malformed versions and nonempty ranges still refuse. The selected release is
  checked again before returning its immutable artifact identity.
- Official source archive builds can select exact pack ids and use each pack's
  own manifest version, retaining its declared engine range in the archive.
  Unknown pack ids and missing publication declarations refuse the build.
- Publication selection can identify changed official packs from committed Git
  ranges, or one exact manifest-versioned pack tag, without reading working-tree
  manifests. Engine-only changes select none; shared vision changes select all
  affected vision packs. Native dependency locking can consume that independent
  selected catalog without replacing its versions with the engine version.
- Official publication requires dependency locks for every selected pack and
  committed supported base before preparing maps or submitting any archive.
  Wrong source/base/lock identities, incompatible Python and nonportable wheel
  origins refuse the entire local preparation without publication requests.
  Prepared maps do not imply scanner acceptance or live default availability.
- Default-set admission checks resolve registry archives against the engine
  under test and require every pack to pass the registry scanner, including
  the isolated executable probe. Missing defaults, rejected packs and security
  review requirements fail the check without publishing any pack.
- Public archive admission verifies deterministic source ZIPs with bounded
  portable paths, validates manifest publication metadata, runs static doctor
  and a bounded isolated executable probe, and distinguishes security blocks
  from files requiring review. These checks need no private checkout, registry
  database or publication credentials. Missing or degraded probe isolation
  refuses rather than skipping executable checks.
- Registry archive inspection verifies the selected digest, safely extracts
  the manifest without importing pack code, and checks its pack id, version
  and engine range. Source publisher and namespaces describe that verified
  content, not the publishing audit actor; they do not authenticate publisher
  authority. Generation admission still rejects source identity mismatches.
  Registry downloads also require the ticket's nonnegative integer archive
  size and reject truncated or excess bytes before acquisition.
- Registry source preparation resolves the committed default ids and minimum
  versions, or an explicitly requested set, using the running engine's version.
  It returns a verified source lock and digest-addressed archives without
  changing a project or generation. Unavailable packs normally refuse the
  operation; optional missing-default reporting can return an empty source set.
- Published dependency-lock preparation verifies each selected registry
  archive's exact base binding, lock size and BLAKE3, and pack/variant identity.
  It exposes the complete set only after every lock verifies, without resolving
  dependencies, modifying the source selection or activating a generation.
- Registry preparation can accept a caller-owned lock producer for an unpublished
  base using the same selected sources, target interpreter and effective constraints.
  Empty custom lock results do not fall back to published base bindings.
- Installed-source preparation re-resolves official registry packs against the
  target engine without adding defaults, dropping installed packs or downgrading
  them. Recorded registries remain authoritative; third-party, local and git
  selections remain unchanged for generation admission. Unavailable releases refuse
  before the candidate source lock is written, even with missing-default opt-in.
- Generation source admission checks each declared pack engine range against
  the admitting interpreter's engine version before recording the source lock.
  An incompatible range names the pack/version, range and target engine; a
  refusal leaves the generation unsealed and its dependency inputs intact.
- Explicit pack application checks declared engine ranges before staging or
  returning an identical-lock no-op. External archives, cached and already
  staged sources, and grouped packs follow the same admission rule. Refusal
  preserves the active selection, staged content and generation history.
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
  User-defined portable variant names do not require a published channel row
  to start; an explicit engine variant must still match the generation.
- Environment-managed engine starts pass the selected generation explicitly,
  including candidate startup before activation and retained rollback starts.
- Stopped engine bindings retain their assigned ports across update and rollback;
  restarting an existing named binding defaults to its saved port.
- Generation construction admits verified pack archives before sealing sources.
  A rejected source discards the incomplete target without changing the active
  generation or its running engine.
- Generations retain cached engine release manifests byte-for-byte in their
  sealed inventory, checked against the selected release identity before pack
  preparation. Missing legacy caches remain supported; present corrupt or
  unreadable manifests refuse construction and discard the incomplete target.
- Admitted generation packs prepare their runtime catalogs in their selected
  interpreters before activation. Missing interpreters, changed sources or failed
  imports refuse the candidate; sealed generations cannot be prepared in place.
- Release mirror reads retry transient HTTP 5xx, connection failures and timeouts
  with bounded backoff. Remote artifact size or SHA256 mismatches share the same
  three-attempt time budget: short transfers resume, while full corrupt bodies
  are discarded before refetch. Terminal failures retain every failed attempt
  and Cloudflare request identifiers for HTTP errors when supplied. HTTP 4xx and
  local mirrors are not retried; exact size and SHA256 verification remain mandatory.
  Contributor setup reads only the selected host platform's published manifests.
- Windows contributor setup and interpreter preparation require uv 0.13.6 or
  newer for pinned CPython 3.14.10. An older uv is refused before those scripts
  download inputs or create environments; the error names both versions and
  the interpreter. Setup never upgrades uv automatically.
- The environment CLI can acquire an explicit mirror selection, build an offline
  generation and health-check it before activation, with confirmation and retained
  rollback. Setup plans show the selected engine binding, port and ordered GPU
  assignment before confirmation. Confirmed published setup selects registry
  defaults against the acquired target engine before creating an activation
  journal. Unavailable defaults are reported, and offline setup can start an
  explicitly empty generation. Update re-resolves installed official selections
  and refuses unavailable compatible versions before journal/project mutation.
  Exact-base dependency locks come from the registry; retained non-registry
  sources keep their recorded bytes/provenance and generation-owned locks.
  Explicit source/dependency inputs remain supported. Every selected
  source pack needs a dependency lock, including empty closures. Confirmed setup
  and updates acquire missing pack wheels from their locked URLs and verify size
  and SHA256 before offline construction, including wheels absent from release
  assets. Download failures identify the pack, URL and verification outcome.
  Offline repair and move still require intact retained wheels. Releases without
  pack acquisition support can still build from complete cached wheels. Confirmed
  updates can select another channel or variant, retaining the old selection until candidate health
  and requirement revalidation succeed. Failure recovers the prior selection and
  serving engine.
- Published engine and base locations can carry readable dated labels alongside
  their hash prefix. Existing hash-only locations remain installable, and local
  stores keep full-hash directory names for both forms. Contributor setup can
  locate an absent legacy base directory through the variant index, matching
  only its pinned full content id and verifying the manifest before installation.
- Status and doctor show installed base and engine labels from identity-matched
  local manifests without consulting the current channel. Legacy and custom
  manifests without labels remain supported and do not acquire invented labels.
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
- Status and doctor identify an active user-defined base by its variant name,
  torch version and torch index. The base's provenance is retained with the
  generation rather than inferred from the current variants file.
  Custom definitions name an exact torch version and HTTPS torch index;
  their optional extras index defaults to PyPI.
  Setup and update accept `--variants PATH`; otherwise definitions come from
  the project or store `variants.toml`. An ordinary update retains the active
  custom base even if its definition file has changed or disappeared. Pass
  `--variants PATH` explicitly to rebuild from a changed definition. Retained
  base metadata must still match the active generation's inventory.
- Custom pack dependency preparation reads the exact selected source archives,
  verifies their identities and resolves standard dependency locks against the
  target generation's constraints. It does not rebuild packs from a checkout
  or require a published lock for the custom variant.
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
- A pip-installed engine has no implicit official pack selection. Managed
  generations serve their admitted registry sources. Checkout development
  selects verified local sources with each pack's declared version, without
  requiring an installed pack distribution.
- Checkout defaults retain their managed source lock without requiring an
  installed standard-suite distribution.
- Development scaffolding is workspace-only and is excluded from engine and
  official-pack releases.
- Engine release assembly records variant-specific dependency wheel identities
  without uploading third-party wheels. Finalization rejects unexpected payloads
  and enforces a 100 MB budget excluding native tools ZIPs.
- Native tools builds retry transient HTTPS input failures up to three times,
  preserving exact size and SHA256 admission. Certificate errors, HTTP 4xx and
  corrupt payloads fail without retry; exhausted failures retain attempt details.
- The environment client acquires its native tools ZIP from the channel's
  independently versioned tools publication without engine or base payloads.
  Planning creates no cache; confirmed acquisition checks manifest and archive
  integrity and refuses corrupt tools rather than falling back to engine assets.
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
- Health reports composed and failed counts when packs are selected and is not
  ready when their composition is empty or any pack failed. An explicitly empty
  source selection starts a healthy zero-pack engine without bundled defaults.
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
  A selected custom generation uses its validated base and inventoried torch
  build metadata for device compatibility, without importing torch or probing
  host hardware. Missing, nonliteral, repeated or conflicting device declarations
  refuse planning rather than guessing a device. GPU-only packs still refuse a
  CPU custom base. Unknown custom comparison targets require their own generation
  rather than a guessed device.
  Project plans can select that target with `--generation`; installed pack
  versions still come from `--source-generation`. Applying a project plan checks
  device compatibility against the target generation, not the previous base.
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
  Rollback rebuilds the preceding software snapshot as a new generation, preserving
  current project settings and any edits made after the recorded activation.
- Unconfirmed `dinkster-env update` and `update --plan` report saved workflow
  requirements without acquiring target payloads or claiming source feasibility.
  Confirmed updates check those requirements against the same registry-selected
  sources used for construction, using the acquired target engine before creating
  the activation journal. Unavailable saved versions and changed workflow inputs
  refuse before construction. Catalog identities and input revisions are checked
  again before application and activation; sealed target checks remain mandatory.
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
- `dinkster-env history` lists successful software activations in order, and
  `history show ID` displays the recorded software and changes. History survives
  generation-directory collection. Snapshots exclude workflows, models, outputs,
  libraries and mounts. Interrupted activation does not become a successful entry;
  completed activation history publication is recoverable without duplication.
- `dinkster-env restore ID` rebuilds a published software snapshot in a new
  generation after confirmation. Recorded identities are preserved; unavailable
  inputs refuse activation. Tools versions are displayed but tools are never
  replaced. Failed attempts remain separate from successful history. Restore
  preserves existing history, project settings, workflows, models and outputs.
- `dinkster-env rollback` rebuilds the preceding successful history entry, even
  after its generation directory was pruned. It uses restore's missing-input and
  proposal-approval contract, appends a source-linked entry with reason `rollback`,
  and leaves project settings and user data unchanged.
