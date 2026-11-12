## Installed pack startup

- Pack installation, retained rollback and serving selection enforce locally
  cached verified security notices. Known malicious artifacts and explicitly
  denied vulnerable releases refuse without deleting retained bytes or changing
  the active selection. Retained vulnerable releases remain usable while their
  decision is pending; a new vulnerable release requires exact-notice approval.
  Installer operations refresh independently trusted issuers before checking,
  including configured mirrors. Failed refresh preserves cached decisions and
  blocks without inventing incidents. Feed age does not refuse by default;
  explicit age limits and vulnerability refusal apply to retained releases too.
  Serving hosts periodically refresh configured security feeds. A newly blocked
  isolated worker stops, including work pinned before the notice arrived;
  pinned work from a retired composition remains subject to the same policy.
  Unrelated workers continue. Pending vulnerable work continues until denied.
  A signed lift clears the classification without resuming a stopped worker.
- Managed engine and pack environments retain the exact selected wheel identities and
  distinguish them from unused lock alternatives and inherited dependencies.
  The records are sealed with the generation and preserved in software snapshots.
- Managed generation pack selection checks cached notices against sealed library
  identities and transitive pack sources. A malicious dependency blocks its
  consumer as affected, not malicious. Missing or changed wheel provenance refuses
  selection when security checking is configured. Generation selection refreshes
  before checking. Catalog preparation also checks before importing pack code,
  using the provisioned wheel receipts before the generation is sealed. Sealed
  generations continue to require the recorded receipt hashes.
- Verified ownership notices accept an explicit operator-owned previous owner.
  An omitted previous owner, empty new publisher or unchanged owner is refused
  without losing cached release notices.
- Plain `pip install dinkster` supports CLI, catalog serving and torch-free
  packs without installing torch or the inference runtime. Install
  `dinkster[inference]` for in-process inference packs; missing inference
  dependencies report that extra rather than installing them implicitly.
  Managed generations and contributor setup retain inference support.
- Static-only registry installs support both the release catalog and the split
  catalog with per-pack version files. Selection and acquisition share a snapshot
  within each operation; the next operation reads current publication state.
  Unknown index schemas refuse with an instruction to update Dinkster.
- Managed generation hosts can opt into local human pack previews with
  `dinkster-serve --generation /absolute/project/generations/N --allow-pack-install`
  or the inherited `DINKSTER_ALLOW_PACK_INSTALL=1` setting. Without opt-in the
  routes do not exist. Exact selections produce a review and copyable commands
  for that project. Run the plan command, inspect its output, then run the apply
  command before its 15-minute approval expires. Reconnect and refresh after
  application. HTTP apply always refuses with `registry.managed-generation-required`;
  it never installs or activates packs. HTTP application is Desktop-gateway-owned.
  Remote, agent and delegated callers cannot use these routes. Authenticated
  loopback callers require a nondelegated human with settings-write authority.
- The default install selection contains all 35 ComfyUI-origin first-party
  packs plus the remote pack. Checkout serving verifies their locked sources
  and isolates every selected pack. The engine carries no draft node packs.
- Official packs retain independent versions when the engine is released;
  changed remote pack dependencies use a new version rather than replacing
  an immutable published artifact.
- Dependency locks declaring engine API compatibility can layer unchanged on
  later engines inside that range. The target must supply the API, and all
  ordinary shared dependency versions must still agree exactly. Older exact
  locks retain their exact-version requirements.
- Published software history can rebuild a pruned generation without restoring
  workflows, models or outputs. Missing engine inputs expose read-only alternatives
  and all substitutions before a proposal-specific approval; `--yes` alone cannot
  approve an approximate restore. Shared tools are recorded, never replaced.
- Restoring recorded registry pack sources reacquires exact digest-pinned archives,
  not newer releases. Acquisition collects unavailable or mismatched sources
  before copying any of the set into the retained cache, without provisioning
  or activating a standalone installation or importing pack code. Interrupted
  software restores retain a failed-attempt receipt while leaving successful
  history unchanged.
- Restore can recover selected source archives retained by another engine release
  when the recorded source cache was pruned. Retained bytes must match the locked
  BLAKE3 identity; corrupt copies cannot displace an intact retained archive.
- Restore planning checks exact pack source availability. When retained archives
  are gone, acquisition probes use disposable storage and verify the recorded
  digest; changed or unavailable sources are disclosed without changing the install.
- Software restore rebuilds isolated and in-process pack dependencies from the
  snapshot's complete locks, including after its old generation was pruned;
  it does not inherit packs from the current generation.
- Software history records the generation interpreter's wheel compatibility tags
  and requirement markers. Replacement previews identify the exact selected
  engine/base wheels; reconstruction must retain those identities.
  Restore planning reports missing pack dependency wheels before activation,
  honors inherited base/engine and in-process dependencies, and verifies retained
  wheel bytes without creating project or cache files.
- Local-development software snapshots rebuild from their recorded manifests,
  retained interpreter and hash-pinned wheels, without resolving a checkout again.
  Missing local interpreter files or wheels are disclosed before activation.
- A missing base can be replaced by an indexed base
  with the same native platform, Python and worker protocol when its locked engine
  and unchanged pack dependency requirements agree. Packs, source locks and
  in-process placement are preserved. Engine replacements and combined engine/base
  proposals also preserve packs when recorded pack manifests declare compatible
  engine ranges and unchanged dependency locks remain compatible and obtainable.
  Entries without those declarations do not guess compatibility. Proposals prefer
  fewer changed identities, preserve an obtainable compatible engine, and disclose
  every substitution for explicit approval rather than silently upgrading it.
- Standalone pack-only roots can be adopted with `setup --adopt-from` into a
  new, separate supervisor root using explicitly chosen published engine/base
  identities and target dependency locks. The plan displays exact pack sources,
  locks and in-process placement; the old root, shared store and user data stay
  untouched. Shared-venv groups currently refuse rather than change topology.
- Adoption displays source pack distribution versions and differences from target
  dependency locks, with separate shared-store accelerator scopes. Inspection is
  metadata-only and never executes old Python or .pth files. Ambient/inherited
  packages are not claimed as captured, and changed metadata invalidates approval.
  Linked source roots, shared stores, archives and their ownership ancestors are
  refused, including Windows directory junctions.
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
  manifest-declared pack identities and node entry points without importing node
  code. New node-pack directories are discovered without a family allowlist.
  Development scaffolding stays unpublished.
- First-party source resolution reads runtime IDs and node entry modules from
  manifests and distribution names from project metadata or installed wheel
  records. Runtime, distribution and import names need not match. Discovery
  does not import node code or add a pack to the selected defaults.
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
- Published lock preparation inspects the whole selected set before creating
  lock output, reporting all missing or invalid bindings together. Its
  `--offline --index-snapshot PATH` form uses retained index and lock metadata
  without network access or dependency resolution; missing or corrupt inputs
  refuse without creating the output or repairing the snapshot.
- Registry source planning can inspect an exact engine release's default pack
  declaration and published locks without acquiring archives or writing a cache.
  Its plan includes selected versions, provenance, archive identities and an
  upper bound for pack wheel bytes. Unavailable releases and invalid locks are
  reported together. Offline metadata discovery reads only retained channel,
  engine and base records and reports missing engine/base records together.
- Registry planning also supports exact additions/replacements and compatible
  updates against an installed source lock. Unselected sources stay unchanged,
  including local development packs. Updates keep recorded registry provenance
  and do not downgrade; missing releases and invalid published locks are reported
  together without acquiring pack payloads or changing the project.
- Published pack install/update commands on an active managed project display
  the retained engine/base/variant, old/new identities, proposed generation,
  published locks, verified cached versus downloadable payloads, wheel-byte
  upper bounds and worker cost before confirmation. Non-terminal use
  requires `--yes` or a saved plan. Declining leaves the project and store
  unchanged. Saved reviews expire after 15 minutes and refuse changed project
  or selection metadata; successful add and update activate fresh generations
  without package-index resolution, retaining the engine and base.
  Missing retained bases refuse early rather than listing later payload failures.
  Local/git selections require explicit development rather than mixed installation.
- Standalone pack installation and serving are development-only. Registry-bearing
  saved plans, snapshots, workflow stamps and rollback targets refuse instead of
  provisioning published dependencies without reviewed base/variant locks.
  Published source acquisition remains available to managed software restore;
  catalog preparation cannot fall back to range resolution outside a labelled
  development session.
- Managed setup and update can require a reviewed published engine and base
  using paired `--engine-release` and `--base-id` arguments. A changed channel
  selection refuses before creating project or store state rather than applying
  different software. These selections cannot use a locally resolved base.
  Explicit source catalogs and dependency locks are displayed with content
  digests and rechecked after confirmation and before activation. Changing an
  input during confirmation refuses before creating project or store state.
- Prepared official source catalogs preserve dependency-owned codecs without
  claiming them as pack-owned declarations. First-party pack node IDs execute
  from these archives without rewriting their source bytes.
- Core official pack manifests declare their existing third-party runtime
  requirements directly, including platform markers. Pack-to-pack dependencies
  and the host API contract remain separate from Python requirements.
- Pack manifests can declare an independent public pack version and a PEP 440
  engine compatibility range. Invalid or incomplete declarations are rejected
  without importing pack code. Every first-party node pack declares both,
  including the development pack.
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
- After a successful same-commit engine release, natural main publication selects
  every declared official pack and library version missing from the public index,
  even when a predecessor never published. Older or newer public versions do not
  mask a missing exact version. Unreadable or invalid index metadata refuses
  selection; interrupted inputs are reused only when they cover the missing set.
- Official publication requires dependency locks for every selected pack and
  committed supported base before preparing maps or submitting any archive.
  Wrong source/base/lock identities, incompatible Python and nonportable wheel
  origins refuse the entire local preparation without publication requests.
  Prepared maps do not imply scanner acceptance or live default availability.
- Default-set admission checks resolve registry archives against the engine
  under test and require every pack to pass the registry scanner, including
  the isolated executable probe. Published defaults use their verified dependency
  locks on the named managed base, without package-index dependency resolution.
  A missing lock, changed artifact or incompatible base refuses admission.
  Missing defaults, rejected packs and security review requirements fail the
  check without publishing any pack.
- Selected-archive behavioral tests use each first-party pack's independent
  distribution, including vision packs. They verify selected source provenance
  and refuse checkout fallbacks or replacement of engine-owned tests.
- Public archive admission verifies deterministic source ZIPs with bounded
  portable paths, validates manifest publication metadata, runs static doctor
  and a bounded isolated executable probe, and distinguishes security blocks
  from files requiring review. These checks need no private checkout, registry
  database or publication credentials. Missing or degraded probe isolation
  refuses rather than skipping executable checks.
- Probe dependency provisioning can use a complete operator-staged CPU wheel
  index for all requirements, including torch and torchvision, without a public
  package-index fallback. Missing or incompatible wheels refuse admission.
  The default CPU dependency source and all probe limits remain unchanged.
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
- Additive source preparation accepts exact absent first-party pack selections
  with `prepare-registry-sources --installed-lock PATH --retain-sources --add PACK@VERSION`.
  Existing source identities remain unchanged instead of being re-resolved.
  Repeated or installed pack names, noncanonical selections, incompatible engines,
  third-party registry publishers and corrupt wheels refuse before a candidate
  lock is written. Additions use the static index's compatible `wheels` entry and
  preserve its authenticated registry publisher, never a manifest trust grant.
  Preparation does not activate a generation or modify project data.
- `prepare-registry-locks --pack PACK` acquires only the named source's published
  dependency lock for the exact base; repeat the option to select multiple packs.
  Other source releases need not remain available in the registry. Duplicate or
  absent selections refuse before fetching; explicit non-registry selections
  cannot be silently omitted with `--registry-only`. Supervisor additive source
  preparation uses these selections, retaining existing source identities and
  leaving old dependency locks to inheritance from the source generation.
- `dinkster-env packs-plan PACK@VERSION --output PATH --json` retains an exact
  software review in a new host-owned disposable directory outside the
  project and shared store. It reports source/dependency locks, retained engine,
  base and placement, wheel downloads and isolation size without a generation
  journal or activation. Exact selections can add or replace published packs
  from any configured registry, not just first-party packs. Replacements use
  recorded registry provenance unless an explicit registry is selected; unrelated
  packs retain their source identities and dependency locks.
  Intact retained registry wheels can supply old archives
  without contacting their release endpoints. Previously installed wheels retain
  their original archive bytes and digest when another pack is added; corrupt
  retained wheels refuse before a candidate source lock is written. Missing
  retained registry archives are fetched and verified at their pinned identity,
  without re-resolving their versions.
- `packs-apply --input PATH --approval DIGEST --json` applies that review once
  within 900 seconds. Changed project settings, current software or reviewed
  input files refuse before construction. Successful application seals and
  activates an ordinary next generation and appends its software history; unselected
  source/dependency locks and placement are preserved. Workflows, models, outputs,
  user data and project settings remain unchanged. Hosts own review-directory
  cleanup and must authorize the caller before invoking these local commands.
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
- A managed engine or supervisor launcher that exits before readiness reports
  its exit code and up to the last 64 KiB of captured output from that launch.
  Older launches are excluded; truncation and unavailable output are explicit.
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
- Environment commands that ask for confirmation refuse non-terminal text-mode
  requests without `--yes`, naming `--yes` to apply and `--plan` to inspect.
  Terminal requests print the plan and ask `Apply this plan? [y/N]`; declining
  does not apply it. Explicit `--plan` and the Desktop client's `--json` preview
  remain read-only without prompting; `--json --yes` applies without prompting.
- Confirmed release acquisition retains channel and verified engine/base metadata.
  Setup and update can inspect those cached manifests when mirrors are unavailable;
  the plan labels use of the last acquired channel selection. Cached channels are
  specific to the configured mirrors, and cached manifests must still match their
  recorded identities. Planning does not create or repair cache files.
- Published runtime plans list exact engine/base payload paths, sizes and hashes,
  verified cache reuse and missing runtime bytes before confirmation. Pack payloads
  are separate from that runtime subtotal. Runtime acquisition also supports
  cache-only operation: missing or corrupt runtime inputs are reported together
  before cache mutation, without contacting mirrors or a package index.
- `dinkster setup --plan --variant VARIANT` inspects the selected release's
  declared default packs and exact-base registry dependency locks alongside its
  runtime inputs. It shows project/store paths, engine binding, exact source
  identities, lock hashes, wheel hashes/sizes and verified cache reuse without
  acquiring payloads or creating installation state. `--json` retains the source
  catalog and dependency-lock documents in its output. Pack bytes are labelled
  as an upper bound across all locked wheel alternatives, not a target-selected
  download promise. `--offline --plan` reads retained release and registry metadata
  only; it does not fetch or repair missing metadata or wheel payloads.
- `dinkster setup --root PATH --variant VARIANT` prints the published input plan
  and asks `Apply this plan? [y/N]`. Declining changes nothing; non-terminal use
  requires `--yes`. Confirmed setup rechecks source and dependency-lock identities,
  acquires their verified pack archives, and builds and activates a managed
  generation. `--yes` skips the prompt, not the plan or identity checks.
  Offline application is refused; `--offline --plan` remains read-only.
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
  their hash prefix. Local base, engine, pack and wheel downloads use readable
  names while retaining full content identity. Different contents with the same
  label stay separate, and relabeling identical bytes reuses the existing copy.
  Existing hash-only local installs remain usable without renaming their
  interpreters or environments. Contributor setup can
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
  Official packs built from the explicitly selected checkout may serve under
  reserved namespaces in that local generation. Other Git, local and pip
  sources do not gain reserved-namespace trust, even with publisher `dinkster`.
  Libraries built alongside those packs supply their locked dependencies
  without a registry download and must match the dependency lock's digest.
  Libraries are not installed or cataloged as node packs.
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
- `dinkster-env --root ROOT space --json` observes managed generation and shared
  store allocation without collecting or changing cleanup settings. Linux uses
  physical extents, counts hardlinks and overlapping clone extents once, and
  separates shared bytes from exclusive rows. It reports automatic cleanup
  policy, generation retention reasons, eligible backlog and additional explicit
  cleanup allocation. User libraries are excluded. Unsupported, changing,
  delayed or encoded allocation is unavailable, not replaced by logical file
  sizes. Shared clone references outside the scan make reclaimability unknown.
  Protected software history and settings are included separately from disposable
  generations and stores. Shared-store metadata includes registrations and cached
  channel, engine and base manifests, without treating them as reclaimable.
  Empty shared-store categories remain visible. Without
  `--json`, the command shows the same report with IEC units, exact byte counts,
  retention reasons and settings guidance.
  Windows uses native allocation and file identities, including resident data,
  and checks the volume's clone capability before counting shared extents.
  Unreadable allocation, reparse points and named streams make sizes unavailable;
  they are never silently omitted or replaced with logical sizes.
- `dinkster-env cleanup --plan` reports retained, candidate and deferred
  generation numbers with reasons without recovering or modifying the project.
  Status includes the same preview and visible pending quarantine warnings.
  Candidate status requires matching sealed software in published history and
  unique completed activation provenance. Missing, damaged, unpublished or
  ambiguous coverage defers the tree. Host and tools observations do not change
  software identity. Explicit legacy imports provide coverage only after their
  completed journal provenance and sealed software are revalidated. Preview
  never launches an interpreter, publishes or repairs history; candidate status
  is advisory rather than deletion authorization.
  Materialized pack payloads are disposable once their software identity is
  recorded; history does not promise to preserve local edits to runtime files.
  Managed interpreter acquisition is serialized against generation collection;
  pending launches and unproven child cleanup retain the generation.
  Stopped assignments carried across updates remain stopped and do not by
  themselves prevent collection after leaving the retained rollback set.
  `cleanup --automatic off --yes` persists an opt-out without collecting;
  `--automatic on` restores the default preference. Preview and status expose
  the same policy. `cleanup --keep-previous N --yes` persists how many previous
  successful generations to keep besides current, from 1 to 100 (default 1).
  Desktop offers the same count beside on/off. Selection follows successful
  activation ancestry, not directory numbering; imports and failed attempts do
  not consume retention slots. Older policy files default to one. Raising the
  count does not rebuild collected trees; lowering it takes effect at the next
  eligible collection. Policy commands do not collect synchronously.
  `cleanup --yes` retries generation-only collection for history-covered unused
  trees when the policy is enabled. Busy projects defer, overlapping retries
  coalesce, and status preserves the last attempt and retryable failures.
  Before selecting trees, retries publish completed activation receipts and
  capture eligible legacy software as marked imports under the project and
  runtime locks. Existing receipts are never replaced. Live, uncertain or
  invalid legacy trees retain their software and expose capture warnings;
  stopped trees can be captured on a later retry. Imports do not change the
  activation chronology, and capture never runs a retained interpreter.
  Interrupted quarantine recovers and committed deletion retries without
  collecting shared caches, rewriting history or stopping processes.
  Successful CLI activation, repair, restore, start and stop attempt collection
  after releasing transaction locks. Managed supervisors attempt startup,
  hourly collection, retrying busy locks sooner. Acknowledged stops attempt
  collection after durably recording the stopped engine assignment.
  Cleanup failures remain warnings rather than invalidating a successful
  update. Legacy trees without provable coverage still defer; Desktop controls
  are separate.
- `dinkster-env doctor --repair` plans an offline rebuild from the active release's
  local wheels and retained base. Confirmation creates and health-checks a new
  generation without editing the drifted tree. Missing or corrupt local inputs
  refuse repair; failures keep the prior generation selected and serving.
- Connected event clients do not need to close their browser before an installed
  generation can stop or be replaced. Authenticated supervisor shutdown closes
  their event connections so clients can reconnect to the active generation.
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
- Source run scripts explicitly select watched checkout development without
  invoking published setup or loading a standalone installed-pack root. The
  terminal and node catalog label the source session as unreviewed and not
  reproducible from dependency ranges; the browser opens after health succeeds.
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
  Release metadata also records that engine's default pack IDs and minimum
  versions, independently of the command interpreter's installed defaults.
- Native tools builds retry transient HTTPS input failures up to three times,
  preserving exact size and SHA256 admission. Certificate errors, HTTP 4xx and
  corrupt payloads fail without retry; exhausted failures retain attempt details.
- The environment client acquires its native tools ZIP from the channel's
  independently versioned tools publication without engine or base payloads.
  Planning creates no cache; confirmed acquisition checks manifest and archive
  integrity and refuses corrupt tools rather than falling back to engine assets.
  Local ZIP folders use dated supervisor versions without hashes; different
  content with the same label receives a numeric suffix. Existing cached tools
  remain reusable, and planning leaves collision allocations unchanged.
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
  registry lock. Cached source archives, in readable or legacy locations, are hash-verified
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
  Linux and Windows managed shutdown drains worker descendants before proving
  exit, including isolated and grouped workers blocked during pack import.
  Collection retains legacy or unsupported-platform stopped generations when
  descendant exit is unproven; it never stops processes to make them collectible.
- `dinkster-env history` lists successful software activations in order, and
  `history show ID` displays the recorded software and changes. History survives
  generation-directory collection. Snapshots exclude workflows, models, outputs,
  libraries and mounts. Interrupted activation does not become a successful entry;
  completed activation history publication is recoverable without duplication.
- Legacy software can be preserved as explicitly marked imports in the same
  history without changing current software, rollback selection or activation
  diffs. `dinkster-env import GENERATION --plan` previews capture; `--yes` confirms
  capture after completed history recovery under the project writer lock.
  Capture never launches the retained interpreter. Imports require intact sealed
  software and unambiguous completed activation ancestry; conflicting captures
  refuse. Their timestamps describe
  observation, and unknown historical tools, host facts and wheel compatibility
  tags stay unknown. `restore ID` accepts an imported entry after its generation
  tree is collected, revalidating its completed activation provenance. Missing
  historical compatibility facts can prevent reconstruction.
- Software snapshots describe canonical reconstruction inputs, not the integrity
  of every materialized pack runtime file. A successful import or exact restore
  does not establish that an existing same-lock runtime has no local edits.
  Recorded software identity in history permits collection of unused generations,
  including their materialized pack runtimes, without a per-file runtime inventory.
  Current and previous generations, live processes and intact sealed metadata
  remain protected. Links inside disposable runtime trees are removed without
  deleting their targets; linked sealed metadata still refuses collection.
- `dinkster-env restore ID` rebuilds a published software snapshot in a new
  generation after confirmation. Recorded identities are preserved; unavailable
  inputs refuse activation. Tools versions are displayed but tools are never
  replaced. Failed attempts remain separate from successful history. Restore
  preserves existing history, project settings, workflows, models and outputs.
- `dinkster-env rollback` rebuilds the preceding successful history entry, even
  after its generation directory was pruned. It uses restore's missing-input and
  proposal-approval contract, appends a source-linked entry with reason `rollback`,
  and leaves project settings and user data unchanged. Planning recovers an
  interrupted publication of a completed activation before selecting that entry.

### Development generations from local folders

Standalone local-folder and `git+URL[@REF]` pack installs remain available as
development inputs without a published dependency lock. Their plans report
range resolution and lack of registry review. Serving an installed local/git
selection, or an explicit local pack, labels both command output and the node
catalog as development and not reproducible from ranges. A registry-only
selection does not gain that label, and standalone install refuses a registry
version rather than resolving it as development.

Select local software explicitly; nothing needs publishing first:

```sh
dinkster develop --root /absolute/project --source /absolute/engine --pack-source /absolute/pack --library-source /absolute/helper --variant linux-cpu --data /absolute/store --plan
dinkster develop --root /absolute/project --source /absolute/engine --pack-source /absolute/pack --library-source /absolute/helper --variant linux-cpu --data /absolute/store --approve sha256:DIGEST_FROM_PLAN --yes
dinkster-env --root /absolute/project status
dinkster-env --root /absolute/project doctor
dinkster-env --root /absolute/project start --yes
```

Repeat `--pack-source` for more packs and `--library-source` for more helper
folders or wheels. Omit library inputs when none are needed. Choose a base variant
matching the host; `--python /absolute/python` explicitly selects an existing
interpreter instead of the configured standalone interpreter. Use `--port PORT`
when the default listener is occupied. Project and store directories must be
outside selected source folders. Building uses verified copies, never editable
installs into author folders. Selected helper wheels participate in engine
resolution before packs are built, including replacements for engine dependencies
whose real versions still satisfy the declared constraints. Explicitly selected
first-party packs may use their reserved namespaces only in this development
capture; this does not grant registry admission. Missing sources, changed plan inputs, incompatible
API ranges and dependency conflicts refuse rather than inventing versions or
dropping constraints.

Development generations are built from local folders, not reviewed by the
registry; dependency ranges resolve at build time. After editing a pack, stop the
engine with `dinkster-env --root /absolute/project stop --yes`, then repeat the
plan and approved `develop` commands to build a new generation. An ordinary
channel update refuses a local generation instead of changing its source.

```sh
dinkster-env --root /absolute/project history
dinkster-env --root /absolute/project history show 1
dinkster-env --root /absolute/project restore 1 --plan
dinkster-env --root /absolute/project restore 1 --yes
```

Restore rebuilds the captured runtime from retained interpreter files, built
wheels, exact dependency artifacts and pack archives. It needs no source checkout
or network when those artifacts remain intact. Doctor reports separately whether
all exact-recovery inputs are retained; a clean running generation can still have
missing recovery artifacts. Current and previous generations use the existing
retention protection. Older history alone is not an artifact backup: collection
of its last inputs can make exact recovery unavailable.

Source paths, package roots, content digests and available Git commit/dirty state
describe the captured working bytes, including selected untracked edits. Recovery
does not reset, clean, overwrite or back up author folders. Workflows, models,
outputs and other project data are not recovered by software restore. No bundled
Git, Git-URL fetching or contributor-environment synchronization is provided by
this operation. When invoking tools through `uv run`, `--no-sync` prevents uv from
silently changing that tool environment to match a checkout lock; it does not
capture an environment or make arbitrary unrecorded edits recoverable.
