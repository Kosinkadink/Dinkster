## Server API and execution

- Supervisor, station and fleet ingress listeners reject unlisted, malformed,
  missing or duplicate HTTP Host headers with `421 host-not-allowed`, including
  WebSocket upgrades. Loopback IP literals and the configured listen hostname
  are accepted; additional names require repeatable `--allow-host HOST`.
- Managed environments persist explicit IPv4/IPv6 public bind addresses, allowed
  Hosts and browser origins, and optional HTTPS certificate/key paths through
  the environment listener command. Defaults remain IPv4 loopback HTTP.
  Multiple specific interfaces share one port; unavailable bindings fail as
  a unit and HTTPS never opens a cleartext fallback listener.
- Managed environments can persist offline runtime mode independently of cached-input
  installation mode. Browser requests stay on the served product origin,
  including API and WebSocket traffic; external resource/connection requests are
  blocked. Partner nodes retain admitted declarations with an actionable refusal
  instead of executing. Online mode restores normal behavior. This is not a host
  or unrelated pack networking firewall; partner-only pack selection remains
  independent. Responses and managed engine status indicate the effective policy.
- Managed environments can opt into persisted HTTP gzip compression. Eligible
  textual/JSON API responses and frontend files negotiate Accept-Encoding;
  disabled or unsupported requests preserve identity bytes. Streams, WebSockets,
  byte ranges and compressed/binary media keep their existing protocols.
- Engines that need a newer library API or an additional library capability
  acquire compatible signed native tools before installation or activation stops
  the current engine. Restoring an older engine retains the selected helper;
  compatible starts need no network access.
- Tools can list local snapshots and projects affected by a proposed helper
  API removal, using the library's recorded engine attachments. Compatible
  machines show no notice. API retirement requires the user's ruling and
  disclosure before tools selection; this release removes no API.
- Tools can browse a page of local projects, saved workflows or recorded files
  without starting an engine. Each request exits with its helper; a missing
  library is not created, and an unreadable library is reported rather than
  presented as empty. Browsing does not migrate the software environment.
- Environment catalog/status/doctor queries share a JSON command interface.
  Catalog preserves the selected channel's variant manifests; status reports
  configured channel/variant and observed engines without starting them.
  Engine rows include the ordered GPU assignment, including an empty list for
  CPU-only launches, so recovery can retain it instead of saved client settings.
  Stopped bindings retain that assignment. Runtime credentials remain private.
  Doctor reports content and pack drift without repairing it.
- Confirmed environment start/stop commands manage named engine bindings with
  explicit device assignments. Repeating an identical ready assignment does
  not restart it; changing an assignment requires stop first. Plan-only calls
  do not launch or stop processes, and empty device lists hide GPUs.
  Shutdown closes live event sockets and interrupts unfinished HTTP proxy
  transfers without waiting for clients that stopped reading.
- Managed local environment launches allow operators to edit project folders.
  Grants persist across controlled restarts. New grants default to read-only;
  writable access remains explicit. Remote connections and separately configured
  server launches do not gain mount-edit authority.
- Schema wire version 2, which carries each node's body tree, with described
  rows, compact-storage and stream input declarations, alpha and mask
  policies, range-aware inputs, and chunk-safe declarations scoped to
  supported choice options. Stream
  execution is not exposed until a node pack supplies matching value codecs
- Resident-resource consumers follow the validated producer execution arm
  without a node-name or native-arm-name allowlist; malformed, unknown, and
  conflicting producer stamps are refused
- Persistent engine startup preserves legacy records in a backed-up,
  restartable migration and serves shared records through the selected tools'
  library helper. Finished history, sources, permissions, folder preferences and
  environment/pack settings survive engine replacement. Accepted work, journals,
  sessions, worker snapshots and values use private engine state outside the library.
- Public-acquisition receipts retain unknown fields and verified-fetch checks.
  Raw receipts are immutable and query-bound pages observe peer writes.
- Runtime and pack preferences use revision-checked helper records. Unknown
  fields survive edits, stale saves refuse, and helper failure does not fall
  back to file writes. Incompatible snapshots are disclosed and refused before
  an engine starts, based on library helper capability rather than version order.
- Routes: health/auth, node catalog, extension/composition diagnostics,
  choices, pack assets, jobs (status, cancellation, event
  replay), values, queue control, live WebSocket events, settings, memory
  governance, cache trim/export; asset library and history routes when a
  library root is configured, including classified bounded latent upload
- The node catalog and native job submission use `/api/nodes` and `/api/jobs`.
  There is no separate ComfyUI schema catalog or prompt submission endpoint.
- A development snapshot's node catalog carries `dinkster.development` with
  `mode: development`, `reproducible: false` and an explanation. Captured runtime
  artifacts remain restorable; resolving source ranges again is not reproducible.
- Graph admission rejects primitive literals that do not match their declared
  input types, including boolean values on INT/FLOAT inputs and strings on
  FLOAT inputs. Integer values remain valid on FLOAT inputs. The same checks
  apply inside lists and to explicitly stamped literals, before nodes execute;
  no node-level coercion is used to repair invalid literals.
- String outputs have a lossless UTF-8 `text` rendition through `/api/values`.
  Long paths and documents can be read in full without the status descriptor's
  256-character inline preview limit. Retrieval retains job/scope authority.
- Library-backed completed-value renditions retain vault read ownership until
  rendering settles, including after request cancellation and before a source
  descriptor opens. Independent readers keep their own lifetime. This does not
  certify legacy migration references or enable cleanup.
- Completed image-document responses retain their declared external resources while
  rendering settles. Durable document copies require their own resource declarations.
- Helper clients can request a file's content digest and reuse its verified
  account-local identity across libraries without reading unchanged bytes.
  Concurrent requests for one folder share the completed result. Helpers
  also support folder scans with progress while project requests remain
  available. Scans reuse verified legacy indexes and worker-written files,
  resume committed work after helper exit, and publish indexes readable by
  existing workers. Engine folder scans use the same account cache across
  libraries and mount names, without rereading unchanged model bytes.
- Persistent projects have library-local browser IDs and independent global
  identities. Registration is idempotent; `default` always exists and cannot
  be deleted. Revision-checked field edits retain unrecognized fields, and
  deletions remain remembered. Project lists support cursor paging and
  filtering by environment. `/api/projects` uses asset read/write permissions.
  `/api/environment` uses asset read permission and reports the managed
  environment's stable ID and running snapshot, or null when that identity is
  unavailable. Reading it does not assign projects. Managed identities survive
  engine restarts, software changes and environment moves; separate environments
  do not share an identity merely because their engines are both named `main`.
  A job that enters execution records its project's managed environment,
  running snapshot and execution-start time when it settles. Unbound projects
  share the running environment by default;
  a project assigned elsewhere refuses execution rather than being reassigned.
  A project may select an environment before any run: both last snapshot and
  last-run time are null until an actual attempt settles. Selecting another
  environment clears the previous environment's run metadata; selecting the same
  one retains it. Revision conflicts and half-filled run metadata are refused.
  Engine-free project queries return the authoritative library record and the
  candidate environment's existing identity and active snapshot. Compatibility
  checks refuse a project assigned elsewhere, unreadable candidates and
  interrupted environment operations without starting or stopping an engine.
  Explicit revision-reviewed selection assigns only that project; a plan leaves
  its association unchanged. Neither queries nor selection create a managed
  identity, download tools or record a run.
  Queued work cancelled before starting does not change that record. Engines
  without a managed identity leave unbound projects unchanged and refuse to run
  projects assigned to a managed environment.
  Helper-backed engines reject unknown or deleted projects in job, upload,
  catalog, selection, library-write, mount, model-setting, asset-guess and
  workflow-planning requests. Workflow creates, reads and
  updates return their owner; project-filtered library browsing selects only
  that owner's workflows, and its cursor binds the project and media type.
- Saved workflows can belong to a project; putting one in another project
  requires a copy. Project pack needs combine packs actually used by saved
  workflows with explicitly added packs. Engines recompute needs from verified
  workflow bytes, using saved node attribution or installed host attribution for
  unstamped nodes. Unrecognized workflows remain unknown rather than guessed.
  Project needs also report the running environment's installed pack IDs and
  versions, including disabled packs and versionless local packs. These facts
  do not enable packs or approve software changes.
  `/api/nodes?project=ID` publishes only that project's enabled packs and core
  nodes; without a project it retains the installed catalog. Removing an extra
  pack does not hide nodes its workflows still use. Model references retain their
  workflow locations and known digests without forcing a scan. Tools can
  browse scoped records without starting an engine, using project, media
  type, name and label filters with query-bound pages. A project filter
  selects only that project's workflows; other record types have no project
  ownership, and legacy workflows belong to `default`. Engine-free asset
  browsing groups recorded aliases and observations by digest within a live
  project and scope, with name, kind, origin and inclusive date filters.
  Pages retain all aliases for a matching digest and do not read asset bytes
  or inventory unrecorded files.
- Generated preview and produced-source bytes use the engine's temporary
  storage separately from uploaded and saved content. Saving a layered image
  document preserves its non-inline resources in durable storage, so deleting
  the temporary source does not remove the saved document's pixels. Each engine
  collects only its own temporary directory. Temporary previews are cleared on
  startup and clean stop unless `--keep-temporary-files` is set. Reopening
  Desktop therefore leaves generated previews empty until the workflow runs
  again; uploaded and saved files remain available. The default running limit
  is 10 GB. Size cleanup removes the oldest eligible files first, and Clear
  removes eligible files immediately. Queued inputs, active job data and open
  readers are protected without browser file lists. The independent disk result
  cache is not emptied: entries naming deleted files miss and recompute, while
  self-contained cached values stay reusable.
- Asset record catalogs upgrade automatically with versioned migrations.
  Matching unversioned catalogs retain scoped records, verified media grants,
  dependency manifests, and derivations; incompatible schemas are refused.
- Result offers are immutable server-created publications for runtime outputs.
  Events, status, and completed-value discovery expose the matching publication;
  nested-value selection keeps the parent output binding. Unknown codecs remain
  incomplete.
- Scoped terminal publication discovery retains original output publications,
  execution identity, and provenance independently of process restart,
  queue/cache retirement, and ordinary history deletion. It never substitutes a
  newer publication or reconstructs legacy output bindings. Failed, cancelled,
  and unknown-codec sets remain incomplete.
- Durable terminal history retains recorded external asset identities from target
  values, saved artifacts, and their optional published previews without storing
  output payloads or changing history responses.
- Host-validated saved-file artifacts are cataloged in the submitted project
  with job and node provenance before the job reports completion. Unavailable
  or revoked files and catalog failures fail the job rather than report a
  successful save; recoverable catalog writes are retried on restart without
  copying files into the vault.
  Native writer saves include exact saved-file descriptors in both in-process
  and isolated execution, including custom mounted destinations and empty
  text files. The host verifies the current writable mount, safe file location,
  content digest and byte size before exposing the descriptor. Returning an
  existing asset without writing it does not report a new saved file.
  Saved images, audio and video are classified from verified bytes for scoped
  media authority and kind-filtered browsing. Text and unrecognized files stay
  generic; a worker-provided media type alone never grants media authority.
  Required encoded result attachments published into the vault are retained
  as collision-safe named files in the project's selected output folder before
  completion. Their result descriptors name the visible files; the catalog
  records both mounted and vault locations with job/node provenance. Transient
  preview events are not retained. A missing writable destination fails the job
  instead of silently redirecting the output.
- Filesystem mount scans report file, byte, and elapsed progress; publish
  indexed assets before the scan finishes; reuse unchanged-file indexes; and
  keep per-mount indexes under the configured library root.
  Prepared model folders can supply a digest list with matching size and
  modified time, so their catalog and digest references are usable without
  reading every model first. Copied file identities are not treated as content
  proof: loading verifies unproven bytes and refuses a mismatching digest,
  including same-size replacements with restored modified time. Changed
  metadata or invalid digests require hashing. Filename-only bindings retain
  their file-revision requirements.
- Windows filename-only model selections require tracked file revisions for
  immediate unverified reads. Mounts without revision tracking wait for the
  background index to verify the file; the same name then resolves as a
  verified asset. Mount settings expose this capability, and refusals name
  the filesystem and current scan progress. No model bytes are hashed during
  filename selection. Unverified selections refuse files with active writers;
  an open unverified read excludes writes, renames and deletion until the
  reader closes. Idle bindings do not block writers.
- Operators can resolve an absolute local file path to a mounted asset in the
  selected project's live folders without copying bytes. Changed or unindexed
  files are verified and cataloged on demand, including read-only inputs.
  Exact-file resolution shares the account digest cache with folder scans;
  selecting one file does not scan or retire its siblings.
  Outside, revoked, excluded, missing and unsafe paths return an explicit
  refusal without revealing host paths. Authentication, malformed requests and
  server failures remain distinct errors, not upload-fallback signals.
- Setup, environment-managed snapshot serving, and serving with mount changes
  enabled initialize a writable
  `<library-root>/imports` folder and a selected writable
  `<library-root>/output` folder when those defaults are not configured.
  Fresh snapshots can save output without enabling runtime mount editing;
  the existing mounts API reports the selected writable destination.
  Existing output selections, folder paths and read-only grants are preserved; unavailable
  configured directories are not recreated or replaced. Removing the import
  or output grant remains effective across restarts and repeated setup.
  New default folders become available to existing projects without resetting
  folder order, exclusions, project-only grants or output selections.
- Managed environments can persist separate generic and classified image, audio,
  video and model3d upload byte caps through `dinkster-env uploads`. Defaults
  remain unchanged; overrides may lower but never bypass transport/media
  ceilings. Declared and chunked lengths enforce the effective caps, oversized
  requests return 413 without publishing partial assets, and restart retains
  the policy. `GET /api/assets/upload-limits` reports the same effective caps
  for clients. Codec, digest and decoded-media validation remain mandatory;
  structured latent/image-document safety budgets are independent.
- Named file and classified media uploads in the managed server use original
  filenames in the configured imports folder, with collision-safe numbering and no durable
  vault duplicate. Requests can select another writable project folder and
  relative subfolder; the returned asset records the actual virtual path.
  Operator location details identify the exact default import mount independently
  of output selection, including custom mount IDs and unavailable defaults.
  Read-only, unavailable, revoked, escaping, or other-project destinations
  fail without silently redirecting the upload.
  Named model/file uploads stream under the configured upload size limit
  without buffering the whole file. Anonymous workflow and digest transfers
  retain deduplicated vault storage; named generic imports do not grant
  classified media authority.
- Explicit scoped digest transfers retain their name, receive time and
  caller-reported source-machine label across restarts, with recoverable
  catalog publication. The label is advisory, not verified machine identity.
  Transfers verify the declared digest, use the existing upload limit and
  streaming vault path, and do not create classified media authority.
- Named and classified-media uploads retain their original names, observed
  virtual locations and upload history across server restarts. The scoped
  asset catalog groups aliases by digest, supports name/kind/origin/date
  filters and paged queries, and keeps workflow records separate. Catalog
  locations do not grant filesystem access or expose physical paths.
  Catalog queries are read-only. Interrupted publication recovers completed
  files on restart or during background maintenance, after checking their
  bytes and current folder authority.
  Unpublished, changed, unavailable or revoked files are not restored or granted.
  Existing scoped media grants and workflow records become discoverable after
  verified vault backfill, with Unknown origin where history was not recorded.
  Missing historical media names display the digest, not an invented filename.
  Workflow records remain document-kind entries; existing project observations
  do not acquire default-project membership, and anonymous bytes do not acquire
  membership in any user's scope through this backfill.
  Human operators can separately inventory verified, unrecorded vault blobs
  by digest. This view excludes bytes recorded in any scope, reports unknown
  names and history without guessing them, and creates no records or grants.
  Agents, delegated users and ordinary asset readers cannot access it.
- Read-only asset availability checks distinguish held vault/granted-folder
  identities from missing ones without downloading, saving, granting or
  cataloging bytes. Replaced, missing and revoked files are unavailable;
  the same digest at a different granted location remains usable. Results
  are scope-authorized, uncached and do not expose physical paths.
  Job preflight uses the same verified local presence rule, never a fetching
  fallback. Corrupted vault content can be replaced only through consented
  acquisition; a digest-shaped filename is not proof that the bytes are held.
- Human operators can inspect the vault directory and verified physical
  locations for a scoped project asset, including identical bytes under
  different observed filenames. Changed, missing, excluded and revoked
  copies and vault symlinks escaping the vault root are not offered as reveal
  targets. Ordinary catalogs remain path-free;
  location reporting does not download assets or open the desktop shell.
- Operators can persist project folder membership, independent display order,
  shared folders, and project output selections through the mount API.
  Removing a project association does not delete files or revoke the grant.
  A selected project output requires a writable replacement before removal;
  projects without an output can remain without one.
  Failed mounts can be explicitly rescanned after reconnecting a drive.
  Saving to a missing output root fails without recreating that root or
  redirecting the output to another folder.
  With fixed pack sandboxes, folder grant edits persist as restart-required
  changes without altering live worker authority or restarting the server.
- Shared model locations accept `always-on`, `default-on` and `opt-in`
  visibility in `model-settings.json`. Omitted visibility is `default-on`.
  Project folder lists and filename selection honor those defaults and saved
  project choices. An always-on model location cannot be disabled through
  project folder updates. Changing the shared mode updates project selection
  without deleting model files or changing read-only filesystem grants.
  Operators can read and replace shared locations through `/api/models/settings`
  using its revision to reject stale edits. A location can name a complete
  ComfyUI-layout root or one existing category directory. Read-only-only
  configurations have no download default. Per-project location switches
  include all category mounts together. Removing a location leaves its files
  on disk. With fixed sandboxes, saved grant and visibility edits remain
  restart-required; live project selection and disabled-folder preflight keep
  the running grants' settings until restart. Project folder updates return
  a restart-required conflict while shared model settings are pending, so an
  unrelated project switch cannot partially change a running location. Mount Browse,
  folder listing and search require the folder to be enabled for the requested
  project; disabled folders return `folder-not-enabled` without file entries.
  Explicit digest references found only in disabled shared model folders return
  a preflight refusal with `disabledLocations`, not acquisition consent. Enabling
  a returned location permits retry without downloading another copy. An enabled
  identical copy or verified internal asset remains usable.
- Jobs retain their project's folder preferences in both in-process and
  isolated execution. Name and directory selections follow that order;
  saved digest references retain their identity independently of folder order.
  Filesystem revocation still takes effect. Host locations are operator-only.
- Scans exclude partial downloads and files changing between observations.
  Missing and replaced asset identities survive rescans and restart without
  being offered as live content; restored content recovers its digest.
- Native job submissions accept bare filenames in model asset inputs before
  mount hashing. Bindings remain explicitly unverified and omit content
  hashes until a scan verifies the same file revision. Missing names return
  `409 assets-missing`; submission does not hash or download model bytes.
  Missing bare model names include category-matched source leads declared in
  official workflow templates, including nested graphs. Equivalent download
  URLs are deduplicated and conflicting sources remain ambiguous. These leads
  are advisory and non-fetchable, without a fabricated content digest; local
  selections and explicit hashes take precedence. Unknown names and unsupported
  categories have no inferred source. Filename-only references to disabled shared
  model locations report `Folder not enabled for this project` without remote
  source suggestions, even before hashing.
  Operators can PUT `{asset, kind}` to `/api/models/hidden/{project}` to hide
  an admitted individual model without deleting its bytes or changing folder grants.
  Model settings return the hidden names and opaque identities; DELETE `{identity}`
  on the same route unhides an existing preference, including an unavailable file.
  Verified content identities hide matching copies; unhashed files use their current
  file revision, surviving restart and hash enrichment but not replacement bytes.
  Hiding applies only to the selected project, suppresses browse/picker entries and
  remote suggestions, and gives explicit references an actionable hidden-model notice.
  Folder settings remain readable by v0.0.14 when no models are hidden, including
  after unhiding the last model or reopening settings saved with empty hiding
  preferences. Nonempty hiding preferences require a newer engine.
  Asset guesses accept an optional `project` (default `default`) and filter local,
  declared and resolver suggestions for that project's folder and hiding preferences.
  Operators with asset-read authority can POST `name`, `kind`, `project` and
  `scope` to `/api/models/downloads/plan`, optionally with a `source` HTTPS
  Hugging Face resolve URL or Civitai model/download URL without credentials.
  An optional `asset` is an exact asset reference whose name must match `name`.
  Its BLAKE3 digest is authoritative: local readiness requires that identity in
  the project's enabled model kind, and the consented transfer verifies it
  before publishing bytes or provenance. A same-named file with another hash
  is not ready. Plans and transfers never submit a workflow run.
  The selected filename must match the needed model. Explicit sources can
  resolve files absent from the template catalog or select among ambiguous
  declarations. Civitai model links can specify `modelVersionId`; matching
  files across versions require a further source choice. Review shows the
  fixed revision/version, provider SHA-256, exact size, reported license and
  physical destination. Downloadable plans include `destinationSpace`: either
  `status: available`, `freeBytes` and an opaque `filesystem` identity, or
  `status: unavailable` without a byte count. Free bytes are an observation,
  not a reservation, and zero means measured zero. Missing destination folders
  use their nearest existing ancestor's filesystem. Selected sizes sharing a
  filesystem are summed before comparison; aliases do not create extra capacity.
  Civitai size comes from transfer headers, not rounded
  `sizeKB`; review opens the response without consuming its body. No model
  file is written or BLAKE3 identity manufactured before explicit consent.
  Civitai access refusals are distinct from Hugging Face sign-in or grants;
  Hugging Face credentials are never sent to Civitai. Local files take
  precedence; local names ambiguous under the same binding rules as execution
  return `duplicate` without a source lookup or download plan. A `model-hidden`
  answer includes `hiddenIdentities`, the matching opaque project preferences
  accepted by the existing hidden-model DELETE operation, including revision-only
  preferences for an exact model stored under another filename. Disabled folders,
  missing writable destinations and ambiguous or absent declarations return
  distinct states. Plans expire after 15 minutes.
  Operators with asset-read authority can POST `project`, `scope` and up to 256
  `models` (`name`, `kind`) to `/api/models/presence` to learn, in order, whether
  each name binds for that project: `local`, `model-hidden`, `folder-disabled`
  (with `disabledLocations`), `duplicate` when local name binding is ambiguous,
  or `missing`, which
  says whether the download plan accepts the name (`plannable`). It reads only local folders, as plans do before
  any source lookup, and holds no plan.
  Operators can GET `/api/models/providers/huggingface?scope=local` with
  asset-read authority to inspect only whether a credential is configured.
  PUT with `{"token":"..."}` additionally requires settings-write authority
  and validates the token against the provider before saving it. DELETE
  forgets the credential. Neither route returns the token or provider response
  body, and all credential responses are non-cacheable.
  POST `/api/models/providers/huggingface/sign-in?scope=local` accepts
  `{"method":"pkce"}` from a local client or `{"method":"device"}` for remote
  sign-in. Both require operator settings-write authority. PKCE returns an
  authorization URL using the registered public client and a temporary loopback
  callback; device sign-in returns the user code, HF verification URI and polling
  interval, never the device secret. Device verification links accept only HTTPS
  on `huggingface.co` or the provider's `hf.co` short domain. PKCE authorization
  stays on `huggingface.co`. Both request only `openid profile gated-repos`.
  POST to `/api/models/providers/huggingface/sign-in/{flowId}?scope=local`
  finishes or polls the same owner's flow; DELETE cancels it. State is one-use,
  flows expire, and replacing/revoking credentials invalidates pending sign-in.
  Completion saves only provider credentials, not download consent. A fresh
  model plan and explicit Download are still required. OAuth credentials honor
  the returned lifetime, refresh once when possible, and otherwise report expired
  sign-in. Lost/refused refresh attempts require new sign-in rather than replay.
  Provider exchanges never follow redirects. Credentials stay in
  engine-local `credentials.json`, separately from model settings, projects
  and workflows, with mode 0600 on POSIX and a current-user-only protected
  Windows ACL without elevation. Forget removes the credential file and retains
  only a nonsecret revocation record. If deletion fails, Forget reports the
  failure and the surviving bearer cannot be reused, including after restart.
  There is no token in model metadata or provenance. Changing
  credentials invalidates outstanding consent plans and cancels transfers
  that have not started publication; a new plan and explicit consent are needed.
  A refused provider returns `409 credential-required` for missing or rejected
  authentication, or `409 provider-grant-required` when an authenticated
  request is forbidden, with a `requirementId`. Provider approval must be
  obtained on Hugging Face; saving a token does not grant repository access.
  Tokens are sent only to `https://huggingface.co` through the pinned HTTPS
  transport, never to a CDN or after a redirect leaves that origin. Public
  credential-free acquisition receipts retain their separate, stricter rules.
  POST `/api/models/downloads/start` takes only `planId` and `consent: true`,
  with optional `scope` query. It requires operator, asset-write and job-submit
  authority. It uses the server's immutable selection, never client-supplied
  provider metadata, and repeated starts of one unexpired plan return one job.
  GET `/api/models/downloads` and `/api/models/downloads/{jobId}` require the
  owning operator's scoped asset-read and job-read authority. They expose byte
  counts, transfer/verification/publication/indexing state, verified result and
  actual path, `reused` and a rename notice. Before transferring, jobs check
  same-size files under the selected physical model location, including unindexed
  files with different names outside conventional category directories. Reuse
  requires provider SHA-256 and computed BLAKE3 verification, plus any workflow
  digest, and retains the existing path without creating another model file.
  Exact-file selections persist across restart for the requested kind while
  neighboring files remain outside that kind's grants. The requested filename
  remains bound to the verified file, including when its existing filename is
  different. A stale binding refuses rather than falling back to different bytes
  at the requested name. Changed files or revoked location/project grants refuse
  reuse. A reused file is not reported as renamed.
  If index refresh fails after verified
  publication, the completed result retains the file and reports `indexingPending`.
  DELETE on a job additionally requires asset-write and job-cancel
  authority; cancellation before publication removes temporary bytes without
  altering existing models. Publication already started cannot be cancelled.
  Transfers verify provider size/SHA-256 and compute BLAKE3 in the same stream.
  Destination grants and project membership are rechecked before publication.
  Four transfers may run concurrently, at most one per physical category root.
  Plans and job records are bounded to 64 each; completed records expire after
  one hour. Jobs are in-memory, not a durable queue or resumable download.
  The list response includes an engine-instance id; after restart, old jobs
  are unavailable and clients must request a new plan and explicit consent.
  Digest-backed public model download plans include `destination` when the host
  has a writable model folder for the project. Without one, the plan is not
  fetchable. Downloading still requires consent to each exact digest.
  Native and translated model loaders accept these local bindings; translated
  loaders retain category-root confinement and reject shadowed file names.
- Inbound authentication with operator-managed static Bearer credentials,
  identity-service Ed25519 JWTs verified against cached JWKS, or both with
  static credentials tried first
- User-delegated agents retain credentials across restarts with optional
  expiry and explicit revocation. Authenticated servers require a recently
  verified user JWT; signing in resumes the same delegation under the user's
  current grants and permission toggles. Auth-off agents do not require JWT
  freshness. Explicitly helper-backed hosts share permission overrides and
  delegation revocations across engines and restarts. Stale permission edits
  return 409; unavailable authorization storage returns 503 without using cached
  grants. Human-kind credentials, whether JWT or static, remain usable during an
  authorization storage outage. Federated asset routes report unavailable
  authorization with their versioned `service-unavailable` error. Open event
  streams close rather than continuing with stale authority.
  Ordinary startup retains legacy authorization storage.
- Declarative resolver-index subscriptions from local JSON files or HTTPS
  URLs, with ETag revalidation, regional mirror priority, digest-verified
  acquisition leads, isolated source removal, and exact-basename suggestions
- Official provider subscription bootstrap from explicit URL and stable-ID
  configuration, with identity verification and preserved unsubscribe
  choices; no built-in provider endpoint or identity
- Workflow templates are disabled by default: `features.templates.enabled`
  defaults to false, and the catalog, template body and thumbnail routes
  return 404. Guide Markdown and node documentation remain available through
  their separate documentation routes; template-opening actions are hidden.
- Asset acquisition checks the local vault and mounted files before packaged
  assets and HTTP sources. Remote bytes verify against the requested digest
  before publication, and job acquisition requires explicit consent.
  Newly acquired assets retain a discoverable name and sanitized winning
  download URL in the submitting scope and project. Interrupted catalog
  publication recovers verified vault bytes without downloading again.
- Helper-capable engines with separate engine-state directories can serve one
  library concurrently, see each other's saved records, and reject stale edits.
  Stopping or killing one leaves the other able to read and save. Direct engines
  still require exclusive library ownership, and migration and format upgrades
  still require exclusive admission. Each engine relays committed library changes
  to connected WebSocket clients with a `library_changed` invalidation ping.
  The ping contains no record IDs, fields or scope details; clients refetch
  through authorized HTTP routes. Working-reference updates alone do not send it.
- Explicit server launches can keep accepted jobs, run journals, collaboration
  sessions, worker snapshots and remote values in an exclusively owned
  engine-state directory outside the library. Unfinished uploads stay in a
  separate engine directory on the vault disk, even when engine state is on
  another disk; linked upload-staging directories refuse. The state directory is
  bound to its library even after shutdown; another library and unbound populated
  state refuse without opening stores. Interrupted first-use binding retries
  automatically without adopting arbitrary existing state or following linked
  staging files. Restarts recover only that engine's accepted work and remove
  only its unfinished latent uploads; finished history remains
  shared. Managed launches declaring `library-helper/1` select engine state
  before engine imports: equal or newer snapshots carry working state forward,
  while an older snapshot starts in a fresh directory and preserves the newer
  state and its uploads. Rebuilt older engine releases start clean even under a
  higher snapshot number; restoring an earlier same-version release is also
  isolated. Selection survives interrupted publication and refuses
  incompatible or unbound populated state. Current engines declare that
  capability; direct persistent launches select account-local engine state when
  no explicit directory is supplied. Stale latent-upload cleanup uses only the
  selected engine's staging on the vault disk, not shared scratch, the shared
  vault or another engine's staging. Legacy working stores are backed up before
  conversion.
- Library API 1 can append, read, page and remove scoped, project-owned finished
  history. Retries keep the original run and cannot resurrect deleted history;
  unknown document fields survive. Pages bind their filters and order equal-time
  runs deterministically. This API refuses an unmigrated legacy history rather
  than hiding it. Explicitly helper-backed hosts share finished history while
  keeping accepted work in separate engine files. A failed publication keeps the
  accepted job for recovery, and a repeated terminal write cannot resurrect a
  deleted run. HTTP lists preserve their existing paging and scope isolation;
  unavailable helpers return 503 without fallback. Ordinary startup remains
  legacy, and engine-side collection refuses these helper-backed stores.
  Finished-history records, dependency and derivation manifests, saved records,
  catalog observations, media grants, result offers, renditions, and terminal
  results persist through Library API 1. Result and terminal publications remain
  bound to their producing engine, scope, job, and attempt; conflicting
  publications are refused. Library API 1 preserves provenance records and resolver snapshots
  larger than a single protocol message, including large merged records and
  source withdrawals. Older helpers still support ordinary-sized calls;
  large calls require the advertised capability. Available memory and native
  database limits still apply. Engine acquisition stores can use helper-backed
  provenance and verified public receipts, observing other helpers' writes and
  preserving unknown stored fields. Unavailable advisory provenance does not
  block declared acquisition sources or requirement planning. Resolver consumers
  can use shared helper subscriptions, atomic multi-source refreshes, regional
  leads, filename guesses and durable official-provider choices. Unavailable
  shared resolver state, including a snapshot this engine cannot parse, supplies
  no optional declarations or guesses; management list and unsubscribe failures
  return structured service-unavailable errors.
  Concurrent unsubscribe prevents a stale refresh from restoring its source.
  Project folder preferences can use shared helper storage for project order,
  exclusions, outputs and hidden models. Reads observe peer changes; stale writes
  refuse without losing unknown fields or another project's choices. Preferences
  do not grant filesystem authority or relocate account-wide model locations.
  Shared filesystem-grant configuration supports validated revision-bound reads
  and edits, preserving unknown fields on retained grants. Stale edits refuse
  atomically. Grant configuration and project preferences can also be read from
  one snapshot and patched together: either both revisions match and both changes
  commit, or neither changes. Explicitly helper-backed folder routes commit grants
  and project membership together before publishing live changes. Stale edits
  return conflict without changing live grants or worker snapshots; unavailable
  helper reads return service-unavailable errors. Peer edits become visible on
  folder access. Restart-only engines retain their live grants and snapshots
  until restart. Explicit helper-backed model settings remain account-wide:
  separate libraries share locations and remembered exact files without granting
  one another project membership. Stale model edits refuse, unknown stored fields
  survive, and helper or unmigrated-account failures return service-unavailable
  errors without falling back to JSON. Restart-only engines retain active model
  grants. Ordinary engine startup still uses legacy folder and model storage.
  Named provenance updates retain extensions
  on surviving records without retaining withdrawn digests. Legacy catalogs remain
  read-only until migration; engine byte verification, history and cleanup are unchanged.
- Durable job queue when history is configured: accepted jobs are recorded
  before acknowledgment, and jobs lost to a server crash or restart appear
  in history as "interrupted" (distinguishing never-started from
  was-running); nothing re-runs without explicit resubmission. Explicit
  `HistoryStore` consumers can separate accepted work per engine while sharing
  finished history; recovering one then leaves the other's pending work intact.
  Ordinary startup still uses the single-file store.
- Execution: parallel ready-set scheduling, priority queueing, configurable
  concurrency, queued/running cancellation, first-class caching with
  cross-run single-flight coalescing, and per-submission `cacheEnabled: false`
  execution that recomputes every node without restarting the server or
  releasing loaded model consumers; partial execution (execute up/between/from
  via target outputs), binary WebSocket preview frames
- Nodes may return `BLOCKED(message)` on any output. Default-policy consumers
  skip without failing the run; a nonempty absence reason produces one
  `node_blocked` event per origin per run, including cache replay. Silent
  blockers skip without a notice. Explicit accept, omit and fail input
  policies retain their normal behavior.
- Opted-in native nodes can reject literal inputs and declared linked types
  before producers execute. Admission validation crosses isolated and remote
  worker boundaries and follows routing and unpinned placement policies.
- Runtime expansion preserves child dynamic choices and ordered output
  membership across local, isolated and remote workers and cache replay.
  Expansions may request child reruns without cache hits or coalescing;
  nested expansions inherit that policy unless they explicitly request reuse
  or rerun. The policy survives persistent cache replay.
- Native loop list selection and result collection preserve child value
  metadata, including resource ownership and producer execution arms.
  Pure list construction and element selection retain cross-run cache reuse.
- Memory governance: budgets, headroom, reservations with renewal,
  governed shedding, admission waiting, and item details with Aimdo model-weight
  page residency when the active Aimdo build exposes it
- Memory status HTTP and live events include worker torch allocated/reserved,
  process RSS and registered pinned-host bytes when available, plus GPU name
  and optional UUID-matched NVML utilization, temperature and power. Device
  peaks record observed driver usage and reset to current usage through
  `POST /memory/reset-peak`. Missing probes remain unavailable, not zero.
  Distinct worker processes contribute their allocator and host counters once;
  device-wide free memory is never added across workers.
  Accelerator policy distinguishes current settings from budgets and Aimdo
  policies applied to each existing worker; changing a restart-bound setting
  never relabels a worker that has not restarted.
  Explicitly helper-backed hosts persist runtime preferences per environment.
  CLI choices retain precedence, unknown preferences survive edits, and stale
  saves return 409 without changing live policy. Reading settings refreshes the
  saved-edit view without applying another engine's policy to this one, even when
  that engine saved preferences this version cannot use. Reopening validates and
  adopts the saved values. Unavailable storage returns 503 without fallback.
  Ordinary startup retains file-backed settings.
- Paused, idle queues support `POST /memory/free` to release volatile
  execution caches and live workers' declared memory consumers without
  starting dormant workers. Results report each worker and consumer;
  persistent caches and assets remain intact, and the queue stays paused.
- Remote workers: `dinkster-serve --remote-workers remotes.toml` composes
  `dinkster_workers.service` daemons on other machines; their node types
  execute remotely with per-remote `@name` device budgets
- Remote worker TLS: daemon `--tls-cert`/`--tls-key` plus `tls_ca_file`
  in `remotes.toml` encrypt the session with server-authenticating TLS
  (pinned certificate; the pre-shared token still authenticates the
  engine)
- Remote worker reconnect: the engine redials a dead or not-yet-up
  remote with backoff and swaps the reattached surface in atomically;
  `GET /api/workers` reports `disconnected` in between
- Same-daemon in-flight resume: an admitted invocation survives a network
  interruption for `--resume-grace` seconds (default 120), then rebinds to
  the same engine and daemon processes. Final results replay reliably;
  disconnected progress and preview chatter is intentionally lossy.
- Remote worker session leases: the daemon's one conversation slot is
  a TTL lease (`--lease-ttl`, default 45 s) - engines heartbeat while
  idle, a silent holder is evicted so a crashed engine never wedges the
  daemon, a competing engine is refused with the holder's name, and the
  engine declares a silent daemon dead on the same clock
- Per-job placement hints select `local` or a configured remote for top-level
  nodes and complete region bodies; worker discovery and per-node execution
  attribution are exposed through the server API
- Remote worker asset staging: declared pack assets and job-referenced
  input assets a dispatched node needs are pulled onto the daemon
  before dispatch, digest-verified, from `--advertise-assets` (or, for
  declared assets, declared URLs); daemon: `--asset-vault`
- Remote worker persistent value transport: with a daemon
  `--value-store` and a serve `--library-root`, bulk boundary values
  cross the network once and later runs send digest references, across
  reconnects and re-runs; per-edge moved-bytes accounting in dev
  diagnostics
