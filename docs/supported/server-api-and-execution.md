## Server API and execution

- Supervisor, station and fleet ingress listeners reject unlisted, malformed,
  missing or duplicate HTTP Host headers with `421 host-not-allowed`, including
  WebSocket upgrades. Loopback IP literals and the configured listen hostname
  are accepted; additional names require repeatable `--allow-host HOST`.
- Managed projects persist explicit IPv4/IPv6 public bind addresses, allowed
  Hosts and browser origins, and optional HTTPS certificate/key paths through
  the environment listener command. Defaults remain IPv4 loopback HTTP.
  Multiple specific interfaces share one port; unavailable bindings fail as
  a unit and HTTPS never opens a cleartext fallback listener.
- Environment catalog/status/doctor queries share a JSON command interface.
  Catalog preserves the selected channel's variant manifests; status reports
  configured channel/variant and observed engines without starting them.
  Doctor reports content and pack drift without repairing it.
- Confirmed environment start/stop commands manage named engine bindings with
  explicit device assignments. Repeating an identical ready assignment does
  not restart it; changing an assignment requires stop first. Plan-only calls
  do not launch or stop processes, and empty device lists hide GPUs.
- Managed local environment launches allow operators to edit project folders.
  Grants persist across controlled restarts. New grants default to read-only;
  writable access remains explicit. Remote connections and separately configured
  server launches do not gain mount-edit authority.
- Schema wire version 1 with stored output descriptors, compact-storage and
  stream input declarations, alpha and mask policies, range-aware inputs, and
  chunk-safe declarations scoped to supported dynamic-combo options. Stream
  execution is not exposed until a node pack supplies matching value codecs
- Resident-resource consumers follow the validated producer execution arm
  without a node-name or native-arm-name allowlist; malformed, unknown, and
  conflicting producer stamps are refused
- Routes: health/auth, node catalog, extension/composition diagnostics,
  choices, pack assets, jobs (status, cancellation, event
  replay), values, queue control, live WebSocket events, settings, memory
  governance, cache trim/export; asset library and history routes when a
  library root is configured, including classified bounded latent upload
- The node catalog and native job submission use `/api/nodes` and `/api/jobs`.
  There is no separate ComfyUI schema catalog or prompt submission endpoint.
- A development generation's node catalog carries `dinkster.development` with
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
- Completed image-document responses retain all declared external resources
  under their exact original result source before returning, including cache
  revalidation. Associations survive library restart and remain protected while
  the source is unclaimed or any source-bound consumer survives. Final release
  or replacement ends that protection; stale requests cannot revive it.
  Original offers and completeness remain unchanged. Durable document copies
  still require their own resource declarations; cleanup remains disabled.
- Asset record catalogs upgrade automatically with versioned migrations.
  Matching unversioned catalogs retain scoped records, verified media grants,
  dependency manifests, and derivations; incompatible schemas are refused.
- Scoped retained-value associations persist explicit asset identities for
  documents, history, shown results and previews without parsing workflow
  contents or granting access to bytes. Revision-checked replacement and
  release prevent stale clients from removing newer associations, including
  after restart or owner reuse. Association registration is explicit, not
  automatic history or preview integration; it does not enable cleanup.
- Clients can read immutable scoped result offers and claim their exact
  server-selected roots into fresh shown-result or preview owners. Durable
  source bindings permit reconciliation after a lost acknowledgment; a claim
  cannot overwrite an existing owner or revive a source after its last
  protecting owner releases it. Library-backed jobs persist source offers
  before fresh, cached and coalesced output events and terminal targets;
  events, status and completed-value discovery expose the exact runtime
  node/output binding. Selecting a nested element retains the complete
  parent-output source rather than narrowing its roots. Unknown
  codecs remain incomplete. Portable VIDEO and legacy tensor-layer results
  have complete reference accounting, including previously stored cache
  manifests when their types are admitted. Previously sealed incomplete
  result offers remain incomplete; loading a codec never rewrites their
  authority. This does not cover live previews or restored
  history, certify frontend coverage, or enable cleanup.
- Scoped terminal publication discovery retains original output tokens,
  execution identity and provenance independently of process restart,
  queue/cache retirement and ordinary history deletion. It never substitutes
  a newer publication or reconstructs tokens for legacy rows. Terminal saved
  artifacts carry original offers bound to the exact saved-file descriptor,
  terminal index and optional published preview, retaining both identities.
  Failed, cancelled and unknown-codec sets remain incomplete. This is source
  discovery, not stored output payloads or completed frontend restoration;
  cleanup remains disabled.
- Durable terminal history retains external asset identities from target
  values, saved artifacts and their optional published previews independently
  of execution-cache entries, without storing output payloads or changing
  history responses. Legacy and interrupted history with unknown output
  references refuses collection inspection. This
  does not retain opaque workflow contents or live previews, or enable deletion.
- Cached image-document renders retain their source, resources and output
  identities independently of catalog records. Verified cache reuse can certify
  previously unknown references without changing render provenance or responses.
  Unknown or unreadable derivation references block collection inspection;
  cleanup remains disabled.
- New composed-server vaults have initialized produced accounting before serving,
  including zero adopted bytes before the first produced output. Missing or corrupt
  inventory in existing vaults remains unavailable and is not recreated.
- Operators can inspect vault/adoption byte observations and held known asset
  references without exposing filesystem paths. Unknown accounting is null,
  with explicit blocked reasons. Cache atoms without exhaustive registered
  reference declarations refuse inspection, including inside nested containers
  and after disk-cache restart. Recognized inactive job scratch reports its
  bytes while host-job and store ownership are exclusively held; live, unsafe
  or unknown scratch remains unavailable. Individual owner registrations do
  not enable collection. Server-admitted client generations can acknowledge
  their exact eight-class owner census separately from vault-wide eligibility.
  Census acknowledgments can exceed the job-submission size limit when the
  actual census requires it; unrelated requests keep their normal size bounds,
  and oversized, altered or stale acknowledgments do not authorize collection.
  Native browser imports of selected immutable pack modules need no retention
  header and do not invalidate the census; their authentication and digest
  checks still apply. Management can observe whether queues are idle without
  changing the asset census or reading job values. Full queue reads, executable
  pack routes and asset reads remain admitted.
  Legacy admission, historical unknown owners, disconnected clients, active
  streams, pending reads and stale acknowledgments block collection. Closing
  streams alone does not authorize collection; a quiescent client must still
  acknowledge its exact owner snapshot. Operator Clear and size-policy enforcement
  collect only relinquished produced assets while all reference and publication
  barriers are held; retained bytes remain protected.
  Vault ownership inspection reports busy for active managed readers, writers
  and server jobs, or unavailable for unknown ownership state. Server jobs keep
  that ownership through terminal recovery and durable history publication;
  media publication keeps it through reference commits despite cancellation.
- Startup recovers explicitly offered publications from inactive job scratch
  through verified engine-vault adoption, without rerunning jobs. Scratch removal
  requires evaluated client completeness and verified durable handoffs; restart
  invalidates known clients, and historical unknown clients remain blocked.
  Active, unknown or corrupt stores remain blocked; recovery of another
  recognized job can continue. Existing untracked assets are not reclassified.
  Success, failure and cancellation also reconcile that job's offered bytes
  after execution settles and before reporting its terminal state. Recovery
  preserves the original job outcome; unrelated jobs are untouched.
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
  keep per-mount indexes under the configured library root
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
  Outside, revoked, excluded, missing and unsafe paths return an explicit
  refusal without revealing host paths. Authentication, malformed requests and
  server failures remain distinct errors, not upload-fallback signals.
- Setup, environment-managed generation serving, and serving with mount changes
  enabled initialize a writable
  `<library-root>/imports` folder and a selected writable
  `<library-root>/output` folder when those defaults are not configured.
  Fresh generations can save output without enabling runtime mount editing;
  the existing mounts API reports the selected writable destination.
  Existing output selections, folder paths and read-only grants are preserved; unavailable
  configured directories are not recreated or replaced. Removing the import
  or output grant remains effective across restarts and repeated setup.
  New default folders become available to existing projects without resetting
  folder order, exclusions, project-only grants or output selections.
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
  Interrupted publication recovers completed files on restart or the next
  catalog query, after checking their bytes and current folder authority.
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
  The selected filename must match the needed model. Explicit sources can
  resolve files absent from the template catalog or select among ambiguous
  declarations. Civitai model links can specify `modelVersionId`; matching
  files across versions require a further source choice. Review shows the
  fixed revision/version, provider SHA-256, exact size, reported license and
  physical destination. Civitai size comes from transfer headers, not rounded
  `sizeKB`; review opens the response without consuming its body. Civitai plans
  and published provenance retain reported model-level usage permissions in
  `terms`, with a version-specific model page as `terms.source`. Missing or
  malformed permissions stay unknown; no license identifier is inferred from
  them, and these facts do not grant upstream access or enforce usage policy.
  No model file is written or BLAKE3 identity manufactured before explicit consent.
  Civitai access refusals are distinct from Hugging Face sign-in or grants;
  Hugging Face credentials are never sent to Civitai. Local files take
  precedence; disabled folders, missing writable destinations and ambiguous or
  absent declarations return distinct states. Plans expire after 15 minutes.
  Operators can GET `/api/models/providers/huggingface?scope=local` or
  `/api/models/providers/civitai?scope=local` with asset-read authority to inspect
  only whether that provider's credential is configured.
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
  A state-matched PKCE refusal ends the flow and releases the callback port;
  the owner's next poll reports refusal without exposing provider error text.
  Civitai uses `/api/models/providers/civitai/sign-in` and the same owner-bound
  poll/cancel routes, with PKCE only. The registered Dinkster public client is
  built in; it has no client secret. `DINKSTER_CIVITAI_CLIENT_ID` overrides that
  public ID, and an explicitly empty value disables browser sign-in.
  Provider status, token-save and Forget replies include `pkceAvailable`,
  separately from credential status. It is false for empty or malformed Civitai
  public client configuration, so clients hide unavailable browser sign-in while
  retaining the token fallback. It does not promise callback-port availability
  or provider reachability.
  Its exact registered callback is `http://localhost:15493/oauth/callback/civitai`,
  served on loopback only. A busy callback port fails without choosing an
  unregistered redirect. Civitai PKCE requires a local client and does not
  offer device sign-in. The authorization endpoint is on `auth.civitai.com`; requested
  ModelsRead scope is 4 and the returned grant must be exactly 5 including
  mandatory UserRead. The original client ID is retained for rotating refresh.
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
  POST `/api/workflows/model-provenance` accepts a saved workflow `document`
  and `project` for local human operators with asset-read authority. It observes
  stored model inputs across graph definitions using that project's folders,
  without execution, provider requests or hashing. The version-1 response uses
  graph/node definition paths, basename, kind, size and available bound identity;
  unavailable files are omitted. This is a current engine observation, not
  proof that a workflow ran or that repeated instances used identical inputs.
  Unknown schemas and non-model inputs do not produce observations. Local
  grants and paths are excluded, and optional provider source hints remain
  advisory. Responses are not cached; unavailable observations do not authorize
  acquisition or grant access to a model folder.
  Native job admission captures model inputs reachable in the final execution
  graph in output metadata under `dinkster.modelProvenance` (or a numbered
  suffix when that key is already supplied). Rows identify the nested node
  path, input, model kind, basename, size and any known bound BLAKE3 identity.
  Optional `sourceHints` are explicitly advisory digest-wide records, not proof
  of the source selected for that occurrence or of permission to use it.
  Recorded SHA-256 and canonical provider URLs come only from bounded fields;
  arbitrary URLs, notes, credentials and local paths are excluded.
  Capture does not hash files or wait for indexing. Hashing a replacement cannot
  enrich an earlier local selection. The admitted snapshot preserves caller
  metadata and describes admission-time reachability, not actual runtime use.
  Compatible output nodes receive it through the existing export metadata
  carrier; embedding and recovery depend on the output node and format.
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
  freshness
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
- Durable job queue when history is configured: accepted jobs are recorded
  before acknowledgment, and jobs lost to a server crash or restart appear
  in history as "interrupted" (distinguishing never-started from
  was-running); nothing re-runs without explicit resubmission
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
