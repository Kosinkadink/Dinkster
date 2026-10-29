## Server API and execution

- Schema wire version 1 with stored output descriptors, compact-storage and
  stream input declarations, alpha and mask policies, range-aware inputs, and
  chunk-safe declarations scoped to supported dynamic-combo options. Stream
  execution is not exposed until a node pack supplies matching value codecs
- Resident-resource consumers follow the validated producer execution arm
  without a node-name or native-arm-name allowlist; malformed, unknown, and
  conflicting producer stamps are refused
- Native-only checkpoint loading errors retain detection and configuration
  failure details
- Comfy API prompts resolve EmptyLatentImage through the native generation
  schema, preserving its dimensions, batch size, and links to all four
  sampling surfaces. Explicit `comfy.EmptyLatentImage` graphs retain their
  legacy `comfy.LATENT` contract
- Routes: health/auth, node catalog, extension/composition diagnostics,
  choices, pack assets/templates, including family/model metadata and
  immutable template thumbnails, jobs (status, cancellation, event
  replay), values, queue control, live WebSocket events, settings, memory
  governance, cache trim/export; asset library and history routes when a
  library root is configured, including classified bounded latent upload
- Filesystem mount scans report file, byte, and elapsed progress; publish
  indexed assets before the scan finishes; reuse unchanged-file indexes; and
  keep per-mount indexes under the configured library root
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
  acquisition leads, isolated source removal, exact-basename suggestions, and
  explicit per-subscription P2P trust and license-authority controls
- Official provider subscription bootstrap from explicit URL and stable-ID
  configuration, with identity verification and preserved trust/unsubscribe
  choices; no built-in provider endpoint or identity
- Default-enabled LAN and global P2P through one shared sidecar session for canonical
  BitTorrent v2 descriptors, with six-hour trust expiry, license metadata,
  safe-format verification, durable rate and seeding budgets, metered
  network closure, and immediate resolver-omission revocation. Global resolver
  transport uses trackerless DHT under `lan-and-internet` scope; LAN
  and HTTP fallback remain available when global transport closes.
- Bounded P2P sidecar status diagnostics for native peer failures, listener
  bindings, discovery errors, and upload-counter comparison; no packet capture
  or persistent diagnostic logging
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
- Concurrent native model jobs run on process-isolated GPU replicas configured
  with `dinkster-serve --multi-gpu-devices INDEX,INDEX`
- Single native sampling jobs can use fixed ordered logical CUDA ranks. The
  shared engine distributes guidance lanes for every family, including
  conditional-only requests, masks, progress and state callbacks, custom
  samplers, and guidance transforms. SD controls and IP-Adapter also work in
  distributed mode. No measured device capability or receipt is required.
  MiniMax H3 FL2VA and REF2VA support explicit sequence mode with compatible
  attention and rank geometry. Measurements cover BF16 guidance on Ada and Blackwell and
  BF16 Ulysses sequence execution on two Blackwell GPUs with SDPA and
  dinkster-kitchen INT8 attention. Unmeasured configurations are diagnosed,
  not refused because they lack measurements. CUDA/NCCL transport, tensor
  dtype/shape consistency, and mode-specific geometry checks still apply.
  Flux packed-grid requests scatter multiple joint windows across ranks with
  deterministic merges and progress and state callbacks. Requests without
  multiple windows use shared guidance evaluation.
  `auto` selects guidance or eligible Flux window scattering, never sequence
  parallelism.
- Memory governance: budgets, headroom, reservations with renewal,
  governed shedding, admission waiting, and item details with Aimdo model-weight
  page residency when the active Aimdo build exposes it
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
