# Whole-backend code review record (2026-07-25)

The retroactive full-codebase review ledgered in ROADMAP.md ("Native
inference" section), pulled forward by user decision. Scope: every
backend package, reviewed for API cohesion, layering, hidden state,
dead surface, and doc drift per the AGENTS.md review discipline.
Method: seven independent read-only review passes over disjoint
package groups, findings verified against consumers before being
acted on, fixes landed with regression tests in the same slice.

Statuses: **fixed** (with tests in this slice), **deferred** (ROADMAP
entry with trigger), **docs** (documentation corrected to match code).

## dinkster-engine (needs work -> fixed)

- fixed: caller-controlled duplicate active `run_id` overwrote the
  per-run schema snapshot, breaking snapshot isolation under hot
  reload; now rejected with `ActiveRunIdError`
  (`tests/test_engine.py::test_concurrent_duplicate_run_id_is_rejected`).
- fixed: unknown region output mode fell through a catch-all to the
  gather path; engine assembly now raises loudly.
- fixed: single-flight waiter could spin without yielding after a
  resource-pin refusal; now yields before retrying ownership.
- fixed: a raising event listener made the owner run fail while
  coalesced waiters succeeded; listener exceptions are now non-fatal
  and logged
  (`tests/test_engine.py::test_raising_listener_is_nonfatal_for_owner_and_coalesced_waiter`).
- fixed (cohesion): unconsumed root exports `EventKind`,
  `output_summary` trimmed (definitions retained).

## dinkster-graph (needs work -> fixed)

- fixed: `snapshot_graph()` aliased mutable literals (lists/dicts,
  `TypedLiteral.value`), so post-snapshot mutation could change
  execution content under an already-computed fingerprint; literals
  are now deep-copied, non-JSON literal shapes rejected
  (`tests/test_engine.py::test_snapshot_graph_copies_nested_literals_and_typed_literal_values`).
- fixed: directly constructed `RegionNode` with unknown
  kind/binding/output-mode silently degraded to while/cross/state;
  validation now emits named diagnostics `unknown-region-kind`,
  `unknown-binding-mode`, `unknown-output-mode`
  (`tests/test_regions.py::test_invalid_direct_region_modes_have_named_diagnostics`).
- fixed (cohesion): unconsumed root exports `AnyNode`, `BindingMode`,
  `RegionKind`, `find_cycle`, `reachable_from` trimmed.

## dinkster-values (minor -> fixed)

- fixed: frozen `ValueMeta`/`ResourceHandle` retained caller-owned
  mappings (mutation changed residency/cost facts after construction,
  including after cache entry); mapping fields now snapshot to private
  copies, residency sequences normalized to tuples (plain dict copies,
  not MappingProxyType, because proxies break the worker-boundary
  pickle/JSON path) (`tests/test_values*` review tests).
- fixed (cohesion): unconsumed root exports `AssetDecoderSpec`,
  `BatchMergeSpec`, `InlinePayload`, `PyObjPayload`,
  `list_fingerprint` trimmed.

## dinkster-caches (needs work -> fixed)

- fixed: disk `_trim()` credited shared-blob bytes per evicting
  manifest and could return while still over budget; now GCs and
  remeasures actual usage after each eviction
  (`tests/test_caches_review.py`, updated `tests/test_cas.py`).
- fixed: malformed pickle metadata escaped the conservative-miss
  boundary (`UnpicklingError`, `EOFError`, etc. now decode as a miss).
- fixed: `put()` early-returned on an existing corrupt CAS blob
  without verifying content; existing blobs are digest-verified and
  atomically replaced on mismatch.
- fixed: memory cache `get()` returned its internal output dict; now
  returns a fresh copy.
- fixed (cohesion): unconsumed root exports `CASError`,
  `ENTRY_WIRE_VERSION`, `EncodedValue`, `iter_manifest_payloads`
  trimmed.

## dinkster-workers (needs work -> fixed; the one deferral shipped 2026-07-25)

- fixed: `IsolatedWorker.start()` leaked the boundary listener and
  `dinkster-iso-*` temp dir when the launcher itself raised; cleanup now
  covers listener creation and launch
  (`tests/test_workers_review_regressions.py`).
- fixed: `InProcessWorker.invoke()` ran synchronous `execute()` on the
  event-loop thread, blocking everything; sync nodes now run via
  `asyncio.to_thread` with reporter context preserved.
- fixed: frozen `DeviceMap`/`LaunchSpec` retained caller-owned
  mappings; now snapshot on construction.
- fixed (cohesion): fourteen unconsumed root exports trimmed
  (`Launcher`, `SubprocessLauncher`, `PlacePolicy`, `run_service`, env
  constants, etc.; definitions retained).
- ~~deferred~~ **revived and shipped** (2026-07-25, user pulled the
  deferral forward): the Invocation/result/event contracts now live in
  `dinkster-protocol`, a leaf over schema/values; workers and caches
  depend on it instead of dinkster-engine, and child venv provisioning no
  longer installs the scheduler (`_HOST_REQUIREMENTS` dropped
  dinkster-engine and dinkster-graph). dinkster-engine re-exports the contracts
  as part of its own API. Enforced by `tests/test_dependency_rule.py`.

## dinkster-memory (minor -> fixed)

- fixed: `MemoryGovernor._admit()` ran shedding unbounded, so a hung
  shedder blocked admission past the caller's timeout; the remaining
  deadline now bounds the shed phase and expiry raises
  `ReservationTimeout` (`tests/test_memory_review_regressions.py`).
- fixed: `LeaseBroker.close()` did not own admission-pending holder
  tasks; all holder tasks are tracked from creation and
  cancelled/awaited on close.
- fixed: frozen `ConsumerItem` mapping snapshot.

## dinkster-server (needs work -> fixed)

- fixed: job failures published raw `str(exc)` with no stable shape;
  public error bodies are now stable `{kind, type, message}` with the
  full exception logged server-side under the job id. The traceback
  is a separate optional field so clients can always render the
  message alone; it defaults ON (user directive 2026-07-25 - local
  tool, developer-friendly like ComfyUI) and locked-down environments
  opt out with the new `debug_errors=False` flag on `create_app` /
  `JobQueue`.
- fixed: events WS waited only on inbound frames while the outbound
  sender ran unsupervised; now FIRST_COMPLETED over both with
  exception retrieval and guaranteed unsubscribe/close.
- fixed: peer cache blob fetch trusted manifest `size` and read
  unbounded; now capped (`DEFAULT_MAX_BLOB_BYTES` 256 MiB,
  configurable), oversized manifests rejected before request, reads
  bounded to size+1 (`tests/test_cache_client_review_limits.py`).
- fixed (deps): missing `dinkster-memory` workspace mapping in
  pyproject.

## dinkster-supervisor (needs work -> fixed)

- fixed: startup timeout left the engine child alive and unmanaged;
  now terminate -> grace -> kill -> await, exit recorded
  (`tests/test_supervisor_review_process.py`).
- fixed: a stalled health response (`TimeoutError`) crashed `_watch()`
  and left the link "starting" forever; now counted as a failed probe.
- fixed: `/supervisor/restart` bypassed the station's per-install
  lock and `EngineProcess.restart()` had no internal lock; lifecycle
  operations are now serialized by an internal asyncio lock.
- fixed (cohesion): unconsumed package-root `main` re-export trimmed.

## dinkster-api (clean)

No findings; the frozen v1 surface is pinned by `tests/test_api_v1.py`.

## src/dinkster composition (minor -> fixed)

- fixed (deps): `aiohttp` now declared directly in the root
  pyproject (was imported via transitive resolution only).
- docs: umbrella docstring claimed the stable v1 extension surface did
  not exist; now points pack authors at `dinkster_api.v1`.
- fixed: reload_api error responses aligned with the new error shape
  (dev-only surface keeps tracebacks on).

## dinkster-assets (needs work -> fixed + one deferral)

- fixed: `AssetWriter` published a zero-byte final file during its
  claim window; publication is now atomic no-replace `os.link` of the
  completed temp file (`tests/test_assets_review_atomicity.py`).
- fixed: sidecar append/prune raced (read-construct-replace could drop
  a concurrent save record); both paths now take a cross-process
  `fcntl` lock.
- fixed: vault `commit()` fallback accepted any existing target file
  on `os.replace` failure; the target's digest is now verified, with
  a named `VaultError` on mismatch.
- docs: size/mtime described honestly as cache-invalidation
  heuristics, not identity proof.
- fixed (2026-07-25, deferral revived by user): fd-verified reads.
  Every byte-consumption boundary now verifies content against the
  digest from the opened descriptor (`dinkster_assets/integrity.py`:
  `open_verified`/`verified_local_path`, bounded POSIX-only
  fingerprint cache, fork-safe); AssetRef.open/read_bytes/local_path,
  both serving endpoints (`/assets/{digest}`, `/api/assets/{digest}`)
  and the metadata probe stream/probe from the SAME verified
  descriptor with cancellation-safe handoff
  (`dinkster_server/asset_stream.py`); mismatches are named
  `AssetIntegrityError` -> HTTP 409 + no-store, never wrong bytes
  (`tests/test_asset_integrity.py`). The (size, mtime) heuristic
  remains for discovery only. `local_path()` is the documented weaker
  path-shaped boundary for path-only third-party APIs (checkpoint
  loading via ComfyUI).

## dinkster-schema (needs work -> fixed)

- fixed: decoder role dispatch turned ANY unknown role into an
  OutputSpec; the role set is now closed
  (`tests/test_schema_review_validation.py`).
- fixed: decoder coerced untrusted fields (`bool()`/`str()`/`int()` -
  `"false"` decoded to True); every field is now shape-validated,
  bool/int cross-typing rejected. Encoder output and
  schema identity were unchanged by that historical fix.
- fixed: widget-socket binding now enforced for BooleanWidget,
  ComboWidget, AssetWidget, SaveTargetWidget (previously only
  Number/String); no node-pack schema violated the new checks.

## dinkster-collab (needs work -> fixed)

- fixed: `SessionOp` retained caller-owned nested values (post-append
  mutation changed the authoritative log) and accessors exposed
  mutable internals; JSON values are now normalized/copied at the
  boundary, accessors return isolated data
  (`tests/test_collab_review_isolation.py`).
- fixed: WS subscribers registered before the session descriptor was
  sent (op-before-descriptor race); descriptor now sent first, then
  registration with revision-deduped replay. Fan-out decoupled from
  the append path via bounded per-subscriber queues; slow/failed
  subscribers are dropped, never stalling the mutation response.
- docs: versioning claims narrowed to operation/session envelopes
  (snapshot routes are unversioned by design).

## Registry model and hosted registry (needs work -> fixed)

The HTTP and storage implementation and its tests enforce these contracts:

- fixed (security): POST /publish passed the artifact digest into
  `ArtifactVault.path_of()` unvalidated - `sha256:../outside`
  traversed out of the vault and the archive was probed before
  digest admission. Digest grammar (64 lowercase hex) now validated
  at the route AND `path_of()` independently enforces grammar +
  containment (`tests/test_registry_service.py` traversal cases).
- fixed: `resolve_review()` treated any decision other than
  "rejected" as acceptance (fail-open); the decision set is closed.
- fixed: `record_acceptance()` trusted caller-supplied
  `Verdict.new_claims`; release claims are now verified against
  grants/new_claims before mutation.
- fixed: `add_member()`/`set_role()` accepted invalid roles that later
  crashed `authorize()` and poisoned the store; roles validated at
  mutation.
- fixed: rehydration now validates release claims and cross-checks
  grant ownership (fail-at-open honored).
- fixed: transactional SQLite failure left in-memory state diverged
  from the rolled-back DB; mutations now rehydrate from SQLite on
  transactional error.

## dinkster-nodes-std (clean) / dinkster-nodes-dev (minor -> fixed)

- std: no findings; the STD_CLAIMED_V1_NAMES cross-pin with compat is
  complete (`tests/test_primitives.py`).
- dev: README claimed numeric bounds / seed controllers / multiline /
  displayName were inexpressible (stale pre-v11 text); corrected.

## dinkster-compat-comfy (needs work -> fixed + one deferral)

- fixed: `_unload_entry()` marked models unloaded and reported
  reclaimed bytes even when the ComfyUI unload hook failed (governor
  over-admission); failure now keeps `loaded=True`, returns 0, warns
  `DINKSTER_COMPAT_UNLOAD_FAILED` (updated `tests/test_compat_pool.py`).
- fixed: skip diagnostics keyed by flat v1 name collided across
  quarantined packs (lost/stale skip reasons); now keyed by
  namespaced identity (`tests/test_compat_review_regressions.py`).
- fixed: the `init_extra_nodes` fallback caught TypeErrors raised
  INSIDE ComfyUI init and re-called (double partial init risk); the
  signature is now probed via `inspect.signature` and the function is
  called exactly once.
- fixed: sampler/scheduler choice enumeration hard-required
  `KSampler.SAMPLERS/SCHEDULERS` at entry import (whole pack failed to
  start on vintages lacking them); now capability-probed with a named
  warning and empty choice lists.
- fixed (cohesion): unconsumed root exports `comfy_model_unload`,
  `PromptTranslation` trimmed.
- **deferred**: translated-combo determinism (frozen
  environment-derived options) - ROADMAP entry with trigger.

## dinkster-inference / dinkster-inference-torch (minor -> fixed)

Per-slice reviews already covered design/port fidelity; this pass
found only cross-slice drift:

- docs: stale "Stage 1 protocols only" / "stage 4b" package
  descriptions replaced with stage-agnostic current descriptions.
- fixed (cohesion): orphaned never-consumed contracts removed
  (`LatentTransform`, `PrecisionPolicy`, `TextEncoder`,
  `TextEncoderDescriptor`); revivable from git if a family-wiring
  slice needs them.
- fixed: FP8 dtype table duplicated between `sources.py` and
  `assemble.py`; assembly now validates against the canonical table.

## Validation

All gates green at completion: ruff clean, pyright 0 (root and torch
projects), full suite 2270 passed / 16 skipped (was 2240 before the
slice - net +30 regression tests), torch suite 423 passed / 85
skipped, CUDA GPU suite 97 passed (2x RTX 4090), live compat suite
10/10 against the real ComfyUI install.
