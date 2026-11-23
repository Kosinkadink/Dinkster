# In-process model packs - design

## Goal

A first-class in-process hosting path for model-related packs that
must share the dinkster-serve process, generalizing InProcessWorker
(packages/dinkster-workers/src/dinkster_workers/in_process.py) beyond the
builtin dev node set, with memory governance participation. Trusted
first-party packs without model residency needs use the same path without
introducing a torch dependency.
Isolation stays the default: removing the policy returns a
pack to its per-pack or grouped isolated placement.

## Constraints

1. No independent residency authority. In-process packs never call
   aimdo or allocate governed model memory directly; they register
   model handles (size estimate + offload/evict callbacks) through an
   ENGINE-OWNED registration API, and the native inference engine
   mediates all aimdo interaction - same residency domain, same
   headroom/budget accounting (e.g. simple_vram_headroom). Packs are
   tenants, not co-owners. Partitioned budgets rejected.
2. Exact aimdo+torch pins. Admission and the install doctor refuse
   any in-process pack whose environment does not resolve to the
   host's EXACT aimdo+torch pins - no compatibility ranges (aimdo
   ships with no backward-compat intent, user directive). The check
   is runtime-installed-version-vs-pin, not lockfile-only (live drift
   example: runner-1 ComfyUI had comfy-aimdo 0.4.10 installed against
   a 0.4.9 requirements pin).
3. Nodes only in v1. In-process packs contribute nodes executing
   against engine-owned models; sampler and model-registry/extension
   contributions are EXPLICITLY REFUSED at composition (loud
   CompositionError), not silently ignored. The contribution contract
   is outside this hosting contract.
4. Torch-free root venv is not negotiable. In-process model-pack
   hosting activates ONLY when serve runs in a torch-capable
   environment, reusing the existing interpreter-split mechanism
   (the --execution-python precedent in src/dinkster/comfy_compose.py). No
   new hosting tier.

## Policy surface

- hosting.toml (src/dinkster/installer.py _load_hosting_groups)
  has an `in-process` list naming packs granted in-process
  placement. Policy stays host-internal: never in manifests,
  lockfiles, registry payloads, or the wire.
- Contradictions refuse at policy load: a pack may not appear in both
  a venv group and the in-process list.
- The per-snapshot sidecar (`generations/<n>.hosting.json`,
  Installer._write_groups/_groups_of) gains an `inProcess` list.
  Snapshot topology remains authoritative for restore replay, with no
  live-policy reinterpretation.

## Admission and doctor gate

- Staging (Installer._stage): an in-process pack gets NO per-pack
  venv provisioning. The doctor probe runs under the designated
  torch-capable serving interpreter and additionally verifies exact
  pins: importlib.metadata versions of comfy-aimdo and torch in that
  interpreter must equal the host baseline recorded at stage time.
- Exact-pin failures are NOT overridable by allow_doctor_findings.
  Mixed aimdo in one process is undefined behavior, not a risk a user
  may accept; the override remains available only for ordinary doctor
  findings.
- Runtime re-check: composition re-verifies installed aimdo/torch
  versions against the recorded pins before loading an in-process model
  pack. Generic first-party packs declare no runtime pins.

## Composition

- PackSpec (src/dinkster/compose.py) gains `in_process: bool = False`,
  mutually exclusive with worker_group/group_manifests/python;
  _validate_worker_group_specs-style preflight refuses
  contradictions before any worker starts.
- add_pack with an in-process spec imports the pack manifest and
  entry in the host process - the documented, deliberate exception
  to the serve.py contract line "pack code is not imported into the
  host" - and builds an InProcessWorker over its node classes, wired
  into routing through the same worker protocol as every other pack.
- Startup gates, all fail-closed with named errors:
  - an in-process pack with model runtime pins in a non-torch-capable
    environment -> refuse the spec at composition (the pack is NOT
    silently demoted to isolated; placement is policy, and policy that
    cannot be honored is an error).
  - engine tenant-registry provider absent -> refuse any in-process
    pack that declares model residency needs.
  - manifest declares sampler/extension/model-registry
    contributions -> refuse (constraint 3).
- Declared assets: in-process packs install and resolve through the
  same per-pack tables and active-pack context as grouped workers;
  the InProcessWorker path gets the same context wrapping around
  load/schema/invocation.
- Hot reload: NOT supported for in-process packs in v1 - Python
  re-import semantics in a live process are unsafe. reload of an
  in-process pack refuses with guidance (replace the snapshot and
  restart serve). Whole-snapshot activation may reuse a byte-identical
  in-process pack already serving, but changing that pack also requires a
  restart. Removal is supported: routes retract, tenant handles unregister,
  and the imported module remains loaded.

## Memory governance: engine-owned tenant registry

Narrow protocol, defined backend-side in a neutral module
(packages/dinkster-memory alongside MemoryGovernor), implemented by the
engine-facing model runtime:

    class ModelTenantHandle(Protocol):
        pack: str                 # attribution
        model_id: str             # stable within the pack
        size_estimate_bytes: int  # admission estimate, IMMUTABLE
        device_preference: str | None   # hint only
        async def offload(self) -> None: ...   # move off-device
        async def evict(self) -> None: ...     # drop entirely

    @dataclass(frozen=True)
    class TenantRegistration:
        registration_id: str
        assigned_device: str      # the ONLY device the pack may
                                  # allocate governed memory on

    class ModelTenantRegistry(Protocol):
        async def register(
            self, handle: ModelTenantHandle
        ) -> TenantRegistration: ...
        async def notify_resident(
            self, registration: TenantRegistration
        ) -> None: ...   # acknowledge residency before allocation
        async def unregister(
            self, registration: TenantRegistration
        ) -> None: ...

Contract clauses, binding on the fake-registry tests and engine implementation:

- Callbacks are ASYNC, IDEMPOTENT (the engine may retry them), and
  NON-REENTRANT: calling register/unregister from inside offload or
  evict is a contract violation refused loudly by the registry.
- The ENGINE ASSIGNS PLACEMENT. device_preference is a hint the
  engine may ignore; register returns the assigned_device and the
  pack allocates governed memory ONLY there - allocating elsewhere
  is a contract violation.
- Pressure ordering: the engine normally calls offload before evict,
  but MAY call evict without prior offload under severe pressure,
  and may call either callback at any time between register and
  unregister. unregister is safe to call while a callback is in
  flight; the engine serializes callback and unregister execution.
- size_estimate_bytes is IMMUTABLE once registered. A pack whose
  model size changes must unregister and re-register; the
  re-register may be refused, and that refusal IS the budget signal.
  In-place growth requires a v2 protocol revision.
- Identity: (pack, model_id) must be unique among live
  registrations; register refuses duplicates loudly rather than
  double-counting the budget.
- In-process pack nodes obtain the registry through a provided
  service/context object; they never import aimdo. The engine folds
  tenants into its existing residency domain so headroom/budget
  accounting sees one owner.
- Registration is required BEFORE any governed allocation; the
  registry may refuse (over budget), and a refused registration
  means the pack must not allocate - enforcement is contractual plus
  review in v1, with the doctor refusing declared aimdo dependencies
  in in-process packs.

Torch-free in-process packs without residency needs use the hosting path
without an engine tenant registry.

## Engine registry

The neutral tenant protocol lives in
`packages/dinkster-memory/src/dinkster_memory/tenants.py`; composition is owned
by `ServingComposer` and `_PackTenantRegistry` in `src/dinkster/compose.py`.

### Module and shape

- New module packages/dinkster-compat-comfy/src/dinkster_compat_comfy/
  tenant_registry.py: torch-free imports, protocol-typed against
  duck-typed manager/coordinator seams exactly like
  native_residency.py.
- NativeModelTenantRegistry implements the dinkster_memory
  ModelTenantRegistry protocol over an EXPLICIT
  NativeResidencyCoordinator constructor argument (no new
  singletons). Serve wiring binds it to the process's
  arm-correct coordinator: default_native_residency() in the eager
  arm, default_native_residency(free_memory=dynamic_free_memory) in
  the aimdo arm.
- Each live registration wraps its ModelTenantHandle in a
  _TenantMechanism adapter implementing ResidencyMechanism, enrolled
  in the coordinator's manager registry: one residency domain, one
  MRU fleet policy, no partitioned budget (settled constraint 1).

### _TenantMechanism semantics (no ResidencyManager changes)

- total_bytes = size_estimate_bytes (immutable); demand_paged False.
- loaded_bytes = 0 from enroll until notify_resident acknowledges;
  size_estimate_bytes while resident; 0 again after a successful
  offload or evict callback completes.
- partially_load is a no-op returning 0: the PACK allocates; the
  engine never loads tenant weights.
- partially_unload(n) runs handle.offload() and reports
  size_estimate_bytes freed; unload() runs handle.evict(). This
  rides manager.free()'s existing shortfall-vs-loaded_bytes split:
  moderate pressure offloads (recoverable), severe pressure evicts
  directly - exactly the settled two-level ordering contract, with
  either callable at any time between register and unregister.

### notify_resident

- async notify_resident(registration) is part of the neutral protocol
  in packages/dinkster-memory/src/dinkster_memory/tenants.py and its
  fake-registry contract tests.
- The pack MUST await notify_resident acknowledgment before
  allocating governed memory - both first materialization after
  register and any re-materialization after a shed.
- On notify_resident the engine, under the coordinator lock, re-runs
  free(size_estimate + reserve) as the (re-)residence admission
  path, then flips the mechanism resident and touches it for MRU
  recency before acknowledging. The call may refuse; that refusal is
  the budget signal.
- A successful offload callback flips the mechanism non-resident.
- Re-materializing without an acknowledged notify_resident is a
  contract violation the engine may treat as DEFECTIVE.
- A never-materialized or shed tenant contributes zero phantom freed
  bytes to any placement pass. Pack-side allocation OOM after
  acknowledgment is retriable with a fresh notify_resident or unregister.

### Serialization (inference hard constraint)

- Tenant callbacks fire ONLY inside manager passes, which run
  exclusively under the coordinator RLock - the same lock held
  across stage() placement AND the caller's full sampling forward.
  A callback can therefore never interleave with an active sampling
  step.
- register/unregister/notify_resident acquire the coordinator lock
  via await asyncio.to_thread(...) so the event loop never blocks
  behind an active sampling run; they wait rather than refuse.
- Callback bridging: the registry owns a dedicated callback event
  loop thread; _TenantMechanism submits handle.offload()/evict() via
  run_coroutine_threadsafe and blocks on the result under the
  budgets below. Callback code must not touch engine residency and
  must not re-enter registry operations (extends the existing
  non-reentrancy clause).
- A per-tenant async lock serializes unregister against in-flight
  callbacks.

### Failure policy and budgets

- A raising or timed-out offload/evict marks the tenant DEFECTIVE:
  accounting zeroed, subsequent registry operations for that tenant
  refuse until unregister, loud log AND structured surfacing in
  composition health/diagnostics (operator-visible, never
  logs-only).
- A timed-out callback coroutine is CANCELLED on the registry loop;
  if cancellation does not complete, the tenant stays DEFECTIVE.
- The engine continues its placement pass on measured free memory;
  a defective tenant never aborts a sampling stage. A poisoned evict
  leaves real memory held while accounting zeroes it - measured-free
  ground truth keeps placement honest.
- Budgets, both constructor-configurable: 15s per callback, 60s
  aggregate per placement pass. Pass-budget exhaustion skips the
  remaining tenant candidates for that pass (treated non-freeable)
  WITHOUT poisoning them. Pass definition and skip mechanics are
  specified below; the aggregate budget resets at each epoch increment.

### Pass definition and skip mechanics

The free() loop recomputes the shortfall from measured free_memory() every
iteration and never credits unload() with declared bytes, so a refusing
no-op mechanism contributes zero phantom progress.

- Placement pass (a): one placement pass = one top-level
  coordinator-lock-held placement operation (stage(),
  advisory_unload(), or a registry-initiated admission free),
  detected by lock-depth/top-level entry in
  NativeResidencyCoordinator. The coordinator gains an ADDITIVE
  pass-epoch hook: the epoch counter increments on each top-level
  entry and is exposed to the registry. stage() already holds the
  RLock across manager.load() INCLUDING its nested free() cascades,
  so the 60s aggregate bounds total tenant-callback time added to
  any single native-sampling-blocking window, nested frees
  included, rather than each nested free() call. The per-callback
  budget is 15s; the aggregate budget resets at each epoch increment.
- Skip via detach-then-re-enroll (b): on pass-budget exhaustion a
  _TenantMechanism refuses work - partially_unload returns 0 and
  unload() no-ops WITHOUT invoking the pack callback - accepts the
  manager's unconditional detach, and the registry re-enrolls the
  mechanism at the next epoch increment (under the coordinator
  lock, before the manager snapshots candidates) or at the next
  registry operation, whichever comes first. The interim
  unshedable window is acceptable: measured-free ground truth
  keeps placement correct, and the window can only persist while
  no placement pressure exists.
- MRU wrinkle (c), documented v1 behavior: a re-enrolled tenant
  re-enters as newest in the MRU registry, losing its prior
  recency position.
- Re-enrollment applies ONLY to healthy, declared-resident
  tenants. A DEFECTIVE tenant is never re-enrolled; it stays out
  of the registry until unregister. A tenant that was
  actually shed (offload/evict callback succeeded) or never
  materialized re-enters the registry only through the
  notify_resident admission path - never through pass-boundary
  re-enroll. Re-enroll must not become a second
  residency-declaration channel.
- The epoch hook and re-enrollment both execute under the
  coordinator lock so they serialize against candidate snapshots;
  no epoch state may be read or advanced lock-free.

### Register / unregister flow

register(handle):
1. Validate handle fields; refuse duplicate live (pack, model_id)
   loudly.
2. Assign placement: v1 assigns the process's native load device;
   device_preference is an ignorable hint, refused only when it
   names a nonexistent device.
3. Admission: refuse when size_estimate_bytes exceeds governed
   capacity (device total minus the unified minimum free reserve).
4. Under the coordinator lock: manager.free(size_estimate + reserve,
   device) to make room (may shed other tenants or native models
   per policy), then enroll the mechanism non-resident
   (loaded_bytes 0 until notify_resident).
5. Return frozen TenantRegistration(registration_id=uuid4,
   assigned_device).

unregister(registration): await any in-flight callback, then
manager.remove([mechanism], unload=False) - the pack owns freeing
its memory before unregistering; the engine does NOT invoke evict on
a voluntary unregister. The _PackTenantRegistry cleanup path
(pack removal) uses the same unregister plus one empty_cache per
affected device.

Ground truth stays measured free memory (get_free_memory /
dynamic_free_memory): declared-accounting errors degrade gracefully
instead of corrupting the budget.

### Reclaim classes in the aimdo arm

Tenants never appear in manager.load()'s todo list (tenant memory is
never loaded through the manager), so in the aimdo arm every native
load pass runs free(skip_demand_paged=True): the for_dynamic credit
branch subtracts each demand-paged native's loaded bytes from the
requirement BEFORE any tenant is considered, and tenants shed only
the residual the pager cannot cover by self-paging. Within the
non-skipped class, MRU candidate ordering applies. Framing: one
residency domain with two reclaim classes - paged native bytes
credit the requirement; explicit shedding targets non-paged
residents (tenants). In the eager arm there is no skip and tenants
compete in plain MRU order with natives. No explicit aimdo
reservation call in v1: measured-free suffices. GPU
validation must include one aimdo-pager-active probe proving a tenant
shed raises measured free within the same locked pass ordering the
pager's next placement decision.

### Validation

- Torch-free tenant-mechanism tests cover synthetic pressure,
  offload-then-evict ordering, pass-budget skip and poisoning.
- _PackTenantRegistry in compose.py checks ownership when forwarding
  notify_resident.
- Composition tests cover the torch-capable gate, coordinator binding
  and ServingComposer tenant_registry injection.
- Capability-gated GPU tests use a real CUDA tenant and forced pressure
  to check callbacks and memory reclamation, including an aimdo-active probe.

## Tests

- Placement neutrality: a pack's schemas, /api/nodes provenance, and
  invocation results byte-identical isolated vs in-process.
- Refusals: non-torch serve env; sampler/extension contribution;
  venv-group + in-process contradiction; doctor exact-pin refusal
  (and its non-overridability); runtime pin drift; reload refusal;
  registry-absent with declared residency needs.
- Tenant lifecycle: register on add, unregister on remove, refused
  registration blocks the pack (fake registry).
- Amendment contracts on the fake registry: duplicate (pack,
  model_id) registration refused loudly; reentrant
  register/unregister from inside offload/evict refused loudly;
  size_estimate immutability (growth = unregister + re-register,
  with refusable re-register as the budget signal); unregister safe
  while a callback is in flight (serialized); pack allocation bound
  to assigned_device.
- Solo/absent-policy behavior byte-identical to today.

## Non-goals

- No wire/schema change in this hosting contract.
- No sampler/model contribution contract.
- No hot reload of in-process packs.
- No relaxation of exact-pin admission without joint review.
