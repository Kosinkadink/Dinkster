# H3 in-process model packs - design

Status: APPROVED WITH AMENDMENTS (2026-07-30). Written by the
backend coordinator after the four-gate scoping round with the
inference thread; the inference thread reviewed the doc at 6edd617
and discharged its gate-1 reservation with the amendments now folded
into the tenant-registry section below. H3.1 shipped 7ed4d76.
H3.2's engine implementation plan was jointly adjudicated 2026-07-30
(backend strawman, inference rulings D1-D4 + finding R1 + question
Q1, backend answers accepted): the "H3.2 engine registry" section
below is the binding spec, H3.2a is approved for delegation with no
further full review round, and H3.2b GPU end-to-end testing runs
under a fresh coordinated hold window as usual.
Program context: ROADMAP
"Hosting topology program (H1-H3)"; H1 declared venv groups shipped
b5bc98e, H2 multi-pack worker groups shipped b2f644a.

## Goal

A first-class in-process hosting path for model-related packs that
must share the dinkster-serve process, generalizing InProcessWorker
(packages/dinkster-workers/src/dinkster_workers/in_process.py) beyond the
builtin dev node set, with memory governance participation. Trusted
first-party packs without model residency needs use the same path without
introducing a torch dependency.
Isolation stays the default and P0: removing the policy returns a
pack to its per-pack (or H1/H2 grouped) isolated placement.

## Settled joint constraints (inference ACK, 2026-07-30)

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
   is its own adjudicated slice after the H2 sampling owner settles
   in production (ROADMAP deferral).
4. Torch-free root venv is not negotiable. In-process model-pack
   hosting activates ONLY when serve runs in a torch-capable
   environment, reusing the existing interpreter-split mechanism
   (the --comfy-python precedent in src/dinkster/comfy_compose.py). No
   new hosting tier.

## Policy surface

- hosting.toml (H1, src/dinkster/installer.py _load_hosting_groups)
  gains an `in-process` list naming packs granted in-process
  placement. Policy stays host-internal: never in manifests,
  lockfiles, registry payloads, or the wire.
- Contradictions refuse at policy load: a pack may not appear in both
  a venv group and the in-process list.
- The per-generation sidecar (generations/<n>.hosting.json,
  Installer._write_groups/_groups_of) gains an `inProcess` list.
  Generation topology remains authoritative for restore/rollback
  replay, exactly as H1 decided - no live-policy reinterpretation.

## Admission and doctor gate

- Staging (Installer._stage): an in-process pack gets NO per-pack
  venv provisioning. The M8.1 doctor probe runs under the designated
  torch-capable serving interpreter and additionally verifies exact
  pins: importlib.metadata versions of comfy-aimdo and torch in that
  interpreter must equal the host baseline recorded at stage time.
- Exact-pin failures are NOT overridable by allow_doctor_findings.
  Mixed aimdo in one process is undefined behavior, not a risk a user
  may accept; the override remains available only for ordinary doctor
  findings. (Divergence from M8.1's override shape - deliberate,
  called out for review.)
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
  same per-pack tables and active-pack context that H2 built for
  grouped workers; the InProcessWorker path gets the same context
  wrapping around load/schema/invocation.
- Hot reload: NOT supported for in-process packs in v1 - Python
  re-import semantics in a live process are unsafe. reload of an
  in-process pack refuses with guidance (replace the generation and
  restart serve). Whole-generation activation may reuse a byte-identical
  in-process pack already serving, but changing that pack also requires a
  restart. Removal is supported: routes retract, tenant handles unregister,
  and the imported module remains loaded.

## Memory governance: engine-owned tenant registry

Narrow protocol, defined backend-side in a neutral module
(packages/dinkster-memory alongside MemoryGovernor), implemented by the
engine (inference-owned packages/dinkster-inference-torch /
dinkster-compat-comfy surfaces: ResidencyManager,
NativeResidencyCoordinator):

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
        ) -> None: ...   # v1 addition, jointly adjudicated
                         # 2026-07-30 (H3.2 plan review); see the
                         # H3.2 engine registry section
        async def unregister(
            self, registration: TenantRegistration
        ) -> None: ...

Contract clauses (inference review amendments, 2026-07-30, all
binding on the fake registry's tests in H3.1 and on the engine
implementation in H3.2):

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
  In-place growth is a v2 protocol rev, jointly adjudicated.
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

## Slicing (demo-first, ownership-split-clean)

- H3.1 (backend): policy + sidecar + admission/doctor exact-pin gate
  + PackSpec.in_process + composition/refusal paths + InProcessWorker
  generalization + declared-asset context + tenant-registry protocol
  with a fake implementation in tests. No engine wiring; a torch-free
  in-process pack (no residency needs) works end to end, proving the
  hosting mechanics without crossing the engine boundary.
- H3.2 (backend-implemented, jointly adjudicated): engine
  implementation of ModelTenantRegistry over
  ResidencyManager/NativeResidencyCoordinator per the binding "H3.2
  engine registry" section below, split H3.2a (registry + protocol
  addition + tests, no serve wiring, no GPU) and H3.2b (serve wiring
  + GPU end-to-end under a coordinated hold window).

## H3.2 engine registry (adjudicated plan, 2026-07-30)

Joint adjudication record: backend strawman reviewed by the
inference thread with rulings D1-D4, finding R1, and question Q1;
all rulings folded below verbatim. Approved for H3.2a delegation
with no further full review round. Amended 2026-07-30 (post-
delegation): the H3.2a delegate proved rulings D2 and "no
ResidencyManager changes" contradictory at the manager layer
(ResidencyMechanism.unload() has no refusal channel; free()
unconditionally detaches an unsatisfied candidate at
residency.py:661-676; load() nests free() with no pass seam at
:585-598). The backend proposal resolving the contradiction at the
coordinator layer was CONFIRMED by the inference thread with
clarifications C1 and C2; see "D2 amendment" below. residency.py
remains untouched. Source anchors at 236e3f2:
packages/dinkster-memory/src/dinkster_memory/tenants.py (neutral
protocol), packages/dinkster-inference-torch/src/dinkster_inference_torch/
residency.py (ResidencyManager fleet policy),
packages/dinkster-compat-comfy/src/dinkster_compat_comfy/
native_residency.py (coordinator lock discipline),
src/dinkster/compose.py ServingComposer(tenant_registry=...) +
_PackTenantRegistry (H3.1 seam).

### Module and shape

- New module packages/dinkster-compat-comfy/src/dinkster_compat_comfy/
  tenant_registry.py: torch-free imports, protocol-typed against
  duck-typed manager/coordinator seams exactly like
  native_residency.py.
- NativeModelTenantRegistry implements the dinkster_memory
  ModelTenantRegistry protocol over an EXPLICIT
  NativeResidencyCoordinator constructor argument (no new
  singletons). Serve wiring (H3.2b) binds it to the process's
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

### notify_resident (v1 protocol addition)

- async notify_resident(registration) added to the neutral protocol
  in packages/dinkster-memory/tenants.py (additive); the H3.1
  fake-registry amendment tests are updated in the same slice.
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
- Closes finding R1(b): a never-materialized or shed tenant
  contributes zero phantom freed bytes to any placement pass. R1(a)
  is ordinary allocation uncertainty: pack-side allocation OOM after
  acknowledgment is retriable (fresh notify_resident, or
  unregister), a documented pack contract clause.

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
  callbacks (existing amendment clause).

### Failure policy (ruling D1: fail-poisoned) and budgets (D2)

- A raising or timed-out offload/evict marks the tenant DEFECTIVE:
  accounting zeroed, subsequent registry operations for that tenant
  refuse until unregister, loud log AND structured surfacing in
  composition health/diagnostics (operator-visible, never
  logs-only).
- A timed-out callback coroutine is CANCELLED on the registry loop;
  if cancellation does not complete, the tenant stays DEFECTIVE.
- The engine continues its placement pass on measured free memory;
  a defective tenant never aborts a sampling stage. Known
  asymmetry, accepted in adjudication: a poisoned evict leaves real
  memory held while accounting zeroes it - measured-free ground
  truth keeps placement honest.
- Budgets, both constructor-configurable: 15s per callback, 60s
  aggregate per placement pass. Pass-budget exhaustion skips the
  remaining tenant candidates for that pass (treated non-freeable)
  WITHOUT poisoning them. Pass definition and skip mechanics are
  specified by the D2 amendment below; the aggregate budget resets
  at each epoch increment.

### D2 amendment: pass definition and skip mechanics (2026-07-30)

Adjudicated post-delegation after the H3.2a delegate proved the
original D2 wording unimplementable without ResidencyManager
changes. Backend proposal CONFIRMED by the inference thread with
clarifications C1/C2; both threads independently verified the
safety keystone in source: free()'s loop recomputes the shortfall
from measured free_memory() every iteration and never credits
unload() with declared bytes, so a refusing no-op mechanism
contributes zero phantom progress (the D1 measured-ground-truth
argument).

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
  included - a more faithful reading of the D2 concern than a
  per-free() budget. The 15s per-callback budget is unchanged. The
  aggregate budget resets at each epoch increment.
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
- C1: re-enrollment applies ONLY to healthy, declared-resident
  tenants. A DEFECTIVE tenant is never re-enrolled (it stays out
  of the registry until unregister, per D1). A tenant that was
  actually shed (offload/evict callback succeeded) or never
  materialized re-enters the registry only through the
  notify_resident admission path - never through pass-boundary
  re-enroll. Re-enroll must not become a second
  residency-declaration channel (keeps R1(b) airtight).
- C2: the epoch hook and re-enrollment both execute under the
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
a voluntary unregister. The H3.1 _PackTenantRegistry cleanup path
(pack removal) uses the same unregister plus one empty_cache per
affected device.

Ground truth stays measured free memory (get_free_memory /
dynamic_free_memory): declared-accounting errors degrade gracefully
instead of corrupting the budget.

### Reclaim classes in the aimdo arm (Q1: intended, documented)

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
reservation call in v1 (ruling D4): measured-free suffices; the
H3.2b GPU validation must include one aimdo-pager-active probe
proving a tenant shed raises measured free within the same locked
pass ordering the pager's next placement decision.

### H3.2 slice contents

- H3.2a (backend delegate): tenant_registry.py, the notify_resident
  protocol addition, the ADDITIVE pass-epoch hook in
  NativeResidencyCoordinator (backend-owned native_residency.py;
  per the D2 amendment - residency.py stays untouched), torch-free
  unit tests (fake manager / coordinator / handles mirroring every
  amendment contract), one .venv-torch integration test driving a
  REAL ResidencyManager with
  tenant mechanisms under synthetic pressure (offload-then-evict
  ordering, pass-budget skip, poisoning) as a new additive file
  under packages/dinkster-inference-torch/tests/ (placement sanctioned
  by the joint plan approval; no edits to existing inference-owned
  files), and doc/ledger rows. Narrow adjudicated exception:
  _PackTenantRegistry adds only the ownership-checking
  notify_resident forwarder in compose.py; this is not serve wiring.
  No serve wiring, no GPU work.
- H3.2b (backend, after H3.2a): serve wiring (torch-capable gate,
  arm-correct coordinator binding, ServingComposer tenant_registry
  injection), composition tests, and the capability-gated GPU test
  (real CUDA tenant, forced pressure, callbacks fire, memory
  actually reclaimed, plus the D4 aimdo-active probe) executed
  under a fresh coordinated hold.

## Tests (H3.1 proving set)

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

- No wire/schema change anywhere in H3.
- No sampler/model contribution contract (deferred, ROADMAP).
- No hot reload of in-process packs (deferred, ROADMAP).
- No relaxation of exact-pin admission, ever, without a new joint
  adjudication round.
