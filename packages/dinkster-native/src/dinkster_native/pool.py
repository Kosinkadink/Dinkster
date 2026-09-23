"""ResidentPool: the residency table as a governed memory consumer
(DESIGN 3.10).

resident.py's table holds loaded models by strong reference "until the
child exits". This pool is the promised follow-up. Pressure on a device
lane gets one of two responses, keyed by what the lane means:

- **vram:* pressure - advisory unloading.** ComfyUI's own model manager
  evicts the model's GPU state while the resident identity - and every
  stub pointing at it - stays valid. A stub resolving after pressure
  still works; comfy reloads the model on next use.
- **ram pressure - invalidate-then-release** (comfy's --cache-ram,
  grown up). Freeing host RAM requires dropping the table's strong
  reference, and *that* is only safe once nothing can resolve the stub
  anymore. So release runs in strict order: every registered invalidator
  drops its cache entries referencing the resident (addressed by the
  RESOURCE_ID_META_KEY its envelopes carry); only if all of them succeed
  is device state unloaded and the reference dropped. The next request
  cache-misses and reruns the loader - recompute through the ordinary
  cache-miss path, no dangling stub ever handed out. An invalidator
  failing aborts that resident's release: a possibly-still-referencing
  cache beats reclaimed bytes.

The pool tracks admission (the codec's rid_for), use (a stub resolving is
a model about to be used), and loaded state; under pressure it unloads
least-recently-used models first. It implements the detail contract, so a
memory panel shows "sd15.safetensors, 4.1 GB on cuda:0" with a stable ID
whose unload button cannot race the pool's own mutation. Display names
come from ``label()`` - the loader knows the checkpoint asset's name; the
pool never invents one from a Python class. Unlabeled residents fall back
to their resident ID, which at least cannot collide.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

from dinkster_assets import AssetError, require_digest, validate_vae_hint_field
from dinkster_memory import (
    ConsumerItem,
    FullReleaseCommitResult,
    FullReleaseResult,
    PageMap,
    PressureSignal,
    ReleaseCandidate,
)
from dinkster_values import COST_META_KEY, ResourcePins, resident_owners

from .devices import comfy_resident_meta
from .native_residency import (
    NativeComponentHandle,
    NativeResidencyBusyError,
    NativeRuntimeHandle,
)
from .resident import ResidencyTable, resident_resource_id

log = logging.getLogger("dinkster.native.pool")


def _source_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return validate_vae_hint_field(field, value)


@dataclass(frozen=True, slots=True)
class VaeSource:
    digest: str
    name: str | None = None
    logical_id: str | None = None
    latent_space: str | None = None

    def hint(self) -> str:
        value: dict[str, object] = {"sourceDigest": self.digest, "version": 1}
        if self.name is not None:
            value["sourceName"] = self.name
        if self.logical_id is not None:
            value["sourceLogicalId"] = self.logical_id
        if self.latent_space is not None:
            value["latentSpace"] = self.latent_space
        return json.dumps(value, sort_keys=True, separators=(",", ":"))


CostFn = Callable[[object], Mapping[str, object]]
"""Meta for a resident object; the pool reads its COST_META_KEY entry."""

UnloadFn = Callable[[object], None]
"""Backend eviction of one resident's device state (not the reference)."""

TerminalReleaseFn = Callable[[object], bool]
"""Terminal backend release before dropping the reference. True means the
callback handled unloading; False leaves the existing Comfy path in charge."""

InvalidateFn = Callable[[str], object]
"""Drop every cache entry referencing a resource id; raising aborts the
release of that resident (the return value is ignored)."""


def _declared_cost(meta: Mapping[str, object]) -> dict[str, int]:
    cost = meta.get(COST_META_KEY)
    if not isinstance(cost, Mapping):
        return {}
    return {
        residency: nbytes
        for residency, nbytes in cast("Mapping[str, object]", cost).items()
        if isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0
    }


def _native_page_map(obj: object) -> PageMap | None:
    if not isinstance(obj, (NativeRuntimeHandle, NativeComponentHandle)):
        return None
    maps: list[PageMap] = []
    for mechanism in obj.mechanisms:
        report = getattr(mechanism, "page_map", None)
        if callable(report) and (pages := cast("PageMap | None", report())) is not None:
            maps.append(pages)
    if not maps or any(pages.page_bytes != maps[0].page_bytes for pages in maps[1:]):
        return None
    return PageMap(
        page_bytes=maps[0].page_bytes,
        flags=tuple(flag for pages in maps for flag in pages.flags),
    )


class _PoolEntry:
    __slots__ = ("cost", "display_name", "last_used", "loaded", "vae_source")

    def __init__(self, cost: dict[str, int], *, loaded: bool = True) -> None:
        self.cost = cost
        self.display_name: str | None = None
        self.last_used = 0
        self.loaded = loaded
        self.vae_source: VaeSource | None = None


class ResidentPool(ResidencyTable):
    """A ResidencyTable that is also a governed, detail-contract consumer.

    Registration paths pass this wherever they would pass a table; the
    codec's rid_for/get calls become admission and use-tracking for free.
    Costs are read once at admission - models do not move their
    load_device mid-life, and footprint() runs on every admission so it
    must stay a dictionary walk, never a device probe.
    """

    def __init__(
        self,
        *,
        cost_of: CostFn = comfy_resident_meta,
        unload: UnloadFn | None = None,
        terminal_release: TerminalReleaseFn | None = None,
    ) -> None:
        super().__init__()
        self._cost_of = cost_of
        self._unload = unload
        self._terminal_release = terminal_release
        self._entries: dict[str, _PoolEntry] = {}
        self._dependent_owners: dict[str, set[str]] = {}
        self._owner_dependents: dict[str, set[str]] = {}
        self._clock = itertools.count(1)
        self._invalidators: list[InvalidateFn | weakref.WeakMethod[Any]] = []
        self._pin_registries: list[weakref.ReferenceType[ResourcePins]] = []
        self._coordination_lock = threading.RLock()

    def register_invalidator(self, invalidate: InvalidateFn) -> None:
        """Register a holder of resident references (a cache's
        ``drop_referencing``). Release calls every invalidator before
        dropping a reference - and refuses to release if any of them
        raises, because a cache that may still hold a stub makes the
        release unsafe."""
        with self._coordination_lock:
            owner = getattr(invalidate, "__self__", None)
            if owner is None:
                self._invalidators.append(invalidate)
                return
            try:
                self._invalidators.append(weakref.WeakMethod(cast("Any", invalidate)))
            except TypeError:
                self._invalidators.append(invalidate)

    def register_pins(self, pins: ResourcePins) -> None:
        """Register one engine's live-run resource gate without retaining it."""
        with self._coordination_lock:
            if any(reference() is pins for reference in self._pin_registries):
                return
            self._pin_registries.append(weakref.ref(pins))

    # -- table surface (admission and use-tracking) ----------------------

    def rid_for(self, obj: object) -> str:
        with self._coordination_lock:
            rid = super().rid_for(obj)
            owners = resident_owners(obj)
            owner_rids = {self.rid_for(owner) for owner in owners if owner is not obj}
            if owner_rids:
                previous = self._dependent_owners.get(rid)
                if previous is not None and previous != owner_rids:
                    raise RuntimeError("resident dependency cannot change owners")
                self._dependent_owners[rid] = owner_rids
                for owner_rid in owner_rids:
                    self._owner_dependents.setdefault(owner_rid, set()).add(rid)
            if owners[0] is not obj:
                return rid
            entry = self._entries.get(rid)
            if entry is None:
                # Comfy patchers conservatively enter loaded as before. Native
                # runtimes assemble on CPU and enter with zero VRAM held; their
                # loaded state changes only after successful device placement.
                entry = self._entries[rid] = _PoolEntry(
                    self._read_cost(obj),
                    loaded=not isinstance(obj, (NativeRuntimeHandle, NativeComponentHandle)),
                )
            entry.last_used = next(self._clock)
            return rid

    def get(self, rid: str) -> object:
        with self._coordination_lock:
            obj = super().get(rid)
            owner = resident_owners(obj)[0]
            if owner is not obj:
                self.rid_for(owner)
                return obj
            entry = self._entries.get(rid)
            if entry is not None:
                entry.last_used = next(self._clock)
                # A Comfy stub resolution means its own manager is about to
                # reload the patcher. Native resolution alone moves nothing;
                # native loaded state is reconciled only after actual placement.
                if not isinstance(obj, (NativeRuntimeHandle, NativeComponentHandle)):
                    entry.loaded = True
            return obj

    def set_loaded(self, obj: object, loaded: bool) -> None:
        """Reconcile actual device placement without recording a use.

        Native stage placement and manager-driven cross-model eviction call
        this while holding the process-wide coordinator lock.
        """
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj:
                return
            entry = self._entries.get(rid)
            if entry is not None:
                entry.loaded = loaded

    def remove(self, rid: str) -> object | None:
        """Drop resident state and its attached provenance as one mutation."""
        with self._coordination_lock:
            return self._remove_tree(rid, set())

    def _remove_tree(self, rid: str, removing: set[str]) -> object | None:
        if rid in removing:
            return None
        removing.add(rid)
        for owner_rid in self._dependent_owners.pop(rid, set()):
            dependents = self._owner_dependents.get(owner_rid)
            if dependents is not None:
                dependents.discard(rid)
                if not dependents:
                    self._owner_dependents.pop(owner_rid, None)
        for dependent_rid in tuple(self._owner_dependents.pop(rid, set())):
            self._remove_tree(dependent_rid, removing)
        obj = super().remove(rid)
        self._entries.pop(rid, None)
        return obj

    def release_resident(self, obj: object, *, honor_pins: bool = True) -> bool:
        """Invalidate references and terminally release one admitted resident."""
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj:
                return False
            return self._release_entry(rid, "ram", honor_pins=honor_pins) is not None

    def _live_invalidators(self) -> tuple[InvalidateFn, ...]:
        live: list[InvalidateFn] = []
        retained: list[InvalidateFn | weakref.WeakMethod[Any]] = []
        for registered in self._invalidators:
            invalidate = registered() if isinstance(registered, weakref.WeakMethod) else registered
            if invalidate is None:
                continue
            live.append(cast("InvalidateFn", invalidate))
            retained.append(registered)
        self._invalidators = retained
        return tuple(live)

    def _live_pins(self) -> tuple[ResourcePins, ...]:
        live: list[ResourcePins] = []
        retained: list[weakref.ReferenceType[ResourcePins]] = []
        for reference in self._pin_registries:
            pins = reference()
            if pins is None:
                continue
            live.append(pins)
            retained.append(reference)
        self._pin_registries = retained
        return tuple(live)

    def label(self, obj: object, display_name: str) -> str:
        """Name a resident for humans (the loader knows the asset's name);
        admits the object if it is not yet resident. Returns its rid."""
        with self._coordination_lock:
            rid = self.rid_for(obj)
            self._entries[rid].display_name = display_name
            return rid

    def label_source(
        self,
        obj: object,
        *,
        digest: str,
        name: str | None = None,
        logical_id: str | None = None,
        latent_space: str | None = None,
    ) -> None:
        """Attach path-free VAE lineage without admitting a resident."""
        source = VaeSource(
            require_digest(digest),
            _source_text(name, "sourceName"),
            _source_text(logical_id, "sourceLogicalId"),
            _source_text(latent_space, "latentSpace"),
        )
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj or rid not in self._entries:
                raise AssetError("VAE source can only label an admitted resident")
            entry = self._entries[rid]
            current = entry.vae_source
            if current is not None and current.digest != source.digest:
                raise AssetError("conflicting VAE source digest for one resident")
            if current is not None:
                source = VaeSource(
                    source.digest,
                    current.name or source.name,
                    current.logical_id or source.logical_id,
                    current.latent_space or source.latent_space,
                )
            entry.vae_source = source

    def source_for(self, obj: object) -> VaeSource | None:
        """Observe VAE lineage without admission or LRU use tracking."""
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj:
                return None
            entry = self._entries.get(rid)
            return None if entry is None else entry.vae_source

    def _read_cost(self, obj: object) -> dict[str, int]:
        try:
            return _declared_cost(self._cost_of(obj))
        except Exception:  # noqa: BLE001 - foreign object shapes; no cost beats a crash
            return {}

    def is_loaded(self, resource_id: str) -> bool | None:
        """Whether this pool currently holds a resource's device state.

        ``resource_id`` is the identity stamped on an envelope, not the
        pool-private resident id. None means this worker does not know the
        resource (for example, a cached envelope from an earlier worker
        lifetime). The query is observational: it does not resolve the
        resident or touch its use clock.
        """
        with self._coordination_lock:
            for rid, entry in self._entries.items():
                if resident_resource_id(rid) == resource_id:
                    return entry.loaded
            return None

    # -- governed consumer (Shedder + DetailedConsumer, structurally) ----

    def footprint(self, device: str) -> int:
        """vram:* lanes are held only while loaded (advisory unload frees
        them); every other lane - "ram", the offload copy - is held for
        the reference's whole lifetime, loaded or not."""
        with self._coordination_lock:
            if device.startswith("vram:"):
                entries = (e for e in self._entries.values() if e.loaded)
            else:
                entries = iter(self._entries.values())
            return sum(e.cost.get(device, 0) for e in entries)

    async def shed(self, pressure: PressureSignal) -> int:
        """Free memory on one lane until the pressure is met.

        The lane picks the action: vram:* pressure unloads device state
        (identity survives, stubs keep resolving, next use reloads); ram
        pressure releases residents outright - invalidate references
        first, then drop (see module docstring). Item-targeted pressure
        acts on exactly the named residents - that is a human's "unload
        this model", and partial obedience would be surprising.
        Untargeted pressure goes least-recently-used first, skipping
        residents that hold nothing on the pressured lane (shedding them
        frees no pressure and costs a reload).
        """
        if not pressure.device.startswith("vram:"):
            # Local release is the two-phase surface with a zero-width gap:
            # propose and release run back to back with no suspension, so
            # every token trivially matches. In-process references stay
            # safe without pinning - envelopes hold the object itself, not
            # a stub, so live Python references outlast the table entry.
            outcome = await self.release(pressure.device, self.propose_release(pressure))
            return sum(outcome.values())
        with self._coordination_lock:
            if pressure.items is not None:
                rids = [rid for rid in pressure.items if rid in self._entries]
            else:
                costed = (
                    (entry.last_used, rid)
                    for rid, entry in self._entries.items()
                    if entry.cost.get(pressure.device, 0) > 0 and entry.loaded
                )
                rids = [rid for _, rid in sorted(costed)]
            freed = 0
            for rid in rids:
                if pressure.items is None and freed >= pressure.bytes_needed:
                    break
                freed += self._unload_entry(rid, pressure.device)
            return freed

    # -- two-phase release (dinkster_memory.ReleasableConsumer) -------------

    def propose_release(self, pressure: PressureSignal) -> list[ReleaseCandidate]:
        """Select residents the ram-lane pressure would release; free
        nothing. Item-targeted pressure names exactly those residents;
        untargeted pressure picks least-recently-used first until the
        declared costs cover the need. Each candidate carries the pool's
        use-clock reading as its token - release() refuses a resident
        used since it was proposed."""
        with self._coordination_lock:
            if pressure.device.startswith("vram:"):
                return []  # vram is the advisory lane; nothing to release
            if pressure.items is not None:
                rids = [rid for rid in pressure.items if rid in self._entries]
            else:
                costed = (
                    (entry.last_used, rid)
                    for rid, entry in self._entries.items()
                    if entry.cost.get(pressure.device, 0) > 0
                )
                rids = [rid for _, rid in sorted(costed)]
            candidates: list[ReleaseCandidate] = []
            claimed = 0
            for rid in rids:
                if pressure.items is None and claimed >= pressure.bytes_needed:
                    break
                entry = self._entries[rid]
                nbytes = entry.cost.get(pressure.device, 0)
                candidates.append(
                    ReleaseCandidate(
                        item_id=rid,
                        resource_id=resident_resource_id(rid),
                        nbytes=nbytes,
                        token=str(entry.last_used),
                    )
                )
                claimed += nbytes
            return candidates

    def propose_full_release(self) -> list[ReleaseCandidate]:
        """Propose every resident, including zero-cost entries."""
        with self._coordination_lock:
            return [
                ReleaseCandidate(
                    item_id=rid,
                    resource_id=resident_resource_id(rid),
                    nbytes=entry.cost.get("ram", 0),
                    token=str(entry.last_used),
                )
                for rid, entry in sorted(self._entries.items(), key=lambda item: item[1].last_used)
            ]

    async def full_release(self) -> FullReleaseResult:
        """Terminally release every resident through the ordinary safety gates."""
        return (await self.release_full(self.propose_full_release())).result

    async def release_full(self, candidates: Sequence[ReleaseCandidate]) -> FullReleaseCommitResult:
        """Terminally release the candidates approved by a reference holder."""
        failures: list[str] = []
        released: dict[str, int] = {}
        with self._coordination_lock:
            for candidate in candidates:
                entry = self._entries.get(candidate.item_id)
                if entry is None or str(entry.last_used) != candidate.token:
                    continue
                freed = self._release_entry(candidate.item_id, "ram", failures=failures)
                if freed is not None:
                    released[candidate.item_id] = freed
            if failures:
                result = FullReleaseResult("error", "; ".join(failures))
            elif self._entries:
                result = FullReleaseResult("busy")
            else:
                result = FullReleaseResult("complete")
            return FullReleaseCommitResult(released, result)

    async def release(self, device: str, candidates: Sequence[ReleaseCandidate]) -> dict[str, int]:
        """Release the still-valid candidates: a stale token means the
        resident was used since proposal and is refused (compare-and-drop,
        never destroy fresh state on a stale decision). The outcome maps
        item id -> bytes freed; refused or aborted candidates are absent,
        so the caller can roll back what it staked on them."""
        with self._coordination_lock:
            outcome: dict[str, int] = {}
            for candidate in candidates:
                entry = self._entries.get(candidate.item_id)
                if entry is None or str(entry.last_used) != candidate.token:
                    continue
                freed = self._release_entry(candidate.item_id, device)
                if freed is not None:
                    outcome[candidate.item_id] = freed
            return outcome

    def _unload_entry(self, rid: str, device: str, *, failures: list[str] | None = None) -> int:
        """Evict one resident's device state; freed bytes are the declared
        cost - the same number footprint() reported, so the governor's
        ledger stays consistent. Already-unloaded entries free nothing."""
        with self._coordination_lock:
            entry = self._entries.get(rid)
            if entry is None or not entry.loaded:
                return 0
            try:
                obj = ResidencyTable.get(self, rid)  # not a use: no last_used touch
            except Exception:  # noqa: BLE001 - entry raced away
                return 0
            if self._unload is not None:
                try:
                    self._unload(obj)
                except Exception:  # noqa: BLE001 - backend code may fail arbitrarily
                    if failures is not None:
                        failures.append(f"resident {rid} unload failed")
                    log.warning(
                        "DINKSTER_COMPAT_UNLOAD_FAILED: resident %s remains loaded",
                        rid,
                        exc_info=True,
                    )
                    return 0
            entry.loaded = False
            return entry.cost.get(device, 0)

    def _release_entry(
        self,
        rid: str,
        device: str,
        *,
        honor_pins: bool = True,
        failures: list[str] | None = None,
    ) -> int | None:
        """Invalidate-then-release one resident: references first, device
        state second, the strong reference last. Any invalidator failing
        aborts the release - a cache that may still hold a stub beats
        reclaimed bytes, because a dangling stub is a workflow failure.
        None means aborted or already gone (distinct from a released
        resident that happened to cost zero bytes on this lane)."""
        with self._coordination_lock:
            entry = self._entries.get(rid)
            if entry is None:
                return None
            try:
                obj = ResidencyTable.get(self, rid)
            except Exception:  # noqa: BLE001 - entry raced away
                return None
            guard = (
                obj.terminal_release_guard()
                if isinstance(obj, NativeComponentHandle)
                else nullcontext()
            )
            condemned: list[ResourcePins] = []
            try:
                with guard:
                    resource_id = resident_resource_id(rid)
                    for invalidate in self._live_invalidators():
                        try:
                            invalidate(resource_id)
                        except Exception:  # noqa: BLE001 - cache may still reference
                            if failures is not None:
                                failures.append(f"resident {rid} cache invalidation failed")
                            return None
                    if honor_pins:
                        for pins in self._live_pins():
                            if not pins.condemn(resource_id):
                                for accepted in condemned:
                                    accepted.absolve(resource_id)
                                return None
                            condemned.append(pins)
                    terminal_handled = False
                    if self._terminal_release is not None:
                        try:
                            terminal_handled = self._terminal_release(obj)
                        except Exception:  # noqa: BLE001 - no proven release
                            if failures is not None:
                                failures.append(f"resident {rid} terminal release failed")
                            for pins in condemned:
                                pins.absolve(resource_id)
                            log.warning(
                                "DINKSTER_COMPAT_RELEASE_FAILED: resident %s remains held",
                                rid,
                                exc_info=True,
                            )
                            return None
                    if entry.loaded and not terminal_handled:
                        failure_count = len(failures) if failures is not None else 0
                        self._unload_entry(rid, device, failures=failures)
                        if failures is not None and len(failures) != failure_count:
                            for pins in condemned:
                                pins.absolve(resource_id)
                            return None
                    self.remove(rid)
                    return entry.cost.get(device, 0)
            except NativeResidencyBusyError:
                return None

    def details(self) -> list[ConsumerItem]:
        with self._coordination_lock:
            rows: list[tuple[str, str, dict[str, int], object | None]] = []
            for rid, entry in sorted(self._entries.items(), key=lambda kv: -kv[1].last_used):
                if entry.loaded:
                    nbytes = dict(entry.cost)
                else:
                    # Unloaded: the GPU lanes are free; only the host-side
                    # lanes ("ram", the offload copy) are still held.
                    nbytes = {
                        lane: cost
                        for lane, cost in entry.cost.items()
                        if not lane.startswith("vram:")
                    }
                rows.append(
                    (
                        rid,
                        entry.display_name or f"resident {rid[:12]}",
                        nbytes,
                        self._objects.get(rid),
                    )
                )
        # Mechanism operations reconcile the pool while holding their own locks.
        # Query them only after releasing the pool lock to preserve lock order.
        return [
            ConsumerItem(
                item_id=rid,
                # Never a Python class name (collides); the rid cannot.
                display_name=display_name,
                bytes_by_residency=nbytes,
                pages=None if obj is None else _native_page_map(obj),
            )
            for rid, display_name, nbytes, obj in rows
        ]

    def __len__(self) -> int:
        with self._coordination_lock:
            return super().__len__()


_pool: ResidentPool | None = None
_compat_unload: Callable[[object], None] | None = None


def configure_compat_unload(unload: Callable[[object], None]) -> None:
    """Install the optional compatibility arm's foreign-model unload hook."""
    global _compat_unload  # noqa: PLW0603 - process-wide provider configuration
    _compat_unload = unload


def default_pool() -> ResidentPool:
    """The child process's single pool - created on first use, and public:
    a host that runs the comfy pack in-process registers it with its
    governor (``governor.register_shedder(default_pool(), name=...)``).
    Isolated workers get the same governance through the memory relay:
    declare ``memory_consumers`` in the pack manifest and the parent's
    governor sees the pool as a relayed consumer (vram-advisory only;
    DESIGN 3.10)."""
    global _pool  # noqa: PLW0603 - module-level singleton, child-process scoped
    if _pool is None:
        _pool = ResidentPool(
            unload=resident_advisory_unload,
            terminal_release=native_terminal_release,
        )
    return _pool


def memory_consumers() -> dict[str, ResidentPool]:
    """Manifest entry (``consumers = "dinkster_native.pool:memory_consumers"``):
    the pack's governed consumers, announced through the worker hello and
    served to the parent governor via the memory relay when this pack runs
    isolated."""
    return {"comfy-models": default_pool()}


def resident_advisory_unload(obj: object) -> None:
    """Dispatch advisory VRAM pressure without conflating backend handles."""
    if isinstance(obj, (NativeRuntimeHandle, NativeComponentHandle)):
        obj.advisory_unload()
        return
    if _compat_unload is None:
        raise TypeError("foreign resident requires a configured compatibility unload hook")
    _compat_unload(obj)


def native_terminal_release(obj: object) -> bool:
    """Deregister native mechanisms before the pool claims RAM released.

    False preserves the existing Comfy release path byte for byte.
    """
    if not isinstance(obj, (NativeRuntimeHandle, NativeComponentHandle)):
        return False
    obj.terminal_release()
    return True
