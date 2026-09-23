# dinkster-caches

`dinkster-caches` provides `CacheStore` implementations for memory, persistent
disk, and fastest-first layered caching, plus the shared cache-entry wire
format and content-addressed blob store. It depends on values, the
`dinkster-protocol` boundary package, memory, and assets - not on the engine;
the engine and server consume its stores.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

It does not install a console script.

## Use

`MemoryLRUCache` is the smallest `CacheStore` implementation:

```python
from dinkster_caches import MemoryLRUCache

cache = MemoryLRUCache(max_entries=256, max_bytes=512 * 1024**2)
await cache.put(cache_key, outputs)
outputs_or_none = await cache.get(cache_key)
```

The default RAM budget is one eighth of the effective host RAM reported by
`dinkster-memory`, including cgroup limits. `max_entries` is a secondary bound
for small or uncosted values. Costs come from each value tree's declared
per-device metadata; live resource references belong to their owner and do
not count twice. RAM admission and governor pressure evict LRU entries with
cost on the constrained device, without dropping unrelated residencies.

The same payload object retained across keys or list positions counts once;
shedding credits only the footprint actually removed when aliases remain.
Distinct payload objects count separately even if their bytes or fingerprints
match. Aliases hidden behind separate payload wrappers or different array
views remain conservatively charged separately; caches do not resolve values
or assign host-global allocation IDs to infer them.
This measures declared cache retention, not process RSS or physical allocator
releases: references outside this cache are not tracked. A view's declared
cost may cover its entire retained backing allocation rather than its logical
elements.

Compose stores fastest-first. A slower hit is promoted into every faster
layer:

```python
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache

disk = DiskCacheStore(cache_root, registry, max_bytes=20 * 1024**3)
cache = LayeredCache(MemoryLRUCache(), disk)
```

For adjacent memory/disk layers, entries declaring at least 1 MiB of RAM cost
stay in RAM until eviction, then write to disk. Smaller and uncosted entries
write through eagerly so they survive restart without waiting for eviction.
`disk_spill_min_bytes` overrides this threshold on `LayeredCache` (zero defers
every positive-cost entry). Eagerly written entries and promoted disk hits are
clean and do not write back on eviction. Peer hits populate disk before RAM.
Other layer combinations retain write-through behavior.

A disk write error propagates while retaining the dirty victim. Failed RAM
admission rolls back the incoming entry, whose outputs remain owned by the
caller. Explicit clear and resource invalidation discard rather than spill;
clear also cancels this cache's queued disk publications and stale in-flight
promotions. Cancellation belongs to the store being invalidated, not other
stores sharing its root: an overlapping publication by another store may
serialize after the clear. A shared root is not a cross-engine reload fence.
Replacing a key invalidates its previous disk result before RAM admission.

Disk mutations and garbage collection use a root-wide thread and process
lock, so engine instances on one machine may safely share a directory. The
disk budget evicts manifests by last hit and removes payload blobs only after
their final reference disappears. Startup removes interrupted manifests and
orphaned blobs. Entries containing live resource references or values without
persistable bytes are refused by disk and remain usable in memory.
Rehydrated compact-media leaves use `dinkster-values`' `encoded_storage_meta`
to charge the receiver's actual codec bytes to RAM, not the producer's device.
Fingerprints and non-residency metadata are preserved.

`LayeredCache.hits_by_layer` counts hit sources, and the engine reports the
source as `cacheLayer` on `node_cached` events.

The package also exports `DiskCAS` and the versioned entry encoding helpers
for disk and trusted peer-cache transports.

## Learn more

See DESIGN 3.4 for the cache contract and persistent CAS, DESIGN 3.2 for
the cache wire trust boundary, and DESIGN 3.13 for value-tree accounting.
Related invariants are in `docs/hazards.md`, especially H4. Focused coverage
is in `tests/test_engine.py`, `tests/test_cache_share.py`, `tests/test_cas.py`,
and `tests/test_memory.py`.
