# Cache operations and accounting cautions

Interpret cache footprint as declared retention, not process RSS or physical
allocator releases. References outside a cache are not tracked. Separate payload
wrappers and distinct array views may retain the same backing allocation but
are conservatively charged separately; caches do not resolve values or assign
host-global allocation IDs to infer those aliases. A view's declared cost may
cover its entire backing allocation rather than its logical elements.

Do not treat a shared disk root as a cross-engine reload fence; coordinate
reloads across owners separately. Asset-reference inspection does not retain
values returned to callers after it ends. Follow the deletion-grant warning in
`DiskCacheStore.hold_asset_references()` when coordinating asset collection.

Use `MemoryLRUCache`, `DiskCacheStore` and `LayeredCache` from
`dinkster_caches`; compose faster stores before slower ones. The authoritative
budget, spill, invalidation, persistence and telemetry behavior is in the
[implementations](../packages/dinkster-caches/src/dinkster_caches)
and their tests, not this operating guide.
