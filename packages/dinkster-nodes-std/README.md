# dinkster-nodes-std

`dinkster-nodes-std` is the install suite for Dinkster's standard first-party node
packs. It depends on:

- `dinkster-nodes-foundation` for lightweight logic, math, text, list, and utility
  nodes.
- `dinkster-nodes-media-io` for asset-backed media loading, saving, and save-target
  construction.
- `dinkster-nodes-image` for deterministic image and mask operations.
- `dinkster-nodes-remote` for catalog-driven remote nodes served through the
  authenticated Dinkster gateway.
- The standard vision components for learned edge, line-art, depth, detection,
  segmentation, tracking, matting, and model-upscale operations.

This distribution contains the exact managed lock selecting those component
packs. It has no pack manifest and no importable node implementation.
`dinkster-serve` verifies each installed component against the lock and loads it
independently, so one can fail without making the others unavailable. Vision
components run in isolated environments; their model assets remain
digest-pinned, require acquisition consent, and download only during submission
preflight. `dinkster-serve` provisions each component's declared runtime
dependencies into a persistent content-addressed environment before the worker
announces, so node execution never installs software.

Existing workflows keep their canonical node IDs; only their pack provenance
changes. Other packs must compose through registered node, capability, and type
contracts rather than importing either implementation.
