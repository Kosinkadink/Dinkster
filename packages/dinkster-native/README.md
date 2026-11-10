# dinkster-native

`dinkster-native` supplies resident-pool and memory planning helpers, device
telemetry, worker attention integration, and legacy checkpoint conversion.
It does not contain a sampling engine or model-family runtime. First-party
packs own media and 3D execution.

The unused modules are preserved on the permanent branch
`archive/dinkster-native-unused-modules` at commit
`602f8bf4936379bc366afe225554cedc0b9e75bf`.

The package depends on an exact `dinkster-inference` PyPI release and imports
`dinkster_inference` for supported SD1.5 and MiniMax H3 inference. First-party
node packs own node registration; this package supplies shared runtime helpers.

To update inference, the sync owner first validates and releases a landed
inference commit with a new version and matching `vX.Y.Z` tag. After the
owner-approved PyPI publication, update this package's exact requirement and
every pack manifest's inference requirement together, run `uv lock`, and
validate the resulting installed release with `tools/verify_pinned_inference.py`.
The lock records the release's distribution hashes. Setup scripts install
this dependency through package metadata, not a separate archive or commit pin.

From the repository root, install the workspace with:

```console
uv sync --all-packages
```
