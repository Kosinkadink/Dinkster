# dinkster-native

`dinkster-native` owns Dinkster's retained native media, mesh/3D, and worker
integration surfaces. It does not contain a sampling engine or model-family
runtime.

The package depends on an exact `dinkster-inference` PyPI release and imports
`dinkster_inference` for supported SD1.5 and MiniMax H3 inference. ComfyUI-backed
legacy nodes remain in `dinkster-compat-comfy`.

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
