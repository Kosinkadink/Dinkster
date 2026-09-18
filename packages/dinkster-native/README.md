# dinkster-native

`dinkster-native` owns Dinkster's native execution schemas, implementations, value
codecs, residency, and worker manifest. Its provider can be imported and
registered without importing ComfyUI or `dinkster-compat-comfy`.

The package is a uv workspace member. From the repository root, install the
whole workspace with:

```console
uv sync --all-packages
```

Native execution still requires the torch inference environment documented in
`packages/dinkster-inference-torch/README.md`. ComfyUI-backed translated nodes and
the remaining compatibility execution bodies live in `dinkster-compat-comfy`.

The worker manifest is generated from the compatibility manifest while
excluding ComfyUI-dependent execution claims. Regenerate it after changing the
source manifest or native catalog:

```console
uv run python tools/gen_native_comfy_manifests.py
```
