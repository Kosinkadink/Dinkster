# dinkster-native

`dinkster-native` owns Dinkster's retained native media, mesh/3D, and worker
integration surfaces. It does not contain a sampling engine or model-family
runtime.

The worker installs the pinned `dinkster-inference` fork and imports
`dinkster_inference` for supported SD1.5 and MiniMax H3 inference. ComfyUI-backed
legacy nodes remain in `dinkster-compat-comfy`.

From the repository root, install the workspace with:

```console
uv sync --all-packages
```
