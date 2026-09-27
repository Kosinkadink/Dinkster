# dinkster-compat-comfy

`dinkster-compat-comfy` is Dinkster's quarantined ComfyUI compatibility layer.
It translates legacy ComfyUI nodes behind the ordinary worker boundary and
keeps application-specific behavior out of the engine.

The worker installs the pinned `dinkster-comfy` fork and imports
`dinkster_comfy` for SD1.5 and MiniMax H3 inference. Other model-family
inference surfaces are unsupported.

The former native-loader and MiniMax alias declarations remain as
`comfy-aliases.inactive.json` evidence. They are not loaded into the serving
composition; equivalent mappings return only after their fork acceptance case
passes.

From the repository root, install the workspace with:

```console
uv sync --all-packages
```

Untranslated legacy nodes require a ComfyUI installation selected with
`--comfy-root`. The torch-free host process never imports the execution runtime.
