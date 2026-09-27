# dinkster-nodes-generation

First-party graph schemas for supported SD1.5 generation and retained mesh/3D
operations. Execution comes from `dinkster-native`: generation delegates to the
pinned `dinkster_comfy` package, while mesh operations remain Dinkster-owned.

The retired generation alias and group declarations remain beside the manifest
as `comfy-aliases.inactive.json` and `comfy-groups.inactive.json`. They are
evidence only and are not loaded into the serving composition.
