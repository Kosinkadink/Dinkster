# my-pack (Dinkster pack template)

A complete, doctor-clean Dinkster pack to copy and rename: a manifest, two
nodes, a pack-owned value type with a declared codec and rendition, an
icon, tests, and a CI workflow that gates publishing on `dinkster doctor`.

The full authoring reference is [docs/pack-authoring.md](../../docs/pack-authoring.md)
in the Dinkster repository.

## Layout

```
dinkster-pack.toml        # the manifest: identity, entries, presentation
my_pack_nodes.py       # nodes + register_types (the only import: dinkster_api.v1)
icon.png               # 64x64 static PNG badge, served by digest
pyproject.toml         # dev tooling only - runtime deps go in the manifest
tests/test_pack.py     # plain-value node tests + the doctor gate
.github/workflows/ci.yml
```

## Renaming

1. `[pack] name` in `dinkster-pack.toml` - your pack id. It prefixes node
   types (`my-pack.shout`), keys attribution and logging, and never
   changes once published.
2. The module name (`my_pack_nodes.py` and the `[pack.entry]` strings).
3. Every `node_type` and value type id (`my-pack.*` -> `your-pack.*`).
4. `[pack.presentation]` and `icon.png` - your badge, pixels only.

## Developing

Dinkster is unpublished, so develop against a checkout and run this pack's
gates through it (`--project` uses the checkout's environment without
touching your pack directory):

```bash
DINKSTER=path/to/Dinkster
uv run --project "$DINKSTER" pytest tests          # from this pack's root
uv run --project "$DINKSTER" dinkster-doctor .        # the publish gate
uv run --project "$DINKSTER" dinkster-serve --pack ./dinkster-pack.toml --watch-packs
```

`dinkster doctor` must come back healthy before you ship: it verifies the
manifest, the icon contract, that your code only imports `dinkster_api.v1`,
that requirements are pinned, that imports are quiet, and that edge-typed
values declare codecs. The CI workflow runs the same gate on every push.
