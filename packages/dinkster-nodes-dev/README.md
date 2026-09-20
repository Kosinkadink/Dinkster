# dinkster-nodes-dev

`dinkster-nodes-dev` is test and demo scaffolding, not a user-facing node pack. It
provides observable scheduling and lightweight numpy image operations so the
engine, value boundary, renditions, lists, and mounted write gate can be
exercised without torch or Pillow.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```console
uv sync --all-packages
```

The package is not published separately yet. Its workspace dependency includes
numpy in addition to `dinkster-api`.

## Use

`DEV_NODES` contains the dev node classes and `register_dev_types` registers
the `dev.image` numpy value type (codec, fingerprint, metadata, renditions)
plus the gallery's marker and shared asset/save-target types.
`Delay` is an async passthrough used to make concurrent scheduling visible:

```python
import asyncio

from dinkster_nodes_dev import Delay

result = asyncio.run(Delay.execute(value="ready", seconds=0.01))
assert result == {"value": "ready"}
```

`dinkster-serve` imports and composes this pack into its in-process core only when
started with `--dev`. Without that flag, `dev.*` nodes and `dev.image` are
absent from the user node surface.

## The widget/socket gallery

`gallery.py` ships `dev.gallery.*` nodes that collectively exercise every
construct the native schema wire can express - primitive widgets and
defaults, static/remote/empty-remote combos, bare and labeled booleans,
asset and save-target widgets, required/optional inputs, optional outputs,
unions, wildcards, lists (optional, of-union, nested), and match
templates - so the frontend can verify rendering and decode against a real
server. It is a coverage surface, not semantics: nodes are passthroughs or
trivial producers (`dev.gallery.exotic_out` is render-only by design; its
union/wildcard outputs cannot be wrapped at execution).

Under `--dev` the composition also serves:

- `/api/choices/dev.gallery.samplers` (populated) and
  `/api/choices/dev.gallery.empty` (legally empty) for the remote combos,
  enumerated from `combo_choices`;
- the `dev-gallery` template (`gallery_template.json`, declared in
  `dinkster-pack.toml` and shipped through the core packs table), a starter
  document instantiating every gallery node with connected and
  unconnected examples of each socket variant.

The gallery module's docstring lists the constructs the native wire cannot
express yet (COLOR, union-of-lists); those are tracked in ROADMAP.md, never
invented as wire fields here. Numeric bounds, seed controllers
(controlAfterGenerate), multiline hints, and per-input display names are
expressible by the current schema and are exercised by the gallery nodes.
`tests/test_gallery.py` pins the coverage.

## Learn more

- [DESIGN.md section 3.2](../../DESIGN.md#32-value-envelopes-nothing-crosses-an-edge-raw)
  describes value codecs and metadata.
- [DESIGN.md section 3.10](../../DESIGN.md#310-parallel-execution-and-memory-governance)
  describes the scheduling behavior made visible by `Delay`.
- [Pack authoring](../../docs/pack-authoring.md) covers node and type entry
  points.
- `tests/test_compose.py`, `tests/test_isolated.py`, and
  `tests/test_renditions.py` exercise this package.
