# dinkster-nodes-dev

Workspace-only node pack for test/demo scaffolding, observable scheduling,
lightweight NumPy images and the widget/socket gallery. It is not a default or
published pack.

## Develop locally

From the repository root, prepare the contributor runtime and run focused tests:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_compose.py tests/test_isolated.py tests/test_renditions.py
```

To serve the pack explicitly and try its starter templates, follow
[the CPU-only feature guide](../../docs/dev-node-features.md).
See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference

- [Pack manifest](dinkster-pack.toml) and [public entry points](src/dinkster_nodes_dev/__init__.py)
- [Gallery schemas and coverage limits](src/dinkster_nodes_dev/gallery.py)
- [Scheduling example](src/dinkster_nodes_dev/util.py) and [image operations](src/dinkster_nodes_dev/image.py)
- [Pack authoring](../../docs/pack-authoring.md) and [architecture](../../DESIGN.md)
