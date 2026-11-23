# Dinkster registry client

Pack artifact, registry admission and installation models used by the engine's
registry client. The registry service is a separate project.

## Develop locally

From the repository root, use the contributor runtime:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_artifact.py tests/test_install.py tests/test_installer.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public exports](src/dinkster_registry/__init__.py), [archive admission](src/dinkster_registry/archive_admission.py) and [release models](src/dinkster_registry/model.py)
- [Registry operating notes](../../docs/registry-operations.md) and [pack submission](../../docs/registry-submission.md)
- [Installation](../../docs/install.md) and [environment updates](../../docs/environments-and-updates.md)
- [Registry service](https://github.com/Kosinkadink/dinkster-registry-service) and [official pack sources](https://github.com/Kosinkadink/dinkster-nodes)
