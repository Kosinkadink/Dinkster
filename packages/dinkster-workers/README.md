# dinkster-workers

Worker implementations, pack provisioning, doctor and sandbox launchers for
Dinkster hosts, without a dependency on the scheduler.

## Develop locally

From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- dinkster-doctor --help
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference

- [Worker exports](src/dinkster_workers/__init__.py)
- [Doctor, sandbox and memory-evidence operations](../../docs/worker-operations.md)
- [Remote workers](../../docs/remote-workers.md)
- [Pack dependency locks](PACK-LOCK.md)
- [Pack authoring](../../docs/pack-authoring.md)
- [Worker architecture](../../DESIGN.md#33-boundary-first-execution-engine)
