# Dinkster supervisor

Engine-independent process supervision, environment installation and station
management. The supervisor owns the public port and launches engine subprocesses;
the station manages multiple configured installations.

## Develop locally

From the repository root, use the contributor runtime:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- dinkster-supervisor --help
python scripts/develop.py run -- dinkster-env --help
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_supervisor.py tests/test_station.py tests/test_fleet_ingress.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Supervisor protocol](src/dinkster_supervisor/supervisor.py), [station](src/dinkster_supervisor/station.py) and [install configuration](src/dinkster_supervisor/installs.py)
- [Environment commands](src/dinkster_supervisor/environment.py) and [contract tests](../../tests/test_environment_cli.py)
- [Environment lifecycle](../../docs/environments-and-updates.md) and [operating notes](../../docs/supervisor-operations.md)
- [Native tools build/bootstrap](../../docs/environment-tools.md) and [channel signing](../../docs/channel-signing.md)
