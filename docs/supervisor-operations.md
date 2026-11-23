# Supervisor and environment operations

Use the [installation guide](install.md) for a published environment and the
[environment guide](environments-and-updates.md) for layout and update concepts.
The environment CLI's `--help` and its subcommand help are the option reference.

## Ports and station configuration

Dinkster uses public port 3639 and station management port 3649 by default.
Keep Dinkster allocations outside the ComfyUI/comfy-runner port neighborhood
(8188/8189/8199) to avoid collisions with those applications.

To run a supervised engine from the contributor runtime:

```sh
python scripts/develop.py run -- dinkster-supervisor -- dinkster-serve
```

The station reads the installations file managed by `dinkster-installs`.
An optional fleet-facing listener can be configured beside its install tables:

```toml
[ingress]
port = 8300
members = ["main", "experiments"]
primary = "main"
state_path = "/var/lib/dinkster/ingress.sqlite3"
```

Keep the ingress ownership database across restarts. Do not treat its assignments
as expiring cache entries. Execution uses the primary; do not assume fleet
spreading without fleet-homogeneity and shared-library proofs.

## Security decisions and recovery

Provision issuer trust independently in the environment's `security.sqlite3`.
A mirror choice in `security.toml` does not establish trust in that issuer:

```toml
refuse_vulnerable = false
request_timeout_seconds = 5.0
refresh_interval_seconds = 60.0
# maximum_feed_age_seconds = 86400

[feed_urls]
"https://registry.example.org" = "https://mirror.example.org/security/index.json"
```

Inspect notices with `dinkster-env --root ROOT security --json list` or
`security --json details --pack PACK`. Use `--snapshot N` to inspect retained
software. Offline inventory is unavailable, not evidence that no jobs or workers
exist. Runtime inventory identifies submitted work, not incident impact. A policy
allowance or lifted notice is not proof that a revoked worker has resumed.
Review a compatible repair plan before treating recovery as complete.

For malicious or affected incidents, blocking is not cleanup. Isolate affected
systems, rotate exposed credentials from a trusted machine, assess exposure and
rebuild from trusted inputs. Do not resume work merely because a notice or local
policy changed. See [channel signing](channel-signing.md) for the separate channel
key authority, provisioning, rotation and compromise response.

## Review before changing an environment

Use `--plan` before confirming a mutation. Review the complete source and
per-variant dependency locks, in-process placement, software substitutions and
project scope, not just the displayed version. Exact restore and approximate
restore are different approvals; `--yes` alone does not approve substitutions.

Before activating a project, recheck its environment candidate: a successful
`project-check` is not a reservation of the library. A project-bound `packs-plan`
requires a reviewed `--scope every-project` or a planned destination for
`--scope only-project`. If construction succeeds but project selection fails,
read back the project and the reported stopped destination before retrying.

When adopting a standalone installation, select engine/base identities from the
intended channel and supply each target-variant pack lock. Ambient dependencies
are not an exact freeze; a distribution missing from a pack lock can still belong
to the target base. Do not infer the old shared store's accelerator selection.

Retain host-owned progress scratch until the command exits, even after requesting
cancellation: an uninterruptible child may still be using it. Progress is local
to a stage, not an overall percentage or timer heartbeat. Likewise, custom pack
lock scratch must survive provisioning and be removed afterward by its owner.

For Python integrations, hold the store writer lock while acquiring/extracting
inputs or preparing an interpreter. A caller-owned prepared interpreter directory
also needs caller-owned cleanup on failure; preparation is not snapshot sealing.
Acquire the store lock before the environment lock when both are needed. Never
remove persistent lock files to bypass a live writer. Treat an ambiguous process
identity as unresolved, not as permission to stop an unrelated process or collect
its software.

Do not edit installed hardlinked files: that can corrupt shared extracted cache
entries too. Use `doctor --repair` with verified retained inputs. Use the managed
move command rather than copying an active installation; retain source data until
destination doctor and engine health succeed. Readable cache directory labels are
not disposable-object identities, and missing references do not prove abandonment.

## Launch policies and storage

Stop all environment engines before changing launch selections or storage policy,
then restart. Include every required pack and capability provider explicitly in a
filtered launch. A normal `start` is not `start --selected-packs`; use the latter
only when the saved launch filter is intended.

For browser-offline runtime, run `network offline --yes` and restart, then reload
open product tabs: a loaded online page does not acquire the new policy
retroactively. This is not an OS firewall and does not constrain unrelated host
or pack networking, navigation outside the product, or other applications.
Setup/update `--offline` instead selects cached installation inputs. Development
live reload must use the public product proxy, not a separate dev-server origin.

Compression is opt-in. Do not enable `compression on --yes` where an attacker can
combine reflected input with secrets in compressed responses and measure sizes.

Select existing absolute storage parents through `previews --directory PATH` or
`scratch --directory PATH`, inspecting `--plan` before `--yes`. Keep previous
preview locations attached. Do not move or remove an assigned preview vault by
hand; select a new location through the command so old references remain readable.

Choose a short scratch parent on Windows: the full assigned worker path, including
the environment prefix and random child name, must fit 259 UTF-16 code units.
Longer paths can deny contained writes or hang native GetTempPath APIs.
The selected filesystem determines the medium; choose a RAM filesystem for RAM.
Ambient TMPDIR does not select managed scratch. Unsandboxed workers are not a
security boundary. Clearing or replacing scratch policy does not remove old
scratch; recover pending produced-output handoffs by reselecting their original
storage rather than deleting it.

## Reference

- [Supervisor protocol and proxy](../packages/dinkster-supervisor/src/dinkster_supervisor/supervisor.py), [station](../packages/dinkster-supervisor/src/dinkster_supervisor/station.py) and [ingress](../packages/dinkster-supervisor/src/dinkster_supervisor/ingress.py)
- [Environment commands](../packages/dinkster-supervisor/src/dinkster_supervisor/environment.py) and [contract tests](../tests/test_environment_cli.py)
- [Native tools build and bootstrap](environment-tools.md)
