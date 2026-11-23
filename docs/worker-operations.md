# Worker operations

## Checking and refreshing packs

Run doctor against the directory containing `dinkster-pack.toml` before
publishing. Use the human report for concrete fixes and `--json` in tooling:

```sh
python scripts/develop.py run -- dinkster-doctor path/to/pack
python scripts/develop.py run -- dinkster-doctor --json path/to/pack
python scripts/develop.py run -- dinkster-doctor --sandbox path/to/pack
```

The import probe executes pack code. Use a trusted pack or a supported probe
sandbox; an ordinary isolated interpreter is not a security boundary.
Catalog preparation validates runtime declarations, not the complete publisher
checks. Refresh catalogs after changing pack code or runtime dependencies:

```sh
python scripts/develop.py run -- dinkster-pack prepare-catalogs --defaults
```

Use `dinkster-doctor --part NAME` when checking a named execution part.
See [pack dependency locks](../packages/dinkster-workers/PACK-LOCK.md),
[pack authoring](pack-authoring.md) and [remote worker setup](remote-workers.md).

## Serving sandbox

Dependency and crash isolation are not host security. Enable the separate
serving sandbox with `dinkster-serve --sandbox-packs` and check the host with
`dinkster-doctor --sandbox-readiness --json`.

Linux serving requires full bubblewrap user, PID, mount and network namespaces;
a mount-only degraded capability is insufficient. macOS uses Seatbelt.
Windows 11 uses the native MXC ProcessContainer when the OS API and trusted
executor are available. Windows 10 requires WSL2 for Linux isolation.
The doctor import sandbox is available on Linux and macOS, not Windows.
A requested unavailable sandbox must not be replaced by an unjailed launch.
Serving isolation does not cover in-process packs or remote worker daemons.

Command-line Windows hosts must provision the same source-built `wxc-exec.exe`
that Desktop bundles at `resources/sandbox/wxc-exec.exe`. Set the host-owned
`DINKSTER_MXC_EXECUTABLE` to its absolute path. Supply the adjacent
`mxc-build.json` for the final signed executable, using the format, source pin,
empty telemetry component and SHA-256 validated by
[the Windows launcher](../packages/dinkster-workers/src/dinkster_workers/windows_sandbox.py).
These are trusted host installation inputs, never pack grants; do not substitute
a PATH lookup or a downloaded release binary.

If a Unix worker refuses an overlong socket path, set `TMPDIR` to a shorter
real directory before starting workers. Do not introduce a TCP fallback that
relaxes the jail's private-network policy.

Choose grants and worker groups as trust boundaries. Every jailed pack can read
the configured asset vault; shared-memory workers can see host `/dev/shm`.
Group members share scratch and the union of member grants. Writable mounts
expose their contents and GPU grants expose the host driver. The HTTPS CONNECT
proxy checks the destination, not tunneled bytes or the destination's
trustworthiness. Windows retains baseline system/GPU access and DNS resolution;
it does not apply Linux process-count or file-size limits. This is not a VM:
kernel/driver exploits, malicious outputs and RAM/VRAM exhaustion remain outside
the boundary.

The platform launchers are the reference for enforced grants and transports:
[Linux](../packages/dinkster-workers/src/dinkster_workers/sandbox.py),
[macOS](../packages/dinkster-workers/src/dinkster_workers/seatbelt.py),
[Windows](../packages/dinkster-workers/src/dinkster_workers/windows_sandbox.py).

## Interpreting memory evidence

A valid benchmark report is not proof of driver-spill avoidance. Shared-usage
samples and warm-to-final differences are observations: pinned staging and
driver spill both contribute, including when usage is stable. Attribute them
with allocation-owner lifetimes, process RAM, device budgets and timings, not a
process-wide or machine-wide growth threshold. Missing or invalid Windows
counter samples mean unknown, not zero. Process RAM means physically resident
memory; virtual mappings are not equivalent to resident pages.

Keep historical reports with their original source and validator rather than
rewriting them. Historical boolean spill flags are not spill proof. The
standalone `dinkster-residency-probe/4` timing-outlier flag is not a spill verdict
either. Canonical null-assessment requirements live in
[benchmark validation](../packages/dinkster-workers/src/dinkster_workers/backend_env.py).
