# Existing-store seed service

`dinkster-seed` runs the P2P controller and its private libtorrent sidecar without
starting the UI, editor, execution engine, or model workers. It shares only
safe-format files matching the explicitly trusted provider's complete export.
License is metadata, not seed authorization.

```sh
uv sync --all-packages --frozen
uv run --frozen dinkster-seed --store /srv/models --state-dir /var/lib/dinkster-seed \
  --provider-url https://provider.example/export.json --provider-id my-provider \
  --listen-port 6881 --upload-bytes-per-second 5242880
uv run --frozen dinkster-seed status --state-dir /var/lib/dinkster-seed
uv run --frozen dinkster-seed disable --state-dir /var/lib/dinkster-seed
uv run --frozen dinkster-seed enable --state-dir /var/lib/dinkster-seed
```

The provider ID must match the export name. Existing saved trust and unsubscribe
choices are preserved. No endpoint or provider identity is built in. Do not put
credentials in provider URLs. Use the system CA store for a private provider CA.

| Flag | Environment | Default |
| --- | --- | --- |
| `--provider-url` | `DINKSTER_SEED_PROVIDER_URL` | required |
| `--provider-id` | `DINKSTER_SEED_PROVIDER_ID` | required |
| repeated `--store` | `DINKSTER_SEED_STORES` (OS path separator) | required |
| `--state-dir` | `DINKSTER_SEED_STATE_DIR` | `~/.local/state/dinkster-seed` |
| `--listen-port` | `DINKSTER_SEED_LISTEN_PORT` | 0 (ephemeral) |
| `--status-port` | `DINKSTER_SEED_STATUS_PORT` | 0 (ephemeral) |
| `--upload-bytes-per-second` | `DINKSTER_SEED_UPLOAD_BYTES_PER_SECOND` | 5242880 |
| `--max-active-seeds` | `DINKSTER_SEED_MAX_ACTIVE_SEEDS` | 512 |
| `--refresh-seconds` | `DINKSTER_SEED_REFRESH_SECONDS` | 300 |
| `--backoff-seconds` | `DINKSTER_SEED_BACKOFF_SECONDS` | 5 |
| `--backoff-max-seconds` | `DINKSTER_SEED_BACKOFF_MAX_SECONDS` | 300 |
| `--network-cost` | `DINKSTER_SEED_NETWORK_COST` | auto |

Upload rate applies to both LAN and internet; zero means unlimited. Seeding is
continuous, with the existing metered-network pause policy. `disable` is one
durable toggle and stops seeding without restarting the host process.
On POSIX, disable acknowledges success only after syncing both the saved boolean
and its parent directory. `enable` also explicitly resumes current authorized
mapped seeds after a network-cost pause, including a pause retained on restart.
It does not resume removed artifacts or clear latches during ordinary startup
or refresh. If network policy still blocks recovery, it returns `resume-refused`
(HTTP 409); capability enablement remains saved, but the pause latch stays set.
Network cost is rechecked every five seconds. `auto` fails closed for internet
when the platform cannot detect cost (common in containers); explicitly choose
`unmetered` for an operator-managed unmetered server, or `metered` to pause
internet seeding. LAN remains separately available.

The bounded capacity admits native seed leases, not just catalog entries. A
file can occupy a LAN and a global slot; the default 512 slots covers 218 files
in both scopes. The range is 1-4096. Over-budget leases are refused, not silently
rotated; readiness stays false until every mapped file has an active seed.
Increase the budget for larger exports and measure memory, descriptors, disk
I/O, and upload throughput on the deployment host.

Nonzero listen ports are strict: TCP or UDP collisions close the listener,
never fall back to a different port. Reapplying an unchanged template retains
the binding and its TCP confirmation. Before reusing a port across changed
templates, the runtime closes the listeners, confirms the empty configuration
through a native settings-ordering barrier, and drains the queued alerts before
resetting listener tracking. It can then reopen the same port in the same
session without treating old bind alerts as confirmation of the new template.
The new binding requires its own TCP confirmation; existing torrent handles
and authorized leases are retained. The barrier orders settings application,
not every pending native callback. Isolated-link restoration remains a separate
limitation tracked in [#1342](https://github.com/Kosinkadink/Dinkster/issues/1342);
safe template reuse does not establish interface-topology recovery.

Control HTTP binds only to loopback. The selected port is recorded in
`<state-dir>/status-port`. `/health` reports process liveness; `/ready` returns
503 until every currently mapped artifact has a ready global native seed. `/status`
lists mapped and globally seeded digests, global peers, upload totals, provider refresh
timestamps/errors, and the existing controller's detailed status. An authorized
catalog row alone never counts as a seed. A local JSON boolean POST to `/enabled`
controls seeding; browser-origin writes are refused. Errors use stable categories
such as `provider-refresh-failed`, `seed-reconcile-failed`, `network-policy-failed`
and `state-write-failed`, not provider URLs or arbitrary exception text. Nested
sidecar errors and CLI output use the same sanitization boundary.

Store directories are never written, copied, or linked into the vault. Directory
symlinks, junctions, and symlink files are skipped. Matching filters by byte size
before hashing. Safe-format header validation and a single bounded BLAKE3/BEP52
scan share one open file, with path and fingerprint checks before and after.
Safe-format and digest verification and BitTorrent descriptor
material are persisted under the private state vault, bound to the absolute
path and the storage owner's six-field fingerprint. Unchanged mappings reuse
that verification on restart; changed/replaced files are reverified. Native
seeds use verified seed mode instead of a full startup check, with native
piece checks on upload. Keep state private to the service account: this is
local verification state, not an import format for provider data.

Refresh failures retry with exponential capped backoff; they do not extend
provider authority. The existing six-hour grant expiry still applies. Complete
refreshes tombstone omitted digests and retire removed or replaced LAN and global
authority before admitting new leases, even when replacement admission exceeds
capacity. State must live outside every store, and each service instance must
have its own state directory.

## systemd

Install the private checkout and its frozen environment under `/opt/dinkster`.
Create a `dinkster-seed` service account and set the real provider values below.

```ini
[Unit]
Description=Dinkster read-only model seeder
After=network-online.target
Wants=network-online.target

[Service]
User=dinkster-seed
Group=dinkster-seed
WorkingDirectory=/opt/dinkster
StateDirectory=dinkster-seed
Environment=DINKSTER_SEED_PROVIDER_URL=https://provider.example/export.json
Environment=DINKSTER_SEED_PROVIDER_ID=my-provider
Environment=DINKSTER_SEED_NETWORK_COST=unmetered
ExecStart=/opt/dinkster/.venv/bin/dinkster-seed --store /srv/models --state-dir /var/lib/dinkster-seed --listen-port 6881
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/srv/models
ReadWritePaths=/var/lib/dinkster-seed
PrivateTmp=true
UMask=0077

[Install]
WantedBy=multi-user.target
```

## Container recipe

Build only from the authorized private checkout. Do not publish the image to a
public registry. This recipe installs the pinned workspace without Torch.

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /opt/dinkster
COPY . .
RUN uv sync --all-packages --frozen --no-dev
RUN useradd --uid 10001 --create-home seeder && mkdir /state && chown seeder /state
USER seeder
ENV DINKSTER_SEED_STATE_DIR=/state
ENTRYPOINT ["/opt/dinkster/.venv/bin/dinkster-seed"]
CMD ["--store", "/models", "--listen-port", "6881"]
```

Exclude `.git`, `.venv*`, model files, secrets, and local state in the build
context's `.dockerignore`. For Linux hosts, host networking preserves LAN
discovery and loopback control. Mount models read-only and provision the state
directory for UID 10001:

```sh
docker run --rm --network host --read-only --tmpfs /tmp \
  --mount type=bind,src=/srv/models,dst=/models,readonly \
  --mount type=bind,src=/var/lib/dinkster-seed,dst=/state \
  -e DINKSTER_SEED_PROVIDER_URL=https://provider.example/export.json \
  -e DINKSTER_SEED_PROVIDER_ID=my-provider \
  -e DINKSTER_SEED_NETWORK_COST=unmetered dinkster-seed-private
```
