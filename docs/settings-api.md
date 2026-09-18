# Runtime settings API

Dinkster exposes its effective runtime controls through an always-registered
read endpoint and category-gated mutation endpoints. The gate is enforced
by the server; hiding a frontend menu is not a security boundary.

## Permissions and discovery

`GET /api/settings` is always available. Its response has this shape:

```json
{
  "categories": {
    "granted": ["jobs"],
    "available": [
      "memory-budgets",
      "memory-headroom",
      "aimdo-policy",
      "dtype-policy",
      "fp8-matmul",
      "worker-comfy-args",
      "jobs",
      "logging",
      "p2p"
    ]
  },
  "settings": {
    "jobs": {
      "value": {"maxRunningJobs": 2},
      "source": "runtime",
      "mutability": "live",
      "writable": true,
      "persistence": {"available": true, "persisted": true}
    }
  }
}
```

The actual response contains all nine sections. `source` is one of
`cli`, `persisted`, `config`, `default`, or `runtime`. `writable` is true
only when the category was granted with `--allow-settings-changes`.
`persistence.available` is false when `--library-root ''` disabled the
persistence root. A successful in-memory edit then visibly reports
`persisted: false`.

## Categories and request bodies

Use `PUT /api/settings/{category}`. A successful response is the updated
section in the same shape used by GET.

| Category | Request JSON | Mutability |
| --- | --- | --- |
| `memory-budgets` | Object mapping device names to integer bytes or K/M/G/T strings, for example `{"ram":"24G"}` | live admission; residency on worker restart |
| `memory-headroom` | Integer bytes or a K/M/G/T string, for example `"256M"` | live |
| `aimdo-policy` | `"auto"`, `"on"`, or `"off"`; Aimdo provides partial weight offload on supported NVIDIA Linux and Windows workers and otherwise explicitly falls back to eager residency | on-worker-restart |
| `dtype-policy` | `{"diffusion":"auto","textEncoder":"float16","vae":"float32"}`; each value is `"auto"`, `"float16"`, `"bfloat16"`, or `"float32"` | on-worker-restart |
| `fp8-matmul` | Boolean | on-worker-restart |
| `worker-comfy-args` | JSON array of strings, for example `["--preview-size","768"]` | on-worker-restart |
| `jobs` | `{"maxRunningJobs":2}` where the value is an integer >= 1 | live |
| `logging` | `{"level":"info","overrides":{"dinkster.pack.a":"debug"}}` | live |
| `p2p` | Closed P2P policy object described below | live |

The `p2p` object contains all of these fields. The shown values are the exact
defaults:

```json
{
  "downloadsEnabled": true,
  "seedingEnabled": true,
  "scope": "lan-and-internet",
  "internetUploadBytesPerSecond": 5242880,
  "internetDownloadBytesPerSecond": 0,
  "lanUploadBytesPerSecond": 0,
  "lanDownloadBytesPerSecond": 0,
  "pauseOnMetered": true,
  "networkCostOverride": "auto",
  "seedMode": "budgeted",
  "internetSeedRatio": 1.0,
  "internetSeedTimeSeconds": 86400,
  "stagingBudgetBytes": 68719476736
}
```

Saved enable and scope choices are preserved. `--disable-p2p` overrides both
enable fields at startup without changing the saved settings or write grants.
The staging budget is a non-negative safe integer (at most 9007199254740991).
It limits aggregate LAN/global partial allocation plus reserved missing bytes;
zero refuses new growth, while no-copy seed files do not count. Lowering it
pauses downloads that no longer fit without deleting data; increasing it does
not resume them. This is admission control, not a filesystem quota: in-flight
writes, allocation granularity, and resume metadata may exceed it before the
next reconciliation pauses growth.

`scope` is `lan-only` or `lan-and-internet`; `networkCostOverride` is `auto`,
`metered`, or `unmetered`; and `seedMode` is `budgeted` or `continuous`. Rates
and seed time are integer values from 0 through 2147483647. The seed ratio is
a finite non-negative number. No sidecar, IPC socket, or peer port is created
while both enable fields are false. When a library root exists, enabling either
field starts its vault's isolated sidecar. Metered network detection closes
global internet features when `pauseOnMetered` is true; unknown network cost
also fails closed. LAN discovery and mappings remain active. An explicit cost
override takes precedence. `GET /api/p2p/status` reports
lifecycle, LAN interfaces and mappings, network state and features, leases,
recovery state, durable internet-attributed download/upload totals, and
per-digest activity, partial bytes, remaining seed budgets, and canonical seed
grants. Shared-handle accounting excludes observed LAN traffic and charges
unattributable disconnect intervals to the internet budget. The response
preserves the host envelope fields `state`, `settings`, `restartCount`,
`lastError`, and `sidecar`; detected network cost is the top-level `network`
field, and the top-level `lan` field reports whether LAN networking is allowed,
the mapping port, and mapped digests. Totals and transfers extend the running
`sidecar` object. Each seed authorization reports whether it is active,
inactive under current policy, or revoked; active and inactive rows include the
canonical grant, including its acquisition-receipt evidence when applicable.
`network.paused` and `sidecar.networkPaused` describe an all-P2P stop, so they
are false during a global-only metered or unknown-cost closure. The reason for
that closure appears in `sidecar.global.closureReason`.

LAN mode discovers digest-mapping endpoints through mDNS. Mapping version 2
advertises the active libtorrent TCP port for the receiving LAN interface.
The controller carries that endpoint in a version 2 download lease, and the
sidecar revalidates the address against its current LAN policy before connecting
immediately. The exact version 1 mapping and download-lease shapes remain
supported. A version 2 mapping client retries the version 1 route only when the
version 2 route returns 404 or 405, within the original request deadline. LSD
remains enabled as fallback peer discovery. LAN mode binds peer and mapping
listeners only to eligible RFC 1918 interfaces and never enables DHT, trackers,
PEX, web seeds, uTP, UPnP, or NAT-PMP. Per-digest POST actions require the host's
`p2p` settings grant, return 204, and are available at
`/api/p2p/transfers/{digest}/{action}` for `pause`, `resume`, `stop`,
`remove-partial`, `reset-budget`, and `continuous-seed`.
Policy-paused transfers require an explicit `resume`; budget reset and
continuous mode do not resume a transfer themselves.

Seeding requires an active trusted P2P descriptor, verified local safetensors
or GGUF bytes, and source evidence. License and gated fields are metadata, not
transfer conditions. Resolver subscriptions are HTTP-only unless `trustedForP2P`
is enabled through this category's permission gate; `licenseAuthoritative`
identifies metadata authority only. Both flags default false. A complete resolver
refresh supplies provider enumeration evidence; omission revokes it, while a
failed or partial refresh does not renew it. The LAN mapping service answers
only an exact digest request and exposes no digest list.

`lan-and-internet` additionally permits internet features on the shared P2P
session only while a current trusted grant exists. Returning to `lan-only`,
revocation, expiry, or budget exhaustion removes global handles and closes DHT,
PEX, TCP, uTP, approved trackers, and NAT traversal without replacing the LAN
session. Metered or unknown-cost policy closes only those internet features and
requires explicit per-digest resume after the closure clears. An explicit
unmetered override permits the internet features. Resolver-index provider
snapshots use trackerless DHT and supply no trackers in version 1.

Memory-budget updates merge the supplied devices into the effective map.
`MemoryGovernor.set_budget` only replaces the declared limit. Lowering a
budget does not shed existing reservations or consumer footprints. It may
make reported availability negative, and constrains future admission; use
the separate shedding surface when bytes must be reclaimed. CUDA compat
workers apply the same budget to classic or AIMDO residency when they start or
reload. `GET /memory/status` exposes the live admission value, each active
worker's applied residency value, and a staleness flag while they differ.

Raising `jobs.maxRunningJobs` wakes queue dispatch so waiting jobs can be
admitted immediately. Logging changes reconfigure this process and update
`DINKSTER_LOG_LEVEL` / `DINKSTER_LOG` for workers spawned later.

`memory-headroom` updates every running compat policy worker immediately.
AIMDO workers preserve their current governor-reservation extra; classic
workers apply only the physical base. Workers spawned or reloaded later
receive the same effective base through their validated argv. `aimdo-policy`
remains pending worker policy: a dev pack reload
(`POST /api/packs/{packId}/reload`) starts the replacement compat worker
with the current effective policy, but does not alter an already running
worker. `worker-comfy-args` follows the same reload seam: values persist
immediately, and a replacement compat worker receives the complete array
in order through explicit child argv. Values never travel through inherited
environment variables.

`dtype-policy` selects diffusion, text-encoder, and VAE compute dtypes
independently. Auto preserves each native family's safe default. Explicit
values are carried in native runtime identity and translated to ComfyUI's
equivalent worker flags for compatibility loaders. Storage that differs from
the selected compute dtype uses cast-at-operation loading; capability policy
selects the next supported compute dtype when stored weights lack native
compute support.

`fp8-matmul` is a global opt-in for native checkpoint loading and defaults
to false. It is folded unconditionally into every native checkpoint cache
identity, so changing it rotates cache tags even when a checkpoint contains
no fp8 weights. The server records the requested policy because its
torch-free environment cannot resolve device support. The selected worker
checks its actual device, compute capability, and torch version before load;
an unsupported request fails loudly and never falls back to false. Planned
e5m2 checkpoint storage is refused during the server's header-only probe.

ComfyUI remains the authority for all passed-through arguments at worker
startup. Dinkster rejects controls it owns before persistence or spawn. The
structured 400 response names `offendingFlag` and `owner`. The deny-list is:

- Dinkster server: `--listen`, `--port`, `--tls-keyfile`, `--tls-certfile`,
  `--enable-cors-header`, `--max-upload-size`, `--auto-launch`,
  `--disable-auto-launch`, `--front-end-version`, `--front-end-root`,
  `--enable-compress-response-body`, and `--multi-user`;
- Dinkster memory/aimdo policy: `--reserve-vram`, `--gpu-only`, `--highvram`,
  `--normalvram`, `--lowvram`, `--novram`, and `--cpu`.
  `--vram-headroom`, `--enable-dynamic-vram`, and
  `--disable-dynamic-vram` are denied for the same owner.
- Dinkster dtype policy: the fp16, bf16, and fp32 UNet, text-encoder, and VAE
  flags.

Bare flags, `--flag=value` forms, and argparse abbreviations of denied flags
are rejected. Unknown arguments
pass validation and are accepted or rejected by the installed ComfyUI
parser when the worker starts.

## Persistence and precedence

Persistence is on whenever a library root exists. Every successful edit
rewrites all runtime-set values to `<library-root>/settings.json` using a
temporary file followed by `os.replace`. `memory.toml` remains a read-only,
hand-edited configuration layer and is never rewritten by this API.

The effective precedence is:

1. an explicitly supplied managed CLI flag;
2. persisted `settings.json`;
3. `memory.toml` or another configuration layer;
4. the built-in default.

Explicit CLI detection uses parser sentinels, so explicitly passing a value
equal to the built-in default still wins over persistence. A mutation in
the current process reports source `runtime`.

The persisted file uses category names directly:

```json
{
  "aimdo-policy": "off",
  "dtype-policy": {"diffusion":"auto","textEncoder":"float16","vae":"float32"},
  "fp8-matmul": true,
  "worker-comfy-args": ["--preview-size", "768"],
  "jobs": {"maxRunningJobs": 2},
  "memory-headroom": 268435456
}
```

## Errors

An ungranted category returns 403:

```json
{
  "error": "settings-changes-disabled",
  "category": "jobs",
  "granted": ["logging"]
}
```

Malformed JSON, an unknown category, or an invalid value returns 400:

```json
{
  "error": "invalid-settings",
  "category": "jobs",
  "message": "jobs.maxRunningJobs must be an integer >= 1"
}
```
