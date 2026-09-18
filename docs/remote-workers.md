# Remote workers

A `dinkster_workers.service` daemon serves one pack over authenticated TCP;
`dinkster-serve` composes named connections to such daemons at startup, so
the daemon's node types join the serving surface and execute on its
machine - indistinguishable to the engine from a local isolated pack.

## Daemon side

On the machine that should execute the nodes:

```
python -m dinkster_workers.service --listen 0.0.0.0:5151 \
    --manifest path/to/dinkster-pack.toml --token-file /etc/dinkster/box.token \
    --asset-vault /var/lib/dinkster/vault \
    --value-store /var/lib/dinkster/value-store --value-store-budget 20G
```

`--listen HOST:PORT` (port `0` binds an ephemeral port, announced on
stdout), `--manifest` names the pack to load, and the pre-shared token
comes from `--token-file` or `$DINKSTER_REMOTE_TOKEN`. `--asset-vault DIR`
and `--asset-root DIR` configure the asset store chain (see "Declared
assets on a daemon" below). `--value-store DIR` gives the daemon a
persistent value store so bulk boundary values cross the network once
(see "Persistent value transport" below); `--value-store-budget SIZE`
caps it (default `10G`), evicting least recently used blobs beyond the
cap. The daemon outlives
its clients: an engine disconnecting ends that conversation, never the
process, and pack state (loaded consumers, resident models) survives
across reconnects. One conversation at a time, held by a TTL lease
(`--lease-ttl SECONDS`, default 45; see "Session leases and liveness"
below).

The token authenticates; it does not encrypt. To encrypt, give the
daemon a certificate and key with `--tls-cert cert.pem --tls-key
key.pem` and pin the certificate on the engine side with `tls_ca_file`
(below) - or deploy on a trusted network or behind an authenticated
tunnel (WireGuard, SSH, mTLS proxy). A self-signed certificate works;
its subjectAltName must name the endpoint the engine dials:

```
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
    -days 3650 -subj "/CN=upscale-box" \
    -addext "subjectAltName=IP:192.168.1.53,DNS:upscale-box"
```

## Engine side

Point `dinkster-serve` at a `remotes.toml` with `--remote-workers PATH`, or
place it at `<library-root>/remotes.toml`:

```toml
[worker.upscale-box]
endpoint = "192.168.1.53:5151"
token_file = "/home/op/.config/dinkster/upscale-box.token"
# tls_ca_file = "/home/op/.config/dinkster/upscale-box.pem"

[worker.upscale-box.memory]
ram = "24G"
"vram:cuda:0" = "20G"
```

`tls_ca_file` pins the daemon's `--tls-cert` certificate (or the CA that
signed it) and turns the connection into server-authenticating TLS; the
token still authenticates the engine, inside the encrypted stream. Set
it iff the daemon serves TLS - there is no unverified-TLS mode, and a
plaintext/TLS mismatch in either direction fails the handshake with an
error naming the mismatch. One mismatch is not free: an engine
configured for plaintext sends its token as its first bytes, so dialing
a TLS daemon without `tls_ca_file` exposes that attempt's token on the
wire - rotate the token if the endpoint was untrusted.

Remotes compose after the packs, one at a time, with pack failure
semantics: a dead or misconfigured daemon is recorded on
`/api/composition` and everything else serves (fatal under
`--strict-packs`). The composed remote appears in the pack table with
`source = "remote:HOST:PORT"` and no version pin, and its node types
attribute to the worker's name on `/api/nodes`.

Protocol 8 service hellos may include `visionProviders` and
`generationProviders`. Each strict declaration identifies the served node and
choice route; vision declarations also carry semantic model, device, dtype,
batching, and declared-asset facts, while generation declarations may carry a
human-facing service label. A present empty list is authoritative evidence
that the daemon provides none. An omitted field from an older daemon is
unknown, not unsupported. The engine can conservatively try that daemon only
when its exact compatible pack surface serves the node, and the resulting
events and receipts name the actual worker, pack, and provider.

## Worker discovery and placement

`GET /api/workers` reports the execution locations known to the server. The
response includes the implicit `local` worker and each configured
`[worker.NAME]` entry:

```json
{
  "workers": [
    {
      "name": "local",
      "status": "connected",
      "routedNodeTypes": ["std.math.add"],
      "deviceQualifiers": []
    },
    {
      "name": "upscale-box",
      "status": "connected",
      "routedNodeTypes": ["esrgan.upscale"],
      "deviceQualifiers": ["@upscale-box"]
    }
  ]
}
```

Status is the connection state already known from composition. The request
does not probe remote endpoints. With server authentication enabled, this is
an authenticated catalog route and requires no additional capability.

Jobs can carry an optional submission sidecar that selects a worker by its
configured name:

```json
{
  "clientId": "browser-1",
  "jobId": "run-42",
  "graph": {},
  "targets": ["save"],
  "placement": {
    "load-model": "local",
    "upscale": "upscale-box"
  }
}
```

Keys are top-level graph node ids. Body node ids and lowered runtime ids are
not accepted. A hint on a region applies recursively to every node in the
region body. The server validates every hint before queueing and returns 400
for an unknown worker, an unknown or non-top-level node id, or a worker that
does not serve the node type. Literal selector nodes and nodes pruned from an
inactive selector branch cannot be hinted. Remote placement of lazy nodes is
also rejected because lazy dispatch is owner-bound. There is no fallback to
local execution.

An explicit hint takes precedence over the configured node-type route and
the placement policy. It cannot move an invocation whose inputs are resident
on another worker; that run fails rather than transferring or reloading the
resident value silently. Node events and job node receipts include `worker`
so run reports identify the execution location.

Placement is not part of the graph document or node input fingerprints. The
selected dispatch arm remains part of cache identity: a hinted and unhinted
run selecting the same arm can reuse an entry, but local and remote arms do
not share entries merely because their schemas and unversioned tags match.
Remote daemons do not advertise an authoritative cross-worker
implementation identity. Placement is part of the active job idempotency
request, so reusing one `(clientId, jobId)` with different placement returns
the same 409 content conflict as changing the graph or targets.

## The @name budget convention

Every device fact a remote reports is qualified into its own namespace:
`upscale-box`'s `vram:cuda:0` becomes `vram:cuda:0@upscale-box`, so two
machines' GPUs never share an accounting row. `[worker.NAME.memory]`
budgets are declared in the remote's own unqualified keys and qualified
by the host before the governor sees them; explicit `--memory-budget`
entries (which must use the qualified form) win on collision.

## Declared assets on a daemon

A daemon-hosted pack resolves `[[pack.assets]]` declarations through the
same environment-assembled store chain a local isolated worker gets:
`--asset-vault DIR` names its verified vault (`$DINKSTER_ASSET_VAULT`,
created on first write), `--asset-root DIR` an operator-managed
read-only library (`$DINKSTER_ASSET_ROOT`), and an exported
`$DINKSTER_MOUNTS_SNAPSHOT` is honored as-is. Execution-time reads never
download: a declared asset must already resolve when the node runs.

Staging closes that gap before dispatch. The pack's declarations ride
the connection handshake (the engine never sees the daemon's manifest),
so before dispatching a node type they name, the engine asks the daemon
which digests its store already resolves and, for the missing ones,
offers candidate HTTP(S) sources: the engine's advertised asset
endpoint (`dinkster-serve --advertise-assets URL`, with a bearer token from
`--advertise-assets-token-file` when the engine serves with `--auth` -
an `auth.toml` token granting `assets:read`), then any remote URLs the
declaration itself carries. The daemon pulls the bytes into its vault
with streamed digest verification: a mismatch or interrupted transfer
rolls back to nothing and fails the dispatch loudly, naming the digest
and the worker. Asset bytes never cross the worker socket - sources are
HTTP pull only. A digest the daemon already resolves costs one round
trip, never a transfer. A daemon that predates staging is refused with
a clear error when staging is needed; pre-seed its store (or drop the
declarations' `nodes` associations) to compose it anyway.

Selected vision-provider assets are admitted earlier, before the job enters
the queue. A current daemon reports which declared model digests it already
holds. Missing digests enter the same explicit acquisition-consent response as
local assets, and approved bytes stage with digest verification before queueing.
The worker checks them again immediately before invocation and asks for a
resubmission if they disappeared. An older daemon that cannot provide this
preflight evidence is not rejected merely for uncertainty: a compatible exact
pack path may run against its pre-seeded store, but execution still never
downloads a model and reports any missing asset honestly.

Job-referenced assets stage the same way. When an invocation's inputs
carry asset values (a library or mount checkpoint picked at the prompt,
for example), the engine asks the daemon which of those digests its
store already resolves and pulls the missing ones from the engine's
advertised asset endpoint before dispatching the node - the endpoint is
the sole source, since input values carry no declaration URLs. A held
digest costs one query round trip, never a transfer. When bytes are
missing and no asset endpoint is configured, or the daemon predates
staging, the dispatch fails loudly at submit time naming the worker,
the digest, and the remedy instead of failing later inside the node.

## Persistent value transport

Without persistent stores, every conversation starts cold: large
values (images, latents, model outputs) re-cross the network on every
reconnect and re-run. With a store on both sides they cross once, ever.

Give the daemon `--value-store DIR`; the engine side is automatic -
`dinkster-serve` keeps its store at `<library-root>/value-store` whenever
a library root is set. When both hellos advertise the capability, an
encoded payload above the CAS threshold crosses as a blake3 digest
reference; the receiver answers with the digests its store is missing
and only those blobs stream, chunked so a multi-GB value never
monopolizes the socket, landing digest-verified before they become
visible. The transport is symmetric - daemon-bound inputs and
engine-bound results use the same frames - so a blob crosses at most
once across conversations, reconnects, and re-runs. Either side
without a store falls back to the conversation-scoped `cas` transport
unchanged, so old daemons and old engines keep working.

Both stores are size-budgeted (daemon: `--value-store-budget`,
default `10G`), evicting least recently used blobs. An evicted blob
is a conservative miss: the peer that still holds the bytes streams
them again. A corrupt stored blob is caught at read, deleted, and the
failing run names the digest and the edge; the next run re-streams it.

Dev diagnostics account for actual movement per edge: `networkBytes`
and `transferMs` are the payload bytes that really streamed - zero on
a store hit, however many runs or reconnects ago the bytes landed.

## Reconnection

The engine redials configured remotes automatically. One supervisor task
per `[worker.NAME]` entry watches the composed session:

- A remote whose startup dial failed keeps its "failed" row on
  `/api/composition` and is retried until the daemon appears.
- A composed remote whose session dies (daemon restart, network loss)
  shows `"status": "disconnected"` on `GET /api/workers`, its
  composition row flips to failed, and the engine redials with
  exponential backoff (1 s doubling to 60 s, jittered). The reattach
  revalidates the new hello like a fresh compose - a daemon redeployed
  with a different node surface swaps in its new surface atomically,
  and a change that would strand a dependent (another worker executing
  a schema this remote owns) is refused, leaving the old surface
  served.

While a remote is disconnected, invocations in flight on the dead
session fail as node errors and placement hints naming it are refused
at submission; nothing queues against a dead worker. After a reattach
the composition row returns to "announced" at a new epoch. A reattach
to a restarted daemon process clears the engine result cache (cache
keys fingerprint schema signature and inputs, not implementation); a
transport-only reconnect to the same process keeps it.

## Session leases and liveness

TCP alone cannot tell an idle peer from a vanished one: a crashed or
partitioned engine leaves a half-open connection that would hold the
daemon's one conversation slot forever. The daemon therefore leases the
slot rather than granting it: any received traffic renews the lease, the
engine sends small heartbeat frames while idle, and a holder silent for
a full `--lease-ttl` (default 45 s) is evicted - its transport aborted,
the slot freed, pack state kept warm for the next engine. A second
engine dialing a held daemon is refused with a frame naming the current
holder and how fresh its lease is (`service is busy: leased to engine
NAME@HOST:pidN (idle 12s, lease ttl 45s)`), so an operator can see who
owns the slot. `--lease-ttl 0` disables eviction; a busy refusal then
lasts until the holder disconnects.

The clock is bilateral. The daemon's hello advertises `leaseTtl`, and
the engine applies the same threshold to the daemon: heartbeats go out
well inside the TTL while idle, and a daemon silent for a full TTL is
declared dead - the session aborts, in-flight invocations fail as node
errors, and the reconnect supervisor (above) redials. Renewal counts
every received frame, not just heartbeat acks, so long-running
invocations that stream results and reports never need separate
liveness traffic.

## Current limitations

Resuming an in-flight invocation across connection loss, concurrent
multi-engine execution against one daemon, direct worker-to-worker
transfer, and hot reload of remotes are not supported.
