"""Remote worker service: a pack served over the network.

    python -m dinkster_workers.service --listen HOST:PORT --manifest PATH \
        [--token-file FILE] [--asset-vault DIR] [--asset-root DIR]

Loads the pack once, listens on TCP, and serves one engine connection at a
time through the same conversation a launched child runs (host.py's
``serve_connection``). RemoteWorker (remote.py) is the engine side.

What differs from the launched-child host, and why:

- **Lifetime.** The service outlives its clients. A client disconnecting
  retains admitted invocations for the reconnect grace; ``shutdown`` or
  grace expiry ends the conversation, never the process. Stopping the
  service is the operator's action (signal), not a peer's frame. Pack state - loaded
  consumers, resident models - survives across conversations, so a
  reconnecting engine finds warm caches, and the relay re-announces them
  through the new hello.
- **Authentication.** A launched child inherits a one-time secret through
  its environment; a network peer cannot. The service requires a
  pre-shared token (file or ``DINKSTER_REMOTE_TOKEN``), presented by the
  client as its first bytes and compared constant-time. The token
  authenticates; it does not encrypt. Pass ``--tls-cert``/``--tls-key``
  to wrap the stream in server-authenticating TLS (the engine pins the
  certificate via ``tls_ca_file`` in remotes.toml), or deploy behind an
  authenticated tunnel (WireGuard, SSH, mTLS proxy). The
  token-first-bytes shape is deliberate: transport security wraps the
  socket without touching the protocol - over TLS the token is simply
  the first decrypted bytes.
- **Negotiation.** A launched pair ships as one build and needs none. A
  remote peer may not: the client opens with ``clientHello`` naming its
  protocol version, and the service answers hello with its own version and
  ``payloadTransports``. Mismatches are refused before any value crosses.
- **No shared memory.** Payloads are inline in the frame, both directions
  (``use_shm=False`` to send, ``accept_shm=False`` to refuse): a segment
  *name* from another machine would attach to unrelated local memory.
  Bulk transfer by content address is the CAS milestone, not a reason to
  let this lie.
- **Liveness.** TCP alone cannot tell an idle peer from a vanished one:
  a half-open connection from a crashed or partitioned engine would hold
  the conversation slot forever. The service hello advertises
  ``leaseTtl``; the engine heartbeats well inside it while idle, and each
  side treats a peer silent for a full TTL as gone - the engine aborts
  and lets its reconnect supervisor redial, the service evicts and frees
  the slot.

One conversation at a time, held by a TTL lease: concurrent clients would
share one pack's consumers with two governors, each translating devices
into its own namespace, so a second engine is refused with a frame naming
the current holder. Any received traffic renews the lease; a holder silent
past ``--lease-ttl`` is evicted (its transport aborted, pack state kept
warm for the next engine). Concurrent multi-engine execution stays out of
scope (DESIGN 3.10 governance).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import logging
import math
import os
import signal
import ssl
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_assets import AssetVault, resolver_from_env, use_declared_asset_pack
from dinkster_assets.declared import ROOT_ENV, VAULT_ENV
from dinkster_caches import DEFAULT_VALUE_STORE_BYTES, BudgetedDiskCAS
from dinkster_memory import parse_size
from dinkster_schema import build_schemas, configure_logging_from_env, install_stream_capture
from dinkster_values import process_instance_token

from .boundary import PROTOCOL_VERSION, BoundaryError, ValueCodec, read_frame, write_frame
from .host import (
    load_choices,
    load_extension_contributions,
    load_pack,
    load_planner,
    load_source_staging,
    parse_comfy_args,
    serve_connection,
)
from .manifest import load_manifest
from .resume import DEFAULT_RESUME_GRACE_S, ResumableConversation
from .staging import AssetStagingService

log = logging.getLogger("dinkster.workers.service")

REMOTE_TOKEN_ENV = "DINKSTER_REMOTE_TOKEN"
_MIN_TOKEN_CHARS = 16
_AUTH_TIMEOUT_S = 10.0
_HELLO_TIMEOUT_S = 30.0
DEFAULT_LEASE_TTL_S = 45.0
_MIN_LEASE_TTL_S = 3.0

READY_LINE_PREFIX = "DINKSTER-SERVICE-LISTENING "
"""Printed to stdout once the socket is bound: ``DINKSTER-SERVICE-LISTENING
tcp:HOST:PORT``. Launchers and tests read the actual port from it when
binding port 0."""


class ServiceError(Exception):
    """The service could not be configured or started."""


def _resolve_token(token_file: str | None) -> bytes:
    if token_file is not None:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    else:
        token = os.environ.get(REMOTE_TOKEN_ENV, "")
    if len(token) < _MIN_TOKEN_CHARS:
        raise ServiceError(
            f"remote service requires a token of at least {_MIN_TOKEN_CHARS} "
            f"characters (from --token-file or ${REMOTE_TOKEN_ENV}); refusing "
            "to serve unauthenticated"
        )
    return token.encode("utf-8")


def _build_server_tls(cert_file: str, key_file: str) -> ssl.SSLContext:
    """A server-authenticating TLS context from a PEM cert/key pair. The
    client authenticates with the pre-shared token inside the encrypted
    stream, so no client certificate is requested."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    except (OSError, ssl.SSLError) as exc:
        raise ServiceError(
            f"cannot load TLS certificate (--tls-cert {cert_file}, --tls-key {key_file}): {exc}"
        ) from exc
    return context


async def run_service(
    host: str,
    port: int,
    manifest_path: str,
    token: bytes,
    *,
    tls: ssl.SSLContext | None = None,
    value_store_root: Path | None = None,
    value_store_bytes: int = DEFAULT_VALUE_STORE_BYTES,
    lease_ttl: float = DEFAULT_LEASE_TTL_S,
    resume_grace: float = DEFAULT_RESUME_GRACE_S,
) -> None:
    if lease_ttl != 0 and not (math.isfinite(lease_ttl) and lease_ttl >= _MIN_LEASE_TTL_S):
        # The engine heartbeats at clamp(ttl/3, 1s, 15s): a positive TTL
        # below the floor's margin would evict every healthy idle session.
        raise ServiceError(
            f"--lease-ttl must be 0 (disabled) or at least "
            f"{_MIN_LEASE_TTL_S:g} seconds, got {lease_ttl:g}"
        )
    if not (math.isfinite(resume_grace) and resume_grace >= 0):
        raise ServiceError(f"--resume-grace must be non-negative, got {resume_grace:g}")
    manifest = load_manifest(Path(manifest_path))
    # One store for the process lifetime, shared across conversations, so a
    # reconnecting engine finds the blobs earlier conversations landed.
    value_store = (
        BudgetedDiskCAS(value_store_root, max_bytes=value_store_bytes)
        if value_store_root is not None
        else None
    )
    # The staging service reads through the SAME environment-assembled
    # chain load_pack installs for declared_asset, so "held" answers and
    # execution-time reads can never disagree. The vault (fetch target)
    # is the chain's verified store; without one the service still
    # answers queries and refuses stageAssets with the reason.
    vault_root = os.environ.get(VAULT_ENV, "")
    asset_staging = AssetStagingService(
        vault=AssetVault(vault_root) if vault_root else None,
        resolver=resolver_from_env(),
    )
    worker, registry, node_classes, arm_workers = load_pack(manifest)
    with use_declared_asset_pack(manifest.name):
        planner = load_planner(manifest)
        consumers = worker.memory_consumers
        source_staging = load_source_staging(manifest)
        choices = load_choices(manifest)
        extension_contributions = load_extension_contributions(manifest)
        schemas = build_schemas(node_classes)
    # One engine owns the loaded pack. Its logical conversation can outlive
    # a physical socket only while admitted invocations remain resumable.
    admission_lock = asyncio.Lock()
    active: ResumableConversation | None = None
    active_transports: tuple[str, ...] | None = None
    conversation_tasks: set[asyncio.Task[None]] = set()
    process_resource_tasks: set[asyncio.Task[None]] = set()
    process_maintenance_operations: set[tuple[object, str]] = set()

    async def serve_resumable(
        conversation: ResumableConversation,
        codec: ValueCodec,
        transports: list[str],
        persistent: bool,
        initial_reader: asyncio.StreamReader,
        initial_writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal active, active_transports
        try:
            with use_declared_asset_pack(manifest.name):
                await serve_connection(
                    initial_reader,
                    initial_writer,
                    pack_name=manifest.name,
                    worker=worker,
                    schemas=schemas,
                    comfy_aliases=manifest.comfy_aliases,
                    comfy_groups=manifest.comfy_groups,
                    planner=planner,
                    consumers=consumers,
                    source_staging=source_staging,
                    choices=choices.static,
                    lazy_choices=choices.lazy,
                    extension_contributions=extension_contributions,
                    codec=codec,
                    arm_workers=arm_workers,
                    body_arms=dict(manifest.arms),
                    vision_providers=manifest.vision_providers,
                    generation_providers=manifest.generation_providers,
                    hello_extra={
                        "protocol": PROTOCOL_VERSION,
                        "payloadTransports": transports,
                        "leaseTtl": lease_ttl,
                        "resumeGrace": resume_grace,
                    },
                    asset_staging=asset_staging,
                    declared_assets=manifest.assets,
                    value_store=value_store if persistent else None,
                    resume=conversation,
                    process_resource_tasks=process_resource_tasks,
                    process_maintenance_operations=process_maintenance_operations,
                )
        finally:
            await conversation.close()
            async with admission_lock:
                if active is conversation:
                    active = None
                    active_transports = None

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal active, active_transports
        try:
            presented = await asyncio.wait_for(
                reader.readexactly(len(token)), timeout=_AUTH_TIMEOUT_S
            )
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        if not hmac.compare_digest(presented, token):
            writer.close()
            return
        ssl_object = writer.get_extra_info("ssl_object")
        if ssl_object is not None:
            # Only after auth, so an unauthenticated prober leaves no line.
            log.info(
                "authenticated connection from %s over %s",
                writer.get_extra_info("peername"),
                ssl_object.version(),
            )
        try:
            frame = await asyncio.wait_for(read_frame(reader), timeout=_HELLO_TIMEOUT_S)
            if frame is None or frame[0].get("type") != "clientHello":
                writer.close()
                return
            client_protocol = frame[0].get("protocol")
            if type(client_protocol) is not int or client_protocol != PROTOCOL_VERSION:
                await write_frame(
                    writer,
                    {
                        "type": "error",
                        "message": (
                            f"protocol mismatch: service speaks {PROTOCOL_VERSION}, "
                            f"client spoke {client_protocol}"
                        ),
                    },
                    [],
                )
                writer.close()
                return
            engine = frame[0].get("engine")
            if not isinstance(engine, Mapping):
                raise BoundaryError("clientHello engine identity is malformed")
            engine_fields = cast("Mapping[str, object]", engine)
            engine_instance_id = engine_fields.get("instanceId")
            raw_label = engine_fields.get("label")
            resumable = engine_fields.get("resumable", False)
            resume_worker_instance = engine_fields.get("resumeWorkerInstance")
            if type(engine_instance_id) is not str or not engine_instance_id:
                raise BoundaryError("clientHello engine instanceId must be a non-empty string")
            if type(raw_label) is not str or not raw_label:
                raise BoundaryError("clientHello engine label must be a non-empty string")
            if type(resumable) is not bool:
                raise BoundaryError("clientHello engine resumable must be a boolean")
            if resume_worker_instance is not None and (
                type(resume_worker_instance) is not str or not resume_worker_instance
            ):
                raise BoundaryError(
                    "clientHello engine resumeWorkerInstance must be a non-empty string"
                )
            client_transports_raw = frame[0].get("payloadTransports")
            if not isinstance(client_transports_raw, list):
                raise BoundaryError("clientHello payloadTransports must be a list of strings")
            transport_entries = cast("list[object]", client_transports_raw)
            if not all(type(item) is str for item in transport_entries):
                raise BoundaryError("clientHello payloadTransports must be a list of strings")
            client_transports = tuple(cast("list[str]", transport_entries))
            if len(client_transports) != len(set(client_transports)):
                raise BoundaryError("clientHello payloadTransports must be unique")
            refusal: str | None = None
            conversation: ResumableConversation | None = None
            is_new = False
            async with admission_lock:
                if active is not None and not active.connected and not active.has_records:
                    await active.close()
                    active = None
                    active_transports = None
                if (
                    active is not None
                    and active.connected
                    and lease_ttl > 0
                    and time.monotonic() - active.last_activity > lease_ttl
                ):
                    log.info(
                        "detaching engine %s (idle %.0fs > lease ttl %.0fs)",
                        active.label,
                        time.monotonic() - active.last_activity,
                        lease_ttl,
                    )
                    await active.detach(active.owner_epoch)
                if active is not None and active.engine_instance_id == engine_instance_id:
                    if resume_worker_instance != active.worker_instance:
                        refusal = "resume refused: daemon process identity does not match"
                    elif active_transports != client_transports:
                        refusal = "resume refused: payload transport offer changed"
                    else:
                        conversation = active
                elif active is not None:
                    idle = time.monotonic() - active.last_activity
                    refusal = f"service is busy: leased to engine {active.label} (idle {idle:.0f}s"
                    refusal += f", lease ttl {lease_ttl:.0f}s)" if lease_ttl > 0 else ")"
                else:
                    conversation = ResumableConversation(
                        engine_instance_id,
                        raw_label,
                        process_instance_token(),
                        grace=resume_grace if resumable else 0,
                    )
                    active = conversation
                    active_transports = client_transports
                    is_new = True
                if conversation is not None:
                    try:
                        await conversation.attach(reader, writer)
                        if is_new:
                            persistent = value_store is not None and (
                                "persistentCas" in client_transports
                            )
                            codec = ValueCodec(
                                registry,
                                use_shm=False,
                                accept_shm=False,
                                value_store=value_store if persistent else None,
                            )
                            if "cas" in client_transports:
                                codec.enable_cas()
                            if persistent:
                                codec.enable_persistent_cas()
                            transports = ["inline", "cas"]
                            if persistent:
                                transports.append("persistentCas")
                            conversation.start_host()
                            task = asyncio.create_task(
                                serve_resumable(
                                    conversation,
                                    codec,
                                    transports,
                                    persistent,
                                    reader,
                                    writer,
                                )
                            )
                            conversation_tasks.add(task)
                            task.add_done_callback(conversation_tasks.discard)
                    except BaseException:
                        if is_new:
                            await conversation.close()
                            if active is conversation:
                                active = None
                                active_transports = None
                        else:
                            await conversation.detach(conversation.owner_epoch)
                        raise
            if refusal is not None:
                await write_frame(writer, {"type": "error", "message": refusal}, [])
                writer.close()
        except (BoundaryError, ConnectionError, TimeoutError) as exc:
            with contextlib.suppress(Exception):
                await write_frame(writer, {"type": "error", "message": str(exc)}, [])
            writer.close()

    if tls is not None:
        # A failed handshake must never wedge the accept loop; the timeout
        # frees the slot the way the auth timeout does for plaintext.
        server = await asyncio.start_server(
            on_connect, host=host, port=port, ssl=tls, ssl_handshake_timeout=_AUTH_TIMEOUT_S
        )
    else:
        server = await asyncio.start_server(on_connect, host=host, port=port)
    bound = server.sockets[0].getsockname()
    if tls is not None:
        log.info("TLS enabled (minimum %s)", tls.minimum_version.name)
    print(f"{READY_LINE_PREFIX}tcp:{bound[0]}:{bound[1]}", flush=True)

    async def watchdog() -> None:
        # A holder on a half-open transport never triggers a read error, so
        # nothing else would notice it; poll well inside the TTL and evict
        # proactively, rather than only when the next engine happens to dial.
        poll = min(lease_ttl / 4.0, 5.0)
        while True:
            await asyncio.sleep(poll)
            async with admission_lock:
                stale = active
                if (
                    stale is not None
                    and stale.connected
                    and time.monotonic() - stale.last_activity > lease_ttl
                ):
                    log.info(
                        "detaching engine %s (idle %.0fs > lease ttl %.0fs)",
                        stale.label,
                        time.monotonic() - stale.last_activity,
                        lease_ttl,
                    )
                    await stale.detach(stale.owner_epoch)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)
    watchdog_task = asyncio.create_task(watchdog()) if lease_ttl > 0 else None
    try:
        async with server:
            await stop.wait()
    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task
        if active is not None:
            await active.close()
        if conversation_tasks:
            await asyncio.gather(*conversation_tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dinkster remote worker service")
    parser.add_argument(
        "--listen",
        required=True,
        help="HOST:PORT to listen on (port 0 binds an ephemeral port, announced on stdout)",
    )
    parser.add_argument("--manifest", required=True, help="path to the pack's dinkster-pack.toml")
    parser.add_argument(
        "--comfy-args-json",
        type=parse_comfy_args,
        default=(),
        metavar="JSON",
        help="validated ComfyUI argv as one JSON array, for packs that read "
        "ComfyUI's import-time CLI (e.g. the compat pack)",
    )
    parser.add_argument(
        "--token-file",
        help=f"file containing the pre-shared token (default: ${REMOTE_TOKEN_ENV})",
    )
    parser.add_argument(
        "--tls-cert",
        help="PEM certificate (chain) presented to connecting engines; "
        "requires --tls-key. Engines pin it via tls_ca_file in "
        "remotes.toml. Generate a self-signed one with: openssl req -x509 "
        "-newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 3650 "
        '-subj "/CN=NAME" -addext "subjectAltName=IP:ADDR,DNS:HOST"',
    )
    parser.add_argument(
        "--tls-key",
        help="PEM private key for --tls-cert",
    )
    parser.add_argument(
        "--asset-vault",
        help="directory for this service's verified asset vault (sets "
        f"${VAULT_ENV}; created if missing). Declared assets the engine "
        "stages land here, and pack code reads them through the same "
        "store chain a local worker uses. Without it the service cannot "
        "stage - engines refuse to dispatch nodes whose declared assets "
        "are not already readable",
    )
    parser.add_argument(
        "--asset-root",
        help="indexed asset library root for read-only resolution (sets "
        f"${ROOT_ENV}). An operator-managed store the service reads but "
        "never writes",
    )
    parser.add_argument(
        "--value-store",
        help="directory for this service's persistent value store (created "
        "if missing). Large boundary values land here once and later runs "
        "reference them by digest instead of re-sending the bytes. Without "
        "it every conversation starts cold and repeat values re-cross the "
        "network",
    )
    parser.add_argument(
        "--value-store-budget",
        type=parse_size,
        default=DEFAULT_VALUE_STORE_BYTES,
        metavar="SIZE",
        help="byte budget for --value-store with an optional K/M/G/T suffix "
        "(default 10G); least recently used blobs are evicted beyond it",
    )
    parser.add_argument(
        "--lease-ttl",
        type=float,
        default=DEFAULT_LEASE_TTL_S,
        metavar="SECONDS",
        help="engine lease: a connected engine silent for this many seconds "
        "is evicted and the slot freed for the next engine (engines "
        "heartbeat automatically while idle, so only a crashed or "
        f"partitioned engine goes silent); minimum {_MIN_LEASE_TTL_S:g}, or "
        "0 to disable eviction so a busy refusal lasts until the holder "
        f"disconnects (default {DEFAULT_LEASE_TTL_S:g})",
    )
    parser.add_argument(
        "--resume-grace",
        type=float,
        default=DEFAULT_RESUME_GRACE_S,
        metavar="SECONDS",
        help="time to retain admitted invocations for the same engine to "
        f"reconnect after a network loss (default {DEFAULT_RESUME_GRACE_S:g})",
    )
    args = parser.parse_args()
    # Logging rides the environment across the process boundary: the host
    # exports DINKSTER_LOG_LEVEL/DINKSTER_LOG and subprocess launches inherit it,
    # so pack log lines land on the shared stderr with their origin intact.
    configure_logging_from_env(os.environ)
    # Node stdout/stderr still reaches this service's terminal, and while a
    # node executes it is also forwarded as attributed execution log events.
    install_stream_capture()
    host, _, port_text = args.listen.rpartition(":")
    if not host or not port_text.isdigit():
        parser.error(f"malformed --listen (want HOST:PORT): {args.listen!r}")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be given together")
    tls = _build_server_tls(args.tls_cert, args.tls_key) if args.tls_cert else None
    # The flags land in the SAME environment variables a launched isolated
    # worker inherits (declared.py's store chain), so a daemon-hosted pack
    # resolves declared assets identically to local execution. An operator
    # may also export the variables directly (including DINKSTER_MOUNTS_SNAPSHOT,
    # which has no flag - snapshots are files an engine host maintains).
    if args.asset_vault:
        os.environ[VAULT_ENV] = args.asset_vault
    if args.asset_root:
        os.environ[ROOT_ENV] = args.asset_root
    token = _resolve_token(args.token_file)
    # Pin the pack bootstrap's argv to the explicit operator-supplied ComfyUI
    # arguments, exactly like the launched worker host does. This applies even
    # to the empty tuple, so no service argument can leak into ComfyUI's
    # import-time parser.
    sys.argv[:] = [sys.argv[0], *args.comfy_args_json]
    asyncio.run(
        run_service(
            host,
            int(port_text),
            args.manifest,
            token,
            tls=tls,
            value_store_root=Path(args.value_store) if args.value_store else None,
            value_store_bytes=args.value_store_budget,
            lease_ttl=args.lease_ttl,
            resume_grace=args.resume_grace,
        )
    )


if __name__ == "__main__":
    main()
