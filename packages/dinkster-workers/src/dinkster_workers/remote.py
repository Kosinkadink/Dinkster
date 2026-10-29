"""RemoteWorker: the Worker protocol across a machine boundary.

The engine cannot tell this apart from InProcessWorker or IsolatedWorker
(hazard H3) - same invoke/prepare/schemas surface, same conversation
(session.py), same conservative failure behavior. What actually changes
when the peer is a service on another machine (service.py):

- **The connection is made, not launched.** There is no subprocess: start()
  dials the configured host:port, presents the pre-shared token, and opens
  with a ``clientHello`` naming this side's protocol version. The service's
  hello must match, and must accept inline payloads, or start() refuses.
- **close() ends the conversation, never the service.** The daemon belongs
  to its operator; a client disconnecting must leave it serving the next
  engine. An admitted invocation can resume only with the same engine and
  daemon process during the negotiated grace; every other loss fails
  conservatively. Relay footprints collapse to zero while disconnected and
  releases claim nothing freed (all session behavior, shared, not
  reimplemented).
- **No shared memory, ever.** Payloads ride inline in the frame both ways;
  a segment name from another machine would attach to unrelated local
  memory. Bulk values are the CAS milestone.
- **Devices are qualified, not trusted.** The remote's ``cuda:0`` is not
  this machine's ``cuda:0``, and its ram is not this machine's ram. Unless
  an explicit DeviceMap is given, every device fact the remote reports is
  suffixed ``@<name>`` (DeviceMap.qualifier), so remote footprints,
  lease requests, and value residency land on remote-namespace budgets
  and lanes - never silently on local ones. Configure budgets for
  ``vram:cuda:0@<name>`` / ``ram@<name>`` to govern the remote's memory
  from this side.

The token authenticates; it does not encrypt. Pass ``tls_ca_file`` (the
daemon's certificate, served with ``--tls-cert``/``--tls-key``) to wrap
the stream in server-authenticating TLS, or deploy across a trusted
network or an authenticated tunnel (see service.py's docstring).
"""

from __future__ import annotations

import asyncio
import math
import os
import secrets
import socket
import ssl
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from dinkster_assets import DeclaredAsset, RemoteSource
from dinkster_memory import MemoryGovernor, ReportedTelemetry, ReservationService
from dinkster_protocol import (
    CompatGateDiagnostic,
    Invocation,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    NodeError,
    OnInvocationEvent,
    ReplicaId,
    WorkGroupDefinition,
)
from dinkster_protocol.pack_surfaces import PackRoute
from dinkster_schema import ComfyAliasRegistry, ComfyGroupRegistry, NodeSchema
from dinkster_values import (
    ASSET_BASE_TYPE,
    Rendition,
    TypeRegistry,
    Value,
    iter_value_tree,
    parse_asset_type_id,
)

from .boundary import PROTOCOL_VERSION, ValueCodec, ValueStore, read_frame, write_frame
from .devices import DeviceMap
from .diagnostics import DiagnosticListener
from .manifest import GenerationProvider, VisionProvider
from .relay import ReleaseGuard, WorkerFullReleaseResult
from .session import BoundarySession, RenditionDeclaration, WorkerDied
from .staging import StageAsset, StagingSource
from .transport import TransportError
from .workgroup import ReplicaEndpoint


class RemoteWorker:
    """A Worker served by a ``dinkster_workers.service`` daemon elsewhere.

    ``name`` identifies this remote in the parent's device namespace: it
    becomes the ``@<name>`` qualifier on every device fact the remote
    reports (unless ``device_map`` overrides the whole translation), and
    should be stable across reconnects so budgets and diagnostics keyed on
    it stay meaningful.
    """

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        registry: TypeRegistry,
        *,
        name: str,
        on_diagnostic: DiagnosticListener | None = None,
        device_map: DeviceMap | None = None,
        reservations: ReservationService | None = None,
        governor: MemoryGovernor | None = None,
        consumer_priority: int = 10,
        release_guard: ReleaseGuard | None = None,
        telemetry: ReportedTelemetry | None = None,
        connect_timeout: float = 30.0,
        asset_endpoint: str | None = None,
        asset_endpoint_token: str | None = None,
        value_store: ValueStore | None = None,
        tls_ca_file: Path | None = None,
        engine_instance_id: str | None = None,
        resume_from: RemoteWorker | None = None,
    ) -> None:
        if not name:
            raise ValueError("RemoteWorker requires a non-empty name")
        if asset_endpoint is not None and not asset_endpoint.startswith(("http://", "https://")):
            raise ValueError(f"asset_endpoint must be an http(s) URL, got: {asset_endpoint!r}")
        self._host = host
        self._port = port
        self._token = token.encode("utf-8")
        self._name = name
        self._connect_timeout = connect_timeout
        self._tls_ca_file = tls_ca_file
        # This engine's advertised asset base URL (peer_asset_sources
        # convention: bytes at <endpoint>/assets/<digest>) and its bearer
        # credential, offered to the daemon as a staging source. The
        # credential rides ONLY requests to this endpoint - declared
        # provenance URLs never see it.
        self._asset_endpoint = asset_endpoint.rstrip("/") if asset_endpoint else None
        self._asset_endpoint_token = asset_endpoint_token
        self._value_store = value_store
        self._engine_instance_id = engine_instance_id or secrets.token_urlsafe(24)
        self._resumable = engine_instance_id is not None

        def new_session() -> tuple[ValueCodec, BoundarySession]:
            codec = ValueCodec(registry, use_shm=False, accept_shm=False, value_store=value_store)
            return codec, BoundarySession(
                registry,
                role="remote worker",
                pack=name,
                codec=codec,
                on_diagnostic=on_diagnostic,
                reservations=reservations,
                device_map=(
                    device_map if device_map is not None else DeviceMap(mapping={}, qualifier=name)
                ),
                governor=governor,
                consumer_priority=consumer_priority,
                release_guard=release_guard,
                telemetry=telemetry,
                death_detail=lambda: f" (connection to {host}:{port} lost)",
                resumable=self._resumable,
            )

        self._session_factory: Callable[[], tuple[ValueCodec, BoundarySession]] = new_session
        self._resume_from = resume_from
        if resume_from is not None:
            if resume_from._engine_instance_id != self._engine_instance_id:
                raise ValueError("resume source belongs to a different engine instance")
            self._codec = resume_from._codec
            self._session = resume_from._session
        else:
            self._codec, self._session = self._session_factory()
        self._started = False

    @property
    def session_identity(self) -> object:
        return self._session

    @property
    def engine_instance_id(self) -> str:
        return self._engine_instance_id

    @property
    def pack(self) -> str:
        return self._session.pack

    @property
    def alive(self) -> bool:
        """Whether the connection's session is currently serving."""
        return self._session.alive

    @property
    def schemas(self) -> Mapping[str, NodeSchema]:
        """The remote pack's node schemas, announced by the service's hello.
        Feed these to the Engine - this machine never imports the pack."""
        return self._session.schemas

    @property
    def comfy_aliases(self) -> ComfyAliasRegistry | None:
        """Maintained ComfyUI import translations announced by the service."""
        return self._session.comfy_aliases

    @property
    def comfy_groups(self) -> ComfyGroupRegistry | None:
        """ComfyUI group translations announced by the service."""
        return self._session.comfy_groups

    @property
    def combo_choices(self) -> Mapping[str, tuple[str, ...]]:
        """The remote pack's combo choice lists (choice-list id -> values),
        as announced by the service's hello; empty when the pack declares
        none. Feed these to the composed surface's /api/choices routes."""
        return self._session.combo_choices

    @property
    def renditions(self) -> tuple[RenditionDeclaration, ...]:
        return self._session.renditions

    async def resolve_rendition(
        self,
        type_id: str,
        kind: str,
        metadata: Mapping[str, object],
        parameters: Mapping[str, str],
    ) -> tuple[str, Mapping[str, str]]:
        return await self._session.resolve_rendition(type_id, kind, metadata, parameters)

    async def resolve_rendition_mime(
        self, type_id: str, kind: str, metadata: Mapping[str, object]
    ) -> str:
        return await self._session.resolve_rendition_mime(type_id, kind, metadata)

    async def render_rendition(
        self, value: Value, kind: str, parameters: Mapping[str, str]
    ) -> Rendition:
        return await self._session.render_rendition(value, kind, parameters)

    @property
    def lazy_choice_ids(self) -> tuple[str, ...]:
        """Choice-list ids the remote pack computes per fetch (see
        ``fetch_choices``), as announced by the service's hello."""
        return self._session.lazy_choice_ids

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        """Run the remote pack's lazy choice provider for ``choice_id``
        and return its values; one invocation per call."""
        return await self._session.fetch_choices(choice_id)

    async def call_pack_route(
        self, route: PackRoute, data: Mapping[str, object]
    ) -> dict[str, object]:
        return await self._session.call_pack_route(route, data)

    @property
    def compat_skips(self) -> Mapping[str, CompatGateDiagnostic]:
        """Classified compat translation skips announced by the service."""
        return self._session.compat_skips

    @property
    def body_arms(self) -> Mapping[str, tuple[str, ...]] | None:
        """The service's registered same-session body capabilities."""
        return self._session.body_arms

    @property
    def vision_providers(self) -> tuple[VisionProvider, ...] | None:
        return self._session.vision_providers

    @property
    def generation_providers(self) -> tuple[GenerationProvider, ...] | None:
        return self._session.generation_providers

    @property
    def attention_route_token(self) -> object | None:
        return self._session.attention_route_token

    @property
    def attention_capabilities(self) -> object | None:
        return self._session.attention_capabilities

    @property
    def extension_contributions(self):
        """RPC-clean extension contribution descriptors from the hello."""
        return self._session.extension_contributions

    @property
    def instance_token(self) -> str | None:
        return self._session.instance_token

    @property
    def device_map_wire(self) -> dict[str, object]:
        return self._session.device_map_wire

    async def full_release(
        self,
        request_id: str,
        worker_instance: str,
        *,
        release_guard: ReleaseGuard | None = None,
    ) -> WorkerFullReleaseResult:
        return await self._session.full_release(
            request_id, worker_instance, release_guard=release_guard
        )

    @property
    def asset_staging(self) -> bool:
        """Whether the service negotiated the asset staging frames."""
        return self._session.asset_staging

    @property
    def declared_assets(self) -> tuple[DeclaredAsset, ...]:
        """The remote pack's [[pack.assets]] declarations from the hello."""
        return self._session.declared_assets

    @property
    def workgroup_capabilities(self) -> frozenset[str]:
        return self._session.workgroup_capabilities

    def bind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> ReplicaEndpoint:
        return self._session.bind_workgroup_endpoint(definition, replica)

    def unbind_workgroup_endpoint(
        self, definition: WorkGroupDefinition, replica: ReplicaId
    ) -> None:
        self._session.unbind_workgroup_endpoint(definition, replica)

    def _build_client_tls(self) -> ssl.SSLContext | None:
        if self._tls_ca_file is None:
            return None
        # Pinned trust: the CA file IS the daemon's certificate (or the CA
        # that signed it). Hostname/IP checking stays on - the daemon's
        # certificate must carry a SAN naming the endpoint dialed here.
        try:
            return ssl.create_default_context(cafile=str(self._tls_ca_file))
        except (OSError, ssl.SSLError) as exc:
            raise TransportError(
                f"remote worker '{self._name}': cannot load TLS CA file {self._tls_ca_file}: {exc}"
            ) from exc

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("RemoteWorker already started")
        self._started = True
        tls = self._build_client_tls()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port, ssl=tls),
                self._connect_timeout,
            )
        except ssl.SSLCertVerificationError as exc:
            raise TransportError(
                f"remote worker '{self._name}': TLS certificate verification "
                f"for {self._host}:{self._port} failed: {exc} (does "
                f"tls_ca_file match the daemon's --tls-cert, and does the "
                f"certificate's SAN name this endpoint?)"
            ) from exc
        except (OSError, TimeoutError) as exc:
            hint = (
                " (TLS was requested; is the daemon serving with --tls-cert/--tls-key?)"
                if tls is not None and isinstance(exc, (ssl.SSLError, ConnectionError))
                else ""
            )
            raise TransportError(
                f"remote worker '{self._name}' could not connect to "
                f"{self._host}:{self._port}: {exc}{hint}"
            ) from exc
        try:
            try:
                writer.write(self._token)
                await writer.drain()
                offered = ["inline", "cas"]
                if self._value_store is not None:
                    # Offered only when this side can actually resolve the
                    # digests a persistentCas descriptor would name.
                    offered.append("persistentCas")
                engine_identity: dict[str, object] = {
                    "label": f"{self._name}@{socket.gethostname()}:pid{os.getpid()}",
                    "instanceId": self._engine_instance_id,
                    "resumable": self._resumable,
                }
                if self._resume_from is not None and self._session.instance_token is not None:
                    engine_identity["resumeWorkerInstance"] = self._session.instance_token
                await write_frame(
                    writer,
                    {
                        "type": "clientHello",
                        "protocol": PROTOCOL_VERSION,
                        "payloadTransports": offered,
                        # A human-readable identity, not a credential: the
                        # daemon names this engine in lease refusals so an
                        # operator can see WHO holds the slot.
                        "engine": engine_identity,
                    },
                    [],
                )
                accepted = await asyncio.wait_for(read_frame(reader), self._connect_timeout)
                if accepted is None:
                    raise ConnectionError("service closed before resume admission")
                admission = accepted[0]
                if admission.get("type") == "error":
                    raise RuntimeError(
                        f"remote worker '{self._name}' refused: "
                        f"{admission.get('message', 'no reason given')}"
                    )
                worker_instance = admission.get("workerInstance")
                owner_epoch = admission.get("ownerEpoch")
                resumed = admission.get("resumed")
                resume_grace = admission.get("resumeGrace")
                if (
                    admission.get("type") != "resumeAccepted"
                    or type(worker_instance) is not str
                    or not worker_instance
                    or type(owner_epoch) is not int
                    or owner_epoch < 1
                    or type(resumed) is not bool
                    or not isinstance(resume_grace, (int, float))
                    or isinstance(resume_grace, bool)
                    or not math.isfinite(float(resume_grace))
                    or resume_grace < 0
                ):
                    raise RuntimeError("service sent malformed resume admission")
                if resumed:
                    if self._resume_from is None or self._session.instance_token != worker_instance:
                        raise RuntimeError("service offered resume without matching daemon state")
                elif self._resume_from is not None:
                    await self._session.close()
                    self._codec, self._session = self._session_factory()
                hello = await self._session.begin(
                    reader,
                    writer,
                    timeout=self._connect_timeout,
                    resuming=resumed,
                    owner_epoch=owner_epoch,
                    resume_grace=float(cast("float", resume_grace)),
                )
                if resumed:
                    await self._session.rebind()
            except ConnectionError as exc:
                # The service hangs up silently on a bad token (it will not
                # tell a prober whether the token was close). A TLS daemon
                # reached without tls_ca_file looks identical from here:
                # its handshake reads these plaintext bytes as a garbled
                # ClientHello and closes.
                tls_hint = (
                    "" if tls is not None else ", or a TLS daemon reached without tls_ca_file"
                )
                raise TransportError(
                    f"remote worker '{self._name}': {self._host}:{self._port} "
                    f"closed the connection during the handshake "
                    f"(wrong token{tls_hint}?)"
                ) from exc
            # begin() validated shape; this side validates negotiation. An
            # old service sends neither field - refuse rather than assume,
            # the whole point of carrying versions is never guessing.
            protocol = int(hello.get("protocol", 0))
            if protocol != PROTOCOL_VERSION:
                raise TransportError(
                    f"remote worker '{self._name}': protocol mismatch "
                    f"(this side speaks {PROTOCOL_VERSION}, service spoke "
                    f"{protocol})"
                )
            transports = hello.get("payloadTransports")
            if not isinstance(transports, list) or "inline" not in transports:
                raise TransportError(
                    f"remote worker '{self._name}': service does not accept inline payloads"
                )
            if "cas" in transports:
                # Both hellos listed cas (ours always does): repeat crossings
                # of the same payload bytes are digest-only from here on.
                self._codec.enable_cas()
            if self._value_store is not None and "persistentCas" in transports:
                # Both sides hold stores: bulk values cross as digest
                # references, the missing blobs stream via the blob frames,
                # and a blob a store already holds never crosses again.
                self._codec.enable_persistent_cas()
                self._session.enable_blob_transfer(self._value_store)
            lease_ttl = hello.get("leaseTtl")
            if isinstance(lease_ttl, (int, float)) and lease_ttl > 0:
                # The daemon's lease clock is the contract: heartbeat well
                # inside it so an idle engine keeps its slot, and treat a
                # daemon silent for a full lease as dead (same threshold the
                # daemon applies to us - neither side waits on TCP).
                ttl = float(lease_ttl)
                self._session.start_keepalive(min(max(ttl / 3.0, 1.0), 15.0), ttl)
        except BaseException:
            writer.close()
            await self.rollback_start()
            raise

    async def prepare(self, node_types: Sequence[str]) -> None:
        # Declared assets stage BEFORE preparation reaches the pack:
        # declared_asset() never downloads at execution time, so a
        # dispatched node's digests must already resolve daemon-side.
        await self._stage_declared_assets(node_types)
        await self._session.prepare(node_types)

    def _endpoint_source(self, digest: str) -> StagingSource | None:
        """This engine's advertised pull URL for one digest, or None when no
        endpoint is configured."""
        if self._asset_endpoint is None:
            return None
        headers = (
            {"Authorization": f"Bearer {self._asset_endpoint_token}"}
            if self._asset_endpoint_token
            else {}
        )
        return StagingSource(url=f"{self._asset_endpoint}/assets/{digest}", headers=headers)

    def _staging_sources(
        self,
        digest: str,
        declared: DeclaredAsset,
        hinted_sources: Sequence[str] = (),
    ) -> tuple[StagingSource, ...]:
        """Candidate pull URLs for one digest, best-first: this engine's
        advertised endpoint (the bytes consented acquisition landed here),
        then the declaration's own remote leads."""
        sources: list[StagingSource] = []
        endpoint = self._endpoint_source(digest)
        if endpoint is not None:
            sources.append(endpoint)
        for source in declared.need.sources:
            if isinstance(source, RemoteSource):
                sources.append(StagingSource(url=source.url))
        known = {source.url for source in sources}
        for url in hinted_sources:
            if url not in known:
                sources.append(StagingSource(url=url))
                known.add(url)
        return tuple(sources)

    @staticmethod
    def _provider_asset_plan(
        declared: DeclaredAsset,
        *,
        status: str,
        sources: Sequence[str],
        detail: str = "",
    ) -> dict[str, object]:
        need = declared.need
        entry: dict[str, object] = {
            "digest": need.digest,
            "name": need.name,
            "status": status,
            "sources": list(sources),
            "fetchable": bool(sources),
        }
        if need.kind:
            entry["kind"] = need.kind
        if need.size >= 0:
            entry["size"] = need.size
        if detail:
            entry["detail"] = detail
        return entry

    async def _query_missing(self, digests: Sequence[str]) -> set[str]:
        """The subset of ``digests`` the daemon's store does not hold.

        The daemon's vault stays the single source of truth for what is
        held: each dispatch asks (assetQuery) and stages only the missing
        digests, so an operator clearing the daemon's store is safe and a
        held digest costs one round trip, never a transfer."""
        query = await self._session.query_assets(sorted(digests))
        query_error = query.get("error")
        if query_error is not None:
            raise TransportError(f"remote worker '{self._name}': asset query failed: {query_error}")
        held_raw = query.get("held")
        missing_raw = query.get("missing")
        if not isinstance(held_raw, list) or not isinstance(missing_raw, list):
            raise TransportError(
                f"remote worker '{self._name}': malformed assetQueryResult "
                "(want 'held' and 'missing' lists)"
            )
        held = {str(digest) for digest in cast("list[Any]", held_raw)}
        missing = {str(digest) for digest in cast("list[Any]", missing_raw)}
        unaccounted = sorted(set(digests) - held - missing)
        if unaccounted:
            raise TransportError(
                f"remote worker '{self._name}': assetQueryResult did not "
                f"account for digests: {', '.join(unaccounted)}"
            )
        return missing

    async def _run_stage(self, assets: Sequence[StageAsset], what: str) -> None:
        """Pull ``assets`` into the daemon's vault; every failure is a loud
        TransportError naming this worker and the digest."""
        result = await self._session.stage_assets(assets)
        stage_error = result.get("error")
        if stage_error is not None:
            raise TransportError(
                f"remote worker '{self._name}': asset staging failed: {stage_error}"
            )
        failed_raw = result.get("failed")
        if isinstance(failed_raw, dict):
            failures = ", ".join(
                f"{digest}: {reason}"
                for digest, reason in sorted(cast("dict[str, Any]", failed_raw).items())
            )
            raise TransportError(
                f"remote worker '{self._name}' could not stage {what} - {failures}"
            )
        staged_raw = result.get("staged")
        staged = (
            {str(digest) for digest in cast("list[Any]", staged_raw)}
            if isinstance(staged_raw, list)
            else set[str]()
        )
        unstaged = sorted({asset.digest for asset in assets} - staged)
        if unstaged:
            raise TransportError(
                f"remote worker '{self._name}': service did not stage "
                f"digests: {', '.join(unstaged)}"
            )

    async def preflight_provider_assets(
        self,
        asset_ids: Sequence[str],
        consented: frozenset[str],
        hinted_sources: Mapping[str, Sequence[str]],
    ) -> list[dict[str, object]]:
        """Materialize selected provider artifacts before a job queues.

        An older daemon without staging is unknown rather than unsupported;
        invocation still gets a download-free read and an actionable error
        if the artifact is absent. A current daemon is queried first, so
        bytes it already holds never require consent or transfer.
        """
        declared_by_id = {asset.id: asset for asset in self._session.declared_assets}
        unknown = sorted(set(asset_ids) - declared_by_id.keys())
        if unknown:
            raise TransportError(
                f"remote worker '{self._name}': provider names undeclared assets: "
                f"{', '.join(unknown)}"
            )
        needed = {
            declared_by_id[asset_id].need.digest: declared_by_id[asset_id] for asset_id in asset_ids
        }
        if not needed or not self._session.asset_staging:
            return []
        try:
            missing = await self._query_missing(sorted(needed))
        except TransportError as exc:
            return [
                self._provider_asset_plan(
                    needed[digest],
                    status="failed",
                    sources=(),
                    detail=str(exc),
                )
                for digest in sorted(needed)
            ]
        plan: list[dict[str, object]] = []
        for digest in sorted(missing):
            declared = needed[digest]
            remote_urls = [
                source.url for source in declared.need.sources if isinstance(source, RemoteSource)
            ]
            for url in hinted_sources.get(digest, ()):
                if url not in remote_urls:
                    remote_urls.append(url)
            if digest not in consented:
                endpoint = self._endpoint_source(digest)
                if endpoint is not None:
                    try:
                        await self._run_stage(
                            (
                                StageAsset(
                                    digest=digest,
                                    name=declared.need.name,
                                    size=declared.need.size,
                                    sources=(endpoint,),
                                ),
                            ),
                            "provider asset",
                        )
                    except TransportError:
                        pass
                    else:
                        continue
                plan.append(
                    self._provider_asset_plan(
                        declared,
                        status="missing",
                        sources=remote_urls,
                    )
                )
                continue
            sources = self._staging_sources(
                digest,
                declared,
                hinted_sources.get(digest, ()),
            )
            if not sources:
                plan.append(
                    self._provider_asset_plan(
                        declared,
                        status="failed",
                        sources=remote_urls,
                        detail=(
                            f"remote worker '{self._name}' does not hold this provider asset "
                            "and has no staging source"
                        ),
                    )
                )
                continue
            try:
                await self._run_stage(
                    (
                        StageAsset(
                            digest=digest,
                            name=declared.need.name,
                            size=declared.need.size,
                            sources=sources,
                        ),
                    ),
                    "provider asset",
                )
            except TransportError as exc:
                plan.append(
                    self._provider_asset_plan(
                        declared,
                        status="failed",
                        sources=remote_urls,
                        detail=str(exc),
                    )
                )
        return plan

    async def _verify_provider_assets(self, invocation: Invocation) -> None:
        if not self._session.asset_staging:
            return
        provider = next(
            (
                declaration
                for declaration in self._session.vision_providers or ()
                if declaration.node == invocation.node_type
            ),
            None,
        )
        if provider is None or not provider.artifacts:
            return
        declared_by_id = {asset.id: asset for asset in self._session.declared_assets}
        needed = {declared_by_id[asset_id].need.digest: asset_id for asset_id in provider.artifacts}
        missing = await self._query_missing(sorted(needed))
        if missing:
            descriptions = ", ".join(f"{needed[digest]} ({digest})" for digest in sorted(missing))
            raise TransportError(
                f"remote worker '{self._name}': provider assets disappeared after job "
                f"admission: {descriptions}; resubmit the job to preflight them again"
            )

    async def _stage_declared_assets(self, node_types: Sequence[str]) -> None:
        """Materialize every declared digest the dispatched types require.

        The daemon's vault stays the single source of truth for what is
        held: each dispatch asks (assetQuery) and stages only the missing
        digests, so an operator clearing the daemon's store is safe and a
        held digest costs one round trip, never a transfer. Every failure
        is a loud TransportError naming this worker and the digest."""
        wanted = set(node_types)
        needed: dict[str, DeclaredAsset] = {}
        for declared in self._session.declared_assets:
            if declared.nodes and not wanted.isdisjoint(declared.nodes):
                needed.setdefault(declared.need.digest, declared)
        if not needed:
            return
        if not self._session.asset_staging:
            raise TransportError(
                f"remote worker '{self._name}': dispatched node types "
                f"require declared assets ({', '.join(sorted(needed))}) but "
                f"the service at {self._host}:{self._port} predates asset "
                "staging - upgrade the daemon, or pre-seed its asset store "
                "and remove the declarations' node associations"
            )
        missing = await self._query_missing(sorted(needed))
        to_stage = sorted(set(needed) & missing)
        if not to_stage:
            return
        assets: list[StageAsset] = []
        for digest in to_stage:
            declared = needed[digest]
            sources = self._staging_sources(digest, declared)
            if not sources:
                raise TransportError(
                    f"remote worker '{self._name}': declared asset "
                    f"{declared.id!r} ({digest}) is not held by the service "
                    "and has no staging sources - declare remote URLs or "
                    "advertise this engine's asset endpoint (asset_endpoint)"
                )
            assets.append(
                StageAsset(
                    digest=digest,
                    name=declared.need.name,
                    size=declared.need.size,
                    sources=sources,
                )
            )
        await self._run_stage(assets, "declared assets")

    async def _stage_job_assets(self, invocation: Invocation) -> None:
        """Materialize every asset the invocation's input values reference.

        Job-referenced assets (engine-library/mount assets such as
        checkpoints) cross the boundary as digest envelopes, never bytes:
        without this preflight the daemon-side AssetRef fails at
        execution time unless an operator pre-seeded the vault. Dispatch
        asks the daemon which digests it holds and pulls the missing ones
        from this engine's advertised asset endpoint - the sole source,
        since unlike declared assets there are no declaration-carried
        URLs. Every failure is loud and dispatch-time, naming this worker
        and the digest, never a resolver error inside the node."""
        needed: dict[str, tuple[str, int]] = {}
        for value in invocation.inputs.values():
            for node in iter_value_tree(value):
                if node.type_id == ASSET_BASE_TYPE or parse_asset_type_id(node.type_id) is not None:
                    references: list[object] = [node.meta.entries]
                elif "asset_refs" in node.meta.entries:
                    nested = node.meta.get("asset_refs", [])
                    if not isinstance(nested, list):
                        raise TransportError(f"{node.type_id} asset_refs must be a list")
                    references = cast("list[object]", nested)
                else:
                    continue
                for reference in references:
                    if not isinstance(reference, Mapping):
                        raise TransportError("asset reference must be a metadata mapping")
                    reference = cast("Mapping[str, object]", reference)
                    digest = reference.get("digest")
                    if not isinstance(digest, str) or not digest:
                        raise TransportError(
                            f"remote worker '{self._name}': asset input of node "
                            f"{invocation.node_id!r} ({invocation.node_type}) "
                            "carries no digest metadata - cannot verify the "
                            "service holds its content"
                        )
                    name = reference.get("name")
                    size = reference.get("size")
                    needed.setdefault(
                        digest,
                        (
                            name if isinstance(name, str) else "",
                            size if isinstance(size, int) else 0,
                        ),
                    )
        if not needed:
            return
        if not self._session.asset_staging:
            raise TransportError(
                f"remote worker '{self._name}': node {invocation.node_id!r} "
                f"({invocation.node_type}) references assets "
                f"({', '.join(sorted(needed))}) but the service at "
                f"{self._host}:{self._port} predates asset staging - "
                "upgrade the daemon, or pre-seed its asset store"
            )
        missing = await self._query_missing(sorted(needed))
        to_stage = sorted(set(needed) & missing)
        if not to_stage:
            return
        assets: list[StageAsset] = []
        for digest in to_stage:
            source = self._endpoint_source(digest)
            if source is None:
                raise TransportError(
                    f"remote worker '{self._name}': asset {digest} referenced "
                    f"by node {invocation.node_id!r} ({invocation.node_type}) "
                    "is not held by the service and this engine advertises no "
                    "asset endpoint (asset_endpoint) - configure one or "
                    "pre-seed the daemon's asset store"
                )
            name, size = needed[digest]
            assets.append(StageAsset(digest=digest, name=name, size=size, sources=(source,)))
        await self._run_stage(assets, "job assets")

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        try:
            await self._verify_provider_assets(invocation)
            await self._stage_job_assets(invocation)
        except WorkerDied:
            # Fall through: session.invoke returns the canonical dead result.
            pass
        except TransportError as exc:
            # Same failure shape the session gives boundary errors: the
            # engine sees a clean node failure, not a torn-down run.
            return InvocationResult(
                error=NodeError(invocation.node_id, invocation.node_type, str(exc))
            )
        return await self._session.invoke(invocation, on_event=on_event)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        return await self._session.check_lazy_status(invocation, on_event=on_event)

    async def close(self) -> None:
        """Disconnect. The service keeps running and keeps its pack state;
        this ends only our conversation (deliberately no ``shutdown`` frame
        - a client must not be able to stop a shared daemon)."""
        await self._session.close()

    async def rollback_start(self) -> None:
        """Undo an unpublished dial without discarding a shared resume session."""
        if self._resume_from is not None and self._session is self._resume_from._session:
            await self._session.detach_transport()
        else:
            await self._session.close()
