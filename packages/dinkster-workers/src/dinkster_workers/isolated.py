"""IsolatedWorker: the Worker protocol across a process boundary.

The engine cannot tell this apart from InProcessWorker (hazard H3) - that
is the point. What actually changes:

- The pack runs in a child process (its own interpreter, optionally its own
  venv via ``provision.ensure_pack_venv``). The parent never imports pack
  code; it learns the pack's interface from the hello handshake, in the
  schema wire format (hazards H1/H5).
- Values cross as envelopes. Fingerprints computed where a value was
  produced travel with it, so cache keys are location-independent (H4).
  Types the parent has not registered still flow through it - carried,
  cached, interrogated via meta, and relayed back over the boundary as
  their original codec bytes - only resolve() requires the type.
- Every invocation reports a BoundaryDiagnostic (DESIGN 3.9) so the cost
  of the boundary is visible, not mysterious.

The conversation itself - hello, invocations, leases, the memory relay,
the ram-release gate - is a BoundarySession (session.py), shared verbatim
with RemoteWorker. This class owns what is genuinely subprocess-shaped:
launching the child, the endpoint listener, and reaping the process. *How*
the child is spawned is a Launcher (launch.py) - plain subprocess by
default, a bubblewrap jail via sandbox.BubblewrapLauncher - and nothing
here or below can tell the difference (DESIGN 3.11).

Lifecycle: ``await start()`` before use, ``await close()`` after. If the
child dies, in-flight and subsequent invocations fail with a NodeError
naming the pack - the engine surfaces it like any node failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dinkster_assets import resolver_from_env
from dinkster_memory import MemoryGovernor, ReportedTelemetry, ReservationService
from dinkster_protocol import (
    CompatGateDiagnostic,
    Invocation,
    InvocationResult,
    KeyedContribution,
    LazyStatusInvocation,
    LazyStatusResult,
    NodeError,
    OnInvocationEvent,
    ReplicaId,
    WorkGroupDefinition,
)
from dinkster_protocol.pack_surfaces import PackRoute
from dinkster_schema import NodeSchema
from dinkster_values import TypeRegistry

from .boundary import DEFAULT_SHM_THRESHOLD, ValueCodec
from .devices import DeviceMap
from .diagnostics import DiagnosticListener
from .headroom import HeadroomMirror, worker_vram_bytes
from .interpreter import InterpreterPreflightError, preflight_interpreter
from .launch import Launcher, LaunchSpec, SubprocessLauncher
from .manifest import load_manifest
from .relay import ReleaseGuard, WorkerFullReleaseResult
from .saved_artifacts import SavedArtifactAuthority
from .session import BoundarySession
from .transport import BoundaryListener, TransportChoice
from .workgroup import ReplicaEndpoint

log = logging.getLogger("dinkster.workers.isolated")

_CHILD_SHUTDOWN_GRACE = 5.0


def _artifact_authority(env: Mapping[str, str]):
    snapshot = env.get("DINKSTER_MOUNTS_SNAPSHOT") or os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
    return SavedArtifactAuthority(snapshot).capture if snapshot else None


async def _await_close(task: asyncio.Task[None]) -> None:
    """Finish owned process cleanup before delivering caller cancellation."""
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        await task
        raise cancelled


def _worker_vram_budgets(
    budgets: Mapping[str, int], device_map: DeviceMap | None
) -> dict[int, int]:
    """Project parent budget keys into the worker's CUDA namespace.

    ServingComposer launches identity-mapped workers, but direct worker users
    may pin a child with CUDA_VISIBLE_DEVICES and supply a DeviceMap. An
    unmappable or ambiguous inverse is not guessed: that budget contributes
    zero activation headroom and emits one named diagnostic.
    """
    return worker_vram_bytes(budgets, device_map)


class IsolatedWorker:
    def __init__(
        self,
        manifest_path: Path | str,
        registry: TypeRegistry,
        *,
        python: str | None = None,
        on_diagnostic: DiagnosticListener | None = None,
        shm_threshold: int = DEFAULT_SHM_THRESHOLD,
        use_shm: bool = True,
        extra_env: Mapping[str, str] | None = None,
        start_timeout: float = 60.0,
        transport: TransportChoice = "auto",
        reservations: ReservationService | None = None,
        device_map: DeviceMap | None = None,
        governor: MemoryGovernor | None = None,
        consumer_priority: int = 10,
        release_guard: ReleaseGuard | None = None,
        telemetry: ReportedTelemetry | None = None,
        launcher: Launcher | None = None,
        aimdo_init: bool = False,
        aimdo_arm: str = "auto",
        vram_budgets: Mapping[str, int] | None = None,
        reserve_vram: int | None = None,
        comfy_args: tuple[str, ...] = (),
        headroom_mirror: HeadroomMirror | None = None,
        on_schema_reload: Callable[[str, str], None] | None = None,
    ) -> None:
        if aimdo_arm not in ("off", "auto", "on"):
            raise ValueError(f"aimdo_arm must be 'off', 'auto', or 'on', got {aimdo_arm!r}")
        self._manifest = load_manifest(manifest_path)
        self._python = python or sys.executable
        self._launcher: Launcher = launcher or SubprocessLauncher()
        self._aimdo_init = aimdo_init
        self._aimdo_arm = aimdo_arm
        self._vram_budgets = _worker_vram_budgets(vram_budgets or {}, device_map)
        if reserve_vram is not None and reserve_vram < 0:
            raise ValueError("reserve_vram must be non-negative")
        self._reserve_vram = reserve_vram
        self._comfy_args = comfy_args
        self._headroom_mirror = headroom_mirror
        self._device_map = device_map
        self._shm_threshold = shm_threshold
        self._use_shm = use_shm
        self._extra_env = dict(extra_env or {})
        self._start_timeout = start_timeout
        self._transport: TransportChoice = transport
        self._session = BoundarySession(
            registry,
            role="isolated worker",
            pack=self._manifest.name,
            codec=ValueCodec(registry, shm_threshold=shm_threshold, use_shm=use_shm),
            on_diagnostic=on_diagnostic,
            reservations=reservations,
            device_map=device_map,
            governor=governor,
            consumer_priority=consumer_priority,
            release_guard=release_guard,
            telemetry=telemetry,
            death_detail=self._death_detail,
            artifact_authority=_artifact_authority(self._extra_env),
            produced_asset_source=resolver_from_env(self._extra_env),
            on_schema_reload=on_schema_reload,
        )
        self._proc: asyncio.subprocess.Process | None = None
        self._process_watch: asyncio.Task[None] | None = None
        self._listener: BoundaryListener | None = None
        self._tmpdir: Path | None = None
        self._close_task: asyncio.Task[None] | None = None

    def _death_detail(self) -> str:
        code = self._proc.returncode if self._proc is not None else None
        return f" (exit code {code})" if code is not None else ""

    @property
    def pack(self) -> str:
        return self._session.pack

    @property
    def alive(self) -> bool:
        """Whether the child's session is currently serving."""
        return self._session.alive

    @property
    def instance_token(self) -> str | None:
        """The child process's lifetime token (hello ``workerInstance``);
        the owner identity its resident envelopes carry. None before
        start() or when the child announced none."""
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
    def attention_route_token(self) -> object | None:
        return self._session.attention_route_token

    @property
    def attention_capabilities(self) -> object | None:
        return self._session.attention_capabilities

    @property
    def schemas(self) -> Mapping[str, NodeSchema]:
        """The pack's node schemas, as announced by the hello handshake.
        Feed these to the Engine - the parent never imports the pack."""
        return self._session.schemas

    @property
    def combo_choices(self) -> Mapping[str, tuple[str, ...]]:
        """The pack's combo choice lists (choice-list id -> values), as
        announced by the hello handshake; empty when the pack declares
        none. Feed these to the composed surface's /api/choices routes."""
        return self._session.combo_choices

    @property
    def lazy_choice_ids(self) -> tuple[str, ...]:
        """Choice-list ids the pack computes per fetch (see
        ``fetch_choices``), as announced by the hello handshake."""
        return self._session.lazy_choice_ids

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        """Run the pack's lazy choice provider for ``choice_id`` in the
        subprocess and return its values; one invocation per call."""
        return await self._session.fetch_choices(choice_id)

    async def call_pack_route(
        self, route: PackRoute, data: Mapping[str, object]
    ) -> dict[str, object]:
        return await self._session.call_pack_route(route, data)

    @property
    def compat_skips(self) -> Mapping[str, CompatGateDiagnostic]:
        """Classified compat translation skips announced by the worker."""
        return self._session.compat_skips

    @property
    def body_arms(self) -> Mapping[str, tuple[str, ...]] | None:
        """The child's registered same-session body capabilities."""
        return self._session.body_arms

    @property
    def can_convert_legacy_checkpoint(self) -> bool:
        return self._session.can_convert_legacy_checkpoint

    @property
    def extension_contributions(self):
        """RPC-clean extension contribution descriptors from worker hello."""
        return self._session.extension_contributions

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

    async def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("IsolatedWorker already started")
        try:
            preflight_interpreter(self._python)
        except InterpreterPreflightError as exc:
            raise RuntimeError(f"isolated worker '{self.pack}' interpreter refused: {exc}") from exc
        self._tmpdir = Path(tempfile.mkdtemp(prefix="dinkster-iso-"))
        try:
            self._listener = await BoundaryListener.create(self._tmpdir, transport=self._transport)
            connected = self._listener.connected
            command = [
                self._python,
                "-m",
                "dinkster_workers.host",
                "--endpoint",
                self._listener.endpoint,
                "--manifest",
                str(self._manifest.path),
                "--shm-threshold",
                str(self._shm_threshold),
                "--comfy-args-json",
                json.dumps(self._comfy_args),
            ]
            if self._aimdo_init or self._aimdo_arm != "off":
                command.append("--aimdo-init")
            command.extend(("--aimdo-arm", self._aimdo_arm))
            if self._reserve_vram is not None:
                command.extend(("--reserve-vram", str(self._reserve_vram)))
            for index, nbytes in sorted(self._vram_budgets.items()):
                command.extend(("--vram-budget", f"{index}={nbytes}"))
            if not self._use_shm:
                command.append("--no-shm")
            self._proc = await self._launcher.launch(
                LaunchSpec(
                    command=tuple(command),
                    env={**self._extra_env, **self._listener.child_env},
                    endpoint_dir=self._tmpdir,
                    pack_root=self._manifest.root,
                    python=self._python,
                    use_shm=self._use_shm,
                )
            )
        except BaseException:
            await self.close()
            raise
        exited = asyncio.ensure_future(self._proc.wait())
        try:
            done, _ = await asyncio.wait(
                {connected, exited},
                timeout=self._start_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if connected not in done:
                detail = (
                    f"exited with code {self._proc.returncode}"
                    if exited in done
                    else f"did not connect within {self._start_timeout}s"
                )
                raise RuntimeError(f"isolated worker '{self.pack}' failed to start: {detail}")
            reader, writer = connected.result()
            await self._session.begin(reader, writer, timeout=self._start_timeout)
            process = self._proc

            async def watch_process() -> None:
                await process.wait()
                self._session.process_died()

            self._process_watch = asyncio.create_task(watch_process())
            if self._headroom_mirror is not None and (
                self._reserve_vram is not None or self._aimdo_arm != "off"
            ):
                await self._headroom_mirror.register(self._session, self._device_map)
        except BaseException:
            await self.close()
            raise
        finally:
            if not exited.done():
                exited.cancel()

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._session.prepare(node_types)

    async def materialize_sampler_registry(
        self, key: str
    ) -> tuple[tuple[str, tuple[KeyedContribution, ...]], ...]:
        return await self._session.materialize_sampler_registry(key)

    async def convert_legacy_checkpoint(
        self, path: Path, logical_name: str
    ) -> tuple[str, str | None] | None:
        return await self._session.convert_legacy_checkpoint(path, logical_name)

    async def materialize_inference_generation(
        self, key: str
    ) -> tuple[tuple[str, tuple[KeyedContribution, ...]], ...]:
        return await self._session.materialize_inference_generation(key)

    async def release_inference_generation(self, key: str) -> None:
        await self._session.release_inference_generation(key)

    async def compile_graph(
        self,
        generation_key: str,
        graph: Mapping[str, Any],
        targets: Sequence[str],
    ) -> dict[str, Any]:
        return await self._session.compile_graph(generation_key, graph, targets)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        result = await self._session.invoke(invocation, on_event=on_event)
        error = result.error
        dead_message = f"isolated worker '{self.pack}' is not running"
        if (
            error is None
            or self._session.alive
            or error.message != dead_message
            or self._proc is None
        ):
            return result

        # Pipe EOF can reach BoundarySession just before asyncio publishes the
        # child's returncode. The process has already stopped serving, so give
        # the child watcher one bounded chance to reap it and preserve the
        # actual exit status/signal in this first in-flight failure too.
        if self._proc.returncode is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), 1.0)
        detail = self._death_detail()
        if not detail:
            return result
        return InvocationResult(
            error=NodeError(
                node_id=error.node_id,
                node_type=error.node_type,
                message=f"{error.message}{detail}",
                traceback=error.traceback,
                hints=error.hints,
            )
        )

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        return await self._session.check_lazy_status(invocation, on_event=on_event)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _await_close(self._close_task)

    async def _close(self) -> None:
        if self._proc is not None:
            if self._session.alive:
                with contextlib.suppress(Exception):
                    await self._session.send({"type": "shutdown"}, [])
            if self._proc.returncode is None:
                try:
                    await asyncio.wait_for(self._proc.wait(), _CHILD_SHUTDOWN_GRACE)
                except TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            self._session.process_died()
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._headroom_mirror is not None:
            self._headroom_mirror.deregister(self._session)
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
        if self._proc is not None:
            self._session.process_died()
        if self._process_watch is not None:
            await self._process_watch
            self._process_watch = None
        await self._session.close()
        if self._listener is not None:
            await self._listener.close()
            self._listener = None
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


class GroupMemberWorker:
    """One pack/session facade owned by a GroupIsolatedWorker."""

    def __init__(self, group: GroupIsolatedWorker, session: BoundarySession) -> None:
        self._group = group
        self._session = session

    @property
    def pack(self) -> str:
        return self._session.pack

    @property
    def alive(self) -> bool:
        return self._session.alive

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
    def attention_route_token(self) -> object | None:
        return self._session.attention_route_token

    @property
    def attention_capabilities(self) -> object | None:
        return self._session.attention_capabilities

    @property
    def schemas(self):
        return self._session.schemas

    @property
    def combo_choices(self):
        return self._session.combo_choices

    @property
    def lazy_choice_ids(self):
        return self._session.lazy_choice_ids

    async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
        return await self._session.fetch_choices(choice_id)

    async def call_pack_route(
        self, route: PackRoute, data: Mapping[str, object]
    ) -> dict[str, object]:
        return await self._session.call_pack_route(route, data)

    @property
    def compat_skips(self):
        return self._session.compat_skips

    @property
    def body_arms(self):
        return self._session.body_arms

    @property
    def can_convert_legacy_checkpoint(self) -> bool:
        return self._session.can_convert_legacy_checkpoint

    @property
    def extension_contributions(self):
        return self._session.extension_contributions

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

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self._session.prepare(node_types)

    async def materialize_sampler_registry(self, key: str):
        return await self._session.materialize_sampler_registry(key)

    async def convert_legacy_checkpoint(
        self, path: Path, logical_name: str
    ) -> tuple[str, str | None] | None:
        return await self._session.convert_legacy_checkpoint(path, logical_name)

    async def materialize_inference_generation(self, key: str):
        return await self._session.materialize_inference_generation(key)

    async def release_inference_generation(self, key: str) -> None:
        await self._session.release_inference_generation(key)

    async def compile_graph(
        self,
        generation_key: str,
        graph: Mapping[str, Any],
        targets: Sequence[str],
    ) -> dict[str, Any]:
        return await self._session.compile_graph(generation_key, graph, targets)

    async def invoke(self, invocation: Invocation, on_event: OnInvocationEvent | None = None):
        result = await self._session.invoke(invocation, on_event=on_event)
        error = result.error
        dead = f"isolated worker '{self.pack}' is not running"
        if error is None or self._session.alive or not error.message.startswith(dead):
            return result
        code = await self._group.reap_code()
        return InvocationResult(
            error=NodeError(
                node_id=error.node_id,
                node_type=error.node_type,
                message=(
                    f"pack {self.pack!r} is not running "
                    f"(worker group {self._group.name!r} exited with code {code})"
                ),
                traceback=error.traceback,
                hints=error.hints,
            )
        )

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        return await self._session.check_lazy_status(invocation, on_event=on_event)


class GroupIsolatedWorker:
    """One subprocess with one independent boundary session per pack."""

    def __init__(
        self,
        name: str,
        manifest_paths: Sequence[Path | str],
        registry: TypeRegistry,
        *,
        python: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        start_timeout: float = 60.0,
        launcher: Launcher | None = None,
        shm_threshold: int = DEFAULT_SHM_THRESHOLD,
        use_shm: bool = True,
        on_diagnostic: DiagnosticListener | None = None,
        transport: TransportChoice = "auto",
        reservations: ReservationService | None = None,
        device_map: DeviceMap | None = None,
        governor: MemoryGovernor | None = None,
        consumer_priority: int = 10,
        release_guard: ReleaseGuard | None = None,
        telemetry: ReportedTelemetry | None = None,
        aimdo_init: bool = False,
        aimdo_arm: str = "auto",
        vram_budgets: Mapping[str, int] | None = None,
        reserve_vram: int | None = None,
        comfy_args: tuple[str, ...] = (),
        headroom_mirror: HeadroomMirror | None = None,
        on_schema_reload: Callable[[str, str], None] | None = None,
    ) -> None:
        if not name or not manifest_paths:
            raise ValueError("worker group requires a name and at least one manifest")
        self.name = name
        self._manifests = tuple(load_manifest(path) for path in manifest_paths)
        if len({m.name for m in self._manifests}) != len(self._manifests):
            raise ValueError(f"worker group {name!r} contains duplicate pack names")
        self._python = python or sys.executable
        self._extra_env = dict(extra_env or {})
        self._start_timeout = start_timeout
        self._launcher = launcher or SubprocessLauncher()
        self._shm_threshold = shm_threshold
        self._use_shm = use_shm
        self._transport: TransportChoice = transport
        self._aimdo_init = aimdo_init
        if aimdo_arm not in ("off", "auto", "on"):
            raise ValueError("aimdo_arm must be 'off', 'auto', or 'on'")
        if reserve_vram is not None and reserve_vram < 0:
            raise ValueError("reserve_vram must be non-negative")
        self._aimdo_arm = aimdo_arm
        self._vram_budgets = _worker_vram_budgets(vram_budgets or {}, device_map)
        self._reserve_vram = reserve_vram
        self._comfy_args = comfy_args
        self._headroom_mirror = headroom_mirror
        self._device_map = device_map
        self._sessions = tuple(
            BoundarySession(
                registry,
                role="isolated worker",
                pack=m.name,
                codec=ValueCodec(registry, shm_threshold=shm_threshold, use_shm=use_shm),
                on_diagnostic=on_diagnostic,
                reservations=reservations,
                device_map=device_map,
                governor=governor,
                consumer_priority=consumer_priority,
                release_guard=release_guard,
                telemetry=telemetry,
                death_detail=lambda m=m: self._death_detail(m.name),
                artifact_authority=_artifact_authority(self._extra_env),
                produced_asset_source=resolver_from_env(self._extra_env),
                on_schema_reload=on_schema_reload,
            )
            for m in self._manifests
        )
        self.members = MappingProxyType(
            {session.pack: GroupMemberWorker(self, session) for session in self._sessions}
        )
        self._proc: asyncio.subprocess.Process | None = None
        self._process_watch: asyncio.Task[None] | None = None
        self._listeners: list[BoundaryListener] = []
        self._tmpdir: Path | None = None
        self._close_task: asyncio.Task[None] | None = None

    def _death_detail(self, pack: str) -> str:
        code = self._proc.returncode if self._proc is not None else None
        return f" (worker group {self.name!r} exited with code {code})"

    async def reap_code(self) -> int | None:
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), 1.0)
        return self._proc.returncode if self._proc is not None else None

    async def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError(f"worker group {self.name!r} already started")
        try:
            preflight_interpreter(self._python)
        except InterpreterPreflightError as exc:
            raise RuntimeError(f"worker group {self.name!r} interpreter refused: {exc}") from exc
        self._tmpdir = Path(tempfile.mkdtemp(prefix="dinkster-group-"))
        try:
            token = secrets.token_hex(32) if self._transport in ("auto", "tcp") else None
            for index, _ in enumerate(self._manifests):
                endpoint_dir = self._tmpdir / str(index)
                endpoint_dir.mkdir()
                self._listeners.append(
                    await BoundaryListener.create(
                        endpoint_dir, transport=self._transport, token=token
                    )
                )
            command = [self._python, "-m", "dinkster_workers.host"]
            for listener, manifest in zip(self._listeners, self._manifests, strict=True):
                command.extend(("--endpoint", listener.endpoint, "--manifest", str(manifest.path)))
            command.extend(
                (
                    "--shm-threshold",
                    str(self._shm_threshold),
                    "--comfy-args-json",
                    json.dumps(self._comfy_args),
                )
            )
            if self._aimdo_init or self._aimdo_arm != "off":
                command.append("--aimdo-init")
            command.extend(("--aimdo-arm", self._aimdo_arm))
            if self._reserve_vram is not None:
                command.extend(("--reserve-vram", str(self._reserve_vram)))
            for index, nbytes in sorted(self._vram_budgets.items()):
                command.extend(("--vram-budget", f"{index}={nbytes}"))
            if not self._use_shm:
                command.append("--no-shm")
            env = dict(self._extra_env)
            for listener in self._listeners:
                env.update(listener.child_env)
            self._proc = await self._launcher.launch(
                LaunchSpec(
                    command=tuple(command),
                    env=env,
                    endpoint_dir=self._tmpdir,
                    pack_root=self._manifests[0].root,
                    python=self._python,
                    use_shm=self._use_shm,
                )
            )
            connected = [listener.connected for listener in self._listeners]
            pairs = await asyncio.wait_for(asyncio.gather(*connected), self._start_timeout)
            await asyncio.wait_for(
                asyncio.gather(
                    *(
                        session.begin(reader, writer, timeout=self._start_timeout)
                        for session, (reader, writer) in zip(self._sessions, pairs, strict=True)
                    )
                ),
                self._start_timeout,
            )
            process = self._proc

            async def watch_process() -> None:
                await process.wait()
                for session in self._sessions:
                    session.process_died()

            self._process_watch = asyncio.create_task(watch_process())
            if self._headroom_mirror is not None and (
                self._reserve_vram is not None or self._aimdo_arm != "off"
            ):
                for session in self._sessions:
                    await self._headroom_mirror.register(session, self._device_map)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _await_close(self._close_task)

    async def _close(self) -> None:
        if self._headroom_mirror is not None:
            for session in self._sessions:
                self._headroom_mirror.deregister(session)

        async def shutdown(session: BoundarySession) -> None:
            if session.alive:
                with contextlib.suppress(Exception):
                    await session.send({"type": "shutdown"}, [])

        await asyncio.gather(*(shutdown(session) for session in self._sessions))
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), _CHILD_SHUTDOWN_GRACE)
            if self._proc.returncode is None:
                self._proc.kill()
                await self._proc.wait()
        if self._proc is not None:
            for session in self._sessions:
                session.process_died()
        if self._process_watch is not None:
            await self._process_watch
            self._process_watch = None
        await asyncio.gather(*(session.close() for session in self._sessions))
        await asyncio.gather(*(listener.close() for listener in self._listeners))
        self._listeners.clear()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
