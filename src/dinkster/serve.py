"""dinkster-serve: run a Dinkster server from a composed node-pack set.

The umbrella package owns this wiring (server + workers + caches + pack
composition); dinkster-server itself stays policy/protocol-only and never picks
concrete workers or caches for you. Each ``--pack`` runs out-of-process as an
IsolatedWorker by default. Trusted installed first-party packs may run in-process
in the shared Dinkster environment.

Startup is PROGRESSIVE: the port binds on a zero-node diagnostic surface,
then installed catalogs are announced without starting their workers, and
each announcement grows /api/nodes, bumps the surface epoch, and emits
{"type": "schema_changed", "epoch": N} - clients interact immediately
instead of waiting for every pack import (the ComfyUI startup-delta
failure). /api/health narrates in-flight composition ("composition":
{"done", "total", "phase"}) for the supervisor to mirror.

Pack failures are RECORDED, not fatal (unless --strict-packs): a pack
that fails to compose - bad manifest, worker death, schema refusal,
namespace collision - lands as {"state": "failed", "error"} on
/api/composition plus a non-droppable pack_failed event, and every pack
that did load serves. Never silently degraded: the error is cached for
the process lifetime so clients can correlate missing node types with
"failed to load: <why>" instead of "unknown node" (the ComfyUI failure
this fixes: extensions that broke at import just... weren't there). The
installed default packs follow the same failure-isolated path as every other pack.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import ipaddress
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import tomllib
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from aiohttp import web
from dinkster_assets import (
    AssetVault,
    LibraryStore,
    MountDef,
    MountsError,
    MountTable,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    ResolverIndexError,
    ResolverSubscriptionStore,
    load_mounts,
    load_output_mount,
    require_region,
)
from dinkster_assets.resolution import ResolutionStore
from dinkster_caches import DEFAULT_DISK_CACHE_BYTES, BudgetedDiskCAS
from dinkster_inference import (
    OpenAICompatibility,
    OpenAIGenerationProvider,
    load_model_output_profile,
)
from dinkster_inference.devices import nvidia_compute_dtypes
from dinkster_memory import (
    BudgetsError,
    GovernorReservationService,
    MemoryGovernor,
    ReportedTelemetry,
    load_budgets,
    parse_size,
)
from dinkster_p2p import default_p2p_settings
from dinkster_protocol import AttentionPolicy, validate_attention_policy
from dinkster_schema import (
    LOG_LEVEL_ENV,
    LOG_OVERRIDES_ENV,
    canonical_name,
    configure_logging,
    core_logger,
    validate_name,
)
from dinkster_server import (
    SETTINGS_CATEGORIES,
    STATE_KEY,
    AuthError,
    CompositeAuthenticator,
    ExecutionJournal,
    HistoryStore,
    JournalStore,
    PathRedactor,
    PrincipalPermissionStore,
    RuntimeSettings,
    ServerLibrary,
    SettingsError,
    SettingsSource,
    TokenAuthenticator,
    TrainingSessionStore,
    comfy_dtype_args,
    create_app,
    load_authenticator,
    load_settings,
    principal_for,
    resolve_scope,
    validate_comfy_args,
)
from dinkster_server.image_document import InvalidDocument, validate_document
from dinkster_workers import (
    PackManifest,
    SandboxPolicy,
    SingleJobMultiGpuConfig,
    ensure_pack_venv,
    load_manifest,
    normalize_egress_origin,
    resolve_accelerator,
)
from dinkster_workers.catalog import read_catalog
from dinkster_workers.doctor import prepare_catalog

from .activation import add_activation_routes
from .benchmark import (
    BenchmarkAssembler,
    HardwareSampler,
    instrument_engine_factory,
    write_record,
)
from .comfy_compose import comfy_compat_specs, comfy_model_roots, comfy_python
from .compat_api import add_comfy_compat_routes
from .compose import (
    CompositionError,
    PackDelta,
    PackSpec,
    RemoveResult,
    ServingComposer,
    default_pack_ids,
    default_pack_spec,
    model_pack_specs,
    resolve_manifest_path,
    training_pack_specs,
)
from .frontend import install_frontend
from .generation_api import GenerationModel, GenerationService, add_generation_routes
from .guess_api import add_guess_routes
from .installer import Installer
from .lan_p2p import LanP2PController
from .mounts_api import MountService, add_mount_routes
from .native_policy import NativeDispatchPolicy, NativePolicyDiagnostic
from .p2p_api import add_p2p_routes
from .reload_api import add_reload_routes, apply_reload
from .remote_reconnect import RemoteReconnectSupervisor
from .remotes import RemotesError, RemoteSpec, load_remotes
from .resolver_api import add_resolver_index_routes
from .storelock import hold_lock
from .watch import PackWatcher

_PERMISSIVE_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_SERVING_PYTHON_ENV = "DINKSTER_SERVING_PYTHON"
_PACK_VENV_LOCK_TIMEOUT = 600.0
_PACK_HOST_WORKSPACE_PACKAGES = (
    "dinkster-api",
    "dinkster-assets",
    "dinkster-caches",
    "dinkster-image-document",
    "dinkster-inference",
    "dinkster-inference-torch",
    "dinkster-memory",
    "dinkster-protocol",
    "dinkster-schema",
    "dinkster-values",
    "dinkster-video",
    "dinkster-workers",
)


def _validate_collaboration_snapshot(
    document_kind: str, document_id: str, snapshot: object
) -> str | None:
    if document_kind == "workflow":
        return None
    try:
        validate_document(snapshot)
    except InvalidDocument as error:
        return f"snapshot is not a valid ImageDocument: {error}"
    if not isinstance(snapshot, dict) or snapshot.get("lineage") != document_id:
        return "ImageDocument lineage must match documentId"
    return None


def _add_collaboration_routes(app: web.Application, database: Path | None) -> bool:
    """Ask the optional collaboration package to register its server extension."""
    try:
        collab = importlib.import_module("dinkster_collab")
    except ModuleNotFoundError as error:
        if error.name != "dinkster_collab":
            raise
        return False

    collab.install_session_extension(
        app,
        database=database,
        snapshot_validator=_validate_collaboration_snapshot,
        principal_for=principal_for,
        resolve_scope=resolve_scope,
    )
    return True


def _default_pack_venv_root(library_root: str) -> Path:
    if library_root:
        return Path(library_root) / "venvs"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "dinkster" / "venvs"


def _pack_runtime_sources(manifest: PackManifest) -> tuple[tuple[Path, ...], str]:
    manifest_root = manifest.root
    source_root = manifest_root
    if (
        not (source_root / "pyproject.toml").is_file()
        and (source_root.parent / "pyproject.toml").is_file()
    ):
        source_root = source_root.parent
    packages = source_root.parent
    if (
        not (source_root / "pyproject.toml").is_file()
        or not (packages.parent / "pyproject.toml").is_file()
    ):
        module = manifest.nodes_entry.partition(":")[0].partition(".")[0]
        if not (manifest_root / module).is_dir():
            raise CompositionError(
                f"installed pack {manifest.name!r} does not bundle its runtime module"
            )
        return (), str(manifest_root)
    workspace = tuple(packages / name for name in _PACK_HOST_WORKSPACE_PACKAGES)
    missing = tuple(path.name for path in workspace if not (path / "pyproject.toml").is_file())
    if missing:
        raise CompositionError(
            f"source workspace is missing pack-host packages: {', '.join(missing)}"
        )
    pythonpath = os.pathsep.join(str(path / "src") for path in (*workspace, source_root))
    return workspace, pythonpath


def _prepare_default_pack(
    spec: PackSpec,
    *,
    venv_root: Path,
    accelerator: str,
) -> PackSpec:
    """Attach a complete interpreter to an isolated installed default pack."""
    if spec.in_process or spec.python is not None:
        return spec
    manifest = load_manifest(resolve_manifest_path(spec.manifest))
    workspace, pythonpath = _pack_runtime_sources(manifest)
    environment = dict(spec.env)
    if pythonpath:
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = pythonpath + (os.pathsep + inherited if inherited else "")
    if configured := os.environ.get(_SERVING_PYTHON_ENV):
        return replace(spec, python=configured, env=environment)
    if spec.packs is None or len(spec.packs) != 1:
        raise CompositionError("an installed default pack must carry exactly one provenance entry")
    info = next(iter(spec.packs.values()))
    algorithm, separator, digest = info.artifact_digest.partition(":")
    if separator != ":" or algorithm != "sha256" or len(digest) != 64:
        raise CompositionError(
            f"installed default pack {manifest.name!r} has invalid artifact digest"
        )
    pack_venv_root = venv_root / accelerator / digest
    with hold_lock(venv_root / ".lock", _PACK_VENV_LOCK_TIMEOUT):
        python = ensure_pack_venv(
            manifest,
            venv_root=pack_venv_root,
            workspace_packages=workspace,
            accelerator=accelerator,
        )
    return replace(spec, python=str(python), env=environment)


def _is_standard_vision_pack(spec: PackSpec) -> bool:
    return (
        spec.packs is not None
        and len(spec.packs) == 1
        and next(iter(spec.packs)).startswith("dinkster-vision-")
    )


def detect_native_compute_dtypes(executing_cuda_indices: Sequence[int] = ()) -> frozenset[str]:
    """Compute dtypes supported by every NVIDIA device that executes native jobs.

    Each device's capability comes from driver-reported facts through the
    pinned-reference gates (should_use_fp16 / should_use_bf16 @ b78cec87);
    the answer is the intersection across the executing devices, because
    replica lanes are interchangeable - a selected dtype must run on
    whichever lane takes the job. ``executing_cuda_indices`` index into the
    CUDA_VISIBLE_DEVICES pool exactly as replica lane launch does; empty
    means the default single worker on the pool's first device. Without an
    explicit CUDA_VISIBLE_DEVICES list the CUDA runtime's device order is
    unknowable here (CUDA_DEVICE_ORDER defaults to fastest-first while
    nvidia-smi reports PCI bus order), so a multi-GPU host answers with the
    intersection across all devices. Probe failure or an unclassifiable
    device selector keeps the permissive answer: absence of nvidia-smi does
    not identify the accelerator, and refusing dtypes on a guess would
    wrongly slow non-NVIDIA hosts.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,compute_cap,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        windows = any(platform.win32_ver())
        indexed: list[tuple[str, str, frozenset[str]]] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, uuid, capability, name = (part.strip() for part in line.split(",", 3))
            major = int(capability.split(".", 1)[0])
            indexed.append((index, uuid, nvidia_compute_dtypes(major, name, windows=windows)))
        if not indexed:
            return _PERMISSIVE_COMPUTE_DTYPES
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
        return _PERMISSIVE_COMPUTE_DTYPES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if len(indexed) > 1:
            return frozenset.intersection(*(capabilities for _, _, capabilities in indexed))
        pool = [index for index, _, _ in indexed]
    else:
        pool = [part.strip() for part in visible.split(",")]
    by_selector: dict[str, frozenset[str]] = {}
    for index, uuid, capabilities in indexed:
        by_selector[index] = capabilities
        by_selector[uuid] = capabilities
    executing: list[frozenset[str]] = []
    for position in executing_cuda_indices or (0,):
        if position < 0 or position >= len(pool):
            return _PERMISSIVE_COMPUTE_DTYPES
        capabilities = by_selector.get(pool[position])
        if capabilities is None:
            return _PERMISSIVE_COMPUTE_DTYPES
        executing.append(capabilities)
    return frozenset.intersection(*executing)


def detect_native_runtime_versions(interpreter: str) -> Mapping[str, str]:
    """Read behavior-bearing package versions from the native worker interpreter."""

    script = """
import json
from importlib.metadata import PackageNotFoundError, version
result = {}
for name in ("torch", "dinkster-kitchen"):
    try:
        result[name] = version(name)
    except PackageNotFoundError:
        pass
print(json.dumps(result, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [interpreter, "-I", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not inspect native runtime package versions") from error
    if not isinstance(raw, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) or not value
        for name, value in raw.items()
    ):
        raise RuntimeError("native runtime package versions are malformed")
    return raw


#: Derived compat mounts: when --comfy-root is configured, the install's
#: own directories become conventional, well-known mounts so imported
#: workflows have a stable target ("comfy-input" is where a legacy
#: LoadImage filename resolves; "comfy-output" is where a legacy
#: filename_prefix lands). Derived per boot, never persisted to
#: mounts.toml, and skipped when the operator configured the id herself
#: (explicit beats derived).
_COMFY_MOUNTS: tuple[tuple[str, str, str], ...] = (
    ("comfy-input", "input", "read"),
    ("comfy-output", "output", "readwrite"),
)

_OPENAI_API_KEY_ENV = "DINKSTER_OPENAI_API_KEY"
_OPENAI_BASE_URL_ENV = "DINKSTER_OPENAI_BASE_URL"
_OPENAI_MODEL_ENV = "DINKSTER_OPENAI_MODEL"
_OPENAI_COMPATIBILITY_ENV = "DINKSTER_OPENAI_COMPATIBILITY"
_OPENAI_STREAM_ENV = "DINKSTER_OPENAI_STREAM"
_OPENAI_TIMEOUT_ENV = "DINKSTER_OPENAI_TIMEOUT"
_REMOTE_CATALOG_BASE_ENV = "DINKSTER_REMOTE_CATALOG_BASE"
_REMOTE_GATEWAY_BASE_ENV = "DINKSTER_REMOTE_GATEWAY_BASE"
_REMOTE_AUTH_TOKEN_FILE_ENV = "DINKSTER_REMOTE_AUTH_TOKEN_FILE"
_REMOTE_CATALOG_POLL_INTERVAL_ENV = "DINKSTER_REMOTE_CATALOG_POLL_INTERVAL"
_IDENTITY_JWKS_URL_ENV = "DINKSTER_IDENTITY_JWKS_URL"
_IDENTITY_ISSUER_ENV = "DINKSTER_IDENTITY_ISSUER"
_IDENTITY_AUDIENCE_ENV = "DINKSTER_IDENTITY_AUDIENCE"
_DEFAULT_REMOTE_GATEWAY_BASE = ""
_FEDERATED_ASSET_PATHS = {
    "catalog": "/api/catalog",
    "candidates": "/api/catalog/candidates",
}
_PROVIDER_ID = re.compile(r"[a-z0-9][a-z0-9._/-]*")


def _is_loopback_bind(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _with_remote_config(
    spec: PackSpec,
    *,
    catalog_base: str | None = None,
    gateway_base: str | None = None,
    poll_interval: float | None = None,
    token_file: str | None = None,
) -> PackSpec:
    if catalog_base is None:
        catalog_base = os.environ.get(_REMOTE_CATALOG_BASE_ENV, _DEFAULT_REMOTE_GATEWAY_BASE)
    if gateway_base is None:
        gateway_base = os.environ.get(_REMOTE_GATEWAY_BASE_ENV, "")
    gateway_base = gateway_base or catalog_base
    if poll_interval is None:
        poll_interval = float(os.environ.get(_REMOTE_CATALOG_POLL_INTERVAL_ENV, "3600"))
    if token_file is None:
        token_file = os.environ.get(_REMOTE_AUTH_TOKEN_FILE_ENV, "")
    environment = {
        _REMOTE_CATALOG_BASE_ENV: catalog_base,
        _REMOTE_GATEWAY_BASE_ENV: gateway_base,
        _REMOTE_CATALOG_POLL_INTERVAL_ENV: format(poll_interval, ".17g"),
    }
    if token_file:
        environment[_REMOTE_AUTH_TOKEN_FILE_ENV] = token_file
    return replace(
        spec,
        env=environment,
        execution_config={"catalog_base": catalog_base, "gateway_base": gateway_base},
    )


def _load_federated_asset_config(
    store_path: str | None,
    policy_path: str | None,
    cursor_key_path: str | None,
    *,
    auth_enabled: bool,
) -> tuple[Path, dict[str, frozenset[str]], bytes] | None:
    configured = (store_path, policy_path, cursor_key_path)
    if all(value is None for value in configured):
        return None
    if any(value is None for value in configured):
        raise ValueError(
            "--federated-assets-store, --federated-assets-policy, and "
            "--federated-assets-cursor-key must be provided together"
        )
    assert store_path is not None and policy_path is not None and cursor_key_path is not None
    with Path(policy_path).open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) != {"version", "scopes"}:
        raise ValueError("federated asset policy must contain exactly version and scopes")
    version = document["version"]
    if type(version) is not int or version != 1:
        raise ValueError("federated asset policy version must be the integer 1")
    scopes = document["scopes"]
    if not isinstance(scopes, dict) or not scopes:
        raise ValueError("federated asset policy scopes must be a non-empty table")
    policy: dict[str, frozenset[str]] = {}
    for scope, providers in scopes.items():
        if not isinstance(scope, str) or not scope or any(char.isspace() for char in scope):
            raise ValueError(
                "federated asset policy scope names must be nonempty and whitespace-free"
            )
        if not isinstance(providers, list):
            raise ValueError(f"federated asset policy scope {scope!r} must contain an array")
        if any(not isinstance(provider, str) for provider in providers):
            raise ValueError(f"federated asset policy scope {scope!r} providers must be strings")
        if any(_PROVIDER_ID.fullmatch(provider) is None for provider in providers):
            raise ValueError(
                f"federated asset policy scope {scope!r} has a noncanonical provider id"
            )
        if len(set(providers)) != len(providers):
            raise ValueError(f"federated asset policy scope {scope!r} has duplicate provider ids")
        policy[scope] = frozenset(providers)
    if not auth_enabled and "local" not in policy:
        raise ValueError("federated asset policy requires a local scope when auth is absent")
    cursor_key = Path(cursor_key_path).read_bytes()
    if len(cursor_key) < 32:
        raise ValueError("federated asset cursor key must contain at least 32 bytes")
    return Path(store_path), policy, cursor_key


def _native_asset_locator(vault: AssetVault, mounts: MountTable) -> Callable[[str], Path | None]:
    """Resolve native probe inputs from verified storage before mount hints."""

    def locate(digest: str) -> Path | None:
        return vault.resolve(digest) or mounts.resolve(digest)

    return locate


async def _schedule_legacy_checkpoint_conversion(
    composer: ServingComposer, path: Path, logical_name: str
) -> tuple[str, str | None]:
    """Send conversion only to a live worker that advertised the optional RPC."""
    async with composer._mutate:
        for owner in tuple(composer.composition._isolated):
            members = getattr(owner, "members", None)
            candidates = (
                tuple(cast("Mapping[str, Any]", members).values())
                if isinstance(members, Mapping)
                else (owner,)
            )
            for worker in candidates:
                try:
                    if not getattr(worker, "can_convert_legacy_checkpoint", False):
                        continue
                    outcome = await worker.convert_legacy_checkpoint(path, logical_name)
                except Exception:  # noqa: BLE001 - worker/session loss is retryable transport
                    continue
                if outcome is not None:
                    return cast("tuple[str, str | None]", outcome)
        return "transport-failure", None


def _log_native_policy_diagnostic(diagnostic: NativePolicyDiagnostic) -> None:
    core_logger("serve.native").info(
        "native checkpoint %s for %s: %s",
        diagnostic.kind,
        diagnostic.digest,
        "; ".join(diagnostic.reasons),
    )


def parse_memory_budget(entry: str) -> tuple[str, int]:
    """Parse one ``--memory-budget DEVICE=SIZE`` entry.

    DEVICE is a residency-class key (``ram``, ``vram:cuda:0``); SIZE is a
    byte count with an optional binary suffix K/M/G/T (case-insensitive),
    e.g. ``ram=8G``. Raises ValueError with a human message on anything
    malformed - the caller turns it into a parser error.
    """
    device, sep, size = entry.partition("=")
    device = device.strip()
    size = size.strip().lower()
    if not sep or not device or not size:
        raise ValueError(f"expected DEVICE=SIZE, got {entry!r}")
    try:
        nbytes = parse_size(size)
    except ValueError:
        raise ValueError(
            f"budget for {device!r} must be a byte count with an optional "
            f"K/M/G/T suffix, got {size!r}"
        ) from None
    return device, nbytes


def merge_remote_budgets(budgets: dict[str, int], remotes: Sequence[RemoteSpec]) -> None:
    """Fold each remote's own-namespace budgets into the effective budgets
    under the remote's ``@<name>`` qualifier - the namespace its qualified
    device facts land on. Callers apply persisted and CLI entries
    afterwards, so those win on collision (CLI beats config)."""
    for spec in remotes:
        for device, nbytes in spec.memory_budgets.items():
            budgets[f"{device}@{spec.name}"] = nbytes


def parse_cuda_devices(value: str) -> tuple[int, ...]:
    """Parse an explicit ordered CUDA device set for native jobs."""
    parts = value.split(",")
    try:
        indices = tuple(int(part) for part in parts)
    except ValueError:
        indices = ()
    if (
        len(indices) < 2
        or any(str(index) != part or index < 0 for index, part in zip(indices, parts, strict=True))
        or len(set(indices)) != len(indices)
    ):
        raise argparse.ArgumentTypeError(
            "CUDA devices must be at least two unique comma-separated non-negative indices"
        )
    return indices


def parse_positive_float(value: str) -> float:
    """Parse one finite positive float for an argparse option."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_positive_int(value: str) -> int:
    """Parse one positive integer for an argparse option."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_positive_size(value: str) -> int:
    """Parse one positive byte size for an argparse option."""
    try:
        parsed = parse_size(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a byte count with an optional K/M/G/T suffix"
        ) from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_resolver_region(value: str) -> str:
    """Parse an optional resolver mirror region."""
    if not value:
        return ""
    try:
        return require_region(value)
    except ResolverIndexError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def openai_response_mode_from_env(value: str) -> str:
    """Normalize the response-mode environment value without silent fallbacks."""
    normalized = value.strip().lower()
    if normalized in ("stream", "1", "true", "yes", "on"):
        return "stream"
    if normalized in ("json", "0", "false", "no", "off"):
        return "json"
    raise ValueError(f"{_OPENAI_STREAM_ENV} must be stream/json or true/false")


def normalize_settings_categories(values: list[str] | None) -> frozenset[str]:
    """Normalize repeatable optional gate values into the granted set."""
    if not values:
        return frozenset()
    if "all" in values:
        return frozenset(SETTINGS_CATEGORIES)
    unknown = set(values) - set(SETTINGS_CATEGORIES)
    if unknown:
        bad = sorted(unknown)[0]
        raise ValueError(
            f"unknown settings category {bad!r}; valid categories: {', '.join(SETTINGS_CATEGORIES)}"
        )
    return frozenset(values)


def _spec_label(entry: PackSpec | str) -> str:
    """Best-effort report key for a spec: its manifest's declared pack
    name, else the manifest path - a manifest that cannot even parse
    still needs a row to hang its failure record on."""
    spec = entry if isinstance(entry, PackSpec) else PackSpec(manifest=entry)
    try:
        return load_manifest(resolve_manifest_path(spec.manifest)).name
    except Exception:
        return str(spec.manifest)


def _install_event_loop_stall_diagnostics(
    app: web.Application, *, threshold: float, logger: logging.Logger
) -> None:
    """Report safe request timing and Python stacks when the loop stops advancing."""

    @web.middleware
    async def time_request(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        route = request.match_info.route
        route_resource = route.resource if route is not None else None
        resource = route_resource.canonical if route_resource is not None else "<unmatched>"
        started = time.monotonic()
        logger.info("request started: %s %s", request.method, resource)
        try:
            response = await handler(request)
        except BaseException:
            logger.info(
                "request failed: %s %s in %.3f seconds",
                request.method,
                resource,
                time.monotonic() - started,
            )
            raise
        else:
            logger.info(
                "request finished: %s %s in %.3f seconds",
                request.method,
                resource,
                time.monotonic() - started,
            )
            return response

    monitor: asyncio.Task[None] | None = None
    watcher: threading.Thread | None = None
    stop_watcher = threading.Event()
    loop_thread_id: int | None = None
    last_heartbeat = time.monotonic()

    async def heartbeat() -> None:
        nonlocal last_heartbeat
        while True:
            last_heartbeat = time.monotonic()
            await asyncio.sleep(min(threshold / 4, 0.25))

    def watch() -> None:
        reported = False
        while not stop_watcher.wait(min(threshold / 4, 0.25)):
            lag = time.monotonic() - last_heartbeat
            if lag < threshold:
                reported = False
                continue
            if reported:
                continue
            thread_id = loop_thread_id
            frame = None if thread_id is None else sys._current_frames().get(thread_id)
            stack = "" if frame is None else "".join(traceback.format_stack(frame))
            logger.warning("event loop unresponsive for %.3f seconds\n%s", lag, stack)
            reported = True

    async def start(_: web.Application) -> None:
        nonlocal loop_thread_id, monitor, watcher
        loop_thread_id = threading.get_ident()
        monitor = asyncio.create_task(heartbeat())
        watcher = threading.Thread(target=watch, name="dinkster-event-loop-watch", daemon=True)
        watcher.start()
        logger.info("event-loop stall diagnostics armed at %.3f seconds", threshold)

    async def stop(_: web.Application) -> None:
        stop_watcher.set()
        if monitor is not None:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        if watcher is not None:
            watcher.join()

    app.middlewares.append(time_request)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)


def _resolve_pack_argument(value: str) -> Path:
    """Freeze a --pack path before any worker or asynchronous startup work."""
    return Path(value).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a Dinkster server from its installed defaults plus configured packs"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3639)
    parser.add_argument("--frontend-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--frontend-dev", default="", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-stale-catalogs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="HOST",
        help="accept this HTTP Host in addition to the bind address and loopback IPs (repeatable)",
    )
    parser.add_argument(
        "--no-default-packs",
        action="store_true",
        help="serve only the managed install root and explicitly configured packs",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        type=_resolve_pack_argument,
        metavar="PATH",
        help="pack to serve: a dinkster-pack.toml or its directory, repeatable; "
        "relative paths resolve from the launch directory; each pack runs "
        "isolated in its own process",
    )
    parser.add_argument(
        "--install-root",
        default=os.environ.get("DINKSTER_INSTALL_ROOT", ""),
        metavar="PATH",
        help="managed install root (dinkster-pack): serve the packs of its "
        "current generation from the immutable store "
        "(default: $DINKSTER_INSTALL_ROOT)",
    )
    parser.add_argument(
        "--comfy-root",
        default=os.environ.get("DINKSTER_COMFYUI_ROOT", ""),
        metavar="PATH",
        help="optional ComfyUI install for untranslated core and legacy packs; "
        "native execution and import schemas need no checkout "
        "(default: $DINKSTER_COMFYUI_ROOT)",
    )
    parser.add_argument(
        "--comfy-python",
        default="",
        metavar="PATH",
        help="interpreter for native or compat execution "
        "(default: $DINKSTER_COMFYUI_PYTHON, the optional ComfyUI venv, or current Python)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get(_OPENAI_BASE_URL_ENV, ""),
        metavar="URL",
        help="OpenAI-compatible /v1 base URL (default: $DINKSTER_OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--remote-catalog-base",
        default=os.environ.get(_REMOTE_CATALOG_BASE_ENV, _DEFAULT_REMOTE_GATEWAY_BASE),
        metavar="URL",
        help=(
            "remote node catalog base URL "
            "(default: $DINKSTER_REMOTE_CATALOG_BASE; disabled when unset)"
        ),
    )
    parser.add_argument(
        "--remote-gateway-base",
        default=os.environ.get(_REMOTE_GATEWAY_BASE_ENV, ""),
        metavar="URL",
        help="remote job gateway base URL (default: catalog base URL)",
    )
    parser.add_argument(
        "--remote-auth-token-file",
        default=os.environ.get(_REMOTE_AUTH_TOKEN_FILE_ENV, ""),
        metavar="PATH",
        help="rotatable remote session bearer token file",
    )
    parser.add_argument(
        "--remote-catalog-poll-interval",
        type=parse_positive_float,
        default=os.environ.get(_REMOTE_CATALOG_POLL_INTERVAL_ENV, "3600"),
        metavar="SECONDS",
        help="remote catalog epoch polling interval (default: 3600)",
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get(_OPENAI_MODEL_ENV, ""),
        metavar="MODEL",
        help="remote generation model id (default: $DINKSTER_OPENAI_MODEL)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get(_OPENAI_API_KEY_ENV, ""),
        metavar="KEY",
        help="optional bearer credential (default: $DINKSTER_OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-compatibility",
        choices=("openai", "llama.cpp"),
        default=os.environ.get(_OPENAI_COMPATIBILITY_ENV, "openai"),
        help="external endpoint sampling dialect (default: openai)",
    )
    try:
        openai_response_mode = openai_response_mode_from_env(
            os.environ.get(_OPENAI_STREAM_ENV, "true")
        )
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument(
        "--openai-response-mode",
        choices=("stream", "json"),
        default=openai_response_mode,
        help="request SSE streaming or one JSON response (default: stream)",
    )
    parser.add_argument(
        "--openai-timeout",
        type=parse_positive_float,
        default=os.environ.get(_OPENAI_TIMEOUT_ENV, "300"),
        metavar="SECONDS",
        help="external generation request timeout (default: 300)",
    )
    parser.add_argument(
        "--comfy-arg",
        action="append",
        nargs="?",
        const=None,
        default=None,
        metavar="ARG",
        help="ComfyUI startup argument for compat workers, repeatable; use "
        "--comfy-arg=--flag when ARG begins with '--'; a bare flag clears "
        "persisted arguments",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--multi-gpu-devices",
        type=parse_cuda_devices,
        default=(),
        metavar="INDEX,INDEX",
        help="run native model jobs across an ordered set of at least two CUDA devices",
    )
    gpu_group.add_argument(
        "--single-job-multi-gpu-devices",
        type=parse_cuda_devices,
        default=(),
        metavar="INDEX,INDEX",
        help="fixed ordered logical CUDA ranks for single-job execution",
    )
    parser.add_argument(
        "--single-job-multi-gpu-mode",
        choices=("auto", "guidance", "sequence", "window"),
        default="auto",
        help="single-job rank recipe selection (default: auto)",
    )
    parser.add_argument(
        "--aimdo",
        choices=("auto", "on", "off"),
        default=None,
        help="weight residency mechanism for native execution: auto enables "
        "dinkster-aimdo where upstream ComfyUI would (NVIDIA CUDA, non-WSL), "
        "on enables dinkster-aimdo wherever the capability chain passes, "
        "off disables dynamic residency (default: auto)",
    )
    parser.add_argument(
        "--fp8-matmul",
        action="store_true",
        default=None,
        help="opt into fp8 matrix multiplication for native checkpoints "
        "(default: persisted fp8-matmul setting, else off)",
    )
    parser.add_argument(
        "--diffusion-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="diffusion model compute dtype (default: auto)",
    )
    parser.add_argument(
        "--text-encoder-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="text encoder compute dtype (default: auto)",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="VAE compute dtype (default: auto)",
    )
    parser.add_argument(
        "--reserve-vram",
        type=parse_size,
        default=None,
        metavar="BYTES",
        help="physical accelerator headroom in bytes with an optional "
        "binary K/M/G/T suffix (default: 256M)",
    )
    parser.add_argument(
        "--legacy-pack",
        action="append",
        default=[],
        type=_resolve_pack_argument,
        metavar="PATH",
        help="unmodified ComfyUI custom node pack (directory or single .py), "
        "repeatable; relative paths resolve from the launch directory; loads "
        "in the legacy quarantine worker and attributes as 'comfy.<pack>' "
        "(requires --comfy-root)",
    )
    parser.add_argument(
        "--strict-packs",
        action="store_true",
        help="a pack that fails to compose aborts the whole process "
        "(default: record the failure on /api/composition, emit "
        "pack_failed, and serve every pack that loaded)",
    )
    parser.add_argument(
        "--sandbox-packs",
        action="store_true",
        help="run isolated pack workers in a fail-closed Linux bubblewrap jail",
    )
    parser.add_argument(
        "--sandbox-grant-gpu",
        action="append",
        default=[],
        metavar="PACK",
        help="allow a pack whose manifest requests GPU access to receive it; repeatable",
    )
    parser.add_argument(
        "--sandbox-grant-network",
        action="append",
        default=[],
        metavar="PACK=HTTPS_ORIGIN",
        help="allow a pack whose manifest requests network access to reach one exact HTTPS "
        "origin through the host proxy; repeatable",
    )
    parser.add_argument(
        "--max-running-jobs",
        type=int,
        default=None,
        help="concurrent jobs (engine admission keeps hardware safe either way)",
    )
    parser.add_argument(
        "--watch-packs",
        action="store_true",
        help="hot-reload node packs on source changes: poll every composed "
        "pack's source directory and, when its files change and settle, "
        "restart that pack's worker and swap its nodes on the live surface "
        "- enables development diagnostics and the same swap and failure "
        "semantics as POST /api/packs/{packId}/reload",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="write one benchmark record per job (DESIGN 3.9): per-occurrence "
        "timings, cache hit/miss attribution, boundary costs, and a "
        "hardware sample timeline (system memory and NVML when installed); implies "
        "cache-miss explanations, adds nothing to the hot path otherwise",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmarks",
        metavar="PATH",
        help="directory for benchmark records (default: ./benchmarks)",
    )
    parser.add_argument(
        "--benchmark-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="hardware sampling interval for --benchmark (default: 0.5)",
    )
    parser.add_argument(
        "--library-root",
        default=os.environ.get("DINKSTER_LIBRARY_ROOT", "dinkster-library"),
        metavar="PATH",
        help="persistence root: uploaded bytes land in PATH/vault "
        "(content-addressed), scoped library records in "
        "PATH/library.sqlite, persistent execution history in "
        "PATH/history.sqlite, and execution-run logs in "
        "PATH/execution.sqlite; pass '' to disable "
        "(default: $DINKSTER_LIBRARY_ROOT, else ./dinkster-library)",
    )
    parser.add_argument(
        "--execution-cache-mode",
        choices=("memory", "layered"),
        default=None,
        help="execution result cache: memory only, or memory in front of persistent disk "
        "(default: layered when a cache directory is available, else memory)",
    )
    parser.add_argument(
        "--execution-cache-memory-entries",
        type=parse_positive_int,
        default=1024,
        metavar="N",
        help="maximum entries in the execution cache memory layer (default: 1024)",
    )
    parser.add_argument(
        "--execution-cache-dir",
        default=None,
        metavar="PATH",
        help="persistent execution cache directory (default: <library-root>/execution-cache)",
    )
    parser.add_argument(
        "--execution-cache-disk-budget",
        type=parse_positive_size,
        default=DEFAULT_DISK_CACHE_BYTES,
        metavar="BYTES",
        help="maximum persistent execution cache bytes, with an optional binary suffix "
        "(default: 10G)",
    )
    parser.add_argument(
        "--resolver-region",
        type=parse_resolver_region,
        default=os.environ.get("DINKSTER_RESOLVER_REGION", ""),
        metavar="REGION",
        help="prefer resolver-index mirrors for this region (default: $DINKSTER_RESOLVER_REGION)",
    )
    parser.add_argument(
        "--official-resolver-url",
        default=os.environ.get("DINKSTER_OFFICIAL_RESOLVER_URL"),
        metavar="URL",
        help=(
            "official provider export URL (or $DINKSTER_OFFICIAL_RESOLVER_URL; no built-in default)"
        ),
    )
    parser.add_argument(
        "--official-resolver-provider-id",
        default=os.environ.get("DINKSTER_OFFICIAL_RESOLVER_PROVIDER_ID"),
        metavar="ID",
        help="official export identity "
        "(or $DINKSTER_OFFICIAL_RESOLVER_PROVIDER_ID; no built-in default)",
    )
    parser.add_argument(
        "--disable-p2p",
        action="store_true",
        help="start with P2P downloading and seeding disabled without changing saved settings",
    )
    parser.add_argument(
        "--execution-log-runs",
        type=int,
        default=200,
        metavar="N",
        help="keep the execution logs (journal replay behind "
        "/api/runs/{runId}/journal) of the last N finished runs "
        "(default: 200)",
    )
    parser.add_argument(
        "--execution-log-days",
        type=float,
        default=30.0,
        metavar="DAYS",
        help="drop execution logs of runs that finished more than DAYS ago (default: 30)",
    )
    parser.add_argument(
        "--allow-mount-changes",
        action="store_true",
        help="enable POST/DELETE /api/mounts: grant and revoke filesystem "
        "mounts while running (the desktop-shell folder-picker flow); "
        "granted mounts persist to <library-root>/mounts.toml. Off by "
        "default: on a remotely reachable server, runtime mount mutation "
        "is a filesystem capability grant and stays operator-only",
    )
    parser.add_argument(
        "--allow-settings-changes",
        action="append",
        nargs="?",
        const="all",
        default=None,
        metavar="CATEGORY",
        help="allow runtime settings mutation for CATEGORY; repeatable, and a bare "
        "flag or 'all' grants every settings category",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="enable CORS for this browser origin (repeatable; '*' allows "
        "any); default: no CORS headers, so pages from other origins "
        "cannot script this server",
    )
    parser.add_argument(
        "--auth",
        metavar="PATH",
        help="accept static Bearer credentials from PATH for inbound API access "
        "(default: off; <library-root>/auth.toml is the conventional location)",
    )
    parser.add_argument(
        "--identity-jwks-url",
        default=os.environ.get(_IDENTITY_JWKS_URL_ENV, ""),
        metavar="URL",
        help="identity-service JWKS endpoint (default: $DINKSTER_IDENTITY_JWKS_URL)",
    )
    parser.add_argument(
        "--identity-issuer",
        default=os.environ.get(_IDENTITY_ISSUER_ENV, ""),
        metavar="URL",
        help="expected identity-service token issuer (default: $DINKSTER_IDENTITY_ISSUER)",
    )
    parser.add_argument(
        "--identity-audience",
        default=os.environ.get(_IDENTITY_AUDIENCE_ENV, ""),
        metavar="AUDIENCE",
        help="expected identity-service token audience (default: $DINKSTER_IDENTITY_AUDIENCE)",
    )
    parser.add_argument(
        "--federated-assets-store",
        metavar="PATH",
        help="SQLite resolution store for the read-only federated asset catalog",
    )
    parser.add_argument(
        "--federated-assets-policy",
        metavar="PATH",
        help="strict version-1 scope-to-provider policy for federated asset reads",
    )
    parser.add_argument(
        "--federated-assets-cursor-key",
        metavar="PATH",
        help="secret raw cursor-signing key file containing at least 32 bytes",
    )
    parser.add_argument(
        "--remote-workers",
        metavar="PATH",
        help="remotes.toml naming remote worker daemons to compose at startup "
        "(default: <library-root>/remotes.toml when --library-root is set)",
    )
    parser.add_argument(
        "--advertise-assets",
        metavar="URL",
        help="asset base URL remote worker daemons can pull declared assets "
        "from when the engine stages them (bytes at URL/assets/{digest}; "
        "advertise this server's API base, e.g. http://HOST:PORT/api). "
        "Without it daemons can stage only from URLs the pack declares",
    )
    parser.add_argument(
        "--advertise-assets-token-file",
        metavar="PATH",
        help="file containing the bearer token daemons present at the "
        "advertised asset endpoint (a static auth.toml token granting "
        "assets:read); required when --auth is set and --advertise-assets "
        "points at this server",
    )
    parser.add_argument(
        "--memory-budget",
        action="append",
        default=None,
        metavar="DEVICE=SIZE",
        help="declared memory budget for one residency class, repeatable "
        "(e.g. --memory-budget ram=24G --memory-budget vram:cuda:0=20G). "
        "SIZE is bytes with an optional binary K/M/G/T suffix. Budgeted "
        "devices gate worker reservations through the memory governor "
        "and appear in /memory/status; unbudgeted devices admit freely",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="verbosity for the whole dinkster tree (debug/info/warning/error/critical)",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        metavar="NAME=LEVEL",
        help="per-origin override, repeatable (e.g. --log dinkster.pack.mypack=debug)",
    )
    parser.add_argument(
        "--event-loop-stall-threshold",
        type=parse_positive_float,
        default=None,
        metavar="SECONDS",
        help="diagnose event-loop stalls longer than SECONDS with safe request timing "
        "and Python thread stacks",
    )
    args = parser.parse_args(argv)
    if (args.official_resolver_url or args.official_resolver_provider_id) and not args.library_root:
        parser.error("official resolver bootstrap requires --library-root")
    openai_values = (args.openai_base_url, args.openai_model, args.openai_api_key)
    if any(openai_values) and not (args.openai_base_url and args.openai_model):
        parser.error("OpenAI generation requires both --openai-base-url and --openai-model")
    try:
        attention_policy: AttentionPolicy = validate_attention_policy(
            os.environ.pop("DINKSTER_ATTENTION_POLICY", "auto")
        )
    except ValueError as exc:
        parser.error(f"DINKSTER_ATTENTION_POLICY: {exc}")
    # The plain launcher inherits the host environment. Capture service
    # settings through argparse, then give them only to their owning PackSpec.
    for name in (
        _OPENAI_API_KEY_ENV,
        _OPENAI_BASE_URL_ENV,
        _OPENAI_MODEL_ENV,
        _OPENAI_COMPATIBILITY_ENV,
        _OPENAI_STREAM_ENV,
        _OPENAI_TIMEOUT_ENV,
        _REMOTE_CATALOG_BASE_ENV,
        _REMOTE_GATEWAY_BASE_ENV,
        _REMOTE_AUTH_TOKEN_FILE_ENV,
        _REMOTE_CATALOG_POLL_INTERVAL_ENV,
        _IDENTITY_JWKS_URL_ENV,
        _IDENTITY_ISSUER_ENV,
        _IDENTITY_AUDIENCE_ENV,
    ):
        os.environ.pop(name, None)
    identity_values = (
        args.identity_jwks_url,
        args.identity_issuer,
        args.identity_audience,
    )
    if any(identity_values) and not all(identity_values):
        parser.error(
            "identity token authentication requires --identity-jwks-url, "
            "--identity-issuer, and --identity-audience"
        )
    try:
        static_authenticator = load_authenticator(Path(args.auth)) if args.auth else None
    except AuthError as exc:
        raise SystemExit(str(exc)) from exc
    token_authenticator = TokenAuthenticator(*identity_values) if all(identity_values) else None
    if static_authenticator is not None and token_authenticator is not None:
        authenticator = CompositeAuthenticator(static_authenticator, token_authenticator)
    else:
        authenticator = static_authenticator or token_authenticator
    if not _is_loopback_bind(args.host) and authenticator is None:
        parser.error("a non-loopback --host requires --auth or identity token authentication")
    try:
        federated_asset_config = _load_federated_asset_config(
            args.federated_assets_store,
            args.federated_assets_policy,
            args.federated_assets_cursor_key,
            auth_enabled=authenticator is not None,
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"federated asset configuration: {exc}")
    try:
        granted_settings = normalize_settings_categories(args.allow_settings_changes)
    except ValueError as exc:
        parser.error(str(exc))

    overrides: dict[str, str] = {}
    for entry in args.log or ():
        name, sep, level = entry.partition("=")
        if not sep or not name or not level:
            parser.error(f"--log expects NAME=LEVEL, got {entry!r}")
        overrides[name.strip()] = level.strip()
    settings_path = Path(args.library_root) / "settings.json" if args.library_root else None
    cache_dir = (
        Path(args.execution_cache_dir)
        if args.execution_cache_dir is not None
        else Path(args.library_root) / "execution-cache"
        if args.library_root
        else None
    )
    cache_mode = args.execution_cache_mode or ("layered" if cache_dir is not None else "memory")
    if cache_mode == "layered" and cache_dir is None:
        parser.error(
            "--execution-cache-mode layered requires --execution-cache-dir "
            "when --library-root is disabled"
        )
    try:
        persisted_settings = load_settings(settings_path) if settings_path is not None else {}
    except SettingsError as exc:
        raise SystemExit(str(exc)) from exc

    if args.library_root:
        try:
            memory_budgets = load_budgets(Path(args.library_root) / "memory.toml")
        except BudgetsError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        memory_budgets = {}
    # Remote worker daemons to compose at startup: an explicit path, or the
    # library root's remotes.toml when present - the memory.toml convention.
    remote_workers_path: Path | None = None
    if args.remote_workers:
        remote_workers_path = Path(args.remote_workers)
        if not remote_workers_path.exists():
            parser.error(f"--remote-workers: {remote_workers_path} not found")
    elif args.library_root:
        remote_workers_path = Path(args.library_root) / "remotes.toml"
    try:
        remote_specs = load_remotes(remote_workers_path) if remote_workers_path else ()
    except RemotesError as exc:
        parser.error(str(exc))
    if args.advertise_assets and not args.advertise_assets.startswith(("http://", "https://")):
        parser.error(f"--advertise-assets must be an http(s) URL, got {args.advertise_assets!r}")
    if args.advertise_assets_token_file and not args.advertise_assets:
        parser.error("--advertise-assets-token-file requires --advertise-assets")
    remote_asset_token: str | None = None
    if args.advertise_assets_token_file:
        try:
            remote_asset_token = Path(args.advertise_assets_token_file).read_text("utf-8").strip()
        except OSError as exc:
            parser.error(f"--advertise-assets-token-file: {exc}")
        if not remote_asset_token:
            parser.error(
                f"--advertise-assets-token-file: {args.advertise_assets_token_file} is empty"
            )
    # Each remote's budgets land on its qualified devices before persisted
    # and CLI entries apply, so those still win on collision.
    merge_remote_budgets(memory_budgets, remote_specs)
    memory_source: SettingsSource = "config" if memory_budgets else "default"
    persisted_budgets = persisted_settings.get("memory-budgets")
    if persisted_budgets is not None:
        memory_budgets.update(dict(persisted_budgets))  # type: ignore[arg-type]
        memory_source = "persisted"
    for entry in args.memory_budget or ():
        try:
            device, nbytes = parse_memory_budget(entry)
        except ValueError as exc:
            parser.error(f"--memory-budget: {exc}")
        memory_budgets[device] = nbytes
    if args.memory_budget is not None:
        memory_source = "cli"

    aimdo = cast("str", persisted_settings.get("aimdo-policy", "auto"))
    aimdo_source: SettingsSource = (
        "persisted" if "aimdo-policy" in persisted_settings else "default"
    )
    if args.aimdo is not None:
        aimdo = args.aimdo
        aimdo_source = "cli"

    fp8_matmul = cast("bool", persisted_settings.get("fp8-matmul", False))
    fp8_matmul_source: SettingsSource = (
        "persisted" if "fp8-matmul" in persisted_settings else "default"
    )
    if args.fp8_matmul is not None:
        fp8_matmul = args.fp8_matmul
        fp8_matmul_source = "cli"

    dtype_policy = cast(
        "dict[str, str]",
        persisted_settings.get(
            "dtype-policy",
            {"diffusion": "auto", "textEncoder": "auto", "vae": "auto"},
        ),
    )
    dtype_policy = dict(dtype_policy)
    dtype_policy_source: SettingsSource = (
        "persisted" if "dtype-policy" in persisted_settings else "default"
    )
    dtype_overrides = {
        "diffusion": args.diffusion_dtype,
        "textEncoder": args.text_encoder_dtype,
        "vae": args.vae_dtype,
    }
    if any(value is not None for value in dtype_overrides.values()):
        dtype_policy.update(
            {component: value for component, value in dtype_overrides.items() if value is not None}
        )
        dtype_policy_source = "cli"

    reserve_vram = cast("int", persisted_settings.get("memory-headroom", 256 * 1024**2))
    reserve_source: SettingsSource = (
        "persisted" if "memory-headroom" in persisted_settings else "default"
    )
    if args.reserve_vram is not None:
        reserve_vram = args.reserve_vram
        reserve_source = "cli"

    comfy_args = cast("tuple[str, ...]", persisted_settings.get("worker-comfy-args", ()))
    comfy_args_source: SettingsSource = (
        "persisted" if "worker-comfy-args" in persisted_settings else "default"
    )
    if args.comfy_arg is not None:
        if None in args.comfy_arg:
            if args.comfy_arg != [None]:
                parser.error("a bare --comfy-arg cannot be combined with argument values")
            args.comfy_arg = []
        try:
            comfy_args = validate_comfy_args(tuple(args.comfy_arg), require_tuple=True)
        except SettingsError as exc:
            parser.error(f"--comfy-arg: {exc}")
        comfy_args_source = "cli"
    effective_comfy_args = comfy_args + comfy_dtype_args(dtype_policy)
    single_job_multi_gpu = (
        SingleJobMultiGpuConfig(args.single_job_multi_gpu_devices, args.single_job_multi_gpu_mode)
        if args.single_job_multi_gpu_devices
        else None
    )
    # Same lane-selection precedence as worker launch: replica lanes win
    # over single-job ranks, and both index into CUDA_VISIBLE_DEVICES.
    executing_cuda_indices: tuple[int, ...] = tuple(args.multi_gpu_devices or ()) or (
        single_job_multi_gpu.cuda_indices if single_job_multi_gpu is not None else ()
    )

    persisted_jobs = persisted_settings.get("jobs", {"maxRunningJobs": 1})
    max_running_jobs = int(dict(persisted_jobs)["maxRunningJobs"])  # type: ignore[arg-type]
    jobs_source: SettingsSource = "persisted" if "jobs" in persisted_settings else "default"
    if args.max_running_jobs is not None:
        if args.max_running_jobs < 1:
            parser.error("--max-running-jobs must be >= 1")
        max_running_jobs = args.max_running_jobs
        jobs_source = "cli"
    elif args.multi_gpu_devices:
        max_running_jobs = max(max_running_jobs, len(args.multi_gpu_devices))
        jobs_source = "cli"

    persisted_logging = persisted_settings.get("logging", {"level": "info", "overrides": {}})
    logging_value = dict(cast("dict[str, object]", persisted_logging))
    logging_source: SettingsSource = "persisted" if "logging" in persisted_settings else "default"
    if args.log_level is not None or args.log is not None:
        if args.log_level is not None:
            logging_value["level"] = args.log_level
        if args.log is not None:
            logging_value["overrides"] = overrides
        logging_source = "cli"
    log_level = str(logging_value["level"])
    effective_overrides = dict(cast("dict[str, str]", logging_value["overrides"]))

    p2p_settings = cast("dict[str, object]", persisted_settings.get("p2p", default_p2p_settings()))
    p2p_source: SettingsSource = "persisted" if "p2p" in persisted_settings else "default"
    if args.disable_p2p:
        p2p_settings = {**p2p_settings, "downloadsEnabled": False, "seedingEnabled": False}
        p2p_source = "cli"

    try:
        configure_logging(log_level, overrides=effective_overrides)
    except ValueError as exc:
        parser.error(str(exc))
    # Worker subprocesses inherit the environment; exporting the resolved
    # config here is how --log-level/--log reach pack processes.
    os.environ[LOG_LEVEL_ENV] = log_level
    if effective_overrides:
        os.environ[LOG_OVERRIDES_ENV] = ",".join(
            f"{name}={level}" for name, level in effective_overrides.items()
        )
    else:
        os.environ.pop(LOG_OVERRIDES_ENV, None)

    runtime_settings = RuntimeSettings(
        {
            "memory-budgets": memory_budgets,
            "memory-headroom": reserve_vram,
            "aimdo-policy": aimdo,
            "dtype-policy": dtype_policy,
            "fp8-matmul": fp8_matmul,
            "worker-comfy-args": comfy_args,
            "jobs": {"maxRunningJobs": max_running_jobs},
            "logging": {"level": log_level, "overrides": effective_overrides},
            "p2p": p2p_settings,
        },
        {
            "memory-budgets": memory_source,
            "memory-headroom": reserve_source,
            "aimdo-policy": aimdo_source,
            "dtype-policy": dtype_policy_source,
            "fp8-matmul": fp8_matmul_source,
            "worker-comfy-args": comfy_args_source,
            "jobs": jobs_source,
            "logging": logging_source,
            "p2p": p2p_source,
        },
        granted=granted_settings,
        path=settings_path,
        persisted_values=persisted_settings,
    )

    if args.legacy_pack and not args.comfy_root:
        parser.error("--legacy-pack requires --comfy-root (or $DINKSTER_COMFYUI_ROOT)")
    if args.allow_mount_changes and not args.library_root:
        parser.error(
            "--allow-mount-changes requires a library root "
            "(granted mounts persist to <library-root>/mounts.toml)"
        )
    if args.sandbox_packs and args.allow_mount_changes:
        parser.error(
            "--allow-mount-changes cannot be combined with --sandbox-packs; "
            "a running mount namespace cannot add or revoke bind grants"
        )
    if (args.sandbox_grant_gpu or args.sandbox_grant_network) and not args.sandbox_packs:
        parser.error("sandbox pack grants require --sandbox-packs")
    for name in args.sandbox_grant_gpu:
        if problem := validate_name(name):
            parser.error(f"invalid sandbox pack grant {name!r}: {problem}")
    sandbox_gpu_grants = frozenset(canonical_name(name) for name in args.sandbox_grant_gpu)
    mutable_network_grants: dict[str, list[str]] = {}
    for value in args.sandbox_grant_network:
        name, separator, origin = value.partition("=")
        if not separator or not name or not origin:
            parser.error("--sandbox-grant-network requires PACK=HTTPS_ORIGIN")
        if problem := validate_name(name):
            parser.error(f"invalid sandbox pack grant {name!r}: {problem}")
        try:
            normalized_origin = normalize_egress_origin(origin)
        except ValueError as exc:
            parser.error(f"invalid sandbox network origin {origin!r}: {exc}")
        mutable_network_grants.setdefault(canonical_name(name), []).append(normalized_origin)
    sandbox_network_grants = {
        name: tuple(dict.fromkeys(origins)) for name, origins in mutable_network_grants.items()
    }

    # Roots whose absolute paths must never reach wire payloads: each
    # becomes a stable token in redacted error/log text (see PathRedactor).
    redaction_roots: list[tuple[str, Path]] = []
    if args.install_root:
        redaction_roots.append(("<install>", Path(args.install_root)))
    if args.comfy_root:
        redaction_roots.append(("<comfy>", Path(args.comfy_root)))
    if args.library_root:
        redaction_roots.append(("<library>", Path(args.library_root)))

    model_roots = ()
    comfy_requirements_checked = False
    if args.library_root and args.comfy_root:
        try:
            model_roots = comfy_model_roots(
                args.comfy_root,
                python=args.comfy_python or None,
                comfy_args=effective_comfy_args,
            )
            comfy_requirements_checked = True
        except CompositionError as exc:
            raise SystemExit(str(exc)) from exc

    # Filesystem mounts: a LIVE table with mounts.toml as the durable
    # record underneath. Config parse errors are fatal (a typo'd grant
    # must not silently vanish); a mount whose directory is missing is a
    # failed ROW, not a failed process - the unplugged-drive case.
    mount_service: MountService | None = None
    mounts_snapshot: Path | None = None
    sandbox_mounts: list[MountDef] = []
    if args.library_root:
        library_root = Path(args.library_root)
        mounts_config = library_root / "mounts.toml"
        mounts_snapshot = library_root / "worker-state" / "mounts-snapshot.json"
        mount_table = MountTable(
            mounts_snapshot,
            index_root=library_root / "asset-indexes",
            output_mount=load_output_mount(mounts_config),
        )
        try:
            for mount in load_mounts(mounts_config):
                mount_table.add(mount, source="config")
                sandbox_mounts.append(mount)
                redaction_roots.append((f"<mount:{mount.id}>", mount.path))
        except MountsError as exc:
            raise SystemExit(str(exc)) from exc
        if args.comfy_root:
            comfy_root = Path(args.comfy_root)
            for mount_id, subdir, mode in _COMFY_MOUNTS:
                directory = comfy_root / subdir
                if mount_table.get(mount_id) is not None:
                    continue  # explicit config wins over derivation
                if not directory.is_dir():
                    continue
                mount = MountDef(id=mount_id, path=directory, mode=mode)
                mount_table.add(mount, source="derived")
                sandbox_mounts.append(mount)
                redaction_roots.append((f"<mount:{mount_id}>", directory))
            for model_root in model_roots:
                if mount_table.get(model_root.mount_id) is not None:
                    raise SystemExit(
                        f"configured mount id {model_root.mount_id!r} collides "
                        "with a derived ComfyUI model root"
                    )
                mount = MountDef(id=model_root.mount_id, path=model_root.path)
                mount_table.add(
                    mount,
                    source="derived",
                    kind=model_root.kind,
                )
                sandbox_mounts.append(mount)
                redaction_roots.append((f"<mount:{model_root.mount_id}>", model_root.path))
        if mount_table.output_mount is None:
            if mount_table.get("output") is not None:
                mount_table.select_output_mount("output")
            elif mount_table.get("comfy-output") is not None:
                mount_table.select_output_mount("comfy-output", require_config=False)
        mount_service = MountService(
            mount_table, mounts_config, allow_changes=args.allow_mount_changes
        )
        # Publish the (initially empty - nothing has scanned yet) snapshot
        # and point THIS process at it: media I/O save nodes run in-process, and
        # isolated pack workers inherit the environment, so one export
        # arms the write gate everywhere. Before the file exists, a save
        # would misreport "no snapshot configured" instead of the true
        # "mount not ready yet".
        mount_table.write_snapshot()
        os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(mounts_snapshot)
        # Same export pattern for the vault: EVERY worker (native packs
        # included, not just the compat children that also receive it via
        # spec env) inherits the store where consented acquisition lands
        # bytes - declared_asset() and asset-input reads resolve there.
        os.environ["DINKSTER_ASSET_VAULT"] = str(library_root / "vault")
    default_pack_failures: dict[str, Exception] = {}
    specs: list[PackSpec | str] = []
    for pack_id in () if args.no_default_packs else default_pack_ids():
        try:
            spec = replace(default_pack_spec(pack_id), require_catalog=True)
            if pack_id == "dinkster-nodes-remote":
                spec = _with_remote_config(
                    spec,
                    catalog_base=args.remote_catalog_base,
                    gateway_base=args.remote_gateway_base,
                    poll_interval=args.remote_catalog_poll_interval,
                    token_file=args.remote_auth_token_file,
                )
            specs.append(spec)
        except Exception as exc:
            # Distribution corruption is a pack failure, not a host-kernel
            # failure. Resolve defaults independently so one broken pack
            # cannot hide another pack's nodes.
            default_pack_failures[pack_id] = exc
    if not args.no_default_packs:
        specs.extend(model_pack_specs())
    try:
        compat_specs = (
            comfy_compat_specs(
                args.comfy_root or None,
                python=args.comfy_python or None,
                _requirements_checked=comfy_requirements_checked,
                legacy_packs=args.legacy_pack,
                asset_vault=(Path(args.library_root) / "vault" if args.library_root else None),
                mounts_snapshot=mounts_snapshot,
                aimdo=aimdo,
                memory_budgets=memory_budgets,
                reserve_vram=reserve_vram,
                comfy_args=effective_comfy_args,
                multi_device_cuda_indices=args.multi_gpu_devices,
                single_job_multi_gpu=single_job_multi_gpu,
            )
            if args.comfy_root or not args.no_default_packs
            else []
        )
    except CompositionError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.comfy_root:
        specs.extend(replace(spec, require_catalog=True) for spec in compat_specs)
    resolved_default_pack_count = len(specs)
    if args.library_root and not args.no_default_packs:
        try:
            specs.extend(
                replace(spec, require_catalog=True)
                for spec in training_pack_specs(Path(args.library_root) / "training.sqlite")
            )
        except Exception as exc:
            default_pack_failures["dinkster-training"] = exc
    installer: Installer | None = None
    if args.install_root:
        # Installed defaults compose first, then managed packs, then explicit
        # dev additions. Any name collision fails composition loudly.
        installer = Installer(Path(args.install_root))
        specs.extend(installer.packs_for_serving())
    specs.extend(args.pack)
    if args.comfy_root:
        specs.extend(compat_specs)
    default_pack_venv_root = _default_pack_venv_root(args.library_root)
    default_pack_accelerator = resolve_accelerator()

    assembler = None
    sampler = None
    if args.benchmark:
        benchmark_log = core_logger("benchmark")
        benchmark_dir = args.benchmark_dir

        def emit_record(record: dict[str, object]) -> None:
            path = write_record(record, benchmark_dir)
            benchmark_log.info("benchmark record: %s", path)

        sampler = HardwareSampler(interval_s=args.benchmark_interval)
        assembler = BenchmarkAssembler(emit_record, sampler=sampler)

    log = core_logger("serve")
    # Composition failures recorded by the drive task: the process must
    # exit nonzero (configuration errors fail loudly), and the abort path
    # is "cancel run_app's main task", which alone would exit silently.
    failure: list[BaseException] = []

    async def build_app() -> web.Application:
        # Publish installed declarations without activating pack runtimes.
        # Explicit development packs use live discovery.
        # One governor per serving instance (DESIGN 3.10): declared budgets
        # from persisted defaults plus per-device CLI overrides gate worker
        # reservations; everything else is observability (footprints,
        # /memory/status, manual shed, leases).
        # Always constructed, so the memory surface exists even with no
        # budgets - an unbudgeted device admits every reservation.
        # Worker-reported measurements land in one ReportedTelemetry store,
        # feeding the governor's probe/enumeration seats: /memory/status
        # shows measured beside declared (and discovers measured-only
        # devices). Informational only - admission stays declared math.
        reported = ReportedTelemetry()
        governor = MemoryGovernor(
            memory_budgets,
            telemetry=reported.probe,
            telemetry_devices=reported.devices,
        )
        vault: AssetVault | None = None
        native_policy: NativeDispatchPolicy | None = None
        composer_ref: list[ServingComposer] = []
        if args.library_root:
            root = Path(args.library_root)
            vault = AssetVault(root / "vault")
            assert mount_service is not None
            native_policy = NativeDispatchPolicy(
                _native_asset_locator(vault, mount_service.table),
                _log_native_policy_diagnostic,
                fp8_matmul=lambda: runtime_settings.fp8_matmul,
                dtype_policy=lambda: runtime_settings.dtype_policy,
                compute_dtypes=lambda: detect_native_compute_dtypes(executing_cuda_indices),
                schedule_conversion=lambda path, logical_name: (
                    _schedule_legacy_checkpoint_conversion(composer_ref[0], path, logical_name)
                ),
                minimax_h3_runtime_versions=(
                    lambda: detect_native_runtime_versions(
                        comfy_python(Path(args.comfy_root), args.comfy_python or None)
                        if args.comfy_root
                        else args.comfy_python
                        or os.environ.get("DINKSTER_COMFYUI_PYTHON")
                        or sys.executable
                    )
                ),
                schemas=lambda: composer_ref[0].composition.schemas,
            )
        sandbox_policy: SandboxPolicy | None = None
        sandbox_writable_mounts: list[str] = []
        if args.sandbox_packs:
            sandbox_ro_binds: list[str] = []
            protected_roots: list[str] = []
            protected_ro_exceptions: list[str] = []
            if args.library_root:
                protected_roots.append(str(Path(args.library_root)))
            if args.install_root:
                protected_roots.append(str(Path(args.install_root)))
            secret_paths = (
                args.auth,
                args.federated_assets_cursor_key,
                args.advertise_assets_token_file,
                args.remote_auth_token_file,
                *(remote.token_file for remote in remote_specs),
            )
            protected_roots.extend(str(Path(path)) for path in secret_paths if path)
            if vault is not None:
                sandbox_ro_binds.append(str(vault.root))
                protected_ro_exceptions.append(str(vault.root))
            if mounts_snapshot is not None:
                # Bind the dedicated directory so atomic snapshot replacement
                # remains visible without exposing the library root beside it.
                sandbox_ro_binds.append(str(mounts_snapshot.parent))
                protected_ro_exceptions.append(str(mounts_snapshot.parent))
            for mount in sandbox_mounts:
                sandbox_ro_binds.append(str(mount.path))
                if mount.mode == "readwrite":
                    sandbox_writable_mounts.append(str(mount.path))
            sandbox_policy = SandboxPolicy(
                ro_binds=tuple(dict.fromkeys(sandbox_ro_binds)),
                protected_roots=tuple(dict.fromkeys(protected_roots)),
                protected_ro_exceptions=tuple(dict.fromkeys(protected_ro_exceptions)),
            )

        schema_reload_requests: asyncio.Queue[str] = asyncio.Queue()
        schema_reload_pending: set[str] = set()

        def request_schema_reload(name: str) -> None:
            if name not in schema_reload_pending:
                schema_reload_pending.add(name)
                schema_reload_requests.put_nowait(name)

        composer = ServingComposer(
            sandbox_policy=sandbox_policy,
            sandbox_gpu_grants=sandbox_gpu_grants,
            sandbox_network_grants=sandbox_network_grants,
            sandbox_writable_mounts=sandbox_writable_mounts,
            pack_scratch_root=(
                Path(args.library_root).resolve() / "scratch" if args.library_root else None
            ),
            dev=args.watch_packs,
            on_diagnostic=(assembler.on_boundary_diagnostic if assembler is not None else None),
            explain_misses=args.benchmark,
            governor=governor,
            reservations=GovernorReservationService(governor),
            telemetry=reported,
            native_policy=native_policy,
            runtime_worker_settings=lambda: (
                runtime_settings.aimdo_policy,
                runtime_settings.memory_headroom,
                runtime_settings.memory_budgets,
                runtime_settings.worker_comfy_args,
            ),
            headroom_base=runtime_settings.memory_headroom,
            remote_asset_endpoint=args.advertise_assets,
            remote_asset_token=remote_asset_token,
            # Persistent value store for remote workers, next to the vault:
            # bulk boundary values land once and later runs cross as digest
            # references. Only with a library root - no root, no disk home.
            remote_value_store=(
                BudgetedDiskCAS(Path(args.library_root) / "value-store")
                if args.library_root
                else None
            ),
            on_schema_reload=request_schema_reload,
            cache_mode=cache_mode,
            cache_memory_entries=args.execution_cache_memory_entries,
            cache_dir=cache_dir,
            cache_disk_budget=args.execution_cache_disk_budget,
            composition_mode="development" if args.watch_packs else "production",
        )
        try:
            ordered_defaults = composer.order_pack_entries(specs[:resolved_default_pack_count])
        except CompositionError:
            # Incremental add_pack calls report each broken default independently.
            pass
        else:
            specs[:resolved_default_pack_count] = ordered_defaults
        composer_ref.append(composer)
        composer.validate_specs(specs)
        if args.prepare_stale_catalogs:
            for entry in specs:
                if not isinstance(entry, PackSpec) or not entry.require_catalog:
                    continue
                prepared = (
                    _prepare_default_pack(
                        entry,
                        venv_root=default_pack_venv_root,
                        accelerator=default_pack_accelerator,
                    )
                    if not entry.in_process and _is_standard_vision_pack(entry)
                    else entry
                )
                for manifest_path in (
                    resolve_manifest_path(prepared.manifest),
                    *prepared.group_manifests,
                ):
                    manifest = load_manifest(manifest_path)
                    if read_catalog(manifest) is None:
                        report = prepare_catalog(
                            manifest_path,
                            interpreter=prepared.python,
                            environment=prepared.env,
                        )
                        if not report.ok:
                            raise CompositionError(
                                f"runtime catalog preparation failed for {report.pack_name}"
                            )
                        print(
                            f"Prepared runtime catalog: {report.pack_name} "
                            f"({len(report.node_types)} nodes)"
                        )
        composer.validate_catalogs(specs)
        composition = composer.composition
        make_engine = composition.make_engine
        if assembler is not None and sampler is not None:
            # Records carry the comparability envelope: which packs served.
            assembler.environment["packs"] = sorted(composition.packs)
            make_engine = instrument_engine_factory(make_engine, assembler)
            sampler.start()
        library = None
        history = None
        training_sessions = None
        execution_journal = None
        resolver_indexes = None
        p2p_manager: LanP2PController | None = None
        principal_permissions = PrincipalPermissionStore(
            Path(args.library_root) / "principals.sqlite" if args.library_root else None
        )
        if args.library_root:
            root = Path(args.library_root)
            assert vault is not None
            provenance = ProvenanceStore(root / "provenance.json")
            receipts = PublicAcquisitionReceiptStore(root / "public-acquisition-receipts.json")
            resolver_indexes = ResolverSubscriptionStore(
                root / "resolver-indexes.json",
                provenance,
                region=args.resolver_region,
            )
            library = ServerLibrary(
                vault=vault,
                store=LibraryStore(root / "library.sqlite"),
                # Digest GETs fall through to ready mounts so a browsed
                # file previews without ever being copied into the vault.
                resolver=(mount_service.table if mount_service is not None else None),
                # Acquisition leads (templates/asset distribution): where
                # bytes for a digest can be obtained. Leads, never
                # authorities - fetched bytes verify or land nowhere.
                provenance=provenance,
                receipts=receipts,
                public_sources_for=resolver_indexes.public_sources,
                # Pack-declared assets ([[pack.assets]]): the composer
                # maintains this catalog across add/reload/remove; the
                # library reads it for job preflight and the sources
                # surface. Same object for the process lifetime - the
                # catalog swaps its CONTENTS, never its identity.
                pack_assets=composition.asset_catalog,
                model_output_profile=lambda path, handle, digest, size: (
                    load_model_output_profile(
                        path,
                        asset_digest=digest,
                        asset_size=size,
                        handle=handle,
                    ).document
                ),
            )
            p2p_manager = LanP2PController(
                vault=vault,
                resolver_indexes=resolver_indexes,
                receipts=receipts,
                local_path_for=library.locate,
                installation_root=Path(args.install_root) if args.install_root else None,
            )
            library = replace(
                library,
                lan_resolve=p2p_manager.resolve_sync,
                p2p_acquired=p2p_manager.notify_acquired,
            )
            # Persistent execution history rides the same persistence root:
            # terminal runs land in history.sqlite with their sourceDocument
            # link back to uploaded workflow assets.
            history = HistoryStore(root / "history.sqlite")
            # Training sessions ride it too: the session ledger and its
            # journal share training.sqlite (one fsync domain), owned by
            # the store and closed by create_app's cleanup.
            training_sessions = TrainingSessionStore(JournalStore(root / "training.sqlite"))
            # Execution-run logs (journal replay for the frontend's
            # execution log): their own SQLite file, so whole-stream
            # retention can never touch the training ledger's file.
            execution_journal = ExecutionJournal(
                JournalStore(root / "execution.sqlite"),
                keep_runs=args.execution_log_runs,
                keep_days=args.execution_log_days,
            )
        resolution_store: ResolutionStore | None = None
        provider_policy: Mapping[str, frozenset[str]] | None = None
        cursor_key: bytes | None = None
        store_closed = False
        if federated_asset_config is not None:
            store_path, provider_policy, cursor_key = federated_asset_config
            resolution_store = ResolutionStore(store_path)

        def close_resolution_store() -> None:
            nonlocal store_closed
            if resolution_store is not None and not store_closed:
                store_closed = True
                resolution_store.close()

        build_task = asyncio.current_task()
        if resolution_store is not None and build_task is not None:
            # run_app owns this task. If any later app-composition step fails,
            # aiohttp has no completed Application whose cleanup it can run.
            def close_on_build_failure(task: asyncio.Task[Any]) -> None:
                if task.cancelled() or task.exception() is not None:
                    close_resolution_store()

            build_task.add_done_callback(close_on_build_failure)

        try:
            generation_service = None
            if args.openai_base_url:
                compatibility = OpenAICompatibility(args.openai_compatibility)

                def make_generation_provider() -> OpenAIGenerationProvider:
                    return OpenAIGenerationProvider(
                        args.openai_base_url,
                        args.openai_model,
                        api_key=args.openai_api_key or None,
                        compatibility=compatibility,
                        stream=args.openai_response_mode == "stream",
                        timeout_s=float(args.openai_timeout),
                    )

                generation_service = GenerationService(
                    (
                        GenerationModel(
                            args.openai_model,
                            "dinkster.openai",
                            make_generation_provider,
                        ),
                    )
                )
            if resolution_store is not None and provider_policy is not None:
                scopes = tuple(sorted(provider_policy))
                if mount_service is None:
                    resolution_store.replace_mount_snapshot(scopes, ())
                else:
                    mount_service.attach_resolution_store(resolution_store, scopes)
            redactor = PathRedactor(redaction_roots)
            app = create_app(
                make_engine,
                composition.schemas,
                pack_route_dispatch=composer.call_pack_route,
                frontend_module_read=composer.read_frontend_module,
                max_running_jobs=max_running_jobs,
                governor=governor,
                packs=composition.packs,
                node_packs=composition.node_packs,
                execution_arms=composition.execution_arms,
                allow_hosts=[args.host, *args.allow_host],
                allow_origins=args.allow_origin,
                authenticator=authenticator,
                principal_permissions=principal_permissions,
                library=library,
                history=history,
                training_sessions=training_sessions,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                schema_owners=composition.schema_owners,
                choice_owners=composition.choice_owners,
                compat_skips=composition.compat_skips,
                settings=runtime_settings,
                redactor=redactor,
                attention_policy=attention_policy,
                execution_journal=execution_journal,
                memory_headroom_changed=composer.set_memory_headroom,
                residency_memory_budgets=composer.residency_memory_budgets,
                workers=lambda: composer.workers(remote_specs),
                place_execution=composer.place_execution,
                full_free=composer.full_free,
                federated_asset_paths=(
                    _FEDERATED_ASSET_PATHS if resolution_store is not None else None
                ),
                federated_asset_store=resolution_store,
                federated_asset_sources=(),
                federated_asset_provider_policy=provider_policy,
                _federated_asset_cursor_key=cursor_key,
            )
            if args.event_loop_stall_threshold is not None:
                _install_event_loop_stall_diagnostics(
                    app,
                    threshold=args.event_loop_stall_threshold,
                    logger=log,
                )
            if p2p_manager is None:
                add_p2p_routes(app, None, runtime_settings)
            else:
                add_p2p_routes(
                    app,
                    p2p_manager,
                    runtime_settings,
                    grant_snapshot=lambda: p2p_manager.grant_snapshot,
                )
        except BaseException:
            close_resolution_store()
            principal_permissions.close()
            raise

        async def close_federated_assets(_: web.Application) -> None:
            close_resolution_store()

        if resolution_store is not None:
            app.on_cleanup.append(close_federated_assets)
        # ComfyUI API prompt submissions translate at this edge; the server
        # package itself stays comfy-free.
        add_comfy_compat_routes(app)
        _add_collaboration_routes(
            app,
            Path(args.library_root) / "sessions.sqlite" if args.library_root else None,
        )
        if resolver_indexes is not None:
            add_resolver_index_routes(
                app,
                resolver_indexes,
                p2p_granted="p2p" in runtime_settings.granted,
                official_url=args.official_resolver_url,
                official_provider_id=args.official_resolver_provider_id,
            )
        # Name-guess lookups for imported ComfyUI workflows: candidates
        # come from mount catalogs, pack declarations, and resolver indexes;
        # accepting one is the client's explicit gesture. Always registered -
        # an empty composition answers with empty candidates, not a 404.
        add_guess_routes(app)
        if mount_service is not None:
            # GET routes serve everyone; the mutation half answers 403
            # mount-changes-disabled unless --allow-mount-changes opted in.
            add_mount_routes(app, mount_service)
        if installer is not None:
            # Live install activation: reconcile the served surface with
            # the install root's current generation without a restart.
            # Not dev-gated - the mutation was authorized at dinkster-pack
            # apply time; this endpoint only delivers it.
            add_activation_routes(app, composer, installer)
        watcher: PackWatcher | None = None
        if args.watch_packs:
            # Pack hot reload is a dev affordance: production installs
            # change packs through the manager's plan/apply flow, never a
            # live mutation endpoint. Not registered = 404, no half-open
            # surface to secure.
            add_reload_routes(app, composer, redactor=redactor)
            if args.watch_packs:
                # The watcher is sugar over the same coordinator the
                # endpoint uses: one reload implementation, two triggers.

                async def reload_changed(name: str) -> None:
                    await apply_reload(app[STATE_KEY], composer, name)

                watcher = PackWatcher(composer.watch_targets, reload_changed)
        if args.frontend_root or args.frontend_dev:
            install_frontend(
                app,
                bundle=Path(args.frontend_root) if args.frontend_root else None,
                development_url=args.frontend_dev or None,
            )

        state = app[STATE_KEY]
        default_failure_count = len(default_pack_failures)
        total = default_failure_count + len(specs) + len(remote_specs)
        # Every intended pack and remote gets a report row up front
        # ("pending"), so /api/composition tells the whole startup story
        # from the first request - including what has not landed yet.
        keys = state.seed_composition(
            [*default_pack_failures]
            + [_spec_label(entry) for entry in specs]
            + [spec.name for spec in remote_specs]
        )
        default_failure_keys = keys[:default_failure_count]
        pack_start = default_failure_count
        remote_start = pack_start + len(specs)
        pack_keys = keys[pack_start:remote_start]
        remote_keys = keys[remote_start:]
        if total:
            # Set before the port binds, so the very first health probe
            # already narrates composition instead of racing it.
            state.narrate_composition(0, total)
        drive_tasks: list[asyncio.Task[None]] = []
        pack_report_keys: dict[str, str] = {}
        # Set once startup composition reaches a terminal state, so the
        # reconnect supervisor never races the initial add_remote pass.
        startup_composed = asyncio.Event()

        async def start_composition(app: web.Application) -> None:
            # on_startup runs inside run_app's main task; capturing it is
            # what lets a composition failure abort the whole process.
            main_task = asyncio.current_task()

            async def drive() -> None:
                async def publish_delta(delta: PackDelta) -> int:
                    validation = await state.prepare_replace(
                        (), (), delta.schemas, delta.packs, delta.node_packs
                    )
                    return state.replace(
                        (),
                        (),
                        delta.schemas,
                        delta.packs,
                        delta.node_packs,
                        execution_arms=delta.execution_arms,
                        remove_choices=tuple(delta.derived_choices),
                        choices={**delta.choices, **delta.derived_choices},
                        lazy_choices=delta.lazy_choices,
                        schema_owners=delta.schema_owners,
                        choice_owners=delta.choice_owners,
                        compat_skips=delta.compat_skips,
                        _validation=validation,
                    )

                async def publish_removal(result: RemoveResult) -> int:
                    validation = await state.prepare_replace(
                        result.removed_types,
                        result.removed_packs,
                        {},
                        {},
                        {},
                    )
                    return state.replace(
                        result.removed_types,
                        result.removed_packs,
                        {},
                        {},
                        {},
                        execution_arms=result.execution_arms,
                        remove_choices=(
                            *result.removed_choices,
                            *result.derived_choices,
                        ),
                        choices=result.derived_choices,
                        choice_owners=result.choice_owners,
                        remove_compat_skips=result.removed_compat_skips,
                        _validation=validation,
                    )

                try:
                    completed_before_packs = 0
                    for (pack_id, pack_failure), failure_key in zip(
                        default_pack_failures.items(), default_failure_keys, strict=True
                    ):
                        state.mark_pack_failed(failure_key, str(pack_failure))
                        log.error(
                            "pack %s failed to resolve: %s",
                            pack_id,
                            pack_failure,
                        )
                        if args.strict_packs:
                            failure.append(pack_failure)
                            if main_task is not None:
                                main_task.cancel()
                            return
                        completed_before_packs += 1
                        state.narrate_composition(completed_before_packs, total)
                    for spec_index, (entry, key) in enumerate(
                        zip(specs, pack_keys, strict=True),
                    ):
                        done = completed_before_packs + spec_index + 1
                        try:
                            if (
                                spec_index < resolved_default_pack_count
                                and isinstance(entry, PackSpec)
                                and not entry.in_process
                                and _is_standard_vision_pack(entry)
                            ):
                                entry = await asyncio.to_thread(
                                    _prepare_default_pack,
                                    entry,
                                    venv_root=default_pack_venv_root,
                                    accelerator=default_pack_accelerator,
                                )
                            async with composer.publication_transaction():
                                delta = await composer.add_pack(entry)
                                epoch = await composer.finish_publication(publish_delta(delta))
                        except asyncio.CancelledError:
                            raise
                        except BaseException as exc:
                            # One pack failing does not cost the others:
                            # add_pack is atomic (a failed pack merges
                            # nothing), so record the error - process-
                            # lifetime cache on /api/composition plus the
                            # pack_failed event - and keep composing.
                            # --strict-packs restores config-is-fatal.
                            state.mark_pack_failed(key, str(exc))
                            log.error("pack %s failed to compose: %s", key, exc)
                            if args.strict_packs:
                                failure.append(exc)
                                if main_task is not None:
                                    main_task.cancel()
                                return
                            state.narrate_composition(done, total)
                            continue
                        state.mark_pack_announced(key, epoch)
                        pack_report_keys[delta.pack] = key
                        if assembler is not None:
                            assembler.environment["packs"] = sorted(composition.packs)
                        state.narrate_composition(done, total)
                        log.info(
                            "pack %s announced (%d/%d, epoch %d)",
                            delta.pack,
                            done,
                            total,
                            epoch,
                        )
                    # Remote workers compose after the packs, one at a
                    # time, with the same per-entry failure handling:
                    # add_remote is atomic, so a dead or misconfigured
                    # daemon is recorded and the rest keep composing.
                    for done, (remote_spec, key) in enumerate(
                        zip(remote_specs, remote_keys, strict=True),
                        start=completed_before_packs + len(specs) + 1,
                    ):
                        try:
                            async with composer.publication_transaction():
                                delta = await composer.add_remote(remote_spec)
                                epoch = await composer.finish_publication(publish_delta(delta))
                        except asyncio.CancelledError:
                            raise
                        except BaseException as exc:
                            state.mark_pack_failed(key, str(exc))
                            log.error("remote worker %s failed to compose: %s", key, exc)
                            if args.strict_packs:
                                failure.append(exc)
                                if main_task is not None:
                                    main_task.cancel()
                                return
                            state.narrate_composition(done, total)
                            continue
                        state.mark_pack_announced(key, epoch)
                        if assembler is not None:
                            assembler.environment["packs"] = sorted(composition.packs)
                        state.narrate_composition(done, total)
                        log.info(
                            "remote worker %s announced (%d/%d, epoch %d)",
                            delta.pack,
                            done,
                            total,
                            epoch,
                        )
                    while incomplete := composer.incomplete_generation_removals():
                        missing_types = sorted(
                            node_type
                            for node_types in incomplete.values()
                            for node_type in node_types
                        )
                        reason = "schema-only nodes have no execution provider: " + ", ".join(
                            missing_types
                        )
                        if args.strict_packs:
                            raise CompositionError(reason)
                        for pack, node_types in incomplete.items():
                            async with composer.publication_transaction():
                                result = await composer.remove_pack(pack)
                                epoch = await composer.finish_publication(publish_removal(result))
                            detail = (
                                "schema-only nodes have no execution provider: "
                                + ", ".join(sorted(node_types))
                                if node_types
                                else f"required pack was retracted because {reason}"
                            )
                            key = pack_report_keys.get(pack, pack)
                            state.mark_pack_failed(key, detail)
                            log.error("pack %s retracted: %s", pack, detail)
                            if assembler is not None:
                                assembler.environment["packs"] = sorted(composition.packs)
                    composer.validate_complete_generation()
                    # The positive "loading actually finished" moment:
                    # clears the health/nodes narration and (when packs
                    # composed) emits composition_complete with the final
                    # epoch - after the last schema_changed, always.
                    # "Complete" means every pack reached a terminal state,
                    # not that every pack loaded: failures ride the event's
                    # "failed" list and stay on /api/composition.
                    state.complete_composition()
                    node_count = len(state.schemas)
                    failed = [
                        key for key in keys if state.composition_packs[key].get("state") == "failed"
                    ]
                    pack_note = f", {len(composition.packs)} pack(s)" if composition.packs else ""
                    fail_note = f", {len(failed)} FAILED (see /api/composition)" if failed else ""
                    print(
                        f"dinkster server: {node_count} node types{pack_note} composed{fail_note}"
                    )
                    startup_composed.set()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    # Host wiring failure (not one pack's): the process
                    # goes down loudly (exit nonzero via main()), exactly
                    # like the pre-progressive startup - just after the
                    # port opened instead of before.
                    failure.append(exc)
                    log.error("composition failed: %s", exc)
                    if main_task is not None:
                        main_task.cancel()

            drive_tasks.append(asyncio.create_task(drive()))

            async def reload_schema_sources() -> None:
                while True:
                    name = await schema_reload_requests.get()
                    try:
                        await apply_reload(app[STATE_KEY], composer, name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.warning("pack %s automatic schema reload failed: %s", name, exc)
                    finally:
                        schema_reload_pending.discard(name)

            drive_tasks.append(asyncio.create_task(reload_schema_sources()))
            if remote_specs:
                supervisor = RemoteReconnectSupervisor(state, composer, remote_specs, remote_keys)

                async def supervise() -> None:
                    # Waits forever when startup composition failed as a
                    # host wiring problem - the process is going down.
                    await startup_composed.wait()
                    await supervisor.run()

                drive_tasks.append(asyncio.create_task(supervise()))
            if mount_service is not None:
                # Mount scans ride their own task, parallel to pack
                # composition: hashing a model library must not delay
                # pack announcements (or vice versa), and each finished
                # scan announces itself with mounts_changed.
                drive_tasks.append(asyncio.create_task(mount_service.scan_all(app)))
            if watcher is not None:
                # Started alongside composition (first sighting of each
                # pack takes a baseline without firing, so watching during
                # startup is safe) and cancelled with the drive tasks.
                drive_tasks.append(asyncio.create_task(watcher.run()))

        async def stop_composition(_: web.Application) -> None:
            errors: list[Exception] = []
            for task in drive_tasks:
                task.cancel()
            await asyncio.gather(*drive_tasks, return_exceptions=True)
            if generation_service is not None:
                try:
                    await asyncio.to_thread(generation_service.close)
                except Exception as error:
                    errors.append(error)
            if mount_service is not None:
                try:
                    await mount_service.close()
                except Exception as error:
                    errors.append(error)
            if sampler is not None:
                try:
                    sampler.stop()
                except Exception as error:
                    errors.append(error)
            try:
                await composition.close()
            except Exception as error:
                errors.append(error)
            if errors:
                raise ExceptionGroup("server cleanup failed", errors)

        app.on_startup.append(start_composition)
        app.on_cleanup.append(stop_composition)
        if generation_service is not None:
            add_generation_routes(app, generation_service, register_cleanup=False)
        pack_note = f", composing {total} pack(s)" if total else ""
        print(f"dinkster server on http://{args.host}:{args.port} (diagnostic core{pack_note})")
        return app

    try:
        # Upgrade tickets travel in the query string; default aiohttp access
        # logging includes it verbatim in the request target.
        web.run_app(
            build_app(),
            host=args.host,
            port=args.port,
            print=None,
            access_log=None,
        )
    except asyncio.CancelledError:
        if failure:
            raise SystemExit(f"composition failed: {failure[0]}") from failure[0]
        raise


if __name__ == "__main__":
    main()
