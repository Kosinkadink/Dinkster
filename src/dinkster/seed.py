"""Headless existing-store seeder; no editor or execution engine is constructed."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, NoReturn, cast

from aiohttp import ClientSession, web
from dinkster_assets import (
    AssetVault,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    ResolverSubscriptionStore,
)
from dinkster_assets.p2p_store import ExistingSeedStore
from dinkster_p2p import P2PManagerError, default_p2p_settings
from dinkster_p2p.settings import normalize_p2p_settings
from dinkster_server.network_cost import detect_network_cost

from .lan_p2p import LanP2PController
from .p2p_api import P2PSidecarActivityProvider

_ERROR_CODES = frozenset(
    {
        "provider-refresh-failed",
        "seed-reconcile-failed",
        "network-policy-failed",
        "status-unavailable",
        "control-failed",
        "resume-refused",
        "state-write-failed",
        "command-failed",
        "invalid-request",
        "operation-failed",
    }
)


def _sanitized_errors(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (item if isinstance(item, str) and item in _ERROR_CODES else "operation-failed")
            if item is not None and key.lower().endswith(("error", "reason", "message", "detail"))
            else _sanitized_errors(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitized_errors(item) for item in value]
    return value


class _SeedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        super().error("invalid arguments; use --help for supported options")


class SeedService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = args.state_dir.resolve()
        if any(self.state.is_relative_to(root.resolve()) for root in args.store):
            raise ValueError("seed state must be outside every read-only store")
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.vault = AssetVault(self.state / "vault")
        self.store = ExistingSeedStore(self.vault, args.store)
        self.resolver = ResolverSubscriptionStore(
            self.state / "resolver.json",
            ProvenanceStore(self.state / "provenance.json"),
            revalidate_seconds=0,
        )
        self.settings = normalize_p2p_settings(
            {
                **default_p2p_settings(),
                "downloadsEnabled": False,
                "seedingEnabled": self._saved_enabled(),
                "seedMode": "continuous",
                "maxActiveSeeds": args.max_active_seeds,
                "listenPort": args.listen_port,
                "internetUploadBytesPerSecond": args.upload_bytes_per_second,
                "lanUploadBytesPerSecond": args.upload_bytes_per_second,
                "networkCostOverride": args.network_cost,
            }
        )
        self.controller = LanP2PController(
            vault=self.vault,
            resolver_indexes=self.resolver,
            receipts=PublicAcquisitionReceiptStore(self.state / "receipts.json"),
            local_path_for=self.store.local_path_for,
        )
        self.activity = P2PSidecarActivityProvider(self.controller, detect_network_cost)
        self.mapped: tuple[str, ...] = ()
        self.error: str | None = None
        self.next_refresh_at: float | None = None
        self.last_refresh_at: float | None = max(
            (
                row.refreshed_at
                for row in self.resolver.subscriptions()
                if row.source == args.provider_url and row.index.name == args.provider_id
            ),
            default=None,
        )
        self._control_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._network_task: asyncio.Task[None] | None = None
        self._subscription_id: str | None = None
        self._failures = 0

    def _saved_enabled(self) -> bool:
        try:
            enabled = json.loads((self.state / "enabled.json").read_text())
        except FileNotFoundError:
            return True
        if type(enabled) is not bool:
            raise ValueError("enabled.json must contain a boolean")
        return enabled

    async def refresh(self) -> float:
        try:
            if self._subscription_id is None:
                subscription = await asyncio.to_thread(
                    self.resolver.bootstrap_official,
                    self.args.provider_url,
                    self.args.provider_id,
                )
                if subscription is None:
                    raise ValueError("configured provider was explicitly unsubscribed")
                self._subscription_id = subscription.id
            rows = await asyncio.to_thread(self.resolver.refresh, self._subscription_id)
            if any(row.get("error") for row in rows):
                raise ValueError("provider refresh failed; retained authority expires normally")
            self.last_refresh_at = max(
                (row.refreshed_at for row in self.resolver.subscriptions()),
                default=None,
            )
            self.error = None
            self._failures = 0
        except Exception:
            self.error = "provider-refresh-failed"
            self._failures += 1
        try:
            snapshots = await asyncio.to_thread(self.resolver.provider_p2p_snapshots)
            self.mapped = await asyncio.to_thread(self.store.refresh, snapshots)
            if self.settings["seedingEnabled"]:
                await self.controller.reconcile(local_files_changed=True)
        except Exception:
            self.error = "seed-reconcile-failed"
            self._failures += 1
        delay = (
            self.args.refresh_seconds
            if not self._failures
            else min(
                self.args.backoff_max_seconds,
                self.args.backoff_seconds * 2 ** min(self._failures - 1, 20),
            )
        )
        self.next_refresh_at = time.time() + delay
        return delay

    async def _refresh_loop(self) -> None:
        while True:
            try:
                delay = await self.refresh()
            except Exception:
                self.error = "seed-reconcile-failed"
                delay = self.args.backoff_max_seconds
                self.next_refresh_at = time.time() + delay
            await asyncio.sleep(delay)

    async def start(self, _app: web.Application) -> None:
        await self.activity.start(self.settings)
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        self._network_task = asyncio.create_task(self._network_loop())

    async def _network_loop(self) -> None:
        while True:
            try:
                await self.activity.reconcile_network_policy()
            except Exception:
                self.error = "network-policy-failed"
            await asyncio.sleep(5)

    async def close(self, _app: web.Application) -> None:
        for task in (self._refresh_task, self._network_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await self.controller.close()

    async def status(self) -> dict[str, object]:
        controller = await self.controller.status()
        sidecar = cast(dict[str, Any], controller.get("sidecar") or {})
        leases = sidecar.get("leases", [])
        seeded = sorted(
            {
                row["digest"]
                for row in leases
                if row.get("kind") == "seed"
                and row.get("state") == "ready"
                and row.get("scope") == "lan-and-internet"
            }
        )
        global_status = sidecar.get("global") or {}
        peers = sum(row.get("peers", 0) for row in global_status.get("transfers", []))
        current = sorted(
            digest for digest in self.mapped if self.store.local_path_for(digest) is not None
        )
        ready = bool(self.settings["seedingEnabled"] and current and set(current) <= set(seeded))
        return {
            "ready": ready,
            "enabled": self.settings["seedingEnabled"],
            "mappedDigests": current,
            "seededDigests": seeded,
            "peers": peers,
            "totals": sidecar.get("totals", {}),
            "lastSuccessfulRefresh": self.last_refresh_at,
            "nextRefresh": self.next_refresh_at,
            "error": self.error,
            "p2p": _sanitized_errors(controller),
        }

    async def get_status(self, request: web.Request) -> web.Response:
        status = await self.status()
        return web.json_response(
            status, status=503 if request.path == "/ready" and not status["ready"] else 200
        )

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"healthy": True})

    async def set_enabled(self, request: web.Request) -> web.Response:
        if request.headers.get("Origin") or request.content_type != "application/json":
            raise web.HTTPForbidden()
        enabled = await request.json()
        if type(enabled) is not bool:
            raise web.HTTPBadRequest(text="expected a JSON boolean")
        async with self._control_lock:
            settings = {**self.settings, "seedingEnabled": enabled}
            await self.activity.update(settings)
            self.settings = settings
            try:
                temporary = self.state / "enabled.tmp"
                with temporary.open("w") as handle:
                    json.dump(enabled, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(self.state / "enabled.json")
                if os.name != "nt":
                    directory = os.open(self.state, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            except OSError:
                raise web.HTTPServiceUnavailable(text="state-write-failed") from None
            if enabled:
                try:
                    await self.controller.resume_seed_transfers(self.mapped)
                except P2PManagerError:
                    raise web.HTTPConflict(text="resume-refused") from None
        return await self.get_status(request)

    def application(self) -> web.Application:
        @web.middleware
        async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
            try:
                return await handler(request)
            except web.HTTPException as error:
                code = error.text if error.text in _ERROR_CODES else "invalid-request"
                return web.json_response({"error": code}, status=error.status)
            except Exception:
                code = "control-failed" if request.path == "/enabled" else "status-unavailable"
                return web.json_response({"error": code}, status=503)

        app = web.Application(client_max_size=1024, middlewares=[errors])
        app.add_routes(
            [
                web.get("/health", self.health),
                web.get("/ready", self.get_status),
                web.get("/status", self.get_status),
                web.post("/enabled", self.set_enabled),
            ]
        )
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.close)
        return app


def parser() -> argparse.ArgumentParser:
    result = _SeedArgumentParser(description=__doc__)
    result.add_argument(
        "command", nargs="?", choices=("run", "status", "enable", "disable"), default="run"
    )
    result.add_argument(
        "--state-dir",
        type=Path,
        default=Path(
            os.environ.get("DINKSTER_SEED_STATE_DIR", "~/.local/state/dinkster-seed")
        ).expanduser(),
    )
    result.add_argument("--store", type=Path, action="append", default=None)
    result.add_argument("--provider-url", default=os.environ.get("DINKSTER_SEED_PROVIDER_URL"))
    result.add_argument("--provider-id", default=os.environ.get("DINKSTER_SEED_PROVIDER_ID"))
    result.add_argument(
        "--network-cost",
        choices=("auto", "metered", "unmetered"),
        default=os.environ.get("DINKSTER_SEED_NETWORK_COST", "auto"),
    )
    for flag, env, default, kind in (
        ("listen-port", "LISTEN_PORT", 0, int),
        ("status-port", "STATUS_PORT", 0, int),
        ("upload-bytes-per-second", "UPLOAD_BYTES_PER_SECOND", 5242880, int),
        ("max-active-seeds", "MAX_ACTIVE_SEEDS", 512, int),
        ("refresh-seconds", "REFRESH_SECONDS", 300, float),
        ("backoff-seconds", "BACKOFF_SECONDS", 5, float),
        ("backoff-max-seconds", "BACKOFF_MAX_SECONDS", 300, float),
    ):
        result.add_argument(
            "--" + flag, type=kind, default=os.environ.get("DINKSTER_SEED_" + env, default)
        )
    return result


async def run(args: argparse.Namespace) -> None:
    if args.command != "run":
        port = int((args.state_dir / "status-port").read_text())
        async with ClientSession() as client:
            url = f"http://127.0.0.1:{port}"
            if args.command == "status":
                response = await client.get(url + "/status")
            else:
                response = await client.post(url + "/enabled", json=args.command == "enable")
            print(json.dumps(_sanitized_errors(await response.json()), indent=2))
            if response.status >= 400:
                raise SystemExit(1)
        return
    service = SeedService(args)
    runner = web.AppRunner(service.application(), access_log=None)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.status_port)
        await site.start()
        port = runner.addresses[0][1]
        (service.state / "status-port").write_text(str(port))
        print(
            "Seeding verified provider-listed files only; disable with "
            f"dinkster-seed disable --state-dir {service.state}",
            flush=True,
        )
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, stopped.set)
        await stopped.wait()
    finally:
        await runner.cleanup()


def main() -> None:
    try:
        arguments = parser()
        args = arguments.parse_args()
        if args.command == "run":
            args.store = args.store or [
                Path(value)
                for value in os.environ.get("DINKSTER_SEED_STORES", "").split(os.pathsep)
                if value
            ]
            if not args.store or not args.provider_url or not args.provider_id:
                arguments.error(
                    "run requires --store, --provider-url and --provider-id "
                    "(or their environment settings)"
                )
            for field in ("refresh_seconds", "backoff_seconds", "backoff_max_seconds"):
                if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
                    arguments.error(field + " must be finite and positive")
            if not 0 <= args.status_port <= 65535:
                arguments.error("status-port must be between 0 and 65535")
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except Exception:
        print(json.dumps({"error": "command-failed"}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
