"""Opt-in, local-human plan/apply transport over the pack manager CLI."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_registry import InstallError, PlanRecord
from dinkster_schema import core_logger
from dinkster_server import STATE_KEY, PathRedactor, principal_for

from .activation import apply_activation
from .compose import ServingComposer
from .installer import Installer
from .registries import parse_registry_spec

_CONFIRMATION_SECONDS = 900
_log = core_logger("pack_install")


def add_pack_install_routes(
    app: web.Application,
    composer: ServingComposer,
    installer: Installer,
    *,
    busy: asyncio.Lock,
    redactor: PathRedactor,
) -> None:
    # One outstanding review per principal bounds retained plans. A restarted
    # backend deliberately forgets confirmations rather than replaying them.
    pending: dict[str, tuple[str, float, str]] = {}

    def failure(status: int, error: str, detail: str = "", **extra: object) -> web.Response:
        return web.json_response(
            redactor.redact_value({"error": error, "detail": detail, **extra}),
            status=status,
            headers={"Cache-Control": "no-store"},
        )

    async def body_for(request: web.Request, field: str) -> tuple[str, object]:
        principal = principal_for(request)
        peer = request.transport.get_extra_info("peername") if request.transport else None
        try:
            address = ipaddress.ip_address(peer[0]) if isinstance(peer, tuple) and peer else None
            loopback = address is not None and address.is_loopback
        except ValueError:
            loopback = False
        if not loopback or not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"local-operator-required"}',
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        try:
            raw: object = await request.json()
        except ValueError:
            raise web.HTTPBadRequest(
                text='{"error":"bad-request"}',
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            ) from None
        if not isinstance(raw, dict) or set(raw) != {field}:
            raise web.HTTPBadRequest(
                text='{"error":"bad-request"}',
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        return principal.principal_id, raw[field]

    async def manager(*arguments: str) -> str:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dinkster.manager",
            "--root",
            str(installer.root),
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, stderr = await asyncio.shield(communication)
        except asyncio.CancelledError:
            # Keep the mutation lock and plan file alive until the manager
            # exits; request cancellation must not orphan a concurrent writer.
            await communication
            raise
        if process.returncode:
            raise InstallError(stderr.decode(errors="replace"))
        return stdout.decode(errors="replace")

    async def handle_plan(request: web.Request) -> web.Response:
        owner, raw = await body_for(request, "packs")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
            return failure(
                400, "bad-request", "packs must contain 1 to 64 exact registry selections"
            )
        selections: list[str] = []
        for item in cast("list[object]", raw):
            if not isinstance(item, dict) or set(item) != {"pack", "version"}:
                return failure(400, "bad-request", "each pack needs only pack and version")
            pack, version = item["pack"], item["version"]
            if not isinstance(pack, str) or not isinstance(version, str):
                return failure(400, "bad-request", "pack and version must be strings")
            spec = parse_registry_spec(f"{pack}@{version}")
            if spec is None or spec != (pack, version):
                return failure(400, "bad-request", "pack must be canonical and version exact")
            selections.append(f"{pack}@{version}")
        if len({selection.partition("@")[0] for selection in selections}) != len(selections):
            return failure(400, "bad-request", "pack selections must be unique")
        if busy.locked():
            return failure(409, "install-in-progress")
        async with busy:
            try:
                with tempfile.TemporaryDirectory(prefix="dinkster-pack-plan-") as directory:
                    path = Path(directory) / "plan.json"
                    review = await manager("install", *selections, "--plan", str(path))
                    if not path.is_file():
                        return failure(409, "already-installed", "the installation already matches")
                    text = path.read_text()
                    record = PlanRecord.from_record_json(text)
            except (InstallError, OSError, ValueError) as exc:
                return failure(422, "plan-failed", str(exc))
            now = time.monotonic()
            for key in tuple(pending):
                if pending[key][1] <= now:
                    del pending[key]
            if owner not in pending and len(pending) >= 64:
                return failure(503, "too-many-plans")
            confirmation = secrets.token_urlsafe(32)
            pending[owner] = (confirmation, now + _CONFIRMATION_SECONDS, text)
            # The temporary file path is not part of the review or wire contract.
            review = review.split("plan written to ", 1)[0].rstrip()
            return web.json_response(
                {
                    "confirmation": confirmation,
                    "plan": json.loads(record.record_json()),
                    "review": redactor.redact_text(review),
                    "expiresInSeconds": _CONFIRMATION_SECONDS,
                },
                headers={"Cache-Control": "no-store"},
            )

    async def handle_apply(request: web.Request) -> web.Response:
        owner, confirmation = await body_for(request, "confirmation")
        if not isinstance(confirmation, str) or not confirmation.isascii():
            return failure(400, "bad-request", "confirmation must be an ASCII string")
        if busy.locked():
            return failure(409, "install-in-progress")
        async with busy:
            state = request.app[STATE_KEY]
            if state.composition_progress is not None:
                return failure(409, "composition-in-progress")
            stored = pending.get(owner)
            if (
                stored is None
                or stored[1] <= time.monotonic()
                or not secrets.compare_digest(stored[0], confirmation)
            ):
                return failure(409, "invalid-confirmation")
            del pending[owner]
            record = PlanRecord.from_record_json(stored[2])
            if not record.matches_base(installer.current_lockfile()):
                return failure(409, "plan-stale", "installation changed; re-plan and review again")

            async def install_and_activate() -> web.Response:
                try:
                    with tempfile.TemporaryDirectory(prefix="dinkster-pack-apply-") as directory:
                        path = Path(directory) / "plan.json"
                        path.write_text(stored[2])
                        await manager("apply", str(path))
                except (InstallError, OSError, ValueError) as exc:
                    code = "plan-stale" if "plan is stale:" in str(exc) else "apply-failed"
                    return failure(409 if code == "plan-stale" else 422, code, str(exc))
                generation = installer.current_number()
                try:
                    activation = await apply_activation(state, composer, installer)
                except Exception as exc:
                    return failure(409, "activation-failed", str(exc), generation=generation)
                if activation.get("failed"):
                    return failure(
                        409,
                        "activation-failed",
                        "installed generation could not be activated; "
                        "previous live surface retained",
                        generation=generation,
                        activation=activation,
                    )
                return web.json_response(
                    {"generation": generation, "activation": activation},
                    headers={"Cache-Control": "no-store"},
                )

            operation = asyncio.create_task(install_and_activate())
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(operation)
                    break
                except asyncio.CancelledError:
                    if operation.cancelled():
                        raise
                    # Repeated request cancellations must not reach the mutation task.
                    cancelled = True
            if cancelled:
                _log.info(
                    "pack installation finished after request cancellation: HTTP %s", result.status
                )
                raise asyncio.CancelledError
            return result

    app.router.add_post("/api/packs/install/plan", handle_plan)
    app.router.add_post("/api/packs/install/apply", handle_apply)
