"""Local-human managed-generation previews; application remains terminal-owned."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import cast

from aiohttp import web
from dinkster_server import PathRedactor, principal_for
from dinkster_supervisor.environment_pack_install import plan_pack_install
from dinkster_supervisor.environment_state import EnvironmentError

from .registries import parse_registry_spec


def add_pack_install_routes(
    app: web.Application,
    project: Path,
    *,
    redactor: PathRedactor,
) -> None:
    pending: dict[str, tuple[float, list[str]]] = {}
    busy = asyncio.Lock()
    operations: set[asyncio.Task[dict[str, object]]] = set()

    async def drain_previews(_: web.Application) -> None:
        await asyncio.gather(*operations, return_exceptions=True)

    app.on_cleanup.insert(0, drain_previews)

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

    def commands_for(selections: list[str]) -> list[str]:
        output = Path(tempfile.gettempdir()) / f"dinkster-pack-review-{uuid.uuid4().hex}"
        receipt = str(output) + ".json"
        if os.name == "nt":

            def quote(value: str) -> str:
                return "'" + value.replace("'", "''") + "'"

            command = "& 'dinkster-env' --root " + quote(str(project))
            return [
                command
                + " packs-plan "
                + " ".join(map(quote, selections))
                + " --output "
                + quote(str(output))
                + " --json > "
                + quote(receipt)
                + "; if ($LASTEXITCODE -ne 0) { throw 'Pack planning failed; do not apply.' }; "
                + "Get-Content -Raw "
                + quote(receipt),
                command
                + " packs-apply --input "
                + quote(str(output))
                + " --approval ((Get-Content -Raw "
                + quote(receipt)
                + ") | ConvertFrom-Json).approval",
            ]
        command = shlex.join(["dinkster-env", "--root", str(project)])
        approval = shlex.join(
            [
                sys.executable,
                "-c",
                "import json,sys; print(json.load(sys.stdin)['approval'])",
            ]
        )
        return [
            command
            + " packs-plan "
            + shlex.join(selections)
            + " --output "
            + shlex.quote(str(output))
            + " --json > "
            + shlex.quote(receipt)
            + " && cat "
            + shlex.quote(receipt),
            command
            + " packs-apply --input "
            + shlex.quote(str(output))
            + ' --approval "$('
            + approval
            + " < "
            + shlex.quote(receipt)
            + ')"',
        ]

    async def handle_plan(request: web.Request) -> web.Response:
        owner, raw = await body_for(request, "packs")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
            return failure(400, "bad-request", "packs must contain 1 to 64 exact selections")
        selections: list[str] = []
        for item in cast("list[object]", raw):
            if not isinstance(item, dict) or set(item) != {"pack", "version"}:
                return failure(400, "bad-request", "each pack needs only pack and version")
            pack, version = item["pack"], item["version"]
            if not isinstance(pack, str) or not isinstance(version, str):
                return failure(400, "bad-request", "pack and version must be strings")
            if parse_registry_spec(f"{pack}@{version}") != (pack, version):
                return failure(400, "bad-request", "pack must be canonical and version exact")
            selections.append(f"{pack}@{version}")
        if len({selection.partition("@")[0] for selection in selections}) != len(selections):
            return failure(400, "bad-request", "pack selections must be unique")
        if busy.locked():
            return failure(409, "install-in-progress")
        now = time.monotonic()
        for key in tuple(pending):
            if pending[key][0] <= now:
                del pending[key]
        if owner not in pending and len(pending) >= 64:
            return failure(503, "too-many-plans")

        def preview() -> dict[str, object]:
            with tempfile.TemporaryDirectory(prefix="dinkster-pack-preview-") as directory:
                return plan_pack_install(project, tuple(selections), Path(directory) / "inputs")

        async with busy:
            operation = asyncio.create_task(asyncio.to_thread(preview))
            operations.add(operation)
            operation.add_done_callback(operations.discard)
            try:
                cancelled = False
                while True:
                    try:
                        result = await asyncio.shield(operation)
                        break
                    except asyncio.CancelledError:
                        if operation.cancelled():
                            raise
                        cancelled = True
                if cancelled:
                    raise asyncio.CancelledError
            except (EnvironmentError, OSError, ValueError) as exc:
                return failure(422, "plan-failed", str(exc))
        commands = commands_for(selections)
        pending[owner] = (time.monotonic() + 900, commands)
        return web.json_response(
            {
                "plan": result["packPlan"],
                "review": redactor.redact_text("\n".join(cast("list[str]", result["plan"]))),
                "expiresInSeconds": result["expiresInSeconds"],
                "commands": commands,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def handle_apply(request: web.Request) -> web.Response:
        owner, confirmation = await body_for(request, "confirmation")
        if not isinstance(confirmation, str) or not confirmation.isascii():
            return failure(400, "bad-request", "confirmation must be an ASCII string")
        stored = pending.get(owner)
        # Commands must retain exact paths, not the diagnostic redactor's labels.
        return web.json_response(
            {
                "error": "registry.managed-generation-required",
                "detail": "Run packs-plan, review it, then run packs-apply in the terminal.",
                "commands": stored[1] if stored is not None else [],
            },
            status=409,
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_post("/api/packs/install/plan", handle_plan)
    app.router.add_post("/api/packs/install/apply", handle_apply)
