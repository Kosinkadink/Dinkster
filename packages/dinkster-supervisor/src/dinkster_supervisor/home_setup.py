"""Always-on setup planning and one cooperatively cancelled CLI installation job."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import socket
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

from aiohttp import web

from .environment_artifacts import discover
from .environment_hardware import (
    detect_graphics_cards,
    hardware_selection,
)
from .environment_state import dinkster_home, library_engine, read_document, shared_library

RELEASE_MIRROR = "https://releases.dinkster.org"


class SetupRefused(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def library_refusal() -> dict[str, str] | None:
    engine = library_engine(shared_library())
    if engine is None:
        return None
    return {
        "code": "library-held",
        "message": f"Another installation's engine ({engine}) is using the library. "
        "Stop that engine before setting up. Nothing was downloaded or changed.",
    }


async def environment_command(root: Path, *arguments: str) -> dict[str, object]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(root),
        *arguments,
        "--mirror",
        RELEASE_MIRROR,
        "--json",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
    )
    output, _ = await process.communicate()
    result = cast("dict[str, object]", json.loads(output))
    if process.returncode and result.get("status") != "cancelled":
        raise SetupRefused(
            str(result.get("errorCode") or "setup-failed"),
            str(result.get("error") or "Setup failed."),
        )
    return result


def setup_facts(root: Path, channel: str, requested: str | None) -> dict[str, object]:
    if not root.is_absolute():
        raise SetupRefused("invalid-root", "Choose an absolute installation folder.")
    parent = root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    refusal = library_refusal()
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        refusal = {"code": "invalid-root", "message": "Choose a new or empty installation folder."}
    facts: dict[str, object] = {
        "root": str(root.resolve()),
        "channel": channel,
        "variant": None,
        "buildLabel": None,
        "buildReason": None,
        "engineRelease": None,
        "baseId": None,
        "detectedCards": None,
        "choices": None,
        "gpus": [],
        "downloadBytes": None,
        "runtimeDownloadBytes": None,
        "downloadEstimate": None,
        "freeBytes": free,
        "refusal": refusal,
    }
    if refusal is not None and refusal["code"] != "library-held":
        return facts
    try:
        catalog = discover((RELEASE_MIRROR,), channel)
    except (OSError, ValueError) as exc:
        facts["refusal"] = refusal or {
            "code": "build-unavailable",
            "message": f"The {channel} build metadata is unavailable: {exc}",
        }
        return facts
    try:
        selection = hardware_selection(catalog.bases, detect_graphics_cards(), requested)
    except ValueError as exc:
        raise SetupRefused("unsupported-build", str(exc)) from exc
    facts.update(selection)
    facts["refusal"] = refusal or selection["refusal"]
    if selection["refusal"] is not None:
        return facts
    variant = cast("str", facts["variant"])
    facts.update(
        engineRelease=catalog.channel.engine_release,
        baseId=catalog.channel.base_for(variant),
    )
    return facts


class SetupJobs:
    def __init__(self, completed: Callable[[Path], Awaitable[None]]) -> None:
        self._completed = completed
        self._plans: dict[str, dict[str, object]] = {}
        self._pack_identities: dict[str, str | None] = {}
        self._jobs: dict[str, dict[str, object]] = {}
        self._task: asyncio.Task[None] | None = None
        self._progress: Path | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def active_job(self) -> dict[str, object] | None:
        job_id = next(reversed(self._jobs), None)
        return self.get(job_id) if job_id is not None else None

    async def plan(self, root: Path, channel: str, variant: str | None) -> dict[str, object]:
        facts = await asyncio.to_thread(setup_facts, root, channel, variant)
        pack_identity = None
        if facts["engineRelease"] is not None:
            try:
                result = await environment_command(
                    root,
                    "setup",
                    "--channel",
                    channel,
                    "--variant",
                    cast("str", facts["variant"]),
                    "--engine-release",
                    cast("str", facts["engineRelease"]),
                    "--base-id",
                    cast("str", facts["baseId"]),
                    "--estimate-packs",
                    "--plan",
                )
            except (OSError, ValueError):
                if facts["refusal"] is None:
                    raise
                result = {}
            facts["runtimeDownloadBytes"] = result.get("runtimeDownloadBytes")
            facts["downloadEstimate"] = result.get("downloadEstimate")
            pack_identity = cast("str | None", result.get("packEstimateIdentity"))
            if (
                facts["refusal"] is None
                and isinstance(facts["runtimeDownloadBytes"], int)
                and facts["runtimeDownloadBytes"] > cast("int", facts["freeBytes"])
            ):
                facts["refusal"] = {
                    "code": "insufficient-space",
                    "message": "The drive has too little free space for the runtime download.",
                }
        plan_id = uuid4().hex
        facts["id"] = plan_id
        self._plans[plan_id] = facts
        self._pack_identities[plan_id] = pack_identity
        while len(self._plans) > 16:
            oldest = next(iter(self._plans))
            del self._plans[oldest]
            del self._pack_identities[oldest]
        return facts

    async def start(self, plan_id: str) -> dict[str, object]:
        async with self._lock:
            if self._closed:
                raise SetupRefused("setup-stopping", "The process is shutting down.")
            if self._task is not None and not self._task.done():
                raise SetupRefused(
                    "setup-active", "Another setup job is still running or cleaning up."
                )
            previous = self._plans.get(plan_id)
            if previous is None:
                raise KeyError(plan_id)
            if refusal := library_refusal():
                raise SetupRefused(refusal["code"], refusal["message"])
            previous_identity = self._pack_identities[plan_id]
            facts = await self.plan(
                Path(cast("str", previous["root"])),
                cast("str", previous["channel"]),
                cast("str | None", previous["variant"]),
            )
            refusal = facts["refusal"]
            if isinstance(refusal, dict):
                refusal = cast("dict[str, str]", refusal)
                raise SetupRefused(refusal["code"], refusal["message"])
            if previous_identity != self._pack_identities[cast("str", facts["id"])] or any(
                facts[key] != previous[key]
                for key in ("root", "channel", "variant", "engineRelease", "baseId", "gpus")
            ):
                raise SetupRefused(
                    "plan-changed", "The setup selection changed. Review a new plan."
                )
            job_id = uuid4().hex
            job: dict[str, object] = {
                "id": job_id,
                "root": facts["root"],
                "channel": facts["channel"],
                "variant": facts["variant"],
                "state": "running",
                "progress": None,
                "error": None,
                "installation": None,
            }
            self._jobs[job_id] = job
            while len(self._jobs) > 16:
                del self._jobs[next(iter(self._jobs))]
            self._progress = dinkster_home() / "setup-jobs" / f"{job_id}.json"
            self._task = asyncio.create_task(
                self._run(job, facts, self._progress, previous_identity)
            )
            return job

    async def _run(
        self,
        job: dict[str, object],
        plan: dict[str, object],
        progress: Path,
        pack_identity: str | None,
    ) -> None:
        root = Path(cast("str", job["root"]))
        try:
            # Setup owns a new listener, not the engine address of the always-on process.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            result = await environment_command(
                root,
                "setup",
                "--channel",
                cast("str", plan["channel"]),
                "--variant",
                cast("str", plan["variant"]),
                "--engine-release",
                cast("str", plan["engineRelease"]),
                "--base-id",
                cast("str", plan["baseId"]),
                *(("--pack-estimate-identity", pack_identity) if pack_identity else ()),
                "--port",
                str(port),
                "--progress",
                str(progress),
                *(
                    argument
                    for gpu in cast("list[str]", plan["gpus"])
                    for argument in ("--gpu", gpu)
                ),
                "--yes",
            )
            if result["status"] == "cancelled":
                job["state"] = "cancelled"
            else:
                await self._completed(root)
                job.update(state="completed", installation={"root": str(root), "engine": "main"})
        except (OSError, ValueError, LookupError, RuntimeError) as exc:
            job.update(
                state="failed",
                error={"code": getattr(exc, "code", "setup-failed"), "message": str(exc)},
            )
        finally:
            progress.unlink(missing_ok=True)
            progress.with_name(progress.name + ".cancel").unlink(missing_ok=True)

    def get(self, job_id: str) -> dict[str, object]:
        job = self._jobs[job_id]
        if job["state"] in {"running", "cancelling"} and self._progress is not None:
            with contextlib.suppress(OSError, ValueError):
                observed = read_document(self._progress)
                total = observed["total"]
                job["progress"] = {
                    "stage": observed["stage"],
                    "completed": observed["completed"],
                    "total": total or None,
                    "unit": "bytes" if observed["stage"] == "downloading" else "items",
                }
        return job

    def cancel(self, job_id: str) -> dict[str, object]:
        job = self.get(job_id)
        if job["state"] in {"running", "cancelling"}:
            assert self._progress is not None
            self._progress.parent.mkdir(parents=True, exist_ok=True)
            self._progress.with_name(self._progress.name + ".cancel").touch()
            job["state"] = "cancelling"
        return job

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._task is not None and not self._task.done():
                assert self.active_job is not None
                self.cancel(cast("str", self.active_job["id"]))
                await self._task


SETUP_KEY = web.AppKey("home_setup", SetupJobs)


async def handle_plan(request: web.Request) -> web.Response:
    try:
        if set(request.query) - {"root", "channel", "variant"}:
            raise ValueError("Unknown setup query field.")
        channel = request.query.get("channel", "nightly")
        if channel not in {"stable", "nightly"}:
            raise ValueError("Choose the stable or nightly channel.")
        jobs = request.app[SETUP_KEY]
        plan = await jobs.plan(
            Path(request.query.get("root", str(dinkster_home() / "project"))),
            channel,
            request.query.get("variant"),
        )
        return web.json_response({"plan": plan, "activeJob": jobs.active_job})
    except (OSError, ValueError) as exc:
        return web.json_response(
            {"error": getattr(exc, "code", "invalid-setup"), "message": str(exc)}, status=400
        )


async def handle_start(request: web.Request) -> web.Response:
    try:
        raw: object = await request.json()
        if not isinstance(raw, dict):
            raise ValueError("Supply only a setup planId.")
        body = cast("dict[str, object]", raw)
        if set(body) != {"planId"} or not isinstance(body["planId"], str):
            raise ValueError("Supply only a setup planId.")
        job = await request.app[SETUP_KEY].start(body["planId"])
        return web.json_response(job, status=202)
    except KeyError:
        return web.json_response(
            {"error": "unknown-plan", "message": "Review a new setup plan."}, status=404
        )
    except SetupRefused as exc:
        return web.json_response({"error": exc.code, "message": str(exc)}, status=409)
    except (OSError, ValueError) as exc:
        return web.json_response({"error": "invalid-setup", "message": str(exc)}, status=400)


async def handle_job(request: web.Request) -> web.Response:
    try:
        jobs = request.app[SETUP_KEY]
        job = (
            jobs.cancel(request.match_info["id"])
            if request.method == "DELETE"
            else jobs.get(request.match_info["id"])
        )
        status = 202 if request.method == "DELETE" and job["state"] == "cancelling" else 200
        return web.json_response(job, status=status)
    except KeyError:
        return web.json_response(
            {"error": "unknown-job", "message": "That setup job is not known to this process."},
            status=404,
        )
