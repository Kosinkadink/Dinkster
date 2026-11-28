"""Always-on installation planning and one cooperatively cancelled job."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import shutil
import socket
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

from aiohttp import web
from packaging.version import Version

from .environment_artifacts import discover
from .environment_hardware import (
    detect_graphics_cards,
    hardware_selection,
)
from .environment_state import (
    Generation,
    Journal,
    content_digest,
    dinkster_home,
    library_engine,
    read_current,
    read_document,
    shared_library,
    update_selection_identity,
)
from .home_context import installation_context, resolve_installation_root

RELEASE_MIRROR = "https://releases.dinkster.org"


class SetupRefused(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def library_refusal(root: Path | None = None) -> dict[str, str] | None:
    if root is not None and (context := installation_context(root)) is not None:
        return cast("dict[str, str] | None", context.call(root, "setup-refusal"))
    engine = library_engine(shared_library())
    if engine is None:
        return None
    return {
        "code": "library-held",
        "message": f"Another installation's engine ({engine}) is using the library. "
        "Stop that engine before setting up. Nothing was downloaded or changed.",
    }


async def environment_command(root: Path, *arguments: str) -> dict[str, object]:
    context = installation_context(root)
    command = [
        sys.executable,
        "-I",
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(root),
        *arguments,
    ]
    if context is not None:
        command = context.environment_command("--root", str(root), *arguments)
    process = await asyncio.create_subprocess_exec(
        *command,
        *(("--mirror", RELEASE_MIRROR) if arguments[0] in {"setup", "update"} else ()),
        "--json",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
    )
    output, _ = await process.communicate()
    result = cast("dict[str, object]", json.loads(output))
    if (
        process.returncode
        and result.get("status") != "cancelled"
        and not (arguments[0] == "doctor" and result.get("status") == "drift")
    ):
        raise SetupRefused(
            str(result.get("errorCode") or f"{arguments[0]}-failed"),
            str(result.get("error") or "Installation operation failed."),
        )
    return result


def cancelled_setup_identity(root: Path) -> str | None:
    """Identify only the unchanged files left by a completed cancellation."""
    if (context := installation_context(root)) is not None:
        return cast("str | None", context.call(root, "cancelled-setup"))
    try:
        if root.is_symlink() or root.is_junction():
            return None
        children = {path.name for path in root.iterdir()}
        if children - {".lock", "journal", "builds", "runtime", "engine-state", "environment-id"}:
            return None
        identity = hashlib.sha256()
        pending = [root]
        while pending:
            path = pending.pop()
            if path.is_symlink() or path.is_junction():
                return None
            info = path.stat()
            if path.is_dir():
                pending.extend(sorted(path.iterdir()))
                content = None
            elif path.is_file():
                content = content_digest(path)
            else:
                return None
            identity.update(
                json.dumps(
                    [str(path.relative_to(root)), info.st_dev, info.st_ino, content]
                ).encode()
            )
        if (root / "builds").exists() and any((root / "builds").iterdir()):
            return None
        journals = list((root / "journal").glob("*"))
        if not journals and children - {".lock"}:
            return None
        for path in journals:
            journal = Journal.read(path)
            if (
                journal.operation != "setup"
                or journal.previous is not None
                or journal.stage != "rolled-back"
                or journal.error not in {"setup cancelled", "pack preparation cancelled"}
            ):
                return None
        return identity.hexdigest()
    except (OSError, ValueError):
        return None


def setup_facts(
    root: Path, channel: str, requested: str | None, *, cancelled: bool = False
) -> dict[str, object]:
    if (context := installation_context(root)) is not None:
        facts = cast(
            "dict[str, object]", context.call(root, "setup-facts", channel, requested, cancelled)
        )
        facts["root"] = str(context.windows_path(cast("str", facts["root"])))
        return facts
    if not root.is_absolute():
        raise SetupRefused("invalid-root", "Choose an absolute installation folder.")
    parent = root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    refusal = library_refusal()
    if (
        root.is_symlink()
        or root.is_junction()
        or (root.exists() and (not root.is_dir() or (not cancelled and any(root.iterdir()))))
    ):
        refusal = {"code": "invalid-root", "message": "Choose a new or empty installation folder."}
    facts: dict[str, object] = {
        "root": str(root.resolve()),
        "channel": channel,
        "variant": None,
        "buildLabel": None,
        "buildReason": None,
        "engineRelease": None,
        "baseId": None,
        "buildDetails": None,
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
    if selection["refusal"] is not None or selection["variant"] is None:
        return facts
    variant = cast("str", facts["variant"])
    facts.update(
        engineRelease=catalog.channel.engine_release,
        baseId=catalog.channel.base_for(variant),
    )
    base = catalog.bases[variant]
    python = base.get("python")
    torch = next(
        (
            cast("str", wheel["version"])
            for wheel in cast("list[dict[str, object]]", base.get("wheels", []))
            if wheel["name"] == "torch"
        ),
        None,
    )
    if python and torch:
        # PyTorch's CUDA wheel version names its bundled runtime, even on Windows.
        cuda = re.fullmatch(r"cu(\d+)(\d)", Version(torch).local or "")
        facts["buildDetails"] = {
            "python": python,
            "pytorch": torch,
            "cuda": f"{cuda[1]}.{cuda[2]}" if cuda else None,
            "dinkster": catalog.engine["version"],
        }
    return facts


def update_facts(root: Path) -> dict[str, object]:
    if (context := installation_context(root)) is not None:
        facts = cast("dict[str, object]", context.call(root, "update-facts"))
        facts["root"] = str(root)
        return facts
    identity = update_selection_identity(root)
    current = read_current(root)
    recorded = Generation.read(root / "builds" / str(current) / "build.json")
    if recorded.number != current or recorded.source == "local":
        raise SetupRefused("update-unavailable", "This selection cannot use a channel update.")
    settings = read_document(root / "environment.json")
    channel, variant = settings.get("channel"), settings.get("variant")
    if (
        not isinstance(channel, str)
        or channel not in {"stable", "nightly"}
        or not isinstance(variant, str)
    ):
        raise SetupRefused("update-unavailable", "The installation selection is unreadable.")
    catalog = discover((RELEASE_MIRROR,), channel)
    if variant not in catalog.bases:
        raise SetupRefused("update-unavailable", "The channel does not offer this build.")
    if identity != update_selection_identity(root):
        raise SetupRefused("plan-changed", "The installation changed. Review a new plan.")
    return {
        "operation": "update",
        "root": str(root),
        "current": current,
        "selectionApproval": identity,
        "channel": channel,
        "variant": variant,
        "engineRelease": catalog.channel.engine_release,
        "baseId": catalog.channel.base_for(variant),
        "version": catalog.engine["version"],
        "gpus": [],
        "refusal": None,
    }


def restore_facts(root: Path, entry: int) -> dict[str, object]:
    from .environment_restore_plan import plan_restore

    if type(entry) is not int or entry < 1:
        raise SetupRefused("invalid-entry", "Choose a recorded history entry.")
    if (context := installation_context(root)) is not None:
        return cast("dict[str, object]", context.call(root, "restore-plan", entry))
    return plan_restore(root, entry)


class SetupJobs:
    def __init__(
        self,
        completed: Callable[[Path], Awaitable[None]],
        *,
        prepare: Callable[[], Awaitable[None]] | None = None,
        installations: Callable[[], frozenset[Path]] | None = None,
        updated: Callable[[Path], Awaitable[None]] | None = None,
    ) -> None:
        self._completed = completed
        self._prepare = prepare
        self._installations = installations
        self._updated = updated
        self._plans: dict[str, dict[str, object]] = {}
        self._pack_identities: dict[str, str | None] = {}
        self._jobs: dict[str, dict[str, object]] = {}
        self._running_plan: dict[str, object] | None = None
        self._cancelled_roots: dict[Path, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._progress: Path | None = None
        self._lock = asyncio.Lock()
        self._progress_lock = asyncio.Lock()
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def active_job(self) -> dict[str, object] | None:
        job_id = next(reversed(self._jobs), None)
        return self._jobs[job_id] if job_id is not None else None

    async def plan(self, root: Path, channel: str, variant: str | None) -> dict[str, object]:
        resolved = await asyncio.to_thread(resolve_installation_root, root)
        if (
            self.busy
            and self._running_plan is not None
            and self._running_plan.get("operation", "setup") == "setup"
            and root.is_absolute()
            and str(resolved) == self._running_plan["root"]
            and channel == self._running_plan["channel"]
            and variant in (None, self._running_plan["variant"])
        ):
            return self._running_plan
        identity = self._cancelled_roots.get(resolved)
        if identity is not None and identity == await asyncio.to_thread(
            cancelled_setup_identity, root
        ):
            facts = await asyncio.to_thread(setup_facts, root, channel, variant, cancelled=True)
        else:
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

    async def plan_update(self, root: Path) -> dict[str, object]:
        if not root.is_absolute() or self._installations is None:
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        root = await asyncio.to_thread(resolve_installation_root, root)
        if root not in await asyncio.to_thread(self._installations):
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        if self.busy and self._running_plan is not None:
            if self._running_plan.get("operation") == "update" and self._running_plan[
                "root"
            ] == str(root):
                return self._running_plan
        facts = await asyncio.to_thread(update_facts, root)
        plan_id = uuid4().hex
        facts["id"] = plan_id
        self._plans[plan_id] = facts
        self._pack_identities[plan_id] = None
        while len(self._plans) > 16:
            oldest = next(iter(self._plans))
            del self._plans[oldest]
            del self._pack_identities[oldest]
        return facts

    async def start(self, plan_id: str, *, operation: str = "setup") -> dict[str, object]:
        async with self._lock:
            if self._closed:
                raise SetupRefused("setup-stopping", "The process is shutting down.")
            if self._task is not None and not self._task.done():
                raise SetupRefused(
                    "setup-active", "Another installation job is still running or cleaning up."
                )
            previous = self._plans.get(plan_id)
            if previous is None or previous.get("operation", "setup") != operation:
                raise KeyError(plan_id)
            if operation == "setup" and (
                refusal := await asyncio.to_thread(
                    library_refusal, Path(cast("str", previous["root"]))
                )
            ):
                raise SetupRefused(refusal["code"], refusal["message"])
            previous_identity = self._pack_identities[plan_id]
            if operation == "check":
                facts = await self.plan_check(Path(cast("str", previous["root"])))
                if facts["root"] != previous["root"]:
                    raise SetupRefused(
                        "plan-changed", "The installation target changed. Review a new plan."
                    )
            elif operation == "restore":
                root = Path(cast("str", previous["root"]))
                if self._installations is None or root not in await asyncio.to_thread(
                    self._installations
                ):
                    raise SetupRefused("unknown-installation", "Choose a registered installation.")
                proposal = await asyncio.to_thread(
                    restore_facts, root, cast("int", previous["entry"])
                )
                if proposal != previous["proposal"]:
                    raise SetupRefused(
                        "plan-changed", "The restore proposal changed. Review a new plan."
                    )
                facts = previous
            elif operation == "update":
                facts = await self.plan_update(Path(cast("str", previous["root"])))
            else:
                facts = await self.plan(
                    Path(cast("str", previous["root"])),
                    cast("str", previous["channel"]),
                    cast("str | None", previous["variant"]),
                )
            refusal = facts.get("refusal")
            if isinstance(refusal, dict):
                refusal = cast("dict[str, str]", refusal)
                raise SetupRefused(refusal["code"], refusal["message"])
            if operation not in {"restore", "check"} and facts["variant"] is None:
                raise SetupRefused("build-not-chosen", "Choose a build before starting setup.")
            if operation not in {"restore", "check"} and (
                previous_identity != self._pack_identities[cast("str", facts["id"])]
                or any(
                    facts[key] != previous[key]
                    for key in (
                        "root",
                        "channel",
                        "variant",
                        "engineRelease",
                        "baseId",
                        "gpus",
                        *(("selectionApproval",) if operation == "update" else ()),
                    )
                )
            ):
                raise SetupRefused(
                    "plan-changed", "The installation selection changed. Review a new plan."
                )
            job_id = uuid4().hex
            job: dict[str, object] = {
                "id": job_id,
                "root": facts["root"],
                "channel": facts.get("channel"),
                "variant": facts.get("variant"),
                "state": "running",
                "progress": None,
                "error": None,
                "installation": None,
                "operation": operation,
            }
            self._jobs[job_id] = job
            while len(self._jobs) > 16:
                del self._jobs[next(iter(self._jobs))]
            self._progress = dinkster_home() / "setup-jobs" / f"{job_id}.json"
            context = installation_context(Path(cast("str", facts["root"])))
            if context is not None:
                self._progress = context.windows_path(
                    f"/home/dinkster/.dinkster/setup-jobs/{job_id}.json"
                )
            self._running_plan = facts
            self._cancelled_roots.pop(Path(cast("str", facts["root"])), None)
            self._task = asyncio.create_task(
                self._run(job, facts, self._progress, previous_identity)
            )
            return job

    async def plan_check(self, root: Path) -> dict[str, object]:
        if not root.is_absolute() or self._installations is None:
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        root = await asyncio.to_thread(resolve_installation_root, root)
        if root not in await asyncio.to_thread(self._installations):
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        plan_id = uuid4().hex
        facts: dict[str, object] = {"id": plan_id, "operation": "check", "root": str(root)}
        self._plans[plan_id] = facts
        self._pack_identities[plan_id] = None
        while len(self._plans) > 16:
            oldest = next(iter(self._plans))
            del self._plans[oldest]
            del self._pack_identities[oldest]
        return facts

    async def plan_restore(self, root: Path, entry: int) -> dict[str, object]:
        if not root.is_absolute() or self._installations is None:
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        root = await asyncio.to_thread(resolve_installation_root, root)
        if root not in await asyncio.to_thread(self._installations):
            raise SetupRefused("unknown-installation", "Choose a registered installation.")
        proposal = await asyncio.to_thread(restore_facts, root, entry)
        alternatives: list[dict[str, object]] = []
        for alternative in cast("list[dict[str, object]]", proposal["alternatives"]):
            plan_id = uuid4().hex
            self._plans[plan_id] = {
                "id": plan_id,
                "operation": "restore",
                "root": str(root),
                "entry": entry,
                "approval": alternative["approval"],
                "proposal": proposal,
            }
            self._pack_identities[plan_id] = None
            alternatives.append({**alternative, "planId": plan_id})
        while len(self._plans) > max(16, len(alternatives)):
            oldest = next(iter(self._plans))
            del self._plans[oldest]
            del self._pack_identities[oldest]
        return {**proposal, "root": str(root), "alternatives": alternatives}

    async def _run(
        self,
        job: dict[str, object],
        plan: dict[str, object],
        progress: Path,
        pack_identity: str | None,
    ) -> None:
        root = Path(cast("str", job["root"]))
        operation = cast("str", job["operation"])
        outcome: dict[str, object] = {
            "state": "failed",
            "error": {
                "code": f"{operation}-interrupted",
                "message": "Installation job was interrupted.",
            },
        }
        try:
            if operation != "check" and self._prepare is not None:
                await self._prepare()
            # Only setup allocates a listener; maintenance preserves assignments.
            context = installation_context(root)
            if operation == "check":
                arguments = ["doctor"]
            elif operation == "restore":
                arguments = [
                    "restore",
                    str(plan["entry"]),
                    "--approve",
                    cast("str", plan["approval"]),
                ]
            else:
                if operation == "update":
                    assignments = ["--selection-approval", cast("str", plan["selectionApproval"])]
                elif context is not None:
                    port = cast("int", await asyncio.to_thread(context.call, root, "port"))
                    assignments = ["--port", str(port)]
                else:
                    with socket.socket() as listener:
                        listener.bind(("127.0.0.1", 0))
                        port = listener.getsockname()[1]
                    assignments = ["--port", str(port)]
                arguments = [
                    operation,
                    "--channel",
                    cast("str", plan["channel"]),
                    "--variant",
                    cast("str", plan["variant"]),
                    "--engine-release",
                    cast("str", plan["engineRelease"]),
                    "--base-id",
                    cast("str", plan["baseId"]),
                    *(("--pack-estimate-identity", pack_identity) if pack_identity else ()),
                    *assignments,
                    *(
                        argument
                        for gpu in cast("list[str]", plan["gpus"])
                        for argument in ("--gpu", gpu)
                    ),
                ]
            result = await environment_command(
                root,
                *arguments,
                "--progress",
                str(progress),
                *(("--yes",) if operation != "check" else ()),
            )
            if result["status"] == "cancelled":
                identity = (
                    await asyncio.to_thread(cancelled_setup_identity, root)
                    if operation == "setup"
                    else None
                )
                if identity is not None:
                    self._cancelled_roots[root] = identity
                    while len(self._cancelled_roots) > 16:
                        del self._cancelled_roots[next(iter(self._cancelled_roots))]
                outcome = {"state": "cancelled"}
            else:
                if operation == "setup":
                    await self._completed(root)
                outcome = {
                    "state": "completed",
                    "installation": {"root": str(root), "engine": "main"},
                    **({"result": result} if operation == "check" else {}),
                }
        except (OSError, ValueError, LookupError, RuntimeError) as exc:
            outcome = {
                "state": "failed",
                "error": {"code": getattr(exc, "code", f"{operation}-failed"), "message": str(exc)},
            }
        finally:
            try:
                async with self._progress_lock:
                    context = installation_context(root)
                    if context is not None:
                        await asyncio.to_thread(context.call, root, "progress", job["id"], "clean")
                    else:
                        progress.unlink(missing_ok=True)
                        progress.with_name(progress.name + ".cancel").unlink(missing_ok=True)
                if job["operation"] in {"update", "restore"} and self._updated is not None:
                    await self._updated(root)
            except (OSError, ValueError, RuntimeError) as exc:
                outcome = {
                    "state": "failed",
                    "error": {"code": f"{operation}-cleanup-failed", "message": str(exc)},
                }
            finally:
                job.update(outcome)
                self._running_plan = None

    async def get(self, job_id: str) -> dict[str, object]:
        job = self._jobs[job_id]
        if job["state"] in {"running", "cancelling"} and self._progress is not None:
            with contextlib.suppress(OSError, ValueError):
                root = Path(cast("str", job["root"]))
                context = installation_context(root)
                observed = (
                    await asyncio.to_thread(read_document, self._progress)
                    if context is None
                    else cast(
                        "dict[str, object]",
                        await asyncio.to_thread(context.call, root, "progress", job_id, "read"),
                    )
                )
                total = observed["total"]
                job["progress"] = {
                    "stage": observed["stage"],
                    "completed": observed["completed"],
                    "total": total or None,
                    "unit": "bytes" if observed["stage"] == "downloading" else "items",
                }
        return job

    async def cancel(self, job_id: str) -> dict[str, object]:
        job = self._jobs[job_id]
        async with self._progress_lock:
            if job["state"] in {"running", "cancelling"}:
                assert self._progress is not None
                root = Path(cast("str", job["root"]))
                context = installation_context(root)
                if context is not None:
                    await asyncio.to_thread(context.call, root, "progress", job_id, "cancel")
                else:
                    self._progress.parent.mkdir(parents=True, exist_ok=True)
                    self._progress.with_name(self._progress.name + ".cancel").touch()
                if job["state"] == "running":
                    job["state"] = "cancelling"
        return job

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._task is not None and not self._task.done():
                assert self.active_job is not None
                await self.cancel(cast("str", self.active_job["id"]))
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
        active = jobs.active_job
        return web.json_response(
            {
                "plan": plan,
                "activeJob": None if active is None else await jobs.get(cast("str", active["id"])),
            }
        )
    except (OSError, ValueError) as exc:
        return web.json_response(
            {"error": getattr(exc, "code", "invalid-setup"), "message": str(exc)}, status=400
        )


async def handle_start(request: web.Request) -> web.Response:
    try:
        raw: object = await request.json()
        if not isinstance(raw, dict):
            raise ValueError("Supply only a reviewed planId.")
        body = cast("dict[str, object]", raw)
        if set(body) != {"planId"} or not isinstance(body["planId"], str):
            raise ValueError("Supply only a reviewed planId.")
        operation = {
            "/supervisor/installations/update": "update",
            "/supervisor/installations/restore": "restore",
            "/supervisor/installations/check": "check",
        }.get(request.path, "setup")
        job = await request.app[SETUP_KEY].start(body["planId"], operation=operation)
        return web.json_response(job, status=202)
    except KeyError:
        return web.json_response(
            {"error": "unknown-plan", "message": "Review a new installation plan."}, status=404
        )
    except SetupRefused as exc:
        return web.json_response({"error": exc.code, "message": str(exc)}, status=409)
    except (OSError, ValueError) as exc:
        return web.json_response({"error": "invalid-setup", "message": str(exc)}, status=400)


async def handle_job(request: web.Request) -> web.Response:
    try:
        jobs = request.app[SETUP_KEY]
        job = (
            await jobs.cancel(request.match_info["id"])
            if request.method == "DELETE"
            else await jobs.get(request.match_info["id"])
        )
        status = 202 if request.method == "DELETE" and job["state"] == "cancelling" else 200
        return web.json_response(job, status=status)
    except KeyError:
        return web.json_response(
            {
                "error": "unknown-job",
                "message": "That installation job is not known to this process.",
            },
            status=404,
        )
    except (OSError, ValueError) as exc:
        return web.json_response({"error": "setup-unavailable", "message": str(exc)}, status=503)
