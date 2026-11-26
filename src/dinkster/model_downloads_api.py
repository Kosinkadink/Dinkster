"""Operator-owned provider selections and bounded, explicitly consented transfers."""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

from aiohttp import web
from dinkster_assets import AssetError, AssetRef, MountDef, open_verified
from dinkster_assets.civitai import CivitaiFile, select_civitai_file
from dinkster_assets.fetch import FetchAborted, ModelAccessDenied, open_model_response
from dinkster_assets.huggingface import HuggingFaceFile, select_huggingface_file
from dinkster_assets.model import model_is_hidden
from dinkster_assets.mounts import MountsError
from dinkster_server.auth import principal_for, resolve_scope
from dinkster_server.lookups import announce_asset_kinds
from dinkster_values.asset_folders import use_asset_folders

from .model_downloads import ModelDownloads
from .model_settings import MODEL_FOLDERS, filesystem_space
from .project_folders import ProjectFolders
from .provider_credentials import ProviderCredentialsError, ProviderCredentialsExpired
from .provider_sign_in import add_provider_sign_in_routes
from .template_model_sources import TemplateModelSources


@dataclass(frozen=True)
class _Plan:
    owner: str
    scope: str
    project: str
    kind: str
    target: MountDef
    selected: HuggingFaceFile | CivitaiFile
    root: Path
    credential_revision: str | None = field(repr=False)
    digest: str | None
    destination_space: dict[str, object]
    expires: float = field(default_factory=lambda: time.monotonic() + 900)

    def wire(self) -> dict[str, object]:
        selected = self.selected
        return {
            "name": Path(selected.file).name,
            **({"digest": self.digest} if self.digest is not None else {}),
            "kind": self.kind,
            "project": self.project,
            "provider": "civitai" if isinstance(selected, CivitaiFile) else "huggingface",
            "source": selected.url,
            "repo": selected.repo,
            "revision": selected.revision,
            "file": selected.file,
            "sha256": selected.sha256,
            "size": selected.size,
            "license": selected.license,
            "destination": str(self.target.path / Path(selected.file).name),
            "mount": self.target.id,
        }


@dataclass
class _Job:
    plan_id: str
    plan: _Plan
    status: str = "starting"
    received: int = 0
    reused: bool = False
    result: dict[str, object] = field(default_factory=dict[str, object])
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def progress(self, status: str, received: int) -> None:
        with self.lock:
            if self.cancel.is_set():
                raise FetchAborted()
            if status == "reusing":
                self.reused = True
            else:
                self.status, self.received = status, received

    def publish(self, _ref: AssetRef, _url: str) -> None:
        self.progress("publishing", self.plan.selected.size)

    def wire(self, job_id: str) -> dict[str, object]:
        with self.lock:
            return {
                **self.plan.wire(),
                "jobId": job_id,
                "status": self.status,
                "received": self.received,
                **self.result,
            }


def add_model_download_routes(app: web.Application, downloads: ModelDownloads) -> None:
    sources = TemplateModelSources()
    planning = asyncio.Semaphore(4)
    plans: dict[str, _Plan] = {}
    jobs: dict[str, _Job] = {}
    tasks: set[asyncio.Task[None]] = set()
    instance = secrets.token_urlsafe(16)
    credential_updates = asyncio.Lock()
    credential_generation = 0

    def invalidate_credentials() -> None:
        nonlocal credential_generation
        credential_generation += 1
        for plan_id, plan in tuple(plans.items()):
            if plan.credential_revision is not None:
                del plans[plan_id]
        for job in jobs.values():
            with job.lock:
                if job.plan.credential_revision is not None and job.status not in {
                    "publishing",
                    "indexing",
                    "completed",
                }:
                    job.cancel.set()

    def validate_token(token: str) -> None:
        with open_model_response(
            "https://huggingface.co/api/whoami-v2", huggingface_token=token, timeout=15
        ) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError()
        raw: object = json.loads(payload)
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
            raise ValueError()

    async def handle_credentials(request: web.Request) -> web.Response:
        authority(
            request,
            "assets:read" if request.method == "GET" else "settings:write",
            request.query.get("scope"),
        )
        try:
            if request.method == "GET":
                configured = await asyncio.to_thread(downloads.credentials.token)
                return web.json_response(
                    {"provider": "huggingface", "configured": configured is not None},
                    headers={"Cache-Control": "no-store"},
                )
            token: str | None = None
            if request.method == "PUT":
                raw: object = await request.json()
                if not isinstance(raw, dict) or set(raw) != {"token"}:
                    raise ValueError()
                candidate = raw["token"]
                if (
                    not isinstance(candidate, str)
                    or not candidate
                    or len(candidate) > 8192
                    or any(not 33 <= ord(character) <= 126 for character in candidate)
                ):
                    raise ValueError()
                token = candidate
            async with credential_updates:
                if token is not None:
                    await asyncio.to_thread(validate_token, token)
                invalidate_credentials()
                saving = asyncio.create_task(
                    asyncio.to_thread(downloads.credentials.set_token, token)
                )
                try:
                    await asyncio.shield(saving)
                except asyncio.CancelledError:
                    await saving
                    raise
                finally:
                    invalidate_credentials()
            return web.json_response(
                {"provider": "huggingface", "configured": token is not None},
                headers={"Cache-Control": "no-store"},
            )
        except ProviderCredentialsExpired:
            return web.json_response(
                {"provider": "huggingface", "configured": False, "expired": True},
                headers={"Cache-Control": "no-store"},
            )
        except ModelAccessDenied:
            return web.json_response(
                {"error": "credential-invalid"}, status=401, headers={"Cache-Control": "no-store"}
            )
        except (AssetError, ProviderCredentialsError, OSError, HTTPException):
            return web.json_response(
                {"error": "provider-credentials-unavailable"},
                status=409,
                headers={"Cache-Control": "no-store"},
            )
        except ValueError:
            return web.json_response(
                {"error": "invalid-provider-credential"},
                status=400,
                headers={"Cache-Control": "no-store"},
            )

    def access_requirement(error: ModelAccessDenied) -> dict[str, object]:
        if error.provider == "civitai":
            return {"error": "civitai-access-required", "provider": "civitai"}
        return {
            "error": "credential-required"
            if (error.status == 401 and not error.gated) or not error.authenticated
            else "provider-grant-required",
            "provider": "huggingface",
            "requirementId": secrets.token_urlsafe(24),
        }

    def prune() -> None:
        now = time.monotonic()
        for plan_id in tuple(plans):
            if plans[plan_id].expires < now:
                del plans[plan_id]
        for job_id, job in tuple(jobs.items()):
            if job.finished is not None and job.finished + 3600 < now:
                del jobs[job_id]

    def authority(request: web.Request, capability: str, scope: object) -> tuple[str, str]:
        principal = principal_for(request)
        if not principal.is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}', content_type="application/json"
            )
        return principal.principal_id, resolve_scope(principal, capability, scope)

    def local_status(
        name: str, kind: str, project: str, asset: AssetRef | None = None
    ) -> dict[str, object] | None:
        """Read project-granted local identity or name bindings without a source lookup."""
        mounts = downloads.mounts
        context = mounts.project_context(project)
        with use_asset_folders(context):
            if asset is not None:
                if downloads.hidden_digest(asset.digest, project):
                    return {
                        "status": "model-hidden",
                        "detail": "Model hidden for this project; unhide it in model settings",
                    }
                refs = [
                    ref
                    for mount in mounts.table.ready_snapshot()
                    if mount.mount_id in context.mounts and mount.asset_kind == kind
                    for ref in mount.refs
                    if ref.digest == asset.digest
                ]
                refs.extend(mounts.table.model_file_refs(kind))
                hidden = False
                for ref in refs:
                    if ref.digest != asset.digest:
                        continue
                    try:
                        path = mounts.table.resolve_location(ref.virtual_path, asset.digest)
                        if path is None:
                            continue
                        if model_is_hidden(path, asset.digest):
                            hidden = True
                            continue
                        with open_verified(path, asset.digest):
                            return {"status": "local", "name": name, "kind": kind}
                    except (AssetError, OSError):
                        pass
                if hidden:
                    return {
                        "status": "model-hidden",
                        "detail": "Model hidden for this project; unhide it in model settings",
                    }
                disabled = downloads.disabled_locations(asset.digest, project, kind=kind)
                if disabled:
                    return {
                        "status": "folder-disabled",
                        "disabledLocations": list(disabled),
                        "detail": "Folder not enabled for this project",
                    }
                return None
            if mounts.table.bind_name(name, kind) is not None:
                return {"status": "local", "name": name, "kind": kind}
            if mounts.table.name_is_ambiguous(name, kind):
                return {"status": "duplicate"}
            if downloads.hidden_name(name, kind):
                return {
                    "status": "model-hidden",
                    "detail": "Model hidden for this project; unhide it in model settings",
                }
            disabled = downloads.disabled_name_locations(name, kind)
            if disabled:
                return {
                    "status": "folder-disabled",
                    "disabledLocations": list(disabled),
                    "detail": "Folder not enabled for this project",
                }
        return None

    def plan(
        name: str,
        kind: str,
        project: str,
        owner: str,
        scope: str,
        source: str | None,
        asset: AssetRef | None,
    ) -> _Plan | dict[str, object]:
        found = local_status(name, kind, project, asset)
        if found is not None:
            return found
        candidates = (source,) if source is not None else sources.for_name(name, kind)
        if len(candidates) != 1:
            return {"status": "picker-required", "sources": list(candidates)}
        target = downloads.target(kind, project)
        if target is None:
            return {"status": "destination-required"}
        credential_revision = None
        if urlsplit(candidates[0]).netloc == "civitai.com":
            selected = select_civitai_file(candidates[0], name)
            if isinstance(selected, tuple):
                return {"status": "picker-required", "sources": list(selected)}
        else:
            token, credential_revision = downloads.credentials.snapshot()
            selected = select_huggingface_file(candidates[0], token=token)
        root = target.path.resolve()
        return _Plan(
            owner,
            scope,
            project,
            kind,
            target,
            selected,
            root,
            credential_revision,
            asset.digest if asset is not None else None,
            filesystem_space(root),
        )

    def plannable(name: str) -> bool:
        return (
            0 < len(name) <= 255
            and name not in {".", ".."}
            and not any(ord(char) < 32 or ord(char) == 127 or char in "/\\" for char in name)
        )

    async def handle_presence(request: web.Request) -> web.Response:
        """Report whether model names bind for a project, without planning or network."""
        try:
            raw: object = await request.json()
            if not isinstance(raw, dict) or set(cast("dict[str, object]", raw)) != {
                "models",
                "project",
                "scope",
            }:
                raise ValueError("expected models, project and scope")
            body = cast("dict[str, object]", raw)
            project, models = body["project"], body["models"]
            if not isinstance(project, str) or not isinstance(models, list):
                raise ValueError("expected a project and a list of models")
            entries = cast("list[object]", models)
            if not 0 < len(entries) <= 256:
                raise ValueError("expected 1 to 256 models")
            wanted: list[tuple[str, str]] = []
            for entry in entries:
                if not isinstance(entry, dict) or set(cast("dict[str, object]", entry)) != {
                    "name",
                    "kind",
                }:
                    raise ValueError("each model is a name and a kind")
                model = cast("dict[str, object]", entry)
                name, kind = model["name"], model["kind"]
                if (
                    not isinstance(name, str)
                    or not name
                    or len(name) > 4096
                    or not isinstance(kind, str)
                    or kind not in MODEL_FOLDERS
                ):
                    raise ValueError("expected a model name and supported kind")
                wanted.append((name, kind))
            ProjectFolders.validate_project(project)
            authority(request, "assets:read", body["scope"])
        except (ValueError, MountsError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

        def read() -> list[dict[str, object]]:
            result: list[dict[str, object]] = []
            for name, kind in wanted:
                found = local_status(name, kind, project)
                if found is not None:
                    result.append({**found, "name": name, "kind": kind})
                    continue
                result.append(
                    {
                        "name": name,
                        "kind": kind,
                        "status": "missing",
                        "plannable": plannable(name),
                    }
                )
            return result

        try:
            models_wire = await asyncio.to_thread(read)
        except (AssetError, OSError):
            return web.json_response(
                {"error": "model-folders-unavailable"},
                status=409,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response({"models": models_wire}, headers={"Cache-Control": "no-store"})

    async def handle_plan(request: web.Request) -> web.Response:
        if not principal_for(request).is_operator:
            raise web.HTTPForbidden(
                text='{"error":"operator-required"}', content_type="application/json"
            )
        try:
            raw: object = await request.json()
            if not isinstance(raw, dict):
                raise ValueError("expected name, kind, project and scope")
            body = cast("dict[str, object]", raw)
            required = {"name", "kind", "project", "scope"}
            if not required <= set(body) or set(body) - required - {"source", "asset"}:
                raise ValueError("expected name, kind, project, scope and optional source or asset")
            name, kind, project = body["name"], body["kind"], body["project"]
            if (
                not isinstance(name, str)
                or not plannable(name)
                or not isinstance(kind, str)
                or kind not in MODEL_FOLDERS
                or not isinstance(project, str)
            ):
                raise ValueError("expected a bare model name, supported kind and project")
            asset = None
            if "asset" in body:
                wire = body["asset"]
                if not isinstance(wire, dict):
                    raise ValueError("expected an exact asset reference")
                try:
                    asset = AssetRef.from_wire(cast("dict[str, object]", wire))
                except AssetError as exc:
                    raise ValueError("expected an exact asset reference") from exc
                if asset.name != name:
                    raise ValueError("asset name must match the model name")
            source = body.get("source")
            if "source" in body:
                if not isinstance(source, str) or not source or len(source) > 8192:
                    raise ValueError("expected a model source URL")
                parsed = urlsplit(source)
                if (
                    parsed.scheme != "https"
                    or parsed.netloc not in {"huggingface.co", "civitai.com"}
                    or parsed.fragment
                    or any(ord(char) <= 32 or ord(char) == 127 for char in source)
                    or (
                        parsed.netloc == "huggingface.co"
                        and (parsed.query or unquote(parsed.path).rsplit("/", 1)[-1] != name)
                    )
                ):
                    raise ValueError("source must identify a Hugging Face file or Civitai model")
            ProjectFolders.validate_project(project)
            owner, scope = authority(request, "assets:read", body["scope"])
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        prune()
        if planning.locked() or len(plans) >= 64:
            return web.json_response({"error": "model-planning-busy"}, status=429)
        async with planning:
            generation = credential_generation
            try:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        plan, name, kind, project, owner, scope, cast("str | None", source), asset
                    )
                )
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            except ProviderCredentialsExpired:
                return web.json_response(
                    {
                        "error": "credential-required",
                        "provider": "huggingface",
                        "expired": True,
                        "requirementId": secrets.token_urlsafe(24),
                    },
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
            except ModelAccessDenied as exc:
                return web.json_response(
                    access_requirement(exc),
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
            except (AssetError, ProviderCredentialsError, OSError, HTTPException):
                return web.json_response(
                    {"error": "model-source-unavailable"},
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
        if (
            generation != credential_generation
            and isinstance(result, _Plan)
            and result.credential_revision is not None
        ):
            return web.json_response(
                {"error": "provider-credentials-changed"},
                status=409,
                headers={"Cache-Control": "no-store"},
            )
        if isinstance(result, _Plan):
            if len(plans) >= 64:
                return web.json_response({"error": "model-planning-busy"}, status=429)
            plan_id = secrets.token_urlsafe(24)
            plans[plan_id] = result
            document = {
                **result.wire(),
                "status": "consent-required",
                "planId": plan_id,
                "destinationSpace": result.destination_space,
            }
        else:
            document = result
        return web.json_response(document, headers={"Cache-Control": "no-store"})

    async def run(job: _Job) -> None:
        try:
            job.progress("starting", 0)
            ref = await asyncio.to_thread(
                downloads.download_selected,
                job.plan.selected,
                job.plan.target,
                job.plan.project,
                job.publish,
                digest=job.plan.digest,
                progress=job.progress,
                expected_root=job.plan.root,
                expected_credential_revision=job.plan.credential_revision,
            )
            job.progress("indexing", ref.size)
            if ref.virtual_path.startswith("model-files/"):
                with use_asset_folders(downloads.mounts.project_context(job.plan.project)):
                    actual_path = downloads.mounts.table.ref(ref.virtual_path).local_path()
            else:
                actual_path = job.plan.root / ref.virtual_path.split("/", 2)[2]
            with job.lock:
                job.result = {
                    "asset": ref.to_wire(),
                    "path": str(actual_path),
                    "reused": job.reused,
                    "renamed": not job.reused and ref.name != Path(job.plan.selected.file).name,
                }
            try:
                await asyncio.to_thread(
                    downloads.mounts.table.refresh_publication, job.plan.target.id
                )
            except (AssetError, OSError):
                with job.lock:
                    job.result["indexingPending"] = True
            else:
                announce_asset_kinds(app, (job.plan.target.kind,))
            with job.lock:
                job.status = "completed"
        except FetchAborted:
            with job.lock:
                job.status = "cancelled"
        except ProviderCredentialsExpired:
            with job.lock:
                job.status = "failed"
                job.result = {
                    "error": "credential-required",
                    "provider": "huggingface",
                    "expired": True,
                    "requirementId": secrets.token_urlsafe(24),
                }
        except ModelAccessDenied as exc:
            with job.lock:
                job.status = "failed"
                job.result = access_requirement(exc)
        except Exception:
            with job.lock:
                job.status = "failed"
                job.result = {"error": "model-download-failed"}
        finally:
            job.finished = time.monotonic()

    async def handle_start(request: web.Request) -> web.Response:
        owner, scope = authority(request, "assets:write", request.query.get("scope"))
        resolve_scope(principal_for(request), "jobs:submit", scope)
        try:
            raw: object = await request.json()
            if not isinstance(raw, dict) or set(raw) != {"planId", "consent"}:
                raise ValueError("expected planId and explicit consent")
            body = cast("dict[str, object]", raw)
            if not isinstance(body["planId"], str) or body["consent"] is not True:
                raise ValueError("expected planId and explicit consent")
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        prune()
        selected = plans.get(body["planId"])
        if selected is None or (selected.owner, selected.scope) != (owner, scope):
            return web.json_response({"error": "download-plan-unavailable"}, status=404)
        with use_asset_folders(downloads.mounts.project_context(selected.project)):
            if (
                downloads.hidden_digest(selected.digest, selected.project)
                if selected.digest is not None
                else downloads.hidden_name(Path(selected.selected.file).name, selected.kind)
            ):
                return web.json_response({"error": "model-hidden"}, status=409)
        try:
            credential_revision = None
            if selected.credential_revision is not None:
                _, credential_revision = await asyncio.to_thread(downloads.credentials.snapshot)
        except ProviderCredentialsExpired:
            return web.json_response(
                {
                    "error": "credential-required",
                    "provider": "huggingface",
                    "expired": True,
                    "requirementId": secrets.token_urlsafe(24),
                },
                status=409,
            )
        except (ProviderCredentialsError, AssetError, OSError, HTTPException):
            return web.json_response({"error": "provider-credentials-unavailable"}, status=409)
        if plans.get(body["planId"]) is not selected or (
            credential_revision != selected.credential_revision
        ):
            return web.json_response({"error": "provider-credentials-changed"}, status=409)
        for job_id, job in jobs.items():
            if job.plan_id == body["planId"]:
                return web.json_response(job.wire(job_id), status=200)
        if (
            downloads.target(selected.kind, selected.project) != selected.target
            or selected.target.path.resolve() != selected.root
        ):
            return web.json_response({"error": "download-destination-changed"}, status=409)
        if (
            len(jobs) >= 64
            or len(tasks) >= 4
            or any(job.finished is None and job.plan.root == selected.root for job in jobs.values())
        ):
            return web.json_response({"error": "model-download-busy"}, status=409)
        job_id = secrets.token_urlsafe(24)
        job = _Job(body["planId"], selected)
        jobs[job_id] = job
        task = asyncio.create_task(run(job))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return web.json_response(
            job.wire(job_id), status=202, headers={"Cache-Control": "no-store"}
        )

    async def handle_jobs(request: web.Request) -> web.Response:
        owner, scope = authority(request, "assets:read", request.query.get("scope"))
        resolve_scope(principal_for(request), "jobs:read", scope)
        prune()
        job_id = request.match_info.get("jobId")
        if job_id is None:
            return web.json_response(
                {
                    "instance": instance,
                    "jobs": [
                        job.wire(key)
                        for key, job in jobs.items()
                        if (job.plan.owner, job.plan.scope) == (owner, scope)
                    ],
                },
                headers={"Cache-Control": "no-store"},
            )
        job = jobs.get(job_id)
        if job is None or (job.plan.owner, job.plan.scope) != (owner, scope):
            return web.json_response({"error": "download-job-unavailable"}, status=404)
        if request.method == "DELETE":
            resolve_scope(principal_for(request), "assets:write", scope)
            resolve_scope(principal_for(request), "jobs:cancel", scope)
            with job.lock:
                if job.status in {"publishing", "indexing", "completed"}:
                    return web.json_response({"error": "download-publication-started"}, status=409)
                if job.finished is None:
                    job.cancel.set()
                    job.status = "cancelling"
        return web.json_response(job.wire(job_id), headers={"Cache-Control": "no-store"})

    async def cleanup(_app: web.Application) -> None:
        for job in jobs.values():
            with job.lock:
                if job.status not in {"publishing", "indexing", "completed"}:
                    job.cancel.set()
        if tasks:
            await asyncio.gather(*tasks)

    app.on_cleanup.append(cleanup)
    add_provider_sign_in_routes(app, downloads.credentials, invalidate_credentials)
    app.router.add_get("/api/models/providers/huggingface", handle_credentials)
    app.router.add_put("/api/models/providers/huggingface", handle_credentials)
    app.router.add_delete("/api/models/providers/huggingface", handle_credentials)
    app.router.add_post("/api/models/presence", handle_presence)
    app.router.add_post("/api/models/downloads/plan", handle_plan)
    app.router.add_post("/api/models/downloads/start", handle_start)
    app.router.add_get("/api/models/downloads", handle_jobs)
    app.router.add_get("/api/models/downloads/{jobId}", handle_jobs)
    app.router.add_delete("/api/models/downloads/{jobId}", handle_jobs)
