"""Filesystem mounts API: the operator's directory-grant surface.

Umbrella-owned wiring like reload_api: the MountTable is
composed in dinkster-serve, dinkster-server stays mount-agnostic, and this
module is the seam.

- GET    /api/mounts                     every mount: id, mode, source,
                                         state, entryCount/scanProgress/error; operator
                                         mounts include real path, semantic
                                         model mounts include kind instead
- GET    /api/mounts/{id}/list?path=     one level of a mount's virtual
                                         tree (file-picker navigation)
- GET    /api/mounts/{id}/entries?q=&path=&recursive=&cursor=&limit=
                                         query-first, cursor-paged catalog
                                         browse; the cursor binds every
                                         result-shaping filter (400 on
                                         mismatch)
- POST   /api/mounts                     {id, path, mode?, priority?} grant a
                                         directory at runtime
- DELETE /api/mounts/{id}                revoke a config-sourced grant

The mutation half is policy-gated (--allow-mount-changes): the routes
are always registered but answer 403 {"error": "mount-changes-disabled"}
when the operator did not opt in - a machine-readable refusal in the
engine-not-ready mold, so a client can tell "disabled by policy" from
"no such endpoint". Runtime changes are DURABLE:
config-sourced mounts rewrite mounts.toml atomically, so a folder granted
from the desktop shell survives a restart. Derived mounts (a ComfyUI
install's own directories) are re-derived each boot: they never persist
and cannot be deleted here - re-point the config instead.

Every mutation and every finished scan publishes {"type":
"mounts_changed"} non-droppable, the schema_changed pattern: a ping to
refetch GET /api/mounts, never a diff.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from aiohttp import web
from dinkster_assets import (
    AssetEntry,
    AssetError,
    AssetScanProgress,
    MountDef,
    MountsError,
    MountTable,
    dump_mounts,
    is_asset_kind,
)
from dinkster_assets.identity import require_digest
from dinkster_assets.integrity import verified_local_path
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import MountMaterialization, ResolutionStore
from dinkster_protocol import AssetFolderContext
from dinkster_schema import core_logger
from dinkster_server import STATE_KEY, decode_cursor, encode_cursor
from dinkster_server.auth import principal_for, resolve_scope
from dinkster_server.library import LIBRARY_KEY, run_owned_thread

from .project_folders import ProjectFolders

__all__ = ["MountService", "add_mount_routes"]

MOUNTS_KEY: web.AppKey[MountService] = web.AppKey("dinkster_mounts")

_log = core_logger("mounts")

_PAGE_LIMIT_DEFAULT = 100
_PAGE_LIMIT_MAX = 500
_SCAN_PROGRESS_INTERVAL = 1.0


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


class MountService:
    """The mount coordinator: one table, one persistence file, one scan
    lane. Scans run one at a time on a thread (hashing a model library
    is hours of IO in the worst case; two scans at once would just fight
    over the disk) and each completion publishes mounts_changed."""

    def __init__(
        self,
        table: MountTable,
        config_path: Path | None = None,
        *,
        allow_changes: bool = False,
        restart_for_changes: bool = False,
    ) -> None:
        self.table = table
        self.allow_changes = allow_changes
        self.restart_for_changes = restart_for_changes
        self.settings_table = table
        if restart_for_changes:
            self.settings_table = self.copy_settings()
        self._config_path = config_path
        self._scan_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._resolution_store: ResolutionStore | None = None
        self._resolution_scopes: tuple[str, ...] = ()
        self.projects = ProjectFolders(
            config_path.with_name("project-folders.json") if config_path is not None else None
        )
        self.projects.initialize(self.mount_ids())

    def copy_settings(self) -> MountTable:
        table = MountTable(output_mount=self.settings_table.output_mount)
        for row in self.settings_table.descriptors():
            mount_id = str(row["id"])
            mount = self.settings_table.get(mount_id)
            assert mount is not None
            table.add(mount, source=str(row["source"]), kind=self.settings_table.kind(mount_id))
        return table

    def mount_ids(self, *, settings: bool = False) -> tuple[str, ...]:
        table = self.settings_table if settings else self.table
        rows = sorted(
            table.descriptors(), key=lambda row: (cast(int, row["priority"]), str(row["id"]))
        )
        return tuple(str(row["id"]) for row in rows)

    def settings_descriptors(self) -> list[dict[str, object]]:
        live = {str(row["id"]): row for row in self.table.descriptors()}
        rows = self.settings_table.descriptors()
        for row in rows:
            mount_id = str(row["id"])
            if self.table.get(mount_id) == self.settings_table.get(mount_id):
                row.update(live[mount_id])
            else:
                row["state"] = "restart-required"
        return rows

    def pending_changes(self) -> dict[str, object]:
        removed = sorted(set(self.mount_ids()) - set(self.mount_ids(settings=True)))
        changed = self.restart_for_changes and (
            {mount.id: mount for mount in self.table.config_defs()}
            != {mount.id: mount for mount in self.settings_table.config_defs()}
            or self.table.output_mount != self.settings_table.output_mount
        )
        return {"restartRequired": changed, "pendingRemovals": removed}

    def project_response(self, project: str) -> dict[str, object]:
        order = self.projects.ordered(project, self.mount_ids(settings=True))
        rows = {str(row["id"]): row for row in self.settings_descriptors()}
        selected = self.projects.output(project, self.settings_table.output_mount)
        return {
            "project": project,
            "mounts": [rows[mount_id] for mount_id in order],
            "outputMount": selected if selected in order else None,
            "activeOutputMount": self.project_context(project).output_mount,
            "mountChangesAllowed": self.allow_changes,
            **self.pending_changes(),
        }

    def project_context(self, project: str) -> AssetFolderContext:
        order = self.projects.ordered(project, self.mount_ids())
        output = self.projects.output(project, self.table.output_mount)
        return AssetFolderContext(order, output if output in order else None)

    def attach_resolution_store(self, store: ResolutionStore, scopes: tuple[str, ...]) -> None:
        if scopes != tuple(sorted(set(scopes))) or any(not scope.strip() for scope in scopes):
            raise AssetError("resolution scopes must be sorted, unique, and nonempty")
        self._resolution_store = store
        self._resolution_scopes = scopes
        self.refresh_resolution_store()

    def refresh_resolution_store(self) -> None:
        if self._resolution_store is None:
            return
        ready = self.table.ready_snapshot()
        rows = tuple(
            MountMaterialization(
                scope,
                mount.mount_id,
                mount.priority,
                mount.asset_kind,
                ref,
            )
            for scope in self._resolution_scopes
            for mount in ready
            for ref in mount.refs
        )
        self._resolution_store.replace_mount_snapshot(self._resolution_scopes, rows)

    def persist(self, table: MountTable | None = None) -> None:
        """Durably replace settings before publishing changed live grants."""
        if self._config_path is None:
            return
        table = self.settings_table if table is None else table
        mounts = table.config_defs()
        configured_ids = {mount.id for mount in mounts}
        selected = table.output_mount
        text = dump_mounts(
            mounts,
            output_mount=selected if selected in configured_ids else None,
        )
        tmp = self._config_path.with_name(self._config_path.name + f".tmp-{os.getpid()}")
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, self._config_path)
        finally:
            tmp.unlink(missing_ok=True)

    async def scan(self, app: web.Application, mount_id: str) -> None:
        """Scan one mount off-loop and announce the change. A failed scan
        is recorded on the mount's row (state "failed" + error), never
        raised past here - one unreadable directory must not take down
        the scan lane."""
        async with self._scan_lock:
            loop = asyncio.get_running_loop()
            announced_files_done = 0

            def publish_progress(progress: AssetScanProgress) -> None:
                _log.info(
                    "mount %s scan: %d/%d files, %d/%d bytes, %.1fs elapsed",
                    mount_id,
                    progress.files_done,
                    progress.files_total,
                    progress.bytes_done,
                    progress.bytes_total,
                    progress.elapsed_seconds,
                )

                def announce() -> None:
                    nonlocal announced_files_done
                    if progress.files_done == announced_files_done:
                        return
                    announced_files_done = progress.files_done
                    self.refresh_resolution_store()
                    self.publish(app)

                try:
                    loop.call_soon_threadsafe(announce)
                except RuntimeError:
                    pass  # the hashing thread may outlive server shutdown

            scan_task: asyncio.Task[int] | None = None
            try:
                scan_task = asyncio.create_task(
                    asyncio.to_thread(
                        self.table.scan,
                        mount_id,
                        on_progress=publish_progress,
                        progress_interval=_SCAN_PROGRESS_INTERVAL,
                    )
                )
                while not scan_task.done():
                    done, _ = await asyncio.wait({scan_task}, timeout=_SCAN_PROGRESS_INTERVAL)
                    if not done:
                        progress = self.table.scan_progress(mount_id)
                        if progress is not None:
                            publish_progress(progress)
                count = await scan_task
            except asyncio.CancelledError:
                if scan_task is not None:
                    scan_task.add_done_callback(
                        lambda task: None if task.cancelled() else task.exception()
                    )
                raise
            except MountsError:
                return  # removed while queued: nothing to narrate
            except Exception as exc:  # noqa: BLE001 - recorded on the row
                _log.warning("mount %s scan failed: %s", mount_id, exc)
                self.refresh_resolution_store()
                self.publish(app)
                return
            _log.info(
                "mount %s: %d entr%s cataloged",
                mount_id,
                count,
                "y" if count == 1 else "ies",
            )
            self.refresh_resolution_store()
            self.publish(app)

    def scan_soon(self, app: web.Application, mount_id: str) -> None:
        """Queue a scan behind whatever is already scanning (POST path)."""
        task = asyncio.create_task(self.scan(app, mount_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def scan_all(self, app: web.Application) -> None:
        for mount_id in self.table.pending():
            await self.scan(app, mount_id)

    async def close(self) -> None:
        """Cancel and await queued POST-path scans so shutdown never
        leaves orphaned tasks (the hashing thread itself runs to its next
        completion point; its result is simply discarded)."""
        for task in tuple(self._tasks):
            task.cancel()
        for task in tuple(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def publish(self, app: web.Application) -> None:
        """Announce that GET /api/mounts changed - a ping, never a diff."""
        app[STATE_KEY].hub.publish({"type": "mounts_changed"}, client_id=None, droppable=False)


async def handle_mounts_list(request: web.Request) -> web.Response:
    service = request.app[MOUNTS_KEY]
    try:
        body = service.project_response(request.query.get("project", "default"))
    except MountsError as exc:
        return _json_error(400, str(exc))
    if not _is_operator(request):
        body["mountChangesAllowed"] = False
        for row in cast(list[dict[str, object]], body["mounts"]):
            row.pop("path", None)
            if "error" in row:
                row["error"] = "mount unavailable"
    return web.json_response(body)


async def handle_mount_locations(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    allowed = {"scope", "project", "digest"}
    if set(request.query) - allowed or any(
        len(request.query.getall(field, [])) != 1 for field in request.query
    ):
        return _json_error(400, "unknown or repeated location query field")
    body: dict[str, object] = {"importMount": None, "resolvePath": True}
    library = request.app.get(LIBRARY_KEY)
    if library is not None:
        body["vault"] = str(library.vault.root.resolve())
        if library.upload_writer is not None and library.project_folders_for is not None:
            body["importMount"] = library.import_mount
    if "digest" in request.query:
        scope = resolve_scope(principal_for(request), "assets:read", request.query.get("scope"))
        project = request.query.get("project", "default")
        try:
            digest = require_digest(request.query["digest"])
            context = service.project_context(project)
            groups = (
                await asyncio.to_thread(
                    library.store.query_assets,
                    scope,
                    project=project,
                    digest=digest,
                    limit=1,
                )
                if library is not None
                else []
            )
        except (AssetError, MountsError):
            return _json_error(400, "invalid project or asset digest")
        locations: list[dict[str, object]] = []
        if groups and library is not None:
            asset = groups[0][-1].asset
            try:
                root = library.vault.root.resolve()
                path = library.vault.resolve(asset.digest)
                if path is not None and path.resolve().is_relative_to(root):
                    path = await asyncio.to_thread(
                        AssetRef(
                            asset.digest, asset.name, asset.size, resolver=library.vault
                        ).local_path
                    )
                    path = path.resolve()
                    if path.is_relative_to(root):
                        locations.append({"storage": "vault", "path": str(path)})
            except (AssetError, OSError):
                pass
            for virtual_path in dict.fromkeys(item.asset.virtual_path for item in groups[0]):
                parts = virtual_path.split("/", 2)
                if len(parts) != 3 or parts[0] != "mounts" or parts[1] not in context.mounts:
                    continue
                try:
                    ref = await asyncio.to_thread(service.table.ref, virtual_path)
                    if ref.digest != asset.digest:
                        continue
                    mount = service.table.get(parts[1])
                    if mount is None:
                        continue
                    root = mount.path.resolve()
                    path = (root / parts[2]).resolve()
                    if not path.is_relative_to(root):
                        continue
                    path = await asyncio.to_thread(verified_local_path, path, asset.digest)
                    if (
                        service.table.get(parts[1]) != mount
                        or parts[1] not in service.project_context(project).mounts
                        or not path.resolve().is_relative_to(root)
                    ):
                        continue
                except (AssetError, OSError):
                    continue
                location: dict[str, object] = {
                    "storage": "mount",
                    "mount": parts[1],
                    "virtualPath": virtual_path,
                    "path": str(path),
                }
                locations.append(location)
        body["locations"] = locations
    table = service.settings_table
    rows = service.settings_descriptors()
    for row in rows:
        mount = table.get(str(row["id"]))
        if mount is not None:
            row["path"] = str(mount.path)
    active_rows = service.table.descriptors()
    for row in active_rows:
        mount = service.table.get(str(row["id"]))
        if mount is not None:
            row["path"] = str(mount.path)
    body.update(mounts=rows, activeMounts=active_rows, **service.pending_changes())
    return web.json_response(body, headers={"Cache-Control": "no-store"})


async def handle_mount_resolve_path(request: web.Request) -> web.Response:
    _require_operator(request)

    def respond(body: dict[str, object], status: int = 200) -> web.Response:
        return web.json_response(body, status=status, headers={"Cache-Control": "no-store"})

    try:
        body = await request.json()
        if (
            request.query
            or not isinstance(body, dict)
            or set(body) != {"path", "project", "scope"}
            or any(not isinstance(body[key], str) or not body[key] for key in body)
            or "\0" in body["path"]
            or any(char.isspace() for char in body["scope"])
        ):
            return respond({"error": "invalid-path-request"}, 400)
        project, native_path = body["project"], body["path"]
        ProjectFolders.validate_project(project)
        path = Path(native_path)
        if not path.is_absolute() and not (
            PureWindowsPath(native_path).is_absolute() or PurePosixPath(native_path).is_absolute()
        ):
            return respond({"error": "invalid-path-request"}, 400)
    except (ValueError, MountsError):
        return respond({"error": "invalid-path-request"}, 400)
    resolve_scope(principal_for(request), "assets:read", body["scope"])
    refused: dict[str, object] = {"status": "refused", "reason": "not-resolvable"}
    if not path.is_absolute():
        return respond(refused)
    service = request.app[MOUNTS_KEY]
    try:
        # Exact-file indexing and whole-mount scans must not replace each other's indexes.
        async with service._scan_lock:
            context = service.project_context(project)
            grants = {mount_id: service.table.get(mount_id) for mount_id in context.mounts}
            ref = await run_owned_thread(lambda: service.table.ref_path(path, context.mounts))
            if ref is None:
                return respond(refused)
            mount_id = ref.virtual_path.split("/", 2)[1]
            if mount_id not in service.project_context(project).mounts or service.table.get(
                mount_id
            ) is not grants.get(mount_id):
                return respond(refused)
            service.refresh_resolution_store()
            service.publish(request.app)
            return respond({"status": "resolved", "asset": ref.to_wire()})
    except Exception:
        return respond({"error": "mount-resolution-failed"}, 500)


async def handle_mount_folder(request: web.Request) -> web.Response:
    service = request.app[MOUNTS_KEY]
    mount_id = request.match_info["mountId"]
    if service.table.get(mount_id) is None:
        return _json_error(404, f"unknown mount: {mount_id}")
    path = request.query.get("path", "")
    try:
        listing = service.table.list_folder(mount_id, path)
    except (AssetError, MountsError) as exc:
        return _json_error(400, str(exc))
    return web.json_response(
        {
            "folders": list(listing.folders),
            "entries": [
                _entry_wire(entry, service.table.kind(mount_id)) for entry in listing.entries
            ],
        }
    )


def _entry_kind(entry: AssetEntry, mount_kind: str) -> str:
    if mount_kind:
        return mount_kind
    major = entry.media_type.partition("/")[0]
    return f"media/{major}" if major in {"audio", "image", "video"} else ""


def _entry_wire(entry: AssetEntry, mount_kind: str = "") -> dict[str, object]:
    """The browse row a client turns into an AssetRef literal: the same
    fields the upload path returns, plus the virtual path for display."""
    wire: dict[str, object] = {
        "virtualPath": entry.virtual_path,
        "name": entry.name,
        "digest": entry.digest,
        "size": entry.size,
        "mediaType": entry.media_type,
    }
    kind = _entry_kind(entry, mount_kind)
    if kind:
        wire["kind"] = kind
    return wire


async def handle_mount_entries(request: web.Request) -> web.Response:
    """Query-first, cursor-paged browse of one mount's catalog. Sorted by
    virtual path ascending; the keyset is the last path served. ``q``
    filters by case-insensitive substring of the path below the mount;
    ``path`` and ``recursive`` optionally scope the folder traversal."""
    service = request.app[MOUNTS_KEY]
    mount_id = request.match_info["mountId"]
    if service.table.get(mount_id) is None:
        return _json_error(404, f"unknown mount: {mount_id}")
    query = request.query.get("q", "").strip()
    kind = request.query.get("kind", "").strip()
    if kind and not is_asset_kind(kind):
        return _json_error(400, "kind must be a namespaced asset kind")
    path = request.query.get("path", "")
    recursive_raw = request.query.get("recursive", "true")
    if recursive_raw not in ("true", "false"):
        return _json_error(400, "recursive must be true or false")
    recursive = recursive_raw == "true"
    try:
        # This is also the /list route's validation and normalization seam.
        # Keep it before all paging state, even for recursive traversal.
        listing = service.table.list_folder(mount_id, path)
    except (AssetError, MountsError) as exc:
        return _json_error(400, str(exc))
    limit_raw = request.query.get("limit", str(_PAGE_LIMIT_DEFAULT))
    try:
        limit = int(limit_raw)
    except ValueError:
        return _json_error(400, "limit must be an integer")
    if not 1 <= limit <= _PAGE_LIMIT_MAX:
        return _json_error(400, f"limit must be in 1..{_PAGE_LIMIT_MAX}")
    bound = {
        "mount": mount_id,
        "q": query.lower(),
        "kind": kind,
        "path": path,
        "recursive": str(recursive).lower(),
    }
    after_path = ""
    cursor = request.query.get("cursor")
    if cursor:
        _, after_path = decode_cursor(cursor, bound)
    if recursive:
        entries = service.table.entries(mount_id)
        scope = f"mounts/{mount_id}"
        if path:
            scope = f"{scope}/{path}"
        scope += "/"
        entries = tuple(entry for entry in entries if entry.virtual_path.startswith(scope))
    else:
        entries = listing.entries
    needle = query.lower()
    mount_kind = service.table.kind(mount_id)
    rows = [
        entry
        for entry in entries
        if (not needle or needle in entry.virtual_path.lower())
        and (not kind or _entry_kind(entry, mount_kind) == kind)
        and entry.virtual_path > after_path
    ]
    page, remainder = rows[:limit], rows[limit:]
    body: dict[str, object] = {"entries": [_entry_wire(entry, mount_kind) for entry in page]}
    if not recursive and "cursor" not in request.query:
        body["folders"] = list(listing.folders)
    if remainder:
        body["cursor"] = encode_cursor(bound, (0.0, page[-1].virtual_path))
    return web.json_response(body)


async def handle_mount_add(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    if not service.allow_changes:
        return _json_error(403, "mount-changes-disabled")
    try:
        raw = await request.json()
    except json.JSONDecodeError as exc:
        return _json_error(400, f"invalid JSON: {exc}")
    if not isinstance(raw, dict):
        return _json_error(400, "request body must be an object")
    body: dict[str, object] = {str(k): v for k, v in raw.items()}
    mount_id, path, mode, priority, kind = (
        body.get("id"),
        body.get("path"),
        body.get("mode", "read"),
        body.get("priority", 0),
        body.get("kind", ""),
    )
    if not isinstance(mount_id, str) or not isinstance(path, str) or not path:
        return _json_error(400, "id and path must be non-empty strings")
    if not isinstance(mode, str):
        return _json_error(400, "mode must be a string")
    if isinstance(priority, bool) or not isinstance(priority, int):
        return _json_error(400, "priority must be an integer")
    if not isinstance(kind, str):
        return _json_error(400, "kind must be a string")
    project = body.get("project", "default")
    shared = body.get("shared", False)
    if not isinstance(project, str) or not isinstance(shared, bool):
        return _json_error(400, "project must be a project id and shared must be a boolean")
    try:
        service.projects.validate_project(project)
        mount = MountDef(
            id=mount_id,
            path=Path(path),
            mode=mode,
            priority=priority,
            kind=kind,
        )
    except MountsError as exc:
        return _json_error(400, str(exc))
    # Add-time directory validation is a POST-only courtesy: the grant
    # gesture deserves immediate feedback, while a config mount whose
    # drive is unplugged degrades to a failed row instead.
    if not mount.path.is_dir():
        return _json_error(400, f"not a directory: {mount.path}")
    candidate = service.copy_settings()
    try:
        if service.restart_for_changes and service.table.get(mount.id) is not None:
            return _json_error(409, "mount id remains active until restart")
        candidate.add(mount, source="config")
    except MountsError as exc:
        return _json_error(409, str(exc))
    try:
        service.persist(candidate)
        service.projects.register(mount.id, project=project, shared=shared)
    except OSError:
        service.persist()
        raise
    if service.restart_for_changes:
        service.settings_table = candidate
    else:
        service.table.add(mount, source="config")
    service.publish(request.app)
    # Fixed sandbox grants cannot be rebound; only live additions are scanned.
    if not service.restart_for_changes:
        service.scan_soon(request.app, mount.id)
    descriptor = next(row for row in service.settings_descriptors() if row["id"] == mount.id)
    descriptor.update(service.pending_changes())
    return web.json_response(descriptor, status=201)


async def handle_mount_remove(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    if not service.allow_changes:
        return _json_error(403, "mount-changes-disabled")
    mount_id = request.match_info["mountId"]
    source = service.settings_table.source(mount_id)
    if source is None:
        return _json_error(404, f"unknown mount: {mount_id}")
    if source != "config":
        return _json_error(
            409,
            f"mount {mount_id!r} is derived from server configuration "
            f"(source {source!r}); re-point the configuration instead",
        )
    affected = service.projects.output_projects(mount_id)
    if affected:
        return _json_error(
            409, f"select replacement output folders for projects: {', '.join(affected)}"
        )
    candidate = service.copy_settings()
    try:
        candidate.remove(mount_id)
    except MountsError as exc:
        return _json_error(409, str(exc))
    service.persist(candidate)
    if service.restart_for_changes:
        service.settings_table = candidate
    else:
        service.table.remove(mount_id)
    service.refresh_resolution_store()
    service.publish(request.app)
    return web.json_response({"removed": mount_id, **service.pending_changes()})


async def handle_output_mount_update(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    if not service.allow_changes:
        return _json_error(403, "mount-changes-disabled")
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return _json_error(400, f"invalid JSON: {exc}")
    mount_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(mount_id, str):
        return _json_error(400, "id must be a mount id")
    candidate = service.copy_settings()
    try:
        candidate.select_output_mount(mount_id)
    except MountsError as exc:
        return _json_error(400, str(exc))
    service.persist(candidate)
    if service.restart_for_changes:
        service.settings_table = candidate
    else:
        service.table.select_output_mount(mount_id)
    service.publish(request.app)
    return web.json_response({"outputMount": mount_id, **service.pending_changes()})


def _is_operator(request: web.Request) -> bool:
    return principal_for(request).is_operator


def _require_operator(request: web.Request) -> None:
    if not _is_operator(request):
        raise web.HTTPForbidden(
            text='{"error":"operator-required"}', content_type="application/json"
        )


async def handle_project_update(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    if not service.allow_changes:
        return _json_error(403, "mount-changes-disabled")
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return _json_error(400, f"invalid JSON: {exc}")
    if not isinstance(body, dict) or set(body) != {"mounts", "outputMount"}:
        return _json_error(400, "expected mounts and outputMount")
    project = request.match_info["project"]
    output = body["outputMount"]
    if output is not None:
        mount = service.settings_table.get(output) if isinstance(output, str) else None
        if mount is None or mount.mode != "readwrite":
            return _json_error(400, "outputMount must name a writable grant or be null")
    try:
        if output is None and service.project_response(project)["outputMount"] is not None:
            return _json_error(409, "select a replacement output folder for the project")
        service.projects.update(
            project, body["mounts"], available=service.mount_ids(settings=True), output=output
        )
    except MountsError as exc:
        return _json_error(400, str(exc))
    service.publish(request.app)
    return web.json_response(service.project_response(project))


async def handle_mount_retry(request: web.Request) -> web.Response:
    _require_operator(request)
    service = request.app[MOUNTS_KEY]
    if not service.allow_changes:
        return _json_error(403, "mount-changes-disabled")
    mount_id = request.match_info["mountId"]
    if service.table.get(mount_id) is None:
        return _json_error(404, f"unknown mount: {mount_id}")
    service.scan_soon(request.app, mount_id)
    return web.json_response({"queued": mount_id}, status=202)


def add_mount_routes(app: web.Application, service: MountService) -> None:
    app[MOUNTS_KEY] = service
    app.router.add_get("/api/mounts", handle_mounts_list)
    app.router.add_get("/api/mounts/locations", handle_mount_locations)
    app.router.add_post("/api/mounts/resolve-path", handle_mount_resolve_path)
    app.router.add_get("/api/mounts/{mountId}/list", handle_mount_folder)
    app.router.add_get("/api/mounts/{mountId}/entries", handle_mount_entries)
    app.router.add_put("/api/mounts/output", handle_output_mount_update)
    app.router.add_put("/api/mounts/projects/{project}", handle_project_update)
    app.router.add_post("/api/mounts/{mountId}/retry", handle_mount_retry)
    app.router.add_post("/api/mounts", handle_mount_add)
    app.router.add_delete("/api/mounts/{mountId}", handle_mount_remove)
