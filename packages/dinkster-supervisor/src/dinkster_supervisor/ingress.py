"""Stateless fleet ingress with SQLite-backed routing ownership."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Set
from dataclasses import dataclass
from typing import cast

import aiohttp
from aiohttp import WSCloseCode, WSMsgType, web

from .leases import LeaseStore, LeaseStoreError
from .limits import (
    INGRESS_CLIENT_MAX_SIZE_BYTES,
    INGRESS_EVENT_FRAME_LIMIT_BYTES,
    INGRESS_EVENT_QUEUE_LIMIT_BYTES,
    INGRESS_JOB_SUBMISSION_LIMIT_BYTES,
)
from .supervisor import EngineLink, install_cors


@dataclass(frozen=True)
class IngressMember:
    owner_id: str
    link: EngineLink


MEMBERS_KEY: web.AppKey[Mapping[str, IngressMember]] = web.AppKey("ingress_members")
PRIMARY_KEY: web.AppKey[str] = web.AppKey("ingress_primary")
STORE_KEY: web.AppKey[LeaseStore] = web.AppKey("ingress_store")
SESSION_KEY: web.AppKey[aiohttp.ClientSession] = web.AppKey("ingress_session")

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_WS_QUEUE_FRAMES = 64
_WS_HEARTBEAT = 30.0
_FANOUT_TIMEOUT = 5.0
_RECONNECT_INITIAL = 0.1
_RECONNECT_MAX = 2.0
# A handshake alone is not stability: short-lived connections retain backoff.
_RECONNECT_STABLE_AFTER = 5.0


class _FrameQueue:
    def __init__(self, max_frames: int, max_bytes: int) -> None:
        self._items: deque[tuple[bool, str | bytes, int]] = deque()
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._changed = asyncio.Condition()

    async def put(self, binary: bool, data: str | bytes) -> None:
        size = len(data) if binary else len(cast("str", data).encode())
        async with self._changed:
            await self._changed.wait_for(
                lambda: (
                    len(self._items) < self._max_frames
                    and self._used_bytes + size <= self._max_bytes
                )
            )
            self._items.append((binary, data, size))
            self._used_bytes += size
            self._changed.notify_all()

    async def get(self) -> tuple[bool, str | bytes]:
        async with self._changed:
            await self._changed.wait_for(self._items.__len__)
            binary, data, size = self._items.popleft()
            self._used_bytes -= size
            self._changed.notify_all()
            return binary, data


def _not_ready(member: IngressMember) -> web.Response:
    return web.json_response(
        {"error": "engine-not-ready", "engine": member.owner_id, "state": member.link.state},
        status=503,
    )


def _owner_unavailable(owner: str) -> web.Response:
    return web.json_response(
        {"error": "engine-not-ready", "engine": owner, "state": "not-configured"},
        status=503,
    )


def _member_or_response(request: web.Request, owner: str) -> IngressMember | web.Response:
    member = request.app[MEMBERS_KEY].get(owner)
    if member is None:
        return _owner_unavailable(owner)
    if member.link.state != "ready":
        return _not_ready(member)
    return member


def _request_headers(request: web.Request) -> dict[str, str]:
    hop_by_hop = _hop_by_hop_names(request.headers.items())
    return {
        name: value
        for name, value in request.headers.items()
        if not _is_hop_by_hop(name, hop_by_hop)
    }


def _hop_by_hop_names(headers: Iterable[tuple[str, str]]) -> frozenset[str]:
    names = set(_HOP_BY_HOP)
    for name, value in headers:
        if name.lower() == "connection":
            names.update(token.strip().lower() for token in value.split(",") if token.strip())
    return frozenset(names)


def _is_hop_by_hop(name: str, names: Set[str]) -> bool:
    lowered = name.lower()
    return lowered in names or lowered.startswith("proxy-")


def _reconnect_delay(delay: float) -> float:
    return min(_RECONNECT_MAX, delay * random.uniform(0.8, 1.2))


async def _read_bounded(content: aiohttp.StreamReader, limit: int) -> bytes | None:
    body = bytearray()
    async for chunk in content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


async def _forward(request: web.Request, owner: str) -> web.StreamResponse:
    found = _member_or_response(request, owner)
    if isinstance(found, web.Response):
        return found
    member = found
    response: web.StreamResponse | None = None
    try:
        async with request.app[SESSION_KEY].request(
            request.method,
            member.link.base_url + request.rel_url.path_qs,
            headers=_request_headers(request),
            data=request.content if request.body_exists else None,
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(status=upstream.status)
            hop_by_hop = _hop_by_hop_names(upstream.headers.items())
            for name, value in upstream.headers.items():
                if name.lower() != "content-length" and not _is_hop_by_hop(name, hop_by_hop):
                    response.headers.add(name, value)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except (TimeoutError, aiohttp.ClientError, ConnectionResetError):
        if response is not None and response.prepared:
            response.force_close()
            if request.transport is not None:
                request.transport.close()
            return response
        return _not_ready(member)


async def _forward_submit(request: web.Request, owner: str, body: bytes) -> web.Response:
    found = _member_or_response(request, owner)
    if isinstance(found, web.Response):
        return found
    member = found
    try:
        async with request.app[SESSION_KEY].request(
            request.method,
            member.link.base_url + request.rel_url.path_qs,
            headers=_request_headers(request),
            data=body,
            allow_redirects=False,
        ) as upstream:
            response_body = await _read_bounded(
                upstream.content, INGRESS_JOB_SUBMISSION_LIMIT_BYTES
            )
            if response_body is None:
                return web.json_response({"error": "accepted-body-too-large"}, status=502)
            response = web.Response(body=response_body, status=upstream.status)
            hop_by_hop = _hop_by_hop_names(upstream.headers.items())
            for name, value in upstream.headers.items():
                if name.lower() != "content-length" and not _is_hop_by_hop(name, hop_by_hop):
                    response.headers.add(name, value)
            return response
    except (TimeoutError, aiohttp.ClientError):
        return _not_ready(member)


async def _submit(request: web.Request) -> web.Response:
    try:
        raw = await _read_bounded(request.content, INGRESS_JOB_SUBMISSION_LIMIT_BYTES)
        if raw is None:
            return web.json_response({"error": "job-body-too-large"}, status=413)
        value: object = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        body = cast("dict[str, object]", value)
        client_id, job_id = body["clientId"], body["jobId"]
        if not isinstance(client_id, str) or not isinstance(job_id, str):
            raise ValueError
    except (ValueError, KeyError, json.JSONDecodeError):
        return web.json_response({"error": "invalid-job-key"}, status=400)
    owner = await request.app[STORE_KEY].claim_key(
        "default", client_id, job_id, request.app[PRIMARY_KEY]
    )
    response = await _forward_submit(request, owner, raw)
    if response.status == 202:
        try:
            value = json.loads(cast("bytes", response.body))
            if not isinstance(value, dict):
                raise ValueError
            wire = cast("dict[str, object]", value)
            if wire.get("clientId") != client_id or wire.get("jobId") != job_id:
                raise ValueError
            job_ref = wire["jobRef"]
            if not isinstance(job_ref, str) or not job_ref:
                raise ValueError
            await request.app[STORE_KEY].record_job(job_ref, owner)
        except LeaseStoreError:
            return web.json_response({"error": "ownership-store-unavailable"}, status=503)
        except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            return web.json_response({"error": "ownership-persistence-failed"}, status=502)
    return response


async def _by_key(request: web.Request) -> web.StreamResponse:
    owner = await request.app[STORE_KEY].lookup_key(
        "default", request.match_info["client_id"], request.match_info["job_id"]
    )
    if owner is None:
        return web.json_response({"error": "no-such-job"}, status=404)
    return await _forward(request, owner)


async def _value_by_key(request: web.Request) -> web.StreamResponse:
    client_id = request.query.get("clientId")
    job_id = request.query.get("jobId")
    if client_id is None or job_id is None:
        return web.json_response({"error": "invalid-job-key"}, status=400)
    owner = await request.app[STORE_KEY].lookup_key("default", client_id, job_id)
    if owner is None:
        return web.json_response({"error": "no-such-job"}, status=404)
    return await _forward(request, owner)


async def _by_ref(request: web.Request, *, legacy_fallback: bool = False) -> web.StreamResponse:
    ref = request.match_info["job_ref"]
    owner = await request.app[STORE_KEY].lookup_job(ref)
    if owner is None and legacy_fallback:
        # Preserve the engine's literal-route fallback for clientId="by-ref".
        owner = await request.app[STORE_KEY].lookup_key("default", "by-ref", ref)
    if owner is None:
        return web.json_response({"error": "no-such-job"}, status=404)
    return await _forward(request, owner)


async def _by_ref_status(request: web.Request) -> web.StreamResponse:
    return await _by_ref(request, legacy_fallback=True)


async def _by_ref_assignment(request: web.Request) -> web.StreamResponse:
    return await _by_ref(request)


@web.middleware
async def _ownership_errors(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    try:
        return await handler(request)
    except LeaseStoreError:
        return web.json_response({"error": "ownership-store-unavailable"}, status=503)


async def _primary(request: web.Request) -> web.StreamResponse:
    return await _forward(request, request.app[PRIMARY_KEY])


async def _aggregate(request: web.Request) -> web.Response:
    async def fetch(
        owner: str, member: IngressMember
    ) -> tuple[dict[str, object] | None, dict[str, str] | None]:
        if member.link.state != "ready":
            return None, {"engine": owner, "state": member.link.state}
        try:
            async with asyncio.timeout(_FANOUT_TIMEOUT):
                async with request.app[SESSION_KEY].get(
                    member.link.base_url + request.rel_url.path_qs
                ) as response:
                    if response.status != 200:
                        return None, {"engine": owner, "state": f"http-{response.status}"}
                    value: object = await response.json()
                    if not isinstance(value, dict):
                        raise ValueError
                    body = cast("dict[str, object]", value)
                    if not _valid_aggregate(request.path, body):
                        raise ValueError
                    return body, None
        except (TimeoutError, aiohttp.ClientError, ValueError):
            return None, {"engine": owner, "state": "unreachable"}

    members = request.app[MEMBERS_KEY].items()
    results = await asyncio.gather(*(fetch(owner, member) for owner, member in members))
    bodies = [body for body, _ in results if body is not None]
    failures = [failure for _, failure in results if failure is not None]
    if failures:
        return web.json_response({"error": "fleet-incomplete", "members": failures}, status=503)
    if request.path == "/api/jobs":
        jobs = [
            job for body in bodies for job in cast("list[dict[str, object]]", body.get("jobs", []))
        ]
        jobs.sort(
            key=lambda job: (
                cast("float", job["submittedAt"]),
                cast("str", job["jobRef"]),
            )
        )
        return web.json_response({"jobs": jobs})
    queued = [job for body in bodies for job in cast("list[object]", body["queued"])]
    running = [job for body in bodies for job in cast("list[object]", body.get("running", []))]
    return web.json_response(
        {
            "queued": queued,
            "running": running,
            "maxRunningJobs": sum(cast("int", body["maxRunningJobs"]) for body in bodies),
            "paused": all(cast("bool", body["paused"]) for body in bodies),
        }
    )


def _history_bound(request: web.Request) -> dict[str, str]:
    return {
        "s": request.query.get("scope", ""),
        "c": request.query.get("clientId", ""),
        "d": request.query.get("sourceDocument", ""),
        "t": request.query.get("state", ""),
    }


def _encode_history_cursor(bound: Mapping[str, str], after: tuple[float, str]) -> str:
    payload = json.dumps({"b": dict(bound), "m": after[0], "i": after[1]})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_history_cursor(cursor: str, bound: Mapping[str, str]) -> tuple[float, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value: object = json.loads(base64.urlsafe_b64decode(padded))
        if not isinstance(value, dict):
            raise TypeError
        data = cast("dict[str, object]", value)
        cursor_bound = data["b"]
        if not isinstance(cursor_bound, dict):
            raise TypeError
        entries = cast("dict[object, object]", cursor_bound).items()
        if {str(key): str(item) for key, item in entries} != dict(bound):
            raise ValueError("query mismatch")
        return float(cast("float | str", data["m"])), str(data["i"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error) as exc:
        raise ValueError("malformed cursor") from exc


async def _history_member_page(
    request: web.Request,
    member: IngressMember,
    after: tuple[float, str] | None,
    wanted: int,
) -> list[dict[str, object]]:
    params = dict(request.query)
    params.pop("cursor", None)
    params["limit"] = str(min(200, wanted))
    records: list[dict[str, object]] = []
    upstream_cursor: str | None = None
    while len(records) < wanted:
        if upstream_cursor is None:
            params.pop("cursor", None)
        else:
            params["cursor"] = upstream_cursor
        async with request.app[SESSION_KEY].get(
            member.link.base_url + "/api/history", params=params
        ) as response:
            if response.status != 200:
                raise ValueError(f"http-{response.status}")
            value: object = await response.json()
        if not isinstance(value, dict):
            raise ValueError("malformed")
        body = cast("dict[str, object]", value)
        page = body.get("records")
        if not isinstance(page, list) or not all(
            _valid_job_row(item, "finishedAt") for item in cast("list[object]", page)
        ):
            raise ValueError("malformed")
        for item in cast("list[dict[str, object]]", page):
            key = (cast("float", item["finishedAt"]), cast("str", item["jobRef"]))
            if after is None or key < after:
                records.append(item)
                if len(records) == wanted:
                    break
        next_cursor = body.get("cursor")
        if next_cursor is None:
            break
        if not isinstance(next_cursor, str) or next_cursor == upstream_cursor:
            raise ValueError("malformed")
        upstream_cursor = next_cursor
    return records


async def _history(request: web.Request) -> web.Response:
    scope = request.query.get("scope")
    if scope is None or not scope.strip():
        return web.json_response({"error": "scope is required (single-user: 'local')"}, status=400)
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        return web.json_response({"error": "limit must be an integer"}, status=400)
    bound = _history_bound(request)
    bound["s"] = scope.strip()
    try:
        after = (
            _decode_history_cursor(request.query["cursor"], bound)
            if "cursor" in request.query
            else None
        )
    except ValueError:
        return web.json_response({"error": "malformed cursor"}, status=400)

    async def fetch(
        owner: str, member: IngressMember
    ) -> tuple[list[dict[str, object]], dict[str, str] | None]:
        if member.link.state != "ready":
            return [], {"engine": owner, "state": member.link.state}
        try:
            async with asyncio.timeout(_FANOUT_TIMEOUT):
                return await _history_member_page(request, member, after, limit + 1), None
        except (TimeoutError, aiohttp.ClientError, ValueError):
            return [], {"engine": owner, "state": "unreachable"}

    results = await asyncio.gather(
        *(fetch(owner, member) for owner, member in request.app[MEMBERS_KEY].items())
    )
    records = [record for member_records, _ in results for record in member_records]
    failures = [failure for _, failure in results if failure is not None]
    if failures:
        return web.json_response({"error": "fleet-incomplete", "members": failures}, status=503)
    records.sort(
        key=lambda record: (
            cast("float", record["finishedAt"]),
            cast("str", record["jobRef"]),
        ),
        reverse=True,
    )
    wire: dict[str, object] = {"records": records[:limit]}
    if len(records) > limit:
        last = records[limit - 1]
        wire["cursor"] = _encode_history_cursor(
            bound, (cast("float", last["finishedAt"]), cast("str", last["jobRef"]))
        )
    return web.json_response(wire)


def _valid_job_row(item: object, timestamp: str) -> bool:
    if not isinstance(item, dict):
        return False
    row = cast("dict[str, object]", item)
    submitted = row.get(timestamp)
    return (
        isinstance(submitted, (int, float))
        and not isinstance(submitted, bool)
        and isinstance(row.get("jobRef"), str)
    )


def _valid_aggregate(path: str, value: dict[str, object]) -> bool:
    if path == "/api/jobs":
        return isinstance(value.get("jobs"), list) and all(
            _valid_job_row(item, "submittedAt") for item in cast("list[object]", value["jobs"])
        )
    return (
        isinstance(value.get("queued"), list)
        and isinstance(value.get("running"), list)
        and isinstance(value.get("maxRunningJobs"), int)
        and not isinstance(value.get("maxRunningJobs"), bool)
        and isinstance(value.get("paused"), bool)
    )


async def _health(request: web.Request) -> web.Response:
    members = {
        name: {"state": member.link.state} for name, member in request.app[MEMBERS_KEY].items()
    }
    ready = all(member.link.state == "ready" for member in request.app[MEMBERS_KEY].values())
    return web.json_response(
        {"ok": ready, "state": "ready" if ready else "degraded", "members": members},
        status=200 if ready else 503,
    )


def _correlated(data: str | bytes, binary: bool) -> bool:
    try:
        raw = data
        if binary:
            payload = cast("bytes", raw)
            if len(payload) < 4:
                return False
            length = int.from_bytes(payload[:4], "big")
            raw = payload[4 : 4 + length].decode()
        value: object = json.loads(cast("str", raw))
        if not isinstance(value, dict):
            return False
        fields = cast("dict[object, object]", value)
        return isinstance(fields.get("jobRef"), str)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


async def _events(request: web.Request) -> web.WebSocketResponse:
    downstream = web.WebSocketResponse(heartbeat=_WS_HEARTBEAT)
    await downstream.prepare(request)
    queue = _FrameQueue(_WS_QUEUE_FRAMES, INGRESS_EVENT_QUEUE_LIMIT_BYTES)

    async def reader(name: str, member: IngressMember) -> None:
        delay = 0.1
        while not downstream.closed:
            if member.link.state != "ready":
                await asyncio.sleep(_reconnect_delay(delay))
                delay = min(delay * 2, _RECONNECT_MAX)
                continue
            try:
                url = member.link.base_url.replace("http://", "ws://").replace("https://", "wss://")
                async with request.app[SESSION_KEY].ws_connect(
                    url + request.rel_url.path_qs,
                    max_msg_size=INGRESS_EVENT_FRAME_LIMIT_BYTES,
                ) as upstream:
                    connected_at = time.monotonic()
                    async for message in upstream:
                        if message.type == WSMsgType.ERROR:
                            error = upstream.exception()
                            if (
                                isinstance(error, aiohttp.WebSocketError)
                                and error.code == WSCloseCode.MESSAGE_TOO_BIG
                            ):
                                await downstream.close(
                                    code=WSCloseCode.MESSAGE_TOO_BIG,
                                    message=b"upstream websocket frame exceeds limit",
                                )
                                return
                            break
                        if message.type not in {WSMsgType.TEXT, WSMsgType.BINARY}:
                            continue
                        binary = message.type == WSMsgType.BINARY
                        if name == request.app[PRIMARY_KEY] or _correlated(message.data, binary):
                            await queue.put(binary, message.data)
                    if upstream.close_code == WSCloseCode.MESSAGE_TOO_BIG:
                        await downstream.close(
                            code=WSCloseCode.MESSAGE_TOO_BIG,
                            message=b"upstream websocket frame exceeds limit",
                        )
                        return
                if time.monotonic() - connected_at >= _RECONNECT_STABLE_AFTER:
                    delay = _RECONNECT_INITIAL
            except (TimeoutError, aiohttp.ClientError):
                pass
            await asyncio.sleep(_reconnect_delay(delay))
            delay = min(delay * 2, _RECONNECT_MAX)

    async def writer() -> None:
        while not downstream.closed:
            binary, data = await queue.get()
            if binary:
                await downstream.send_bytes(cast("bytes", data))
            else:
                await downstream.send_str(cast("str", data))

    tasks = [
        asyncio.create_task(reader(name, member))
        for name, member in request.app[MEMBERS_KEY].items()
    ]
    tasks.append(asyncio.create_task(writer()))
    try:
        async for _ in downstream:
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return downstream


async def _status(request: web.Request) -> web.Response:
    return web.json_response({"protocol": 1, "state": "ready", "role": "ingress"})


async def _unknown_supervisor(request: web.Request) -> web.Response:
    return web.json_response({"error": "unknown-supervisor-path"}, status=404)


async def _session(app: web.Application):  # type: ignore[no-untyped-def]
    app[SESSION_KEY] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=_FANOUT_TIMEOUT),
        auto_decompress=False,
    )
    yield
    await app[SESSION_KEY].close()


def create_ingress_app(
    members: Mapping[str, IngressMember],
    primary: str,
    store: LeaseStore,
    *,
    allow_origins: tuple[str, ...] = (),
) -> web.Application:
    if not members or primary not in members:
        raise ValueError("ingress members must be non-empty and contain the primary")
    if any(name != member.owner_id for name, member in members.items()):
        raise ValueError("ingress member keys must match their owner ids")
    app = web.Application(
        middlewares=[_ownership_errors], client_max_size=INGRESS_CLIENT_MAX_SIZE_BYTES
    )
    app[MEMBERS_KEY], app[PRIMARY_KEY], app[STORE_KEY] = dict(members), primary, store
    app.cleanup_ctx.append(_session)
    install_cors(app, frozenset(allow_origins))
    app.router.add_get("/supervisor/status", _status)
    app.router.add_route(
        "*",
        "/supervisor/{tail:.*}",
        _unknown_supervisor,
    )
    app.router.add_post("/api/jobs", _submit)
    app.router.add_get("/api/events", _events)
    app.router.add_get("/api/jobs/by-ref/{job_ref}", _by_ref_status)
    app.router.add_delete("/api/jobs/by-ref/{job_ref}", _by_ref_status)
    app.router.add_get("/api/jobs/by-ref/{job_ref}/events", _by_ref_assignment)
    app.router.add_get("/api/jobs/{client_id}/{job_id}", _by_key)
    app.router.add_delete("/api/jobs/{client_id}/{job_id}", _by_key)
    app.router.add_get("/api/values", _value_by_key)
    app.router.add_get("/api/history/{job_ref}", _by_ref_assignment)
    app.router.add_get("/api/jobs", _aggregate)
    app.router.add_get("/api/queue", _aggregate)
    app.router.add_get("/api/history", _history)
    app.router.add_get("/api/health", _health)
    # Explicit primary-only matrix for catalog, state, and mutation surfaces.
    for prefix in (
        "nodes",
        "choices",
        "composition",
        "templates",
        "packs",
        "diagnostics",
        "assets",
        "library",
        "mounts",
        "settings",
        "memory",
        "cache",
        "queue",
        "compat/comfy/prompt",
    ):
        app.router.add_route("*", f"/api/{prefix}", _primary)
        app.router.add_route("*", f"/api/{prefix}/{{tail:.*}}", _primary)
    for prefix in ("assets", "memory", "cache"):
        app.router.add_route("*", f"/{prefix}", _primary)
        app.router.add_route("*", f"/{prefix}/{{tail:.*}}", _primary)
    return app
