from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import aiohttp
import pytest
from aiohttp import WSCloseCode, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_supervisor import (
    EngineLink,
    IngressDef,
    IngressMember,
    LeaseConflictError,
    LeaseStore,
    LeaseStoreError,
    StationConfig,
    create_ingress_app,
    dump_station_config,
    parse_station_config,
)
from dinkster_supervisor import ingress as ingress_module
from dinkster_supervisor.install_manager import _write_config
from dinkster_supervisor.installs import InstallsError
from multidict import CIMultiDict


async def _fleet(
    tmp_path: Path, apps: dict[str, web.Application], *, primary: str | None = None
) -> tuple[TestClient, LeaseStore, list[TestServer]]:
    servers: list[TestServer] = []
    members: dict[str, IngressMember] = {}
    for name, app in apps.items():
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        members[name] = IngressMember(
            name, EngineLink(state="ready", base_url=str(server.make_url("")))
        )
    store = LeaseStore(tmp_path / "fleet.sqlite3")
    await store.initialize()
    client = TestClient(TestServer(create_ingress_app(members, primary or next(iter(apps)), store)))
    await client.start_server()
    return client, store, servers


async def _close(client: TestClient, servers: list[TestServer]) -> None:
    await client.close()
    await asyncio.gather(*(server.close() for server in servers))


def _engine(name: str, submissions: list[dict[str, object]]) -> web.Application:
    app = web.Application()

    async def submit(request: web.Request) -> web.Response:
        body = await request.json()
        if submissions:
            previous = submissions[0]
            if previous != body:
                return web.json_response({"error": "job-key-in-use"}, status=409)
            return web.json_response(
                {**body, "jobRef": f"{name}-ref", "duplicate": True}, status=202
            )
        submissions.append(body)
        return web.json_response({**body, "jobRef": f"{name}-ref"}, status=202)

    async def echo(request: web.Request) -> web.Response:
        return web.json_response({"engine": name, "path": request.path})

    app.router.add_post("/api/jobs", submit)
    app.router.add_route("*", "/{tail:.*}", echo)
    return app


def test_two_lease_stores_atomically_claim_one_key(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    first, second = LeaseStore(path), LeaseStore(path)

    async def claim() -> tuple[str, str]:
        await first.initialize()
        await second.initialize()
        return await asyncio.gather(
            first.claim_key("default", "client", "job", "alpha"),
            second.claim_key("default", "client", "job", "beta"),
        )

    winners = asyncio.run(claim())
    assert winners[0] == winners[1]
    assert winners[0] in {"alpha", "beta"}


def test_job_owner_conflict_is_never_overwritten(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "leases.sqlite3")

    async def check() -> None:
        await store.record_job("ref", "alpha")
        with pytest.raises(LeaseConflictError):
            await store.record_job("ref", "beta")
        assert await store.lookup_job("ref") == "alpha"
        assert not await store.delete_job_if_owner("ref", "beta")
        assert await store.delete_job_if_owner("ref", "alpha")
        assert await store.lookup_job("ref") is None

    asyncio.run(check())


def test_ingress_config_roundtrip() -> None:
    config = parse_station_config(
        '[installs.alpha]\nroot="/alpha"\nport=8200\n'
        '[installs.beta]\nroot="/beta"\nport=8201\n'
        '[ingress]\nport=8300\nmembers=["alpha", "beta"]\n'
    )
    assert config.ingress == IngressDef(8300, ("alpha", "beta"), "alpha")
    assert parse_station_config(dump_station_config(config)) == config
    assert isinstance(config, StationConfig)


def test_install_writer_preserves_ingress_and_rejects_member_removal(tmp_path: Path) -> None:
    path = tmp_path / "installs.toml"
    path.write_text(
        '[installs.alpha]\nroot="/alpha"\nport=8200\n[ingress]\nport=8300\nmembers=["alpha"]\n'
    )
    config = parse_station_config(path.read_text())
    beta = type(config.installs[0])("beta", Path("/beta"), 8201)
    _write_config(path, (*config.installs, beta))
    written = parse_station_config(path.read_text())
    assert written.ingress is not None
    assert written.ingress.members == ("alpha",)
    assert {install.name for install in written.installs} == {"alpha", "beta"}
    with pytest.raises(InstallsError, match="unknown members"):
        _write_config(path, (beta,))
    assert parse_station_config(path.read_text()) == written


@pytest.mark.parametrize(
    "ingress",
    [
        'port=3649\nmembers=["alpha"]',
        "port=8300\nmembers=[]",
        'port=8300\nmembers=[""]',
        'port=8300\nmembers=["alpha", "alpha"]',
        'port=8300\nmembers=["missing"]',
        'port=8300\nmembers=["alpha"]\nprimary="missing"',
    ],
)
def test_ingress_config_strict_validation(ingress: str) -> None:
    with pytest.raises(InstallsError):
        parse_station_config('[installs.alpha]\nroot="/alpha"\nport=8200\n[ingress]\n' + ingress)


def test_concurrent_same_key_different_content_chooses_one_owner_and_engine_409(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        submissions: list[dict[str, object]] = []
        client, store, servers = await _fleet(tmp_path, {"alpha": _engine("alpha", submissions)})
        try:
            one = {"clientId": "c", "jobId": "j", "graph": {"one": 1}}
            two = {"clientId": "c", "jobId": "j", "graph": {"two": 2}}
            responses = await asyncio.gather(
                client.post("/api/jobs", json=one), client.post("/api/jobs", json=two)
            )
            assert sorted(response.status for response in responses) == [202, 409]
            assert await store.lookup_key("default", "c", "j") == "alpha"
            assert len(submissions) == 1
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_lost_202_disconnect_retry_stays_owner_and_recovers_duplicate_true(tmp_path: Path) -> None:
    async def scenario() -> None:
        submissions: list[dict[str, object]] = []
        app = web.Application()

        async def submit(request: web.Request) -> web.Response:
            body = await request.json()
            if not submissions:
                submissions.append(body)
                assert request.transport is not None
                request.transport.close()
                return web.json_response({}, status=202)
            return web.json_response({**body, "jobRef": "alpha-ref", "duplicate": True}, status=202)

        app.router.add_post("/api/jobs", submit)
        client, store, servers = await _fleet(tmp_path, {"alpha": app})
        body = {"clientId": "c", "jobId": "j", "graph": {}}
        try:
            lost = await client.post("/api/jobs", json=body)
            assert lost.status == 503
            assert await store.lookup_key("default", "c", "j") == "alpha"
            assert await store.lookup_job("alpha-ref") is None
            response = await client.post("/api/jobs", json=body)
            assert response.status == 202
            assert (await response.json())["duplicate"] is True
            assert await store.lookup_job("alpha-ref") == "alpha"
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_restart_same_store_preserves_keyed_and_by_ref_routes(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = TestServer(_engine("alpha", []))
        await server.start_server()
        member = IngressMember(
            "alpha", EngineLink(state="ready", base_url=str(server.make_url("")))
        )
        path = tmp_path / "fleet.sqlite3"
        store = LeaseStore(path)
        await store.claim_key("default", "c", "j", "alpha")
        await store.record_job("ref", "alpha")
        first = TestClient(TestServer(create_ingress_app({"alpha": member}, "alpha", store)))
        await first.start_server()
        try:
            for route in ("/api/jobs/c/j", "/api/jobs/by-ref/ref"):
                assert (await first.get(route)).status == 200
            await first.close()
            restarted_store = LeaseStore(path)
            second = TestClient(
                TestServer(create_ingress_app({"alpha": member}, "alpha", restarted_store))
            )
            await second.start_server()
            try:
                for route in ("/api/jobs/c/j", "/api/jobs/by-ref/ref"):
                    assert (await second.get(route)).status == 200
            finally:
                await second.close()
        finally:
            await first.close()
            await server.close()

    asyncio.run(scenario())


def test_assignment_exists_before_downstream_202(tmp_path: Path) -> None:
    async def scenario() -> None:
        class ObservedStore(LeaseStore):
            def __init__(self, path: Path) -> None:
                super().__init__(path)
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def record_job(self, job_ref: str, owner_id: str) -> None:
                self.entered.set()
                await self.release.wait()
                await super().record_job(job_ref, owner_id)

        server = TestServer(_engine("alpha", []))
        await server.start_server()
        store = ObservedStore(tmp_path / "fleet.sqlite3")
        member = IngressMember(
            "alpha", EngineLink(state="ready", base_url=str(server.make_url("")))
        )
        client = TestClient(TestServer(create_ingress_app({"alpha": member}, "alpha", store)))
        await client.start_server()
        try:
            pending = asyncio.create_task(
                client.post("/api/jobs", json={"clientId": "c", "jobId": "j"})
            )
            await asyncio.wait_for(store.entered.wait(), 10)
            assert not pending.done()
            store.release.set()
            response = await pending
            assert response.status == 202
            assert await store.lookup_job("alpha-ref") == "alpha"
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_legacy_by_ref_fallback_unknown_assignment_and_owner_down_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _engine("alpha", [])
        server = TestServer(app)
        await server.start_server()
        link = EngineLink(state="ready", base_url=str(server.make_url("")))
        store = LeaseStore(tmp_path / "fleet.sqlite3")
        await store.claim_key("default", "by-ref", "legacy", "alpha")
        client = TestClient(
            TestServer(create_ingress_app({"alpha": IngressMember("alpha", link)}, "alpha", store))
        )
        await client.start_server()
        try:
            assert (await client.get("/api/jobs/by-ref/legacy")).status == 200
            assert (await client.get("/api/history/legacy?scope=local")).status == 404
            assert (await client.get("/api/jobs/by-ref/unknown")).status == 404
            link.state = "crashed"
            response = await client.get("/api/jobs/by-ref/legacy")
            assert response.status == 503
            assert await response.json() == {
                "error": "engine-not-ready",
                "engine": "alpha",
                "state": "crashed",
            }
            assert await store.lookup_key("default", "by-ref", "legacy") == "alpha"
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_aggregates_complete_merge_history_and_fail_closed_on_bad_member(tmp_path: Path) -> None:
    def aggregate_engine(marker: str, order: float, malformed: bool = False) -> web.Application:
        app = web.Application()

        async def reply(request: web.Request) -> web.Response:
            if malformed:
                return web.json_response({"wrong": []})
            if request.path == "/api/jobs":
                return web.json_response({"jobs": [{"jobRef": marker, "submittedAt": order}]})
            if request.path == "/api/history":
                return web.json_response({"records": [{"jobRef": marker, "finishedAt": order}]})
            return web.json_response(
                {"queued": [marker], "running": [], "maxRunningJobs": 1, "paused": True}
            )

        app.router.add_get("/{tail:.*}", reply)
        return app

    async def scenario() -> None:
        client, _, servers = await _fleet(
            tmp_path,
            {"alpha": aggregate_engine("a", 1), "beta": aggregate_engine("b", 2)},
        )
        try:
            assert [
                item["jobRef"] for item in (await (await client.get("/api/jobs")).json())["jobs"]
            ] == ["a", "b"]
            queue = await (await client.get("/api/queue")).json()
            assert queue == {
                "queued": ["a", "b"],
                "running": [],
                "maxRunningJobs": 2,
                "paused": True,
            }
            assert {
                item["jobRef"]
                for item in (await (await client.get("/api/history?scope=local")).json())["records"]
            } == {"a", "b"}
        finally:
            await _close(client, servers)

        bad, _, bad_servers = await _fleet(
            tmp_path,
            {
                "alpha": aggregate_engine("a", 1),
                "beta": aggregate_engine("b", 2, True),
            },
        )
        try:
            for route in ("/api/jobs", "/api/queue", "/api/history?scope=local"):
                response = await bad.get(route)
                assert response.status == 503
                assert (await response.json())["error"] == "fleet-incomplete"
        finally:
            await _close(bad, bad_servers)

    asyncio.run(scenario())


def test_history_merge_honors_global_limit_and_cursor(tmp_path: Path) -> None:
    def history_engine(name: str, timestamps: list[float]) -> web.Application:
        app = web.Application()

        async def history(request: web.Request) -> web.Response:
            offset = int(request.query.get("cursor", "0"))
            limit = int(request.query["limit"])
            page = timestamps[offset : offset + limit]
            body: dict[str, object] = {
                "records": [
                    {"jobRef": f"{name}-{timestamp}", "finishedAt": timestamp} for timestamp in page
                ]
            }
            if offset + limit < len(timestamps):
                body["cursor"] = str(offset + limit)
            return web.json_response(body)

        app.router.add_get("/api/history", history)
        return app

    async def scenario() -> None:
        client, _, servers = await _fleet(
            tmp_path,
            {
                "alpha": history_engine("a", [9, 7, 5, 3, 1]),
                "beta": history_engine("b", [10, 8, 6, 4, 2]),
            },
        )
        try:
            first = await (await client.get("/api/history?scope=local&limit=3")).json()
            assert [record["finishedAt"] for record in first["records"]] == [10, 9, 8]
            assert isinstance(first.get("cursor"), str)
            second = await (
                await client.get(
                    "/api/history",
                    params={"scope": "local", "limit": "3", "cursor": first["cursor"]},
                )
            ).json()
            assert [record["finishedAt"] for record in second["records"]] == [7, 6, 5]
            mismatch = await client.get(
                "/api/history",
                params={"scope": "other", "limit": "3", "cursor": first["cursor"]},
            )
            assert mismatch.status == 400
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_exact_primary_and_local_route_matrix(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, store, servers = await _fleet(
            tmp_path,
            {"beta": _engine("beta", []), "alpha": _engine("alpha", [])},
            primary="alpha",
        )
        try:
            for route in (
                "/api/nodes",
                "/api/choices/model",
                "/api/composition",
                "/api/templates",
                "/api/packs",
                "/api/diagnostics",
                "/api/assets",
                "/api/library",
                "/api/mounts",
                "/api/settings",
                "/api/memory",
                "/api/cache",
                "/assets",
                "/memory/status",
                "/cache/trim",
            ):
                response = await client.get(route)
                assert response.status == 200, route
                assert (await response.json())["engine"] == "alpha"
            response = await client.post("/api/queue/pause")
            assert response.status == 200
            assert (await response.json())["engine"] == "alpha"
            assert (await (await client.get("/supervisor/status")).json())["role"] == "ingress"
            assert (await client.get("/supervisor/not-real")).status == 404
            await store.claim_key("default", "client", "job", "alpha")
            response = await client.get("/api/values?clientId=client&jobId=job&nodeId=n&outputId=o")
            assert response.status == 200
            assert (await response.json())["engine"] == "alpha"
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_malformed_accepted_wire_and_removed_owner_fail_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = web.Application()

        async def malformed(_: web.Request) -> web.Response:
            return web.json_response(["not", "a", "job"], status=202)

        app.router.add_post("/api/jobs", malformed)
        client, store, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            response = await client.post("/api/jobs", json={"clientId": "c", "jobId": "j"})
            assert response.status == 502
            assert await store.lookup_key("default", "c", "j") == "alpha"
            assert await store.lookup_job("missing") is None
        finally:
            await _close(client, servers)

        stale = LeaseStore(tmp_path / "stale.sqlite3")
        await stale.claim_key("default", "c", "j", "removed-member")
        stale_client = TestClient(
            TestServer(
                create_ingress_app(
                    {"alpha": IngressMember("alpha", EngineLink(state="stopped"))},
                    "alpha",
                    stale,
                )
            )
        )
        await stale_client.start_server()
        try:
            response = await stale_client.get("/api/jobs/c/j")
            assert response.status == 503
            assert await response.json() == {
                "error": "engine-not-ready",
                "engine": "removed-member",
                "state": "not-configured",
            }
            assert await stale.lookup_key("default", "c", "j") == "removed-member"
        finally:
            await stale_client.close()

    asyncio.run(scenario())


def test_accepted_job_store_failure_is_structured_503(tmp_path: Path) -> None:
    class FailingStore(LeaseStore):
        async def record_job(self, job_ref: str, owner_id: str) -> None:
            raise LeaseStoreError("disk full")

    async def scenario() -> None:
        server = TestServer(_engine("alpha", []))
        await server.start_server()
        member = IngressMember(
            "alpha", EngineLink(state="ready", base_url=str(server.make_url("")))
        )
        store = FailingStore(tmp_path / "fleet.sqlite3")
        client = TestClient(TestServer(create_ingress_app({"alpha": member}, "alpha", store)))
        await client.start_server()
        try:
            response = await client.post("/api/jobs", json={"clientId": "c", "jobId": "j"})
            assert response.status == 503
            assert await response.json() == {"error": "ownership-store-unavailable"}
            assert await store.lookup_key("default", "c", "j") == "alpha"
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_conditional_delete_requires_owner_and_ambiguous_failures_preserve(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = web.Application()

        async def missing(_: web.Request) -> web.Response:
            return web.json_response({}, status=404)

        app.router.add_get("/{tail:.*}", missing)
        client, store, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            await store.claim_key("default", "c", "gone", "alpha")
            assert (await client.get("/api/jobs/c/gone")).status == 404
            assert await store.lookup_key("default", "c", "gone") == "alpha"
            assert not await store.delete_key_if_owner("default", "c", "gone", "beta")
            assert await store.delete_key_if_owner("default", "c", "gone", "alpha")
            assert await store.lookup_key("default", "c", "gone") is None
            await store.claim_key("default", "c", "kept", "alpha")
            # Closing the owner makes forwarding ambiguous; its lease remains.
            await servers[0].close()
            assert (await client.get("/api/jobs/c/kept")).status == 503
            assert await store.lookup_key("default", "c", "kept") == "alpha"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_ws_verbatim_filter_serialized_bounded_merge_and_member_reconnect(tmp_path: Path) -> None:
    # The old test relied on the unbounded upstream frame setting. This
    # remains a large legal frame; oversize closure has a dedicated proof.
    large_binary = b"x" * (2 * 1024 * 1024 + 1)

    def ws_engine(primary: bool, connections: list[int]) -> web.Application:
        app = web.Application()

        async def events(request: web.Request) -> web.WebSocketResponse:
            connection_number = len(connections)
            connections.append(1)
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            if primary and connection_number == 0:
                await ws.send_str("primary-text")
                await ws.send_bytes(large_binary)
                for index in range(96):
                    await ws.send_str(f"burst-{index}")
            elif not primary and connection_number == 0:
                await ws.send_str('{"kind":"broadcast"}')
                await ws.send_str('{"jobRef":"member-job","kind":"state"}')
                header = b'{"jobRef":"binary-job"}'
                await ws.send_bytes(len(header).to_bytes(4, "big") + header + b"payload")
            await ws.close()
            return ws

        app.router.add_get("/api/events", events)
        return app

    async def scenario() -> None:
        primary_connections: list[int] = []
        member_connections: list[int] = []
        client, _, servers = await _fleet(
            tmp_path,
            {
                "alpha": ws_engine(True, primary_connections),
                "beta": ws_engine(False, member_connections),
            },
        )
        try:
            ws = await client.ws_connect("/api/events", max_msg_size=0)
            received: list[str | bytes] = []
            async with asyncio.timeout(2):
                while len(received) < 100:
                    received.append((await ws.receive()).data)
                while len(primary_connections) < 2 or len(member_connections) < 2:
                    await asyncio.sleep(0.02)
            assert "primary-text" in received
            assert large_binary in received
            assert '{"jobRef":"member-job","kind":"state"}' in received
            assert any(isinstance(item, bytes) and b"binary-job" in item for item in received)
            assert '{"kind":"broadcast"}' not in received
            assert {f"burst-{index}" for index in range(96)} <= set(received)
            assert not ws.closed
            await ws.close()
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_streams_large_bodies_and_strips_hop_headers(tmp_path: Path) -> None:
    upload_chunks = [b"upload-one" * (128 * 1024), b"upload-two" * (128 * 1024)]
    download_chunks = [b"download-one" * (128 * 1024), b"download-two" * (128 * 1024)]
    first_upload_received = asyncio.Event()
    first_download_received = asyncio.Event()
    seen: dict[str, object] = {}
    app = web.Application(client_max_size=16 * 1024 * 1024)

    async def transfer(request: web.Request) -> web.StreamResponse:
        chunks = [await request.content.readany()]
        first_upload_received.set()
        chunks.extend([chunk async for chunk in request.content.iter_any()])
        seen["body"] = b"".join(chunks)
        seen["header_names"] = {name.lower() for name in request.headers}
        response = web.StreamResponse(
            headers={
                "Connection": "X-Response-Hop, close",
                "Proxy-Authenticate": "removed",
                "X-Response-Hop": "secret",
            }
        )
        response.headers.add("Connection", "X-Response-Hop-Two")
        response.headers.add("X-Response-Hop-Two", "secret")
        response.headers.add("Set-Cookie", "one=1")
        response.headers.add("Set-Cookie", "two=2")
        await response.prepare(request)
        await response.write(download_chunks[0])
        await first_download_received.wait()
        await response.write(download_chunks[1])
        await response.write_eof()
        return response

    app.router.add_put("/api/assets/upload", transfer)

    async def scenario() -> None:
        async def upload() -> AsyncIterator[bytes]:
            yield upload_chunks[0]
            await first_upload_received.wait()
            yield upload_chunks[1]

        client, _, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            async with asyncio.timeout(3):
                headers = CIMultiDict(
                    {
                        "Connection": "X-Hop",
                        "X-Hop": "secret",
                        "Proxy-Authorization": "secret",
                        "Proxy-Custom": "removed",
                        "TE": "trailers",
                    }
                )
                headers.add("Connection", "X-Hop-Two")
                headers.add("X-Hop-Two", "secret")
                response = await client.put(
                    "/api/assets/upload",
                    data=upload(),
                    headers=headers,
                )
                first_download = await response.content.readany()
                assert first_download
                first_download_received.set()
                received = first_download + await response.read()
                assert response.status == 200
                assert received == b"".join(download_chunks)
                assert seen["body"] == b"".join(upload_chunks)
                header_names = cast("set[str]", seen["header_names"])
                assert "x-hop" not in header_names
                assert "x-hop-two" not in header_names
                assert "proxy-authorization" not in header_names
                assert "proxy-custom" not in header_names
                assert "te" not in header_names
                assert "Connection" not in response.headers
                assert "Proxy-Authenticate" not in response.headers
                assert "X-Response-Hop" not in response.headers
                assert "X-Response-Hop-Two" not in response.headers
                assert response.headers.getall("Set-Cookie") == ["one=1", "two=2"]
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_submit_request_and_accepted_response_body_caps(tmp_path: Path) -> None:
    limit = ingress_module._SUBMIT_BODY_LIMIT

    def request_body(job_id: str, size: int) -> bytes:
        prefix = f'{{"clientId":"c","jobId":"{job_id}","padding":"'.encode()
        suffix = b'"}'
        body = prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix
        assert len(body) == size
        return body

    def accepted_body(job_id: str, size: int) -> bytes:
        prefix = (
            f'{{"clientId":"c","jobId":"{job_id}","jobRef":"ref-{job_id}","padding":"'
        ).encode()
        suffix = b'"}'
        body = prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix
        assert len(body) == size
        return body

    app = web.Application(client_max_size=16 * 1024 * 1024)

    async def submit(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = cast("str", body["jobId"])
        size = limit + 1 if job_id == "response-over" else limit
        response = web.Response(
            body=accepted_body(job_id, size),
            status=202,
            headers={"Connection": "X-Buffered-Hop", "X-Buffered-Hop": "secret"},
        )
        response.headers.add("Connection", "X-Buffered-Hop-Two")
        response.headers.add("X-Buffered-Hop-Two", "secret")
        return response

    app.router.add_post("/api/jobs", submit)

    async def scenario() -> None:
        client, _, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            exact_request = await client.post(
                "/api/jobs", data=request_body("request-exact", limit)
            )
            assert exact_request.status == 202
            over_request = await client.post(
                "/api/jobs", data=request_body("request-over", limit + 1)
            )
            assert over_request.status == 413
            assert (await over_request.json())["error"] == "job-body-too-large"

            exact_response = await client.post(
                "/api/jobs", data=request_body("response-exact", 128)
            )
            assert exact_response.status == 202
            assert len(await exact_response.read()) == limit
            assert "Connection" not in exact_response.headers
            assert "X-Buffered-Hop" not in exact_response.headers
            assert "X-Buffered-Hop-Two" not in exact_response.headers
            over_response = await client.post("/api/jobs", data=request_body("response-over", 128))
            assert over_response.status == 502
            assert (await over_response.json())["error"] == "accepted-body-too-large"
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_upstream_failure_after_prepare_truncates_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = web.Application()
    not_ready_calls = 0
    original_not_ready = ingress_module._not_ready

    def tracked_not_ready(member: IngressMember) -> web.Response:
        nonlocal not_ready_calls
        not_ready_calls += 1
        return original_not_ready(member)

    monkeypatch.setattr(ingress_module, "_not_ready", tracked_not_ready)

    async def transfer(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first-chunk")
        assert request.transport is not None
        request.transport.close()
        return response

    app.router.add_get("/api/assets/file", transfer)

    async def scenario() -> None:
        client, _, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            response = await client.get("/api/assets/file")
            assert response.status == 200
            assert await response.content.readexactly(len(b"first-chunk")) == b"first-chunk"
            with pytest.raises(aiohttp.ClientPayloadError):
                await response.read()
            assert not_ready_calls == 0
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_reconnect_jitter_never_exceeds_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingress_module.random, "uniform", lambda _low, _high: 1.2)
    assert ingress_module._reconnect_delay(ingress_module._RECONNECT_MAX) == (
        ingress_module._RECONNECT_MAX
    )


def test_ws_queue_blocks_on_multiple_large_frames_and_has_heartbeat() -> None:
    async def scenario() -> None:
        large = b"x" * (2 * 1024 * 1024)
        queue = ingress_module._FrameQueue(max_frames=64, max_bytes=4 * 1024 * 1024)
        await queue.put(True, large)
        await queue.put(True, large)
        blocked = asyncio.create_task(queue.put(True, large))
        await asyncio.sleep(0)
        assert not blocked.done()
        assert await queue.get() == (True, large)
        await asyncio.wait_for(blocked, 1)
        assert await queue.get() == (True, large)
        assert await queue.get() == (True, large)

    asyncio.run(scenario())
    assert ingress_module._WS_HEARTBEAT == 30


def test_oversized_upstream_ws_frame_loudly_closes_downstream(tmp_path: Path) -> None:
    app = web.Application()

    async def events(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_bytes(b"x" * (4 * 1024 * 1024 + 1))
        await ws.close()
        return ws

    app.router.add_get("/api/events", events)

    async def scenario() -> None:
        client, _, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            ws = await client.ws_connect("/api/events", max_msg_size=0)
            message = await asyncio.wait_for(ws.receive(), 2)
            assert message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            assert ws.close_code == WSCloseCode.MESSAGE_TOO_BIG
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_immediate_ws_close_uses_growing_reconnect_backoff(tmp_path: Path) -> None:
    connections: list[float] = []
    app = web.Application()

    async def events(request: web.Request) -> web.WebSocketResponse:
        connections.append(time.monotonic())
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close()
        return ws

    app.router.add_get("/api/events", events)

    async def scenario() -> None:
        client, _, servers = await _fleet(tmp_path, {"alpha": app})
        try:
            ws = await client.ws_connect("/api/events")
            await asyncio.sleep(0.75)
            assert 2 <= len(connections) <= 4
            await ws.close()
        finally:
            await _close(client, servers)

    asyncio.run(scenario())


def test_aggregate_fanout_is_concurrent_and_times_out_by_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    concurrent_release = asyncio.Event()
    concurrent_requests = 0

    def engine(*, hangs: bool = False, prove_concurrency: bool = False) -> web.Application:
        app = web.Application()

        async def jobs(_: web.Request) -> web.Response:
            nonlocal concurrent_requests
            if hangs:
                await release.wait()
            elif prove_concurrency:
                concurrent_requests += 1
                if concurrent_requests == 2:
                    concurrent_release.set()
                await concurrent_release.wait()
            return web.json_response({"jobs": []})

        app.router.add_get("/api/jobs", jobs)
        return app

    async def scenario() -> None:
        client, _, servers = await _fleet(
            tmp_path,
            {
                "alpha": engine(prove_concurrency=True),
                "beta": engine(prove_concurrency=True),
            },
        )
        try:
            response = await asyncio.wait_for(client.get("/api/jobs"), 2)
            assert response.status == 200
            assert concurrent_requests == 2
        finally:
            await _close(client, servers)

        monkeypatch.setattr(ingress_module, "_FANOUT_TIMEOUT", 0.15)
        client, _, servers = await _fleet(tmp_path, {"alpha": engine(), "beta": engine(hangs=True)})
        try:
            started = time.monotonic()
            response = await client.get("/api/jobs")
            elapsed = time.monotonic() - started
            assert response.status == 503
            assert elapsed < 0.4
            assert (await response.json())["members"] == [
                {"engine": "beta", "state": "unreachable"}
            ]
        finally:
            release.set()
            await _close(client, servers)

    asyncio.run(scenario())
