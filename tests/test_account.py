"""Native account flow against local identity HTTP contracts, never production."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_library import AsyncLibraryClient, LibraryClient
from dinkster_server.auth import Principal, install_auth
from dinkster_server.projects import PROJECTS_KEY, project_request

from dinkster.account import AccountError, AccountSession, AccountSignIn
from dinkster.account_api import add_account_routes
from dinkster.cli import main
from dinkster.services import resolve_services
from tests.async_scenarios import run_scenario

PROJECT = "89790f18-fdad-4f48-bf67-823fdddb89da"


@pytest.fixture
def identity():
    observed: dict = {"requests": [], "challenge": "", "deny": False, "redirect": False}

    class Identity(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, value, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            observed["requests"].append((self.path, self.headers.get("Authorization")))
            assert not self.headers.get("Cookie")
            assert self.headers.get("Authorization") in {"Bearer oauth-first", "Bearer oauth-next"}
            if observed["redirect"]:
                self.send_response(302)
                self.send_header("Location", observed["base"] + "/not-allowed")
                self.end_headers()
                return
            if self.path == "/userinfo":
                if observed.get("profile_denied"):
                    self.reply({"error": "private-provider-detail"}, 403)
                    return
                self.reply(
                    {
                        "sub": "account-id",
                        "name": observed.get("name", "  Test Person  "),
                        "email": observed.get("email", " person@example.test "),
                        "picture": "https://images.example.test/avatar.png",
                        "access_token": "must-not-leak",
                        "refresh_token": "must-not-leak",
                    }
                )
                return
            offset = parse_qs(urlsplit(self.path).query)["offset"][0]
            rows = (
                [
                    {
                        "id": f"00000000-0000-0000-0000-{index:012d}",
                        "display_name": f"Project {index}",
                    }
                    for index in range(100)
                ]
                if offset == "0"
                else [
                    {
                        "id": PROJECT,
                        "display_name": "My project",
                        **observed.get("project_extra", {}),
                    }
                ]
            )
            self.reply(rows)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            assert not self.headers.get("Cookie")
            observed["requests"].append((self.path, self.headers.get("Authorization")))
            if self.path == "/register":
                document = json.loads(raw)
                assert document["scope"] == "openid profile email projects:gateway offline_access"
                assert document["token_endpoint_auth_method"] == "none"
                assert document["grant_types"] == ["authorization_code", "refresh_token"]
                assert document["redirect_uris"][0].startswith("http://127.0.0.1:")
                self.reply({"client_id": "public-install-client"})
            elif self.path == "/token":
                fields = parse_qs(raw.decode())
                if observed.get("lost_refresh") and fields["grant_type"] == ["refresh_token"]:
                    self.connection.close()
                    return
                assert fields["client_id"] == ["public-install-client"]
                if fields["grant_type"] == ["authorization_code"]:
                    assert fields["code"] == ["one-use-code"]
                    challenge = (
                        base64.urlsafe_b64encode(
                            hashlib.sha256(fields["code_verifier"][0].encode()).digest()
                        )
                        .rstrip(b"=")
                        .decode()
                    )
                    assert challenge == observed["challenge"]
                    if "exchange_started" in observed:
                        observed["exchange_started"].set()
                        assert observed["exchange_continue"].wait(timeout=5)
                    access, refresh = "oauth-first", "refresh-first"
                else:
                    assert fields["refresh_token"] == ["refresh-first"]
                    access, refresh = "oauth-next", "refresh-next"
                self.reply(
                    {
                        "access_token": access,
                        "refresh_token": refresh,
                        "scope": observed.get(
                            "scope", "openid profile email projects:gateway offline_access"
                        ),
                        "token_type": "Bearer",
                        "expires_in": 30
                        if access == "oauth-first"
                        else observed.get("oauth_lifetime", 600),
                    }
                )
            else:
                assert self.path == f"/v1/oauth/projects/{PROJECT}/gateway-token"
                assert raw == b"{}"
                assert self.headers["Authorization"] == "Bearer oauth-next"
                if observed["deny"]:
                    self.reply({"error": "project_access_revoked"}, 403)
                else:
                    self.reply(
                        {
                            "access_token": observed.get("gateway_token", "project-gateway-token"),
                            "token_type": "Bearer",
                            "expires_in": observed.get("gateway_lifetime", 600),
                        }
                    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Identity)
    observed["base"] = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def browser(identity: dict):
    threads = []
    identity["callback_statuses"] = []

    def open_browser(url: str):
        fields = parse_qs(urlsplit(url).query)
        assert fields["scope"] == ["openid profile email projects:gateway offline_access"]
        assert fields["code_challenge_method"] == ["S256"]
        assert fields["resource"] == [identity["base"]]
        identity["challenge"] = fields["code_challenge"][0]

        def callback():
            target = fields["redirect_uri"][0]
            for wrong in ({"state": "wrong"}, {"iss": "https://foreign.example"}):
                params = {
                    "state": fields["state"][0],
                    "iss": identity["base"],
                    "code": "one-use-code",
                    **wrong,
                }
                with pytest.raises(HTTPError) as failure:
                    urlopen(target + "?" + urlencode(params), timeout=5)
                identity["callback_statuses"].append(failure.value.code)
            params = {"state": fields["state"][0], "iss": identity["base"], "code": "one-use-code"}
            with urlopen(target + "?" + urlencode(params), timeout=5) as response:
                identity["callback_statuses"].append(response.status)

        thread = threading.Thread(target=callback, daemon=True)
        thread.start()
        threads.append(thread)

    return open_browser, threads


def test_pkce_project_selection_refresh_revocation_and_endpoint_isolation(
    identity: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=tmp_path / "user", install_root=tmp_path / "install")
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert identity["callback_statuses"] == [400, 400, 200]
    assert not session.token_path.exists()
    projects = session.projects()
    assert len(projects) == 101
    assert projects[0]["display_name"] == "Project 0"
    assert projects[-1] == {"id": PROJECT, "display_name": "My project"}
    session.select_project(PROJECT)
    assert session.token_path.read_text() == "project-gateway-token"
    state = json.loads(session.state_path.read_text())
    assert state["refresh_token"] == "refresh-next"
    assert state["project_id"] == PROJECT
    assert session.state_path.parent != session.token_path.parent
    if os.name != "nt":
        assert session.state_path.stat().st_mode & 0o777 == 0o600
        assert session.token_path.stat().st_mode & 0o777 == 0o600
        assert session.state_path.parent.stat().st_mode & 0o777 == 0o700
    assert not (tmp_path / "install").exists()
    other = AccountSession(
        replace(services, gateway_audience="gateway-qa", environment="qa"),
        home=tmp_path / "user",
        install_root=tmp_path / "install",
    )
    assert other.token_path != session.token_path
    with pytest.raises(AccountError, match="Sign in"):
        other.projects()
    original_state = session.state_path.read_bytes()
    original_token = session.token_path.read_bytes()
    qa_opener, qa_callbacks = browser(identity)
    other.login(open_browser=qa_opener, timeout=5)
    for thread in qa_callbacks:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert other.projects() == projects
    assert session.state_path.read_bytes() == original_state
    assert session.token_path.read_bytes() == original_token
    qa_state = other.state_path.read_bytes()
    assert session.projects() == projects
    assert other.state_path.read_bytes() == qa_state
    other.logout()
    assert session.state_path.read_bytes() == original_state
    assert session.token_path.read_bytes() == original_token
    assert not (tmp_path / "install").exists()
    second = AccountSession(services, home=tmp_path / "user", install_root=tmp_path / "second")
    assert second.state_path != session.state_path
    with pytest.raises(AccountError, match="Sign in"):
        second.projects()
    identity["redirect"] = True
    with pytest.raises(AccountError, match="HTTP 302"):
        session.projects()
    assert not any(path == "/not-allowed" for path, _ in identity["requests"])
    identity["redirect"] = False
    identity["deny"] = True
    before = len(identity["requests"])
    with pytest.raises(AccountError, match="HTTP 403"):
        session.refresh_gateway()
    assert len(identity["requests"]) == before + 1
    assert not session.token_path.exists()
    session.logout()
    assert not session.state_path.exists()
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "user"))
    assert (
        main(
            [
                "account",
                "--identity-base",
                identity["base"],
                "--gateway-base",
                identity["base"],
                "--catalog-base",
                identity["base"],
                "projects",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "Sign in" in captured.err
    for secret in (
        "oauth-first",
        "oauth-next",
        "refresh-first",
        "refresh-next",
        "project-gateway-token",
    ):
        assert secret not in captured.out + captured.err


def test_automatic_refresh_boundary_and_lost_rotation_are_fail_closed(
    identity: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import account

    clock = [account.time.time()]
    monkeypatch.setattr(account.time, "time", lambda: clock[0])
    identity["oauth_lifetime"] = 3600
    session = AccountSession(
        resolve_services(identity_base=identity["base"], catalog_base=identity["base"]),
        home=tmp_path,
    )
    session.ensure_gateway()
    assert not identity["requests"]
    assert not session.token_path.exists()
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    session.ensure_gateway()
    assert not session.token_path.exists()
    session.select_project(PROJECT)
    before = len(identity["requests"])
    clock[0] += 539
    session.ensure_gateway()
    assert len(identity["requests"]) == before
    identity["gateway_token"] = "replacement-bearer"
    clock[0] += 2
    session.ensure_gateway()
    assert len(identity["requests"]) == before + 1
    assert session.token_path.read_text() == "replacement-bearer"
    session.token_path.unlink()
    session.ensure_gateway()
    assert len(identity["requests"]) == before + 2
    assert session.token_path.read_text() == "replacement-bearer"
    identity["lost_refresh"] = True
    clock[0] += 3600
    with pytest.raises(AccountError, match="no exchange was retried"):
        session.ensure_gateway()
    after_failure = len(identity["requests"])
    assert not session.state_path.exists()
    assert not session.token_path.exists()
    session.ensure_gateway()
    assert len(identity["requests"]) == after_failure


def test_engine_refreshes_only_the_selected_install_account(identity: dict, tmp_path: Path) -> None:
    install = tmp_path / "install"
    home = tmp_path / "user"
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=home, install_root=install)
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    identity["gateway_lifetime"] = 1
    session.select_project(PROJECT)
    identity["gateway_token"] = "engine-startup-bearer"
    identity["gateway_lifetime"] = 600
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    log = tmp_path / "engine.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dinkster.serve",
                "--no-default-packs",
                "--account-root",
                str(install),
                "--account-identity-base",
                identity["base"],
                "--remote-catalog-base",
                identity["base"],
                "--remote-gateway-base",
                identity["base"],
                "--library-root",
                str(tmp_path / "library"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env={**os.environ, "DINKSTER_HOME": str(home)},
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:

        def wait_for_bearer(expected: str) -> None:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                if session.token_path.exists() and session.token_path.read_text() == expected:
                    return
                time.sleep(0.05)
            pytest.fail("engine did not refresh the install bearer: " + log.read_text())

        wait_for_bearer("engine-startup-bearer")
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
            assert response.status == 200
        identity["gateway_token"] = "engine-periodic-bearer"
        session.token_path.unlink()
        wait_for_bearer("engine-periodic-bearer")
        with urlopen(f"http://127.0.0.1:{port}/api/account", timeout=5) as response:
            status = json.load(response)
        assert status["signedIn"] is True
        assert status["profile"] == {"displayName": "Test Person", "email": "person@example.test"}
        request = Request(
            f"http://127.0.0.1:{port}/api/account/sign-out",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            assert json.load(response)["signedIn"] is False
        assert not session.token_path.exists()
        assert not session.state_path.exists()
    finally:
        process.terminate()
        process.wait(timeout=30)
    assert "oauth-first" not in log.read_text()
    assert "refresh-next" not in log.read_text()


def test_consent_denial_never_exchanges_or_persists_tokens(identity: dict, tmp_path: Path) -> None:
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=tmp_path)
    callbacks = []
    statuses = []

    def deny(url: str) -> None:
        fields = parse_qs(urlsplit(url).query)
        target = fields["redirect_uri"][0]
        params = {"state": fields["state"][0], "iss": identity["base"], "error": "access_denied"}

        def callback() -> None:
            with urlopen(target + "?" + urlencode(params), timeout=5) as response:
                statuses.append(response.status)

        thread = threading.Thread(target=callback, daemon=True)
        thread.start()
        callbacks.append(thread)

    with pytest.raises(AccountError, match="consent was refused"):
        session.login(open_browser=deny, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    assert statuses == [200]
    assert [path for path, _ in identity["requests"]] == ["/register"]
    assert not session.state_path.exists()
    assert not session.token_path.exists()


def test_cancel_during_exchange_never_saves_late_tokens(identity, tmp_path):
    session = AccountSession(
        resolve_services(identity_base=identity["base"], catalog_base=identity["base"]),
        home=tmp_path,
    )
    attempt = AccountSignIn()
    identity["exchange_started"] = threading.Event()
    identity["exchange_continue"] = threading.Event()
    opener, callbacks = browser(identity)
    with ThreadPoolExecutor() as pool:
        result = pool.submit(session.login, open_browser=opener, timeout=5, sign_in=attempt)
        try:
            assert identity["exchange_started"].wait(timeout=5)
            attempt.cancel()
        finally:
            identity["exchange_continue"].set()
        with pytest.raises(AccountError, match="cancelled"):
            result.result(timeout=5)
    for callback in callbacks:
        callback.join(timeout=5)
    assert not session.state_path.exists()
    assert not session.token_path.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_browser_account_native_flow_cancel_profile_and_local_logout(identity, tmp_path, legacy):
    if legacy:
        identity["scope"] = "projects:gateway offline_access"
    run_scenario(_browser_account_native_flow(identity, tmp_path))


async def _browser_account_native_flow(identity, tmp_path):
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=tmp_path)
    app = web.Application()
    install_auth(app)
    add_account_routes(app, session)
    async with TestClient(TestServer(app)) as client:
        out = await client.get("/api/account")
        assert await out.json() == {
            "available": True,
            "canStartSignIn": True,
            "signInUnavailableReason": None,
            "signedIn": False,
            "profile": None,
            "signIn": {"state": "idle"},
        }
        start = await client.post("/api/account/sign-in", json={})
        assert start.status == 202
        pending = await start.json()
        assert pending["signIn"] == {"state": "pending"}
        choices = await client.get("/api/account/projects")
        assert choices.status == 409
        assert await choices.json() == {"error": "account-sign-in-pending"}
        fields = parse_qs(urlsplit(pending["authorizationUrl"]).query)
        assert fields["nonce"]
        duplicate = await client.post("/api/account/sign-in", json={})
        assert (await duplicate.json())["authorizationUrl"] == pending["authorizationUrl"]
        cancelled = await client.delete("/api/account/sign-in")
        assert (await cancelled.json())["signIn"] == {"state": "idle"}
        assert not session.state_path.exists()
        assert not session.token_path.exists()
        assert [path for path, _ in identity["requests"]] == ["/register"]
        start = await client.post("/api/account/sign-in", json={})
        fields = parse_qs(urlsplit((await start.json())["authorizationUrl"]).query)
        denied_url = (
            fields["redirect_uri"][0]
            + "?"
            + urlencode(
                {"state": fields["state"][0], "iss": identity["base"], "error": "private-error"}
            )
        )

        def deny():
            with urlopen(denied_url, timeout=5) as response:
                assert response.status == 200

        await asyncio.to_thread(deny)
        document = {}
        for _ in range(100):
            document = await (await client.get("/api/account")).json()
            if document["signIn"]["state"] != "pending":
                break
            await asyncio.sleep(0.01)
        assert document["signIn"] == {"state": "failed", "error": "account-sign-in-failed"}
        assert document["signedIn"] is False
        assert not session.state_path.exists()
        assert not session.token_path.exists()
        start = await client.post("/api/account/sign-in", json={})
        fields = parse_qs(urlsplit((await start.json())["authorizationUrl"]).query)
        identity["challenge"] = fields["code_challenge"][0]
        callback = (
            fields["redirect_uri"][0]
            + "?"
            + urlencode(
                {"state": fields["state"][0], "iss": identity["base"], "code": "one-use-code"}
            )
        )

        def finish():
            with urlopen(callback, timeout=5) as response:
                assert response.status == 200

        await asyncio.to_thread(finish)
        document = {}
        response = out
        for _ in range(100):
            response = await client.get("/api/account")
            document = await response.json()
            if document.get("signedIn"):
                break
            await asyncio.sleep(0.01)
        assert document == {
            "available": True,
            "canStartSignIn": True,
            "signInUnavailableReason": None,
            "signedIn": True,
            "profile": None
            if identity.get("scope")
            else {"displayName": "Test Person", "email": "person@example.test"},
            "signIn": {"state": "idle"},
        }
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        if identity.get("scope"):
            assert not any(path == "/userinfo" for path, _ in identity["requests"])
        else:
            identity.update(name=" \t", email={"unsafe": "value"})
            empty_profile = await (await client.get("/api/account")).json()
            assert empty_profile["profile"] == {"displayName": None, "email": None}
            identity["profile_denied"] = True
            unavailable = await client.get("/api/account")
            assert unavailable.status == 503
            assert await unavailable.json() == {"error": "account-status-unavailable"}
        await asyncio.to_thread(session.select_project, PROJECT)
        assert session.token_path.exists()
        signed_out = await client.post("/api/account/sign-out", json={})
        assert (await signed_out.json())["signedIn"] is False
        assert not session.state_path.exists()
        assert not session.token_path.exists()


def test_account_project_link_is_sanitized_authorized_and_revision_guarded(identity, tmp_path):
    identity["project_extra"] = {"access_token": "private-provider-token", "owner": "private-owner"}
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=tmp_path / "account")
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)

    class Auth:
        async def authenticate(self, token):
            if token in {"operator", "agent", "editor"}:
                capabilities = {"assets:read", "assets:write", "settings:read"}
                if token != "editor":
                    capabilities.add("settings:write")
                return Principal(
                    token,
                    {"*": frozenset(capabilities)},
                    kind="agent" if token == "agent" else "human",
                )
            return None

    async def scenario():
        with LibraryClient(tmp_path / "library") as helper:
            helper.call("projects.register", id="film")
            helper.call("projects.register", id="other")
            helper.call("projects.register", id="deleted")
            helper.call("projects.delete", id="deleted", revision=1)
            app = web.Application()
            app[PROJECTS_KEY] = AsyncLibraryClient(helper)
            install_auth(app, Auth())
            add_account_routes(app, session)
            app.router.add_patch("/api/projects/{project}", project_request)
            headers = {"Authorization": "Bearer operator"}
            async with TestClient(TestServer(app), headers=headers) as client:
                for token, expected in (("unknown", 401), ("agent", 403), ("editor", 403)):
                    for method, path in (
                        ("GET", "/api/account/projects"),
                        ("PUT", "/api/account/projects/film"),
                    ):
                        response = await client.request(
                            method,
                            path,
                            headers={"Authorization": f"Bearer {token}"},
                            json={"revision": 1, "accountProject": PROJECT},
                        )
                        assert response.status == expected
                response = await client.get("/api/account/projects")
                choices = await response.json()
                assert response.status == 200
                assert choices["issuer"] == identity["base"]
                assert len(choices["projects"]) == 101
                assert choices["projects"][-1] == {"id": PROJECT, "displayName": "My project"}
                assert all(set(row) == {"id", "displayName"} for row in choices["projects"])
                assert response.headers["Cache-Control"] == "no-store"
                identity["project_extra"]["display_name"] = " \t"
                response = await client.get("/api/account/projects")
                assert (await response.json())["projects"][-1]["displayName"] is None
                path = "/api/account/projects/film"
                for body in (
                    {},
                    {"revision": True, "accountProject": PROJECT},
                    {"revision": 1, "accountProject": "not-a-uuid"},
                    {"revision": 1, "accountProject": PROJECT, "issuer": "https://wrong.test"},
                ):
                    assert (await client.put(path, json=body)).status == 400
                unknown = "ffffffff-ffff-4fff-bfff-ffffffffffff"
                assert (
                    await client.put(path, json={"revision": 1, "accountProject": unknown})
                ).status == 400
                for project in ("missing", "deleted"):
                    assert (
                        await client.put(
                            f"/api/account/projects/{project}",
                            json={"revision": 1, "accountProject": PROJECT},
                        )
                    ).status == 404
                direct = await client.patch(
                    "/api/projects/film",
                    json={
                        "revision": 1,
                        "patch": {"accountProject": {"issuer": identity["base"], "id": PROJECT}},
                    },
                )
                assert direct.status == 400
                response = await client.put(path, json={"revision": 1, "accountProject": PROJECT})
                assert response.status == 200
                linked = (await response.json())["project"]
                assert linked["accountProject"] == {"issuer": identity["base"], "id": PROJECT}
                assert linked["revision"] == 2
                other = cast(dict, helper.call("projects.get", id="other")["project"])
                assert other["accountProject"] is None
                assert not session.token_path.exists()
                stale = await client.put(path, json={"revision": 1, "accountProject": None})
                assert stale.status == 409
                await asyncio.to_thread(session.logout)
                unavailable = await client.put(
                    path, json={"revision": 2, "accountProject": PROJECT}
                )
                assert unavailable.status == 503
                assert helper.call("projects.get", id="film")["project"] == linked
                response = await client.put(path, json={"revision": 2, "accountProject": None})
                assert response.status == 200
                assert (await response.json())["project"]["accountProject"] is None
                assert not session.state_path.exists()

    run_scenario(scenario())


def test_unconfigured_account_routes_do_not_offer_sign_in():
    async def scenario():
        app = web.Application()
        install_auth(app)
        add_account_routes(app, None)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/api/account")
            assert await response.json() == {
                "available": False,
                "canStartSignIn": False,
                "signInUnavailableReason": "account-unavailable",
                "signedIn": False,
                "profile": None,
                "signIn": {"state": "idle"},
            }
            response = await client.post("/api/account/sign-in", json={})
            assert response.status == 409
            assert await response.json() == {"error": "account-unavailable"}

    run_scenario(scenario())


def test_account_routes_require_operator_and_bind_pending_owner(identity, tmp_path):
    run_scenario(_account_route_authority(identity, tmp_path))


async def _account_route_authority(identity, tmp_path):
    class Auth:
        async def authenticate(self, token):
            if token == "operator":
                return Principal("owner", {"*": frozenset({"settings:read", "settings:write"})})
            if token == "other":
                return Principal("other", {"*": frozenset({"settings:read", "settings:write"})})
            if token == "agent":
                return Principal(
                    "agent", {"*": frozenset({"settings:read", "settings:write"})}, kind="agent"
                )
            return None

    session = AccountSession(
        resolve_services(identity_base=identity["base"], catalog_base=identity["base"]),
        home=tmp_path,
    )
    app = web.Application()
    install_auth(app, Auth())
    add_account_routes(app, session)
    async with TestClient(TestServer(app)) as client:
        for method, path in (
            ("GET", "/api/account"),
            ("POST", "/api/account/sign-in"),
            ("DELETE", "/api/account/sign-in"),
            ("POST", "/api/account/sign-out"),
        ):
            response = await client.request(method, path, json={})
            assert response.status == 401
            response = await client.request(
                method, path, json={}, headers={"Authorization": "Bearer agent"}
            )
            assert response.status == 403
        started = await client.post(
            "/api/account/sign-in", json={}, headers={"Authorization": "Bearer operator"}
        )
        assert started.status == 202
        for method, path in (
            ("POST", "/api/account/sign-in"),
            ("DELETE", "/api/account/sign-in"),
            ("POST", "/api/account/sign-out"),
        ):
            response = await client.request(
                method, path, json={}, headers={"Authorization": "Bearer other"}
            )
            assert response.status == 403
        remote_status = await client.get(
            "/api/account", headers={"Authorization": "Bearer operator", "Host": "remote.example"}
        )
        remote_document = await remote_status.json()
        assert remote_document["signIn"] == {"state": "pending"}
        cancelled = await client.delete(
            "/api/account/sign-in",
            headers={"Authorization": "Bearer operator", "Host": "remote.example"},
        )
        assert cancelled.status == 200
        assert (await cancelled.json())["signIn"] == {"state": "idle"}
        assert remote_document["canStartSignIn"] is False
        assert (
            remote_document["signInUnavailableReason"] == "account-sign-in-requires-engine-machine"
        )
        remote = await client.post(
            "/api/account/sign-in",
            json={},
            headers={"Authorization": "Bearer operator", "Host": "remote.example"},
        )
        assert remote.status == 409
        assert (await remote.json())["error"] == "account-sign-in-requires-engine-machine"
        started = await client.post(
            "/api/account/sign-in", json={}, headers={"Authorization": "Bearer operator"}
        )
        fields = parse_qs(urlsplit((await started.json())["authorizationUrl"]).query)
        identity["challenge"] = fields["code_challenge"][0]
        callback_url = (
            fields["redirect_uri"][0]
            + "?"
            + urlencode(
                {"state": fields["state"][0], "iss": identity["base"], "code": "one-use-code"}
            )
        )

        def complete():
            with urlopen(callback_url, timeout=5) as response:
                assert response.status == 200

        await asyncio.to_thread(complete)
        for _ in range(100):
            response = await client.get("/api/account", headers={"Authorization": "Bearer other"})
            if (await response.json()).get("signedIn"):
                break
            await asyncio.sleep(0.01)
        response = await client.post(
            "/api/account/sign-out", json={}, headers={"Authorization": "Bearer other"}
        )
        assert response.status == 200
        assert (await response.json())["signedIn"] is False
