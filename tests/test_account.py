"""Native account flow against local identity HTTP contracts, never production."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen

import pytest

from dinkster.account import AccountError, AccountSession
from dinkster.cli import main
from dinkster.services import resolve_services

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
                else [{"id": PROJECT, "display_name": "My project"}]
            )
            self.reply(rows)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            assert not self.headers.get("Cookie")
            observed["requests"].append((self.path, self.headers.get("Authorization")))
            if self.path == "/register":
                document = json.loads(raw)
                assert document["scope"] == "projects:gateway offline_access"
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
                    access, refresh = "oauth-first", "refresh-first"
                else:
                    assert fields["refresh_token"] == ["refresh-first"]
                    access, refresh = "oauth-next", "refresh-next"
                self.reply(
                    {
                        "access_token": access,
                        "refresh_token": refresh,
                        "scope": "projects:gateway offline_access",
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
        assert fields["scope"] == ["projects:gateway offline_access"]
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
        replace(services, gateway_audience="gateway-qa"),
        home=tmp_path / "user",
        install_root=tmp_path / "install",
    )
    assert other.token_path != session.token_path
    with pytest.raises(AccountError, match="Sign in"):
        other.projects()
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
        session.logout()
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
