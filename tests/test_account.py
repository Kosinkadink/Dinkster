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


def _legacy_selection(
    session: AccountSession, project: str = PROJECT, token: str = "legacy-bearer"
) -> None:
    state = json.loads(session.state_path.read_text())
    state.update(project_id=project, gateway_expires_at=time.time() + 600)
    session.state_path.write_text(json.dumps(state))
    session.token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    session.token_path.write_text(token)
    session.token_path.chmod(0o600)


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
                project = self.path.split("/")[4]
                assert self.path == f"/v1/oauth/projects/{project}/gateway-token"
                assert project in {PROJECT, *observed.get("project_tokens", {})}
                assert raw == b"{}"
                assert self.headers["Authorization"] == "Bearer oauth-next"
                if observed["deny"] or project in observed.get("deny_projects", ()):
                    self.reply({"error": "project_access_revoked"}, 403)
                else:
                    self.reply(
                        {
                            "access_token": observed.get("project_tokens", {}).get(
                                project, observed.get("gateway_token", "project-gateway-token")
                            ),
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
    session.prepare_run("selected", {"issuer": identity["base"], "id": PROJECT})
    token = next(session.run_token_directory.iterdir())
    assert token.read_text() == "project-gateway-token"
    state = json.loads(session.state_path.read_text())
    assert state["refresh_token"] == "refresh-next"
    assert "project_id" not in state
    assert not session.token_path.exists()
    assert session.state_path.parent != token.parent
    if os.name != "nt":
        assert session.state_path.stat().st_mode & 0o777 == 0o600
        assert token.stat().st_mode & 0o777 == 0o600
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
    original_token = token.read_bytes()
    qa_opener, qa_callbacks = browser(identity)
    other.login(open_browser=qa_opener, timeout=5)
    for thread in qa_callbacks:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert other.projects() == projects
    assert session.state_path.read_bytes() == original_state
    assert token.read_bytes() == original_token
    qa_state = other.state_path.read_bytes()
    assert session.projects() == projects
    assert other.state_path.read_bytes() == qa_state
    other.logout()
    assert session.state_path.read_bytes() == original_state
    assert token.read_bytes() == original_token
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
    token.unlink()
    with pytest.raises(AccountError, match="unavailable"):
        session.refresh_runs()
    assert len(identity["requests"]) == before + 1
    assert not token.exists()
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


def test_run_credentials_stay_project_bound_and_cannot_survive_new_consent(identity, tmp_path):
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=tmp_path)
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    _legacy_selection(session)
    legacy = session.token_path.read_bytes()
    other = "c660f30c-c170-4312-a08a-542d6ce39e93"
    identity["project_tokens"] = {PROJECT: "film-bearer", other: "other-bearer"}
    identity["gateway_lifetime"] = 1
    film = {"issuer": identity["base"], "id": PROJECT}
    second = {"issuer": identity["base"], "id": other}
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: session.prepare_run(*item), (("film", film), ("other", second))))
    film_token = session.run_token_directory / hashlib.sha256(b"film").hexdigest()
    other_token = session.run_token_directory / hashlib.sha256(b"other").hexdigest()
    assert film_token.read_text() == "film-bearer"
    assert other_token.read_text() == "other-bearer"
    assert session.token_path.read_bytes() == legacy
    assert json.loads(session.state_path.read_text())["project_id"] == PROJECT
    if os.name != "nt":
        assert film_token.stat().st_mode & 0o777 == 0o600
        assert session.run_token_directory.stat().st_mode & 0o777 == 0o700
    session.prepare_run("unlinked", None)
    assert len(list(session.run_token_directory.iterdir())) == 2
    for run, link in (("film", second), ("unlinked", film)):
        with pytest.raises(AccountError, match="already selected"):
            session.prepare_run(run, link)
    before = len(identity["requests"])
    with pytest.raises(AccountError, match="issuer"):
        session.prepare_run("wrong-service", {"issuer": "https://wrong.test", "id": PROJECT})
    assert len(identity["requests"]) == before
    identity["project_tokens"] = {PROJECT: "film-refreshed", other: "other-refreshed"}
    session.refresh_runs()
    assert film_token.read_text() == "film-refreshed"
    assert other_token.read_text() == "other-refreshed"
    identity["deny_projects"] = [PROJECT]
    identity["project_tokens"][other] = "other-still-usable"
    identity["gateway_lifetime"] = 600
    with pytest.raises(AccountError, match="unavailable"):
        session.refresh_runs()
    assert not film_token.exists()
    assert other_token.read_text() == "other-still-usable"
    peer = AccountSession(services, home=tmp_path)
    peer.logout()
    assert not other_token.exists()
    opener, callbacks = browser(identity)
    peer.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    before = len(identity["requests"])
    session.refresh_runs()
    assert len(identity["requests"]) == before
    assert not list(session.run_token_directory.iterdir())
    session.prepare_run("new-consent", second)
    assert len(list(session.run_token_directory.iterdir())) == 1
    session.finish_run("new-consent")
    assert not list(session.run_token_directory.iterdir())
    session.prepare_run("signed-out-run", second)
    session.logout()
    assert not list(session.run_token_directory.iterdir())
    with pytest.raises(AccountError, match="already selected"):
        session.prepare_run("signed-out-run", film)
    session.finish_run("signed-out-run")


def test_server_run_credentials_use_library_links_and_settle_all_terminal_states(
    identity: dict, tmp_path: Path
) -> None:
    from collections.abc import Mapping

    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine
    from dinkster_schema import (
        InputSpec,
        Node,
        NodeSchema,
        OutputSpec,
        TypeExpr,
        build_node_types,
        build_schemas,
    )
    from dinkster_server import create_app
    from dinkster_server.library import run_owned_thread
    from dinkster_server.projects import add_project_routes
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker, current_execution_context

    session = AccountSession(
        resolve_services(identity_base=identity["base"], catalog_base=identity["base"]),
        home=tmp_path / "credentials",
    )
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    other = "c660f30c-c170-4312-a08a-542d6ce39e93"
    identity["project_tokens"] = {PROJECT: "film-token", other: "other-token"}

    async def scenario() -> None:
        ready: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        release = asyncio.Event()

        class Probe(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                string = TypeExpr.concrete("STRING")
                return NodeSchema(
                    node_type="test.run-account",
                    inputs=(InputSpec("action", string),),
                    outputs=(OutputSpec("result", string),),
                    idempotent=False,
                    io_bound=True,
                )

            @classmethod
            async def execute(cls, *, action: str) -> Mapping[str, object]:
                context = current_execution_context()
                assert context is not None and context.job_ref
                path = (
                    session.run_token_directory
                    / hashlib.sha256(context.job_ref.encode()).hexdigest()
                )
                token = path.read_text() if path.exists() else "unlinked"
                await ready.put((action, token))
                await release.wait()
                if action == "fail":
                    raise ValueError("intentional local execution failure")
                return cls.outputs(result=action)

        registry = TypeRegistry()
        register_core_types(registry)
        schemas = build_schemas([Probe])
        worker = InProcessWorker(build_node_types([Probe]), registry)

        async def prepare(job_ref: str, link: Mapping[str, object] | None) -> None:
            await run_owned_thread(lambda: session.prepare_run(job_ref, link))

        async def finish(job_ref: str) -> None:
            await run_owned_thread(lambda: session.finish_run(job_ref))

        app = create_app(
            lambda event: Engine(
                schemas=schemas,
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=event,
            ),
            schemas,
            max_running_jobs=3,
            prepare_remote_run=prepare,
            finish_remote_run=finish,
        )
        add_project_routes(app, tmp_path / "library", engine_version="0.0.17")
        async with TestClient(TestServer(app)) as client:
            helper = app[PROJECTS_KEY]
            for local, remote in (("film", PROJECT), ("other", other)):
                project = (await helper.call("projects.register", id=local))["project"]
                assert isinstance(project, dict)
                await helper.call(
                    "projects.patch",
                    id=local,
                    revision=project["revision"],
                    patch={"accountProject": {"issuer": identity["base"], "id": remote}},
                )
            for action, local in (("cancel", "film"), ("fail", "other"), ("complete", "default")):
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "client",
                        "jobId": action,
                        "project": local,
                        "graph": {
                            "nodes": {
                                "probe": {
                                    "nodeType": "test.run-account",
                                    "inputs": {"action": action},
                                }
                            }
                        },
                        "targets": ["probe"],
                    },
                )
                assert response.status == 202, await response.text()
            async with asyncio.timeout(10):
                observed = dict([await ready.get() for _ in range(3)])
                assert observed == {
                    "cancel": "film-token",
                    "fail": "other-token",
                    "complete": "unlinked",
                }
                assert len(list(session.run_token_directory.iterdir())) == 2
                assert (await client.delete("/api/jobs/client/cancel")).status == 200
                release.set()
                for action, terminal in (
                    ("cancel", "cancelled"),
                    ("fail", "failed"),
                    ("complete", "completed"),
                ):
                    while True:
                        status = await (await client.get(f"/api/jobs/client/{action}")).json()
                        if status["state"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.005)
                    assert status["state"] == terminal, status
            assert not list(session.run_token_directory.iterdir())
            assert not session.token_path.exists()

    run_scenario(scenario())


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
    session.refresh_runs()
    assert not identity["requests"]
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    session.refresh_runs()
    assert not session.token_path.exists()
    session.prepare_run("running", {"issuer": identity["base"], "id": PROJECT})
    token = next(session.run_token_directory.iterdir())
    before = len(identity["requests"])
    clock[0] += 539
    session.refresh_runs()
    assert len(identity["requests"]) == before
    identity["gateway_token"] = "replacement-bearer"
    clock[0] += 2
    session.refresh_runs()
    assert len(identity["requests"]) == before + 1
    assert token.read_text() == "replacement-bearer"
    token.unlink()
    session.refresh_runs()
    assert len(identity["requests"]) == before + 2
    assert token.read_text() == "replacement-bearer"
    assert not session.token_path.exists()
    identity["lost_refresh"] = True
    clock[0] += 3600
    with pytest.raises(AccountError, match="no exchange was retried"):
        session.refresh_runs()
    after_failure = len(identity["requests"])
    assert not session.state_path.exists()
    assert not session.token_path.exists()
    assert not list(session.run_token_directory.iterdir())
    session.refresh_runs()
    assert len(identity["requests"]) == after_failure


@pytest.mark.parametrize(
    "migration", ["unique", "shared", "unknown", "other-library", "linked", "invalid"]
)
def test_engine_migrates_only_a_known_project_without_refreshing_legacy_authority(
    identity: dict, tmp_path: Path, migration: str
) -> None:
    install = tmp_path / "install"
    home = tmp_path / "user"
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=home, install_root=install)
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    identity["gateway_lifetime"] = 1
    _legacy_selection(session)
    library_root = tmp_path / "library"
    environment = "a" * 32
    install.mkdir()
    if migration != "unknown":
        (install / "environment-id").write_text(
            "invalid" if migration == "invalid" else environment
        )
    (install / "environment.json").write_text(
        json.dumps({"libraryRoot": str(library_root if migration != "other-library" else home)})
    )
    existing_link = {"issuer": identity["base"], "id": "c660f30c-c170-4312-a08a-542d6ce39e93"}
    with LibraryClient(library_root) as library:
        library.call("projects.register", id="film")
        library.call(
            "projects.patch",
            id="film",
            revision=1,
            patch={
                "environment": {"id": environment},
                **({"accountProject": existing_link} if migration == "linked" else {}),
            },
        )
        if migration == "shared":
            library.call("projects.register", id="other")
            library.call(
                "projects.patch", id="other", revision=1, patch={"environment": {"id": environment}}
            )
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
            env={
                **os.environ,
                "DINKSTER_HOME": str(home),
                "DINKSTER_ENGINE_IDENTITY": json.dumps(
                    {"identityVersion": 1, "environment": environment, "snapshot": "1"}
                ),
            },
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 45
        while True:
            assert process.poll() is None, log.read_text()
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
                    assert response.status == 200
                break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.fail("engine did not start: " + log.read_text())
                time.sleep(0.05)
        assert session.token_path.exists() == (migration == "invalid")
        with LibraryClient(library_root) as library:
            if migration in {"unknown", "other-library"}:
                (install / "environment-id").write_text(environment)
                (install / "environment.json").write_text(
                    json.dumps({"libraryRoot": str(library_root)})
                )
                session.migrate_project_link(
                    library, library_root=library_root, environment=environment
                )
            linked = cast(dict, library.call("projects.get", id="film")["project"])
            assert linked["accountProject"] == (
                {"issuer": identity["base"], "id": PROJECT}
                if migration == "unique"
                else existing_link
                if migration == "linked"
                else None
            )
            assert (
                cast(dict, library.call("projects.get", id="default")["project"])["accountProject"]
                is None
            )
            if migration == "shared":
                assert (
                    cast(dict, library.call("projects.get", id="other")["project"])[
                        "accountProject"
                    ]
                    is None
                )
                library.call("projects.delete", id="other", revision=2)
                session.migrate_project_link(
                    library, library_root=library_root, environment=environment
                )
                assert (
                    cast(dict, library.call("projects.get", id="film")["project"])["accountProject"]
                    is None
                )
        assert not list(session.run_token_directory.iterdir())
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
    if migration == "invalid":
        assert "Legacy account link unavailable" in log.read_text()
        assert "invalid environment identity" not in log.read_text()


@pytest.mark.parametrize("interruption", ["before-write", "after-write", "unlink", "delete"])
def test_legacy_link_migration_resumes_without_retargeting(
    identity, tmp_path, monkeypatch, interruption
):
    root = tmp_path / "library"
    install = tmp_path / "install"
    install.mkdir()
    environment = "b" * 32
    (install / "environment-id").write_text(environment)
    (install / "environment.json").write_text(json.dumps({"libraryRoot": str(root)}))
    session = AccountSession(
        resolve_services(identity_base=identity["base"], catalog_base=identity["base"]),
        home=tmp_path / "account",
        install_root=install,
    )
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    _legacy_selection(session)
    requests = list(identity["requests"])
    with LibraryClient(root) as library:
        library.call("projects.register", id="film")
        library.call(
            "projects.patch", id="film", revision=1, patch={"environment": {"id": environment}}
        )
        original_call = library.call

        def interrupted(operation, **arguments):
            if operation == "projects.patch":
                if interruption != "before-write":
                    original_call(operation, **arguments)
                raise OSError("interrupted migration")
            return original_call(operation, **arguments)

        with monkeypatch.context() as patch:
            patch.setattr(library, "call", interrupted)
            with pytest.raises(OSError, match="interrupted migration"):
                session.migrate_project_link(library, library_root=root, environment=environment)
        assert not session.token_path.exists()
        record = cast(dict, library.call("projects.get", id="film")["project"])
        if interruption == "unlink":
            library.call(
                "projects.patch",
                id="film",
                revision=record["revision"],
                patch={"accountProject": None},
            )
        elif interruption == "delete":
            library.call("projects.delete", id="film", revision=record["revision"])
            library.call("projects.register", id="replacement")
            library.call(
                "projects.patch",
                id="replacement",
                revision=1,
                patch={"environment": {"id": environment}},
            )
        session = AccountSession(session.services, home=tmp_path / "account", install_root=install)
        for _ in range(2):
            session.migrate_project_link(library, library_root=root, environment=environment)
        film = cast(dict, library.call("projects.get", id="film")["project"])
        if interruption in {"before-write", "after-write"}:
            assert film["accountProject"] == {"issuer": identity["base"], "id": PROJECT}
            assert film["revision"] == 3
            library.call("projects.patch", id="film", revision=3, patch={"accountProject": None})
            session.migrate_project_link(library, library_root=root, environment=environment)
            assert (
                cast(dict, library.call("projects.get", id="film")["project"])["accountProject"]
                is None
            )
        elif interruption == "unlink":
            assert film["accountProject"] is None
            assert film["revision"] == 4
        else:
            assert film["deleted"] is True
            assert (
                cast(dict, library.call("projects.get", id="replacement")["project"])[
                    "accountProject"
                ]
                is None
            )
        assert (
            cast(dict, library.call("projects.get", id="default")["project"])["accountProject"]
            is None
        )
        assert not session.token_path.exists()
        assert identity["requests"] == requests


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
        await asyncio.to_thread(
            session.prepare_run, "browser-run", {"issuer": identity["base"], "id": PROJECT}
        )
        assert list(session.run_token_directory.iterdir())
        signed_out = await client.post("/api/account/sign-out", json={})
        assert (await signed_out.json())["signedIn"] is False
        assert not session.state_path.exists()
        assert not session.token_path.exists()


def test_cli_account_link_changes_only_the_named_library_project(
    identity, tmp_path, monkeypatch, capsys
):
    home = tmp_path / "account"
    install = tmp_path / "install"
    root = tmp_path / "library"
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    services = resolve_services(identity_base=identity["base"], catalog_base=identity["base"])
    session = AccountSession(services, home=home, install_root=install)
    opener, callbacks = browser(identity)
    session.login(open_browser=opener, timeout=5)
    for thread in callbacks:
        thread.join(timeout=5)
    assert session.projects()[-1]["id"] == PROJECT
    legacy_project = "c660f30c-c170-4312-a08a-542d6ce39e93"
    _legacy_selection(session, legacy_project, "legacy-gateway-token")
    original_state = session.state_path.read_bytes()
    original_token = session.token_path.read_bytes()
    prefix = [
        "account",
        "--root",
        str(install),
        "--identity-base",
        identity["base"],
        "--catalog-base",
        identity["base"],
        "--gateway-base",
        identity["base"],
    ]
    for arguments in (("select-project", legacy_project), ("refresh",)):
        with pytest.raises(SystemExit) as removed:
            main([*prefix, *arguments])
        assert removed.value.code == 2
    for exists in (False, True):
        if exists:
            root.mkdir()
        assert (
            main(
                [
                    *prefix,
                    "link-project",
                    PROJECT,
                    "--library-root",
                    str(root),
                    "--project",
                    "default",
                    "--revision",
                    "1",
                ]
            )
            == 1
        )
        assert not root.exists() if not exists else not any(root.iterdir())
    with LibraryClient(root) as library:
        for project in ("film", "other", "deleted"):
            library.call("projects.register", id=project)
        library.call("projects.delete", id="deleted", revision=1)
        other = library.call("projects.get", id="other")

        def command(operation, *, project="film", revision=1, account=PROJECT):
            return main(
                [
                    *prefix,
                    operation,
                    "--library-root",
                    str(root),
                    "--project",
                    project,
                    "--revision",
                    str(revision),
                    *([account] if operation == "link-project" else []),
                ]
            )

        for project in ("missing", "deleted"):
            assert command("link-project", project=project) == 1
        assert library.call("projects.get", id="missing")["project"] is None
        assert command("link-project", account="not-a-uuid") == 1
        assert command("link-project", account="ffffffff-ffff-4fff-bfff-ffffffffffff") == 1
        assert command("link-project") == 0
        linked = cast(dict, library.call("projects.get", id="film")["project"])
        assert linked["accountProject"] == {"issuer": identity["base"], "id": PROJECT}
        assert linked["revision"] == 2
        assert command("unlink-project") == 1
        assert library.call("projects.get", id="film")["project"] == linked
        assert library.call("projects.get", id="other") == other
        assert session.state_path.read_bytes() == original_state
        assert session.token_path.read_bytes() == original_token
        session.logout()
        assert command("link-project", revision=2) == 1
        assert library.call("projects.get", id="film")["project"] == linked
        assert command("unlink-project", revision=2) == 0
        unlinked = cast(dict, library.call("projects.get", id="film")["project"])
        assert unlinked["accountProject"] is None
        assert unlinked["revision"] == 3
        assert library.call("projects.get", id="other") == other
        assert not session.state_path.exists()
        assert not session.token_path.exists()
    output = capsys.readouterr()
    for secret in ("oauth-first", "oauth-next", "refresh-next", "legacy-gateway-token"):
        assert secret not in output.out + output.err


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
