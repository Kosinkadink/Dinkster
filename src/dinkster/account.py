"""Native account consent and selected-project gateway credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

from dinkster_library import LibraryClient, LibraryError

from .services import Services, resolve_services
from .setup import default_project
from .storelock import hold_lock

_SCOPE = "openid profile email projects:gateway offline_access"


class AccountError(ValueError):
    """Sign-in or current project authority is unavailable."""


class AccountSignIn:
    """Fence cancellation against the native sign-in's final session write."""

    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.lock = threading.Lock()

    def cancel(self) -> None:
        with self.lock:
            self.cancelled.set()

    def check(self) -> None:
        if self.cancelled.is_set():
            raise AccountError("Account sign-in cancelled")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _private_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _text(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise AccountError(f"Identity response requires {key}")
    return value


def _expires(document: Mapping[str, object]) -> float:
    value = document.get("expires_in")
    if type(value) is not int or value < 1:
        raise AccountError("Identity response requires a positive expires_in")
    if _text(document, "token_type").lower() != "bearer":
        raise AccountError("Identity did not issue a bearer token")
    return time.time() + value


class AccountSession:
    """Own OAuth secrets outside projects; expose only a project gateway-token file.

    The endpoint/audience key prevents a custom or QA service from reusing a
    production credential. Refresh rotation and project minting serialize across
    the CLI and engine processes, with no automatic retry of ambiguous exchanges.
    """

    def __init__(
        self,
        services: Services,
        *,
        home: Path | None = None,
        install_root: Path | None = None,
    ) -> None:
        if not services.identity_base or not services.gateway_base:
            raise AccountError("Account sign-in requires identity and gateway addresses")
        self.services = services
        self.install_root = (install_root or default_project()).expanduser().resolve()
        root = home or Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster"))
        key = hashlib.sha256(
            json.dumps(
                [
                    str(self.install_root),
                    services.identity_base,
                    services.catalog_base,
                    services.gateway_base,
                    services.gateway_audience,
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.state_path = root.expanduser() / "accounts" / key / "session.json"
        self.token_path = root.expanduser() / "gateway-tokens" / key / "access-token"
        self.lock_path = self.state_path.with_suffix(".lock")
        self.run_token_directory = self.token_path.parent / "runs"
        self._runs: dict[str, tuple[str, str, float] | None] = {}

    def _request(
        self,
        path: str,
        *,
        token: str | None = None,
        document: Mapping[str, object] | None = None,
        form: Mapping[str, str] | None = None,
    ) -> object:
        headers = {"Accept": "application/json"}
        data = None
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if document is not None:
            data = json.dumps(document).encode()
            headers["Content-Type"] = "application/json"
        elif form is not None:
            data = urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = Request(self.services.identity_base + path, data=data, headers=headers)
        try:
            with build_opener(_NoRedirect()).open(request, timeout=15) as response:
                raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise AccountError("Identity response exceeds the account response limit")
                return json.loads(raw)
        except HTTPError as exc:
            # The body or redirect location may contain credentials. Never log either.
            exc.close()
            raise AccountError(f"Identity refused the account request (HTTP {exc.code})") from None
        except (OSError, URLError, ValueError) as exc:
            if isinstance(exc, AccountError):
                raise
            raise AccountError("Identity account request failed; no exchange was retried") from None

    def _object(
        self,
        path: str,
        *,
        token: str | None = None,
        document: Mapping[str, object] | None = None,
        form: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        value = self._request(path, token=token, document=document, form=form)
        if not isinstance(value, dict):
            raise AccountError("Identity response must be an object")
        return cast("dict[str, object]", value)

    def _read(self) -> dict[str, object]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise AccountError("Sign in to use remote nodes") from None
        if not isinstance(value, dict) or value.get("format") != "dinkster.account/1":
            raise AccountError("Account session is invalid; sign in again")
        return cast("dict[str, object]", value)

    def _save(self, state: Mapping[str, object]) -> None:
        _private_write(self.state_path, json.dumps(state, separators=(",", ":")))

    def _oauth(self, state: dict[str, object]) -> str:
        expiry = state.get("expires_at")
        if not isinstance(expiry, (int, float)) or not math.isfinite(expiry):
            raise AccountError("Account session expiry is invalid; sign in again")
        if expiry > time.time() + 60:
            return _text(state, "access_token")
        try:
            result = self._object(
                "/token",
                form={
                    "grant_type": "refresh_token",
                    "client_id": _text(state, "client_id"),
                    "refresh_token": _text(state, "refresh_token"),
                },
            )
            self._accept_oauth(state, result)
            self._save(state)
        except (AccountError, OSError):
            # A lost reply may already have rotated the refresh family. Never replay it.
            self.token_path.unlink(missing_ok=True)
            self._clear_run_tokens()
            self.state_path.unlink(missing_ok=True)
            raise
        return _text(state, "access_token")

    def _accept_oauth(self, state: dict[str, object], result: Mapping[str, object]) -> None:
        if not {"projects:gateway", "offline_access"}.issubset(_text(result, "scope").split()):
            raise AccountError("Identity consent lacks projects:gateway or offline_access")
        state.update(
            access_token=_text(result, "access_token"),
            refresh_token=_text(result, "refresh_token"),
            expires_at=_expires(result),
            scope=_text(result, "scope"),
        )

    def status(self) -> dict[str, object]:
        with hold_lock(self.lock_path, timeout=20):
            if not self.state_path.exists():
                return {"signedIn": False, "profile": None}
            state = self._read()
            token = self._oauth(state)
            if not {"openid", "profile", "email"}.issubset(str(state.get("scope", "")).split()):
                return {"signedIn": True, "profile": None}
            document = self._object("/userinfo", token=token)
            profile: dict[str, object] = {}
            for source, target in (
                ("name", "displayName"),
                ("email", "email"),
            ):
                value = document.get(source)
                profile[target] = (
                    value.strip() or None if isinstance(value, str) and len(value) <= 2048 else None
                )
            return {"signedIn": True, "profile": profile}

    def login(
        self,
        *,
        open_browser: Callable[[str], object] = webbrowser.open,
        timeout: float = 180,
        client_id: str | None = None,
        sign_in: AccountSignIn | None = None,
    ) -> None:
        attempt = sign_in or AccountSignIn()
        attempt.check()
        if not math.isfinite(timeout) or timeout <= 0:
            raise AccountError("Sign-in timeout must be positive and finite")
        state_value = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(
            b"="
        )
        received: dict[str, str] = {}
        issuer = self.services.identity_base

        class Callback(BaseHTTPRequestHandler):
            def setup(self) -> None:
                self.request.settimeout(min(1.0, timeout))
                super().setup()

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)
                denied = len(query.get("error", [])) == 1 and "code" not in query
                valid = (
                    parsed.path == "/callback"
                    and query.get("state") == [state_value]
                    and query.get("iss") == [issuer]
                    and (
                        denied
                        or (
                            len(query.get("code", [])) == 1
                            and bool(query["code"][0])
                            and "error" not in query
                        )
                    )
                )
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b"Return to Dinkster." if valid else b"Invalid sign-in callback.")
                if valid:
                    received["error" if denied else "code"] = query["error" if denied else "code"][
                        0
                    ]

        with HTTPServer(("127.0.0.1", 0), Callback) as server:
            server.timeout = min(1.0, timeout)
            redirect = f"http://127.0.0.1:{server.server_port}/callback"
            with hold_lock(self.lock_path, timeout=20):
                if client_id is None:
                    registered = self._object(
                        "/register",
                        document={
                            "client_name": "Dinkster install",
                            "redirect_uris": [redirect],
                            "scope": _SCOPE,
                            "token_endpoint_auth_method": "none",
                            "grant_types": ["authorization_code", "refresh_token"],
                            "response_types": ["code"],
                        },
                    )
                    client_id = _text(registered, "client_id")
                parameters = urlencode(
                    {
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": redirect,
                        "scope": _SCOPE,
                        "state": state_value,
                        "nonce": secrets.token_urlsafe(32),
                        "code_challenge": challenge.decode(),
                        "code_challenge_method": "S256",
                        "resource": issuer,
                    }
                )
                attempt.check()
                open_browser(issuer + "/authorize?" + parameters)
                deadline = time.monotonic() + timeout
                while not received and time.monotonic() < deadline:
                    attempt.check()
                    server.handle_request()
                attempt.check()
                if not received:
                    raise AccountError("Account sign-in timed out")
                if "error" in received:
                    raise AccountError("Account consent was refused")
                result = self._object(
                    "/token",
                    form={
                        "grant_type": "authorization_code",
                        "client_id": client_id,
                        "code": received["code"],
                        "redirect_uri": redirect,
                        "code_verifier": verifier,
                    },
                )
                state: dict[str, object] = {
                    "format": "dinkster.account/1",
                    "client_id": client_id,
                    "session_id": secrets.token_hex(32),
                }
                self._accept_oauth(state, result)
                with attempt.lock:
                    attempt.check()
                    self.token_path.unlink(missing_ok=True)
                    self._clear_run_tokens()
                    self._save(state)

    def projects(self) -> list[dict[str, object]]:
        with hold_lock(self.lock_path, timeout=20):
            state = self._read()
            token = self._oauth(state)
            projects: list[dict[str, object]] = []
            offset = 0
            while True:
                rows = self._request(f"/v1/oauth/projects?limit=100&offset={offset}", token=token)
                if (
                    not isinstance(rows, list)
                    or len(rows) > 100
                    or not all(isinstance(row, dict) for row in rows)
                ):
                    raise AccountError("Identity project list is invalid")
                projects.extend(cast("list[dict[str, object]]", rows))
                if len(rows) < 100:
                    return projects
                offset += len(rows)

    def project_link(self, project_id: str) -> dict[str, str] | None:
        project = str(UUID(project_id))
        if not any(str(UUID(str(row.get("id", "")))) == project for row in self.projects()):
            return None
        return {"issuer": self.services.identity_base, "id": project}

    def migrate_project_link(
        self, library: LibraryClient, *, library_root: Path, environment: str
    ) -> None:
        """Carry legacy selection only to its uniquely recorded local project."""
        from dinkster_supervisor.environment_state import (
            library_root as configured_library,
        )
        from dinkster_supervisor.environment_state import (
            read_environment_identity,
        )

        with hold_lock(self.lock_path, timeout=20):
            if not self.state_path.exists():
                return
            state = self._read()
            if "project_id" not in state and "legacy_project_link" not in state:
                return
            known = (
                read_environment_identity(self.install_root) == environment
                and Path(configured_library(self.install_root)).resolve() == library_root.resolve()
            )
            if "project_id" in state:
                account_project = str(UUID(_text(state, "project_id")))
                target = None
                if known:
                    page = library.call("projects.list", environment=environment, limit=2)
                    projects = cast("list[dict[str, object]]", page["items"])
                    if len(projects) == 1 and page["cursor"] is None:
                        target = projects[0]
                # Reserve the target revision before the cross-store write. A restart
                # must not choose a different project or undo a later unlink.
                state["legacy_project_link"] = {
                    "library": str(library_root.resolve()),
                    "environment": environment,
                    "project": target["id"] if target else None,
                    "revision": target["revision"] if target else None,
                    "id": account_project,
                }
                state.pop("project_id")
                state.pop("gateway_expires_at", None)
                self._save(state)
            self.token_path.unlink(missing_ok=True)
            pending = state["legacy_project_link"]
            if not isinstance(pending, dict):
                raise AccountError("Invalid legacy project link")
            pending = cast("dict[str, object]", pending)
            if (
                not known
                or pending.get("library") != str(library_root.resolve())
                or pending.get("environment") != environment
                or pending.get("project") is None
            ):
                return
            project = cast(
                "dict[str, object] | None",
                library.call("projects.get", id=pending["project"])["project"],
            )
            link = {"issuer": self.services.identity_base, "id": str(UUID(_text(pending, "id")))}
            if (
                project is None
                or project["deleted"]
                or not isinstance(project["environment"], dict)
                or project["environment"].get("id") != environment
            ):
                return
            if project["accountProject"] != link:
                if project["revision"] != pending.get("revision") or project["accountProject"]:
                    return
                library.call(
                    "projects.patch",
                    id=project["id"],
                    revision=project["revision"],
                    patch={"accountProject": link},
                )
            state.pop("legacy_project_link")
            self._save(state)

    def select_project(self, project_id: str) -> None:
        project = str(UUID(project_id))
        with hold_lock(self.lock_path, timeout=20):
            state = self._read()
            # A project change must not leave a bearer for the previous project usable.
            self.token_path.unlink(missing_ok=True)
            state.pop("project_id", None)
            self._save(state)
            self._mint(state, project)

    def _gateway(self, state: dict[str, object], project: str) -> tuple[str, float]:
        token = self._oauth(state)
        result = self._object(
            f"/v1/oauth/projects/{project}/gateway-token", token=token, document={}
        )
        return _text(result, "access_token"), _expires(result)

    def _mint(self, state: dict[str, object], project: str) -> None:
        gateway, expiry = self._gateway(state, project)
        state.update(project_id=project, gateway_expires_at=expiry)
        self._save(state)
        _private_write(self.token_path, gateway)

    def _run_token_path(self, job_ref: str) -> Path:
        return self.run_token_directory / hashlib.sha256(job_ref.encode()).hexdigest()

    def prepare_run(self, job_ref: str, link: Mapping[str, object] | None) -> None:
        """Pin a run to its library link, never the install-wide selection."""
        with hold_lock(self.lock_path, timeout=20):
            if job_ref in self._runs:
                raise AccountError("Run account project is already selected")
            self._runs[job_ref] = None
            path = self._run_token_path(job_ref)
            path.unlink(missing_ok=True)
            if link is None:
                return
            if set(link) != {"issuer", "id"} or link["issuer"] != self.services.identity_base:
                raise AccountError("Project account issuer does not match the configured service")
            project = str(UUID(_text(link, "id")))
            state = self._read()
            session = _text(state, "session_id" if "session_id" in state else "client_id")
            gateway, expiry = self._gateway(state, project)
            _private_write(path, gateway)
            self._runs[job_ref] = (project, session, expiry)

    def refresh_runs(self) -> None:
        with hold_lock(self.lock_path, timeout=20):
            if not any(self._runs.values()):
                return
            try:
                state = self._read()
                session = _text(state, "session_id" if "session_id" in state else "client_id")
            except (ValueError, OSError):
                self._clear_run_tokens()
                raise
            failed = False
            for job_ref, binding in tuple(self._runs.items()):
                if binding is None:
                    continue
                project, owner, expiry = binding
                path = self._run_token_path(job_ref)
                if owner != session:
                    path.unlink(missing_ok=True)
                    self._runs[job_ref] = None
                    continue
                if expiry > time.time() + 60 and path.is_file():
                    continue
                try:
                    gateway, expiry = self._gateway(state, project)
                    _private_write(path, gateway)
                    self._runs[job_ref] = (project, owner, expiry)
                except (ValueError, OSError):
                    path.unlink(missing_ok=True)
                    self._runs[job_ref] = None
                    if not self.state_path.exists():
                        self._clear_run_tokens()
                        raise
                    failed = True
            if failed:
                raise AccountError("Remote account project credentials are unavailable")

    def finish_run(self, job_ref: str) -> None:
        with hold_lock(self.lock_path, timeout=20):
            self._run_token_path(job_ref).unlink(missing_ok=True)
            self._runs.pop(job_ref, None)

    def _clear_run_tokens(self) -> None:
        for path in self.run_token_directory.glob("*"):
            if len(path.name) == 64 and all(
                character in "0123456789abcdef" for character in path.name
            ):
                path.unlink(missing_ok=True)
        self._runs = dict.fromkeys(self._runs)

    def refresh_gateway(self) -> None:
        with hold_lock(self.lock_path, timeout=20):
            state = self._read()
            project = str(UUID(_text(state, "project_id")))
            try:
                self._mint(state, project)
            except (AccountError, OSError):
                self.token_path.unlink(missing_ok=True)
                raise

    def ensure_gateway(self) -> None:
        """Keep a selected project's bearer fresh without opening browser consent."""
        with hold_lock(self.lock_path, timeout=20):
            self.token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name != "nt":
                self.token_path.parent.chmod(0o700)
            if not self.state_path.exists():
                self.token_path.unlink(missing_ok=True)
                return
            try:
                state = self._read()
                if "project_id" not in state:
                    self.token_path.unlink(missing_ok=True)
                    return
                project = str(UUID(_text(state, "project_id")))
                expiry = state.get("gateway_expires_at")
                if (
                    isinstance(expiry, (int, float))
                    and math.isfinite(expiry)
                    and expiry > time.time() + 60
                    and self.token_path.is_file()
                ):
                    return
                self._mint(state, project)
            except (ValueError, OSError):
                self.token_path.unlink(missing_ok=True)
                raise

    def logout(self) -> None:
        with hold_lock(self.lock_path, timeout=20):
            self.token_path.unlink(missing_ok=True)
            self._clear_run_tokens()
            self.state_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign an install into Dinkster remote services")
    parser.add_argument("--root", type=Path, default=default_project(), help="managed install root")
    parser.add_argument("--services", default=os.environ.get("DINKSTER_SERVICES", "production"))
    parser.add_argument("--identity-base", default=os.environ.get("DINKSTER_ACCOUNT_IDENTITY_BASE"))
    parser.add_argument("--gateway-base", default=os.environ.get("DINKSTER_REMOTE_GATEWAY_BASE"))
    parser.add_argument("--catalog-base", default=os.environ.get("DINKSTER_REMOTE_CATALOG_BASE"))
    commands = parser.add_subparsers(dest="operation", required=True)
    login = commands.add_parser("login")
    login.add_argument("--client-id")
    login.add_argument("--no-browser", action="store_true")
    commands.add_parser("projects")
    select = commands.add_parser("select-project")
    select.add_argument("project_id")
    for operation in ("link-project", "unlink-project"):
        link = commands.add_parser(operation)
        link.add_argument("--library-root", type=Path, required=True)
        link.add_argument("--project", required=True, help="existing library project ID")
        link.add_argument(
            "--revision", type=int, required=True, help="current library project revision"
        )
        if operation == "link-project":
            link.add_argument("account_project_id")
    commands.add_parser("refresh")
    commands.add_parser("logout")
    args = parser.parse_args(argv)
    try:
        session = AccountSession(
            resolve_services(
                args.services,
                identity_base=args.identity_base,
                catalog_base=args.catalog_base,
                gateway_base=args.gateway_base,
            ),
            install_root=args.root,
        )
        if args.operation == "login":
            opener = print if args.no_browser else webbrowser.open
            session.login(open_browser=opener, client_id=args.client_id)
            print("Signed in. List projects, then select one for remote nodes.")
        elif args.operation == "projects":
            for project in session.projects():
                print(f"{project['id']}  {project['display_name']}")
        elif args.operation in {"link-project", "unlink-project"}:
            if not (args.library_root / "records.sqlite").is_file():
                raise AccountError("Project library is unavailable")
            with LibraryClient(args.library_root) as library:
                project = cast(
                    "dict[str, object] | None",
                    library.call("projects.get", id=args.project)["project"],
                )
                if project is None or project["deleted"]:
                    raise AccountError("Library project is unavailable")
                if project["revision"] != args.revision:
                    raise AccountError("Library project changed; read it again before saving")
                account_link = None
                if args.operation == "link-project":
                    account_link = session.project_link(args.account_project_id)
                    if account_link is None:
                        raise AccountError("Account project is unavailable")
                library.call(
                    "projects.patch",
                    id=args.project,
                    revision=args.revision,
                    patch={"accountProject": account_link},
                )
            print("Project account link saved.")
        elif args.operation == "select-project":
            session.select_project(args.project_id)
            print("Remote project selected.")
        elif args.operation == "refresh":
            session.refresh_gateway()
            print("Remote project token refreshed.")
        else:
            session.logout()
            print("Signed out on this install.")
    except (ValueError, OSError, LibraryError) as exc:
        print(f"account: {exc}", file=sys.stderr)
        return 1
    return 0
