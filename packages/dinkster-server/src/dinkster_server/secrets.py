"""Account-scoped node credentials, outside workflows and library settings."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import stat
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from aiohttp import web
from cryptography.fernet import Fernet, InvalidToken
from dinkster_graph import Graph, RegionNode, elaborate_graph
from dinkster_schema import NodeSchema
from dinkster_schema.model import SECRET_NAME_PATTERN

from .auth import Principal, principal_for

SECRETS_KEY: web.AppKey[SecretStore] = web.AppKey("secrets")


class SecretStoreError(ValueError):
    """A credential could not be stored or read without disclosure."""


class SecretStore:
    """Store encrypted values separately from portable engine data.

    The local master key and database belong to the engine's OS account.
    Neither is part of a library, project, workflow or execution record.
    A root of None is an ephemeral store for embedded servers.
    """

    def __init__(self, root: Path | None = None, *, execution_available: bool = True) -> None:
        self.execution_available = execution_available
        self._root = root
        self._memory: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def check_name(name: str) -> None:
        if SECRET_NAME_PATTERN.fullmatch(name) is None:
            raise SecretStoreError(
                "Secret names use letters, digits, underscores, dots and hyphens"
            )

    def _private_root(self) -> Path:
        assert self._root is not None
        root = self._root
        if root.is_symlink():
            raise SecretStoreError("Cannot open private secret storage")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            if root.stat().st_uid != os.getuid():
                raise SecretStoreError("Secret storage must be owned by this user")
            root.chmod(0o700)
        else:
            literal = str(root).replace("'", "''")
            script = (
                "$ErrorActionPreference = 'Stop'; "
                "$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User; "
                "$acl = [System.Security.AccessControl.DirectorySecurity]::new(); "
                "$acl.SetOwner($sid); $acl.SetAccessRuleProtection($true, $false); "
                "$rule = [System.Security.AccessControl.FileSystemAccessRule]::new("
                "$sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow'); "
                "$acl.AddAccessRule($rule); "
                f"[System.IO.Directory]::SetAccessControl('{literal}', $acl)"
            )
            powershell = (
                Path(os.environ.get("SystemRoot", r"C:\Windows"))
                / "System32/WindowsPowerShell/v1.0/powershell.exe"
            )
            try:
                result = subprocess.run(
                    [
                        str(powershell),
                        "-NoProfile",
                        "-NonInteractive",
                        "-EncodedCommand",
                        base64.b64encode(script.encode("utf-16-le")).decode("ascii"),
                    ],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                raise SecretStoreError("Cannot protect private secret storage") from None
            if result.returncode:
                raise SecretStoreError("Cannot protect private secret storage")
        return root

    def _cipher(self) -> Fernet:
        root = self._private_root()
        key_path = root / "master.key"
        if key_path.is_symlink() or (root / "credentials.sqlite3").is_symlink():
            raise SecretStoreError("Cannot open private secret storage")
        if not key_path.exists():
            if (root / "credentials.sqlite3").exists():
                raise SecretStoreError("Secret storage key is unavailable")
            descriptor, temporary = tempfile.mkstemp(dir=root, prefix=".key-")
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(Fernet.generate_key())
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, key_path)
                except FileExistsError:
                    pass
            finally:
                Path(temporary).unlink(missing_ok=True)
        descriptor = os.open(
            key_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (os.name != "nt" and info.st_uid != os.getuid())
            ):
                raise SecretStoreError("Cannot read private secret storage")
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o600)
            return Fernet(stream.read(45))

    def _connection(self) -> sqlite3.Connection:
        assert self._root is not None
        path = self._root / "credentials.sqlite3"
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (os.name != "nt" and info.st_uid != os.getuid())
            ):
                raise SecretStoreError("Cannot open private secret storage")
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS credentials "
            "(owner TEXT NOT NULL, name TEXT NOT NULL, value BLOB NOT NULL, "
            "PRIMARY KEY (owner, name))"
        )
        return connection

    def _get(self, owner: str, name: str) -> str | None:
        if self._root is None:
            return self._memory.get((owner, name))
        cipher = self._cipher()
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT value FROM credentials WHERE owner = ? AND name = ?", (owner, name)
            ).fetchone()
            if row is None:
                return None
            raw: object = json.loads(cipher.decrypt(row[0]))
            if not isinstance(raw, list):
                raise SecretStoreError("Cannot read private secret storage")
            record = cast("list[object]", raw)
            if len(record) != 3 or record[:2] != [owner, name] or not isinstance(record[2], str):
                raise SecretStoreError("Cannot read private secret storage")
            return record[2]
        finally:
            connection.close()

    def resolve(self, principal: Principal, name: str) -> tuple[str | None, str | None, bool]:
        """Resolve only the caller's value, or a deployment value they may use.

        This host-only method is for invocation. HTTP exposes presence only.
        An unauthorized caller never reads or probes the deployment entry.
        """
        self.check_name(name)
        with self._lock:
            try:
                own = self._get("account:" + principal.principal_id, name)
                deployment = (
                    self._get("deployment", name)
                    if principal.is_operator or principal.allows("secrets:use")
                    else None
                )
                if own is not None:
                    return own, "account", deployment is not None
                return deployment, "deployment" if deployment is not None else None, False
            except (OSError, sqlite3.Error, InvalidToken, ValueError):
                raise SecretStoreError("Cannot read private secret storage") from None

    def resolve_for_execution(self, principal: Principal, name: str) -> str | None:
        value = self.resolve(principal, name)[0]
        if value is not None and not self.execution_available:
            raise SecretStoreError("This base does not support secret execution")
        return value

    def names(self, principal: Principal) -> list[str]:
        """List only entries this account may resolve, without reading values."""
        owners = ["account:" + principal.principal_id]
        if principal.is_operator or principal.allows("secrets:use"):
            owners.append("deployment")
        with self._lock:
            try:
                if self._root is None:
                    return sorted({name for owner, name in self._memory if owner in owners})
                self._cipher()
                connection = self._connection()
                try:
                    rows = connection.execute(
                        "SELECT DISTINCT name FROM credentials WHERE owner IN ("
                        + ",".join("?" for _ in owners)
                        + ") ORDER BY name",
                        owners,
                    ).fetchall()
                    return [row[0] for row in rows]
                finally:
                    connection.close()
            except (OSError, sqlite3.Error, ValueError):
                raise SecretStoreError("Cannot read private secret storage") from None

    def status(self, principal: Principal, name: str) -> dict[str, object]:
        """Presence and authorized fallback only; never return a credential to a client."""
        value, source, fallback = self.resolve(principal, name)
        return {"name": name, "set": value is not None, "source": source, "fallback": fallback}

    def preflight(
        self,
        principal: Principal,
        graph: Graph,
        schemas: Mapping[str, NodeSchema],
    ) -> list[dict[str, object]]:
        """Reject missing required credentials without exporting their values.

        Check the elaborated interface so inactive choice rows are ignored and
        selected group members and nested regions receive their own diagnostics.
        """
        problems: list[dict[str, object]] = []
        pending = [("", graph)]
        while pending:
            prefix, level = pending.pop()
            effective, _diagnostics = elaborate_graph(level, schemas)
            for node_id, node in level.nodes.items():
                path = prefix + node_id
                if isinstance(node, RegionNode):
                    pending.append((path + "/", node.body))
                    continue
                schema = effective.get(node_id)
                if schema is None:
                    continue
                for spec in schema.inputs:
                    if not spec.secret:
                        continue
                    name = node.inputs.get(spec.id, spec.default)
                    if name is None and not spec.required:
                        continue
                    invalid = name is not None and (
                        not isinstance(name, str) or SECRET_NAME_PATTERN.fullmatch(name) is None
                    )
                    if invalid:
                        problems.append(
                            {
                                "code": "invalid-secret-name",
                                "message": "Choose a secret name; secret inputs cannot be wired",
                                "nodeId": path,
                                "inputId": spec.id,
                            }
                        )
                    elif not self.execution_available and (
                        spec.required or self.status(principal, cast("str", name))["set"]
                    ):
                        problems.append(
                            {
                                "code": "secret-channel-unavailable",
                                "message": "This dependency base does not support secret execution",
                                "nodeId": path,
                                "inputId": spec.id,
                            }
                        )
                    elif spec.required and (
                        name is None or not self.status(principal, cast("str", name))["set"]
                    ):
                        problems.append(
                            {
                                "code": "secret-not-set",
                                "message": "A required secret is not set",
                                "nodeId": path,
                                "inputId": spec.id,
                                "action": "Set secret",
                                **({"name": name} if name is not None else {}),
                            }
                        )
        return problems

    def write(
        self,
        principal: Principal,
        name: str,
        value: object,
        *,
        scope: Literal["account", "deployment"] = "account",
    ) -> None:
        """Replace or remove one value; never return the stored credential."""
        self.check_name(name)
        if principal.kind != "human" or principal.delegation_id is not None:
            raise PermissionError("Only a person can change secrets")
        if scope not in ("account", "deployment"):
            raise SecretStoreError("Choose Your account or Deployment")
        if scope == "deployment" and not principal.is_operator:
            raise PermissionError("Only an administrator can change deployment secrets")
        if value is not None and (not isinstance(value, str) or not value or len(value) > 8192):
            raise SecretStoreError("Enter a secret of at most 8192 characters")
        owner = "deployment" if scope == "deployment" else "account:" + principal.principal_id
        with self._lock:
            try:
                if self._root is None:
                    if value is None:
                        self._memory.pop((owner, name), None)
                    else:
                        self._memory[owner, name] = value
                    return
                cipher = self._cipher()
                connection = self._connection()
                try:
                    with connection:
                        if value is None:
                            connection.execute(
                                "DELETE FROM credentials WHERE owner = ? AND name = ?",
                                (owner, name),
                            )
                        else:
                            encrypted = cipher.encrypt(json.dumps([owner, name, value]).encode())
                            connection.execute(
                                "INSERT INTO credentials VALUES (?, ?, ?) "
                                "ON CONFLICT(owner, name) DO UPDATE SET value = excluded.value",
                                (owner, name, encrypted),
                            )
                finally:
                    connection.close()
            except (OSError, sqlite3.Error, ValueError):
                raise SecretStoreError("Cannot save private secret storage") from None


def add_secret_routes(app: web.Application, store: SecretStore) -> None:
    app[SECRETS_KEY] = store

    async def secret(request: web.Request) -> web.Response:
        principal = principal_for(request)
        name = request.match_info["name"]
        try:
            if request.method == "PUT":
                raw: object = await request.json()
                if not isinstance(raw, dict):
                    raise SecretStoreError("Provide a value and optional scope")
                body = cast("dict[str, object]", raw)
                if set(body) - {"value", "scope"} or "value" not in body:
                    raise SecretStoreError("Provide a value and optional scope")
                scope = body.get("scope", "account")
                if scope not in ("account", "deployment"):
                    raise SecretStoreError("Choose Your account or Deployment")
                await asyncio.to_thread(
                    store.write,
                    principal,
                    name,
                    body["value"],
                    scope=scope,
                )
            elif request.method == "DELETE":
                scope = request.query.get("scope", "account")
                if scope not in ("account", "deployment"):
                    raise SecretStoreError("Choose Your account or Deployment")
                await asyncio.to_thread(store.write, principal, name, None, scope=scope)
            status = await asyncio.to_thread(store.status, principal, name)
            status["canManageDeployment"] = principal.is_operator
        except PermissionError:
            return web.json_response(
                {"error": "Secret access is not permitted"},
                status=403,
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, UnicodeError):
            return web.json_response(
                {"error": "Cannot access secret. Try again"},
                status=400,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    async def names(request: web.Request) -> web.Response:
        principal = principal_for(request)
        try:
            available = await asyncio.to_thread(store.names, principal)
        except SecretStoreError:
            return web.json_response(
                {"error": "Cannot access secret. Try again"},
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"names": available, "canManageDeployment": principal.is_operator},
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/api/secrets", names)
    app.router.add_get("/api/secrets/{name}", secret)
    app.router.add_put("/api/secrets/{name}", secret)
    app.router.add_delete("/api/secrets/{name}", secret)
