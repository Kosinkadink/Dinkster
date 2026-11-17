"""Engine-local provider secrets, separate from model and project settings."""

from __future__ import annotations

import base64
import json
import math
import os
import secrets
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from dinkster_assets.fetch import post_huggingface_form

from .storelock import StoreLockTimeout, hold_lock

HUGGINGFACE_CLIENT_ID = "ce11b06c-0297-4960-b6c3-6f3ed436c5d1"


def oauth_credential(response: dict[str, object]) -> tuple[str, float, str | None]:
    """Validate issued authority, lifetime and optional refresh without retaining ID tokens."""
    token, lifetime = response.get("access_token"), response.get("expires_in")
    refresh = response.get("refresh_token")
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 8192
        or any(not 33 <= ord(character) <= 126 for character in token)
        or type(lifetime) is not int
        or not 1 <= lifetime <= 366 * 86400
        or not isinstance(response.get("token_type"), str)
        or str(response.get("token_type")).lower() != "bearer"
        or (
            refresh is not None
            and (
                not isinstance(refresh, str)
                or not refresh
                or len(refresh) > 8192
                or any(not 33 <= ord(character) <= 126 for character in refresh)
            )
        )
    ):
        raise ProviderCredentialsError("Invalid Hugging Face sign-in response")
    return token, time.time() + lifetime, cast("str | None", refresh)


class ProviderCredentialsError(ValueError):
    """Provider credentials cannot be read or changed safely."""


class ProviderCredentialsExpired(ProviderCredentialsError):
    """The provider's issued lifetime ended; a new sign-in is required."""

    def __init__(self, message: str, *, provider: str = "huggingface") -> None:
        super().__init__(message)
        self.provider = provider


def _private_windows_file(path: Path) -> None:
    # The staging file is still empty when its inherited ACL is removed.
    literal = str(path).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User; "
        "$acl = [System.Security.AccessControl.FileSecurity]::new(); "
        "$acl.SetOwner($sid); $acl.SetAccessRuleProtection($true, $false); "
        "$rule = [System.Security.AccessControl.FileSystemAccessRule]::new("
        "$sid, 'FullControl', 'Allow'); $acl.AddAccessRule($rule); "
        # Persist only the owner/DACL sections modified above, never the SACL.
        f"[System.IO.File]::SetAccessControl('{literal}', $acl)"
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
        raise ProviderCredentialsError("Cannot protect provider credentials") from None
    if result.returncode:
        raise ProviderCredentialsError("Cannot protect provider credentials")


class ProviderCredentials:
    """Serialize updates without putting secrets in a portable settings document."""

    def __init__(self, home: Path | None = None) -> None:
        root = home or Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster"))
        self.path = root.expanduser() / "credentials.json"
        self._revocation_path = self.path.with_name("credentials-revoked.json")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with hold_lock(self.path.with_suffix(".lock"), timeout=5):
                yield
        except StoreLockTimeout:
            raise ProviderCredentialsError("Provider credentials are busy; try again") from None

    def _read(self) -> dict[str, object]:
        if self.path.is_symlink() or self._revocation_path.is_symlink():
            raise ProviderCredentialsError("Provider credentials must be a private regular file")
        # A durable revocation wins even if replacement of the bearer file failed.
        if not self._revocation_path.exists():
            return self._read_file(self.path)
        revoked = self._read_file(self._revocation_path)
        names = revoked.get("revoked")
        if names is None:
            return revoked
        if (
            not isinstance(names, list)
            or not names
            or any(
                not isinstance(name, str) or name not in {"huggingface", "civitai"}
                for name in names
            )
        ):
            raise ProviderCredentialsError("Invalid provider credential revocation")
        state = self._read_file(self.path)
        providers = cast("dict[str, object]", state["providers"])
        revisions = cast("dict[str, object]", state.setdefault("revisions", {}))
        if not isinstance(revisions, dict):
            raise ProviderCredentialsError("Invalid provider credential revisions")
        for name in names:
            providers.pop(name, None)
            revisions[name] = self._revision(revoked, name)
        return state

    def _read_file(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {"version": 1, "providers": {}}
        try:
            if os.name == "nt":
                _private_windows_file(path)
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or (os.name != "nt" and info.st_uid != os.getuid())
                ):
                    raise ProviderCredentialsError(
                        "Provider credentials must be owned by this user"
                    )
                if os.name != "nt":
                    os.fchmod(stream.fileno(), 0o600)
                raw: object = json.loads(stream.read(64 * 1024 + 1))
            if (
                not isinstance(raw, dict)
                or raw.get("version") != 1
                or not isinstance(raw.get("providers"), dict)
                or (path == self._revocation_path and raw["providers"] != {})
            ):
                raise ValueError()
            return cast("dict[str, object]", raw)
        except (OSError, ValueError):
            raise ProviderCredentialsError("Cannot read private provider credentials") from None

    def token(self, provider: str = "huggingface") -> str | None:
        """Return the selected provider bearer only to the engine's provider transport."""
        return self.snapshot(provider)[0]

    def _revision(self, state: dict[str, object], provider: str) -> str:
        if provider not in {"huggingface", "civitai"}:
            raise ProviderCredentialsError("Unsupported credential provider")
        revisions = state.get("revisions", {})
        if not isinstance(revisions, dict):
            raise ProviderCredentialsError("Invalid provider credential revisions")
        legacy = state.get("revision", "initial") if provider == "huggingface" else "initial"
        revision = revisions.get(provider, legacy)
        if not isinstance(revision, str) or not revision:
            raise ProviderCredentialsError("Invalid provider credential revision")
        return revision

    def snapshot(self, provider: str = "huggingface") -> tuple[str | None, str]:
        """Read bearer and mutation identity together, including revoked credentials."""
        with self._locked():
            state = self._read()
            revision = self._revision(state, provider)
            providers = cast("dict[str, object]", state["providers"])
            entry = providers.get(provider)
            if entry is None:
                return None, revision
            if not isinstance(entry, dict):
                raise ProviderCredentialsError("Invalid provider credentials; sign in again")
            token = entry.get("access_token")
            if (
                not isinstance(token, str)
                or not token
                or len(token) > 8192
                or any(not 33 <= ord(character) <= 126 for character in token)
            ):
                raise ProviderCredentialsError("Invalid provider credentials; sign in again")
            expires = entry.get("expires_at")
            if expires is None:
                return token, revision
            if type(expires) not in {int, float} or not math.isfinite(cast("float", expires)):
                raise ProviderCredentialsError("Invalid provider credential expiry")
            if cast("float", expires) > time.time():
                return token, revision
            refresh = entry.get("refresh_token")
            if provider != "huggingface" or not isinstance(refresh, str) or not refresh:
                raise ProviderCredentialsExpired("Provider sign-in expired", provider=provider)
            # Consume the attempt before I/O; revocation must not wait on the provider.
            entry.pop("refresh_token")
            revision = secrets.token_urlsafe(32)
            revisions = cast("dict[str, object]", state.setdefault("revisions", {}))
            revisions[provider] = revision
            self._write(state)
        response = post_huggingface_form(
            "token",
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": HUGGINGFACE_CLIENT_ID,
            },
        )
        if response.get("error"):
            raise ProviderCredentialsExpired("Hugging Face sign-in expired")
        token, expires_at, rotated = oauth_credential(response)
        revision = self.set_token(
            token,
            expires_at=expires_at,
            refresh_token=rotated or refresh,
            expected_revision=revision,
        )
        return token, revision

    def revision(self, provider: str = "huggingface") -> str:
        """Read mutation identity even when the stored bearer has expired."""
        with self._locked():
            return self._revision(self._read(), provider)

    def set_token(
        self,
        token: str | None,
        *,
        provider: str = "huggingface",
        expires_at: float | None = None,
        expected_revision: str | None = None,
        refresh_token: str | None = None,
    ) -> str:
        """Store a validated provider bearer, or forget it without touching other providers."""
        if token is not None and (
            not token
            or len(token) > 8192
            or any(not 33 <= ord(character) <= 126 for character in token)
        ):
            raise ProviderCredentialsError("Invalid provider credential")
        if expires_at is not None and (not math.isfinite(expires_at) or expires_at <= time.time()):
            raise ProviderCredentialsError("Invalid provider credential expiry")
        with self._locked():
            state = self._read()
            current = self._revision(state, provider)
            if expected_revision is not None and current != expected_revision:
                raise ProviderCredentialsError("Provider credentials changed; sign in again")
            revision = secrets.token_urlsafe(32)
            revisions = cast("dict[str, object]", state.setdefault("revisions", {}))
            revisions[provider] = revision
            providers = cast("dict[str, object]", state["providers"])
            if token is None:
                providers.pop(provider, None)
            else:
                providers[provider] = {
                    "access_token": token,
                    "method": "oauth" if expires_at is not None else "token",
                    **({"expires_at": expires_at} if expires_at is not None else {}),
                    **({"refresh_token": refresh_token} if refresh_token is not None else {}),
                }
            try:
                if token is None:
                    # Persist revocation before replacing a file that may still hold authority.
                    self._write(
                        {
                            "version": 1,
                            "providers": {},
                            "revoked": [
                                name for name in ("huggingface", "civitai") if name not in providers
                            ],
                            "revisions": {
                                name: self._revision(state, name)
                                for name in ("huggingface", "civitai")
                            },
                        },
                        self._revocation_path,
                    )
                if not providers:
                    self.path.unlink(missing_ok=True)
                else:
                    self._write(state)
                    self._revocation_path.unlink(missing_ok=True)
            except (OSError, ProviderCredentialsError):
                action = "forget" if token is None else "save"
                raise ProviderCredentialsError(
                    f"Cannot {action} private provider credentials"
                ) from None
            return revision

    def _write(self, state: dict[str, object], path: Path | None = None) -> None:
        path = path or self.path
        descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                if os.name == "nt":
                    _private_windows_file(Path(temporary))
                else:
                    os.fchmod(stream.fileno(), 0o600)
                json.dump(state, stream, separators=(",", ":"), ensure_ascii=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError:
            raise ProviderCredentialsError("Cannot save private provider credentials") from None
        finally:
            Path(temporary).unlink(missing_ok=True)
