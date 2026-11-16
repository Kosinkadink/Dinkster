"""Engine-local provider secrets, separate from model and project settings."""

from __future__ import annotations

import base64
import json
import os
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from .storelock import hold_lock


class ProviderCredentialsError(ValueError):
    """Provider credentials cannot be read or changed safely."""


def _private_windows_file(path: Path) -> None:
    # The staging file is still empty when its inherited ACL is removed.
    literal = str(path).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User; "
        "$acl = New-Object System.Security.AccessControl.FileSecurity; "
        "$acl.SetOwner($sid); $acl.SetAccessRuleProtection($true, $false); "
        "$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("
        "$sid, 'FullControl', 'Allow'); $acl.AddAccessRule($rule); "
        f"Set-Acl -LiteralPath '{literal}' -AclObject $acl"
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

    def _read(self) -> dict[str, object]:
        if self.path.is_symlink():
            raise ProviderCredentialsError("Provider credentials must be a private regular file")
        if not self.path.exists():
            return {"version": 1, "providers": {}}
        try:
            if os.name == "nt":
                _private_windows_file(self.path)
            descriptor = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
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
            ):
                raise ValueError()
            return cast("dict[str, object]", raw)
        except (OSError, ValueError):
            raise ProviderCredentialsError("Cannot read private provider credentials") from None

    def token(self) -> str | None:
        """Return the current HF bearer only to the engine's provider transport."""
        return self.snapshot()[0]

    def snapshot(self) -> tuple[str | None, str]:
        """Read bearer and mutation identity together, including revoked credentials."""
        with hold_lock(self.path.with_suffix(".lock"), timeout=5):
            state = self._read()
            revision = state.get("revision", "initial")
            if not isinstance(revision, str) or not revision:
                raise ProviderCredentialsError("Invalid provider credential revision")
            providers = cast("dict[str, object]", state["providers"])
            entry = providers.get("huggingface")
            if entry is None:
                return None, revision
            if not isinstance(entry, dict):
                raise ProviderCredentialsError("Invalid Hugging Face credentials; sign in again")
            token = entry.get("access_token")
            if (
                not isinstance(token, str)
                or not token
                or len(token) > 8192
                or any(not 33 <= ord(character) <= 126 for character in token)
            ):
                raise ProviderCredentialsError("Invalid Hugging Face credentials; sign in again")
            return token, revision

    def set_token(self, token: str | None) -> None:
        """Store a validated provider bearer, or forget it without touching other providers."""
        if token is not None and (
            not token
            or len(token) > 8192
            or any(not 33 <= ord(character) <= 126 for character in token)
        ):
            raise ProviderCredentialsError("Invalid Hugging Face credential")
        with hold_lock(self.path.with_suffix(".lock"), timeout=5):
            state = self._read()
            state["revision"] = secrets.token_urlsafe(32)
            providers = cast("dict[str, object]", state["providers"])
            if token is None:
                providers.pop("huggingface", None)
            else:
                providers["huggingface"] = {"access_token": token, "method": "token"}
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
                os.replace(temporary, self.path)
            except OSError:
                raise ProviderCredentialsError("Cannot save private provider credentials") from None
            finally:
                Path(temporary).unlink(missing_ok=True)
