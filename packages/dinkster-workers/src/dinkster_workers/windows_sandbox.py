"""Windows 11 ProcessContainer launcher. A requested sandbox never falls back."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ._pipe_transport import bridge_process
from .interpreter import module_command
from .launch import HOST_OWNED_ENVIRONMENT, LaunchSpec
from .sandbox import (
    SandboxError,
    SandboxPolicy,
    SandboxUnavailable,
    _interpreter_mounts,
    _validate_protected_binds,
)

MXC_COMMIT = "c66201839d63cae2239db5e92f1f1d6076cf5181"
MXC_EXECUTABLE_ENV = "DINKSTER_MXC_EXECUTABLE"
_WSL_HELP = "Use the Desktop WSL2 option for the Linux pack sandbox."


@dataclass(frozen=True)
class WindowsSandboxCapability:
    available: bool
    executable: str | None
    detail: str


def detect_windows_sandbox() -> WindowsSandboxCapability:
    if sys.platform != "win32":
        return WindowsSandboxCapability(False, None, "native Windows sandbox requires Windows 11")
    selected = os.environ.get(MXC_EXECUTABLE_ENV, "")
    executable = Path(selected)
    if not selected or not executable.is_absolute() or not executable.is_file():
        return WindowsSandboxCapability(
            False,
            None,
            f"Dinkster's source-built MXC executor is missing; repair Desktop. {_WSL_HELP}",
        )
    try:
        metadata = json.loads(executable.with_name("mxc-build.json").read_text(encoding="utf-8"))
        digest = hashlib.file_digest(executable.open("rb"), "sha256").hexdigest()
        if (
            metadata.get("format") != "dinkster.mxc/1"
            or metadata.get("commit") != MXC_COMMIT
            or metadata.get("telemetryComponent") != ""
            or metadata.get("sha256") != digest
        ):
            raise ValueError("executor does not match the pinned unrouted build")
        result = subprocess.run(
            [str(executable), "--probe"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        probe = json.loads(result.stdout)
        if (
            result.returncode != 0
            or probe.get("tier") != "base-container"
            or probe.get("probes", {}).get("baseContainerApiPresent") is not True
            or probe.get("needsDaclAugmentation") is not False
        ):
            raise ValueError("Windows process isolation is unavailable")
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return WindowsSandboxCapability(
            False,
            None,
            f"Windows native sandbox unavailable: {exc}. "
            f"Windows 11 24H2/25H2 with the August 2026 update or newer is required. {_WSL_HELP}",
        )
    return WindowsSandboxCapability(True, str(executable), "Windows process isolation is ready")


def build_windows_policy(
    spec: LaunchSpec, policy: SandboxPolicy
) -> tuple[dict[str, object], list[str]]:
    if policy.egress_allowlist:
        raise SandboxError(
            "Windows sandbox refuses network grants: MXC proxy mode permits DNS lookups "
            "outside the origin allowlist (MXC issue 1666). Use the Desktop WSL2 option."
        )
    if spec.use_shm:
        raise SandboxError("Windows sandbox transports values over pipes, not named shared memory")
    command = list(spec.command)
    endpoints: list[str] = []
    for index, argument in enumerate(command[:-1]):
        if argument == "--endpoint":
            endpoints.append(command[index + 1])
            command[index + 1] = f"pipe:{len(endpoints) - 1}"
    if not endpoints:
        raise SandboxError("Windows pack sandbox requires a worker boundary endpoint")
    interpreter_roots, _ = _interpreter_mounts(spec.python)
    readonly = [*interpreter_roots, str(spec.pack_root), *policy.ro_binds]
    writable = [str(spec.endpoint_dir), *policy.rw_binds]
    for path in (*readonly, *writable):
        resolved = Path(path).resolve()
        if not Path(path).is_absolute() or resolved == Path(resolved.anchor):
            raise SandboxError(f"sandbox refuses broad or relative path grant {path!r}")
    _validate_protected_binds(
        readonly,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_ro_exceptions,
        role="read-only",
    )
    _validate_protected_binds(
        writable,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_rw_exceptions,
        role="writable",
    )
    systemroot = os.environ.get("SYSTEMROOT", "C:\\Windows")
    localappdata = os.environ.get("LOCALAPPDATA")
    if not localappdata:
        raise SandboxUnavailable("Windows sandbox requires LOCALAPPDATA for the container profile")
    environment = {
        "SYSTEMROOT": systemroot,
        "LOCALAPPDATA": localappdata,
        "PATH": str(Path(systemroot) / "System32"),
        "PYTHONNOUSERSITE": "1",
    }
    for name in policy.env_passthrough:
        if name not in HOST_OWNED_ENVIRONMENT and name in os.environ:
            environment[name] = os.environ[name]
    environment.update(spec.env)
    # These switches belong to the operator, not to ambient or pack environment entries.
    environment.pop(MXC_EXECUTABLE_ENV, None)
    return {
        "version": "1.0.0",
        "containment": "processcontainer",
        "lifecycle": {"destroyOnExit": True, "preservePolicy": False},
        "process": {
            "commandLine": subprocess.list2cmdline(module_command(command)),
            "cwd": str(spec.endpoint_dir),
            "env": [f"{name}={value}" for name, value in sorted(environment.items())],
            "inheritDefaultEnv": False,
            "timeout": 0,
        },
        "filesystem": {
            "readonlyPaths": list(dict.fromkeys(readonly)),
            "readwritePaths": list(dict.fromkeys(writable)),
        },
        "network": {
            "egress": {"default": "deny"},
            "ingress": {"default": "deny", "hostLoopback": "deny"},
        },
        "ui": {"disable": False, "clipboard": "none", "injection": False},
    }, endpoints


class WindowsProcessLauncher:
    supports_shared_memory = False

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        configuration, endpoints = build_windows_policy(spec, self.policy)
        capability = await asyncio.to_thread(detect_windows_sandbox)
        if not capability.available or capability.executable is None:
            raise SandboxUnavailable(capability.detail)
        path = spec.endpoint_dir / "mxc-policy.json"
        with path.open("x", encoding="utf-8") as output:
            os.chmod(path, 0o600)
            json.dump(configuration, output)
        try:
            process = await asyncio.create_subprocess_exec(
                capability.executable,
                str(path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        async def supervise() -> None:
            bridge = asyncio.create_task(bridge_process(process, endpoints, spec.env))
            exited = asyncio.create_task(process.wait())
            try:
                await asyncio.wait({bridge, exited}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    await process.wait()
                bridge.cancel()
                await asyncio.gather(bridge, return_exceptions=True)
                await exited
                path.unlink(missing_ok=True)

        task = asyncio.create_task(supervise())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        return process
