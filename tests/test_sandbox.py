"""Sandboxing is a launcher concern (DESIGN 3.11): bwrap argv construction
is pure and inspectable, capability detection classifies this machine
honestly, and a requested sandbox that cannot be built refuses loudly
instead of running the child unjailed. The live test runs the whole worker
boundary inside a real bwrap jail - and skips, with the remediation in the
skip reason, on machines where the kernel blocks unprivileged user
namespaces (Ubuntu 24.04's default posture)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import aiohttp
import dinkster_workers.egress as egress_module
import dinkster_workers.sandbox as sandbox_module
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import (
    BubblewrapCapability,
    BubblewrapLauncher,
    IsolatedWorker,
    LaunchSpec,
    SandboxError,
    SandboxPolicy,
    SandboxUnavailable,
    build_bwrap_command,
    detect_bubblewrap,
)
from dinkster_workers.launch import SubprocessLauncher

from tests.async_scenarios import run_scenario

REPO_ROOT = Path(__file__).parents[1]
DEV_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"

CAPABILITY = detect_bubblewrap() if sys.platform == "linux" else None


def test_workers_import_without_a_home_directory() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from unittest.mock import patch; "
            "guard = patch('pathlib.Path.home', side_effect=RuntimeError('home unavailable')); "
            "guard.start(); import dinkster_workers",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows process containment")
def test_windows_readonly_vault_worker_can_hold_produced_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers.windows_sandbox import detect_windows_sandbox

    from dinkster import compose

    if not os.environ.get("DINKSTER_MXC_EXECUTABLE"):
        pytest.skip("Desktop MXC executor is not configured")
    capability = detect_windows_sandbox()
    if not capability.available:
        if "Windows process isolation is unavailable" in capability.detail:
            pytest.skip(capability.detail)
        pytest.fail(capability.detail)
    local = tmp_path / "local"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    vault = tmp_path / "library" / "vault"
    vault.mkdir(parents=True)
    sentinel = vault / "sentinel.txt"
    sentinel.write_text("keep", encoding="ascii")
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname="lease-probe"\npublisher="dinkster"\nnamespaces=["leaseprobe"]\n'
        '[pack.entry]\nnodes="lease_probe:NODES"\n',
        encoding="ascii",
    )
    (pack / "lease_probe.py").write_text(
        "import os\nfrom pathlib import Path\nfrom contextlib import closing\n"
        "from dinkster_assets.scratch import hold_produced_vault\n"
        "from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, Output, TypeExpr\n"
        "class Probe(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='leaseprobe.read', body=(\n"
        "            Output('value', TypeExpr.concrete(CORE_STRING)),))\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        vault = Path(os.environ['DINKSTER_ASSET_VAULT'])\n"
        "        with closing(hold_produced_vault(vault)):\n"
        "            value = (vault / 'sentinel.txt').read_text()\n"
        "            for path in (vault / 'forbidden.txt', vault / '.leases' / 'sibling.lock'):\n"
        "                try:\n"
        "                    path.write_text('escape')\n"
        "                except PermissionError:\n"
        "                    value += ':denied'\n"
        "            return cls.outputs(value=value + ':leased')\n"
        "NODES=(Probe,)\n",
        encoding="ascii",
    )

    async def scenario() -> None:
        composition = await compose.compose_serving(
            (
                compose.PackSpec(
                    manifest,
                    env={"DINKSTER_ASSET_VAULT": str(vault), "PYTHONPATH": str(pack)},
                ),
            ),
            include_default_packs=False,
            sandbox_policy=SandboxPolicy(ro_binds=(str(REPO_ROOT),)),
        )
        try:
            result = await composition.make_engine(lambda event: None).run(
                Graph(nodes={"read": GraphNode("leaseprobe.read", {})}), ["read"]
            )
            assert result.outputs["read"]["value"].resolve() == "keep:denied:denied:leased"
            assert sentinel.read_text(encoding="ascii") == "keep"
            assert not (vault / "forbidden.txt").exists()
            assert not (vault / ".leases" / "sibling.lock").exists()
        finally:
            await composition.close()

    run_scenario(scenario())


def test_windows_policy_keeps_environment_and_network_grants_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers.windows_sandbox import build_windows_policy

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("UNGRANTED_SANDBOX_TEST_VALUE", "ambient-secret")
    monkeypatch.setenv("GRANTED_SANDBOX_TEST_VALUE", "explicit-value")
    monkeypatch.setenv("DINKSTER_MXC_EXECUTABLE", str(tmp_path / "host-executor"))
    spec = make_spec(
        tmp_path,
        command=(sys.executable, "-m", "dinkster_workers.host", "--endpoint", "tcp:127.0.0.1:1234"),
        use_shm=False,
        env={"DINKSTER_BOUNDARY_TOKEN": "host-test-token", "https_proxy": "http://untrusted.test"},
    )
    policy = SandboxPolicy(
        env_passthrough=("GRANTED_SANDBOX_TEST_VALUE", "DINKSTER_MXC_EXECUTABLE")
    )
    configuration, endpoints = build_windows_policy(spec, policy)
    rendered = json.loads(json.dumps(configuration))
    assert endpoints == ["tcp:127.0.0.1:1234"]
    assert "pipe:0" in rendered["process"]["commandLine"]
    assert "tcp:" not in rendered["process"]["commandLine"]
    assert rendered["process"]["inheritDefaultEnv"] is False
    environment = dict(entry.split("=", 1) for entry in rendered["process"]["env"])
    assert environment["GRANTED_SANDBOX_TEST_VALUE"] == "explicit-value"
    assert "UNGRANTED_SANDBOX_TEST_VALUE" not in environment
    assert "DINKSTER_BOUNDARY_TOKEN" not in environment
    assert "DINKSTER_MXC_EXECUTABLE" not in environment
    assert not any("proxy" in name.lower() for name in environment)
    assert rendered["network"] == {
        "egress": {"default": "deny"},
        "ingress": {"default": "deny", "hostLoopback": "deny"},
    }
    assert "runtimeConfig" not in rendered
    with pytest.raises(SandboxError, match="not named shared memory"):
        build_windows_policy(replace(spec, use_shm=True), policy)
    with pytest.raises(SandboxError, match="private per-worker proxy"):
        build_windows_policy(
            spec, replace(policy, egress_allowlist=("https://allowed.example.test",))
        )
    allowed, _ = build_windows_policy(
        spec,
        replace(policy, egress_allowlist=("https://allowed.example.test",)),
        proxy_url="http://dinkster:private-test-token@127.0.0.1:54321",
    )
    proxy_mode = json.loads(json.dumps(allowed))
    assert proxy_mode["runtimeConfig"] == {"networkProxy": "http://127.0.0.1:54321"}
    assert proxy_mode["network"]["egress"] == {"default": "deny"}
    assert proxy_mode["network"]["ingress"] == {"default": "allow", "hostLoopback": "allow"}


@pytest.mark.parametrize("layout", ["python/python.exe", "venv/Scripts/python.exe"])
def test_windows_policy_only_exposes_interpreter_inside_protected_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    from dinkster_workers.windows_sandbox import build_windows_policy

    generation = tmp_path / "generation"
    python = generation / layout
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test-interpreter")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    spec = make_spec(
        tmp_path,
        python=str(python),
        use_shm=False,
        command=(str(python), "-m", "dinkster_workers.host", "--endpoint", "tcp:127.0.0.1:1234"),
    )
    policy = SandboxPolicy(protected_roots=(str(generation),))
    config, _ = build_windows_policy(spec, policy)
    paths = json.loads(json.dumps(config))["filesystem"]["readonlyPaths"]
    prefix = python.parent.parent if python.parent.name == "Scripts" else python.parent
    assert str(prefix) in paths
    assert str(generation) not in paths
    with pytest.raises(SandboxError, match="protected root"):
        build_windows_policy(spec, replace(policy, ro_binds=(str(generation),)))


def test_windows_launcher_reaps_worker_after_invalid_frame_and_output_flood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers.transport import BoundaryListener
    from dinkster_workers.windows_sandbox import WindowsProcessLauncher, WindowsSandboxCapability

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(
        "dinkster_workers.windows_sandbox.detect_windows_sandbox",
        lambda: WindowsSandboxCapability(True, sys.executable, "test backend"),
    )
    spawn = asyncio.create_subprocess_exec

    async def backend(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        return await spawn(
            sys.executable,
            "-c",
            "import sys,struct,time; sys.stdout.buffer.write(struct.pack('!II',999,1)); "
            "sys.stdout.buffer.write(b'x' * (5 * 1024 * 1024)); "
            "sys.stdout.buffer.flush(); time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr("dinkster_workers.windows_sandbox.asyncio.create_subprocess_exec", backend)

    async def scenario() -> None:
        listener = await BoundaryListener.create(tmp_path, transport="tcp")
        spec = make_spec(
            tmp_path,
            use_shm=False,
            env=listener.child_env,
            command=(
                sys.executable,
                "-m",
                "dinkster_workers.host",
                "--endpoint",
                listener.endpoint,
            ),
        )
        try:
            process = await WindowsProcessLauncher().launch(spec)
            assert await asyncio.wait_for(process.wait(), 5) != 0
        finally:
            if listener.connected.done() and not listener.connected.cancelled():
                _, writer = listener.connected.result()
                writer.close()
                await writer.wait_closed()
            await listener.close()

    run_scenario(scenario(), timeout=10)


def test_windows_readiness_refuses_tampered_or_routed_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from types import SimpleNamespace

    from dinkster_workers.windows_sandbox import MXC_COMMIT, detect_windows_sandbox

    executable = tmp_path / "wxc-exec.exe"
    executable.write_bytes(b"trusted-host-test-executor")
    metadata = {
        "format": "dinkster.mxc/1",
        "commit": MXC_COMMIT,
        "telemetryComponent": "",
        "sha256": hashlib.sha256(b"trusted-host-test-executor").hexdigest(),
    }
    manifest = tmp_path / "mxc-build.json"
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setenv("DINKSTER_MXC_EXECUTABLE", str(executable))
    monkeypatch.setattr("dinkster_workers.windows_sandbox.sys.platform", "win32")
    monkeypatch.setattr(
        "dinkster_workers.windows_sandbox.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "tier": "base-container",
                    "probes": {"baseContainerApiPresent": True},
                    "needsDaclAugmentation": False,
                }
            ),
        ),
    )
    readiness = detect_windows_sandbox()
    assert readiness.available
    assert "On native Windows 11, DNS lookups can leave the sandbox" in readiness.detail
    assert (
        "even when HTTPS destinations are restricted (Microsoft MXC issue 1666)."
        in readiness.detail
    )
    executable.write_bytes(b"changed-host-test-executor")
    assert not detect_windows_sandbox().available
    executable.write_bytes(b"trusted-host-test-executor")
    metadata["telemetryComponent"] = "mxc_telemetry.dll"
    manifest.write_text(json.dumps(metadata))
    assert not detect_windows_sandbox().available
    monkeypatch.delenv("DINKSTER_MXC_EXECUTABLE")
    assert not detect_windows_sandbox().available


@pytest.mark.parametrize("outcome", ["success", "denied", "malformed", "executor-error", "timeout"])
def test_windows_readiness_requires_contained_wakeup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from types import SimpleNamespace

    from dinkster_workers.windows_sandbox import (
        WindowsSandboxCapability,
        check_windows_sandbox,
    )

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "redirected-profile"))
    monkeypatch.setenv("UNGRANTED_SANDBOX_TEST_VALUE", "ambient-secret")
    monkeypatch.setattr(
        "dinkster_workers.windows_sandbox.detect_windows_sandbox",
        lambda: WindowsSandboxCapability(True, "test-executor", "available with DNS caveat"),
    )
    paths: list[Path] = []

    def executor(command: list[str], **kwargs: object) -> SimpleNamespace:
        path = Path(command[1])
        paths.append(path)
        policy = json.loads(path.read_text())
        assert policy["containment"] == "processcontainer"
        assert policy["process"]["inheritDefaultEnv"] is False
        assert "dinkster_workers._windows_loop" in policy["process"]["commandLine"]
        assert "pipe:0" in policy["process"]["commandLine"]
        assert "ambient-secret" not in path.read_text()
        assert policy["network"] == {
            "egress": {"default": "deny"},
            "ingress": {"default": "deny", "hostLoopback": "deny"},
        }
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        if outcome == "malformed":
            return SimpleNamespace(returncode=0, stdout="executor API exists", stderr="")
        if outcome == "executor-error":
            return SimpleNamespace(
                returncode=72, stdout="", stderr="\x1bcontainer profile unavailable\n" + "x" * 2000
            )
        return SimpleNamespace(
            returncode=0 if outcome == "success" else 1,
            stdout=json.dumps(
                {
                    "format": "dinkster.windows-wakeup/1",
                    "ok": outcome == "success",
                    "detail": "[Errno 5] Windows IOCP PostQueuedCompletionStatus failed: denied",
                }
            ),
        )

    monkeypatch.setattr("dinkster_workers.windows_sandbox.subprocess.run", executor)
    report = check_windows_sandbox()
    assert report.available is (outcome == "success")
    assert paths and all(not path.parent.exists() for path in paths)
    if outcome == "success":
        assert report.detail == "available with DNS caveat"
    else:
        assert "Windows contained wakeup unavailable" in report.detail
        assert (
            f"policy process.env LOCALAPPDATA={str(tmp_path / 'redirected-profile')!r}"
            in report.detail
        )
        assert "WSL2" in report.detail
        if outcome == "denied":
            assert "[Errno 5]" in report.detail and "PostQueuedCompletionStatus" in report.detail
        elif outcome == "executor-error":
            assert "executor exit 72" in report.detail
            assert "container profile unavailable" in report.detail
            assert "Expecting value" not in report.detail
            assert "\x1b" not in report.detail and "x" * 1001 not in report.detail


@pytest.mark.skipif(sys.platform != "win32", reason="Windows IOCP event loop")
def test_windows_worker_wakes_without_sockets_or_temp_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import tempfile
    import threading

    from dinkster_workers._windows_loop import check_wakeup, worker_loop

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("contained wakeup must not create sockets or select a temp path")

    monkeypatch.setattr(socket, "socketpair", forbidden)
    monkeypatch.setattr(tempfile, "gettempdir", forbidden)
    monkeypatch.setenv("TEMP", "C:\\" + "long-profile-" * 20)
    check_wakeup()
    loop = worker_loop()
    delivered: list[int] = []
    expired = threading.Event()

    def timeout() -> None:
        expired.set()
        loop.stop()

    try:
        for _ in range(2):
            done = threading.Event()

            def callbacks(done: threading.Event = done) -> None:
                for value in range(100):
                    loop.call_soon_threadsafe(delivered.append, value)
                loop.call_soon_threadsafe(done.set)
                loop.call_soon_threadsafe(loop.stop)

            sender = threading.Timer(0.02, callbacks)
            sender.start()
            watchdog = loop.call_later(2, timeout)
            loop.run_forever()
            sender.join()
            watchdog.cancel()
            assert done.is_set() and not expired.is_set()
        assert delivered == list(range(100)) * 2
        # Close must drain a posted completion without running its callback.
        loop.call_soon_threadsafe(delivered.append, -1)
    finally:
        loop.close()
    assert delivered == list(range(100)) * 2
    loop.close()
    with pytest.raises(RuntimeError, match="closed"):
        loop.call_soon_threadsafe(lambda: None)


def make_spec(tmp_path: Path, **overrides: object) -> LaunchSpec:
    endpoint_dir = tmp_path / "endpoint"
    endpoint_dir.mkdir(exist_ok=True)
    pack_root = tmp_path / "pack"
    pack_root.mkdir(exist_ok=True)
    fields: dict[str, object] = {
        "command": (sys.executable, "-m", "dinkster_workers.host"),
        "env": {"DINKSTER_BOUNDARY_TOKEN": "sekrit"},
        "endpoint_dir": endpoint_dir,
        "pack_root": pack_root,
        "python": sys.executable,
        "use_shm": True,
    }
    fields.update(overrides)
    return LaunchSpec(**fields)  # type: ignore[arg-type]


def worker_failure(error: str, stderr_path: Path) -> AssertionError:
    return AssertionError(f"{error}\nworker stderr:\n{stderr_path.read_text(encoding='utf-8')}")


@pytest.mark.parametrize("mxc_key", ["DINKSTER_MXC_EXECUTABLE", "dinkster_mxc_executable"])
def test_plain_launcher_strips_host_executor_but_preserves_explicit_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mxc_key: str,
) -> None:
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", "/inherited/untrusted")
    monkeypatch.setenv(mxc_key, "/host/executor")
    absent_result = tmp_path / "absent.txt"
    trusted_result = tmp_path / "trusted.txt"
    script = (
        "import os, sys; from pathlib import Path; "
        "assert not any(key.upper() == 'DINKSTER_MXC_EXECUTABLE' for key in os.environ); "
        "Path(sys.argv[1]).write_text(os.environ.get('DINKSTER_PACK_SCRATCH', '<absent>'))"
    )

    async def scenario() -> None:
        launcher = SubprocessLauncher()
        absent = await launcher.launch(
            make_spec(
                tmp_path,
                command=(sys.executable, "-c", script, str(absent_result)),
                env={},
            )
        )
        assert await absent.wait() == 0
        trusted = await launcher.launch(
            make_spec(
                tmp_path,
                command=(sys.executable, "-c", script, str(trusted_result)),
                env={"DINKSTER_PACK_SCRATCH": "/host-owned", mxc_key: "/explicit/executor"},
            )
        )
        assert await trusted.wait() == 0

    run_scenario(scenario())
    assert absent_result.read_text(encoding="utf-8") == "<absent>"
    assert trusted_result.read_text(encoding="utf-8") == "/host-owned"


def pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    """All (a, b) argument pairs following a two-argument bwrap flag."""
    return [(argv[i + 1], argv[i + 2]) for i, arg in enumerate(argv) if arg == flag]


class TestBuildBwrapCommand:
    @pytest.mark.skipif(sys.platform != "linux", reason="requires Linux WSL mount semantics")
    def test_wsl_gpu_grant(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        exists = os.path.exists
        isdir = os.path.isdir
        glob = Path.glob
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/dev/dxg" or exists(p))
        monkeypatch.setattr(os.path, "isdir", lambda p: p == "/usr/lib/wsl" or isdir(p))
        monkeypatch.setattr(
            Path,
            "glob",
            lambda p, pattern: (
                iter((Path("/dev/dxg"),))
                if p == Path("/dev") and pattern == "dxg"
                else glob(p, pattern)
            ),
        )
        spec = make_spec(tmp_path)
        granted = build_bwrap_command(spec, SandboxPolicy(gpu=True))
        assert ("/dev/dxg", "/dev/dxg") in pairs(granted, "--dev-bind")
        assert ("/usr/lib/wsl", "/usr/lib/wsl") in pairs(granted, "--ro-bind")
        denied = build_bwrap_command(spec, SandboxPolicy())
        assert ("/dev/dxg", "/dev/dxg") not in pairs(denied, "--dev-bind")
        assert ("/usr/lib/wsl", "/usr/lib/wsl") not in pairs(denied, "--ro-bind")

    @pytest.mark.skipif(sys.platform != "linux", reason="requires Linux WSL mount semantics")
    def test_wsl_refuses_windows_drives_and_interop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drive = tmp_path / "Windows drive"
        drive.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(drive, target_is_directory=True)
        read_text = Path.read_text
        monkeypatch.setattr(platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
        escaped_drive = str(drive).replace(" ", r"\040")
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda p, *a, **kw: (
                (f"C: {escaped_drive} 9p rw,aname=drvfs;path=C: 0 0\n/dev/sdc / ext4 rw 0 0\n")
                if p == Path("/proc/mounts")
                else read_text(p, *a, **kw)
            ),
        )
        spec = make_spec(tmp_path)
        for path in (str(drive), str(alias / "models"), "/run/WSL/123_interop", "/init"):
            for grant in (SandboxPolicy(ro_binds=(path,)), SandboxPolicy(rw_binds=(path,))):
                with pytest.raises(SandboxError, match="Windows drive or WSL interop"):
                    build_bwrap_command(spec, grant)
            with pytest.raises(SandboxError, match="Windows drive or WSL interop"):
                build_bwrap_command(make_spec(tmp_path, env={"PYTHONPATH": path}), SandboxPolicy())
        build_bwrap_command(spec, SandboxPolicy(ro_binds=(str(tmp_path / "pack"),)))

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX root semantics")
    def test_pythonpath_cannot_dissolve_the_jail(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path, env={"PYTHONPATH": "/", "DINKSTER_BOUNDARY_TOKEN": "s"})
        with pytest.raises(SandboxError, match="PYTHONPATH"):
            build_bwrap_command(spec, SandboxPolicy())

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX root semantics")
    def test_policy_binds_land_and_forbidden_roots_refuse(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir()
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy(ro_binds=(str(models),)))
        assert (str(models), str(models)) in pairs(argv, "--ro-bind")
        for root in ("/", "/home", "/etc"):
            with pytest.raises(SandboxError):
                build_bwrap_command(make_spec(tmp_path), SandboxPolicy(ro_binds=(root,)))
        with pytest.raises(SandboxError):
            build_bwrap_command(make_spec(tmp_path), SandboxPolicy(rw_binds=("/tmp",)))

        with pytest.raises(SandboxError, match="pack root"):
            build_bwrap_command(make_spec(tmp_path, pack_root=Path("/home")), SandboxPolicy())

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
    def test_symlink_to_forbidden_root_refuses(self, tmp_path: Path) -> None:
        grant = tmp_path / "grant"
        grant.symlink_to("/home")
        with pytest.raises(SandboxError, match="dissolve the sandbox"):
            build_bwrap_command(make_spec(tmp_path), SandboxPolicy(ro_binds=(str(grant),)))

        interpreter = tmp_path / "interpreter"
        interpreter.symlink_to("/home")
        python = interpreter / "bin" / "python"
        with pytest.raises(SandboxError, match="interpreter"):
            build_bwrap_command(
                make_spec(tmp_path, python=str(python), command=(str(python), "-V")),
                SandboxPolicy(),
            )

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
    def test_forbidden_root_that_is_itself_a_link_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # macOS firmlink topology: the forbidden root is itself a link, so a
        # grant symlinked to it resolves past it (/home resolves to
        # /System/Volumes/Data/home) and never equals the literal root name.
        real_home = tmp_path / "data" / "home"
        real_home.mkdir(parents=True)
        root = tmp_path / "root_home"
        root.symlink_to(real_home)
        monkeypatch.setattr(sandbox_module, "FORBIDDEN_BIND_ROOTS", frozenset({str(root)}))
        grant = tmp_path / "grant"
        grant.symlink_to(root)
        with pytest.raises(SandboxError, match="dissolve the sandbox"):
            build_bwrap_command(make_spec(tmp_path), SandboxPolicy(ro_binds=(str(grant),)))

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX interpreter symlinks")
    def test_managed_interpreter_mounts_do_not_expose_the_protected_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        editable = tmp_path / "editable-source"
        editable.mkdir()
        monkeypatch.syspath_prepend(str(editable))
        generation = tmp_path / "generation"
        runtime = generation / "python"
        pack = generation / "packs" / "example"
        for prefix in (runtime, pack):
            (prefix / "bin").mkdir(parents=True)
        (runtime / "bin" / "python3").touch()
        python = pack / "bin" / "python"
        python.symlink_to(runtime / "bin" / "python3")
        policy = SandboxPolicy(
            protected_roots=(str(generation),),
            protected_ro_exceptions=(str(pack),),
        )
        spec = make_spec(tmp_path, python=str(python), pack_root=pack)
        argv = build_bwrap_command(spec, policy)
        binds = pairs(argv, "--ro-bind")
        assert (str(runtime), str(runtime)) in binds
        assert (str(editable), str(editable)) in binds
        assert (str(pack), str(pack)) in binds
        assert (str(generation), str(generation)) not in binds
        settings = generation / "settings"
        settings.mkdir()
        for field in ("ro_binds", "rw_binds"):
            with pytest.raises(SandboxError, match="protected root"):
                build_bwrap_command(spec, replace(policy, **{field: (str(settings),)}))
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(spec, replace(policy, ro_binds=(str(generation),)))
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(spec, replace(policy, rw_binds=(str(runtime),)))
        secret = runtime / "auth.token"
        secret.touch()
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(
                spec, replace(policy, protected_roots=(str(generation), str(secret)))
            )
        monkeypatch.setenv("PYTHONPATH", str(settings))
        monkeypatch.syspath_prepend(str(settings))
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(replace(spec, env={"PYTHONPATH": str(settings)}), policy)
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(spec, policy)
        monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(settings)])
        monkeypatch.syspath_prepend(str(generation))
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(spec, policy)

    def test_protected_roots_and_read_only_grants_cannot_be_overridden(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        vault = library / "vault"
        vault.mkdir(parents=True)
        pack = tmp_path / "pack"
        pack.mkdir(exist_ok=True)
        policy = SandboxPolicy(
            ro_binds=(str(vault),),
            protected_roots=(str(library),),
            protected_ro_exceptions=(str(vault),),
        )
        build_bwrap_command(make_spec(tmp_path, pack_root=pack), policy)

        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(make_spec(tmp_path, pack_root=tmp_path), policy)
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(
                make_spec(tmp_path, pack_root=tmp_path),
                SandboxPolicy(
                    protected_roots=(str(library),),
                    protected_ro_exceptions=(str(tmp_path),),
                ),
            )
        with pytest.raises(SandboxError, match="overlaps read-only bind"):
            build_bwrap_command(
                make_spec(tmp_path, pack_root=pack),
                SandboxPolicy(ro_binds=(str(vault),), rw_binds=(str(library),)),
            )
        endpoint = pack / "endpoint"
        endpoint.mkdir()
        with pytest.raises(SandboxError, match="overlaps read-only bind"):
            build_bwrap_command(make_spec(tmp_path, pack_root=pack, endpoint_dir=endpoint), policy)
        if os.path.exists("/dev/shm"):
            with pytest.raises(SandboxError, match="writable launch"):
                build_bwrap_command(
                    make_spec(tmp_path, pack_root=pack),
                    SandboxPolicy(protected_roots=("/dev/shm/dinkster-library",)),
                )
        shared = tmp_path / "shared"
        shared.mkdir()
        configured_auth = shared / "auth.toml"
        configured_auth.touch()
        with pytest.raises(SandboxError, match="protected root"):
            build_bwrap_command(
                make_spec(tmp_path, pack_root=pack),
                SandboxPolicy(
                    ro_binds=(str(shared),),
                    protected_roots=(str(configured_auth),),
                ),
            )


class TestDetection:
    @pytest.mark.skipif(sys.platform != "linux", reason="Linux bubblewrap")
    @pytest.mark.parametrize("missing", ["--disable-userns", "--seccomp"])
    def test_missing_hardening_option_refuses_before_pack_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        bwrap = shutil.which("bwrap")
        if bwrap is None or CAPABILITY is None or not CAPABILITY.available:
            pytest.skip("requires working bubblewrap")
        old_bwrap = tmp_path / "bwrap"
        old_bwrap.write_text(
            f'#!/bin/sh\nfor arg do\nif [ "$arg" = "{missing}" ]; then\n'
            f'echo "bwrap: Unknown option {missing}" >&2\nexit 1\nfi\ndone\n'
            f'exec "{bwrap}" "$@"\n'
        )
        old_bwrap.chmod(0o700)
        monkeypatch.setattr(shutil, "which", lambda _: str(old_bwrap))
        capability = detect_bubblewrap()
        assert not capability.available
        assert f"Unknown option {missing}" in capability.detail
        marker = tmp_path / "executed"

        async def scenario() -> None:
            with pytest.raises(SandboxUnavailable, match=f"Unknown option {missing}"):
                await BubblewrapLauncher().launch(
                    make_spec(
                        tmp_path,
                        command=(sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"),
                    )
                )

        run_scenario(scenario())
        assert not marker.exists()

    def test_unavailable_capability_refuses_loudly(self, tmp_path: Path) -> None:
        launcher = BubblewrapLauncher(
            capability=BubblewrapCapability(
                available=False, bwrap=None, user_namespaces=False, detail="blocked"
            )
        )

        async def scenario() -> None:
            with pytest.raises(SandboxUnavailable, match="blocked"):
                await launcher.launch(make_spec(tmp_path))

        run_scenario(scenario())

    def test_degraded_mode_refuses_even_without_secret_environment(self, tmp_path: Path) -> None:
        launcher = BubblewrapLauncher(
            capability=BubblewrapCapability(
                available=True,
                bwrap="/usr/bin/bwrap",
                user_namespaces=False,
                detail="mount-only",
            )
        )

        async def scenario() -> None:
            with pytest.raises(SandboxUnavailable, match="mount-only degraded mode is refused"):
                await launcher.launch(make_spec(tmp_path, env={}))

        run_scenario(scenario())


@pytest.mark.skipif(
    CAPABILITY is None or not CAPABILITY.available,
    reason=(
        "bubblewrap jail cannot be built here: "
        + (CAPABILITY.detail if CAPABILITY is not None else "not Linux")
    ),
)
class TestLiveJail:
    def test_helper_published_indexes_resolve_with_only_snapshot_and_folder_grants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dinkster_assets import digest_bytes
        from dinkster_library import LibraryClient

        library = tmp_path / "library"
        state = library / "worker-state"
        root = tmp_path / "models"
        root.mkdir()
        model = root / "model.bin"
        model.write_bytes(b"sandboxed model")
        index = state / "asset-indexes" / "models.json"
        snapshot = state / "mounts.json"
        monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))
        with LibraryClient(library) as helper:
            scan = helper.call("index.scan", root=str(root), output=str(index))
            deadline = time.monotonic() + 10
            while scan["state"] in ("queued", "scanning") and time.monotonic() < deadline:
                time.sleep(0.02)
                scan = helper.call("index.progress", id=scan["id"])
            assert scan["state"] == "complete", scan
        snapshot.write_text(
            json.dumps(
                {
                    "mounts": [
                        {
                            "id": "models",
                            "root": str(root),
                            "index": str(index),
                            "mode": "readonly",
                            "kind": "model/checkpoint",
                            "priority": 0,
                        }
                    ]
                }
            ),
            "utf-8",
        )
        script = f"""
from pathlib import Path
from dinkster_assets import AssetRef, MountSnapshotResolver
resolver = MountSnapshotResolver({str(snapshot)!r})
ref = AssetRef({digest_bytes(b"sandboxed model")!r}, 'model.bin', 15, resolver=resolver)
assert ref.read_bytes() == b'sandboxed model'
assert not Path({str(library / "records.sqlite")!r}).exists()
"""

        async def scenario() -> None:
            policy = SandboxPolicy(
                ro_binds=(str(REPO_ROOT), str(state), str(root)),
                protected_roots=(str(library),),
                protected_ro_exceptions=(str(state),),
            )
            process = await BubblewrapLauncher(policy).launch(
                make_spec(
                    tmp_path,
                    command=(sys.executable, "-c", script),
                )
            )
            try:
                assert await asyncio.wait_for(process.wait(), timeout=15) == 0
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

        run_scenario(scenario())

    @pytest.mark.parametrize("sandbox,granted", [(False, False), (True, False), (True, True)])
    @pytest.mark.parametrize("catalog", [False, True])
    @pytest.mark.parametrize("parts", [False, True])
    def test_cpu_group_survives_ungranted_gpu_member(
        self, tmp_path: Path, sandbox: bool, granted: bool, catalog: bool, parts: bool
    ) -> None:
        from dinkster_workers.doctor import prepare_catalog

        from dinkster.compose import CompositionError, PackSpec, ServingComposer

        generation = tmp_path / "generations" / "2"
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "sandbox_group_helper.py").write_text("MARK = 17\n", encoding="ascii")
        denied_path = str(generation / "pack-sources" / "gpu-denied")
        manifests = []
        for name, gpu in (
            ("cpu-one", False),
            ("cpu-two", False),
            ("gpu-only", True),
            ("gpu-denied", True),
        ):
            root = generation / "pack-sources" / name
            root.mkdir(parents=True)
            module = name.replace("-", "_")
            manifest = root / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname="{name}"\npublisher="dinkster"\nnamespaces=["{module}"]\n'
                f"[pack.sandbox]\ngpu={str(not gpu if parts else gpu).lower()}\n"
                f'[pack.entry]\nnodes="{module}:NODES"\n'
                + (
                    '[[pack.parts]]\nname="selected"\n'
                    f"[pack.parts.sandbox]\ngpu={str(gpu).lower()}\n"
                    f'[pack.parts.entry]\nnodes="{module}:NODES"\n'
                    if parts
                    else ""
                ),
                encoding="ascii",
            )
            (root / f"{module}.py").write_text(
                "import json, os\n"
                "from dinkster_api.v1 import Node, NodeSchema, Output, TypeExpr\n"
                "class Probe(Node):\n"
                "    @classmethod\n"
                "    def define_schema(cls):\n"
                f"        return NodeSchema(node_type='{module}.echo', body=(\n"
                "            Output('report', TypeExpr.concrete('STRING')),))\n"
                "    @classmethod\n"
                "    def execute(cls):\n"
                "        report = {key: os.readlink('/proc/self/ns/' + key) "
                "for key in ('user', 'pid', 'net')}\n"
                "        report['process'] = os.getpid()\n"
                "        from sandbox_group_helper import MARK\n"
                "        report['helper'] = MARK\n"
                "        report['gpu_device'] = os.path.exists('/dev/nvidiactl')\n"
                f"        report['denied_pack'] = os.path.exists({denied_path!r})\n"
                f"        report['value'] = '{name}: Pack sandbox | 966'\n"
                "        return cls.outputs(report=json.dumps(report))\n"
                "NODES=(Probe,)\n",
                encoding="ascii",
            )
            manifests.append(manifest)
        part = "selected" if parts else None
        if catalog:
            for manifest in manifests:
                assert prepare_catalog(manifest, part=part).ok
        specs = tuple(
            PackSpec(
                manifest,
                part=part,
                env={
                    "PYTHONPATH": os.pathsep.join(
                        [str(extra), *(str(path.parent) for path in manifests)]
                    )
                },
                worker_group="inference",
                group_manifests=tuple(manifests),
                group_parts=(part,) * len(manifests),
                require_catalog=catalog,
            )
            for manifest in manifests
        )
        host = {key: os.readlink("/proc/self/ns/" + key) for key in ("user", "pid", "net")}

        async def scenario() -> None:
            composer = ServingComposer(
                sandbox_policy=(
                    SandboxPolicy(protected_roots=(str(generation),)) if sandbox else None
                ),
                sandbox_gpu_grants=("gpu-only",) if granted else (),
                worker_env={"CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"},
            )
            try:
                if sandbox:
                    with pytest.raises(CompositionError, match="not a member of worker group"):
                        await composer.add_pack(
                            replace(
                                specs[0],
                                group_manifests=tuple(manifests[1:]),
                                group_parts=(part,) * (len(manifests) - 1),
                            )
                        )
                if granted or not sandbox:
                    await composer.add_pack(specs[2])
                else:
                    with pytest.raises(
                        CompositionError,
                        match=f"not granted for: gpu-only{'~selected' if parts else ''}$",
                    ):
                        await composer.add_pack(specs[2])
                if sandbox:
                    with pytest.raises(
                        CompositionError,
                        match=f"not granted for: gpu-denied{'~selected' if parts else ''}$",
                    ):
                        await composer.add_pack(specs[3])
                else:
                    await composer.add_pack(specs[3])
                for spec in specs[:2]:
                    await composer.add_pack(spec)
                for _ in range(2):
                    engine = composer.composition.make_engine(lambda _event: None)
                    names = (
                        ("cpu_one", "cpu_two", "gpu_only")
                        if granted or not sandbox
                        else ("cpu_one", "cpu_two")
                    )
                    nodes = {name: GraphNode(name + ".echo", {}) for name in names}
                    result = await engine.run(Graph(nodes=nodes), list(nodes))
                    reports = []
                    for name in nodes:
                        report = result.outputs[name]["report"].resolve()
                        assert isinstance(report, str)
                        reports.append(json.loads(report))
                    assert reports[0]["value"] == "cpu-one: Pack sandbox | 966"
                    assert reports[1]["value"] == "cpu-two: Pack sandbox | 966"
                    assert len({report["process"] for report in reports}) == 1
                    if len(reports) == 3:
                        assert reports[2]["value"] == "gpu-only: Pack sandbox | 966"
                    for report in reports:
                        assert all((report[key] != host[key]) == sandbox for key in host)
                        assert report["helper"] == 17
                        assert report["denied_pack"] is not sandbox
                        if sandbox and not granted:
                            assert report["gpu_device"] is False
                    await composer.reload_pack("cpu-one~selected" if parts else "cpu-one", specs[0])
            finally:
                await composer.close()

        run_scenario(scenario())
        if sandbox:
            from dinkster_engine import ExecutionError

            private = generation / "private"
            private.mkdir()

            async def refuse_unrelated_path() -> None:
                composer = ServingComposer(
                    sandbox_policy=SandboxPolicy(protected_roots=(str(generation),)),
                    worker_env={"CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"},
                )
                try:
                    with pytest.raises(
                        (SandboxError, ExecutionError), match="private.*overlaps protected root"
                    ):
                        await composer.add_pack(
                            replace(
                                specs[0],
                                env={
                                    "PYTHONPATH": specs[0].env["PYTHONPATH"]
                                    + os.pathsep
                                    + str(private)
                                },
                            )
                        )
                        engine = composer.composition.make_engine(lambda _event: None)
                        await engine.run(Graph(nodes={"n": GraphNode("cpu_one.echo", {})}), ["n"])
                finally:
                    await composer.close()

            run_scenario(refuse_unrelated_path())

    def test_kernel_interfaces_are_denied_but_dataloader_and_shared_memory_work(
        self, tmp_path: Path
    ) -> None:
        report = tmp_path / "endpoint" / "hardening.json"
        script = f"""
import ctypes
import errno
import json
import os
import threading
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory
import torch
from torch.utils.data import DataLoader, TensorDataset

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
seccomp = ctypes.CDLL('libseccomp.so.2')
seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
results = {{}}
def denied(name, args, expected):
    number = seccomp.seccomp_syscall_resolve_name(name.encode())
    assert number >= 0, name
    ctypes.set_errno(0)
    value = libc.syscall(ctypes.c_long(number), *args)
    results[name] = [value, ctypes.get_errno()]
    assert results[name] == [-1, expected], (name, results[name])

# These calls normally succeed, or return an error other than EPERM. Testing
# them avoids mistaking an existing kernel capability check for our filter.
denied('personality', [0xffffffff], errno.EPERM)
denied('process_vm_readv', [os.getpid(), 0, 0, 0, 0, 0], errno.EPERM)
denied('ioctl', [-1, 0x5412, 0], errno.EPERM)
denied('unshare', [0x10000000], errno.EPERM)
denied('clone', [0x10000000 | 17, 0, 0, 0, 0], errno.EPERM)
denied('clone3', [0, 0], errno.ENOSYS)
denied('io_uring_setup', [0, 0], errno.ENOSYS)
for name in ('syslog', 'mount', 'bpf', 'userfaultfd', 'perf_event_open', 'keyctl', 'setns'):
    denied(name, [0, 0, 0, 0, 0, 0], errno.EPERM)

values = []
thread = threading.Thread(target=lambda: values.append(37))
thread.start()
thread.join()
assert values == [37]
segment = SharedMemory(create=True, size=16)
try:
    reader = SharedMemory(name=segment.name)
    try:
        segment.buf[:4] = b'pack'
        assert bytes(reader.buf[:4]) == b'pack'
    finally:
        reader.close()
finally:
    segment.close()
    segment.unlink()
data = torch.arange(13, dtype=torch.int64)
loader = DataLoader(TensorDataset(data), batch_size=4, num_workers=2)
assert torch.cat([batch[0] for batch in loader]).tolist() == list(range(13))
Path({str(report)!r}).write_text(json.dumps(results))
"""

        async def scenario() -> None:
            launcher = BubblewrapLauncher()
            process = await launcher.launch(
                make_spec(tmp_path, command=(sys.executable, "-c", script))
            )
            assert await asyncio.wait_for(process.wait(), timeout=60) == 0

        run_scenario(scenario())
        assert json.loads(report.read_text())["clone3"] == [-1, 38]

    def test_asset_selections_read_granted_members_without_write_or_parent_access(
        self, tmp_path: Path
    ) -> None:
        from dinkster_assets import MountDef, MountTable

        state = tmp_path / "state"
        state.mkdir()
        snapshot = state / "snapshot.json"
        roots = {
            "model/vae-approx": {
                "taesd_encoder.safetensors": b"encoder",
                "taesd_decoder.safetensors": b"decoder",
            },
            "model/diffusers": {"model/model_index.json": b"{}", "model/weights.bin": b"weights"},
            "directory/input": {"pictures/caption.txt": b"caption"},
            "directory/dataset": {"training/metadata.json": b"{}"},
        }
        table = MountTable(snapshot)
        granted = []
        for kind, files in roots.items():
            root = tmp_path / kind.replace("/", "-")
            for relative, data in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            mount_id = kind.replace("/", "-")
            table.add(MountDef(id=mount_id, path=root, kind=kind))
            table.scan(mount_id)
            granted.append(str(root))
        hidden = tmp_path / "ungranted.txt"
        hidden.write_text("not granted")
        script = f"""
from pathlib import Path
from dinkster_api.v1 import AssetError, resolve_asset_selection
pair = resolve_asset_selection('model/vae', 'taesd')
assert [member.read_bytes() for member in pair] == [b'encoder', b'decoder']
try:
    pair[0].local_path().write_bytes(b'no')
except OSError:
    pass
else:
    raise AssertionError('selected pair member became writable')
assert resolve_asset_selection('model/vae', 'pixel_space') is None
for kind, name, member, content in (
    ('model/diffusers', 'model', 'weights.bin', b'weights'),
    ('directory/input', 'pictures', 'caption.txt', b'caption'),
    ('directory/dataset', 'training', 'metadata.json', b'{{}}'),
):
    path = resolve_asset_selection(kind, name)
    assert (path / member).read_bytes() == content
    try:
        (path / 'forbidden.txt').write_text('no')
    except OSError:
        pass
    else:
        raise AssertionError('selected subtree became writable')
    try:
        resolve_asset_selection(kind, '../ungranted.txt')
    except AssetError:
        pass
    else:
        raise AssertionError('selection escaped its grant')
assert not Path({str(hidden)!r}).exists()
"""

        async def scenario() -> None:
            launcher = BubblewrapLauncher(
                SandboxPolicy(ro_binds=(str(REPO_ROOT), str(state), *granted))
            )
            process = await launcher.launch(
                make_spec(
                    tmp_path,
                    command=(sys.executable, "-c", script),
                    env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)},
                )
            )
            try:
                assert await asyncio.wait_for(process.wait(), timeout=15) == 0
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

        run_scenario(scenario())

    def test_public_embedding_roots_remain_read_only_and_scoped(self, tmp_path: Path) -> None:
        from dinkster_assets import MountDef, MountTable

        state = tmp_path / "state"
        state.mkdir()
        snapshot = state / "snapshot.json"
        embedding = tmp_path / "embedding"
        embedding.mkdir()
        (embedding / "sentinel.txt").write_text("embedding bytes")
        hidden = tmp_path / "hidden"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("not granted")
        table = MountTable(snapshot)
        table.add(MountDef(id="embedding", path=embedding), kind="model/embedding")
        table.scan("embedding")
        table.add(MountDef(id="hidden", path=hidden), kind="model/checkpoint")
        table.scan("hidden")
        observed = tmp_path / "endpoint" / "roots.json"
        script = f"""
import json
from pathlib import Path
from dinkster_api.v1 import mount_roots_for_kind
roots = mount_roots_for_kind('model/embedding')
assert roots == (Path({str(embedding)!r}),)
assert (roots[0] / 'sentinel.txt').read_text() == 'embedding bytes'
try:
    (roots[0] / 'forbidden.txt').write_text('no')
except OSError:
    pass
else:
    raise AssertionError('embedding grant became writable')
assert not Path({str(hidden / "secret.txt")!r}).exists()
Path({str(observed)!r}).write_text(json.dumps([str(root) for root in roots]))
"""

        async def scenario() -> None:
            launcher = BubblewrapLauncher(
                SandboxPolicy(ro_binds=(str(REPO_ROOT), str(state), str(embedding)))
            )
            spec = make_spec(
                tmp_path,
                command=(sys.executable, "-c", script),
                env={"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)},
            )
            process = await launcher.launch(spec)
            try:
                assert await asyncio.wait_for(process.wait(), timeout=15) == 0
                assert json.loads(observed.read_text()) == [str(embedding)]
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

        run_scenario(scenario())

    def test_secret_reaches_child_without_appearing_in_process_arguments(
        self, tmp_path: Path
    ) -> None:
        async def scenario() -> None:
            observed = tmp_path / "endpoint" / "observed.txt"
            script = (
                "import os, pathlib, time; "
                f"pathlib.Path({str(observed)!r}).write_text("
                "os.environ['DINKSTER_BOUNDARY_TOKEN'], encoding='utf-8'); "
                "time.sleep(30)"
            )
            process = await BubblewrapLauncher().launch(
                make_spec(tmp_path, command=(sys.executable, "-c", script))
            )
            try:
                async with asyncio.timeout(10):
                    while not observed.exists() or observed.stat().st_size == 0:
                        if process.returncode is not None:
                            raise AssertionError(f"sandbox child exited with {process.returncode}")
                        await asyncio.sleep(0.01)
                assert observed.read_text("utf-8") == "sekrit"
                command_line = Path(f"/proc/{process.pid}/cmdline").read_bytes()
                assert b"sekrit" not in command_line
            finally:
                process.terminate()
                await process.wait()

        run_scenario(scenario())

    def test_egress_uses_the_proxy_while_host_loopback_stays_hidden(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def scenario() -> None:
            async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                assert await reader.readexactly(4) == b"ping"
                writer.write(b"pong")
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_server(echo, "127.0.0.1", 0)
            port = int(server.sockets[0].getsockname()[1])

            async def test_resolver(host: str, resolved_port: int) -> tuple[str, ...]:
                assert host == "allowed.example.test"
                assert resolved_port == port
                return ("127.0.0.1",)

            monkeypatch.setattr(egress_module, "_resolve_public_addresses", test_resolver)
            observed = tmp_path / "endpoint" / "egress.txt"
            script = (
                "import os, pathlib, socket; "
                "direct = True; "
                "\ntry:\n"
                f" socket.create_connection(('127.0.0.1', {port}), timeout=0.2).close()\n"
                "except OSError:\n direct = False\n"
                "proxy = socket.socket(socket.AF_UNIX); "
                "proxy.settimeout(5); "
                "proxy.connect(os.environ['DINKSTER_EGRESS_PROXY']); "
                f"proxy.sendall(b'CONNECT allowed.example.test:{port} HTTP/1.1\\r\\n\\r\\n'); "
                "head = b''; "
                "\nwhile b'\\r\\n\\r\\n' not in head:\n head += proxy.recv(4096)\n"
                "assert head.startswith(b'HTTP/1.1 200 '); "
                "proxy.sendall(b'ping'); "
                "reply = proxy.recv(4); "
                f"pathlib.Path({str(observed)!r}).write_text(f'{{direct}}|{{reply.decode()}}')"
            )
            launcher = BubblewrapLauncher(
                SandboxPolicy(egress_allowlist=(f"https://allowed.example.test:{port}",))
            )
            process = await launcher.launch(
                make_spec(tmp_path, command=(sys.executable, "-c", script))
            )
            try:
                assert await asyncio.wait_for(process.wait(), timeout=15) == 0
                assert observed.read_text("utf-8") == "False|pong"
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                server.close()
                await server.wait_closed()

        run_scenario(scenario())

    def test_openai_transport_uses_unix_egress_with_loopback_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def scenario() -> None:
            upstream_bytes: list[bytes] = []

            async def capture(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                try:
                    upstream_bytes.append(await reader.read(4096))
                finally:
                    writer.close()
                    await writer.wait_closed()

            server = await asyncio.start_server(capture, "127.0.0.1", 0)
            port = int(server.sockets[0].getsockname()[1])

            async def test_resolver(host: str, resolved_port: int) -> tuple[str, ...]:
                assert (host, resolved_port) == ("allowed.example.test", port)
                return ("127.0.0.1",)

            monkeypatch.setattr(egress_module, "_resolve_public_addresses", test_resolver)
            observed = tmp_path / "endpoint" / "openai-egress.txt"
            script = f"""
import os
from pathlib import Path
from dinkster_inference_wire import (
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationStopConditions,
    OpenAIGenerationError,
    OpenAIGenerationProvider,
)

provider = OpenAIGenerationProvider(
    "https://allowed.example.test:{port}/v1",
    "test-model",
    timeout_s=2.0,
    proxy_socket=os.environ["DINKSTER_EGRESS_PROXY"],
)
request = GenerationRequest(
    provider.id,
    provider.model_identity,
    prompt="hello",
    sampler=GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),)),
    stop=GenerationStopConditions(1),
)
try:
    with provider.generate(request, cancelled=lambda: False) as stream:
        tuple(stream)
except OpenAIGenerationError:
    pass
else:
    raise AssertionError("test TLS peer unexpectedly completed generation")
finally:
    provider.close()
Path({str(observed)!r}).write_text("proxied", encoding="utf-8")
"""
            launcher = BubblewrapLauncher(
                SandboxPolicy(
                    egress_allowlist=(f"https://allowed.example.test:{port}",),
                    ro_binds=(str(REPO_ROOT),),
                )
            )
            process = await launcher.launch(
                make_spec(tmp_path, command=(sys.executable, "-c", script))
            )
            try:
                assert await asyncio.wait_for(process.wait(), timeout=15) == 0
                assert observed.read_text("utf-8") == "proxied"
                assert upstream_bytes and upstream_bytes[0].startswith(b"\x16\x03")
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                server.close()
                await server.wait_closed()

        run_scenario(scenario())

    @pytest.mark.parametrize("project_run", [False, True])
    def test_gateway_bearer_rotation_is_visible_without_oauth_access(
        self, tmp_path: Path, project_run: bool
    ) -> None:
        from dinkster.compose import PackSpec, compose_serving
        from dinkster.serve import _with_remote_config
        from dinkster.services import resolve_services

        home = tmp_path / "account-home"
        legacy = home / "gateway-tokens" / "install" / "access-token"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("legacy-bearer")
        token = (
            legacy.parent / "runs" / hashlib.sha256(b"credential-run").hexdigest()
            if project_run
            else legacy
        )
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text("first-bearer")
        secret = home / "accounts" / "install" / "session.json"
        secret.parent.mkdir(parents=True)
        secret.write_text('{"refresh_token":"must-stay-private"}')
        generation = tmp_path / "generation"
        pack_source = generation / "pack-sources" / "credential-probe"
        pack = pack_source / "credential_probe"
        pack.mkdir(parents=True)
        generation_secret = generation / "settings.json"
        generation_secret.write_text('{"private":"host-only"}')
        manifest = pack / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname="credential-probe"\npublisher="dinkster"\n'
            'namespaces=["credentialprobe"]\n[pack.entry]\nnodes="credential_probe:NODES"\n'
        )
        (pack / "__init__.py").write_text(
            "import hashlib, json, os\nfrom pathlib import Path\n"
            "from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, Output, TypeExpr\n"
            "from dinkster_api.v1 import current_execution_context\n"
            "class Probe(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            "        return NodeSchema(node_type='credentialprobe.read', idempotent=False, "
            "body=(Output('report', TypeExpr.concrete(CORE_STRING)),))\n"
            "    @classmethod\n"
            "    async def execute(cls):\n"
            "        if directory := os.environ.get('DINKSTER_REMOTE_RUN_TOKEN_DIRECTORY'):\n"
            "            run = current_execution_context().job_ref\n"
            "            token = Path(directory) / hashlib.sha256(run.encode()).hexdigest()\n"
            "        else:\n"
            "            token = Path(os.environ['DINKSTER_REMOTE_AUTH_TOKEN_FILE'])\n"
            "        try:\n"
            "            token.write_text('worker-must-not-write')\n"
            "        except OSError:\n"
            "            writable = False\n"
            "        else:\n"
            "            writable = True\n"
            f"        secret = Path({str(secret)!r})\n"
            f"        generation_secret = Path({str(generation_secret)!r})\n"
            f"        legacy_visible = {project_run!r} and Path({str(legacy)!r}).exists()\n"
            "        return cls.outputs(report=json.dumps([token.read_text() if token.exists() "
            "else '<missing>', secret.exists(), generation_secret.exists(), writable, "
            "legacy_visible]))\n"
            "NODES = (Probe,)\n"
        )
        spec = _with_remote_config(
            PackSpec(
                manifest,
                env={
                    "PYTHONPATH": str(pack_source),
                    "DINKSTER_REMOTE_AUTH_TOKEN_FILE": str(secret),
                    "DINKSTER_REMOTE_RUN_TOKEN_DIRECTORY": str(secret.parent),
                },
            ),
            services=resolve_services(),
            token_file=str(legacy),
            token_directory=legacy.parent,
            run_token_directory=token.parent if project_run else None,
        )

        async def scenario() -> None:
            policy = SandboxPolicy(
                ro_binds=(str(REPO_ROOT),), protected_roots=(str(home), str(generation))
            )
            if project_run:
                from dinkster.compose import CompositionError

                for directory in (None, token.parent.parent):
                    with pytest.raises(CompositionError, match="dedicated directory"):
                        await compose_serving(
                            (replace(spec, remote_token_directory=directory),),
                            include_default_packs=False,
                            sandbox_policy=policy,
                        )
            composition = await compose_serving(
                (spec,), include_default_packs=False, sandbox_policy=policy
            )
            try:
                engine = composition.make_engine(lambda event: None)
                graph = Graph(nodes={"read": GraphNode("credentialprobe.read", {})})
                first = await engine.run(graph, ["read"], run_id="credential-run")
                report = first.outputs["read"]["report"].resolve()
                assert isinstance(report, str)
                assert json.loads(report) == ["first-bearer", False, False, False, False]
                replacement = token.with_suffix(".new")
                replacement.write_text("second-bearer")
                replacement.replace(token)
                second = await engine.run(graph, ["read"], run_id="credential-run")
                report = second.outputs["read"]["report"].resolve()
                assert isinstance(report, str)
                assert json.loads(report) == ["second-bearer", False, False, False, False]
                token.unlink()
                revoked = await engine.run(graph, ["read"], run_id="credential-run")
                report = revoked.outputs["read"]["report"].resolve()
                assert isinstance(report, str)
                assert json.loads(report) == ["<missing>", False, False, False, False]
            finally:
                await composition.close()

        run_scenario(scenario())

    def test_isolated_worker_runs_inside_bwrap(self) -> None:
        """The full worker boundary - hello, schemas, an image graph - inside
        a real jail. The policy grants only the repo checkout (editable
        installs resolve into it); everything else is the launch's own
        necessities."""

        async def scenario() -> None:
            registry = TypeRegistry()
            register_core_types(registry)
            worker = IsolatedWorker(
                DEV_MANIFEST,
                registry,
                launcher=BubblewrapLauncher(SandboxPolicy(ro_binds=(str(REPO_ROOT),))),
            )
            await worker.start()
            try:
                engine = Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                )
                graph = Graph(
                    nodes={
                        "g": GraphNode("dev.image.gradient", {"width": 16, "height": 8}),
                        "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
                        "s": GraphNode("dev.image.stats", {"image": Link("i", "image")}),
                    }
                )
                result = await engine.run(graph, ["s"])
                mean = result.outputs["s"]["mean"].resolve()
                assert 0.0 < float(mean) < 1.0  # type: ignore[arg-type]
            finally:
                await worker.close()

        run_scenario(scenario())

    def test_dinkster_serve_sandbox_hides_host_secrets_and_loopback(self, tmp_path: Path) -> None:
        """The production flag reaches a real pack worker. Its declared vault
        and private scratch remain usable, while sibling library state,
        another pack's scratch and venv, and host loopback stay hidden."""
        from dinkster_graph import Graph, GraphNode, graph_to_wire

        library = tmp_path / "library"
        library.mkdir()
        auth = library / "auth.toml"
        auth.write_text("host-secret", encoding="utf-8")
        other_venv = library / "packs" / "other" / "venv" / "secret.txt"
        other_venv.parent.mkdir(parents=True)
        other_venv.write_text("other-pack-secret", encoding="utf-8")
        state = tmp_path / "engine-state"
        own_scratch = state / "scratch" / "packs" / "sandbox-probe"
        own_scratch_seed = own_scratch / "seed.txt"
        own_scratch_output = own_scratch / "worker-output.txt"
        other_scratch = state / "scratch" / "packs" / "other-pack" / "secret.txt"
        vault_file = library / "vault" / "sentinel.txt"
        vault_file.parent.mkdir()
        vault_file.write_text("allowed", encoding="utf-8")
        read_mount = tmp_path / "read-mount"
        read_mount.mkdir()
        read_mount_file = read_mount / "sentinel.txt"
        read_mount_file.write_text("mounted", encoding="utf-8")
        write_mount = tmp_path / "write-mount"
        write_mount.mkdir()
        write_mount_file = write_mount / "worker-output.txt"
        (library / "mounts.toml").write_text(
            "[mounts.read-test]\n"
            f"path = {json.dumps(str(read_mount))}\n\n"
            "[mounts.write-test]\n"
            f"path = {json.dumps(str(write_mount))}\n"
            'mode = "readwrite"\n',
            encoding="utf-8",
        )

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]

        pack = tmp_path / "probe-pack"
        pack.mkdir()
        (pack / "dinkster-pack.toml").write_text(
            '[pack]\nname = "sandbox-probe"\npublisher = "dinkster"\n'
            'namespaces = ["sandboxprobe"]\n'
            "[pack.sandbox]\nwritable-mounts = true\n"
            '[pack.entry]\nnodes = "probe_nodes:NODES"\n',
            encoding="utf-8",
        )
        principals = library / "principals.sqlite"
        principals.touch()
        (pack / "probe_nodes.py").write_text(
            "import json, os, resource, socket\n"
            "from pathlib import Path\n"
            "from collections.abc import Mapping\n"
            "from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, Output, TypeExpr\n"
            "STRING = TypeExpr.concrete(CORE_STRING)\n"
            "def readable(path):\n"
            "    try:\n"
            "        Path(path).read_bytes()\n"
            "    except OSError:\n"
            "        return False\n"
            "    return True\n"
            "def writeable(path):\n"
            "    try:\n"
            "        Path(path).write_text('sandbox-write', encoding='utf-8')\n"
            "    except OSError:\n"
            "        return False\n"
            "    return True\n"
            "class Inspect(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            "        return NodeSchema(node_type='sandboxprobe.inspect',  "
            "body=(Output('report', STRING),))\n"
            "    @classmethod\n"
            "    async def execute(cls):\n"
            "        scratch = Path(os.environ['DINKSTER_PACK_SCRATCH'])\n"
            "        try:\n"
            f"            with socket.create_connection(('127.0.0.1', {port}), timeout=0.2):\n"
            "                loopback = True\n"
            "        except OSError:\n"
            "            loopback = False\n"
            "        report = {\n"
            f"            'auth': readable({str(auth)!r}),\n"
            f"            'principals': readable({str(principals)!r}),\n"
            f"            'other_venv': readable({str(other_venv)!r}),\n"
            "            'scratch': ''.join('1' if result else '0' for result in (\n"
            "                readable(scratch / 'seed.txt'),\n"
            "                writeable(scratch / 'worker-output.txt'),\n"
            f"                readable({str(other_scratch)!r}),\n"
            "            )),\n"
            f"            'vault': readable({str(vault_file)!r}),\n"
            f"            'vault_write': writeable({str(vault_file)!r}),\n"
            f"            'mount_read': readable({str(read_mount_file)!r}),\n"
            f"            'mount_read_write': writeable({str(read_mount_file)!r}),\n"
            f"            'mount_write': writeable({str(write_mount_file)!r}),\n"
            "            'loopback': loopback,\n"
            "            'fsize_limit': resource.getrlimit(resource.RLIMIT_FSIZE)[0],\n"
            "            'process_limit': resource.getrlimit(resource.RLIMIT_NPROC)[0],\n"
            "        }\n"
            "        return cls.outputs(report=json.dumps(report, sort_keys=True))\n"
            "NODES = (Inspect,)\n",
            encoding="utf-8",
        )

        stderr_path = tmp_path / "serve.stderr"
        stderr_log = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dinkster.serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--library-root",
                str(library),
                "--engine-state-root",
                str(state),
                "--sandbox-packs",
                "--no-default-packs",
                "--pack",
                str(pack),
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "DINKSTER_SERVING_PYTHON": sys.executable,
                "PYTHONPATH": os.pathsep.join((str(REPO_ROOT), str(pack))),
            },
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            text=True,
        )

        async def scenario() -> None:
            base = f"http://127.0.0.1:{port}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with asyncio.timeout(60):
                    while True:
                        if process.poll() is not None:
                            raise worker_failure("serve died", stderr_path)
                        try:
                            async with session.get(base + "/api/composition") as response:
                                report = await response.json()
                            pack_state = report["packs"]["sandbox-probe"]
                            if pack_state["state"] == "failed":
                                raise worker_failure(pack_state["error"], stderr_path)
                            if pack_state["state"] == "announced":
                                break
                        except (aiohttp.ClientError, KeyError):
                            pass
                        await asyncio.sleep(0.05)

                own_scratch_seed.write_text("persistent", encoding="utf-8")
                other_scratch.parent.mkdir()
                other_scratch.write_text("other-scratch-secret", encoding="utf-8")
                job = {
                    "clientId": "sandbox-e2e",
                    "jobId": "probe",
                    "graph": graph_to_wire(
                        Graph(nodes={"probe": GraphNode("sandboxprobe.inspect", {})})
                    ),
                    "targets": ["probe"],
                }
                async with session.post(base + "/api/jobs", json=job) as response:
                    assert response.status == 202
                async with asyncio.timeout(30):
                    while True:
                        async with session.get(base + "/api/jobs/sandbox-e2e/probe") as response:
                            status = await response.json()
                        if status["state"] in ("completed", "failed"):
                            break
                        await asyncio.sleep(0.05)
                assert status["state"] == "completed", status.get("error")
                async with session.get(
                    base
                    + "/api/values?clientId=sandbox-e2e&jobId=probe&nodeId=probe&outputId=report"
                ) as response:
                    value = await response.json()
                observed = json.loads(value["descriptor"]["value"])
                assert 0 < observed.pop("fsize_limit") <= 64 * 1024**3
                assert 0 < observed.pop("process_limit") <= 4096
                assert observed == {
                    "auth": False,
                    "loopback": False,
                    "mount_read": True,
                    "mount_read_write": False,
                    "mount_write": True,
                    "other_venv": False,
                    "principals": False,
                    "scratch": "110",
                    "vault": True,
                    "vault_write": False,
                }
                assert write_mount_file.read_text("utf-8") == "sandbox-write"
                assert own_scratch_output.read_text("utf-8") == "sandbox-write"

        try:
            run_scenario(scenario())
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            stderr_log.close()


# -- probe jail (the doctor's import probe under bwrap) ------------------------


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX root semantics")
def test_probe_jail_refuses_dissolving_root(tmp_path: Path) -> None:
    from dinkster_workers import build_probe_bwrap_command

    with pytest.raises(SandboxError):
        build_probe_bwrap_command(sys.executable, "/home")
    with pytest.raises(SandboxError):
        build_probe_bwrap_command(sys.executable, tmp_path, source_roots=(Path("/home"),))
