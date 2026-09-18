"""Sandboxing is a launcher concern (DESIGN 3.11): bwrap argv construction
is pure and inspectable, capability detection classifies this machine
honestly, and a requested sandbox that cannot be built refuses loudly
instead of running the child unjailed. The live test runs the whole worker
boundary inside a real bwrap jail - and skips, with the remediation in the
skip reason, on machines where the kernel blocks unprivileged user
namespaces (Ubuntu 24.04's default posture)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiohttp
import dinkster_workers.egress as egress_module
import dinkster_workers.sandbox as sandbox_module
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import LOG_LEVEL_ENV, LOG_OVERRIDES_ENV
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
    load_manifest,
)
from dinkster_workers.accelerator import ACCELERATOR_ENV
from dinkster_workers.launch import SubprocessLauncher

REPO_ROOT = Path(__file__).parents[1]
DEV_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"

CAPABILITY = detect_bubblewrap() if sys.platform == "linux" else None


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


def test_plain_launcher_only_accepts_pack_scratch_from_launch_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", "/inherited/untrusted")
    absent_result = tmp_path / "absent.txt"
    trusted_result = tmp_path / "trusted.txt"
    script = (
        "import os, sys; from pathlib import Path; "
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
                env={"DINKSTER_PACK_SCRATCH": "/host-owned"},
            )
        )
        assert await trusted.wait() == 0

    asyncio.run(scenario())
    assert absent_result.read_text(encoding="utf-8") == "<absent>"
    assert trusted_result.read_text(encoding="utf-8") == "/host-owned"


def pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    """All (a, b) argument pairs following a two-argument bwrap flag."""
    return [(argv[i + 1], argv[i + 2]) for i, arg in enumerate(argv) if arg == flag]


class TestBuildBwrapCommand:
    def test_deny_by_default_shape(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy())
        assert argv[0] == "bwrap"
        assert "--die-with-parent" in argv
        assert "--new-session" in argv
        assert "--unshare-user" in argv
        assert "--unshare-pid" in argv
        # Network is always unshared; granted egress uses a Unix socket proxy.
        assert "--unshare-net" in argv
        # Private tmpfs /tmp, fresh /proc and /dev.
        assert argv[argv.index("--tmpfs") + 1] == "/tmp"
        assert argv[argv.index("--proc") + 1] == "/proc"
        assert argv[argv.index("--dev") + 1] == "/dev"
        # The rlimit stub becomes the original child command.
        assert tuple(argv[-2:]) == ("-m", "dinkster_workers.host")
        stub = argv[-3]
        assert "RLIMIT_FSIZE" in stub
        assert "RLIMIT_NPROC" in stub

    def test_degraded_mode_omits_user_namespaces(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy(), user_namespaces=False)
        assert "--unshare-user" not in argv
        assert "--unshare-pid" not in argv
        assert "--unshare-net" in argv  # network isolation is orthogonal

    def test_egress_grant_keeps_network_unshared(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(
            make_spec(tmp_path),
            SandboxPolicy(egress_allowlist=("https://api.example.test",)),
        )
        assert "--unshare-net" in argv

    def test_environment_uses_private_file_descriptor_never_argv(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy(), env_fd=42)
        assert "--clearenv" in argv
        assert "--setenv" not in argv
        assert "sekrit" not in argv
        file_index = argv.index("--file")
        assert argv[file_index + 1 : file_index + 3] == [
            "42",
            "/run/dinkster/worker-env.json",
        ]
        stub = argv[-3]
        assert "json.load(_env_file)" in stub
        assert "os.unlink('/run/dinkster/worker-env.json')" in stub

    def test_relative_interpreter_is_resolved_before_environment_clear(
        self, tmp_path: Path
    ) -> None:
        argv = build_bwrap_command(
            make_spec(tmp_path, python="python3", command=("python3", "-V")),
            SandboxPolicy(),
        )
        assert argv[-4] == os.path.abspath(shutil.which("python3") or "python3")

    def test_endpoint_dir_is_writable_and_pack_root_read_only(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path)
        argv = build_bwrap_command(spec, SandboxPolicy())
        assert (str(spec.endpoint_dir), str(spec.endpoint_dir)) in pairs(argv, "--bind")
        assert (str(spec.pack_root), str(spec.pack_root)) in pairs(argv, "--ro-bind")

    def test_shm_bind_follows_use_shm(self, tmp_path: Path) -> None:
        with_shm = build_bwrap_command(make_spec(tmp_path), SandboxPolicy())
        without = build_bwrap_command(make_spec(tmp_path, use_shm=False), SandboxPolicy())
        if os.path.exists("/dev/shm"):
            assert ("/dev/shm", "/dev/shm") in pairs(with_shm, "--bind")
        assert ("/dev/shm", "/dev/shm") not in pairs(without, "--bind")

    def test_gpu_denied_by_default(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy())
        assert ("/sys", "/sys") not in pairs(argv, "--ro-bind")
        assert "--dev-bind" not in argv

    def test_gpu_grant_binds_devices(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(make_spec(tmp_path), SandboxPolicy(gpu=True))
        assert ("/sys", "/sys") in pairs(argv, "--ro-bind")
        if any(Path("/dev").glob("nvidia*")):
            assert "--dev-bind" in argv

    def test_pythonpath_entries_are_bound(self, tmp_path: Path) -> None:
        extra = tmp_path / "extra-code"
        extra.mkdir()
        spec = make_spec(tmp_path, env={"PYTHONPATH": str(extra), "DINKSTER_BOUNDARY_TOKEN": "s"})
        argv = build_bwrap_command(spec, SandboxPolicy())
        assert (str(extra), str(extra)) in pairs(argv, "--ro-bind")

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX root semantics")
    def test_pythonpath_cannot_dissolve_the_jail(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path, env={"PYTHONPATH": "/", "DINKSTER_BOUNDARY_TOKEN": "s"})
        with pytest.raises(SandboxError, match="PYTHONPATH"):
            build_bwrap_command(spec, SandboxPolicy())

    def test_worker_rlimits_can_be_overridden_or_disabled(self, tmp_path: Path) -> None:
        argv = build_bwrap_command(
            make_spec(tmp_path),
            SandboxPolicy(max_file_size_bytes=1234, max_processes=None),
        )
        stub = argv[-3]
        assert "_limit = 1234" in stub
        assert "min(_limit, _soft)" in stub
        assert "min(_limit, _hard)" in stub
        assert "setrlimit(resource.RLIMIT_FSIZE, (_limit, _limit))" in stub
        assert "RLIMIT_NPROC" not in stub

        unlimited = build_bwrap_command(
            make_spec(tmp_path),
            SandboxPolicy(max_file_size_bytes=None, max_processes=None),
        )
        assert tuple(unlimited[-2:]) == ("-m", "dinkster_workers.host")
        assert "RLIMIT_FSIZE" not in unlimited[-3]
        assert "RLIMIT_NPROC" not in unlimited[-3]

    def test_invalid_worker_rlimit_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(SandboxError, match="max_file_size_bytes"):
            build_bwrap_command(make_spec(tmp_path), SandboxPolicy(max_file_size_bytes=0))
        with pytest.raises(SandboxError, match="max_processes"):
            build_bwrap_command(make_spec(tmp_path), SandboxPolicy(max_processes=0))

    def test_windows_does_not_probe_interpreter_reparse_points(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def unsupported_reparse_point(_path: Path) -> str:
            raise ValueError("unsupported reparse tag")

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "readlink", unsupported_reparse_point)
        assert (
            sandbox_module._next_symlink_target(tmp_path / "python.exe")  # pyright: ignore[reportPrivateUsage]
            is None
        )

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
    def test_absolute_interpreter_symlink_chain_is_bound(self, tmp_path: Path) -> None:
        uv_python_root = tmp_path / "uv" / "python"
        resolved_prefix = uv_python_root / "cpython-3.13.14-linux-x86_64-gnu"
        resolved_bin = resolved_prefix / "bin"
        resolved_bin.mkdir(parents=True)
        executable = resolved_bin / "python3.13"
        executable.touch()

        alias_prefix = uv_python_root / "cpython-3.13-linux-x86_64-gnu"
        alias_prefix.symlink_to(resolved_prefix)
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin" / "python"
        python.symlink_to(alias_prefix / "bin" / "python3.13")

        argv = build_bwrap_command(
            make_spec(tmp_path, python=str(python), command=(str(python), "-V")),
            SandboxPolicy(),
        )
        read_only = pairs(argv, "--ro-bind")
        assert (str(venv), str(venv)) in read_only
        assert (str(alias_prefix), str(alias_prefix)) in read_only
        assert (str(resolved_prefix), str(resolved_prefix)) in read_only

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

    def test_environment_is_allowlisted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
        monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
        monkeypatch.setenv(LOG_OVERRIDES_ENV, "dinkster.pack.probe=warning")
        monkeypatch.setenv(ACCELERATOR_ENV, "xpu")
        environment = sandbox_module._launch_environment(  # pyright: ignore[reportPrivateUsage]
            make_spec(tmp_path), SandboxPolicy()
        )
        assert environment["HOME"] == "/tmp"
        assert environment["TMPDIR"] == "/tmp"
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["DINKSTER_BOUNDARY_TOKEN"] == "sekrit"
        assert environment[LOG_LEVEL_ENV] == "debug"
        assert environment[LOG_OVERRIDES_ENV] == "dinkster.pack.probe=warning"
        # The accelerator pin travels without a GPU grant: explicit CPU or
        # device selection must survive into every sandboxed worker.
        assert environment[ACCELERATOR_ENV] == "xpu"
        # GPU env only travels with a GPU grant.
        assert "CUDA_VISIBLE_DEVICES" not in environment
        gpu_environment = sandbox_module._launch_environment(  # pyright: ignore[reportPrivateUsage]
            make_spec(tmp_path), SandboxPolicy(gpu=True)
        )
        assert gpu_environment["CUDA_VISIBLE_DEVICES"] == "7"


class TestDetection:
    def test_detection_is_honest_about_this_machine(self) -> None:
        capability = detect_bubblewrap()
        if sys.platform != "linux":
            assert not capability.available
            assert "Linux" in capability.detail
        elif not capability.available:
            # Unavailable must come with a remediation, not a shrug.
            assert capability.detail
        else:
            assert capability.bwrap is not None

    def test_unavailable_capability_refuses_loudly(self, tmp_path: Path) -> None:
        launcher = BubblewrapLauncher(
            capability=BubblewrapCapability(
                available=False, bwrap=None, user_namespaces=False, detail="blocked"
            )
        )

        async def scenario() -> None:
            with pytest.raises(SandboxUnavailable, match="blocked"):
                await launcher.launch(make_spec(tmp_path))

        asyncio.run(scenario())

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

        asyncio.run(scenario())

    def test_bubblewrap_process_inherits_no_parent_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: dict[str, Any] = {}
        sentinel = object()

        async def fake_create_subprocess_exec(*argv: str, **kwargs: object) -> object:
            observed["argv"] = argv
            observed.update(kwargs)
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple)
            env_fd = pass_fds[0]
            assert isinstance(env_fd, int)
            with os.fdopen(os.dup(env_fd), "rb") as environment_file:
                observed["worker_env"] = json.load(environment_file)
            return sentinel

        monkeypatch.setenv("DINKSTER_PACK_SCRATCH", "/inherited/untrusted")
        monkeypatch.setenv("DINKSTER_EGRESS_PROXY", "/inherited/proxy.sock")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        launcher = BubblewrapLauncher(
            policy=SandboxPolicy(
                env_passthrough=("DINKSTER_EGRESS_PROXY", "DINKSTER_PACK_SCRATCH")
            ),
            capability=BubblewrapCapability(
                available=True,
                bwrap="/usr/bin/bwrap",
                user_namespaces=True,
                detail="",
            ),
        )

        async def scenario() -> None:
            process = await launcher.launch(make_spec(tmp_path))
            assert process is sentinel

        asyncio.run(scenario())
        assert observed["env"] == {}
        assert "sekrit" not in observed["argv"]
        assert "DINKSTER_PACK_SCRATCH" not in observed["worker_env"]
        assert "DINKSTER_EGRESS_PROXY" not in observed["worker_env"]


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_egress_proxy_lives_with_the_sandbox_process(
    tmp_path: Path,
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.exited = asyncio.Event()

        async def wait(self) -> int:
            await self.exited.wait()
            return 0

    async def scenario() -> None:
        process = FakeProcess()

        async def fake_create_subprocess_exec(*argv: str, **kwargs: object) -> object:
            observed["argv"] = argv
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple)
            env_fd = pass_fds[0]
            assert isinstance(env_fd, int)
            with os.fdopen(os.dup(env_fd), "rb") as environment_file:
                observed["worker_env"] = json.load(environment_file)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        launcher = BubblewrapLauncher(
            policy=SandboxPolicy(egress_allowlist=("https://api.example.test",)),
            capability=BubblewrapCapability(True, "/usr/bin/bwrap", True, ""),
        )
        launched = await launcher.launch(make_spec(tmp_path, endpoint_dir=unix_socket_dir))
        assert launched is process
        socket_path = unix_socket_dir / "egress.sock"
        assert socket_path.is_socket()
        assert "--unshare-net" in observed["argv"]
        assert observed["worker_env"]["DINKSTER_EGRESS_PROXY"] == str(socket_path)
        process.exited.set()
        for _ in range(100):
            if not socket_path.exists():
                break
            await asyncio.sleep(0.01)
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_serving_policy_derives_pack_grants_and_proxied_egress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    vault = tmp_path / "vault"
    vault.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    writable = tmp_path / "writable"
    writable.mkdir()
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "policy-pack"\nnamespaces = ["policy"]\n'
        "[pack.sandbox]\ngpu = true\nnetwork = true\nwritable-mounts = true\n"
        '[pack.entry]\nnodes = "policy_pack:NODES"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dinkster.compose.detect_bubblewrap",
        lambda: BubblewrapCapability(True, "/usr/bin/bwrap", True, ""),
    )
    composer = ServingComposer(
        sandbox_policy=SandboxPolicy(ro_binds=(str(models), str(writable))),
        sandbox_writable_mounts=(str(writable),),
    )
    loaded = load_manifest(manifest_path)
    spec = PackSpec(
        manifest_path,
        env={
            "DINKSTER_ASSET_VAULT": str(vault),
            "DINKSTER_COMFY_API_BASE": "https://api.example.test",
            "DINKSTER_OPENAI_BASE_URL": "https://llm.example.test/v1",
        },
        aimdo="auto",
    )
    launcher = composer._sandbox_launcher(spec, (loaded,), spec.env)
    try:
        assert launcher is not None
        assert launcher.policy.egress_allowlist == (
            "https://api.example.test",
            "https://llm.example.test",
        )
        assert launcher.policy.gpu is True
        assert set(launcher.policy.ro_binds) >= {
            str(models),
            str(vault),
            str(manifest_path.parent),
        }
        assert str(writable) not in launcher.policy.ro_binds
        assert str(writable) in launcher.policy.rw_binds
    finally:
        asyncio.run(composer.close())


def test_serving_policy_requires_both_manifest_requests_and_host_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster.compose import CompositionError, PackSpec, ServingComposer

    def manifest(name: str, sandbox: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        path = root / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n'
            f"[pack.sandbox]\n{sandbox}"
            f'[pack.entry]\nnodes = "{name.replace("-", "_")}:NODES"\n',
            encoding="utf-8",
        )
        return path

    gpu_path = manifest("gpu-pack", "gpu = true\n")
    network_path = manifest("network-pack", "network = true\n")
    network_two_path = manifest("network-two", "network = true\n")
    plain_path = manifest("plain-pack", "")
    gpu = load_manifest(gpu_path)
    network = load_manifest(network_path)
    network_two = load_manifest(network_two_path)
    plain = load_manifest(plain_path)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    denied = ServingComposer(sandbox_policy=SandboxPolicy())
    with pytest.raises(CompositionError, match="GPU access requested but not granted"):
        denied._sandbox_launcher(PackSpec(gpu_path), (gpu,), {})
    with pytest.raises(CompositionError, match="network access requested but not granted"):
        denied._sandbox_launcher(PackSpec(network_path), (network,), {})

    partial = ServingComposer(
        sandbox_policy=SandboxPolicy(),
        sandbox_network_grants={"network-pack": ("https://api.example.test",)},
    )
    with pytest.raises(CompositionError, match="network-two"):
        partial._sandbox_launcher(
            PackSpec(network_path),
            (network, network_two),
            {},
        )

    granted = ServingComposer(
        sandbox_policy=SandboxPolicy(),
        sandbox_gpu_grants=("gpu_pack",),
        sandbox_network_grants={
            "network.pack": (
                "https://api.example.test",
                "https://storage.example.test:8443",
            ),
            "network_pack": ("https://alias.example.test",),
            "network-two": ("https://other.example.test",),
        },
    )
    combined = granted._sandbox_launcher(
        PackSpec(gpu_path),
        (gpu, network, network_two),
        {},
    )
    assert combined is not None
    assert combined.policy.gpu
    assert combined.policy.egress_allowlist == (
        "https://api.example.test",
        "https://storage.example.test:8443",
        "https://alias.example.test",
        "https://other.example.test",
    )

    staged = granted.spawn_empty()
    staged_launcher = staged._sandbox_launcher(
        PackSpec(gpu_path),
        (gpu, network, network_two),
        {},
    )
    assert staged_launcher is not None
    assert staged_launcher.policy.gpu
    assert staged_launcher.policy.egress_allowlist == combined.policy.egress_allowlist

    host_only = ServingComposer(
        sandbox_policy=SandboxPolicy(
            gpu=True,
            egress_allowlist=("https://global.example.test",),
        )
    )
    plain_launcher = host_only._sandbox_launcher(PackSpec(plain_path), (plain,), {})
    assert plain_launcher is not None
    assert not plain_launcher.policy.gpu
    assert not plain_launcher.policy.egress_allowlist


def test_host_readwrite_mount_stays_read_only_without_manifest_request(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    writable = tmp_path / "host-write"
    writable.mkdir()
    manifest_path = tmp_path / "plain" / "dinkster-pack.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        '[pack]\nname = "plain"\n[pack.sandbox]\n[pack.entry]\nnodes = "plain:NODES"\n',
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    composer = ServingComposer(
        sandbox_policy=SandboxPolicy(ro_binds=(str(writable),)),
        sandbox_writable_mounts=(str(writable),),
    )

    launcher = composer._sandbox_launcher(PackSpec(manifest_path), (manifest,), {})

    assert launcher is not None
    assert str(writable) in launcher.policy.ro_binds
    assert str(writable) not in launcher.policy.rw_binds


def test_remote_pack_gets_scoped_vault_write_token_read_and_gateway_egress(
    tmp_path: Path,
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    manifest_path = REPO_ROOT / "packages/dinkster-nodes-remote/dinkster-pack.toml"
    manifest = load_manifest(manifest_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    token = tmp_path / "remote-token"
    token.write_text("secret", encoding="utf-8")
    env = {
        "DINKSTER_ASSET_VAULT": str(vault),
        "DINKSTER_REMOTE_AUTH_TOKEN_FILE": str(token),
        "DINKSTER_REMOTE_CATALOG_BASE": "https://catalog.example.test",
        "DINKSTER_REMOTE_GATEWAY_BASE": "https://gateway.example.test/v1",
    }
    composer = ServingComposer(
        sandbox_policy=SandboxPolicy(
            ro_binds=(str(vault),),
            protected_roots=(str(tmp_path),),
            protected_ro_exceptions=(str(vault),),
        )
    )
    spec = PackSpec(
        manifest_path,
        trust_reserved=True,
        env=env,
        asset_vault_write=True,
    )

    launcher = composer._sandbox_launcher(spec, (manifest,), env)

    assert launcher is not None
    assert str(vault) not in launcher.policy.ro_binds
    assert str(vault) in launcher.policy.rw_binds
    assert str(vault) in launcher.policy.protected_rw_exceptions
    assert str(token) in launcher.policy.ro_binds
    assert str(token) in launcher.policy.protected_ro_exceptions
    assert str(token) not in launcher.policy.rw_binds
    assert launcher.policy.egress_allowlist == (
        "https://catalog.example.test",
        "https://gateway.example.test",
    )
    with pytest.raises(ValueError, match="in-process"):
        PackSpec(manifest_path, in_process=True, asset_vault_write=True)


def test_pack_scratch_reaches_sandboxed_and_unsandboxed_workers(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    library = tmp_path / "library"
    scratch_root = library / "scratch"
    expected = scratch_root / "packs" / "dinkster-nodes-dev"
    manifest = load_manifest(DEV_MANIFEST)
    spec = PackSpec(DEV_MANIFEST, env={"DINKSTER_PACK_SCRATCH": "/untrusted"})

    sandboxed = ServingComposer(
        sandbox_policy=SandboxPolicy(protected_roots=(str(library),)),
        pack_scratch_root=scratch_root,
    )
    sandboxed_worker = sandboxed._isolated_worker(  # pyright: ignore[reportPrivateUsage]
        spec,
        manifest,
        TypeRegistry(),
    )
    assert (  # pyright: ignore[reportPrivateUsage]
        sandboxed_worker._extra_env["DINKSTER_PACK_SCRATCH"] == str(expected)
    )
    launcher = sandboxed_worker._launcher  # pyright: ignore[reportPrivateUsage]
    assert isinstance(launcher, BubblewrapLauncher)
    assert str(expected) in launcher.policy.rw_binds
    assert str(expected) in launcher.policy.protected_rw_exceptions
    assert str(library) not in launcher.policy.rw_binds

    unsandboxed = ServingComposer(pack_scratch_root=scratch_root)
    unsandboxed_worker = unsandboxed._isolated_worker(  # pyright: ignore[reportPrivateUsage]
        spec,
        manifest,
        TypeRegistry(),
    )
    assert (  # pyright: ignore[reportPrivateUsage]
        unsandboxed_worker._extra_env["DINKSTER_PACK_SCRATCH"] == str(expected)
    )
    assert expected.is_dir()
    if os.name == "posix":
        assert expected.stat().st_mode & 0o777 == 0o700
    asyncio.run(sandboxed.close())
    asyncio.run(unsandboxed.close())


def test_pack_scratch_is_scoped_for_groups_and_staging(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    def make_manifest(name: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        path = root / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n'
            f'[pack.entry]\nnodes = "{name}:NODES"\n',
            encoding="utf-8",
        )
        return path

    alpha_path = make_manifest("alpha")
    beta_path = make_manifest("beta")
    paths = (alpha_path, beta_path)
    manifests = tuple(load_manifest(path) for path in paths)
    scratch_root = tmp_path / "library" / "scratch"
    composer = ServingComposer(pack_scratch_root=scratch_root)
    group_spec = PackSpec(
        alpha_path,
        worker_group="Models",
        group_manifests=paths,
    )
    group_env = composer._worker_environment(  # pyright: ignore[reportPrivateUsage]
        group_spec,
        manifests,
    )
    solo_env = composer._worker_environment(  # pyright: ignore[reportPrivateUsage]
        PackSpec(alpha_path),
        (manifests[0],),
    )
    assert group_env["DINKSTER_PACK_SCRATCH"] == str(scratch_root / "groups" / "Models")
    assert solo_env["DINKSTER_PACK_SCRATCH"] == str(scratch_root / "packs" / "alpha")

    staged = composer.spawn_empty()
    staged_env = staged._worker_environment(  # pyright: ignore[reportPrivateUsage]
        group_spec,
        manifests,
    )
    assert staged_env["DINKSTER_PACK_SCRATCH"] == group_env["DINKSTER_PACK_SCRATCH"]
    asyncio.run(staged.close())
    asyncio.run(composer.close())


def test_pack_scratch_refuses_symlink_and_unsafe_group_name(tmp_path: Path) -> None:
    from dinkster.compose import CompositionError, PackSpec, ServingComposer

    manifest = load_manifest(DEV_MANIFEST)
    scratch_root = tmp_path / "library" / "scratch"
    composer = ServingComposer(pack_scratch_root=scratch_root)
    if os.name == "posix":
        target = tmp_path / "outside"
        target.mkdir()
        scratch = scratch_root / "packs" / "dinkster-nodes-dev"
        scratch.parent.mkdir(parents=True)
        scratch.symlink_to(target, target_is_directory=True)
        with pytest.raises(CompositionError, match="not a real directory"):
            composer._worker_environment(  # pyright: ignore[reportPrivateUsage]
                PackSpec(DEV_MANIFEST),
                (manifest,),
            )
    with pytest.raises(CompositionError, match="not path-safe"):
        composer._worker_environment(  # pyright: ignore[reportPrivateUsage]
            PackSpec(
                DEV_MANIFEST,
                worker_group="../escape",
                group_manifests=(DEV_MANIFEST,),
            ),
            (manifest,),
        )
    asyncio.run(composer.close())


def test_pack_scratch_is_absent_without_a_library_root() -> None:
    from dinkster.compose import PackSpec, ServingComposer

    manifest = load_manifest(DEV_MANIFEST)
    composer = ServingComposer(worker_env={"DINKSTER_PACK_SCRATCH": "/untrusted"})
    environment = composer._worker_environment(  # pyright: ignore[reportPrivateUsage]
        PackSpec(DEV_MANIFEST),
        (manifest,),
    )
    assert "DINKSTER_PACK_SCRATCH" not in environment
    asyncio.run(composer.close())


def test_single_job_sandbox_lanes_share_writable_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster.compose import PackSpec, ServingComposer, _SingleJobWorkerPool

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "dinkster.compose.detect_bubblewrap",
        lambda: BubblewrapCapability(True, "/usr/bin/bwrap", True, ""),
    )
    composer = ServingComposer(sandbox_policy=SandboxPolicy())
    manifest = load_manifest(DEV_MANIFEST)
    pool = composer._isolated_worker(  # pyright: ignore[reportPrivateUsage]
        PackSpec(DEV_MANIFEST, single_job_cuda_indices=(0, 1)),
        manifest,
        TypeRegistry(),
    )
    assert isinstance(pool, _SingleJobWorkerPool)
    rendezvous_values = {
        lane.worker._extra_env["DINKSTER_SINGLE_JOB_RENDEZVOUS"]  # pyright: ignore[reportPrivateUsage]
        for lane in pool.lanes
    }
    assert len(rendezvous_values) == 1
    rendezvous = Path(rendezvous_values.pop().removeprefix("file://"))
    assert rendezvous.parent.is_dir()
    for lane in pool.lanes:
        launcher = lane.worker._launcher  # pyright: ignore[reportPrivateUsage]
        assert isinstance(launcher, BubblewrapLauncher)
        assert str(rendezvous.parent) in launcher.policy.rw_binds
    asyncio.run(pool.close())
    assert not rendezvous.parent.exists()


@pytest.mark.parametrize("single_job", [False, True])
def test_multi_gpu_sandbox_lanes_share_pack_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_job: bool,
) -> None:
    from dinkster.compose import PackSpec, ServingComposer, _ReplicaWorkerPool

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "dinkster.compose.detect_bubblewrap",
        lambda: BubblewrapCapability(True, "/usr/bin/bwrap", True, ""),
    )
    scratch = tmp_path / "library" / "scratch"
    composer = ServingComposer(
        sandbox_policy=SandboxPolicy(protected_roots=(str(tmp_path / "library"),)),
        pack_scratch_root=scratch,
    )
    pool = composer._isolated_worker(  # pyright: ignore[reportPrivateUsage]
        PackSpec(
            DEV_MANIFEST,
            replica_cuda_indices=() if single_job else (0, 1),
            single_job_cuda_indices=(0, 1) if single_job else (),
        ),
        load_manifest(DEV_MANIFEST),
        TypeRegistry(),
    )
    assert isinstance(pool, _ReplicaWorkerPool)
    expected = str(scratch / "packs" / "dinkster-nodes-dev")
    for lane in pool.lanes:
        assert (  # pyright: ignore[reportPrivateUsage]
            lane.worker._extra_env["DINKSTER_PACK_SCRATCH"] == expected
        )
        launcher = lane.worker._launcher  # pyright: ignore[reportPrivateUsage]
        assert isinstance(launcher, BubblewrapLauncher)
        assert expected in launcher.policy.rw_binds
    asyncio.run(pool.close())
    asyncio.run(composer.close())


@pytest.mark.skipif(
    CAPABILITY is None or not CAPABILITY.available,
    reason=(
        "bubblewrap jail cannot be built here: "
        + (CAPABILITY.detail if CAPABILITY is not None else "not Linux")
    ),
)
class TestLiveJail:
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

        asyncio.run(scenario())

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

        asyncio.run(scenario())

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
from dinkster_inference import (
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

        asyncio.run(scenario())

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

        asyncio.run(scenario())

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
        own_scratch = library / "scratch" / "packs" / "sandbox-probe"
        own_scratch.mkdir(parents=True)
        own_scratch_seed = own_scratch / "seed.txt"
        own_scratch_seed.write_text("persistent", encoding="utf-8")
        own_scratch_output = own_scratch / "worker-output.txt"
        other_scratch = library / "scratch" / "packs" / "other-pack" / "secret.txt"
        other_scratch.parent.mkdir()
        other_scratch.write_text("other-scratch-secret", encoding="utf-8")
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
            '[pack]\nname = "sandbox-probe"\nnamespaces = ["sandboxprobe"]\n'
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
            "from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, OutputSpec, TypeExpr\n"
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
            "        return NodeSchema(node_type='sandboxprobe.inspect', inputs=(), "
            "outputs=(OutputSpec('report', STRING),))\n"
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
            stderr=subprocess.PIPE,
            text=True,
        )

        async def scenario() -> None:
            base = f"http://127.0.0.1:{port}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with asyncio.timeout(60):
                    while True:
                        if process.poll() is not None:
                            error = process.stderr.read() if process.stderr else "serve died"
                            raise AssertionError(error)
                        try:
                            async with session.get(base + "/api/composition") as response:
                                report = await response.json()
                            pack_state = report["packs"]["sandbox-probe"]
                            if pack_state["state"] == "failed":
                                raise AssertionError(pack_state["error"])
                            if pack_state["state"] == "announced":
                                break
                        except (aiohttp.ClientError, KeyError):
                            pass
                        await asyncio.sleep(0.05)

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
            asyncio.run(scenario())
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


# -- probe jail (the doctor's import probe under bwrap) ------------------------


def _ro_bound(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "--ro-bind"]


def test_probe_jail_command_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe jail is stricter than the worker jail: network ALWAYS
    unshared (no policy knob), nothing writable but the private tmpfs,
    environment cleared."""
    from dinkster_workers import build_probe_bwrap_command

    secret = "must-not-appear-in-probe-argv"
    monkeypatch.setenv(LOG_LEVEL_ENV, secret)
    argv = build_probe_bwrap_command(sys.executable, tmp_path)
    assert "--unshare-net" in argv
    assert "--die-with-parent" in argv
    assert "--clearenv" in argv
    assert "--setenv" not in argv
    assert secret not in argv
    assert pairs(argv, "--file") == [("3", "/run/dinkster/probe-env.json")]
    assert "--bind" not in argv  # no writable bind anywhere
    tmpfs = argv.index("--tmpfs")
    assert argv[tmpfs : tmpfs + 2] == ["--tmpfs", "/tmp"]
    assert str(tmp_path) in _ro_bound(argv)


def test_default_probe_environment_preserves_the_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    monkeypatch.setenv("DINKSTER_UNRELATED_HOST_SENTINEL", "must-not-leak")
    environment = sandbox_module.probe_environment(None)
    assert environment[LOG_LEVEL_ENV] == "debug"
    assert environment["HOME"] == "/tmp"
    assert environment["TMPDIR"] == "/tmp"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "DINKSTER_UNRELATED_HOST_SENTINEL" not in environment


def test_probe_jail_binds_the_import_surface(tmp_path: Path) -> None:
    """Every existing absolute sys.path entry of the parent is readable in
    the jail (bound directly or under a bound prefix) - editable installs
    and site-packages alike, so the probe child can import exactly what
    its parent can."""
    from dinkster_workers import build_probe_bwrap_command

    bound = _ro_bound(build_probe_bwrap_command(sys.executable, tmp_path))
    for entry in sys.path:
        if not entry or not os.path.isabs(entry) or not os.path.exists(entry):
            continue
        normalized = os.path.normpath(entry)
        assert any(normalized == b or normalized.startswith(b + os.sep) for b in bound), (
            f"sys.path entry {normalized} is not readable inside the probe jail"
        )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX root semantics")
def test_probe_jail_refuses_dissolving_root(tmp_path: Path) -> None:
    from dinkster_workers import build_probe_bwrap_command

    with pytest.raises(SandboxError):
        build_probe_bwrap_command(sys.executable, "/home")


def test_probe_jail_wrap_applies_environment_and_rlimits_via_bootstrap(tmp_path: Path) -> None:
    """wrap() prefixes bwrap and inserts the bootstrap between the
    interpreter and probe module, without a thread-unsafe preexec_fn."""
    from dinkster_workers import ProbeJail

    jail = ProbeJail(bwrap="bwrap", user_namespaces=True)
    argv = [sys.executable, "-m", "dinkster_workers._doctor_probe", "x"]
    wrapped = jail.wrap(argv, tmp_path)
    assert wrapped[0] == "bwrap"
    assert wrapped[-3:] == ["-m", "dinkster_workers._doctor_probe", "x"]
    assert wrapped[-5] == "-c"
    assert "setrlimit" in wrapped[-4] and "RLIMIT_CPU" in wrapped[-4]
    assert "from dinkster_workers import _doctor_probe as _probe" in wrapped[-4]
    assert wrapped[-4].index("_doctor_probe as _probe") < wrapped[-4].index("sys.path[1:1]")
    assert "_probe.main()" in wrapped[-4]
    assert "os.environ.clear()" in wrapped[-4]
    assert "os.unlink('/run/dinkster/probe-env.json')" in wrapped[-4]
    assert wrapped[-6] == sys.executable


def test_detect_probe_jail_refuses_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested probe jail never degrades: unavailable bwrap (or a
    non-Linux platform) raises with the remediation."""
    from dinkster_workers import detect_probe_jail

    if sys.platform == "linux":
        monkeypatch.setattr(
            "dinkster_workers.sandbox.detect_bubblewrap",
            lambda: BubblewrapCapability(
                available=False, bwrap=None, user_namespaces=False, detail="no bwrap"
            ),
        )
    with pytest.raises(SandboxUnavailable):
        detect_probe_jail()
