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
import socket
import subprocess
import sys
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

    run_scenario(scenario())
    assert absent_result.read_text(encoding="utf-8") == "<absent>"
    assert trusted_result.read_text(encoding="utf-8") == "/host-owned"


def pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    """All (a, b) argument pairs following a two-argument bwrap flag."""
    return [(argv[i + 1], argv[i + 2]) for i, arg in enumerate(argv) if arg == flag]


class TestBuildBwrapCommand:
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
