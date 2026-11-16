"""Native Seatbelt boundaries, transport and doctor behavior."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from dinkster_workers.launch import LaunchSpec
from dinkster_workers.sandbox import SandboxError, SandboxPolicy, SandboxUnavailable
from dinkster_workers.seatbelt import (
    SeatbeltCapability,
    SeatbeltLauncher,
    build_seatbelt_profile,
    detect_seatbelt,
)


def make_spec(tmp_path: Path, script: str) -> LaunchSpec:
    endpoint = tmp_path / "endpoint"
    endpoint.mkdir(exist_ok=True)
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    return LaunchSpec((sys.executable, "-c", script), {}, endpoint, pack, sys.executable, True)


@pytest.mark.skipif(os.name == "nt", reason="Seatbelt profiles use POSIX paths")
@pytest.mark.parametrize("grant", ["/Library", "/private", "/Users", str(Path.home())])
def test_broad_grants_refuse(tmp_path: Path, grant: str) -> None:
    spec = make_spec(tmp_path, "pass")
    with pytest.raises(SandboxError, match="dissolve"):
        build_seatbelt_profile(
            spec, SandboxPolicy(rw_binds=(grant,)), scratch=spec.endpoint_dir / "tmp"
        )


@pytest.mark.skipif(os.name == "nt", reason="Seatbelt profiles use POSIX paths")
def test_interpreter_outside_macos_system_paths_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "/opt/dinkster-test-interpreter"
    exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda path: str(path) == prefix or exists(path))
    spec = make_spec(tmp_path, "pass")
    python = prefix + "/bin/python3.14"
    spec = LaunchSpec((python, "-c", "pass"), {}, spec.endpoint_dir, spec.pack_root, python, False)
    profile = build_seatbelt_profile(spec, SandboxPolicy(), scratch=spec.endpoint_dir / "tmp")
    assert f'(allow file-read* (subpath "{prefix}"))' in profile
    assert '(allow file-read* (subpath "/opt"))' not in profile


@pytest.mark.skipif(os.name == "nt", reason="Seatbelt profiles use POSIX paths")
@pytest.mark.parametrize("cache_name", ["com.apple.metal", "com.apple.metalfe"])
def test_shared_metal_cache_grants_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_name: str
) -> None:
    cache_root = tmp_path / "user-cache"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "confstr", lambda key: str(cache_root) if key == 65538 else None)
    spec = make_spec(tmp_path, "pass")
    with pytest.raises(SandboxError, match="dissolve"):
        build_seatbelt_profile(
            spec,
            SandboxPolicy(rw_binds=(str(cache_root / cache_name),)),
            scratch=spec.endpoint_dir / "tmp",
        )


def test_unavailable_launcher_refuses(tmp_path: Path) -> None:
    launcher = SeatbeltLauncher(capability=SeatbeltCapability(False, None, "test unavailable"))
    with pytest.raises(SandboxUnavailable, match="test unavailable"):
        asyncio.run(launcher.launch(make_spec(tmp_path, "raise AssertionError('must not run')")))


native = pytest.mark.skipif(sys.platform != "darwin", reason="native macOS Seatbelt")


@native
@pytest.mark.parametrize(
    "gpu,mps,warning",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (True, "ImportError", False),
        (True, "RuntimeError", False),
    ],
)
def test_unavailable_mps_diagnostic_preserves_cpu_execution(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], gpu: bool, mps: bool | str, warning: bool
) -> None:
    spec = make_spec(tmp_path, "print('CPU worker ran')")
    if mps == "ImportError":
        source = "raise ImportError('optional Torch is unavailable')\n"
    else:
        source = (
            "from types import SimpleNamespace\n"
            "def available():\n"
            + (
                "    raise RuntimeError('MPS inspection failed')\n"
                if mps == "RuntimeError"
                else f"    return {mps!r}\n"
            )
            + "backends = SimpleNamespace(mps=SimpleNamespace(is_available=available))\n"
        )
    (spec.pack_root / "torch.py").write_text(source)
    spec = LaunchSpec(
        spec.command,
        {"PYTHONPATH": str(spec.pack_root)},
        spec.endpoint_dir,
        spec.pack_root,
        spec.python,
        False,
    )

    async def scenario() -> None:
        process = await SeatbeltLauncher(SandboxPolicy(gpu=gpu)).launch(spec)
        assert await asyncio.wait_for(process.wait(), 30) == 0

    asyncio.run(scenario())
    output = capfd.readouterr()
    assert output.out.strip() == "CPU worker ran"
    assert ("MPS is unavailable inside the sandbox" in output.err) is warning
    assert ("could not determine sandboxed MPS availability" in output.err) is isinstance(mps, str)
    if isinstance(mps, str):
        assert mps in output.err
    if warning:
        assert "AppleParavirtDeviceUserClient" in output.err
        assert "CPU execution remains available" in output.err


@native
def test_files_processes_network_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert detect_seatbelt().available
    secret_file = tmp_path / "ungranted"
    secret_file.write_text("private")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-inherited")
    spec = make_spec(tmp_path, "")
    output = spec.endpoint_dir / "tmp" / "result.json"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    script = f"""
import ctypes, json, os, socket
from pathlib import Path
results = {{}}
for name, action in [
    ('file', lambda: Path({str(secret_file)!r}).read_text()),
    ('write', lambda: Path({str(secret_file)!r}).write_text('changed')),
    ('signal', lambda: os.kill({os.getpid()}, 0)),
    ('tcp', lambda: socket.create_connection(('127.0.0.1', {port}), timeout=1)),
    ('home', lambda: os.listdir({str(Path.home())!r})),
]:
    try:
        action()
        results[name] = False
    except OSError:
        results[name] = True
lib = ctypes.CDLL('/usr/lib/libproc.dylib')
buffer = ctypes.create_string_buffer(4096)
results['inspect'] = lib.proc_pidpath({os.getpid()}, buffer, len(buffer)) == 0
results['credentials'] = 'AWS_SECRET_ACCESS_KEY' not in os.environ
results['self'] = lib.proc_pidpath(os.getpid(), buffer, len(buffer)) > 0
Path({str(output)!r}).write_text(json.dumps(results))
"""
    spec = LaunchSpec(
        (sys.executable, "-c", script), {}, spec.endpoint_dir, spec.pack_root, sys.executable, True
    )

    async def scenario() -> None:
        process = await SeatbeltLauncher().launch(spec)
        assert await asyncio.wait_for(process.wait(), 30) == 0

    try:
        asyncio.run(scenario())
    finally:
        listener.close()
    assert json.loads(output.read_text()) == {
        "file": True,
        "write": True,
        "signal": True,
        "tcp": True,
        "home": True,
        "inspect": True,
        "credentials": True,
        "self": True,
    }
    assert secret_file.read_text() == "private"


@native
def test_real_unix_transport_and_shared_memory(tmp_path: Path) -> None:
    from multiprocessing.shared_memory import SharedMemory

    from dinkster_workers.boundary import read_frame, write_frame
    from dinkster_workers.transport import BoundaryListener

    spec = make_spec(tmp_path, "")
    # The normal host endpoint is created by IsolatedWorker in the OS temp
    # folder, rather than under this potentially long test checkout path.
    endpoint = Path(tempfile.mkdtemp(prefix="dinkster-iso-"))

    async def scenario() -> None:
        listener = await BoundaryListener.create(endpoint)
        name = "dinkster" + uuid.uuid4().hex[:16]
        segment = SharedMemory(name=name, create=True, size=17)
        assert segment.buf is not None
        segment.buf[:17] = b"asymmetric bytes!"
        try:
            script = f"""
import asyncio
from multiprocessing.shared_memory import SharedMemory
from dinkster_workers.boundary import read_frame, write_frame
from dinkster_workers.transport import connect_endpoint
async def main():
    reader, writer = await connect_endpoint({listener.endpoint!r})
    value = await read_frame(reader)
    segment = SharedMemory(name={name!r})
    assert bytes(segment.buf[:17]) == b'asymmetric bytes!'
    segment.buf[0] = 90
    segment.close()
    await write_frame(writer, value[0], value[1])
    writer.close()
    await writer.wait_closed()
asyncio.run(main())
"""
            launched = LaunchSpec(
                (sys.executable, "-c", script), {}, endpoint, spec.pack_root, sys.executable, True
            )
            process = await SeatbeltLauncher(
                SandboxPolicy(ro_binds=(str(Path(__file__).parents[1]),))
            ).launch(launched)
            reader, writer = await asyncio.wait_for(listener.connected, 30)
            payload = {"type": "probe", "value": "transport"}
            blob = b"asymmetric transport bytes" * 4096
            await write_frame(writer, payload, [blob])
            assert await asyncio.wait_for(read_frame(reader), 30) == (
                {**payload, "blobs": [len(blob)]},
                [blob],
            )
            assert await asyncio.wait_for(process.wait(), 30) == 0
            assert segment.buf[0] == 90
            writer.close()
            await writer.wait_closed()
        finally:
            segment.close()
            segment.unlink()
            await listener.close()

    asyncio.run(scenario())


@native
def test_private_metal_cache_is_per_pack(tmp_path: Path) -> None:
    assert sys.platform == "darwin"
    base = os.confstr(65538)
    assert base

    async def launch(pack: Path, marker: Path | None = None) -> Path:
        endpoint = Path(tempfile.mkdtemp(prefix="dinkster-iso-"))
        output = endpoint / "tmp" / "cache.json"
        pack.mkdir(exist_ok=True)
        script = f"""
import json, os
from pathlib import Path
cache = os.confstr(65538)
assert cache and cache != {base!r}
path = Path(cache)
assert path.stat().st_mode & 0o777 == 0o700
(path / 'marker').write_text('own pack')
other = {str(marker) if marker else ""!r}
if other:
    try:
        Path(other).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('another pack cache was readable')
Path({str(output)!r}).write_text(json.dumps(cache))
"""
        spec = LaunchSpec((sys.executable, "-c", script), {}, endpoint, pack, sys.executable, False)
        process = await SeatbeltLauncher(SandboxPolicy(gpu=True)).launch(spec)
        assert await asyncio.wait_for(process.wait(), 30) == 0
        return Path(json.loads(output.read_text()))

    async def scenario() -> None:
        first = await launch(tmp_path / "first-pack")
        second = await launch(tmp_path / "second-pack", first / "marker")
        repeated = await launch(tmp_path / "first-pack", second / "marker")
        assert first == repeated
        assert first != second

    asyncio.run(scenario())


@native
def test_isolated_worker_runs_image_graph() -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    root = Path(__file__).parents[1]

    async def scenario() -> None:
        composer = ServingComposer(sandbox_policy=SandboxPolicy(ro_binds=(str(root),)))
        try:
            await composer.add_pack(
                PackSpec(
                    root / "packages/dinkster-nodes-dev/dinkster-pack.toml", trust_reserved=True
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(
                Graph(
                    nodes={
                        "g": GraphNode("dev.image.gradient", {"width": 2048, "height": 1024}),
                        "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
                        "s": GraphNode("dev.image.stats", {"image": Link("i", "image")}),
                    }
                ),
                ["s"],
            )
            mean = result.outputs["s"]["mean"].resolve()
            assert isinstance(mean, float)
            assert mean == pytest.approx(0.5, abs=0.001)
        finally:
            await composer.close()

    asyncio.run(scenario())


@native
def test_doctor_uses_seatbelt(tmp_path: Path) -> None:
    from dinkster_workers.doctor import diagnose, render_text
    from dinkster_workers.sandbox import detect_probe_jail

    from tests.test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES, write_pack

    victim = tmp_path / "ungranted.txt"
    victim.write_text("host-owned")
    source = (
        HEALTHY_NODES
        + f"""
from pathlib import Path
try:
    Path({str(victim)!r}).write_text('overwritten')
except PermissionError:
    pass
else:
    raise AssertionError('host file was writable')
"""
    )
    manifest = write_pack(tmp_path / "healthy", HEALTHY_MANIFEST, "healthy_nodes", source)
    report = diagnose(manifest, probe_jail=detect_probe_jail())
    assert report.ok, render_text(report)
    assert report.node_types == ("healthy.doubler", "healthy.tagger")
    assert victim.read_text() == "host-owned"


@native
def test_egress_and_unrelated_unix_socket_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_workers.egress as egress

    endpoint = Path(tempfile.mkdtemp(prefix="dinkster-iso-"))
    other = Path(tempfile.mkdtemp(prefix="dinkster-other-")) / "other.sock"
    spec = make_spec(tmp_path, "")
    output = endpoint / "tmp" / "result.txt"

    async def scenario() -> None:
        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            assert await reader.readexactly(4) == b"ping"
            writer.write(b"pong")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        assert sys.platform == "darwin"
        other_server = await asyncio.start_unix_server(echo, path=other)
        port = server.sockets[0].getsockname()[1]

        async def resolver(host: str, resolved_port: int) -> tuple[str, ...]:
            assert (host, resolved_port) == ("allowed.example.test", port)
            return ("127.0.0.1",)

        monkeypatch.setattr(egress, "_resolve_public_addresses", resolver)
        script = f"""
import os, socket
from pathlib import Path
for address in [{str(other)!r}, ('127.0.0.1', {port})]:
    family = socket.AF_UNIX if isinstance(address, str) else socket.AF_INET
    with socket.socket(family) as direct:
        direct.settimeout(2)
        try:
            direct.connect(address)
        except OSError:
            pass
        else:
            raise AssertionError('direct connection escaped')
def connect(target):
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(5)
    client.connect(os.environ['DINKSTER_EGRESS_PROXY'])
    client.sendall(('CONNECT ' + target + ' HTTP/1.1\\r\\n\\r\\n').encode())
    head = b''
    while b'\\r\\n\\r\\n' not in head:
        head += client.recv(4096)
    return client, head
client, head = connect('denied.example.test:443')
assert head.startswith(b'HTTP/1.1 403 '), head
client.close()
client, head = connect('allowed.example.test:{port}')
assert head.startswith(b'HTTP/1.1 200 '), head
client.sendall(b'ping')
assert client.recv(4) == b'pong'
client.close()
Path({str(output)!r}).write_text('proxied; direct sockets denied')
"""
        launcher = SeatbeltLauncher(
            SandboxPolicy(egress_allowlist=(f"https://allowed.example.test:{port}",))
        )
        launched = LaunchSpec(
            (sys.executable, "-c", script), {}, endpoint, spec.pack_root, sys.executable, False
        )
        try:
            process = await launcher.launch(launched)
            assert await asyncio.wait_for(process.wait(), 30) == 0
            assert output.read_text() == "proxied; direct sockets denied"
            async with asyncio.timeout(5):
                while (endpoint / "egress.sock").exists():
                    await asyncio.sleep(0)
            assert not (endpoint / "egress.sock").exists()
        finally:
            server.close()
            other_server.close()
            await server.wait_closed()
            await other_server.wait_closed()

    asyncio.run(scenario())
