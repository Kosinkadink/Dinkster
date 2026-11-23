"""dinkster-serve progressive pack startup, end to end.

The real process: the port binds on a zero-node diagnostic surface, pack
workers compose behind it, each announcement grows /api/nodes and bumps the
epoch, /api/health narrates the in-flight composition, and a failing
pack is recorded on /api/composition while the survivors serve (or, under
--strict-packs, takes the whole process down with a nonzero exit) - never
a silently degraded surface: absence always has a queryable reason.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest
from test_compose import write_iso_manifest

from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parent
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
pytestmark = pytest.mark.usefixtures("installed_default_catalogs")


@pytest.fixture(autouse=True)
def _use_current_python_for_standard_pack_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    monkeypatch.setenv("DINKSTER_SERVING_PYTHON", sys.executable)
    account = tmp_path_factory.mktemp("serve-account")
    (account / "model-settings.json").write_text(
        '{"version":1,"locations":[],"downloadLocation":null,"disabled":true}'
    )
    monkeypatch.setenv("DINKSTER_HOME", str(account))


@pytest.fixture
def published_library(tmp_path: Path) -> Path:
    from dinkster_library._migration import publish_records

    library = tmp_path / "library"
    publish_records(
        library,
        Path(os.environ["DINKSTER_HOME"]),
        "12" * 16,
        tmp_path / "bootstrap-state",
        environment="test",
    )
    return library


async def _default_pack_names() -> tuple[str, ...]:
    from dinkster_workers import load_manifest

    from dinkster.compose import ServingComposer, default_pack_specs

    composer = ServingComposer()
    try:
        specs = composer.order_pack_entries(default_pack_specs())
        return tuple(load_manifest(Path(spec.manifest)).name for spec in specs)
    finally:
        await composer.close()


def test_folded_engine_serves_catalog_jobs_and_sessions_without_supervisor(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    from dinkster_graph import Graph, GraphNode, graph_to_wire

    fixture_pack = TESTS_DIR / "fixtures/pack-testing"
    environment = tmp_path / "core-environment"
    requirements = subprocess.check_output(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--package",
            "dinkster",
            "--no-emit-package",
            "dinkster-supervisor",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
        ],
        cwd=TESTS_DIR.parent,
        text=True,
    )
    cpu_requirements = tmp_path / "core-requirements.txt"
    cpu_requirements.write_text(
        "\n".join(
            line
            for line in requirements.splitlines()
            if not line.startswith(("nvidia-", "triton=="))
        )
        + "\n"
    )
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            "--torch-backend",
            "cpu",
            str(cpu_requirements),
        ],
        cwd=TESTS_DIR.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    isolated_runtime = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata as m, importlib.util as u; "
            "assert u.find_spec('torch') is None; "
            "assert not any(d.metadata['Name'].lower().startswith(('nvidia-', 'triton')) "
            "for d in m.distributions()); "
            "assert u.find_spec('dinkster_collab') is not None; "
            "assert u.find_spec('dinkster_supervisor') is None; "
            "assert all(not any(d.metadata['Name'] == n for d in m.distributions()) for n in "
            "('dinkster-collab', 'dinkster-supervisor', 'dinkster-nodes-std', "
            "'dinkster-nodes-dev')); "
            "print('No torch, NVIDIA or triton distributions')",
        ],
        text=True,
    )
    request.node.add_report_section(
        "call",
        "Folded engine CPU environment",
        installed.stdout + installed.stderr + isolated_runtime,
    )

    port = free_port()
    process_environment = {
        **os.environ,
        "DINKSTER_REMOTE_CATALOG_BASE": "",
        "DINKSTER_REMOTE_GATEWAY_BASE": "",
        "DINKSTER_SERVING_PYTHON": str(python),
    }
    log = tmp_path / "server.log"
    output = log.open("w")
    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "dinkster.serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--library-root",
            "",
            "--pack",
            str(fixture_pack),
        ],
        cwd=tmp_path,
        env={**process_environment, "PYTHONPATH": str(fixture_pack / "src")},
        stdout=output,
        stderr=output,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(120):
                while True:
                    assert process.poll() is None, "core-only server died"
                    try:
                        async with session.get(base + "/api/nodes") as response:
                            if response.status == 200:
                                payload = await response.json()
                                if "composing" not in payload:
                                    break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            nodes = payload["nodes"]
            assert isinstance(nodes, dict)
            assert {"example-tally.shout", "example-tally.tally"} <= nodes.keys()
            assert all(
                isinstance(node_id, str) and isinstance(schema, dict)
                for node_id, schema in nodes.items()
            )
            assert nodes["example-tally.shout"]["pack"] == "example-tally"
            for method, path in (
                ("GET", "/api/compat/comfy/schemas"),
                ("POST", "/api/compat/comfy/prompt"),
                ("POST", "/api/compat/comfy/prompt?dryRun=1"),
            ):
                async with session.request(method, base + path) as response:
                    assert response.status == 404, await response.text()
            graph = Graph(nodes={"integer": GraphNode("example-tally.shout", {"text": "hello"})})
            async with session.post(
                base + "/api/jobs",
                json={
                    "clientId": "optional-package-proof",
                    "jobId": "integer",
                    "graph": graph_to_wire(graph),
                    "targets": ["integer"],
                },
            ) as response:
                assert response.status == 202, await response.text()
            async with asyncio.timeout(30):
                while True:
                    async with session.get(
                        base + "/api/jobs/optional-package-proof/integer"
                    ) as response:
                        job = await response.json()
                    if job["state"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)
            assert job["state"] == "completed", job
            async with session.get(base + "/api/sessions") as response:
                assert response.status == 200, await response.text()

    try:
        run_scenario(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        output.close()


def test_parse_memory_budget() -> None:
    from dinkster.serve import parse_memory_budget

    assert parse_memory_budget("ram=1024") == ("ram", 1024)
    assert parse_memory_budget("vram:cuda:0=20g") == ("vram:cuda:0", 20 * 1024**3)
    assert parse_memory_budget("ram=8K") == ("ram", 8192)
    assert parse_memory_budget(" ram = 2M ") == ("ram", 2 * 1024**2)
    assert parse_memory_budget("ram=1T") == ("ram", 1024**4)
    for bad in ("ram", "=5", "ram=", "ram=5x", "ram=g", "ram=-5", "ram=1.5G"):
        with pytest.raises(ValueError):
            parse_memory_budget(bad)


def test_parse_cuda_devices_requires_canonical_unique_indices() -> None:
    from dinkster.serve import parse_cuda_devices

    assert parse_cuda_devices("1,0") == (1, 0)
    assert parse_cuda_devices("2,1,0") == (2, 1, 0)
    for bad in ("", "0", "0,0", "0,-1", "0, 1", "00,1", "gpu0,1", "0,1,"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_cuda_devices(bad)


_ADA = "0, GPU-aaaaaaaa-1111-2222-3333-444444444444, 8.9, NVIDIA GeForce RTX 4090"
_PASCAL = "1, GPU-bbbbbbbb-5555-6666-7777-888888888888, 6.1, NVIDIA GeForce GTX 1080"
_ALL_DTYPES = frozenset({"float16", "bfloat16", "float32"})


def _probe_smi(monkeypatch: pytest.MonkeyPatch, rows: list[str] | None) -> None:
    """nvidia-smi answering ``rows``, or absent entirely when None."""
    from dinkster import serve

    def run(command: list[str], **_kwargs: object) -> object:
        assert command[0] == "nvidia-smi"
        if rows is None:
            raise FileNotFoundError("nvidia-smi")
        return type("Completed", (), {"stdout": "\n".join(rows) + "\n"})()

    monkeypatch.setattr(serve.subprocess, "run", run)
    monkeypatch.setattr(serve.platform, "win32_ver", lambda: ("",))


def test_native_compute_dtypes_single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import serve

    _probe_smi(monkeypatch, [_ADA])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert serve.detect_native_compute_dtypes() == _ALL_DTYPES


def test_native_compute_dtypes_intersects_unordered_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without CUDA_VISIBLE_DEVICES the CUDA runtime's device order is not
    # knowable from nvidia-smi, so every device must support the answer.
    from dinkster import serve

    _probe_smi(monkeypatch, [_ADA, _PASCAL])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert serve.detect_native_compute_dtypes() == frozenset({"float32"})
    assert serve.detect_native_compute_dtypes((0,)) == frozenset({"float32"})


def test_native_compute_dtypes_follows_visible_device_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    _probe_smi(monkeypatch, [_ADA, _PASCAL])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert serve.detect_native_compute_dtypes() == _ALL_DTYPES
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert serve.detect_native_compute_dtypes() == frozenset({"float32"})
    # Positions index into the visible pool, exactly as lane launch does.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert serve.detect_native_compute_dtypes((1,)) == _ALL_DTYPES
    assert serve.detect_native_compute_dtypes((0, 1)) == frozenset({"float32"})


@pytest.mark.parametrize(
    ("rows", "visible", "indices"),
    [
        # No nvidia-smi: the accelerator is unidentified, stay permissive.
        (None, None, ()),
        # Probe ran but reported nothing.
        ([], None, ()),
        # A selector the probe cannot classify (MIG slice).
        ([_ADA], "MIG-aaaaaaaa-1111-2222-3333-444444444444", ()),
        # An executing index outside the visible pool: worker launch is
        # the layer that refuses this, the probe must not crash serve.
        ([_ADA], "0", (5,)),
    ],
)
def test_native_compute_dtypes_falls_back_permissive(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[str] | None,
    visible: str | None,
    indices: tuple[int, ...],
) -> None:
    from dinkster import serve

    _probe_smi(monkeypatch, rows)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert serve.detect_native_compute_dtypes(indices) == _ALL_DTYPES


def test_settings_gate_unknown_is_startup_parser_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--allow-settings-changes=comfy-args"],
    )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    error = capsys.readouterr().err
    assert "comfy-args" in error
    assert "memory-budgets" in error


def test_sandbox_host_grants_require_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--sandbox-grant-gpu", "gpu-pack"],
    )

    with pytest.raises(SystemExit, match="2"):
        serve.main()

    assert "require --sandbox-packs" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("grant", "message"),
    (
        ("network-pack", "requires PACK=HTTPS_ORIGIN"),
        ("network-pack=http://api.example.test", "must use https://"),
        ("bad/name=https://api.example.test", "invalid sandbox pack grant"),
    ),
)
def test_sandbox_network_grants_are_strict_origins(
    grant: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--sandbox-packs", "--sandbox-grant-network", grant],
    )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert message in capsys.readouterr().err


def test_execution_cache_layered_mode_requires_a_persistent_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--library-root", "", "--execution-cache-mode", "layered"],
    )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert "requires --execution-cache-dir" in capsys.readouterr().err


def test_auth_file_malformed_refuses_serve_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve

    path = tmp_path / "auth.toml"
    path.write_text("version = 1\ntokens = []\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--auth", str(path)])
    with pytest.raises(SystemExit, match="tokens must be a non-empty"):
        serve.main()


def test_non_loopback_bind_without_authentication_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit, match="2"):
        serve.main()

    assert "a non-loopback --host requires --auth" in capsys.readouterr().err


def test_partial_identity_auth_configuration_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    for name in (
        "DINKSTER_IDENTITY_JWKS_URL",
        "DINKSTER_IDENTITY_ISSUER",
        "DINKSTER_IDENTITY_AUDIENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--identity-jwks-url",
            "https://identity.example/.well-known/jwks.json",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        serve.main()

    assert "requires --identity-jwks-url" in capsys.readouterr().err


def test_serve_single_job_and_replica_multi_gpu_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--multi-gpu-devices",
            "0,1",
            "--single-job-multi-gpu-devices",
            "0,1",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        serve.main()
    assert exc.value.code == 2


def test_serve_is_governed_with_declared_budgets(tmp_path: Path) -> None:
    """dinkster-serve runs governed: /memory/status reports a live governor
    (not the ungoverned null) carrying exactly the --memory-budget
    declarations, and unbudgeted devices stay absent - declared
    accounting, never an invented budget."""
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--memory-budget", "ram=64M"),
        env=dict(os.environ),
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/health") as resp:
                            if resp.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            async with session.get(base + "/memory/status") as resp:
                assert resp.status == 200
                body = await resp.json()
            report = body["memoryGovernor"]
            assert report is not None  # governed, not the ungoverned null
            assert body["leases"] is not None  # lease broker rides along
            assert report["ram"]["budgetBytes"] == 64 * 1024**2
            assert report["ram"]["reservedBytes"] == 0
            assert "vram:cuda:0" not in report

    try:
        run_scenario(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


# Ports below the kernel's ephemeral range (32768+) cannot be claimed by
# outgoing connections between this probe and the eventual server bind. Each
# test process starts in its own block so concurrent checkouts do not race for
# the same deterministic ports.
_PORT_MIN = 22100
_PORT_COUNT = 32000 - _PORT_MIN
_PORT_BLOCK = 32
_port_start = (os.getpid() % (_PORT_COUNT // _PORT_BLOCK)) * _PORT_BLOCK
_port_candidates = (
    _PORT_MIN + (_port_start + offset) % _PORT_COUNT for offset in range(_PORT_COUNT)
)


def free_port() -> int:
    for candidate in _port_candidates:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("no free test port below the ephemeral range")


def serve_command(port: int, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dinkster.serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--library-root",
        "",
        *extra,
    ]


def serve_with_library_command(port: int, library_root: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dinkster.serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--library-root",
        str(library_root),
        *extra,
    ]


def test_video_preview_pack_route_event_and_module_end_to_end(tmp_path: Path) -> None:
    import hashlib

    import av
    import numpy as np
    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire

    from tests.test_pack_surfaces import PROOF_EVENT, PROOF_PACK, PROOF_ROOT, PROOF_ROUTE

    source = tmp_path / "source"
    source.mkdir()
    with av.open(str(source / "clip.mp4"), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for index in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), index * 10, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    (source / "dinkster-pack.toml").write_text("""
[pack]
name = "preview-source"
publisher = "dinkster"
[pack.entry]
nodes = "preview_source:NODES"
types = "preview_source:register_types"
""")
    (source / "preview_source.py").write_text("""
from pathlib import Path
from dinkster_api.v1 import (
    VIDEO_TYPE,
    Node,
    NodeSchema,
    Output,
    TypeExpr,
    register_video_value_type,
    video_from_source,
)
class Source(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type="preview-source.clip",
                          body=(Output("video", TypeExpr.concrete(VIDEO_TYPE)),))
    @classmethod
    def execute(cls):
        source = Path(__file__).with_name("clip.mp4").read_bytes()
        return cls.outputs(video=video_from_source(source))
def register_types(registry):
    register_video_value_type(registry, VIDEO_TYPE)
NODES = [Source]
""")
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--no-default-packs", "--pack", str(PROOF_ROOT), "--pack", str(source)),
        env={**os.environ, "PYTHONPATH": str(source)},
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/nodes") as response:
                            nodes = await response.json()
                            if (
                                "video-preview.initialize" in nodes.get("nodes", {})
                                and "preview-source.clip" in nodes["nodes"]
                                and "composing" not in nodes
                            ):
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            async with session.get(base + "/api/composition") as response:
                report = await response.json()
            assert report["packs"][PROOF_PACK]["state"] == "announced"
            assert report["packs"]["preview-source"]["state"] == "announced"
            async with session.get(base + PROOF_ROUTE) as response:
                assert response.status == 200
                assert await response.json() == {
                    "defaultFps": 24.0,
                    "maxFrames": 120,
                    "maxWidth": 512,
                }
            async with session.get(base + "/api/extensions/snapshot") as response:
                raw = await response.read()
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            assert digest == nodes["extensionSnapshotDigest"]
            extension = next(
                item for item in json.loads(raw)["extensions"] if item["id"] == PROOF_PACK
            )
            module = extension["frontend"][0]
            async with session.get(base + module["moduleUrl"]) as response:
                assert response.status == 200
                assert (
                    "sha256:" + hashlib.sha256(await response.read()).hexdigest()
                    == module["moduleDigest"]
                )
            async with session.ws_connect(base + "/api/events?clientId=preview-proof") as ws:
                graph = Graph(
                    nodes={
                        "source": GraphNode("preview-source.clip", {}),
                        "preview": GraphNode(
                            "video-preview.initialize", {"video": Link("source", "video")}
                        ),
                    }
                )
                async with session.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "preview-proof",
                        "jobId": "initialize",
                        "graph": graph_to_wire(graph),
                        "targets": ["preview"],
                    },
                ) as response:
                    assert response.status == 202, await response.text()
                received = None
                async with asyncio.timeout(30):
                    while True:
                        event = await ws.receive_json()
                        if event.get("event") == PROOF_EVENT:
                            received = event
                        if event.get("type") == "job_state" and event.get("state") in {
                            "completed",
                            "failed",
                            "cancelled",
                        }:
                            assert event["state"] == "completed", event
                            break
                assert received is not None
                assert received["data"] == {
                    "fps": 10.0,
                    "frameCount": 20,
                    "height": 32,
                    "width": 64,
                }
                assert received["schemaVersion"] == 1
                assert received["extensionSnapshotDigest"] == digest
                assert received["pack"] == PROOF_PACK
                assert received["worker"] == "local"
                assert received["executionArm"] == "native"
                assert received["nodeId"] == "preview"
                assert received["jobId"] == "initialize"
                assert isinstance(received["seq"], int)

    try:
        run_scenario(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def write_shutdown_pack(directory: Path, index: int, *, startup_delay: float = 0) -> Path:
    directory.mkdir(parents=True)
    module = f"shutdown_pack_{index}"
    namespace = f"shutdown{index}"
    (directory / f"{module}.py").write_text(
        "import time\n"
        "from dinkster_schema import Node, NodeSchema\n"
        f"time.sleep({startup_delay!r})\n"
        f"class Probe{index}(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        f"        return NodeSchema(node_type='{namespace}.probe')\n"
        f"NODES = (Probe{index},)\n",
        encoding="utf-8",
    )
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "shutdown-pack-{index}"\npublisher = "dinkster"\n'
        f'namespaces = ["{namespace}"]\n\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\n',
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    ("failed_pack", "development"),
    [
        (False, "published"),
        (True, "local"),
        (False, "captured"),
        (False, "watched"),
        (False, "installed-local"),
        (False, "installed-git"),
        (False, "managed-registry"),
    ],
)
def test_serve_empty_generation_health_distinguishes_failed_selection(
    tmp_path: Path, failed_pack: bool, development: str
) -> None:
    from dinkster_registry import Lockfile

    from dinkster.generation_packs import stage_generation_packs

    generation = tmp_path / "generation"
    generation.mkdir()
    if development != "managed-registry":
        stage_generation_packs(generation, Lockfile.of(()), {})
        (generation / "build.json").write_text(
            json.dumps({"format": "dinkster.generation/1", "variant": "linux-cpu"})
        )
    if development == "captured":
        (generation / "development-sources.json").write_text(
            json.dumps({"format": "dinkster.development-sources/1", "sources": []})
        )
    extra = (
        ("--no-default-packs", "--pack", str(write_iso_manifest(tmp_path / "bad", name="core")))
        if failed_pack
        else ("--generation", str(generation))
    )
    if development == "watched":
        extra = ("--no-default-packs", "--watch-packs")
    if development.startswith("installed-") or development == "managed-registry":
        from dataclasses import replace

        from dinkster.installer import Installer, lock_local_pack

        source = tmp_path / "source"
        source.mkdir()
        (source / "example").mkdir()
        (source / "example/__init__.py").write_text(
            "from dinkster_api.v1 import Node, NodeSchema\n"
            "class Probe(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            "        return NodeSchema(node_type='example.probe')\n"
            "NODES = (Probe,)\n"
        )
        (source / "dinkster-pack.toml").write_text(
            '[pack]\nname="example"\npublisher="acme"\n'
            'version="1.0"\nengine=">=0.0.2.dev0,<1"\nnamespaces=["example"]\n'
            '[pack.entry]\nnodes="example:NODES"\n[pack.sandbox]\n'
        )
        installer = Installer(tmp_path / "installed")
        entry, archive = lock_local_pack(source, installer.artifacts_dir)
        if development == "installed-git":
            entry = replace(entry, source="git:https://example.invalid/pack@" + "a" * 40)
        if development == "managed-registry":
            from dinkster_workers.doctor import prepare_catalog

            entry = replace(entry, source="registry:public", publisher="acme", version="1.0")
            stage_generation_packs(generation, Lockfile.of([entry]), {"example": archive})
            prepared = generation / "pack-sources/example"
            report = prepare_catalog(
                prepared / "dinkster-pack.toml", environment={"PYTHONPATH": str(prepared)}
            )
            assert report.ok, report
            (generation / "build.json").write_text(
                json.dumps(
                    {
                        "format": "dinkster.generation/1",
                        "variant": "linux-cpu",
                        "inProcessPacks": ["example"],
                    }
                )
            )
            extra = ("--generation", str(generation))
        else:
            installer.apply(Lockfile.of([entry]), venvs=False)
            extra = ("--no-default-packs", "--install-root", str(installer.root))
    port = free_port()
    log = tmp_path / "empty-generation.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            serve_command(port, *extra),
            cwd=tmp_path,
            stdout=output,
            stderr=output,
        )
        try:
            report = wait_for_composition(process, port)
            packs = report["packs"]
            assert isinstance(packs, dict)
            if failed_pack:
                assert packs["core"]["state"] == "failed"
            elif development.startswith("installed-") or development == "managed-registry":
                assert packs["example"]["state"] == "announced"
            else:
                assert packs == {}

            async def check() -> None:
                async with aiohttp.ClientSession() as client:
                    async with client.get(f"http://127.0.0.1:{port}/api/health") as response:
                        assert response.status == 200
                        assert (await response.json())["ok"] is (not failed_pack)
                    async with client.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                        assert response.status == 200
                        catalog = await response.json()
                        assert set(catalog["nodes"]) == (
                            {"example.probe"}
                            if development.startswith("installed-")
                            or development == "managed-registry"
                            else set()
                        )
                        if development not in {"published", "managed-registry"}:
                            label = catalog["dinkster"]["development"]
                            assert label["mode"] == "development"
                            assert label["reproducible"] is False
                            assert "not reproducible from dependency ranges" in label["description"]
                        else:
                            assert "development" not in catalog["dinkster"]

            run_scenario(check())
            assert ("development: not reviewed by the registry" in log.read_text()) is (
                development not in {"published", "managed-registry"}
            )
        except Exception as exc:
            exc.add_note(log.read_text())
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


@pytest.mark.parametrize(
    ("in_process", "output_configuration", "engine_variant"),
    [
        (False, "fresh", "linux-cpu"),
        (True, "fresh", "linux-cpu"),
        (True, "custom", "linux-cpu"),
        (True, "read-only", "linux-cpu"),
        (True, "removed", "linux-cpu"),
        (True, "fresh", "linux-custom-cu130"),
        (True, "fresh", "linux-nvidia"),
        (True, "fresh", "mac-arm64"),
        (False, "fresh", "mac-arm64"),
    ],
)
def test_serve_explicit_generation_uses_only_its_sources(
    tmp_path: Path, in_process: bool, output_configuration: str, engine_variant: str
) -> None:
    import venv

    from dinkster_assets import MountConfiguration, MountDef, dump_mounts
    from dinkster_library import LibraryClient
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog

    from dinkster.generation_packs import stage_generation_packs

    manifest = write_shutdown_pack(tmp_path / "source", 57)
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    generation = tmp_path / "builds" / "1"
    generation.mkdir(parents=True)
    pack = "shutdown-pack-57"
    lock = Lockfile.of([LockedPack(pack, "1.2.3", digest, "dinkster", ("shutdown57",))])
    stage_generation_packs(generation, lock, {pack: archive})
    (generation / "build.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "variant": engine_variant,
                "inProcessPacks": [pack] if in_process else [],
            }
        )
    )
    if not in_process:
        venv.EnvBuilder(system_site_packages=True).create(generation / "packs" / pack)
    source = generation / "pack-sources" / pack
    report = prepare_catalog(source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)})
    assert report.ok, report
    library_root = tmp_path / "library"
    expected_output = "output" if output_configuration == "fresh" else None
    output_path = library_root / "output"
    existing_mount = None
    if output_configuration != "fresh":
        library_root.mkdir()
        if output_configuration == "removed":
            (library_root / ".output-initialized").touch()
        else:
            output_path = tmp_path / "custom-output"
            if output_configuration == "custom":
                expected_output = "renders"
                output_path.mkdir()
            existing_mount = MountDef(
                expected_output or "output",
                output_path,
                "readwrite" if expected_output else "read",
            )
            (library_root / "mounts.toml").write_text(
                dump_mounts((existing_mount,), output_mount=expected_output), encoding="utf-8"
            )
    port = free_port()
    log = tmp_path / "server.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            serve_with_library_command(
                port,
                tmp_path / "library",
                "--generation",
                str(generation),
                "--strict-packs",
                *(["--engine-variant", engine_variant] if in_process else []),
            ),
            cwd=tmp_path,
            stdout=output,
            stderr=output,
        )
        try:
            wait_for_composition(process, port)

            async def check() -> None:
                async with aiohttp.ClientSession() as client:
                    async with client.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                        data = await response.json()
                    async with client.get(f"http://127.0.0.1:{port}/api/mounts") as response:
                        mounts = await response.json()
                    assert mounts["outputMount"] == expected_output, mounts
                    if expected_output is not None:
                        selected = next(
                            row for row in mounts["mounts"] if row["id"] == expected_output
                        )
                        async with asyncio.timeout(60):
                            while selected["state"] != "ready":
                                await asyncio.sleep(0.05)
                                async with client.get(
                                    f"http://127.0.0.1:{port}/api/mounts"
                                ) as response:
                                    assert response.status == 200
                                    mounts = await response.json()
                                selected = next(
                                    row for row in mounts["mounts"] if row["id"] == expected_output
                                )
                        assert selected["mode"] == "readwrite"
                        assert Path(selected["path"]) == output_path
                        async with client.post(
                            f"http://127.0.0.1:{port}/api/assets",
                            params={
                                "scope": "local",
                                "mount": expected_output,
                                "name": "first-run.txt",
                            },
                            data=b"first generation output",
                        ) as response:
                            assert response.status == 201, await response.text()
                            saved = await response.json()
                        assert (
                            saved["asset"]["virtualPath"]
                            == f"mounts/{expected_output}/first-run.txt"
                        )
                        assert (
                            output_path / "first-run.txt"
                        ).read_bytes() == b"first generation output"
                    else:
                        assert not output_path.exists()
                    if existing_mount is not None:
                        with LibraryClient(library_root) as helper:
                            assert existing_mount in MountConfiguration.load(helper).mounts
                    assert not (library_root / "mounts.toml").exists()
                    assert mounts["mountChangesAllowed"] is False
                    workflow = {
                        "format": "dinkster-workflow",
                        "formatVersion": 1,
                        "graphs": {"root": {"nodes": {"probe": {"type": "shutdown57.probe"}}}},
                        "environment": {
                            "nodes": {"shutdown57.probe": {"pack": pack}},
                            "packs": {pack: {"version": "1.2.3"}},
                        },
                    }
                    for plan_variant in (
                        (None, "win-nvidia", "linux-nvidia")
                        if engine_variant != "linux-custom-cu130"
                        else ("win-nvidia", "linux-nvidia")
                    ):
                        body: dict[str, object] = {"workflows": [workflow]}
                        if plan_variant is not None:
                            body["variant"] = plan_variant
                        async with client.post(
                            f"http://127.0.0.1:{port}/api/workflows/plan", json=body
                        ) as response:
                            plan = await response.json()
                            assert response.status == 200, plan
                            assert plan == {"packs": [], "assets": [], "errors": []}
                assert set(data["nodes"]) == {"shutdown57.probe"}, data
                assert data["nodes"]["shutdown57.probe"]["pack"] == pack

            run_scenario(check())
        except Exception as exc:
            exc.add_note(log.read_text())
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

    if in_process:
        for selected_variant, explicit_variant, error in (
            ("linux-cpu", "win-nvidia", "--engine-variant differs"),
            (None, "linux-cpu", "snapshot variant must be a non-empty string"),
            ("../custom", "linux-cpu", "snapshot variant must be a portable variant name"),
        ):
            (generation / "build.json").write_text(
                json.dumps(
                    {
                        "format": "dinkster.generation/1",
                        "variant": selected_variant,
                        "inProcessPacks": [pack],
                    }
                )
            )
            refused = subprocess.run(
                serve_command(
                    free_port(),
                    "--generation",
                    str(generation),
                    "--engine-variant",
                    explicit_variant,
                ),
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert refused.returncode != 0
            assert error in refused.stderr


def child_pids(parent: subprocess.Popen[bytes]) -> tuple[int, ...]:
    children = Path(f"/proc/{parent.pid}/task/{parent.pid}/children")
    return tuple(int(pid) for pid in children.read_text("ascii").split())


@pytest.mark.parametrize("startup", ["managed", "direct", "explicit-state"])
def test_start_publishes_legacy_records_before_serving(tmp_path: Path, startup: str) -> None:
    from dinkster_assets import LibraryStore, digest_bytes
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_store import library_lock

    library = tmp_path / "library"
    store = LibraryStore(library / "library.sqlite")
    try:
        record = store.create(
            "local", "Preserved workflow", digest_bytes(b"workflow"), "application/json"
        )
    finally:
        store.close()
    (library / "settings.json").write_text('{"jobs":{"maxRunningJobs":3}}')
    port = free_port()
    manifest = write_iso_manifest(tmp_path / "pack")
    pack_args = ("--no-default-packs", "--pack", str(manifest))
    engine = "sha256:" + "ab" * 32
    command = [
        sys.executable,
        str(library_lock.__file__),
        str(library),
        "test engine",
        "--engine-state",
        str(tmp_path / "engine"),
        "1",
        json.dumps({"engine": engine, "readers": [engine]}),
        "--",
        *serve_with_library_command(port, library, *pack_args)[3:],
    ]
    if startup != "managed":
        command = serve_with_library_command(port, library, *pack_args)
        if startup == "explicit-state":
            command.extend(["--engine-state-root", str(tmp_path / "engine")])
    environment = dict(os.environ, PYTHONPATH=str(TESTS_DIR))
    process, log = start_logged_serve(command, tmp_path, env=environment)
    error: BaseException | None = None
    try:
        wait_for_composition(process, port)

        async def scenario() -> None:
            async with aiohttp.ClientSession() as client:
                async with client.get(
                    f"http://127.0.0.1:{port}/api/library/{record.id}?scope=local"
                ) as response:
                    assert response.status == 200
                    document = await response.json()
                    assert document["id"] == record.id
                    assert document["name"] == "Preserved workflow"
                async with client.get(f"http://127.0.0.1:{port}/api/settings") as response:
                    assert response.status == 200
                    settings = await response.json()
                    assert settings["settings"]["jobs"]["value"]["maxRunningJobs"] == 3
                base = f"http://127.0.0.1:{port}"
                async with client.patch(
                    base + f"/api/library/{record.id}",
                    json={"scope": "local", "revision": 1, "name": "Saved after migration"},
                ) as response:
                    assert response.status == 200, await response.text()
                    assert (await response.json())["name"] == "Saved after migration"
                async with client.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "migration",
                        "jobId": "after-migration",
                        "graph": graph_to_wire(
                            Graph(nodes={"talk": GraphNode("iso.talky", {"value": "migrated"})})
                        ),
                        "targets": ["talk"],
                    },
                ) as response:
                    assert response.status == 202, await response.text()
                async with asyncio.timeout(30):
                    while True:
                        async with client.get(
                            base + "/api/jobs/migration/after-migration"
                        ) as response:
                            assert response.status == 200, await response.text()
                            job = await response.json()
                        if job["state"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.05)
                assert job["state"] == "completed", job

        run_scenario(scenario())
        assert json.loads((library / "record-migration.json").read_text())["complete"] is True
        assert not (library / "history.sqlite").exists()
        assert not (library / "principals.sqlite").exists()
        assert not (library / "settings.json").exists()
        assert len(list((library / "backups").glob("before-migration-*/manifest.json"))) == 1
    except BaseException as exc:
        error = exc
        raise
    finally:
        stop_logged_serve(process, log, error)


def test_explicit_engine_state_keeps_recovery_sessions_and_uploads_separate(tmp_path: Path) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_graph import Graph
    from dinkster_library._migration import publish_records
    from dinkster_server import HistoryStore
    from dinkster_server.queue import Job, JobKey

    publish_records(
        tmp_path / "other-library",
        Path(os.environ["DINKSTER_HOME"]),
        "34" * 16,
        tmp_path / "other-bootstrap",
        environment="test",
    )
    library = tmp_path / "library"
    first, second = [tmp_path / "environment" / name for name in ("first", "second")]
    upload_roots: dict[Path, Path] = {}
    for state in (first, second):
        port = free_port()
        process, log = start_logged_serve(
            serve_with_library_command(
                port, library, "--no-default-packs", "--engine-state-root", str(state)
            ),
            tmp_path,
        )
        pending_error: BaseException | None = None
        try:
            wait_for_composition(process, port)
            candidates = set((library / ".engine-uploads").iterdir())
            assert len(candidates) == len(upload_roots) + 1
            selected = candidates - set(upload_roots.values())
            assert len(selected) == 1
            upload_roots[state] = selected.pop()
            assert upload_roots[state].stat().st_dev == (library / "vault").stat().st_dev
            assert not (state / "uploads").exists()
        except BaseException as error:
            pending_error = error
            raise
        finally:
            stop_logged_serve(process, log, pending_error)
    from dinkster_library import LibraryClient

    legacy_path = tmp_path / "unrelated-library/history.sqlite"
    for state, run_id in ((None, "legacy"), (first, "first"), (second, "second")):
        helper = LibraryClient(library)
        store = HistoryStore(
            helper if state else legacy_path,
            accepted_path=state / "accepted.sqlite" if state else None,
        )
        try:
            store.job_accepted(
                Job(
                    key=JobKey("client", run_id),
                    graph=Graph(nodes={}),
                    targets=(),
                    priority=0,
                    run_id=run_id,
                    fingerprint=run_id,
                )
            )
        finally:
            store.close()
            helper.close()
    for directory in (library / "vault", upload_roots[first], upload_roots[second]):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".latent-upload-pending").write_bytes(b"unfinished")
    session_id: str | None = None
    for state, expected_runs in (
        (first, {"first"}),
        (second, {"first", "second"}),
        (first, {"first", "second"}),
    ):
        port = free_port()
        process, log = start_logged_serve(
            serve_with_library_command(
                port, library, "--no-default-packs", "--engine-state-root", str(state)
            ),
            tmp_path,
        )
        pending_error: BaseException | None = None
        try:
            wait_for_composition(process, port)

            async def scenario(port: int, state: Path, expected_runs: set[str]) -> None:
                nonlocal session_id
                base = f"http://127.0.0.1:{port}"
                payload = f"engine-local {state.name} {len(expected_runs)} upload".encode()

                async def upload_body():
                    yield payload[:5]
                    async with asyncio.timeout(10):
                        while not list(upload_roots[state].glob(".media-upload-*")):
                            await asyncio.sleep(0.01)
                    assert not (state / "uploads").exists()
                    other = second if state == first else first
                    assert not list(upload_roots[other].glob(".media-upload-*"))
                    yield payload[5:]

                async with aiohttp.ClientSession() as client:
                    async with client.post(
                        base + "/api/assets/transfers?scope=local&name=staged.bin",
                        data=upload_body(),
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Dinkster-Source-Machine": "peer-engine",
                            "X-Dinkster-Digest": digest_bytes(payload),
                        },
                    ) as response:
                        assert response.status == 201, await response.text()
                        asset = (await response.json())["asset"]
                    assert not list(upload_roots[state].glob(".media-upload-*"))
                    async with client.get(base + "/api/assets/" + asset["digest"]) as response:
                        assert response.status == 200
                        assert await response.read() == payload
                    async with client.get(base + "/api/history?scope=local") as response:
                        assert response.status == 200
                        records = (await response.json())["records"]
                        assert {row["runId"] for row in records} == expected_runs
                        assert all(row["state"] == "interrupted" for row in records)
                    async with client.get(base + "/api/sessions?scope=local") as response:
                        assert response.status == 200
                        sessions = (await response.json())["sessions"]
                        assert [row["sessionId"] for row in sessions] == (
                            [session_id] if state == first and session_id else []
                        )
                    if session_id is None:
                        async with client.post(
                            base + "/api/sessions",
                            json={"scope": "local", "documentId": "one", "snapshot": {"v": 7}},
                        ) as response:
                            assert response.status == 201, await response.text()
                            session_id = (await response.json())["sessionId"]

            run_scenario(scenario(port, state, expected_runs))
            assert (state / "worker-state" / "mounts-snapshot.json").is_file()
            assert (state / "execution.sqlite").is_file()
            assert (state / "sessions.sqlite").is_file()
            assert (state / "value-store").is_dir()
            assert not (upload_roots[state] / ".latent-upload-pending").exists()
            assert (library / "vault" / ".latent-upload-pending").read_bytes() == b"unfinished"
            if state == first and expected_runs == {"first"}:
                assert (
                    upload_roots[second] / ".latent-upload-pending"
                ).read_bytes() == b"unfinished"
                competitor = subprocess.run(
                    serve_with_library_command(
                        free_port(),
                        tmp_path / "other-library",
                        "--no-default-packs",
                        "--engine-state-root",
                        str(first),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                assert competitor.returncode != 0
                assert "Engine state is already in use" in competitor.stderr
                assert not (tmp_path / "other-library" / "history.sqlite").exists()
        except BaseException as error:
            pending_error = error
            raise
        finally:
            stop_logged_serve(process, log, pending_error)
    helper = LibraryClient(library)
    original = HistoryStore(helper, accepted_path=first / "accepted.sqlite")
    try:
        original.job_accepted(
            Job(
                key=JobKey("client", "not-for-another-library"),
                graph=Graph(nodes={}),
                targets=(),
                priority=0,
                run_id="not-for-another-library",
                fingerprint="not-for-another-library",
                scope="private",
                source_document="blake3:" + "cd" * 32,
            )
        )
    finally:
        original.close()
        helper.close()
    process, log = start_logged_serve(
        serve_with_library_command(
            free_port(),
            tmp_path / "other-library",
            "--no-default-packs",
            "--engine-state-root",
            str(first),
        ),
        tmp_path,
    )
    pending_error = None
    try:
        assert process.wait(timeout=10) != 0
        assert "Engine state belongs to another library" in log.read_text("utf-8")
        assert not (tmp_path / "other-library" / "history.sqlite").exists()
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)
    helper = LibraryClient(library)
    original = HistoryStore(helper, accepted_path=first / "accepted.sqlite")
    try:
        recovered = original.recover_interrupted()
        assert [(record.run_id, record.scope, record.source_document) for record in recovered] == [
            ("not-for-another-library", "private", "blake3:" + "cd" * 32)
        ]
    finally:
        original.close()
        helper.close()
    legacy = HistoryStore(legacy_path)
    try:
        assert [record.run_id for record in legacy.recover_interrupted()] == ["legacy"]
    finally:
        legacy.close()
    for name in ("worker-state", "execution.sqlite", "sessions.sqlite", "value-store"):
        assert not (library / name).exists()


def test_launcher_state_carries_forward_without_recovering_newer_work(tmp_path: Path) -> None:
    from dinkster_graph import Graph
    from dinkster_library import LibraryClient
    from dinkster_server import HistoryStore
    from dinkster_server.queue import Job, JobKey
    from dinkster_store import library_lock

    library, state = tmp_path / "library", tmp_path / "environment/engine-state/main"
    environment = dict(os.environ)
    directories: list[Path] = []
    newer_upload: Path | None = None
    for snapshot, expected in (
        (5, set()),
        (8, {"older"}),
        (7, {"older"}),
        (8, {"older"}),
        (9, {"older"}),
    ):
        port = free_port()
        writer = "sha256:" + ("b" if snapshot == 8 else "a") * 64
        readers = ["sha256:" + "a" * 64, *([writer] if snapshot == 8 else [])]
        command = [
            sys.executable,
            str(library_lock.__file__),
            str(library),
            "test engine",
            "--engine-state",
            str(state),
            str(snapshot),
            json.dumps({"engine": writer, "readers": readers}),
            "--",
            *serve_with_library_command(port, library, "--no-default-packs")[3:],
        ]
        if snapshot in (5, 7):
            interrupted = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import os
import runpy
import sys
from pathlib import Path
original = os.replace
before_publish = sys.argv[1] == "5"

def replace(source, destination):
    if Path(destination).name == "state.json":
        if not before_publish:
            original(source, destination)
        os._exit(73)
    return original(source, destination)

os.replace = replace
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
""",
                    str(snapshot),
                    *command[1:],
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert interrupted.returncode == 73, interrupted.stderr
            if snapshot == 5:
                assert not (state / "state.json").exists()
                assert not (library / "history.sqlite").exists()
            else:
                selected = json.loads((state / "state.json").read_text("utf-8"))
                assert selected["lastSnapshot"] == 7
                assert not (state / selected["directory"]).exists()
        process, log = start_logged_serve(command, tmp_path, env=environment)
        pending_error: BaseException | None = None
        try:
            wait_for_composition(process, port)

            async def scenario(port: int = port, expected: set[str] = expected) -> None:
                async with aiohttp.ClientSession() as client:
                    async with client.get(
                        f"http://127.0.0.1:{port}/api/history?scope=local"
                    ) as response:
                        assert response.status == 200
                        records = (await response.json())["records"]
                        assert {row["runId"] for row in records} == expected

            run_scenario(scenario())
            if snapshot == 5:
                before = (state / "state.json").read_bytes()
                competitor = subprocess.run(
                    [*command[:2], str(tmp_path / "other-library"), *command[3:]],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                assert competitor.returncode != 0
                assert "already in use" in competitor.stderr
                assert (state / "state.json").read_bytes() == before
        except BaseException as error:
            pending_error = error
            raise
        finally:
            stop_logged_serve(process, log, pending_error)
        selected = json.loads((state / "state.json").read_text("utf-8"))
        assert selected["lastSnapshot"] == snapshot
        working = state / selected["directory"]
        directories.append(working)
        if len(directories) in (1, 2, 4):
            run_id = {1: "older", 2: "newer", 4: "rebuilt-newer"}[len(directories)]
            helper = LibraryClient(library)
            store = HistoryStore(helper, accepted_path=working / "accepted.sqlite")
            try:
                store.job_accepted(
                    Job(
                        key=JobKey("client", run_id),
                        graph=Graph(nodes={}),
                        targets=(),
                        priority=0,
                        run_id=run_id,
                        fingerprint=run_id,
                    )
                )
            finally:
                store.close()
                helper.close()
            if len(directories) == 2:
                upload_root = next((library / ".engine-uploads").iterdir())
                newer_upload = upload_root / ".latent-upload-pending"
                newer_upload.write_bytes(b"newer snapshot's unfinished upload")
    assert directories[0] == directories[1]
    assert directories[2] == directories[3]
    assert directories[0] != directories[2]
    assert directories[4] not in directories[:4]
    assert newer_upload is not None
    assert newer_upload.read_bytes() == b"newer snapshot's unfinished upload"
    with LibraryClient(library) as helper:
        for directory, expected_run in (
            (directories[0], "newer"),
            (directories[3], "rebuilt-newer"),
        ):
            retained = HistoryStore(helper, accepted_path=directory / "accepted.sqlite")
            try:
                assert [record.run_id for record in retained.recover_interrupted()] == [
                    expected_run
                ]
            finally:
                retained.close()


@pytest.mark.parametrize("linked_parent", [False, True])
def test_engine_upload_staging_refuses_linked_directories(
    tmp_path: Path, linked_parent: bool
) -> None:
    library = tmp_path / "library"
    state = tmp_path / "state"
    port = free_port()
    command = serve_with_library_command(
        port, library, "--no-default-packs", "--engine-state-root", str(state)
    )
    process, log = start_logged_serve(command, tmp_path)
    pending_error: BaseException | None = None
    try:
        wait_for_composition(process, port)
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)
    parent = library / ".engine-uploads"
    selected = parent if linked_parent else next(parent.iterdir())
    retained = tmp_path / "retained"
    selected.rename(retained)
    pending = retained / ".latent-upload-pending"
    pending.write_bytes(b"do not sweep this directory")
    selected.symlink_to(retained, target_is_directory=True)
    refused = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert refused.returncode != 0
    assert "engine upload staging must not contain linked directories" in refused.stderr
    assert pending.read_bytes() == b"do not sweep this directory"
    assert not list(retained.glob(".media-upload-*"))


@pytest.mark.parametrize(
    "point", ["before-write", "partial-write", "before-publish", "after-publish"]
)
def test_engine_state_binding_recovers_after_abrupt_exit(
    tmp_path: Path, point: str, published_library: Path
) -> None:
    library = published_library
    state = tmp_path / "state"
    port = free_port()
    command = serve_with_library_command(
        port, library, "--no-default-packs", "--engine-state-root", str(state)
    )
    interrupted = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import os
import sys
from pathlib import Path
from dinkster.serve import main

point = sys.argv[1]
original_dump, original_replace = json.dump, os.replace

def dump(value, stream, *args, **kwargs):
    if isinstance(value, dict) and value.get("format") == "dinkster.engine-state/1":
        if point == "before-write":
            os._exit(73)
        if point == "partial-write":
            stream.write('{"format":')
            stream.flush()
            os.fsync(stream.fileno())
            os._exit(73)
    return original_dump(value, stream, *args, **kwargs)

def replace(source, destination):
    if Path(destination).name == "owner.json":
        if point == "after-publish":
            original_replace(source, destination)
        os._exit(73)
    return original_replace(source, destination)

json.dump, os.replace = dump, replace
main(sys.argv[2:])
""",
            point,
            *command[3:],
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert interrupted.returncode == 73, interrupted.stderr
    assert not (library / "history.sqlite").exists()
    assert not (state / "accepted.sqlite").exists()
    process, log = start_logged_serve(command, tmp_path)
    pending_error: BaseException | None = None
    try:
        wait_for_composition(process, port)
        assert (state / "accepted.sqlite").is_file()
        assert (state / "sessions.sqlite").is_file()
        assert (library / "records.sqlite").is_file()
        assert not (library / "history.sqlite").exists()
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


@pytest.mark.parametrize("hardlink", [False, True])
def test_engine_state_refuses_linked_pending_ownership(
    tmp_path: Path, hardlink: bool, published_library: Path
) -> None:
    library = published_library
    state = tmp_path / "state"
    state.mkdir()
    retained = tmp_path / "retained"
    retained.write_bytes(b"must not overwrite this file")
    staged = state / ".owner.pending"
    if hardlink:
        staged.hardlink_to(retained)
    else:
        staged.symlink_to(retained)
    result = subprocess.run(
        serve_with_library_command(
            free_port(), library, "--no-default-packs", "--engine-state-root", str(state)
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Cannot verify engine state ownership" in result.stderr
    assert retained.read_bytes() == b"must not overwrite this file"
    assert not (state / "owner.json").exists()
    assert not (library / "history.sqlite").exists()


@pytest.mark.parametrize("owner", [None, "not json", "[]"])
def test_engine_state_refuses_unbound_or_invalid_existing_state(
    tmp_path: Path, owner: str | None, published_library: Path
) -> None:
    library = published_library
    state = tmp_path / "state"
    state.mkdir()
    retained = state / "retained"
    retained.write_bytes(b"keep this engine's state")
    if owner is not None:
        (state / "owner.json").write_text(owner, encoding="utf-8")
    else:
        (state / ".owner.pending").write_text('{"format":', encoding="utf-8")
    result = subprocess.run(
        serve_with_library_command(
            free_port(), library, "--no-default-packs", "--engine-state-root", str(state)
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Cannot verify engine state ownership" in result.stderr
    assert retained.read_bytes() == b"keep this engine's state"
    assert not (library / "history.sqlite").exists()
    assert not (state / "sessions.sqlite").exists()
    if owner is not None:
        assert (state / "owner.json").read_text("utf-8") == owner
    else:
        assert not (state / "owner.json").exists()


@pytest.mark.parametrize("choice", ["relative", "inside", "ancestor", "linked", "disabled"])
def test_engine_state_refuses_invalid_location_before_opening_stores(
    tmp_path: Path,
    choice: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    library = tmp_path / "library"
    state = tmp_path / "engine"
    if choice == "relative":
        state = Path("relative-state")
    elif choice == "inside":
        state = library / "state"
    elif choice == "ancestor":
        state = tmp_path
    elif choice == "linked":
        target = tmp_path / "target"
        target.mkdir()
        state.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        sys,
        "argv",
        serve_with_library_command(free_port(), library, "--engine-state-root", str(state))[2:],
    )
    if choice == "disabled":
        monkeypatch.setattr(
            sys, "argv", serve_command(free_port(), "--engine-state-root", str(state))[2:]
        )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert "--engine-state-root" in capsys.readouterr().err
    assert not library.exists()


@pytest.mark.parametrize("abrupt", [False, True])
def test_library_owner_refuses_competitors_and_releases_on_engine_exit(
    tmp_path: Path, abrupt: bool
) -> None:
    from dinkster_store import library_lock

    library = tmp_path / "library"
    first_port = free_port()
    arguments = serve_with_library_command(first_port, library, "--no-default-packs")[3:]
    engine = "sha256:" + "ab" * 32
    owner = subprocess.Popen(
        [
            sys.executable,
            str(library_lock.__file__),
            str(library),
            "engine alpha",
            "--engine-state",
            str(tmp_path / "engine"),
            "1",
            json.dumps({"engine": engine, "readers": [engine]}),
            "--",
            *arguments,
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_composition(owner, first_port)
        marker = library / "library-format.json"
        published_format = marker.read_bytes()
        competitor = subprocess.run(
            serve_with_library_command(free_port(), library, "--no-default-packs"),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert competitor.returncode != 0
        assert "already in use by engine alpha" in competitor.stderr
        assert marker.read_bytes() == published_format
        assert owner.poll() is None
        if abrupt:
            owner.kill()
        else:
            owner.terminate()
        owner.wait(timeout=30)
        next_port = free_port()
        replacement = subprocess.Popen(
            serve_with_library_command(next_port, library, "--no-default-packs"),
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_composition(replacement, next_port)
        finally:
            replacement.terminate()
            replacement.wait(timeout=30)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=30)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_composition(
    process: subprocess.Popen[bytes], port: int, *, complete: bool = True
) -> dict[str, object]:
    async def wait() -> dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(
                            f"http://127.0.0.1:{port}/api/composition"
                        ) as response:
                            report = await response.json()
                        if not complete or "composing" not in report:
                            return report
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)

    return run_scenario(wait())


def assert_served_memory_budgets(
    process: subprocess.Popen[bytes], port: int, expected: dict[str, int], log: Path
) -> None:
    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/health") as resp:
                            if resp.status == 200:
                                break
                    except (aiohttp.ClientError, TimeoutError):
                        pass
                    await asyncio.sleep(0.05)
            async with session.get(base + "/memory/status") as resp:
                assert resp.status == 200
                report = (await resp.json())["memoryGovernor"]
            assert report is not None
            assert {device: entry["budgetBytes"] for device, entry in report.items()} == expected

    pending_error: BaseException | None = None
    try:
        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


def start_logged_serve(
    command: list[str], cwd: Path, *, env: dict[str, str] | None = None
) -> tuple[subprocess.Popen[bytes], Path]:
    log = cwd / "serve.log"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=dict(os.environ) if env is None else env,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    return process, log


def stop_logged_serve(
    process: subprocess.Popen[bytes], log: Path, pending_error: BaseException | None
) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    if pending_error is not None:
        pending_error.add_note(
            "server output:\n" + log.read_text(encoding="utf-8", errors="replace")
        )


@pytest.mark.parametrize(
    "pack_path", ["packages/dinkster-nodes-dev", "packages/dinkster-nodes-dev/dinkster-pack.toml"]
)
def test_serve_plain_dev_pack_is_announced(tmp_path: Path, pack_path: str) -> None:
    port = free_port()
    pack = Path(__file__).resolve().parent.parent / pack_path
    process, log = start_logged_serve(
        serve_with_library_command(
            port,
            tmp_path / "library",
            "--no-default-packs",
            "--pack",
            str(pack),
            "--engine-variant",
            "linux-cpu",
        ),
        tmp_path,
    )
    pending_error: BaseException | None = None
    try:
        report = wait_for_composition(process, port)
        packs = report["packs"]
        assert isinstance(packs, dict)
        assert packs["dinkster-nodes-dev"]["state"] == "announced"

        async def plan() -> None:
            workflow = {
                "format": "dinkster-workflow",
                "formatVersion": 1,
                "graphs": {"root": {"nodes": {}}},
            }
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"http://127.0.0.1:{port}/api/workflows/plan",
                    json={"workflows": [workflow]},
                ) as response:
                    assert response.status == 200, await response.text()
                    assert await response.json() == {"packs": [], "assets": [], "errors": []}

        run_scenario(plan())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


@pytest.mark.parametrize(
    ("variable", "arguments", "expected"),
    [
        (None, (), "production"),
        ("qa", (), "qa"),
        ("qa", ("--services", "production"), "production"),
        ("invalid", ("--services", "qa"), "qa"),
        (None, ("--remote-gateway-base", "http://127.0.0.1:9"), "custom"),
    ],
)
def test_serve_reports_process_local_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str | None,
    arguments: tuple[str, ...],
    expected: str,
) -> None:
    from dinkster.services import resolve_services

    if variable is None:
        monkeypatch.delenv("DINKSTER_SERVICES", raising=False)
    else:
        monkeypatch.setenv("DINKSTER_SERVICES", variable)
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DINKSTER_REMOTE_CATALOG_BASE", raising=False)
    monkeypatch.delenv("DINKSTER_REMOTE_GATEWAY_BASE", raising=False)
    monkeypatch.delenv("DINKSTER_ACCOUNT_IDENTITY_BASE", raising=False)
    port = free_port()
    process, log = start_logged_serve(
        serve_command(port, "--no-default-packs", *arguments), tmp_path
    )
    pending_error: BaseException | None = None

    async def scenario() -> None:
        async with aiohttp.ClientSession() as client:
            async with asyncio.timeout(30):
                while True:
                    assert process.poll() is None, log.read_text()
                    try:
                        async with client.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                            document = await response.json()
                        break
                    except aiohttp.ClientError:
                        await asyncio.sleep(0.05)
            services = document["dinkster"]["services"]
            hosted = resolve_services("qa" if expected == "qa" else "production")
            gateway = "http://127.0.0.1:9" if expected == "custom" else hosted.gateway_base
            assert services == {
                "name": expected,
                "identityBase": hosted.identity_base,
                "gatewayBase": gateway,
            }
            line = f"Services: {expected}; identity: {hosted.identity_base}; gateway: {gateway}"
            output = log.read_text()
            assert line in output
            assert f"dinkster.serve: {line}" in output

    try:
        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


@pytest.mark.parametrize("channel", ["stable", "nightly"])
def test_services_remain_process_local_across_generation_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    from dinkster_registry import Lockfile

    from dinkster.generation_packs import stage_generation_packs

    project = tmp_path / "project"
    generation = project / "builds" / "1"
    generation.mkdir(parents=True)
    stage_generation_packs(generation, Lockfile.of(()), {})
    (project / "environment.json").write_text(json.dumps({"channel": channel}))
    (generation / "build.json").write_text(
        json.dumps({"format": "dinkster.generation/1", "variant": "linux-cpu"})
    )
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "home"))
    for name in (
        "DINKSTER_SERVICES",
        "DINKSTER_REMOTE_CATALOG_BASE",
        "DINKSTER_REMOTE_GATEWAY_BASE",
        "DINKSTER_ACCOUNT_IDENTITY_BASE",
    ):
        monkeypatch.delenv(name, raising=False)
    before = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    for selected, arguments in (("qa", ("--services", "qa")), ("production", ())):
        port = free_port()
        process, log = start_logged_serve(
            serve_command(port, "--generation", str(generation), *arguments), tmp_path
        )
        pending_error: BaseException | None = None

        async def scenario(
            process: subprocess.Popen[bytes], log: Path, port: int, selected: str
        ) -> None:
            async with aiohttp.ClientSession() as client:
                async with asyncio.timeout(30):
                    while True:
                        assert process.poll() is None, log.read_text()
                        try:
                            async with client.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                                document = await response.json()
                            assert document["dinkster"]["services"]["name"] == selected
                            return
                        except aiohttp.ClientError:
                            await asyncio.sleep(0.05)

        try:
            run_scenario(scenario(process, log, port, selected))
        except BaseException as error:
            pending_error = error
            raise
        finally:
            stop_logged_serve(process, log, pending_error)
        after = {
            path.relative_to(project): path.read_bytes()
            for path in project.rglob("*")
            if path.is_file()
        }
        assert after == before


def test_serve_loads_memory_budgets_from_library_config(tmp_path: Path) -> None:
    (tmp_path / "memory.toml").write_text(
        '[budgets]\nram = "64M"\n"vram:cuda:0" = 33554432\n', "utf-8"
    )
    port = free_port()
    process, log = start_logged_serve(serve_with_library_command(port, tmp_path), tmp_path)
    assert_served_memory_budgets(
        process, port, {"ram": 64 * 1024**2, "vram:cuda:0": 32 * 1024**2}, log
    )


def test_cli_memory_budget_overrides_one_config_device(tmp_path: Path) -> None:
    (tmp_path / "memory.toml").write_text(
        '[budgets]\nram = "64M"\n"vram:cuda:0" = "32M"\n', "utf-8"
    )
    port = free_port()
    process, log = start_logged_serve(
        serve_with_library_command(port, tmp_path, "--memory-budget", "ram=96M"), tmp_path
    )
    assert_served_memory_budgets(
        process, port, {"ram": 96 * 1024**2, "vram:cuda:0": 32 * 1024**2}, log
    )


def test_settings_precedence_cli_over_persisted_over_memory_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "memory.toml").write_text(
        '[budgets]\nram = "64M"\n"vram:cuda:0" = "32M"\n', "utf-8"
    )
    (tmp_path / "settings.json").write_text(
        '{"memory-budgets":{"ram":"80M","vram:cuda:0":"48M"}}\n',
        "utf-8",
    )
    port = free_port()
    process, log = start_logged_serve(
        serve_with_library_command(port, tmp_path, "--memory-budget", "ram=96M"), tmp_path
    )
    assert_served_memory_budgets(
        process, port, {"ram": 96 * 1024**2, "vram:cuda:0": 48 * 1024**2}, log
    )


@pytest.mark.parametrize("no_defaults", [False, True])
@pytest.mark.parametrize("launcher", [False, True])
def test_library_startup_composes_without_pack_workers(
    tmp_path: Path, no_defaults: bool, launcher: bool
) -> None:
    import psutil

    from tools.benchmark_schema_catalog import ENTRY, bound_server, terminate_children

    entry = ENTRY
    port = free_port()
    extra = ("--no-default-packs",) if no_defaults else ()
    command = serve_with_library_command(port, tmp_path, *extra)
    command[1:3] = ["-c", entry]
    if launcher:
        command = [
            sys.executable,
            "-c",
            "import subprocess, sys; sys.exit(subprocess.Popen(sys.argv[1:]).wait())",
            *command,
        ]
    log = tmp_path / "serve.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=tmp_path, stdout=output, stderr=output)

    async def scenario() -> None:
        async with aiohttp.ClientSession() as session:
            try:
                async with asyncio.timeout(120):
                    while True:
                        assert process.poll() is None, log.read_text()
                        try:
                            async with session.get(
                                f"http://127.0.0.1:{port}/api/composition"
                            ) as resp:
                                report = await resp.json()
                            if not report.get("composing"):
                                break
                        except aiohttp.ClientError:
                            pass
                        await asyncio.sleep(0.05)
            except TimeoutError as error:
                error.add_note("server output:\n" + log.read_text())
                raise
            expected = set() if no_defaults else set(await _default_pack_names())
            assert set(report["packs"]) == expected, report
            assert all(pack["state"] == "announced" for pack in report["packs"].values()), report
            output = log.read_text()
            assert "BLOCKED_EXECUTION_IMPORT" not in output, output
            server = bound_server(process.pid, output.splitlines())
            if launcher:
                assert server.pid != process.pid
            children = server.children(recursive=True)
            # Windows virtualenvs can put a redirector above the helper interpreter.
            assert len(children) in ((1, 2) if sys.platform == "win32" else (1,)), children
            if len(children) == 2:
                assert children[1].ppid() == children[0].pid
            for child in children:
                assert child.cmdline()[1:] == [
                    "-m",
                    "dinkster_library",
                    "--library",
                    str(tmp_path),
                ]

    try:
        run_scenario(scenario())
    finally:
        children = terminate_children(process.pid)
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=15)
        _, alive = psutil.wait_procs(children, timeout=15)
        assert not alive, f"server descendants survived teardown: {alive}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory aliases such as macOS /var")
def test_library_startup_resolves_posix_directory_alias(tmp_path: Path) -> None:
    parent = tmp_path / "physical"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    port = free_port()
    process, log = start_logged_serve(
        serve_with_library_command(port, alias / "library", "--no-default-packs"), tmp_path
    )
    pending_error: BaseException | None = None
    try:
        report = wait_for_composition(process, port)
        assert report["packs"] == {}

        async def scenario() -> None:
            async with aiohttp.ClientSession() as client:
                async with client.get(f"http://127.0.0.1:{port}/api/health") as response:
                    assert response.status == 200
                    assert (await response.json())["ok"] is True

        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


@pytest.mark.parametrize("inventory", ["new", "missing", "corrupt", "pending", "linked-parent"])
def test_serve_cleanup_does_not_depend_on_durable_vault_inventory(
    tmp_path: Path, inventory: str
) -> None:
    from dinkster_assets.produced import ProducedAssetInventory

    library = tmp_path / "library"
    if inventory == "linked-parent":
        physical = tmp_path / "physical"
        physical.mkdir()
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(physical, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")
        library = alias / "library"
    if inventory not in {"new", "linked-parent"}:
        vault = library / "vault"
        vault.mkdir(parents=True)
        if inventory == "corrupt":
            (vault / ".produced-assets.sqlite").write_bytes(b"not a database")
        elif inventory == "pending":
            ProducedAssetInventory(vault).pending("blake3:" + "ab" * 32, 17)
    port = free_port()
    process, log = start_logged_serve(
        serve_with_library_command(port, library, "--no-default-packs"), tmp_path
    )
    error: BaseException | None = None
    try:

        async def scenario() -> None:
            base = f"http://127.0.0.1:{port}"
            async with aiohttp.ClientSession() as client:
                async with asyncio.timeout(60):
                    while True:
                        assert process.poll() is None, "serve process died"
                        try:
                            async with client.get(base + "/api/health") as response:
                                if response.status == 200:
                                    break
                        except aiohttp.ClientError:
                            pass
                        await asyncio.sleep(0.05)
                async with client.get(base + "/api/library/storage") as response:
                    assert response.status == 200
                    status = await response.json()
                assert status["adoptedBytes"] == 0
                assert status["vaultOwnershipAvailable"] is True
                assert status["collectionEnabled"]
                assert status["blocked"] == []
                if inventory == "corrupt":
                    assert (
                        library / "vault" / ".produced-assets.sqlite"
                    ).read_bytes() == b"not a database"
                elif inventory == "pending":
                    entries = ProducedAssetInventory(library / "vault").entries()
                    assert len(entries) == 1 and entries[0].adopted_at_ns is None
                elif inventory == "missing":
                    assert not (library / "vault" / ".produced-assets.sqlite").exists()

        run_scenario(scenario())
    except BaseException as exc:
        error = exc
        raise
    finally:
        stop_logged_serve(process, log, error)


def test_degraded_default_ordering_composes_provider_before_consumer(
    tmp_path: Path,
) -> None:
    """A broken pack must not make healthy consumers compose before providers."""
    import shutil

    from dinkster_server import PackInfo
    from test_compose import write_contract_manifest

    import dinkster.serve as serve
    from dinkster.compose import (
        CompositionError,
        PackSpec,
        ServingComposer,
        load_manifest,
        resolve_manifest_path,
    )

    provider = tmp_path / "provider"
    shutil.copytree(TESTS_DIR / "fixtures/pack-testing", provider)
    consumer = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.dependencies]\nexample-tally = ">=0.0.1,<1"\n',
        namespace="consumer",
        nodes_entry="consumer_nodes:NODES",
    )
    (consumer.parent / "consumer_nodes.py").write_text("""
from dinkster_api.v1 import Input, Node, NodeSchema, Output, TypeExpr
class Count(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="consumer.count",
            body=(Input("counts", TypeExpr.concrete("example-tally.counts")),
            Output("count", TypeExpr.concrete("INT")),),
        )
    @classmethod
    def execute(cls, *, counts):
        return cls.outputs(count=sum(counts.values()))
NODES = [Count]
""")
    broken = write_contract_manifest(
        tmp_path / "broken",
        "ordering-probe",
        '[pack.requirements.capabilities]\n"ordering.probe" = ">=1,<2"\n',
    )
    specs = [
        PackSpec(consumer, env={"PYTHONPATH": str(consumer.parent)}),
        PackSpec(
            provider,
            env={"PYTHONPATH": str(provider / "src")},
            packs={
                "example-tally": PackInfo(
                    display_name="Example tally",
                    version="0.0.1",
                    artifact_digest="blake3:" + "a" * 64,
                )
            },
        ),
        PackSpec(broken),
    ]
    composer = ServingComposer()

    async def scenario() -> None:
        # The probe's unresolvable capability makes full contract ordering
        # raise; the defaults must still compose providers before consumers.
        ordered = serve._order_default_pack_specs(composer, specs, len(specs))
        errors: dict[str, str] = {}
        deltas: dict[str, set[str]] = {}
        for spec in ordered:
            pack = load_manifest(resolve_manifest_path(spec.manifest)).name
            try:
                delta = await composer.add_pack(spec)
            except CompositionError as error:
                errors[pack] = str(error)
            else:
                deltas[pack] = set(delta.schemas)
        assert list(errors) == ["ordering-probe"]
        assert "ordering.probe" in errors["ordering-probe"]
        assert "example-tally.shout" in deltas["example-tally"]
        assert "consumer" in deltas, errors
        assert "consumer.count" in deltas["consumer"]

    async def compose_and_close() -> None:
        try:
            await scenario()
        finally:
            await composer.close()

    run_scenario(compose_and_close())


def prepare_tally_manifest(directory: Path) -> Path:
    import shutil

    from dinkster_workers.doctor import prepare_catalog

    root = directory / "default-pack"
    shutil.copytree(TESTS_DIR / "fixtures/pack-testing", root)
    manifest = root / "dinkster-pack.toml"
    assert prepare_catalog(manifest, environment={"PYTHONPATH": str(root / "src")}).ok
    return manifest


def fixture_default_serve_command(port: int, directory: Path, *args: str) -> list[str]:
    manifest = prepare_tally_manifest(directory)
    environment = {"PYTHONPATH": str(manifest.parent / "src")}
    bootstrap = f"""
from pathlib import Path
import sys
from dinkster import serve
from dinkster.compose import PackSpec

sys.path.insert(0, {str(manifest.parent / "src")!r})
serve.default_pack_ids = lambda: ("example-tally",)
serve.default_pack_spec = lambda _pack_id: PackSpec(Path({str(manifest)!r}), env={environment!r})
serve.main()
"""
    return [sys.executable, "-c", bootstrap, *serve_command(port, *args)[3:]]


def test_serve_progressive_pack_announcement(tmp_path: Path) -> None:
    """The default and explicit packs announce after the diagnostic host binds."""
    manifest = write_iso_manifest(tmp_path / "pack")
    default_packs = ("example-tally",)
    final_epoch = len(default_packs) + 2
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR)}
    process, log = start_logged_serve(
        fixture_default_serve_command(
            port,
            tmp_path,
            "--pack",
            str(manifest.relative_to(tmp_path)),
            "--event-loop-stall-threshold",
            "4",
        ),
        tmp_path,
        env=env,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # The port answers; composition may still be in flight.
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/health") as resp:
                            if resp.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            # Default packs may already have announced by the first
            # fetch; they are still ordinary first-party packs, not core.
            async with session.get(base + "/api/nodes") as resp:
                data = await resp.json()
            # A server without the development manifest never serves its scaffolding.
            assert not any(t.startswith("dev.") for t in data["nodes"])
            # All packs land as live announcements after the epoch-1 core.
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    async with session.get(base + "/api/nodes") as resp:
                        data = await resp.json()
                    if "iso.chatty" in data["nodes"] and "composing" not in data:
                        break
                    await asyncio.sleep(0.1)
            async with session.get(base + "/api/composition") as resp:
                composition = await resp.json()
            assert data["epoch"] == final_epoch, composition
            assert data["nodes"]["example-tally.shout"]["pack"] == "example-tally"
            assert data["nodes"]["example-tally.tally"]["pack"] == "example-tally"
            assert data["nodes"]["iso.chatty"]["pack"] == "isopack"
            assert "dinkster.route.switch_by_name" not in data["nodes"]
            # Fully composed, positively: the last announcement and the
            # narration clear land atomically on the event loop, so a
            # table containing the final pack never claims "composing".
            assert "composing" not in data
            async with session.get(base + "/api/health") as resp:
                body = await resp.json()
            assert body == {
                "ok": True,
                "compositionState": {
                    "composed": len(data["nodes"]),
                    "failed": 0,
                    "epoch": final_epoch,
                },
            }
            # The pollable report tells the same story: the pack announced
            # at their respective epochs, with nothing pending.
            async with session.get(base + "/api/composition") as resp:
                report = await resp.json()
            assert report["epoch"] == final_epoch
            assert "composing" not in report
            expected_packs = {
                name: {"state": "announced", "epoch": index + 2}
                for index, name in enumerate(default_packs)
            }
            expected_packs["isopack"] = {"state": "announced", "epoch": final_epoch}
            assert "StringLength" not in data["nodes"]
            assert "dinkster.ksampler" not in data["nodes"]
            assert report["packs"] == expected_packs

    pending_error: BaseException | None = None
    try:
        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


@pytest.mark.parametrize("failure_first", [True, False])
@pytest.mark.timeout(360)
def test_unavailable_default_pack_is_reported_after_diagnostic_host_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_first: bool
) -> None:
    from aiohttp.test_utils import TestClient, TestServer

    from dinkster import serve
    from dinkster.compose import PackSpec

    manifest = prepare_tally_manifest(tmp_path)
    failed_pack, surviving_pack = "isopack", "example-tally"
    present_node, missing_node = "example-tally.shout", "iso.chatty"
    defaults = (failed_pack, surviving_pack) if failure_first else (surviving_pack, failed_pack)
    observed: dict[str, object] = {}

    def fail_one_default(pack_id: str) -> PackSpec:
        if pack_id == failed_pack:
            raise RuntimeError("default pack bytes unavailable")
        return PackSpec(manifest, env={"PYTHONPATH": str(manifest.parent / "src")})

    monkeypatch.setattr(serve, "default_pack_ids", lambda: defaults)

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                # Full default composition under hosted coverage can exceed 120 s.
                async with asyncio.timeout(300):
                    while True:
                        report = await (await client.get("/api/composition")).json()
                        entry = report["packs"][failed_pack]
                        survivor = report["packs"][surviving_pack]
                        if (
                            entry["state"] == "failed"
                            and survivor["state"] == "announced"
                            and all(row["state"] != "pending" for row in report["packs"].values())
                        ):
                            observed.update(entry)
                            break
                        await asyncio.sleep(0.05)
                async with asyncio.timeout(300):
                    while "composing" in await (await client.get("/api/composition")).json():
                        await asyncio.sleep(0.05)
                nodes = await (await client.get("/api/nodes")).json()
                assert missing_node not in nodes["nodes"]
                assert nodes["nodes"][present_node]["pack"] == surviving_pack
                health = await (await client.get("/api/health")).json()
                failed_count = sum(row["state"] == "failed" for row in report["packs"].values())
                assert health == {
                    "ok": False,
                    "compositionState": {
                        "composed": len(nodes["nodes"]),
                        "failed": failed_count,
                        "epoch": nodes["epoch"],
                    },
                }
            finally:
                await client.close()

        run_scenario(inspect(), timeout=330)

    monkeypatch.setattr(serve, "default_pack_spec", fail_one_default)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])

    catalog_base = os.environ.get("DINKSTER_REMOTE_CATALOG_BASE")
    gateway_base = os.environ.get("DINKSTER_REMOTE_GATEWAY_BASE")
    assert catalog_base == ""
    assert gateway_base == ""
    serve.main()

    assert observed == {
        "state": "failed",
        "error": "default pack bytes unavailable",
    }


@pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="requires Linux /proc")
@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
def test_serve_signal_reaps_unresponsive_children_concurrently(
    tmp_path: Path, shutdown_signal: signal.Signals
) -> None:
    manifests = [write_shutdown_pack(tmp_path / f"pack-{index}", index) for index in range(3)]
    expected_children = len(manifests)
    args = [arg for manifest in manifests for arg in ("--pack", str(manifest))]
    port = free_port()
    bootstrap = f"""
import asyncio
import dinkster_workers.isolated as isolated
from dinkster.serve import main

isolated._CHILD_SHUTDOWN_GRACE = 0.05
original_close = isolated.IsolatedWorker._close
closing = set()
all_closing = asyncio.Event()

async def close_together(worker):
    if worker._proc is not None:
        closing.add(worker._proc.pid)
        if len(closing) == {expected_children}:
            all_closing.set()
        await asyncio.wait_for(all_closing.wait(), 8)
    await original_close(worker)

isolated.IsolatedWorker._close = close_together
main()
"""
    stderr_path = tmp_path / "serve.stderr.log"
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                bootstrap,
                *serve_command(port, "--no-default-packs", *args)[3:],
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(str(manifest.parent) for manifest in manifests),
            },
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    children: tuple[int, ...] = ()

    try:
        report = wait_for_composition(process, port)
        packs = report["packs"]
        assert isinstance(packs, dict)
        assert all(isinstance(row, dict) and row["state"] == "announced" for row in packs.values())
        children = child_pids(process)
        assert len(children) == expected_children
        for pid in children:
            os.kill(pid, _SIGSTOP)

        process.send_signal(shutdown_signal)
        process.wait(timeout=8)

        assert process.returncode == 0
        assert not any(process_exists(pid) for pid in children)
        with socket.socket() as released:
            released.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        for pid in children:
            if process_exists(pid):
                os.kill(pid, _SIGKILL)


@pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="requires Linux /proc")
def test_serve_signal_during_progressive_startup_reaps_partial_child(tmp_path: Path) -> None:
    manifest = write_shutdown_pack(tmp_path / "slow-pack", 0, startup_delay=30)
    expected_children = 1
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--no-default-packs", "--pack", str(manifest)),
        env={**os.environ, "PYTHONPATH": str(manifest.parent)},
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    children: tuple[int, ...] = ()
    try:
        report = wait_for_composition(process, port, complete=False)
        packs = report["packs"]
        assert isinstance(packs, dict)
        assert packs["shutdown-pack-0"] == {"state": "pending"}
        deadline = time.monotonic() + 60
        while len(children) < expected_children:
            assert time.monotonic() < deadline
            children = child_pids(process)
            time.sleep(0.01)
        assert len(children) == expected_children

        process.send_signal(signal.SIGTERM)
        process.wait(timeout=8)

        assert process.returncode == 0
        assert not any(process_exists(pid) for pid in children)
        with socket.socket() as released:
            released.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        for pid in children:
            if process_exists(pid):
                os.kill(pid, _SIGKILL)


@pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="requires Linux /proc")
def test_serve_signal_reaps_already_dead_child(tmp_path: Path) -> None:
    manifest = write_shutdown_pack(tmp_path / "pack", 0)
    expected_children = 1
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--no-default-packs", "--pack", str(manifest)),
        env={**os.environ, "PYTHONPATH": str(manifest.parent)},
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    children: tuple[int, ...] = ()
    try:
        wait_for_composition(process, port)
        children = child_pids(process)
        assert len(children) == expected_children
        os.kill(children[-1], _SIGKILL)

        process.send_signal(signal.SIGTERM)
        process.wait(timeout=8)

        assert process.returncode == 0
        assert not any(process_exists(pid) for pid in children)
        with socket.socket() as released:
            released.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        for pid in children:
            if process_exists(pid):
                os.kill(pid, _SIGKILL)


def test_serve_pack_failure_recorded_and_survivors_serve(tmp_path: Path) -> None:
    """A pack that cannot compose (reserved name) does NOT take the process
    down by default: the failure is cached on /api/composition with its
    error, the healthy pack keeps serving, and composition still completes -
    positively, with the failure on the record instead of a dead server
    (the anti-ComfyUI: broken extensions were just silently absent)."""
    manifest = write_iso_manifest(tmp_path / "pack", name="core")
    port = free_port()
    process = subprocess.Popen(
        fixture_default_serve_command(port, tmp_path, "--pack", str(manifest)),
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/composition") as resp:
                            if resp.status == 200:
                                report = await resp.json()
                                if "composing" not in report:
                                    break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            entry = report["packs"]["core"]
            assert entry["state"] == "failed"
            assert "reserved" in entry["error"]
            # The survivor serves, and the table says
            # complete - the failure is on the record, not on the surface.
            async with session.get(base + "/api/nodes") as resp:
                data = await resp.json()
            assert data["nodes"]["example-tally.shout"]["pack"] == "example-tally"
            assert report["packs"]["example-tally"]["state"] == "announced"
            assert "composing" not in data
            assert process.poll() is None, "pack failure must not kill serve"

    try:
        run_scenario(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def test_serve_retracts_schema_owner_without_execution_provider(tmp_path: Path) -> None:
    manifest = write_shutdown_pack(tmp_path / "pack", 99)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'namespaces = ["shutdown99"]\n',
            'namespaces = ["shutdown99"]\nschema-only = ["shutdown99.probe"]\n',
        ),
        encoding="utf-8",
    )
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--no-default-packs", "--pack", str(manifest)),
        env={**os.environ, "PYTHONPATH": str(manifest.parent)},
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def nodes() -> dict[str, object]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/nodes") as response:
                return await response.json()

    try:
        report = wait_for_composition(process, port)
        packs = report["packs"]
        assert isinstance(packs, dict)
        entry = packs["shutdown-pack-99"]
        assert isinstance(entry, dict)
        assert entry["state"] == "failed"
        error = entry["error"]
        assert isinstance(error, str)
        assert "shutdown99.probe" in error
        served = run_scenario(nodes())
        served_nodes = served["nodes"]
        assert isinstance(served_nodes, dict)
        assert "shutdown99.probe" not in served_nodes
        assert process.poll() is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def test_serve_strict_packs_failure_exits_nonzero(tmp_path: Path) -> None:
    """--strict-packs restores configuration-is-fatal: a pack that cannot
    compose aborts the whole process loudly after the port opened -
    nonzero exit, the failure on stderr (the CI/reproducibility mode)."""
    manifest = write_iso_manifest(tmp_path / "pack", name="core")
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--no-default-packs", "--pack", str(manifest), "--strict-packs"),
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        stderr = b""
        deadline = time.monotonic() + 60
        while process.poll() is None:
            assert time.monotonic() < deadline, "serve process did not exit"
            time.sleep(0.05)
        assert process.stderr is not None
        stderr = process.stderr.read()
        assert process.returncode != 0
        assert b"composition failed" in stderr
        assert b"reserved" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        if process.stderr is not None:
            process.stderr.close()


def test_serve_runtime_mounts_end_to_end(
    tmp_path: Path, shared_model_folders: dict[str, Path]
) -> None:
    """Real dinkster-serve with a library root and --allow-mount-changes:
    grant a directory over the API, watch it scan to ready, browse its
    catalog, stream a mounted file by digest, and revoke it - the desktop
    folder-picker flow end to end, with helper-owned durable grants."""
    from dinkster_assets import MountConfiguration
    from dinkster_library import LibraryClient

    from dinkster.project_folders import ProjectFolders

    library_root = tmp_path / "library"
    library_root.mkdir()
    projects = ProjectFolders(library_root / "project-folders.json")
    projects.initialize(())
    projects.update("film", [], available=(), output=None)
    granted = tmp_path / "granted"
    granted.mkdir(parents=True)
    payload = b"mounted-bytes"
    (granted / "pic.png").write_bytes(payload)
    port = free_port()
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
            str(library_root),
            "--allow-mount-changes",
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/health") as resp:
                            if resp.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            async with session.get(base + "/api/mounts") as resp:
                listing = await resp.json()
                rows = {row["id"]: row for row in listing["mounts"]}
                assert set(rows) == {"imports", "output", *shared_model_folders}
                for mount_id, path in shared_model_folders.items():
                    assert rows[mount_id]["path"] == str(path)
                    assert rows[mount_id]["mode"] == "readwrite"
                    assert rows[mount_id]["source"] == "shared-models"
                assert listing["outputMount"] == "output"
                output = rows["output"]
                assert output["path"] == str(library_root / "output")
                assert output["mode"] == "readwrite"
                imports = rows["imports"]
                assert imports["id"] == "imports"
                assert imports["path"] == str(library_root / "imports")
                assert imports["mode"] == "readwrite"
            async with session.post(base + "/api/projects", json={"id": "film"}) as resp:
                assert resp.status == 200, await resp.text()
            async with session.get(base + "/api/mounts", params={"project": "film"}) as resp:
                listing = await resp.json()
                assert {row["id"] for row in listing["mounts"]} == {
                    "imports",
                    "output",
                    *shared_model_folders,
                }
                assert listing["outputMount"] is None
            assert (library_root / "imports").is_dir()
            assert (library_root / "output").is_dir()
            async with asyncio.timeout(60):
                while imports["state"] != "ready":
                    await asyncio.sleep(0.05)
                    async with session.get(base + "/api/mounts") as resp:
                        rows = {row["id"]: row for row in (await resp.json())["mounts"]}
                        imports = rows["imports"]
            from io import BytesIO

            from PIL import Image

            image = BytesIO()
            Image.new("RGB", (2, 1), (17, 42, 93)).save(image, format="PNG")
            async with session.post(
                base + "/api/assets/media",
                params={
                    "scope": "local",
                    "kind": "media/image",
                    "name": "dropped.png",
                    "project": "film",
                },
                data=image.getvalue(),
                headers={"Content-Type": "image/png"},
            ) as resp:
                assert resp.status == 201, await resp.text()
                imported = (await resp.json())["asset"]
            async with session.get(
                base + "/api/mounts/locations",
                params={"scope": "local", "project": "film", "digest": imported["digest"]},
            ) as resp:
                assert resp.status == 200
                assert (await resp.json())["locations"] == [
                    {
                        "storage": "mount",
                        "mount": "imports",
                        "virtualPath": "mounts/imports/dropped.png",
                        "path": str(library_root / "imports" / "dropped.png"),
                    }
                ]
            assert imported["virtualPath"] == "mounts/imports/dropped.png"
            assert (library_root / "imports" / "dropped.png").read_bytes() == image.getvalue()
            from dinkster_assets import AssetVault

            assert not AssetVault(library_root / "vault").has(imported["digest"])
            async with session.get(base + "/api/assets/" + imported["digest"]) as resp:
                assert resp.status == 200
                assert await resp.read() == image.getvalue()
            async with session.get(
                base + "/api/assets/availability",
                params={"scope": "local", "digest": imported["digest"]},
            ) as resp:
                assert resp.status == 200
                assert await resp.json() == {
                    "assets": [{"digest": imported["digest"], "status": "available"}]
                }
            async with session.post(
                base + "/api/mounts", json={"id": "shots", "path": str(granted)}
            ) as resp:
                assert resp.status == 201
                created = await resp.json()
            assert (created["id"], created["source"]) == ("shots", "config")
            # Durable immediately, ready shortly after the background scan.
            with LibraryClient(library_root) as helper:
                assert "shots" in {mount.id for mount in MountConfiguration.load(helper).mounts}
            async with asyncio.timeout(60):
                while True:
                    async with session.get(base + "/api/mounts") as resp:
                        rows = {row["id"]: row for row in (await resp.json())["mounts"]}
                        assert set(rows) == {"imports", "output", "shots", *shared_model_folders}
                        row = rows["shots"]
                    if row["state"] == "ready":
                        break
                    await asyncio.sleep(0.05)
            assert row["entryCount"] == 1
            snapshots = list(
                Path(os.environ["DINKSTER_HOME"]).glob(
                    "engine-state/*/worker-state/mounts-snapshot.json"
                )
            )
            assert len(snapshots) == 1
            snapshot = snapshots[0]
            published = json.loads(snapshot.read_text("utf-8"))["mounts"]
            shots = next(item for item in published if item["id"] == "shots")
            assert Path(shots["index"]).is_relative_to(snapshot.parent)
            assert json.loads(Path(shots["index"]).read_text("utf-8"))["pic.png"]["size"] == len(
                payload
            )
            async with session.get(base + "/api/mounts/shots/entries") as resp:
                (entry,) = (await resp.json())["entries"]
            assert entry["virtualPath"] == "mounts/shots/pic.png"
            # The browsed file previews by digest with no vault copy.
            async with session.get(base + "/api/assets/" + entry["digest"]) as resp:
                assert resp.status == 200
                assert await resp.read() == payload
            (granted / "selected.txt").write_bytes(b"selected after the folder scan")
            (granted / "unselected.txt").write_bytes(b"not selected")
            for _ in range(2):
                async with session.post(
                    base + "/api/mounts/resolve-path",
                    json={
                        "path": str(granted / "selected.txt"),
                        "project": "default",
                        "scope": "local",
                    },
                ) as resp:
                    assert resp.status == 200
                    selected = await resp.json()
                    assert selected["status"] == "resolved"
                    assert selected["asset"]["virtualPath"] == "mounts/shots/selected.txt"
                async with session.get(base + "/api/assets/" + selected["asset"]["digest"]) as resp:
                    assert resp.status == 200
                    assert await resp.read() == b"selected after the folder scan"
            rows = json.loads(Path(shots["index"]).read_text("utf-8"))
            assert set(rows) == {"pic.png", "selected.txt"}
            assert rows["pic.png"].get("missing") is not True
            async with session.delete(base + "/api/mounts/shots") as resp:
                assert resp.status == 200
            async with session.get(base + "/api/mounts") as resp:
                assert {row["id"] for row in (await resp.json())["mounts"]} == {
                    "imports",
                    "output",
                    *shared_model_folders,
                }
            async with session.get(base + "/api/assets/" + entry["digest"]) as resp:
                assert resp.status == 404
            async with session.get(
                base + "/api/assets/availability",
                params={"scope": "local", "digest": entry["digest"]},
            ) as resp:
                assert resp.status == 200
                assert await resp.json() == {
                    "assets": [{"digest": entry["digest"], "status": "missing"}]
                }
            with LibraryClient(library_root) as helper:
                assert "shots" not in {mount.id for mount in MountConfiguration.load(helper).mounts}
            assert not (library_root / "mounts.toml").exists()

    try:
        run_scenario(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


async def _job_finish(session: aiohttp.ClientSession, base: str, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(60):
        while True:
            async with session.get(base + f"/api/jobs/e2e/{job_id}") as resp:
                status: dict[str, object] = await resp.json()
            if status["state"] in ("completed", "failed", "cancelled"):
                return status
            await asyncio.sleep(0.05)


async def _peek_value(
    session: aiohttp.ClientSession, base: str, job_id: str, output_id: str
) -> object:
    url = base + "/api/values?clientId=e2e&jobId=" + job_id + "&nodeId=save&outputId=" + output_id
    async with session.get(url) as resp:
        assert resp.status == 200
        data = await resp.json()
    assert data["available"] is True
    return data["descriptor"]["value"]


@pytest.mark.parametrize("external", [True, False])
@pytest.mark.parametrize("source", ["defaults", "generation"])
def test_sandboxed_serve_rotates_remote_bearer_without_exposing_private_credentials(
    tmp_path: Path,
    external: bool,
    source: str,
) -> None:
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_workers import detect_bubblewrap
    from dinkster_workers.doctor import prepare_catalog

    capability = detect_bubblewrap()
    if not capability.available:
        pytest.skip("bubblewrap jail cannot be built here: " + capability.detail)
    from dinkster.account import AccountSession
    from dinkster.services import resolve_services

    credentials = tmp_path / "credentials"
    credentials.mkdir()
    install = tmp_path / "install"
    endpoint = "http://127.0.0.1:8123"
    account = AccountSession(
        resolve_services(identity_base=endpoint, catalog_base=endpoint),
        home=credentials,
        install_root=install,
    )
    token = credentials / "bearer" if external else account.token_path
    neighbor = credentials / "private-oauth.json" if external else account.state_path
    token.parent.mkdir(parents=True, exist_ok=True)
    neighbor.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("first-external-bearer")
    neighbor.write_text(
        json.dumps(
            {
                "format": "dinkster.account/1",
                "project_id": "89790f18-fdad-4f48-bf67-823fdddb89da",
                "gateway_expires_at": time.time() + 3600,
                "refresh_token": "must-stay-private",
            }
        )
    )
    if external:
        account.token_path.parent.mkdir(parents=True)
        account.token_path.write_text("account-bearer-must-not-win")
        account.state_path.parent.mkdir(parents=True)
        account.state_path.write_text(neighbor.read_text())
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname="dinkster-nodes-remote"\npublisher="dinkster"\n'
        'namespaces=["credentialprobe"]\n[pack.entry]\nnodes="credential_probe:NODES"\n'
    )
    (pack / "credential_probe.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, Output, TypeExpr\n"
        "class Probe(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='credentialprobe.read', idempotent=False, "
        "body=(Output('report', TypeExpr.concrete(CORE_STRING)),))\n"
        "    @classmethod\n"
        "    async def execute(cls):\n"
        "        token = Path(os.environ['DINKSTER_REMOTE_AUTH_TOKEN_FILE'])\n"
        "        bearer = token.read_text() if token.exists() else '<missing>'\n"
        f"        private = any(Path(p).exists() for p in "
        f"{[str(neighbor), str(account.state_path)]!r})\n"
        "        return cls.outputs(report=json.dumps([bearer, private, os.getpid(), "
        "os.environ.get('DINKSTER_REMOTE_PARTNER_MODE'), "
        "os.environ.get('DINKSTER_REMOTE_FAKE_OUTCOME')]))\n"
        "NODES = (Probe,)\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(pack), *(entry for entry in sys.path if Path(entry).name == "src")]
        )
    }
    assert prepare_catalog(manifest, environment=environment).ok
    bootstrap = (
        "from pathlib import Path\nfrom dinkster import serve\n"
        "from dinkster.compose import PackSpec\n"
        "serve.default_pack_ids = lambda: ('dinkster-nodes-remote',)\n"
        f"serve.default_pack_spec = lambda _: PackSpec(Path({str(manifest)!r}), "
        f"env={environment!r})\nserve.main()\n"
    )
    extra: tuple[str, ...] = ()
    if source == "generation":
        import shutil
        import venv

        from dinkster_registry import LockedPack, Lockfile, build_artifact

        from dinkster.generation_packs import stage_generation_packs

        archive = tmp_path / "remote.zip"
        digest = build_artifact(pack, archive)
        generation = tmp_path / "generation"
        generation.mkdir()
        pack_id = "dinkster-nodes-remote"
        lock = Lockfile.of([LockedPack(pack_id, "1.0.0", digest, "dinkster", ("credentialprobe",))])
        stage_generation_packs(generation, lock, {pack_id: archive})
        (generation / "build.json").write_text(
            json.dumps({"format": "dinkster.generation/1", "variant": "linux-cpu"})
        )
        runtime = generation / "packs" / pack_id
        venv.EnvBuilder().create(runtime)
        site_packages = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        for entry in sys.path:
            root = Path(entry)
            if root.name == "src" and root.is_dir():
                shutil.copytree(root, site_packages, dirs_exist_ok=True)
        (site_packages / "runtime-dependencies.pth").write_text(
            str(
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            + "\n"
        )
        staged = generation / "pack-sources" / pack_id
        assert prepare_catalog(
            staged / "dinkster-pack.toml", environment={"PYTHONPATH": str(staged)}
        ).ok
        extra = ("--generation", str(generation))
    port = free_port()
    process, log = start_logged_serve(
        [
            sys.executable,
            "-c",
            bootstrap,
            *serve_command(port)[3:],
            *extra,
            "--sandbox-packs",
            *(["--remote-auth-token-file", str(token)] if external else []),
            "--account-root",
            str(install),
            "--account-identity-base",
            endpoint,
            "--remote-catalog-base",
            endpoint,
        ],
        tmp_path,
        env={
            **os.environ,
            "DINKSTER_HOME": str(credentials),
            "DINKSTER_REMOTE_PARTNER_MODE": "fake",
            "DINKSTER_REMOTE_FAKE_OUTCOME": "slow",
        },
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as session:
            serial = 0

            async def read() -> list[object]:
                nonlocal serial
                serial += 1
                job = f"bearer-{serial}"
                graph = Graph(nodes={"save": GraphNode("credentialprobe.read", {})})
                async with session.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "e2e",
                        "jobId": job,
                        "graph": graph_to_wire(graph),
                        "targets": ["save"],
                    },
                ) as response:
                    assert response.status == 202, await response.text()
                status = await _job_finish(session, base, job)
                assert status["state"] == "completed", status
                value = await _peek_value(session, base, job, "report")
                assert isinstance(value, str)
                return json.loads(value)

            first = await read()
            assert first[:2] == ["first-external-bearer", False]
            assert first[3:] == ["fake", "slow"]
            replacement = token.with_suffix(".new")
            replacement.write_text("second-external-bearer")
            replacement.replace(token)
            async with asyncio.timeout(10):
                while (rotated := await read())[0] != "second-external-bearer":
                    await asyncio.sleep(0.1)
            assert rotated == ["second-external-bearer", False, first[2], "fake", "slow"]
            if external:
                token.unlink()
            else:
                account.logout()
            async with asyncio.timeout(10):
                while (revoked := await read())[0] != "<missing>":
                    await asyncio.sleep(0.1)
            assert revoked == ["<missing>", False, first[2], "fake", "slow"]

    pending_error: BaseException | None = None
    try:
        report = wait_for_composition(process, port)
        packs = report["packs"]
        assert isinstance(packs, dict)
        assert packs["dinkster-nodes-remote"]["state"] == "announced", report
        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


def test_serve_mounted_save_end_to_end(
    tmp_path: Path, shared_model_folders: dict[str, Path]
) -> None:
    """The write path, real process to real bytes: grant a readwrite mount
    over the API while dinkster-serve runs, execute the PGM saver in
    that same process, verify the file landed inside the mount with the
    digest the job reported, stream it back via /api/assets, then revoke
    the mount and prove the very next save job fails - no restart
    anywhere."""
    from dinkster_assets import MountConfiguration, digest_bytes
    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
    from dinkster_library import LibraryClient

    library_root = tmp_path / "library"
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    pack = tmp_path / "mounted-save"
    pack.mkdir()
    manifest = pack / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "mounted-save"\npublisher = "dinkster"\n'
        'type-namespaces = ["dev"]\n[pack.entry]\n'
        'nodes = "mounted_save:NODES"\n'
        'types = "dinkster_nodes_dev.image:register_dev_types"\n'
    )
    (pack / "mounted_save.py").write_text(
        "from dataclasses import replace\n"
        "from dinkster_nodes_dev.image import GradientImage, SaveImagePGM\n"
        "class Gradient(GradientImage):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return replace(super().define_schema(), node_type='mounted-save.gradient')\n"
        "class Save(SaveImagePGM):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return replace(super().define_schema(), node_type='mounted-save.save_pgm')\n"
        "NODES = [Gradient, Save]\n"
    )
    port = free_port()
    process, log = start_logged_serve(
        [
            sys.executable,
            "-m",
            "dinkster.serve",
            "--no-default-packs",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--library-root",
            str(library_root),
            "--allow-mount-changes",
            "--pack",
            str(manifest),
            "--event-loop-stall-threshold",
            "4",
        ],
        tmp_path,
        env={**os.environ, "PYTHONPATH": str(pack)},
    )

    def save_job(job_id: str) -> dict[str, object]:
        graph = Graph(
            nodes={
                "g": GraphNode("mounted-save.gradient", {"width": 3, "height": 2}),
                "save": GraphNode(
                    "mounted-save.save_pgm",
                    {
                        "image": Link("g", "image"),
                        "target": {"mount": "renders", "prefix": "shots/frame"},
                    },
                ),
            }
        )
        return {
            "clientId": "e2e",
            "jobId": job_id,
            "graph": graph_to_wire(graph),
            "targets": ["save"],
        }

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "serve process died"
                    try:
                        async with session.get(base + "/api/health") as resp:
                            if resp.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)

            async with asyncio.timeout(60):
                while True:
                    async with session.get(base + "/api/nodes?wire=43") as resp:
                        nodes = (await resp.json())["nodes"]
                    if "mounted-save.save_pgm" in nodes:
                        break
                    await asyncio.sleep(0.05)

            # Before any grant: the save refuses with the mount's name in
            # the error - never a silent fallback directory.
            async with session.post(base + "/api/jobs", json=save_job("j0")) as resp:
                assert resp.status == 202
            status = await _job_finish(session, base, "j0")
            assert status["state"] == "failed"
            assert "renders" in str(status["error"])

            # Grant the directory readwrite while the process runs.
            async with session.post(
                base + "/api/mounts",
                json={"id": "renders", "path": str(outdir), "mode": "readwrite"},
            ) as resp:
                assert resp.status == 201
            async with asyncio.timeout(60):
                while True:
                    async with session.get(base + "/api/mounts") as resp:
                        rows = {row["id"]: row for row in (await resp.json())["mounts"]}
                        assert set(rows) == {"imports", "output", "renders", *shared_model_folders}
                        row = rows["renders"]
                    if row["state"] == "ready":
                        break
                    await asyncio.sleep(0.05)
            assert row["mode"] == "readwrite"
            # The grant is durable, mode included.
            with LibraryClient(library_root) as helper:
                grants = {mount.id: mount for mount in MountConfiguration.load(helper).mounts}
                assert grants["renders"].path == outdir
                assert grants["renders"].mode == "readwrite"
            assert not (library_root / "mounts.toml").exists()

            # Execute the save for real, in the running server.
            async with session.post(base + "/api/jobs", json=save_job("j1")) as resp:
                assert resp.status == 202
            status = await _job_finish(session, base, "j1")
            assert status["state"] == "completed", status.get("error")
            path_value = await _peek_value(session, base, "j1", "path")
            digest_value = await _peek_value(session, base, "j1", "digest")
            assert path_value == "mounts/renders/shots/frame_00001.pgm"

            # The bytes on disk are the job's bytes: counter-named PGM
            # inside the mount, digest agreeing with the reported one.
            landed = outdir / "shots" / "frame_00001.pgm"
            payload = landed.read_bytes()
            assert payload.startswith(b"P5 3 2 255\n")
            assert len(payload) == len(b"P5 3 2 255\n") + 6  # 3x2 gray pixels
            assert digest_value == digest_bytes(payload)

            # Immediately streamable by digest - the writes sidecar makes
            # the just-saved file resolvable before any rescan.
            async with session.get(base + "/api/assets/" + str(digest_value)) as resp:
                assert resp.status == 200
                assert await resp.read() == payload

            # A second run allocates the next counter name, never overwrites.
            async with session.post(base + "/api/jobs", json=save_job("j2")) as resp:
                assert resp.status == 202
            status = await _job_finish(session, base, "j2")
            assert status["state"] == "completed", status.get("error")
            assert await _peek_value(session, base, "j2", "path") == (
                "mounts/renders/shots/frame_00002.pgm"
            )
            assert (outdir / "shots" / "frame_00002.pgm").read_bytes() == payload

            # Revoke and save again in the same process: refusal, and the
            # already-written files stay untouched on disk.
            async with session.delete(base + "/api/mounts/renders") as resp:
                assert resp.status == 200
            async with session.post(base + "/api/jobs", json=save_job("j3")) as resp:
                assert resp.status == 202
            status = await _job_finish(session, base, "j3")
            assert status["state"] == "failed"
            assert "renders" in str(status["error"])
            assert sorted(p.name for p in (outdir / "shots").iterdir()) == [
                ".dinkster-asset-writer.lock",
                "frame_00001.pgm",
                "frame_00002.pgm",
            ]

    pending_error: BaseException | None = None
    try:
        run_scenario(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)
