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
import tomllib
from pathlib import Path
from unittest.mock import Mock

import aiohttp
import pytest
from packaging.requirements import Requirement
from test_compose import write_iso_manifest

TESTS_DIR = Path(__file__).parent
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
pytestmark = pytest.mark.usefixtures("installed_default_catalogs")


@pytest.fixture(autouse=True)
def _use_current_python_for_standard_pack_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_SERVING_PYTHON", sys.executable)


async def _default_pack_names() -> tuple[str, ...]:
    from dinkster_workers import load_manifest

    from dinkster.comfy_compose import comfy_compat_specs
    from dinkster.compose import ServingComposer, default_pack_specs, model_pack_specs

    composer = ServingComposer()
    try:
        specs = composer.order_pack_entries(
            (*default_pack_specs(), *model_pack_specs(), *comfy_compat_specs())
        )
        return tuple(load_manifest(Path(spec.manifest)).name for spec in specs)
    finally:
        await composer.close()


def test_core_server_serves_real_catalog_without_optional_packages(tmp_path: Path) -> None:
    from dinkster_graph import Graph, GraphNode, graph_to_wire

    environment = tmp_path / "core-environment"
    sync_environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(environment),
    }
    subprocess.run(
        [
            "uv",
            "sync",
            "--locked",
            "--no-dev",
            "--no-install-package",
            "dinkster-collab",
            "--no-install-package",
            "dinkster-supervisor",
            "--no-install-package",
            "dinkster-p2p",
        ],
        cwd=TESTS_DIR.parent,
        env=sync_environment,
        check=True,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m, importlib.util as u; "
            "assert all(u.find_spec(n) is None for n in "
            "('dinkster_collab', 'dinkster_supervisor', 'dinkster_p2p')); "
            "assert all(not any(d.metadata['Name'] == n for d in m.distributions()) for n in "
            "('dinkster-collab', 'dinkster-supervisor', 'dinkster-p2p'))",
        ],
        check=True,
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
            "--disable-p2p",
        ],
        cwd=tmp_path,
        env=process_environment,
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
            assert len(nodes) >= 100
            assert all(
                isinstance(node_id, str) and isinstance(schema, dict)
                for node_id, schema in nodes.items()
            )
            assert any(node_id.startswith("std.") for node_id in nodes)
            graph = Graph(nodes={"sum": GraphNode("std.math.add_ints", {"a": 2, "b": 3})})
            async with session.post(
                base + "/api/jobs",
                json={
                    "clientId": "optional-package-proof",
                    "jobId": "sum",
                    "graph": graph_to_wire(graph),
                    "targets": ["sum"],
                },
            ) as response:
                assert response.status == 202, await response.text()
            async with asyncio.timeout(30):
                while True:
                    async with session.get(
                        base + "/api/jobs/optional-package-proof/sum"
                    ) as response:
                        job = await response.json()
                    if job["state"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.05)
            assert job["state"] == "completed", job
            async with session.get(base + "/api/sessions") as response:
                assert response.status == 404
            assert "p2p unavailable" in log.read_text("utf-8")

    try:
        asyncio.run(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        output.close()


def test_installed_collaboration_package_registers_session_routes() -> None:
    async def scenario() -> None:
        from aiohttp.test_utils import TestClient, TestServer
        from dinkster_caches import MemoryLRUCache
        from dinkster_engine import Engine
        from dinkster_server import create_app
        from dinkster_values import TypeRegistry, register_core_types
        from dinkster_workers import InProcessWorker

        from dinkster.serve import _add_collaboration_routes

        registry = TypeRegistry()
        register_core_types(registry)

        def make_engine(on_event):
            return Engine(
                schemas={},
                registry=registry,
                worker=InProcessWorker({}, registry),
                cache=MemoryLRUCache(),
                on_event=on_event,
            )

        app = create_app(make_engine, {})
        assert _add_collaboration_routes(app, None) is True
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/sessions",
                json={"scope": "local", "documentId": "doc", "snapshot": {}},
            )
            assert response.status == 201
        finally:
            await client.close()

    asyncio.run(scenario())


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


def test_native_runtime_versions_are_read_from_worker_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type(
            "Completed",
            (),
            {"stdout": '{"dinkster-kitchen": "0.2.31", "torch": "2.13.0+cu130"}'},
        )()

    monkeypatch.setattr(serve.subprocess, "run", run)

    assert serve.detect_native_runtime_versions("worker-python") == {
        "dinkster-kitchen": "0.2.31",
        "torch": "2.13.0+cu130",
    }
    assert calls[0][:3] == ["worker-python", "-I", "-c"]
    assert len(calls[0]) == 4


@pytest.mark.parametrize("pack_name", ["dinkster-vision-hed", "dinkster-vision-birefnet"])
def test_standard_vision_pack_provisions_declared_runtime_before_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pack_name: str
) -> None:
    from dinkster_workers import PackManifest

    from dinkster import serve
    from dinkster.compose import default_pack_spec

    spec = default_pack_spec(pack_name)
    assert spec.packs is not None
    digest = spec.packs[pack_name].artifact_digest.removeprefix("blake3:")
    calls: list[dict[str, object]] = []
    selected_python = tmp_path / "prepared" / "bin" / "python"

    def provision(manifest: object, **kwargs: object) -> Path:
        calls.append({"manifest": manifest, **kwargs})
        return selected_python

    monkeypatch.delenv("DINKSTER_SERVING_PYTHON")
    monkeypatch.setattr(serve, "ensure_pack_venv", provision)
    prepared = serve._prepare_default_pack(
        spec,
        venv_root=tmp_path / "runtime-venvs",
        accelerator="cuda",
    )

    assert prepared.python == str(selected_python)
    assert len(calls) == 1
    manifest = calls[0]["manifest"]
    assert isinstance(manifest, PackManifest)
    assert manifest.name == pack_name
    assert "torch==2.13.0" in manifest.requires
    assert calls[0]["venv_root"] == tmp_path / "runtime-venvs" / "cuda" / digest
    assert calls[0]["accelerator"] == "cuda"
    workspace = calls[0]["workspace_packages"]
    assert isinstance(workspace, tuple)
    assert {path.name for path in workspace} == set(serve._PACK_HOST_WORKSPACE_PACKAGES)
    workspace_names = set(serve._PACK_HOST_WORKSPACE_PACKAGES)
    package_names = {
        tomllib.loads(path.read_text(encoding="utf-8"))["project"]["name"]
        for path in (TESTS_DIR.parent / "packages").glob("*/pyproject.toml")
    }
    for path in workspace:
        project = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        dependencies = {Requirement(value).name for value in project.get("dependencies", ())}
        assert dependencies & package_names <= workspace_names
    assert TESTS_DIR.parent / "packages" / "dinkster-video" in workspace
    if pack_name == "dinkster-vision-birefnet":
        assert "dinkster-inference-torch==0.0.1" in manifest.requires
        assert TESTS_DIR.parent / "packages" / "dinkster-inference-torch" in workspace
    assert str(manifest.root.parent / "src") in prepared.env["PYTHONPATH"].split(os.pathsep)


def test_configured_serving_python_skips_standard_pack_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve
    from dinkster.compose import default_pack_spec

    def unexpected(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("configured serving interpreter must not provision")

    monkeypatch.setenv("DINKSTER_SERVING_PYTHON", "/runtime/python")
    monkeypatch.setattr(serve, "ensure_pack_venv", unexpected)
    prepared = serve._prepare_default_pack(
        default_pack_spec("dinkster-vision-hed"),
        venv_root=tmp_path / "runtime-venvs",
        accelerator="cpu",
    )

    assert prepared.python == "/runtime/python"


def test_only_standard_vision_defaults_are_selected_for_runtime_provisioning() -> None:
    from dinkster import serve
    from dinkster.compose import default_pack_spec

    assert serve._is_standard_vision_pack(default_pack_spec("dinkster-vision-hed"))
    assert not serve._is_standard_vision_pack(default_pack_spec("dinkster-nodes-remote"))


def test_bundled_standard_pack_exposes_its_artifact_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import PackInfo

    from dinkster import serve
    from dinkster.compose import PackSpec

    artifact = tmp_path / "artifact"
    module = artifact / "bundled_pack"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    manifest = artifact / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "bundled-pack"\nnamespaces = ["bundled"]\n'
        '[pack.entry]\nnodes = "bundled_pack:NODES"\n'
    )
    selected_python = tmp_path / "prepared" / "bin" / "python"
    calls: list[dict[str, object]] = []

    def provision(loaded: object, **kwargs: object) -> Path:
        calls.append({"manifest": loaded, **kwargs})
        return selected_python

    monkeypatch.delenv("DINKSTER_SERVING_PYTHON")
    monkeypatch.setattr(serve, "ensure_pack_venv", provision)

    prepared = serve._prepare_default_pack(
        PackSpec(
            manifest=manifest,
            packs={
                "bundled-pack": PackInfo(
                    display_name="Bundled pack",
                    artifact_digest=f"blake3:{'1' * 64}",
                )
            },
        ),
        venv_root=tmp_path / "runtime-venvs",
        accelerator="cpu",
    )

    assert prepared.python == str(selected_python)
    assert prepared.env["PYTHONPATH"] == str(artifact)
    assert calls[0]["workspace_packages"] == ()


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


def test_native_compute_dtypes_matches_uuid_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import serve

    _probe_smi(monkeypatch, [_ADA, _PASCAL])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaaaaaa-1111-2222-3333-444444444444")
    assert serve.detect_native_compute_dtypes() == _ALL_DTYPES


def test_native_compute_dtypes_windows_enables_10_series_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    _probe_smi(monkeypatch, [_PASCAL])
    monkeypatch.setattr(serve.platform, "win32_ver", lambda: ("10", "", "", ""))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert serve.detect_native_compute_dtypes() == frozenset({"float16", "float32"})


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


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (None, frozenset()),
        (
            ["all"],
            frozenset(
                {
                    "memory-budgets",
                    "memory-headroom",
                    "aimdo-policy",
                    "dtype-policy",
                    "fp8-matmul",
                    "worker-comfy-args",
                    "jobs",
                    "logging",
                    "p2p",
                }
            ),
        ),
        (["jobs"], frozenset({"jobs"})),
        (["jobs", "logging"], frozenset({"jobs", "logging"})),
        (
            ["jobs", "all"],
            frozenset(
                {
                    "memory-budgets",
                    "memory-headroom",
                    "aimdo-policy",
                    "dtype-policy",
                    "fp8-matmul",
                    "worker-comfy-args",
                    "jobs",
                    "logging",
                    "p2p",
                }
            ),
        ),
    ],
)
def test_settings_gate_normalization(values: list[str] | None, expected: frozenset[str]) -> None:
    from dinkster.serve import normalize_settings_categories

    assert normalize_settings_categories(values) == expected


def test_settings_gate_rejects_unknown_category() -> None:
    from dinkster.serve import normalize_settings_categories

    with pytest.raises(ValueError, match="comfy-args.*memory-budgets"):
        normalize_settings_categories(["comfy-args"])


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], frozenset()),
        (
            ["--allow-settings-changes"],
            frozenset(
                {
                    "memory-budgets",
                    "memory-headroom",
                    "aimdo-policy",
                    "dtype-policy",
                    "fp8-matmul",
                    "worker-comfy-args",
                    "jobs",
                    "logging",
                    "p2p",
                }
            ),
        ),
        (
            ["--allow-settings-changes=all"],
            frozenset(
                {
                    "memory-budgets",
                    "memory-headroom",
                    "aimdo-policy",
                    "dtype-policy",
                    "fp8-matmul",
                    "worker-comfy-args",
                    "jobs",
                    "logging",
                    "p2p",
                }
            ),
        ),
        (["--allow-settings-changes=jobs"], frozenset({"jobs"})),
        (
            ["--allow-settings-changes=jobs", "--allow-settings-changes=logging"],
            frozenset({"jobs", "logging"}),
        ),
        (
            ["--allow-settings-changes=jobs", "--allow-settings-changes"],
            frozenset(
                {
                    "memory-budgets",
                    "memory-headroom",
                    "aimdo-policy",
                    "dtype-policy",
                    "fp8-matmul",
                    "worker-comfy-args",
                    "jobs",
                    "logging",
                    "p2p",
                }
            ),
        ),
    ],
)
def test_settings_gate_argparse_matrix(
    flags: list[str],
    expected: frozenset[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    captured: list[frozenset[str]] = []

    def record_settings(*_args: object, **kwargs: object) -> object:
        captured.append(kwargs["granted"])  # type: ignore[arg-type]
        return object()

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "RuntimeSettings", record_settings)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", "", *flags])
    serve.main()
    assert captured == [expected]


@pytest.mark.parametrize("persisted_enabled", [None, False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_serve_p2p_defaults_off_preserves_saved_choice_and_allows_cli_disable(
    persisted_enabled: bool | None,
    disabled: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_p2p import default_p2p_settings

    from dinkster import serve

    saved = {
        **default_p2p_settings(),
        "downloadsEnabled": persisted_enabled,
        "seedingEnabled": persisted_enabled,
        "scope": "lan-only",
    }
    path = tmp_path / "settings.json"
    if persisted_enabled is not None:
        saved.pop("stagingBudgetBytes")
        path.write_text(json.dumps({"p2p": saved}), encoding="utf-8")
    original = path.read_bytes() if path.exists() else None
    captured: list[tuple[dict[str, object], dict[str, object]]] = []

    def record_settings(values: object, sources: object, **kwargs: object) -> object:
        captured.append((dict(values), dict(sources)))  # type: ignore[arg-type]
        assert kwargs["granted"] == frozenset()
        return object()

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "RuntimeSettings", record_settings)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            str(tmp_path),
            *(["--disable-p2p"] if disabled else []),
        ],
    )
    serve.main()
    value = captured[0][0]["p2p"]
    assert isinstance(value, dict)
    enabled = not disabled and persisted_enabled is True
    assert value["downloadsEnabled"] is enabled
    assert value["seedingEnabled"] is enabled
    assert value["stagingBudgetBytes"] == 64 * 1024**3
    assert value["scope"] == ("lan-and-internet" if persisted_enabled is None else "lan-only")
    assert captured[0][1]["p2p"] == (
        "cli" if disabled else "default" if persisted_enabled is None else "persisted"
    )
    assert (path.read_bytes() if path.exists() else None) == original


@pytest.mark.parametrize("configuration", ["environment", "cli", "absent"])
def test_serve_official_bootstrap_configuration_preserves_p2p_off_and_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configuration: str
) -> None:
    from aiohttp.test_utils import TestClient, TestServer
    from dinkster_p2p import default_p2p_settings

    from dinkster import serve

    source = "https://fixture.invalid/export"
    provider_id = "fixture-provider"
    monkeypatch.delenv("DINKSTER_OFFICIAL_RESOLVER_URL", raising=False)
    monkeypatch.delenv("DINKSTER_OFFICIAL_RESOLVER_PROVIDER_ID", raising=False)
    flags = ["--disable-p2p"]
    settings_path = tmp_path / "settings.json"
    if configuration != "absent":
        monkeypatch.setenv("DINKSTER_OFFICIAL_RESOLVER_URL", source)
        monkeypatch.setenv("DINKSTER_OFFICIAL_RESOLVER_PROVIDER_ID", provider_id)
    if configuration == "cli":
        source += "?cli=1"
        provider_id = "cli-provider"
        flags += ["--official-resolver-url", source, "--official-resolver-provider-id", provider_id]
    elif configuration == "environment":
        flags = []
        settings_path.write_text(
            json.dumps(
                {
                    "p2p": {
                        **default_p2p_settings(),
                        "downloadsEnabled": False,
                        "seedingEnabled": False,
                    }
                }
            ),
            encoding="utf-8",
        )
    original = settings_path.read_bytes() if settings_path.exists() else None
    calls: list[tuple[str | None, str | None]] = []

    def bootstrap(_self: object, url: str | None, identity: str | None) -> None:
        calls.append((url, identity))

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.get("/api/settings")
                assert response.status == 200
                settings = await response.json()
                assert settings["categories"]["granted"] == []
                p2p = settings["settings"]["p2p"]
                assert p2p["value"]["downloadsEnabled"] is False
                assert p2p["value"]["seedingEnabled"] is False
                assert p2p["source"] == ("persisted" if configuration == "environment" else "cli")
            finally:
                await client.close()

        asyncio.run(inspect())

    monkeypatch.setattr(serve.ResolverSubscriptionStore, "bootstrap_official", bootstrap)
    monkeypatch.setattr(serve, "default_pack_ids", lambda: ())
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", str(tmp_path), *flags])
    serve.main()
    assert calls == ([(None, None)] if configuration == "absent" else [(source, provider_id)])
    assert (settings_path.read_bytes() if settings_path.exists() else None) == original


def test_serve_official_bootstrap_requires_persistent_library(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--official-resolver-provider-id",
            "fixture",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert "official resolver bootstrap requires --library-root" in capsys.readouterr().err


def test_comfy_compat_composition_failure_exits_with_one_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    message = (
        "ComfyUI requirement module 'einops' is unavailable in interpreter "
        "'/selected/python' selected by --execution-python"
    )

    def fail_specs(*_args: object, **_kwargs: object) -> None:
        raise serve.CompositionError(message)

    monkeypatch.setattr(serve, "comfy_compat_specs", fail_specs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--no-default-packs",
            "--comfy-root",
            str(comfy_root),
            "--execution-python",
            "/selected/python",
        ],
    )

    with pytest.raises(SystemExit) as caught:
        serve.main()

    assert str(caught.value) == message


def test_retired_comfy_python_flag_exits_naming_the_replacement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.delenv("DINKSTER_COMFYUI_PYTHON", raising=False)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--comfy-python", "/x/python"])
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert "--comfy-python is retired; pass --execution-python instead" in capsys.readouterr().err


def test_retired_comfy_python_env_exits_naming_the_replacement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster import serve

    monkeypatch.setenv("DINKSTER_COMFYUI_PYTHON", "/x/python")
    monkeypatch.setattr(sys, "argv", ["dinkster-serve"])
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert (
        "DINKSTER_COMFYUI_PYTHON is retired; set DINKSTER_EXECUTION_PYTHON instead"
        in capsys.readouterr().err
    )


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


def test_openai_generation_routes_mount_and_lazy_provider_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    made: list[FakeOpenAIProvider] = []

    class FakeOpenAIProvider:
        id = "dinkster.openai"
        model_identity = "openai:test-model"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False
            made.append(self)

        def close(self) -> None:
            self.closed = True

    class JsonRequest:
        def __init__(self, value: object) -> None:
            self._body = json.dumps(value).encode()
            self.content_length = len(self._body)

        async def read(self) -> bytes:
            return self._body

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            routes = {
                (route.method, route.resource.canonical): route.handler
                for route in app.router.routes()
            }
            expected = {
                ("POST", "/api/generation"),
                ("GET", "/api/generation/models"),
                ("POST", "/api/generation/models/load"),
                ("POST", "/api/generation/models/unload"),
                ("DELETE", "/api/generation/sessions/{session_id}"),
                ("GET", "/v1/models"),
                ("POST", "/v1/completions"),
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/responses"),
            }
            assert expected <= routes.keys()
            assert made == []
            response = await routes[("POST", "/api/generation/models/load")](
                JsonRequest({"model": "test-model"})  # type: ignore[arg-type]
            )
            assert response.status == 200
            assert len(made) == 1
            assert made[0].closed is False
            app.freeze()
            await app.cleanup()
            assert made[0].closed is True

        asyncio.run(inspect())

    monkeypatch.setenv("DINKSTER_OPENAI_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("DINKSTER_OPENAI_MODEL", "test-model")
    monkeypatch.setattr(serve, "OpenAIGenerationProvider", FakeOpenAIProvider)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])

    serve.main()


def test_generation_cleanup_failure_does_not_skip_composition_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    made: list[FakeOpenAIProvider] = []
    composition_closed: list[bool] = []
    real_composer = serve.ServingComposer

    class FakeOpenAIProvider:
        id = "dinkster.openai"
        model_identity = "openai:test-model"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            made.append(self)

        def close(self) -> None:
            raise RuntimeError("generation cleanup failed")

    class JsonRequest:
        def __init__(self, value: object) -> None:
            self._body = json.dumps(value).encode()
            self.content_length = len(self._body)

        async def read(self) -> bytes:
            return self._body

    def recording_composer(*args: object, **kwargs: object) -> object:
        composer = real_composer(*args, **kwargs)  # type: ignore[arg-type]
        original_close = composer.composition.close

        async def close_composition() -> None:
            composition_closed.append(True)
            await original_close()

        monkeypatch.setattr(composer.composition, "close", close_composition)
        return composer

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            routes = {
                (route.method, route.resource.canonical): route.handler
                for route in app.router.routes()
            }
            response = await routes[("POST", "/api/generation/models/load")](
                JsonRequest({"model": "test-model"})  # type: ignore[arg-type]
            )
            assert response.status == 200
            app.freeze()
            with pytest.raises(ExceptionGroup, match="server cleanup failed"):
                await app.cleanup()

        asyncio.run(inspect())

    monkeypatch.setenv("DINKSTER_OPENAI_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("DINKSTER_OPENAI_MODEL", "test-model")
    monkeypatch.setattr(serve, "OpenAIGenerationProvider", FakeOpenAIProvider)
    monkeypatch.setattr(serve, "ServingComposer", recording_composer)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])

    serve.main()

    assert len(made) == 1
    assert composition_closed == [True]


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({"DINKSTER_OPENAI_BASE_URL": "https://api.example.test/v1"}, "requires both"),
        ({"DINKSTER_OPENAI_TIMEOUT": "not-a-number"}, "must be a number"),
        ({"DINKSTER_OPENAI_TIMEOUT": "inf"}, "must be finite and positive"),
        ({"DINKSTER_OPENAI_STREAM": "sometimes"}, "must be stream/json or true/false"),
    ),
)
def test_openai_generation_environment_fails_with_argparse_diagnostic(
    environment: dict[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])

    with pytest.raises(SystemExit, match="2"):
        serve.main()

    assert message in capsys.readouterr().err


def test_sandbox_refuses_live_mount_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            str(tmp_path / "library"),
            "--sandbox-packs",
            "--allow-mount-changes",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        serve.main()
    assert "cannot be combined with --sandbox-packs" in capsys.readouterr().err


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


def test_sandbox_host_grants_reach_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    captured: list[dict[str, object]] = []
    real_composer = serve.ServingComposer

    def recording_composer(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return real_composer(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    monkeypatch.setattr(serve, "ServingComposer", recording_composer)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--sandbox-packs",
            "--sandbox-grant-gpu",
            "gpu_pack",
            "--sandbox-grant-network",
            "network.pack=https://API.example.test/",
            "--sandbox-grant-network",
            "network_pack=https://storage.example.test:8443",
        ],
    )

    serve.main()

    assert len(captured) == 1
    assert captured[0]["sandbox_gpu_grants"] == frozenset({"gpu-pack"})
    assert captured[0]["sandbox_network_grants"] == {
        "network-pack": (
            "https://api.example.test",
            "https://storage.example.test:8443",
        )
    }
    assert captured[0]["pack_scratch_root"] is None


def test_library_root_configures_pack_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    captured: list[dict[str, object]] = []
    real_composer = serve.ServingComposer

    def recording_composer(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return real_composer(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    library = tmp_path / "library"
    monkeypatch.setattr(serve, "ServingComposer", recording_composer)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", str(library)])

    serve.main()

    assert len(captured) == 1
    assert captured[0]["pack_scratch_root"] == library.resolve() / "scratch"
    assert captured[0]["cache_mode"] == "layered"
    assert captured[0]["cache_memory_entries"] == 1024
    assert captured[0]["cache_dir"] == library / "execution-cache"
    assert captured[0]["cache_disk_budget"] == 10 * 1024**3


def test_execution_cache_cli_supports_explicit_root_without_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    captured: list[dict[str, object]] = []
    real_composer = serve.ServingComposer

    def recording_composer(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return real_composer(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    cache = tmp_path / "cache"
    monkeypatch.setattr(serve, "ServingComposer", recording_composer)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--execution-cache-dir",
            str(cache),
            "--execution-cache-memory-entries",
            "7",
            "--execution-cache-disk-budget",
            "2M",
        ],
    )

    serve.main()

    assert len(captured) == 1
    assert captured[0]["cache_mode"] == "layered"
    assert captured[0]["cache_memory_entries"] == 7
    assert captured[0]["cache_dir"] == cache
    assert captured[0]["cache_disk_budget"] == 2 * 1024**2


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


def test_sandbox_protects_configured_auth_outside_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import SandboxPolicy

    from dinkster import serve

    library = tmp_path / "library"
    library.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    auth = shared / "auth.toml"
    auth.write_text(
        "version = 1\n"
        "[[tokens]]\n"
        'token = "operator-secret"\n'
        'principalId = "operator"\n'
        "[tokens.grants]\n"
        'local = ["jobs:read"]\n',
        encoding="utf-8",
    )
    (library / "mounts.toml").write_text(
        f"[mounts.shared]\npath = {json.dumps(str(shared))}\n",
        encoding="utf-8",
    )
    captured: list[SandboxPolicy] = []
    real_composer = serve.ServingComposer

    def recording_composer(*args: object, **kwargs: object) -> object:
        policy = kwargs.get("sandbox_policy")
        assert isinstance(policy, SandboxPolicy)
        captured.append(policy)
        return real_composer(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **kwargs: object) -> None:
        assert kwargs["host"] == "0.0.0.0"

        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    monkeypatch.setattr(serve, "ServingComposer", recording_composer)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            str(library),
            "--sandbox-packs",
            "--auth",
            str(auth),
            "--host",
            "0.0.0.0",
            "--allow-host",
            "lan.example",
        ],
    )

    serve.main()

    assert len(captured) == 1
    assert str(shared) in captured[0].ro_binds
    assert str(auth) in captured[0].protected_roots


def test_read_only_output_mount_is_not_selected_as_the_save_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve

    library = tmp_path / "library"
    output = tmp_path / "shared-output"
    library.mkdir()
    output.mkdir()
    (library / "mounts.toml").write_text(
        f"[mounts.output]\npath = {json.dumps(str(output))}\n",
        encoding="utf-8",
    )

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            str(library),
            "--no-default-packs",
            "--disable-p2p",
        ],
    )

    serve.main()


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


def test_identity_auth_environment_configures_token_authenticator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    configured = {
        "DINKSTER_IDENTITY_JWKS_URL": "https://identity.example/.well-known/jwks.json",
        "DINKSTER_IDENTITY_ISSUER": "https://identity.example",
        "DINKSTER_IDENTITY_AUDIENCE": "dinkster-session",
    }
    captured: list[tuple[str, str, str]] = []

    class StubTokenAuthenticator:
        def __init__(self, jwks_url: str, issuer: str, audience: str) -> None:
            captured.append((jwks_url, issuer, audience))

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    for name, value in configured.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(serve, "TokenAuthenticator", StubTokenAuthenticator)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])

    serve.main()

    assert captured == [
        (
            "https://identity.example/.well-known/jwks.json",
            "https://identity.example",
            "dinkster-session",
        )
    ]
    assert all(name not in os.environ for name in configured)


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


@pytest.mark.parametrize(
    "paths",
    [
        ("store.sqlite", None, None),
        (None, "policy.toml", None),
        (None, None, "cursor.key"),
        ("store.sqlite", "policy.toml", None),
    ],
)
def test_federated_asset_config_is_all_or_none(paths: tuple[str | None, ...]) -> None:
    from dinkster.serve import _load_federated_asset_config

    with pytest.raises(ValueError, match="must be provided together"):
        _load_federated_asset_config(*paths, auth_enabled=False)


@pytest.mark.parametrize(
    "policy",
    [
        "version = true\n[scopes]\nlocal = []\n",
        "version = 2\n[scopes]\nlocal = []\n",
        "version = 1\nextra = 1\n[scopes]\nlocal = []\n",
        "version = 1\n[scopes]\n",
        "version = 1\n[scopes]\n'bad scope' = []\n",
        "version = 1\n[scopes]\nlocal = 'provider'\n",
        "version = 1\n[scopes]\nlocal = [1]\n",
        "version = 1\n[scopes]\nlocal = ['Provider']\n",
        "version = 1\n[scopes]\nlocal = ['provider', 'provider']\n",
    ],
)
def test_federated_asset_policy_is_strict(policy: str, tmp_path: Path) -> None:
    from dinkster.serve import _load_federated_asset_config

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(policy, encoding="utf-8")
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(b"k" * 32)
    with pytest.raises(ValueError):
        _load_federated_asset_config(
            str(tmp_path / "store.sqlite"),
            str(policy_path),
            str(key_path),
            auth_enabled=False,
        )


def test_federated_asset_config_preserves_policy_and_raw_cursor_key(tmp_path: Path) -> None:
    from dinkster.serve import _load_federated_asset_config

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        "version = 1\n[scopes]\nlocal = []\nworkspace = ['provider-a', 'org/provider.b']\n",
        encoding="utf-8",
    )
    key = b" raw-cursor-key-with-newline-bytes\n"
    assert len(key) >= 32
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(key)

    config = _load_federated_asset_config(
        str(tmp_path / "store.sqlite"),
        str(policy_path),
        str(key_path),
        auth_enabled=False,
    )

    assert config is not None
    store_path, policy, loaded_key = config
    assert store_path == tmp_path / "store.sqlite"
    assert policy == {
        "local": frozenset(),
        "workspace": frozenset({"provider-a", "org/provider.b"}),
    }
    assert loaded_key == key


def test_federated_asset_auth_off_requires_local_scope(tmp_path: Path) -> None:
    from dinkster.serve import _load_federated_asset_config

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text("version = 1\n[scopes]\nworkspace = []\n", encoding="utf-8")
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(b"k" * 32)
    paths = (str(tmp_path / "store.sqlite"), str(policy_path), str(key_path))

    with pytest.raises(ValueError, match="local"):
        _load_federated_asset_config(*paths, auth_enabled=False)
    assert _load_federated_asset_config(*paths, auth_enabled=True) is not None


def test_federated_asset_cursor_key_minimum_is_raw_bytes(tmp_path: Path) -> None:
    from dinkster.serve import _load_federated_asset_config

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text("version = 1\n[scopes]\nlocal = []\n", encoding="utf-8")
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(b"k" * 31 + b"\n")
    assert (
        _load_federated_asset_config(
            str(tmp_path / "store.sqlite"),
            str(policy_path),
            str(key_path),
            auth_enabled=False,
        )
        is not None
    )
    key_path.write_bytes(b"k" * 31)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        _load_federated_asset_config(
            str(tmp_path / "store.sqlite"),
            str(policy_path),
            str(key_path),
            auth_enabled=False,
        )


@pytest.mark.parametrize("with_library_root", [False, True])
def test_federated_asset_cli_wires_canonical_read_only_composition_and_cleanup(
    with_library_root: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text("version = 1\n[scopes]\nlocal = []\n", encoding="utf-8")
    key = b"k" * 32
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(key)
    stores: list[object] = []
    close_counts: list[int] = []
    replacement_calls: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
    captured: list[dict[str, object]] = []
    registered: set[str] = set()
    original_create_app = serve.create_app

    class FakeStore:
        def __init__(self, path: Path) -> None:
            self.path = path
            stores.append(self)
            close_counts.append(0)

        def close(self) -> None:
            close_counts[0] += 1

        def replace_mount_snapshot(self, scopes: tuple[str, ...], rows: tuple[object, ...]) -> None:
            replacement_calls.append((scopes, rows))

    def capture_create_app(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return original_create_app(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            registered.update(resource.canonical for resource in app.router.resources())
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    monkeypatch.setattr(serve, "ResolutionStore", FakeStore)
    monkeypatch.setattr(serve, "create_app", capture_create_app)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    argv = [
        "dinkster-serve",
        "--library-root",
        str(tmp_path / "library") if with_library_root else "",
    ]
    argv.extend(
        (
            "--federated-assets-store",
            str(tmp_path / "resolution.sqlite"),
            "--federated-assets-policy",
            str(policy_path),
            "--federated-assets-cursor-key",
            str(key_path),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)

    serve.main()

    assert len(stores) == 1
    assert captured[0]["federated_asset_paths"] == {
        "catalog": "/api/catalog",
        "candidates": "/api/catalog/candidates",
    }
    assert captured[0]["federated_asset_store"] is stores[0]
    assert captured[0]["federated_asset_sources"] == ()
    assert captured[0]["federated_asset_provider_policy"] == {"local": frozenset()}
    assert captured[0]["_federated_asset_cursor_key"] == key
    assert {"/api/catalog", "/api/catalog/candidates"} <= registered
    assert "/api/catalog/resolve" not in registered
    assert replacement_calls == [(("local",), ())]
    assert close_counts == [1]


@pytest.mark.parametrize("failure_site", ["create-app", "composition-route"])
def test_federated_asset_store_closes_when_app_construction_fails(
    failure_site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text("version = 1\n[scopes]\nlocal = []\n", encoding="utf-8")
    key_path = tmp_path / "cursor.key"
    key_path.write_bytes(b"k" * 32)
    closed: list[bool] = []

    class FakeStore:
        def __init__(self, _path: Path) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

        def replace_mount_snapshot(
            self, _scopes: tuple[str, ...], _rows: tuple[object, ...]
        ) -> None:
            pass

    def fail_create_app(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("app-build-failed")

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        asyncio.run(awaitable)  # type: ignore[arg-type]

    monkeypatch.setattr(serve, "ResolutionStore", FakeStore)
    if failure_site == "create-app":
        monkeypatch.setattr(serve, "create_app", fail_create_app)
    else:
        monkeypatch.setattr(serve, "add_comfy_compat_routes", fail_create_app)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--federated-assets-store",
            str(tmp_path / "resolution.sqlite"),
            "--federated-assets-policy",
            str(policy_path),
            "--federated-assets-cursor-key",
            str(key_path),
        ],
    )

    with pytest.raises(RuntimeError, match="app-build-failed"):
        serve.main()
    assert closed == [True]


def test_ordinary_serve_passes_no_federated_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import serve

    captured: list[dict[str, object]] = []
    original_create_app = serve.create_app

    def capture_create_app(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return original_create_app(*args, **kwargs)  # type: ignore[arg-type]

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            registered = {resource.canonical for resource in app.router.resources()}
            assert "/api/catalog" not in registered
            assert "/api/catalog/candidates" not in registered
            assert "/api/catalog/resolve" not in registered
            app.freeze()
            await app.cleanup()

        asyncio.run(inspect())

    monkeypatch.setattr(serve, "create_app", capture_create_app)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])
    monkeypatch.setenv("DINKSTER_ATTENTION_POLICY", "flash")

    serve.main()

    assert captured[0]["attention_policy"] == "flash"
    assert "DINKSTER_ATTENTION_POLICY" not in os.environ
    assert captured[0]["federated_asset_paths"] is None
    assert captured[0]["federated_asset_store"] is None
    assert captured[0]["federated_asset_provider_policy"] is None


@pytest.mark.parametrize("devices", (None, "0,1"))
def test_serve_without_comfy_mounts_native_provider(
    monkeypatch: pytest.MonkeyPatch,
    devices: str | None,
) -> None:
    from dinkster import serve

    calls: list[tuple[object, dict[str, object]]] = []

    def native_specs(root: object, **kwargs: object) -> list[object]:
        calls.append((root, kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", native_specs)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    argv = ["dinkster-serve", "--library-root", ""]
    if devices:
        argv.extend(("--single-job-multi-gpu-devices", devices))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("DINKSTER_COMFYUI_ROOT", raising=False)

    serve.main()
    assert len(calls) == 1
    root, options = calls[0]
    assert root is None
    if devices:
        config = options["single_job_multi_gpu"]
        assert isinstance(config, serve.SingleJobMultiGpuConfig)
        assert config.cuda_indices == (0, 1)
    else:
        assert options["single_job_multi_gpu"] is None


def test_serve_resolves_legacy_pack_from_launch_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    launch_directory = tmp_path / "launch"
    legacy_pack = launch_directory / "packs" / "legacy"
    legacy_pack.mkdir(parents=True)
    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    captured: list[Path] = []

    def capture_specs(_root: str, **kwargs: object) -> list[object]:
        captured.extend(kwargs["legacy_packs"])  # type: ignore[arg-type]
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.chdir(launch_directory)
    monkeypatch.setattr(serve, "comfy_compat_specs", capture_specs)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--comfy-root",
            str(comfy_root),
            "--legacy-pack",
            str(legacy_pack.relative_to(launch_directory)),
        ],
    )

    serve.main()

    assert captured == [legacy_pack.resolve()]


def test_comfy_root_defaults_include_compat_specs_in_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve
    from dinkster.compose import PackSpec

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    compat = (
        PackSpec(manifest=tmp_path / "generation"),
        PackSpec(manifest=tmp_path / "compat"),
    )
    captured: list[tuple[int, int]] = []

    def capture_ordering(
        _composer: object,
        specs: object,
        default_count: int,
    ) -> tuple[PackSpec, ...]:
        entries = tuple(specs)  # type: ignore[arg-type]
        captured.append((len(entries), default_count))
        raise RuntimeError("captured ordering")

    def run_app(awaitable: object, **_kwargs: object) -> None:
        with pytest.raises(RuntimeError, match="captured ordering"):
            asyncio.run(awaitable)  # type: ignore[arg-type]

    monkeypatch.setattr(serve, "default_pack_ids", lambda: ())
    monkeypatch.setattr(serve, "model_pack_specs", lambda: ())
    monkeypatch.setattr(serve, "comfy_compat_specs", lambda *_args, **_kwargs: compat)
    monkeypatch.setattr(serve, "_order_default_pack_specs", capture_ordering)
    monkeypatch.setattr(serve.web, "run_app", run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--comfy-root",
            str(comfy_root),
        ],
    )

    serve.main()

    assert captured == [(2, 2)]


@pytest.mark.parametrize("mode", [None, "auto", "on", "off"])
def test_serve_aimdo_cli_defaults_and_reaches_compat_specs(
    mode: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    captured: list[dict[str, object]] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    argv = [
        "dinkster-serve",
        "--comfy-root",
        str(comfy_root),
        "--library-root",
        "",
    ]
    if mode is not None:
        argv.extend(("--aimdo", mode))
    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", argv)

    serve.main()

    assert len(captured) == 1
    assert captured[0]["aimdo"] == ("auto" if mode is None else mode)
    assert captured[0]["reserve_vram"] == 256 * 1024**2
    assert captured[0]["memory_budgets"] == {}


def test_serve_multi_gpu_cli_reaches_compat_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    captured: list[dict[str, object]] = []
    captured_running: list[int] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        frame = awaitable.cr_frame  # type: ignore[attr-defined]
        assert frame is not None
        captured_running.append(frame.f_locals["max_running_jobs"])
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--comfy-root",
            str(comfy_root),
            "--library-root",
            "",
            "--multi-gpu-devices",
            "1,0",
        ],
    )

    serve.main()

    assert captured[0]["multi_device_cuda_indices"] == (1, 0)
    assert captured_running == [2]


@pytest.mark.parametrize(
    ("devices", "mode"),
    (
        ("2,0", "guidance"),
        ("3,1,2", "guidance"),
        ("1,0", "sequence"),
        ("1,0", "window"),
    ),
)
def test_serve_single_job_multi_gpu_preserves_logical_rank_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    devices: str,
    mode: str,
) -> None:
    from dinkster import serve

    captured: list[object] = []

    def compat_specs(*args: object, **kwargs: object) -> list[object]:
        del args
        captured.append(kwargs["single_job_multi_gpu"])
        raise RuntimeError("captured")

    monkeypatch.setattr(serve, "comfy_compat_specs", compat_specs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            "",
            "--comfy-root",
            str(tmp_path),
            "--single-job-multi-gpu-devices",
            devices,
            "--single-job-multi-gpu-mode",
            mode,
        ],
    )
    with pytest.raises(RuntimeError, match="captured"):
        serve.main()

    config = captured[0]
    assert config.cuda_indices == tuple(int(value) for value in devices.split(","))  # type: ignore[attr-defined]
    assert config.mode == mode  # type: ignore[attr-defined]


def test_serve_rejects_removed_single_job_model_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import serve

    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--single-job-multi-gpu-mode", "model"],
    )
    with pytest.raises(SystemExit):
        serve.main()


def test_serve_rejects_removed_dev_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import serve

    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--dev"])
    with pytest.raises(SystemExit) as exc:
        serve.main()
    assert exc.value.code == 2


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


def test_serve_headroom_cli_reaches_compat_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    captured: list[dict[str, object]] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve, "comfy_model_roots", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--comfy-root",
            str(comfy_root),
            "--library-root",
            "",
            "--reserve-vram",
            "128M",
            "--memory-budget",
            "vram:cuda:0=20G",
            "--memory-budget",
            "ram=8G",
        ],
    )

    serve.main()

    assert captured[0]["reserve_vram"] == 128 * 1024**2
    assert captured[0]["memory_budgets"] == {
        "vram:cuda:0": 20 * 1024**3,
        "ram": 8 * 1024**3,
    }


def test_serve_comfy_arg_cli_beats_persisted_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    library_root = tmp_path / "library"
    library_root.mkdir()
    (library_root / "settings.json").write_text(
        '{"worker-comfy-args":["--preview-size","111"]}\n', "utf-8"
    )
    captured: list[dict[str, object]] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve, "comfy_model_roots", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--comfy-root",
            str(comfy_root),
            "--library-root",
            str(library_root),
            "--comfy-arg=--preview-size",
            "--comfy-arg=321",
        ],
    )

    serve.main()

    assert captured[0]["comfy_args"] == ("--preview-size", "321")


def test_serve_bare_comfy_arg_clears_persisted_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    library_root = tmp_path / "library"
    library_root.mkdir()
    (library_root / "settings.json").write_text(
        '{"worker-comfy-args":["--preview-size","111"]}\n', "utf-8"
    )
    captured: list[dict[str, object]] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve, "comfy_model_roots", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--comfy-root",
            str(comfy_root),
            "--library-root",
            str(library_root),
            "--comfy-arg",
        ],
    )

    serve.main()

    assert captured[0]["comfy_args"] == ()


def test_serve_component_dtype_flags_reach_compat_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve

    comfy_root = tmp_path / "ComfyUI"
    comfy_root.mkdir()
    captured: list[dict[str, object]] = []

    def fake_specs(_root: str, **kwargs: object) -> list[object]:
        captured.append(dict(kwargs))
        return []

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "comfy_compat_specs", fake_specs)
    monkeypatch.setattr(serve, "comfy_model_roots", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--comfy-root",
            str(comfy_root),
            "--diffusion-dtype",
            "bfloat16",
            "--text-encoder-dtype",
            "float16",
            "--vae-dtype",
            "float32",
        ],
    )

    serve.main()

    assert captured[0]["comfy_args"] == (
        "--bf16-unet",
        "--fp16-text-enc",
        "--fp32-vae",
    )


def test_serve_aimdo_cli_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--aimdo", "sometimes"])
    with pytest.raises(SystemExit):
        serve.main()


@pytest.mark.parametrize(
    ("persisted", "flags", "expected", "source"),
    [
        ({}, [], False, "default"),
        ({"fp8-matmul": True}, [], True, "persisted"),
        ({"fp8-matmul": False}, ["--fp8-matmul"], True, "cli"),
    ],
)
def test_serve_fp8_matmul_cli_beats_persisted_then_defaults_off(
    persisted: dict[str, object],
    flags: list[str],
    expected: bool,
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import serve

    library_root = tmp_path / "library"
    library_root.mkdir()
    if persisted:
        (library_root / "settings.json").write_text(json.dumps(persisted) + "\n", "utf-8")
    captured: list[tuple[dict[str, object], dict[str, object]]] = []

    def record_settings(values: object, sources: object, **_kwargs: object) -> object:
        captured.append((dict(values), dict(sources)))  # type: ignore[arg-type]
        return object()

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        awaitable.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(serve, "RuntimeSettings", record_settings)
    monkeypatch.setattr(serve.web, "run_app", fake_run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster-serve", "--library-root", str(library_root), *flags],
    )
    serve.main()

    assert captured[0][0]["fp8-matmul"] is expected
    assert captured[0][1]["fp8-matmul"] == source


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
        asyncio.run(scenario())
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


def test_unprepared_library_catalog_fails_before_api_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import serve
    from dinkster.compose import CompositionError, PackSpec

    manifest = write_iso_manifest(tmp_path / "unprepared")
    pack = PackSpec(manifest, require_catalog=True)
    attempted = False

    def run_app(awaitable: object, **_kwargs: object) -> None:
        nonlocal attempted
        attempted = True
        with pytest.raises(
            CompositionError,
            match="schema catalog is missing or stale; run dinkster-pack prepare-catalogs",
        ):
            asyncio.run(awaitable)  # type: ignore[arg-type]

    monkeypatch.setattr(serve, "default_pack_ids", lambda: ())
    monkeypatch.setattr(serve, "comfy_compat_specs", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(serve, "_resolve_pack_argument", lambda _path: pack)
    monkeypatch.setattr(serve.web, "run_app", run_app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-serve",
            "--library-root",
            str(tmp_path),
            "--disable-p2p",
            "--pack",
            "fixture",
        ],
    )

    serve.main()

    assert attempted


def test_video_preview_pack_route_event_and_module_end_to_end(tmp_path: Path) -> None:
    import hashlib

    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire

    from tests.test_pack_surfaces import PROOF_EVENT, PROOF_PACK, PROOF_ROOT, PROOF_ROUTE
    from tests.test_video_runtime import _source

    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.mp4").write_bytes(_source())
    (source / "dinkster-pack.toml").write_text("""
[pack]
name = "preview-source"
[pack.entry]
nodes = "preview_source:NODES"
types = "preview_source:register_types"
""")
    (source / "preview_source.py").write_text("""
from pathlib import Path
from dinkster_api.v1 import (
    Node, NodeSchema, OutputSpec, TypeExpr, register_video_value_type, video_from_source,
)
class Source(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type="preview-source.clip",
                          outputs=(OutputSpec("video", TypeExpr.concrete("comfy.VIDEO")),))
    @classmethod
    def execute(cls):
        source = Path(__file__).with_name("clip.mp4").read_bytes()
        return cls.outputs(video=video_from_source(source))
def register_types(registry):
    register_video_value_type(registry, "comfy.VIDEO")
NODES = [Source]
""")
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--pack", str(PROOF_ROOT), "--pack", str(source)),
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
        asyncio.run(scenario())
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
        f'[pack]\nname = "shutdown-pack-{index}"\n'
        f'namespaces = ["{namespace}"]\n\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\n',
        encoding="utf-8",
    )
    return manifest


def child_pids(parent: subprocess.Popen[bytes]) -> tuple[int, ...]:
    children = Path(f"/proc/{parent.pid}/task/{parent.pid}/children")
    return tuple(int(pid) for pid in children.read_text("ascii").split())


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

    return asyncio.run(wait())


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
        asyncio.run(scenario())
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


def test_event_loop_stall_diagnostics_log_route_templates() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from dinkster import serve

    logger = Mock()
    app = web.Application()

    async def answer(_: web.Request) -> web.Response:
        time.sleep(0.1)
        return web.json_response({"ok": True})

    app.router.add_get("/api/jobs/{client_id}/{job_id}", answer)
    serve._install_event_loop_stall_diagnostics(app, threshold=0.05, logger=logger)

    async def scenario() -> None:
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            async with client.get("/api/jobs/private-client/private-job?token=secret") as response:
                assert response.status == 200
        finally:
            await client.close()

    asyncio.run(scenario())

    messages = [call.args for call in logger.info.call_args_list]
    assert ("request started: %s %s", "GET", "/api/jobs/{client_id}/{job_id}") in messages
    assert any(
        message[:3]
        == ("request finished: %s %s in %.3f seconds", "GET", "/api/jobs/{client_id}/{job_id}")
        for message in messages
    )
    assert "private-client" not in repr(messages)
    assert "private-job" not in repr(messages)
    assert "secret" not in repr(messages)
    warning = logger.warning.call_args.args
    assert warning[:2] == (
        "event loop unresponsive for %.3f seconds\n%s",
        pytest.approx(0.05, abs=0.05),
    )
    assert "in answer" in warning[2]


def test_memory_budget_readiness_retries_response_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        status = 200

        async def __aenter__(self) -> Response:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def json(self) -> dict[str, object]:
            return {"memoryGovernor": {"ram": {"budgetBytes": 1}}}

    class TimedOutResponse(Response):
        async def __aenter__(self) -> Response:
            raise TimeoutError("response headers stalled")

    class Session:
        health_attempts = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        def get(self, url: str) -> Response:
            if url.endswith("/api/health"):
                type(self).health_attempts += 1
                if type(self).health_attempts == 1:
                    return TimedOutResponse()
            return Response()

    process = Mock()
    process.poll.return_value = None
    log = tmp_path / "serve.log"
    log.write_text("", encoding="utf-8")
    monkeypatch.setattr(aiohttp, "ClientSession", Session)

    assert_served_memory_budgets(process, 1, {"ram": 1}, log)

    assert Session.health_attempts == 2
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=30)


def test_memory_budget_failure_retains_server_output(tmp_path: Path) -> None:
    process = Mock()
    process.poll.return_value = 1
    log = tmp_path / "serve.log"
    log.write_text("startup reached p2p\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="serve process died") as raised:
        assert_served_memory_budgets(process, 1, {}, log)

    assert "server output:\nstartup reached p2p" in "\n".join(raised.value.__notes__)
    process.terminate.assert_not_called()
    process.wait.assert_called_once_with(timeout=30)


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
@pytest.mark.parametrize("disable_p2p", [False, True])
def test_library_startup_composes_without_pack_workers(
    tmp_path: Path, no_defaults: bool, launcher: bool, disable_p2p: bool
) -> None:
    import psutil

    from tools.benchmark_schema_catalog import ENTRY, bound_server, terminate_children

    entry = ENTRY
    port = free_port()
    extra = ("--no-default-packs",) if no_defaults else ()
    if disable_p2p:
        extra += ("--disable-p2p",)
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
            async with asyncio.timeout(30):
                while True:
                    assert process.poll() is None, log.read_text()
                    try:
                        async with session.get(f"http://127.0.0.1:{port}/api/composition") as resp:
                            report = await resp.json()
                        if not report.get("composing"):
                            break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)
            expected = set() if no_defaults else set(await _default_pack_names())
            assert set(report["packs"]) == expected, report
            assert all(pack["state"] == "announced" for pack in report["packs"].values()), report
            output = log.read_text()
            assert "BLOCKED_EXECUTION_IMPORT" not in output, output
            server = bound_server(process.pid, output.splitlines())
            if launcher:
                assert server.pid != process.pid
            async with session.get(f"http://127.0.0.1:{port}/api/settings") as resp:
                assert resp.status == 200
                settings = await resp.json()
            panel_value = settings["settings"]["p2p"]["value"]
            assert panel_value["downloadsEnabled"] is False
            assert panel_value["seedingEnabled"] is False
            async with session.get(f"http://127.0.0.1:{port}/api/p2p/status") as resp:
                assert resp.status == 200
                p2p = await resp.json()
            assert p2p["state"] == "disabled", p2p
            assert p2p["settings"]["downloadsEnabled"] is False, p2p
            assert p2p["settings"]["seedingEnabled"] is False, p2p
            assert p2p["sidecar"] is None, p2p
            assert server.children(recursive=True) == []

    try:
        asyncio.run(scenario())
    finally:
        children = terminate_children(process.pid)
        try:
            _, alive = psutil.wait_procs(children, timeout=15)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
        assert not alive, f"server descendants survived teardown: {alive}"


def test_degraded_default_ordering_composes_media_io_before_image(
    tmp_path: Path,
) -> None:
    """One pack with unresolvable contracts must not send the healthy defaults
    back to load order: dinkster-nodes-image sorts before dinkster-nodes-media-io
    alphabetically, so composing in load order validates the image pack's
    media-io requirement before the media-io pack has announced.

    Regression for Kosinkadink/comfy-vibe-station#242, where exactly this took
    down every hosted pack depending on the media-io chain.
    """
    import dinkster.serve as serve
    from dinkster.compose import (
        CompositionError,
        PackSpec,
        ServingComposer,
        default_pack_spec,
        load_manifest,
        resolve_manifest_path,
    )

    broken = tmp_path / "ordering-probe"
    broken.mkdir()
    (broken / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-ordering-probe"\nnamespaces = ["probe"]\n'
        "[pack.requirements.capabilities]\n"
        '"dinkster.ordering.probe" = ">=1.0.0,<2.0.0"\n'
        '[pack.entry]\nnodes = "probe_ordering_nodes:NODES"\n',
        encoding="utf-8",
    )
    specs: list[PackSpec] = [
        default_pack_spec("dinkster-nodes-foundation"),
        default_pack_spec("dinkster-nodes-image"),
        default_pack_spec("dinkster-nodes-media-io"),
        PackSpec(manifest=broken / "dinkster-pack.toml"),
    ]
    # Load order is alphabetical, and it puts the consumer before its provider.
    from dinkster.compose import default_pack_ids

    load_order = list(default_pack_ids())
    assert load_order.index("dinkster-nodes-image") < load_order.index("dinkster-nodes-media-io")
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
        assert list(errors) == ["dinkster-ordering-probe"]
        assert "dinkster.ordering.probe" in errors["dinkster-ordering-probe"]
        assert "dinkster-nodes-media-io" in deltas
        assert "dinkster.load_video" in deltas["dinkster-nodes-media-io"]
        assert "dinkster-nodes-image" in deltas, errors
        assert "dinkster.image.resize" in deltas["dinkster-nodes-image"]

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(composer.close())


def test_degraded_default_ordering_places_generation_before_model_packs(
    tmp_path: Path,
) -> None:
    from dinkster_workers import load_manifest

    import dinkster.serve as serve
    from dinkster.comfy_compose import comfy_compat_specs
    from dinkster.compose import PackSpec, ServingComposer, default_pack_specs, model_pack_specs

    broken = tmp_path / "ordering-probe"
    broken.mkdir()
    (broken / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-ordering-probe"\nnamespaces = ["probe"]\n'
        "[pack.requirements.capabilities]\n"
        '"dinkster.ordering.probe" = ">=1.0.0,<2.0.0"\n'
        '[pack.entry]\nnodes = "probe_ordering_nodes:NODES"\n',
        encoding="utf-8",
    )
    defaults = (*default_pack_specs(), *model_pack_specs(), *comfy_compat_specs())
    specs = (*defaults, PackSpec(manifest=broken / "dinkster-pack.toml"))
    composer = ServingComposer()
    try:
        ordered = serve._order_default_pack_specs(composer, specs, len(specs))
    finally:
        asyncio.run(composer.close())

    names = [load_manifest(Path(spec.manifest)).name for spec in ordered]
    generation = names.index("dinkster-nodes-generation")
    for model in (
        "dinkster-model-qwen-image",
        "dinkster-model-triposplat",
        "dinkster-model-wan",
    ):
        assert generation < names.index(model)


def test_serve_progressive_pack_announcement(tmp_path: Path) -> None:
    """The default and explicit packs announce after the diagnostic host binds."""
    manifest = write_iso_manifest(tmp_path / "pack")
    default_packs = asyncio.run(_default_pack_names())
    final_epoch = len(default_packs) + 2
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR)}
    process, log = start_logged_serve(
        serve_command(
            port,
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
            assert any(t.startswith("std.") for t in data["nodes"])
            assert data["nodes"]["std.math.add_ints"]["pack"] == "dinkster-nodes-foundation"
            assert data["nodes"]["dinkster.load_video"]["pack"] == "dinkster-nodes-media-io"
            assert data["nodes"]["dinkster.image.resize"]["pack"] == "dinkster-nodes-image"
            assert data["nodes"]["iso.chatty"]["pack"] == "isopack"
            named_route = data["nodes"]["dinkster.route.switch_by_name"]
            assert named_route["schemaVersion"] == 1
            assert named_route["interface"][0]["widget"] == {
                "type": "COMBO",
                "optionSource": {"inputFamily": "values"},
            }
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
            assert "dinkster.ksampler" in data["nodes"]
            assert report["packs"] == expected_packs

    pending_error: BaseException | None = None
    try:
        asyncio.run(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)


def test_default_catalog_publishes_only_owned_translation_carriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiohttp.test_utils import TestClient, TestServer

    from dinkster import serve

    def run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                # Catalog composition under coverage took about 78 s on self-hosted Linux (#160).
                async with asyncio.timeout(300):
                    while True:
                        payload = await (await client.get("/api/nodes")).json()
                        if "composing" not in payload:
                            break
                        await asyncio.sleep(0.05)
                report = await (await client.get("/api/composition")).json()
                assert report["packs"]
                assert all(row["state"] == "announced" for row in report["packs"].values()), report
                for pack_id, pack in payload["packs"].items():
                    for key in ("comfyAliases", "comfyGroups"):
                        for record in pack.get(key, {}).get("records", []):
                            assert payload["nodes"][record["carrier"]]["pack"] == pack_id
                for pack_id in ("dinkster-nodes-generation", "dinkster-nodes-image"):
                    for key in ("comfyAliases", "comfyGroups"):
                        assert payload["packs"][pack_id][key]["records"]
                # CI passes this real HTTP response to the frontend validator.
                if output := os.environ.get("DINKSTER_CATALOG_WIRE_OUTPUT"):
                    Path(output).write_text(json.dumps(payload), encoding="utf-8")
            finally:
                await client.close()

        asyncio.run(inspect())

    monkeypatch.setattr(serve.web, "run_app", run_app)
    monkeypatch.setattr(sys, "argv", ["dinkster-serve", "--library-root", ""])
    serve.main()


@pytest.mark.parametrize(
    ("failed_pack", "surviving_pack", "present_node", "missing_node"),
    [
        (
            "dinkster-nodes-foundation",
            "dinkster-nodes-media-io",
            "dinkster.load_video",
            "std.math.add_ints",
        ),
        (
            "dinkster-nodes-media-io",
            "dinkster-nodes-foundation",
            "std.math.add_ints",
            "dinkster.load_video",
        ),
        (
            "dinkster-nodes-image",
            "dinkster-nodes-foundation",
            "std.math.add_ints",
            "dinkster.image.resize",
        ),
    ],
)
def test_unavailable_default_pack_is_reported_after_diagnostic_host_binds(
    monkeypatch: pytest.MonkeyPatch,
    failed_pack: str,
    surviving_pack: str,
    present_node: str,
    missing_node: str,
) -> None:
    from aiohttp.test_utils import TestClient, TestServer

    from dinkster import serve

    observed: dict[str, object] = {}

    real_default_pack_spec = serve.default_pack_spec

    def fail_one_default(pack_id: str) -> object:
        if pack_id == failed_pack:
            raise RuntimeError("default pack bytes unavailable")
        return real_default_pack_spec(pack_id)

    def fake_run_app(awaitable: object, **_kwargs: object) -> None:
        async def inspect() -> None:
            app = await awaitable  # type: ignore[misc]
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                async with asyncio.timeout(10):
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
                        await asyncio.sleep(0)
                async with asyncio.timeout(10):
                    while "composing" in await (await client.get("/api/composition")).json():
                        await asyncio.sleep(0)
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

        asyncio.run(inspect())

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
    process = subprocess.Popen(
        serve_command(port, *args),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(str(manifest.parent) for manifest in manifests),
        },
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

        started = time.monotonic()
        process.send_signal(shutdown_signal)
        process.wait(timeout=8)
        elapsed = time.monotonic() - started

        assert process.returncode == 0
        assert elapsed < 8
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
        serve_command(port, "--pack", str(manifest)),
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
        serve_command(port, "--pack", str(manifest)),
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
    error, the default packs keep serving, and composition still completes -
    positively, with the failure on the record instead of a dead server
    (the anti-ComfyUI: broken extensions were just silently absent)."""
    manifest = write_iso_manifest(tmp_path / "pack", name="core")
    port = free_port()
    process = subprocess.Popen(
        serve_command(port, "--pack", str(manifest)),
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
            # The survivors (here: the default packs) serve, and the table says
            # complete - the failure is on the record, not on the surface.
            async with session.get(base + "/api/nodes") as resp:
                data = await resp.json()
            assert any(t.startswith("std.") for t in data["nodes"])
            assert "composing" not in data
            assert process.poll() is None, "pack failure must not kill serve"

    try:
        asyncio.run(scenario())
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
        serve_command(port, "--pack", str(manifest)),
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
        served = asyncio.run(nodes())
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
        serve_command(port, "--pack", str(manifest), "--strict-packs"),
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


def test_serve_runtime_mounts_end_to_end(tmp_path: Path) -> None:
    """Real dinkster-serve with a library root and --allow-mount-changes:
    grant a directory over the API, watch it scan to ready, browse its
    catalog, stream a mounted file by digest, and revoke it - the desktop
    folder-picker flow end to end, with mounts.toml as the durable record."""
    library_root = tmp_path / "library"
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
                assert (await resp.json())["mounts"] == []
            async with session.post(
                base + "/api/mounts", json={"id": "shots", "path": str(granted)}
            ) as resp:
                assert resp.status == 201
                created = await resp.json()
            assert (created["id"], created["source"]) == ("shots", "config")
            # Durable immediately, ready shortly after the background scan.
            assert "shots" in (library_root / "mounts.toml").read_text("utf-8")
            async with asyncio.timeout(60):
                while True:
                    async with session.get(base + "/api/mounts") as resp:
                        (row,) = (await resp.json())["mounts"]
                    if row["state"] == "ready":
                        break
                    await asyncio.sleep(0.05)
            assert row["entryCount"] == 1
            async with session.get(base + "/api/mounts/shots/entries") as resp:
                (entry,) = (await resp.json())["entries"]
            assert entry["virtualPath"] == "mounts/shots/pic.png"
            # The browsed file previews by digest with no vault copy.
            async with session.get(base + "/api/assets/" + entry["digest"]) as resp:
                assert resp.status == 200
                assert await resp.read() == payload
            async with session.delete(base + "/api/mounts/shots") as resp:
                assert resp.status == 200
            async with session.get(base + "/api/mounts") as resp:
                assert (await resp.json())["mounts"] == []
            async with session.get(base + "/api/assets/" + entry["digest"]) as resp:
                assert resp.status == 404
            assert "shots" not in (library_root / "mounts.toml").read_text("utf-8")

    try:
        asyncio.run(scenario())
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


def test_serve_mounted_save_end_to_end(tmp_path: Path) -> None:
    """The write path, real process to real bytes: grant a readwrite mount
    over the API while dinkster-serve runs, execute dev.image.save_pgm in
    that same process, verify the file landed inside the mount with the
    digest the job reported, stream it back via /api/assets, then revoke
    the mount and prove the very next save job fails - no restart
    anywhere."""
    from dinkster_assets import digest_bytes
    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire

    library_root = tmp_path / "library"
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    port = free_port()
    process, log = start_logged_serve(
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
            "--pack",
            str(
                Path(__file__).parent.parent
                / "packages"
                / "dinkster-nodes-dev"
                / "dinkster-pack.toml"
            ),
            "--event-loop-stall-threshold",
            "4",
        ],
        tmp_path,
    )

    def save_job(job_id: str) -> dict[str, object]:
        graph = Graph(
            nodes={
                "g": GraphNode("dev.image.gradient", {"width": 3, "height": 2}),
                "save": GraphNode(
                    "dev.image.save_pgm",
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
                    if "dev.image.save_pgm" in nodes:
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
                        (row,) = (await resp.json())["mounts"]
                    if row["state"] == "ready":
                        break
                    await asyncio.sleep(0.05)
            assert row["mode"] == "readwrite"
            # The grant is durable, mode included.
            config_text = (library_root / "mounts.toml").read_text("utf-8")
            assert "renders" in config_text and "readwrite" in config_text

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
                "frame_00001.pgm",
                "frame_00002.pgm",
            ]

    pending_error: BaseException | None = None
    try:
        asyncio.run(scenario())
    except BaseException as error:
        pending_error = error
        raise
    finally:
        stop_logged_serve(process, log, pending_error)
