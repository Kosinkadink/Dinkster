"""Multi-pack composition: --pack manifests become one serving surface.

What this proves: compose_serving merges default + configured pack schemas behind one
RoutingWorker (the engine cannot tell where a node runs), pack provenance
flows from the manifest's loading record to /api/nodes (attribution is
host-side, never a schema's own claim), and misconfiguration - reserved
pack ids, duplicate names, node-type collisions - fails loudly at startup
with no orphaned pack processes.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_caches import LayeredCache, MemoryLRUCache
from dinkster_compat_comfy import ResidentPool
from dinkster_engine import EngineEvent
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_memory import (
    GovernorReservationService,
    MeasuredMemory,
    MemoryGovernor,
    ReportedTelemetry,
)
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_image import IMAGE_NODES
from dinkster_nodes_media_io import MEDIA_IO_NODES
from dinkster_protocol import GRAPH_COMPILERS_SURFACE, KeyedContribution, extension_behavior_hash
from dinkster_schema import ComfyAliasRegistry, ComfyGroupRegistry, build_schemas
from dinkster_server import PackInfo, ServerLibrary, create_app
from dinkster_values import EncodedPayload, TypeRegistry, Value, ValueMeta, default_encode
from dinkster_workers import load_manifest
from dinkster_workers.doctor import prepare_catalog

from dinkster.compose import (
    Composition,
    CompositionError,
    PackSpec,
    ServingComposer,
    _merge_pack_entry,
    compose_serving,
    default_pack_spec,
    default_pack_specs,
    resolve_manifest_path,
)

TESTS_DIR = Path(__file__).parent
DEV_PACK_MANIFEST = TESTS_DIR.parent / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"
ATTENTION_PROVIDER = TESTS_DIR / "fixtures/attention_provider"
WORKER_ENV = {"PYTHONPATH": os.pathsep.join((str(ATTENTION_PROVIDER), str(TESTS_DIR)))}
DEFAULT_NODES = [*FOUNDATION_NODES, *MEDIA_IO_NODES, *IMAGE_NODES]
STANDARD_OWNER_PACKS = {
    "dinkster-nodes-foundation",
    "dinkster-nodes-image",
    "dinkster-nodes-media-io",
}
STANDARD_VISION_PACKS = {
    "dinkster-vision-birefnet",
    "dinkster-vision-depth-anything-v2",
    "dinkster-vision-depth-anything-v3",
    "dinkster-vision-detr",
    "dinkster-vision-efficient-sam",
    "dinkster-vision-hed",
    "dinkster-vision-rtdetr",
    "dinkster-vision-sam31",
    "dinkster-vision-upscale",
}
HED_PROVIDER_NODES = (
    "dinkster.preprocess.model_edges",
    "dinkster.preprocess.lineart_realistic",
    "dinkster.preprocess.lineart_anime",
    "dinkster.preprocess.lineart_manga",
    "dinkster.preprocess.anyline",
    "dinkster.preprocess.teed",
    "dinkster.preprocess.mlsd",
)


@pytest.fixture
def isolated_full_free_pool(monkeypatch: pytest.MonkeyPatch) -> ResidentPool:
    pool = ResidentPool()
    monkeypatch.setattr("dinkster.compose.default_pool", lambda: pool)
    return pool


def test_lazy_media_pack_resolves_assets_after_compat_host_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    import numpy as np
    from PIL import Image

    from dinkster.comfy_compose import register_comfy_host_types

    pixels = np.zeros((3, 5, 3), dtype=np.uint8)
    pixels[..., 1] = 255
    encoded = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(encoded, format="PNG")
    data = encoded.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    for key in ("DINKSTER_PACK_SCRATCH", "DINKSTER_MOUNTS_SNAPSHOT", "DINKSTER_ASSET_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    ref = AssetRef(digest, "input.png", len(data), "image/png")

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            media = replace(default_pack_spec("dinkster-nodes-media-io"), require_catalog=True)
            assert prepare_catalog(media.manifest, environment=os.environ).ok
            await composer.add_pack(media)
            register_comfy_host_types(composer.composition._registry)

            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(
                Graph(nodes={"load": GraphNode("dinkster.load_image", {"image": ref.to_wire()})}),
                ["load"],
            )
            loaded = cast("np.ndarray", result.outputs["load"]["image"].resolve())
            np.testing.assert_array_equal(loaded[0] * 255, pixels)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_comfy_model_roots_preserve_every_category_root_with_safe_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    shared = tmp_path / "shared-models"
    nested = shared / "nested"
    payload = {
        "checkpoints": {
            "kind": "model/checkpoint",
            "roots": [str(shared), str(nested)],
        },
        "loras": {
            "kind": "model/lora",
            "roots": [str(shared)],
        },
    }
    captured: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        comfy_compose,
        "find_compat_manifest",
        lambda _name: tmp_path / "dinkster-pack.toml",
    )
    monkeypatch.setattr(
        comfy_compose,
        "_comfy_python_selection",
        lambda _root, _explicit=None: comfy_compose._ComfyPythonSelection(
            "/comfy/python", "current Python"
        ),
    )
    monkeypatch.setattr(
        comfy_compose,
        "dinkster_pythonpath",
        lambda _manifest: "/dinkster/packages",
    )
    monkeypatch.setattr(comfy_compose, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(comfy_compose.subprocess, "run", fake_run)

    roots = comfy_compose.comfy_model_roots(
        root,
        comfy_args=("--extra-model-paths-config", "extra.yaml"),
    )
    assert [(entry.mount_id, entry.category, entry.kind, entry.path) for entry in roots] == [
        (
            "comfy-model-checkpoints-1",
            "checkpoints",
            "model/checkpoint",
            shared.resolve(),
        ),
        (
            "comfy-model-checkpoints-2",
            "checkpoints",
            "model/checkpoint",
            nested.resolve(),
        ),
        ("comfy-model-loras-1", "loras", "model/lora", shared.resolve()),
    ]
    assert captured["cwd"] == root
    assert captured["command"] == (
        "/comfy/python",
        "-c",
        "import json; from dinkster_compat_comfy.bootstrap import comfy_model_roots; "
        "print(json.dumps(comfy_model_roots(), separators=(',', ':')))",
        "--extra-model-paths-config",
        "extra.yaml",
    )
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["DINKSTER_COMFYUI_ROOT"] == str(root)
    assert str(env["PYTHONPATH"]).startswith("/dinkster/packages")


def test_comfy_model_roots_rejects_unusable_probe_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    monkeypatch.setattr(
        comfy_compose,
        "find_compat_manifest",
        lambda _name: tmp_path / "dinkster-pack.toml",
    )
    monkeypatch.setattr(
        comfy_compose,
        "_comfy_python_selection",
        lambda _root, _explicit=None: comfy_compose._ComfyPythonSelection(
            "/comfy/python", "current Python"
        ),
    )
    monkeypatch.setattr(comfy_compose, "dinkster_pythonpath", lambda _manifest: "")
    monkeypatch.setattr(comfy_compose, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(
        comfy_compose.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"checkpoints": {"kind": "model/checkpoint", "roots": ["relative"]}}),
            stderr="",
        ),
    )
    with pytest.raises(CompositionError, match="not absolute"):
        comfy_compose.comfy_model_roots(root)


def test_comfy_blake3_preflight_is_exact_isolated_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    captured: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(comfy_compose.subprocess, "run", fake_run)

    comfy_compose._probe_comfy_blake3("/selected/comfy/python")

    assert captured == {
        "command": ("/selected/comfy/python", "-I", "-c", "import blake3"),
        "capture_output": True,
        "text": True,
        "timeout": 5,
        "check": False,
        "shell": False,
    }


def test_comfy_python_selection_reports_resolution_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster.comfy_compose import _comfy_python_selection

    root = tmp_path / "ComfyUI"
    root.mkdir()
    monkeypatch.delenv("DINKSTER_COMFYUI_PYTHON", raising=False)
    assert _comfy_python_selection(root, "/cli/python") == ("/cli/python", "--comfy-python")

    monkeypatch.setenv("DINKSTER_COMFYUI_PYTHON", "/env/python")
    assert _comfy_python_selection(root) == ("/env/python", "DINKSTER_COMFYUI_PYTHON")

    monkeypatch.delenv("DINKSTER_COMFYUI_PYTHON")
    venv_python = root / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert _comfy_python_selection(root) == (
        str(venv_python),
        "<comfy-root>/venv/bin/python",
    )

    venv_python.unlink()
    assert _comfy_python_selection(root) == (sys.executable, "current Python")


@pytest.mark.parametrize("missing", [None, "einops"])
def test_comfy_requirements_probe_with_fake_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    requirements = root / "requirements.txt"
    requirements.write_text("einops\n")
    captured: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0 if missing is None else 1,
            stdout="{}" if missing is None else json.dumps({"missing": missing}),
            stderr="",
        )

    monkeypatch.setattr(comfy_compose.subprocess, "run", fake_run)
    selection = comfy_compose._ComfyPythonSelection("/fake/python", "--comfy-python")

    if missing is None:
        comfy_compose._probe_comfy_requirements(root, selection)
    else:
        with pytest.raises(CompositionError) as caught:
            comfy_compose._probe_comfy_requirements(root, selection)
        assert str(caught.value) == (
            "ComfyUI requirement module 'einops' is unavailable in interpreter "
            "'/fake/python' selected by --comfy-python"
        )

    command = captured.pop("command")
    assert isinstance(command, tuple)
    assert command[:3] == ("/fake/python", "-I", "-c")
    assert "packages_distributions" in command[3]
    assert command[4] == str(requirements)
    assert captured == {
        "capture_output": True,
        "text": True,
        "timeout": 60,
        "check": False,
        "shell": False,
    }


def test_comfy_requirements_probe_executes_imports(tmp_path: Path) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    requirements = root / "requirements.txt"
    selection = comfy_compose._ComfyPythonSelection(sys.executable, "current Python")

    requirements.write_text("json\n")
    comfy_compose._probe_comfy_requirements(root, selection)

    requirements.write_text("definitely-missing-comfy-requirement\n")
    with pytest.raises(CompositionError) as caught:
        comfy_compose._probe_comfy_requirements(root, selection)
    assert "definitely_missing_comfy_requirement" in str(caught.value)
    assert sys.executable in str(caught.value)
    assert "current Python" in str(caught.value)


def test_comfy_requirements_probe_preserves_backslashes_in_interpreter_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "requirements.txt").write_text("einops\n")
    interpreter = r"C:\actions-runners\Dinkster\_work\Dinkster\.venv\Scripts\python.exe"
    monkeypatch.setattr(
        comfy_compose.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout=json.dumps({"missing": "einops"}),
            stderr="",
        ),
    )

    with pytest.raises(CompositionError) as caught:
        comfy_compose._probe_comfy_requirements(
            root,
            comfy_compose._ComfyPythonSelection(interpreter, "current Python"),
        )

    assert str(caught.value) == (
        "ComfyUI requirement module 'einops' is unavailable in interpreter "
        f"'{interpreter}' selected by current Python"
    )
    assert r"\\" not in str(caught.value)


@pytest.mark.parametrize("stderr", ["No module named 'blake3'", "broken native extension"])
def test_comfy_blake3_preflight_refuses_missing_or_broken_import(
    stderr: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    monkeypatch.setattr(
        comfy_compose.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=stderr),
    )

    with pytest.raises(CompositionError) as caught:
        comfy_compose._probe_comfy_blake3("/selected/comfy/python")
    message = str(caught.value)
    assert "/selected/comfy/python" in message
    assert "/selected/comfy/python -m pip install blake3" in message
    assert stderr in message
    assert len(message) < 1_024


@pytest.mark.parametrize(
    "failure",
    [OSError("cannot spawn"), subprocess.TimeoutExpired(("python",), 5)],
)
def test_comfy_blake3_preflight_refuses_spawn_failure_and_timeout(
    failure: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(comfy_compose.subprocess, "run", fail)

    with pytest.raises(CompositionError) as caught:
        comfy_compose._probe_comfy_blake3("/selected/comfy/python")
    message = str(caught.value)
    assert "/selected/comfy/python" in message
    assert "/selected/comfy/python -m pip install blake3" in message


def test_comfy_compat_specs_probe_selected_interpreter_once_with_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import comfy_compose

    root = tmp_path / "ComfyUI"
    root.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(comfy_compose, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(
        comfy_compose,
        "_probe_comfy_requirements",
        lambda _root, selection: calls.append(f"requirements:{selection.interpreter}"),
    )
    monkeypatch.setattr(
        comfy_compose,
        "_probe_comfy_blake3",
        lambda interpreter: calls.append(f"blake3:{interpreter}"),
    )

    specs = comfy_compose.comfy_compat_specs(
        root,
        python="/selected/comfy/python",
        legacy_packs=[legacy],
    )

    assert len(specs) == 3
    assert Path(specs[0].manifest).name == "dinkster-pack.toml"
    assert specs[0].in_process is True
    assert calls == [
        "requirements:/selected/comfy/python",
        "blake3:/selected/comfy/python",
    ]


def write_sampling_host_manifest(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "sampling-host"\n'
        'namespaces = ["dinkster", "comfy"]\n\n'
        '[pack.arms]\nnative = ["dinkster.ksampler"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_host:NODES"\n'
        'arm_nodes = "s1_sampler_host:ARM_NODES"\n'
        'choices = "s1_sampler_host:choices"\n',
        encoding="utf-8",
    )
    return manifest


def write_inference_extension_manifest(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_empty:NODES"\n\n'
        '[pack.extension]\ninference = "unused_parent_fake:register"\n'
        'privileges = ["inference"]\n',
        encoding="utf-8",
    )
    return manifest


def graph_compiler(
    compiler_id: str, order: int | str, *, aliases: tuple[str, ...] = ()
) -> KeyedContribution:
    return KeyedContribution(
        surface_id=GRAPH_COMPILERS_SURFACE,
        id=compiler_id,
        aliases=aliases,
        behavior_metadata=(("contractVersion", 1), ("order", order)),
    )


def write_iso_manifest(
    directory: Path,
    name: str = "isopack",
    *,
    presentation: bool = True,
    namespaces: tuple[str, ...] | None = ("iso",),
) -> Path:
    """The iso test pack's nodes are iso.*, so its claim is "iso" - pack id
    and node namespace need not coincide (the std/comfy shape). None omits
    the field, leaving the default claim (the pack name)."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    claims = (
        ""
        if namespaces is None
        else "namespaces = [" + ", ".join(f'"{c}"' for c in namespaces) + "]\n"
    )
    body = (
        f'[pack]\nname = "{name}"\n{claims}\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    if presentation:
        body += (
            '\n[pack.presentation]\ndisplay_name = "Iso Pack"\n'
            'abbr = "ISO"\nmark = "\\U0001F9EA"\ncolor = "#336699"\n'
        )
    manifest.write_text(body)
    return manifest


def write_contract_manifest(
    directory: Path,
    name: str,
    declarations: str = "",
    *,
    namespace: str = "iso",
    nodes_entry: str = "isopack_nodes:NODES",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    types_entry = (
        'types = "isopack_nodes:register_types"\n' if nodes_entry == "isopack_nodes:NODES" else ""
    )
    manifest.write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["{namespace}"]\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        f"{declarations}"
        f'[pack.entry]\nnodes = "{nodes_entry}"\n'
        f"{types_entry}",
        encoding="utf-8",
    )
    return manifest


def test_resolve_manifest_path(tmp_path: Path) -> None:
    manifest = write_iso_manifest(tmp_path / "pack")
    assert resolve_manifest_path(manifest) == manifest
    assert resolve_manifest_path(manifest.parent) == manifest
    with pytest.raises(CompositionError, match="not found"):
        resolve_manifest_path(tmp_path / "nowhere")


def test_pack_contract_resolver_orders_dependencies_and_capability_providers(
    tmp_path: Path,
) -> None:
    provider = write_contract_manifest(
        tmp_path / "provider",
        "provider",
        '[pack.capabilities]\n"provider.video-generation" = "2.1.0"\n',
    )
    consumer = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.dependencies]\nprovider = ">=2,<3"\n'
        "[pack.requirements.capabilities]\n"
        '"provider.video-generation" = ">=2,<3"\n',
    )
    digest = "blake3:" + "a" * 64
    provider_spec = PackSpec(
        provider,
        packs={
            "provider": PackInfo(display_name="Provider", version="2.0.0", artifact_digest=digest)
        },
    )
    consumer_spec = PackSpec(
        consumer,
        packs={
            "consumer": PackInfo(display_name="Consumer", version="1.0.0", artifact_digest=digest)
        },
    )
    composer = ServingComposer()
    try:
        ordered = composer.order_pack_entries((consumer_spec, provider_spec))
        assert [Path(spec.manifest) for spec in ordered] == [provider, consumer]
    finally:
        asyncio.run(composer.close())


def test_serving_composer_registers_inference_boundary_types() -> None:
    from dinkster_values import TypeRegistry

    inference_types = {
        "dinkster.conditioning",
        "dinkster.latent",
        "dinkster.control",
        "dinkster.model",
        "dinkster.clip",
        "dinkster.vae",
    }
    default = ServingComposer()
    supplied_registry = TypeRegistry()
    supplied_registry.register("test.caller_value")
    supplied = ServingComposer(registry=supplied_registry)
    try:
        assert all(type_id in default.composition._registry for type_id in inference_types)
        assert all(type_id in supplied.composition._registry for type_id in inference_types)
        assert "test.caller_value" in supplied.composition._registry
    finally:
        asyncio.run(default.close())
        asyncio.run(supplied.close())


def test_compose_serving_orders_provider_before_consumer(tmp_path: Path) -> None:
    provider = write_contract_manifest(
        tmp_path / "provider",
        "provider",
        '[pack.capabilities]\n"provider.video-generation" = "2.1.0"\n',
        namespace="provider",
        nodes_entry="empty_pack_nodes:NODES",
    )
    consumer = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.dependencies]\nprovider = ">=2,<3"\n'
        "[pack.requirements.capabilities]\n"
        '"provider.video-generation" = ">=2,<3"\n',
    )
    digest = "blake3:" + "d" * 64
    specs = (
        PackSpec(
            consumer,
            packs={
                "consumer": PackInfo(
                    display_name="Consumer", version="1.0.0", artifact_digest=digest
                )
            },
        ),
        PackSpec(
            provider,
            packs={
                "provider": PackInfo(
                    display_name="Provider", version="2.0.0", artifact_digest=digest
                )
            },
        ),
    )

    async def scenario() -> None:
        composition = await compose_serving(
            specs, include_default_packs=False, worker_env=WORKER_ENV
        )
        try:
            assert set(composition.packs) == {"consumer", "provider"}
            assert [item.pack for item in composition.generation.packs] == [
                "consumer",
                "provider",
            ]
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_pack_contract_resolver_refuses_cycles_collisions_and_missing_registry_ids(
    tmp_path: Path,
) -> None:
    digest = "blake3:" + "b" * 64

    def spec(path: Path, name: str) -> PackSpec:
        return PackSpec(
            path,
            packs={name: PackInfo(display_name=name, version="1.0.0", artifact_digest=digest)},
        )

    alpha = write_contract_manifest(
        tmp_path / "alpha", "alpha", '[pack.dependencies]\nbeta = ">=1"\n'
    )
    beta = write_contract_manifest(
        tmp_path / "beta", "beta", '[pack.dependencies]\nalpha = ">=1"\n'
    )
    composer = ServingComposer()
    try:
        with pytest.raises(CompositionError, match="dependency cycle"):
            composer.order_pack_entries((spec(alpha, "alpha"), spec(beta, "beta")))

        first = write_contract_manifest(
            tmp_path / "first",
            "first",
            '[pack.capabilities]\n"shared.provider-id" = "1.0.0"\n',
        )
        second = write_contract_manifest(
            tmp_path / "second",
            "second",
            '[pack.capabilities]\n"shared.provider_id" = "2.0.0"\n',
        )
        with pytest.raises(CompositionError, match="provided by both"):
            composer.order_pack_entries((spec(first, "first"), spec(second, "second")))

        missing = write_contract_manifest(
            tmp_path / "missing",
            "missing",
            '[pack.requirements.registry]\n"dinkster.model-families" = ["missing.family"]\n',
        )
        with pytest.raises(CompositionError, match="unavailable"):
            composer.order_pack_entries((spec(missing, "missing"),))

        invalid_provenance = PackSpec(
            write_contract_manifest(tmp_path / "invalid-provenance", "invalid-provenance"),
            packs={
                "invalid-provenance": PackInfo(
                    display_name="Invalid", artifact_digest="not-a-digest"
                )
            },
        )
        with pytest.raises(CompositionError, match="invalid artifact digest"):
            composer.order_pack_entries((invalid_provenance,))

        invalid_version = PackSpec(
            write_contract_manifest(tmp_path / "invalid-version", "invalid-version"),
            packs={"invalid-version": PackInfo(display_name="Invalid", version="latest")},
        )
        with pytest.raises(CompositionError, match="invalid version"):
            composer.order_pack_entries((invalid_version,))

        contract_mismatch = write_contract_manifest(tmp_path / "mismatch", "mismatch")
        contract_mismatch.write_text(
            contract_mismatch.read_text().replace("dinkster-pack-host/1", "dinkster-pack-host/2")
        )
        with pytest.raises(CompositionError, match="requires host contract"):
            composer.order_pack_entries((spec(contract_mismatch, "mismatch"),))
    finally:
        asyncio.run(composer.close())


def test_pack_registry_providers_order_consumers_and_report_conflicts(tmp_path: Path) -> None:
    digest = "blake3:" + "9" * 64

    def spec(path: Path, name: str) -> PackSpec:
        return PackSpec(
            path,
            packs={name: PackInfo(display_name=name, version="1.0.0", artifact_digest=digest)},
        )

    provider = write_contract_manifest(
        tmp_path / "provider",
        "provider",
        '[pack.provides.registry]\n"dinkster.samplers" = ["provider.sampler"]\n'
        '"dinkster.model-families" = ["provider.family"]\n',
    )
    consumer = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.requirements.registry]\n"dinkster.samplers" = ["provider.sampler"]\n'
        '"dinkster.model-families" = ["provider.family"]\n',
    )
    composer = ServingComposer()
    try:
        ordered = composer.order_pack_entries(
            (spec(consumer, "consumer"), spec(provider, "provider"))
        )
        assert [Path(item.manifest) for item in ordered] == [provider, consumer]

        duplicate = write_contract_manifest(
            tmp_path / "duplicate",
            "duplicate",
            '[pack.provides.registry]\n"dinkster.samplers" = ["provider.sampler"]\n',
        )
        with pytest.raises(CompositionError, match="provided by both"):
            composer.order_pack_entries((spec(provider, "provider"), spec(duplicate, "duplicate")))

        builtin_collision = write_contract_manifest(
            tmp_path / "builtin-collision",
            "builtin-collision",
            '[pack.provides.registry]\n"dinkster.samplers" = ["dinkster.euler"]\n',
        )
        with pytest.raises(CompositionError, match="dinkster-inference/1.*builtin-collision"):
            composer.order_pack_entries((spec(builtin_collision, "builtin-collision"),))

        alpha = write_contract_manifest(
            tmp_path / "registry-alpha",
            "registry-alpha",
            '[pack.provides.registry]\n"dinkster.samplers" = ["alpha.sampler"]\n'
            '[pack.requirements.registry]\n"dinkster.schedulers" = ["beta.scheduler"]\n',
        )
        beta = write_contract_manifest(
            tmp_path / "registry-beta",
            "registry-beta",
            '[pack.provides.registry]\n"dinkster.schedulers" = ["beta.scheduler"]\n'
            '[pack.requirements.registry]\n"dinkster.samplers" = ["alpha.sampler"]\n',
        )
        with pytest.raises(
            CompositionError,
            match=(
                "dependency cycle: registry-alpha -> registry-beta, registry-beta -> registry-alpha"
            ),
        ):
            composer.order_pack_entries(
                (spec(alpha, "registry-alpha"), spec(beta, "registry-beta"))
            )
    finally:
        asyncio.run(composer.close())


def test_composed_generation_records_contract_and_registry_resolution(tmp_path: Path) -> None:
    manifest = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.requirements.registry]\n"dinkster.samplers" = ["dinkster.euler"]\n',
    )
    digest = "blake3:" + "c" * 64

    async def scenario() -> None:
        composition = await compose_serving(
            [
                PackSpec(
                    manifest,
                    packs={
                        "consumer": PackInfo(
                            display_name="Consumer",
                            version="1.0.0",
                            artifact_digest=digest,
                        )
                    },
                )
            ],
            include_default_packs=False,
            worker_env=WORKER_ENV,
            composition_mode="production",
        )
        try:
            generation = composition.generation
            assert generation.mode == "production"
            assert generation.packs[0].artifact_digest == digest
            receipts = {
                (item.kind, item.requirement, item.provider) for item in generation.resolutions
            }
            assert receipts == {
                ("host", "dinkster-pack-host/1", "dinkster-pack-host/1"),
                ("api", "dinkster-api/v1", "dinkster-api/v1"),
                ("registry", "dinkster.samplers:dinkster.euler", "dinkster-inference/1"),
            }
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_unpinned_pack_records_development_mode_when_production_was_requested(
    tmp_path: Path,
) -> None:
    manifest = write_contract_manifest(tmp_path / "local", "local")

    async def scenario() -> None:
        composition = await compose_serving(
            [manifest],
            include_default_packs=False,
            worker_env=WORKER_ENV,
            composition_mode="production",
        )
        try:
            assert composition.generation.mode == "development"
            assert composition.generation.packs[0].artifact_digest == ""
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_compose_defaults_only(tmp_path: Path) -> None:
    """Installed defaults provide the standard surface through independent packs."""

    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert set(composition.schemas) == set(build_schemas(DEFAULT_NODES))
            assert not any(t.startswith("dev.") for t in composition.schemas)
            assert set(composition.packs) == (
                STANDARD_OWNER_PACKS | STANDARD_VISION_PACKS | {"dinkster-nodes-remote"}
            )
            assert set(composition.node_packs.values()) == STANDARD_OWNER_PACKS
            assert len(FOUNDATION_NODES) == 52
            assert len(IMAGE_NODES) == 87
            assert len(MEDIA_IO_NODES) == 66
            assert {
                node_type
                for node_type, pack_id in composition.node_packs.items()
                if pack_id == "dinkster-nodes-foundation"
            } == set(build_schemas(FOUNDATION_NODES))
            assert {
                node_type
                for node_type, pack_id in composition.node_packs.items()
                if pack_id == "dinkster-nodes-media-io"
            } == set(build_schemas(MEDIA_IO_NODES))
            assert {
                node_type
                for node_type, pack_id in composition.node_packs.items()
                if pack_id == "dinkster-nodes-image"
            } == set(build_schemas(IMAGE_NODES))
            locked = default_pack_specs()
            assert {
                next(iter(spec.packs))
                for spec in locked
                if spec.packs is not None and not spec.in_process
            } == STANDARD_VISION_PACKS | {"dinkster-nodes-remote"}
            for node_type in HED_PROVIDER_NODES:
                assert composition.choices[f"{node_type}.providers"] == ("dinkster-vision-hed",)
            expected_digests = {
                pack_id: info.artifact_digest
                for spec in locked
                if spec.packs is not None
                for pack_id, info in spec.packs.items()
            }
            assert {
                pack.pack: pack.artifact_digest for pack in composition.generation.packs
            } == expected_digests
            engine = composition.make_engine(lambda event: None)
            runtime = engine.pin_execution()
            assert runtime.resolve_providers is not None
            for node_type in HED_PROVIDER_NODES:
                unresolved = Graph(nodes={"vision": GraphNode(node_type, {})})
                resolved = runtime.resolve_providers(unresolved)
                unresolved_node = unresolved.nodes["vision"]
                resolved_node = resolved.nodes["vision"]
                assert isinstance(unresolved_node, GraphNode)
                assert isinstance(resolved_node, GraphNode)
                assert "provider" not in unresolved_node.inputs
                assert resolved_node.inputs["provider"] == "dinkster-vision-hed"
            graph = Graph(nodes={"g": GraphNode("std.math.add_ints", {"a": 2, "b": 3})})
            result = await engine.run(graph, ["g"])
            assert result.outputs["g"]["sum"].resolve() == 5

            library = ServerLibrary(
                vault=AssetVault(tmp_path / "vault"),
                store=LibraryStore(tmp_path / "library.sqlite"),
                pack_assets=composition.asset_catalog,
            )
            app = create_app(
                composition.make_engine,
                composition.schemas,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                library=library,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                exact_graph = Graph(
                    nodes={
                        "source": GraphNode(
                            "dinkster.image.generate",
                            {"width": 8, "height": 8},
                            slot_variants={"color_source": "hex", "operation": "solid"},
                        ),
                        "realistic-line-art": GraphNode(
                            "dinkster.preprocess.lineart_realistic",
                            {"image": Link("source", "image")},
                        ),
                    }
                )
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "realistic-line-art",
                        "jobId": "preflight",
                        "graph": graph_to_wire(exact_graph),
                        "targets": ["realistic-line-art"],
                    },
                )
                payload = await response.json()
                assert response.status == 409, payload
                assert payload["error"] == "assets-missing"
                assert {asset["name"] for asset in payload["assets"]} == {
                    "ControlNet realistic line-art detector",
                    "ControlNet coarse realistic line-art detector",
                }
                assert all(asset["fetchable"] for asset in payload["assets"])
                assert {asset["name"]: asset["sources"] for asset in payload["assets"]} == {
                    "ControlNet realistic line-art detector": [
                        "https://huggingface.co/lllyasviel/Annotators/resolve/"
                        "982e7edaec38759d914a963c48c4726685de7d96/sk_model.pth"
                    ],
                    "ControlNet coarse realistic line-art detector": [
                        "https://huggingface.co/lllyasviel/Annotators/resolve/"
                        "982e7edaec38759d914a963c48c4726685de7d96/sk_model2.pth"
                    ],
                }
                exact_node = exact_graph.nodes["realistic-line-art"]
                assert isinstance(exact_node, GraphNode)
                assert "provider" not in exact_node.inputs
            finally:
                await client.close()
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_compose_serving_default_resolution_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_defaults() -> tuple[PackSpec, ...]:
        raise CompositionError("default pack unavailable")

    monkeypatch.setattr("dinkster.compose.default_pack_specs", fail_defaults)

    with pytest.raises(CompositionError, match="default pack unavailable"):
        asyncio.run(compose_serving())


def test_graph_compiler_generation_orders_identity_and_binds_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    write_sampling_host_manifest(tmp_path / "host"),
                    trust_reserved=True,
                )
            )
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None
            declarations: list[tuple[str, tuple[KeyedContribution, ...]]] = []
            materialized: list[str] = []
            released: list[str] = []
            compile_calls: list[tuple[str, object, object]] = []
            compile_result = {"graph": {}, "targets": []}

            async def materialize(key: str):
                materialized.append(key)
                return tuple(declarations)

            async def release(key: str) -> None:
                released.append(key)

            async def compile_graph(key: str, graph: object, targets: object):
                compile_calls.append((key, graph, targets))
                return compile_result

            monkeypatch.setattr(worker, "materialize_inference_generation", materialize)
            monkeypatch.setattr(worker, "release_inference_generation", release)
            monkeypatch.setattr(worker, "compile_graph", compile_graph)

            declarations[:] = [
                ("proof_b", (graph_compiler("proof_b.second", 2),)),
            ]
            await composer.add_pack(
                write_inference_extension_manifest(tmp_path / "proof_b", "proof_b")
            )
            old_runtime = composer._runtime_seat.pin()
            old_digest = "sha256:" + extension_behavior_hash(old_runtime.extension_snapshot)

            declarations[:] = [
                (
                    "proof_a",
                    (
                        graph_compiler("proof_a.tie", 2),
                        graph_compiler("proof_a.first", -1),
                    ),
                ),
                ("proof_b", (graph_compiler("proof_b.second", 2),)),
            ]
            await composer.add_pack(
                write_inference_extension_manifest(tmp_path / "proof_a", "proof_a")
            )
            runtime = composer._runtime_seat.pin()
            digest = "sha256:" + extension_behavior_hash(runtime.extension_snapshot)
            assert tuple(
                contribution.id for contribution in runtime.graph_compiler_registry.contributions
            ) == ("proof_a.first", "proof_a.tie", "proof_b.second")
            assert tuple(
                contribution.id
                for extension in runtime.extension_snapshot.extensions
                for contribution in extension.keyed_contributions
            ) == ("proof_a.tie", "proof_a.first", "proof_b.second")
            assert runtime.graph_compile_transport is not None

            graph = {"node": {"type": "std.math.add_ints"}}
            targets = ("node", "other")
            assert await runtime.graph_compile_transport(digest, graph, targets) is compile_result
            assert compile_calls == [(digest, graph, targets)]
            with pytest.raises(AssertionError, match="owning digest"):
                await runtime.graph_compile_transport("sha256:" + "0" * 64, graph, targets)
            assert compile_calls == [(digest, graph, targets)]

            assert old_runtime.graph_compile_transport is not None
            await old_runtime.graph_compile_transport(old_digest, graph, targets)
            assert compile_calls[-1] == (old_digest, graph, targets)
            with pytest.raises(AssertionError):
                await old_runtime.graph_compile_transport(digest, graph, targets)
            assert compile_calls[-1] == (old_digest, graph, targets)

            class UnexpectedWorker:
                async def compile_graph(self, *_args: object):
                    raise AssertionError("transport resolved the current worker")

            with monkeypatch.context() as current_lookup:
                current_lookup.setattr(
                    composer,
                    "_sampling_worker",
                    lambda _topology: UnexpectedWorker(),
                )
                await runtime.graph_compile_transport(digest, graph, targets)
            assert compile_calls[-1] == (digest, graph, targets)
            assert worker is composer._sampling_worker(composer._topology)
            assert materialized[-1] == digest
            assert any(key.startswith("candidate:") for key in released)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_noncompiler_inference_generation_has_no_compile_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_inference import INFERENCE_SAMPLERS_SURFACE

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    write_sampling_host_manifest(tmp_path / "host"),
                    trust_reserved=True,
                )
            )
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None
            sampler = KeyedContribution(
                surface_id=INFERENCE_SAMPLERS_SURFACE,
                id="sampler_only.proof",
            )
            compile_calls: list[object] = []

            async def materialize(_key: str):
                return (("sampler_only", (sampler,)),)

            async def compile_graph(*args: object):
                compile_calls.append(args)
                return {}

            monkeypatch.setattr(worker, "materialize_inference_generation", materialize)
            monkeypatch.setattr(worker, "compile_graph", compile_graph)
            await composer.add_pack(
                write_inference_extension_manifest(tmp_path / "sampler_only", "sampler_only")
            )
            runtime = composer._runtime_seat.pin()
            assert runtime.graph_compiler_registry.contributions == ()
            assert runtime.graph_compile_transport is None
            assert compile_calls == []
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_invalid_graph_compilers_fail_before_final_generation_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    write_sampling_host_manifest(tmp_path / "host"),
                    trust_reserved=True,
                )
            )
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None
            declarations: list[tuple[str, tuple[KeyedContribution, ...]]] = []
            materialized: list[str] = []
            released: list[str] = []

            async def materialize(key: str):
                materialized.append(key)
                return tuple(declarations)

            async def release(key: str) -> None:
                released.append(key)

            monkeypatch.setattr(worker, "materialize_inference_generation", materialize)
            monkeypatch.setattr(worker, "release_inference_generation", release)

            declarations[:] = [
                ("good", (graph_compiler("good.compiler", 0),)),
            ]
            await composer.add_pack(write_inference_extension_manifest(tmp_path / "good", "good"))
            published = composer._runtime_seat.pin()
            catalog_before = json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))
            final_count = sum(not key.startswith("candidate:") for key in materialized)

            invalid_cases = (
                (
                    "malformed",
                    (graph_compiler("malformed.compiler", "not-an-int"),),
                ),
                (
                    "duplicate",
                    (
                        graph_compiler("duplicate.compiler", 1),
                        graph_compiler("duplicate.compiler", 1),
                    ),
                ),
            )
            for name, invalid in invalid_cases:
                declarations[:] = sorted(
                    [
                        (name, invalid),
                        ("good", (graph_compiler("good.compiler", 0),)),
                    ]
                )
                before_calls = len(materialized)
                with pytest.raises(CompositionError, match="invalid graph compiler composition"):
                    await composer.add_pack(
                        write_inference_extension_manifest(tmp_path / name, name)
                    )
                attempted = materialized[before_calls:]
                assert len(attempted) == 1 and attempted[0].startswith("candidate:")
                assert attempted[0] in released
                assert composer._runtime_seat.pin() is published
                assert name not in composer.pack_specs()
                assert (
                    json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))
                    == catalog_before
                )
                assert sum(not key.startswith("candidate:") for key in materialized) == final_count
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("pack_id", "expected_digest"),
    [
        (
            "dinkster-nodes-foundation",
            "blake3:4b2babc83d81f1275238b69031fd1c8a5e5aed07065ed75b56fc432c23a8f064",
        ),
        (
            "dinkster-nodes-media-io",
            "blake3:89fc8de3ac0477fad3622c57fbc884facb6546125757115eb6bdb9223530892f",
        ),
        (
            "dinkster-nodes-image",
            "blake3:2f872d26be5cf8dc3a592384e22fe3d227f516e93f1313d49dc8a3354bbad3cb",
        ),
        (
            "dinkster-nodes-remote",
            "blake3:4685f09942fd5d6a3d85bb559f42fd99d69484dd8f6253f5dda0f2a1c3e7be39",
        ),
    ],
)
def test_default_pack_prefers_loaded_source_tree_over_bundled_copy(
    pack_id: str,
    expected_digest: str,
) -> None:
    source_root = Path(__file__).parent.parent / "packages" / pack_id
    spec = default_pack_spec(pack_id)
    assert Path(spec.manifest).resolve() == (source_root / "dinkster-pack.toml").resolve()
    assert spec.packs is not None
    info = spec.packs[pack_id]
    assert info.source == f"python:{pack_id}==0.0.1"
    assert info.version == "0.0.1"
    assert info.artifact_digest == expected_digest


def test_foundation_default_pack_ships_docs() -> None:
    spec = default_pack_spec("dinkster-nodes-foundation")
    assert spec.packs is not None
    docs = spec.packs["dinkster-nodes-foundation"].docs
    assert docs is not None
    assert [(page.kind, page.id, page.locale) for page in docs.pages] == [
        ("node", "std.math.add_ints", "en"),
        ("guide", "map-and-gather", "en"),
    ]
    page = next(page for page in docs.pages if page.kind == "node")
    assert page.title == "Add Integers"
    assert page.summary == "Adds two integer values."
    assert page.schema_version == 1
    assert page.data == (
        b"\n# Add Integers\n\nConnect integer values to `a` and `b`. "
        b"The `sum` output is their arithmetic sum.\n"
    )
    assert page.digest == "sha256:" + hashlib.sha256(page.data).hexdigest()


def test_default_pack_artifact_files_have_explicit_line_ending_policy() -> None:
    from dinkster import compose

    repo_root = TESTS_DIR.parent
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "packages"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    tracked_paths = {Path(raw.decode("utf-8")) for raw in listed.split(b"\0") if raw}
    artifact_paths: set[Path] = set()
    for pack_id, module_name in compose._FIRST_PARTY_PACK_MODULES.items():
        distribution_id = compose._FIRST_PARTY_PACK_DISTRIBUTIONS.get(pack_id, pack_id)
        pack_root = Path("packages") / distribution_id
        sidecar_root = (
            pack_root / f"{pack_id.replace('-', '_')}_pack"
            if distribution_id != pack_id
            else pack_root
        )
        manifest_path = sidecar_root / "dinkster-pack.toml"
        manifest = tomllib.loads((repo_root / manifest_path).read_text(encoding="utf-8"))
        docs = manifest.get("pack", {}).get("docs", {})
        docs_dir = docs.get("dir") if isinstance(docs, dict) else None
        module_root = pack_root / "src" / Path(*module_name.split("."))
        prefixes = [module_root, sidecar_root / "locales"]
        if isinstance(docs_dir, str):
            prefixes.append(sidecar_root / docs_dir)
        artifact_paths.add(manifest_path)
        module_init = module_root.parent / "__init__.py"
        if distribution_id != pack_id and module_init in tracked_paths:
            artifact_paths.add(module_init)
        if module_root.parent.name == "dinkster_nodes_vision":
            artifact_paths.update(path for path in tracked_paths if path.parent == sidecar_root)
        artifact_paths.update(
            sidecar_root / filename
            for filename in compose._PACK_ARTIFACT_SIDECARS
            if sidecar_root / filename in tracked_paths
        )
        artifact_paths.update(
            path
            for path in tracked_paths
            if any(path.is_relative_to(prefix) for prefix in prefixes)
        )

    checked = subprocess.run(
        [
            "git",
            "check-attr",
            "-z",
            "text",
            "eol",
            "--",
            *(path.as_posix() for path in sorted(artifact_paths)),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    attributes: dict[str, dict[str, str]] = {}
    for index in range(0, len(checked) - 1, 3):
        path, attribute, value = (part.decode("utf-8") for part in checked[index : index + 3])
        attributes.setdefault(path, {})[attribute] = value

    uncovered = [
        path.as_posix()
        for path in sorted(artifact_paths)
        if attributes[path.as_posix()]["eol"] != "lf"
        and attributes[path.as_posix()]["text"] != "unset"
    ]
    assert uncovered == [], (
        "pack artifact files must declare eol=lf for text or -text for binary: "
        + ", ".join(uncovered)
    )


@pytest.mark.all_file_shards
def test_vision_pack_license_worktree_bytes_are_lf() -> None:
    repo_root = TESTS_DIR.parent
    vision_root = repo_root / "packages/dinkster-nodes-vision"
    licenses = sorted(vision_root.glob("*_pack/*_LICENSE"))

    assert [path.relative_to(vision_root).as_posix() for path in licenses] == [
        "dinkster_vision_hed_pack/LINEART_LICENSE",
        "dinkster_vision_hed_pack/MANGA_LICENSE",
        "dinkster_vision_hed_pack/MLSD_LICENSE",
        "dinkster_vision_hed_pack/TEED_LICENSE",
        "dinkster_vision_sam31_pack/CLIP_LICENSE",
        "dinkster_vision_sam31_pack/SAM_LICENSE",
    ]
    assert [
        path.relative_to(repo_root).as_posix() for path in licenses if b"\r" in path.read_bytes()
    ] == []


def test_default_pack_publishes_alias_registry_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import compose

    aliases = ComfyAliasRegistry(source_schemas=(), records=())
    real_load_manifest = compose.load_manifest

    def load_manifest_with_aliases(path: Path | str):
        return replace(real_load_manifest(path), comfy_aliases=aliases)

    monkeypatch.setattr(compose, "load_manifest", load_manifest_with_aliases)
    spec = default_pack_spec("dinkster-nodes-foundation")
    assert spec.packs is not None
    assert spec.packs["dinkster-nodes-foundation"].comfy_aliases is aliases


def test_default_pack_publishes_group_registry_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import compose

    groups = ComfyGroupRegistry(source_schemas=(), group_schemas=(), records=())
    real_load_manifest = compose.load_manifest

    def load_manifest_with_groups(path: Path | str):
        return replace(real_load_manifest(path), comfy_groups=groups)

    monkeypatch.setattr(compose, "load_manifest", load_manifest_with_groups)
    spec = default_pack_spec("dinkster-nodes-foundation")
    assert spec.packs is not None
    assert spec.packs["dinkster-nodes-foundation"].comfy_groups is groups


def test_default_pack_uses_embedded_wheel_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import compose

    lock = compose._default_pack_lock()
    source = Path(__file__).parent.parent / "packages/dinkster-nodes-media-io"
    module = tmp_path / "site/dinkster_nodes_media_io/__init__.py"
    shutil.copytree(source / "src/dinkster_nodes_media_io", module.parent)
    bundled = tmp_path / "site/dinkster_nodes_media_io_pack/dinkster-pack.toml"
    bundled.parent.mkdir(parents=True)
    shutil.copy2(source / "dinkster-pack.toml", bundled)
    shutil.copy2(source / "comfy-aliases.json", bundled.parent / "comfy-aliases.json")
    shutil.copytree(
        source / "src/dinkster_nodes_media_io", bundled.parent / "dinkster_nodes_media_io"
    )
    for filename in compose._PACK_ARTIFACT_SIDECARS:
        sidecar = source / filename
        if sidecar.is_file():
            shutil.copy2(sidecar, bundled.parent / filename)
    distribution = SimpleNamespace(version="0.0.1", locate_file=lambda _path: bundled)

    monkeypatch.setattr(
        "dinkster.compose.importlib.metadata.distribution", lambda _name: distribution
    )
    monkeypatch.setattr("dinkster.compose._default_pack_lock", lambda: lock)
    monkeypatch.setattr(
        "dinkster.compose.importlib.util.find_spec",
        lambda _name: SimpleNamespace(origin=str(module)),
    )

    spec = default_pack_spec("dinkster-nodes-media-io")
    assert spec.manifest == bundled
    assert spec.packs is not None
    info = spec.packs["dinkster-nodes-media-io"]
    assert info.source == "python:dinkster-nodes-media-io==0.0.1"
    locked = lock.get("dinkster-nodes-media-io")
    assert locked is not None
    assert info.artifact_digest == locked.artifact_digest


@pytest.mark.parametrize("filename", ("comfy-aliases.json", "comfy-groups.json"))
def test_installed_pack_digest_includes_standard_sidecars(tmp_path: Path, filename: str) -> None:
    from dinkster import compose

    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "dinkster-pack.toml"
    manifest.write_text('[pack]\nname = "test"\nversion = "1.0.0"\n', encoding="utf-8")
    module = source / "test_nodes"
    module.mkdir()
    (module / "__init__.py").write_text("NODES = []\n", encoding="utf-8")
    sidecar = source / filename
    sidecar.write_text('{"records":[]}\n', encoding="utf-8")

    first = compose._installed_pack_digest(manifest, module)
    sidecar.write_text('{"records":[{"id":"changed"}]}\n', encoding="utf-8")
    second = compose._installed_pack_digest(manifest, module)

    assert first != second


def test_installed_pack_digest_rejects_docs_symlinks(tmp_path: Path) -> None:
    from dinkster_registry import ArtifactError

    from dinkster import compose

    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "test"\nversion = "1.0.0"\n[pack.docs]\ndir = "docs"\n',
        encoding="utf-8",
    )
    module = source / "test_nodes"
    module.mkdir()
    (module / "__init__.py").write_text("NODES = []\n", encoding="utf-8")
    docs = source / "docs"
    docs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("external bytes\n", encoding="utf-8")
    try:
        (docs / "linked.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ArtifactError, match="symlink"):
        compose._installed_pack_digest(manifest, module)


def test_installed_pack_digest_includes_locale_catalogs(tmp_path: Path) -> None:
    from dinkster import compose

    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "dinkster-pack.toml"
    manifest.write_text('[pack]\nname = "test"\nversion = "1.0.0"\n', encoding="utf-8")
    module = source / "test_nodes"
    module.mkdir()
    (module / "__init__.py").write_text("NODES = []\n", encoding="utf-8")
    locales = source / "locales"
    locales.mkdir()
    catalog = locales / "en.json"
    catalog.write_text('{"nodes":{}}\n', encoding="utf-8")

    first = compose._installed_pack_digest(manifest, module)
    catalog.write_text('{"nodes":{"test.echo":{"displayName":"Echo"}}}\n', encoding="utf-8")
    second = compose._installed_pack_digest(manifest, module)

    assert first != second


def test_installed_pack_digest_rejects_locale_symlinks(tmp_path: Path) -> None:
    from dinkster_registry import ArtifactError

    from dinkster import compose

    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "dinkster-pack.toml"
    manifest.write_text('[pack]\nname = "test"\nversion = "1.0.0"\n', encoding="utf-8")
    module = source / "test_nodes"
    module.mkdir()
    (module / "__init__.py").write_text("NODES = []\n", encoding="utf-8")
    locales = source / "locales"
    locales.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        (locales / "en.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ArtifactError, match="symlink"):
        compose._installed_pack_digest(manifest, module)


def test_installed_pack_digest_matches_bundled_artifact(tmp_path: Path) -> None:
    from dinkster import compose

    source = Path(__file__).parent.parent / "packages/dinkster-nodes-image"
    bundled = tmp_path / "dinkster_nodes_image_pack"
    bundled.mkdir()
    shutil.copy2(source / "dinkster-pack.toml", bundled / "dinkster-pack.toml")
    shutil.copytree(source / "src/dinkster_nodes_image", bundled / "dinkster_nodes_image")
    for filename in compose._PACK_ARTIFACT_SIDECARS:
        shutil.copy2(source / filename, bundled / filename)

    source_digest = compose._installed_pack_digest(
        source / "dinkster-pack.toml", source / "src/dinkster_nodes_image"
    )
    bundled_digest = compose._installed_pack_digest(bundled / "dinkster-pack.toml", None)

    assert source_digest == bundled_digest


def test_vision_pack_license_checkout_endings_do_not_change_digest(tmp_path: Path) -> None:
    from dinkster import compose

    source = Path(__file__).parent.parent / "packages/dinkster-nodes-vision"
    copied = shutil.copytree(source, tmp_path / source.name)
    manifest = copied / "dinkster_vision_hed_pack/dinkster-pack.toml"
    module = copied / "src/dinkster_nodes_vision/hed"
    expected = compose._installed_pack_digest(manifest, module)
    assert expected == ("blake3:0421374ac24c9b910a97fc5a5b2982c681b9785ddbeb415f27a08b934bfc1a48")
    license_file = manifest.parent / "MLSD_LICENSE"
    license_bytes = license_file.read_bytes().replace(b"\r\n", b"\n")
    license_file.write_bytes(license_bytes.replace(b"\n", b"\r\n"))

    assert compose._installed_pack_digest(manifest, module) == expected


def test_default_pack_refuses_version_outside_suite_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster import compose

    lock = compose._default_pack_lock()
    real_distribution = compose.importlib.metadata.distribution

    def distribution(name: str):
        installed = real_distribution(name)
        if name == "dinkster-nodes-foundation":
            return SimpleNamespace(version="9.0.0", locate_file=installed.locate_file)
        return installed

    monkeypatch.setattr("dinkster.compose.importlib.metadata.distribution", distribution)
    monkeypatch.setattr("dinkster.compose._default_pack_lock", lambda: lock)

    with pytest.raises(CompositionError, match="locks dinkster-nodes-foundation version 0.0.1"):
        default_pack_spec("dinkster-nodes-foundation")


def test_default_pack_refuses_bytes_outside_suite_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import compose

    lock = compose._default_pack_lock()
    source = Path(__file__).parent.parent / "packages/dinkster-nodes-foundation"
    copied = tmp_path / "source"
    copied.mkdir()
    shutil.copy2(source / "dinkster-pack.toml", copied / "dinkster-pack.toml")
    shutil.copytree(
        source / "src/dinkster_nodes_foundation", copied / "src/dinkster_nodes_foundation"
    )
    module = copied / "src/dinkster_nodes_foundation/__init__.py"
    module.write_text(
        module.read_text(encoding="utf-8") + "\nLOCK_DRIFT = True\n",
        encoding="utf-8",
    )
    distribution = SimpleNamespace(
        version="0.0.1",
        locate_file=lambda _path: tmp_path / "missing/dinkster-pack.toml",
    )

    monkeypatch.setattr(
        "dinkster.compose.importlib.metadata.distribution", lambda _name: distribution
    )
    monkeypatch.setattr("dinkster.compose._default_pack_lock", lambda: lock)
    monkeypatch.setattr(
        "dinkster.compose.importlib.util.find_spec",
        lambda _name: SimpleNamespace(origin=str(module)),
    )

    with pytest.raises(CompositionError, match="locks dinkster-nodes-foundation artifact"):
        default_pack_spec("dinkster-nodes-foundation")


def test_default_suite_refuses_malformed_managed_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import compose

    lock = tmp_path / "dinkster.lock"
    lock.write_text('{"format":"dinkster.lock/999","packs":[]}', encoding="utf-8")
    distribution = SimpleNamespace(locate_file=lambda _path: lock)
    compose._default_pack_lock.cache_clear()
    monkeypatch.setattr(
        "dinkster.compose.importlib.metadata.distribution", lambda _name: distribution
    )

    with pytest.raises(CompositionError, match="installed default suite lock is invalid"):
        compose._default_pack_lock()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ('{"format":"dinkster.lock/1","packs":[]}', "selects no packs"),
        (
            '{"format":"dinkster.lock/1","packs":[{'
            '"artifactDigest":"blake3:1111111111111111111111111111111111111111111111111111111111111111",'
            '"claims":["other"],"pack":"other","publisher":"dinkster",'
            '"source":"registry","version":"1.0.0"}]}',
            "selects unsupported packs",
        ),
        (
            '{"format":"dinkster.lock/1","packs":[{'
            '"artifactDigest":"blake3:1111111111111111111111111111111111111111111111111111111111111111",'
            '"claims":["dinkster"],"pack":"dinkster-nodes-generation","publisher":"other",'
            '"source":"registry","version":"1.0.0"}]}',
            "contains non-Dinkster publishers",
        ),
    ],
)
def test_default_suite_refuses_non_suite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
    message: str,
) -> None:
    from dinkster import compose

    lock = tmp_path / "dinkster.lock"
    lock.write_text(record, encoding="utf-8")
    distribution = SimpleNamespace(locate_file=lambda _path: lock)
    compose._default_pack_lock.cache_clear()
    monkeypatch.setattr(
        "dinkster.compose.importlib.metadata.distribution", lambda _name: distribution
    )

    with pytest.raises(CompositionError, match=message):
        compose._default_pack_lock()


def test_default_suite_package_matches_managed_lock() -> None:
    root = Path(__file__).parent.parent / "packages/dinkster-nodes-std"
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = json.loads((root / "dinkster.lock").read_text(encoding="utf-8"))
    locked_distributions = list(
        dict.fromkeys(entry["source"].removeprefix("python:") for entry in lock["packs"])
    )
    assert configuration["project"]["dependencies"] == locked_distributions
    wheel = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["force-include"] == {
        "dinkster.lock": "dinkster_nodes_std_suite/dinkster.lock",
    }


def test_compose_dev_pack_adds_scaffold_nodes() -> None:
    """The development manifest composes its complete node surface."""
    from dinkster_nodes_dev import PACK_NODES

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            expected = set(build_schemas(DEFAULT_NODES)) | set(build_schemas(PACK_NODES))
            assert set(composition.schemas) == expected
            engine = composition.make_engine(lambda event: None)
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
            result = await engine.run(graph, ["g"])
            assert "g" in result.outputs
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_conformance_proof_nodes_follow_explicit_dev_pack_composition() -> None:
    """Conformance probes appear only when the development pack is composed."""

    async def scenario() -> None:
        dev = await compose_serving([DEV_PACK_MANIFEST])
        try:
            assert "dev.conformance.preview" in dev.schemas
            assert "dev.conformance.cancellable" in dev.schemas
        finally:
            await dev.close()

        production = await compose_serving()
        try:
            assert not any(
                node_type.startswith("dev.conformance.") for node_type in production.schemas
            )
            assert not any(
                node_type.startswith("dev.conformance.") for node_type in production.node_packs
            )
        finally:
            await production.close()

        manifest_composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            assert "dev.image.gradient" in manifest_composition.schemas
            assert "dev.conformance.preview" in manifest_composition.schemas
        finally:
            await manifest_composition.close()

    asyncio.run(scenario())


def test_compose_dev_mode(caplog: pytest.LogCaptureFixture) -> None:
    """dev=True: engines explain cache misses, and boundary diagnostics
    render as one structured log line per crossing."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST], dev=True)
        try:
            events: list[object] = []
            engine = composition.make_engine(events.append)
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
            await engine.run(graph, ["g"])
            kinds = [getattr(e, "kind", "") for e in events]
            assert "cache_miss" in kinds
        finally:
            await composition.close()

    asyncio.run(scenario())

    from dinkster_workers import BoundaryDiagnostic, EdgeCost

    from dinkster.compose import log_boundary_diagnostic

    diagnostic = BoundaryDiagnostic(
        invocation_id="inv1",
        node_id="n",
        node_type="iso.echo",
        pack="isopack",
        inputs=(EdgeCost("text", "core.string", "inline", 12, 0.2, True, False),),
        outputs=(EdgeCost("out", "iso.blob", "shm", 4096, 1.5, False, False),),
        execute_ms=10.0,
        round_trip_ms=14.0,
    )
    with caplog.at_level("INFO", logger="dinkster.dev.boundary"):
        log_boundary_diagnostic(diagnostic)
    line = " ".join(r.getMessage() for r in caplog.records)
    assert "n (iso.echo, pack isopack)" in line
    assert "execute 10.0ms, boundary 4.0ms" in line
    assert "out=iso.blob 4096B shm 1.5ms FALLBACK-CODEC" in line


def test_composed_layered_cache_reuses_disk_then_promotes_to_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
        cache_dir = tmp_path / "execution-cache"

        async def compose():
            return await compose_serving(
                [DEV_PACK_MANIFEST],
                include_default_packs=False,
                dev=True,
                cache_mode="layered",
                cache_memory_entries=1,
                cache_dir=cache_dir,
                cache_disk_budget=1024**2,
            )

        first = await compose()
        try:
            first_result = await first.make_engine(lambda _event: None).run(graph, ["g"])
            assert first_result.executed == ("g",)
        finally:
            await first.close()

        restarted = await compose()
        try:
            events: list[EngineEvent] = []
            engine = restarted.make_engine(events.append)
            assert isinstance(engine.cache, LayeredCache)

            disk_result = await engine.run(graph, ["g"])
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_result.cached == ("g",)
            assert disk_hit.detail["cacheLayer"] == "disk"

            events.clear()
            memory_result = await engine.run(graph, ["g"])
            memory_hit = next(event for event in events if event.kind == "node_cached")
            assert memory_result.cached == ("g",)
            assert memory_hit.detail["cacheLayer"] == "memory"
        finally:
            await restarted.close()

    asyncio.run(scenario())


def test_full_free_clears_memory_execution_cache_but_preserves_disk(
    tmp_path: Path, isolated_full_free_pool: ResidentPool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            dev=True,
            cache_mode="layered",
            cache_memory_entries=1,
            cache_dir=tmp_path / "execution-cache",
            cache_disk_budget=1024**2,
        )
        try:
            await composer.add_pack(DEV_PACK_MANIFEST)
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
            first = await engine.run(graph, ["g"])
            assert first.executed == ("g",)
            events.clear()

            staged = composer.spawn_empty()
            await staged.add_pack(DEV_PACK_MANIFEST)
            old = composer.adopt(staged)
            await old.close()
            assert isinstance(engine.cache, LayeredCache)
            memory = engine.cache.layers[0]
            assert isinstance(memory, MemoryLRUCache)
            assert len(memory) == 1

            local = (await composer.full_free("maintenance-cache"))[0]
            assert local["status"] == "complete"
            consumers = cast("list[dict[str, object]]", local["consumers"])
            assert consumers[0] == {
                "consumer": "execution-cache",
                "status": "complete",
            }

            second = await engine.run(graph, ["g"])
            assert second.cached == ("g",)
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_hit.detail["cacheLayer"] == "disk"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_full_free_releases_tensors_held_by_memory_execution_cache(
    isolated_full_free_pool: ResidentPool,
) -> None:
    torch = pytest.importorskip("torch")

    async def scenario() -> None:
        composer = ServingComposer(dev=True, cache_mode="memory", cache_memory_entries=1)
        try:
            engine = composer.composition.make_engine(lambda _event: None)
            registry = TypeRegistry()
            registry.register("test.tensor", fingerprint=lambda _tensor: "schedule-output")

            async def cache_tensor() -> weakref.ReferenceType[object]:
                tensor = torch.ones(4)
                reference = weakref.ref(tensor)
                await engine.cache.put(
                    "schedule-output",
                    {"sigmas": registry.wrap("test.tensor", tensor)},
                )
                return reference

            tensor_reference = await cache_tensor()
            gc.collect()
            assert tensor_reference() is not None

            (local,) = await composer.full_free("maintenance-tensor-cache")
            assert local["status"] == "complete"
            gc.collect()
            assert tensor_reference() is None
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_compose_rejects_reserved_pack_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, name="core", presentation=False)
        with pytest.raises(CompositionError, match="reserved"):
            await compose_serving([manifest], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_rejects_duplicate_pack_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", presentation=False)
        second = write_iso_manifest(tmp_path / "b", presentation=False)
        with pytest.raises(CompositionError, match="duplicate pack name"):
            await compose_serving([first, second], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_rejects_namespace_collision(tmp_path: Path) -> None:
    """Two packs claiming one namespace is a hard startup error naming
    both owners - never a silent precedence pick. This subsumes the old
    pack-vs-pack node-type collision: two isolated packs cannot announce
    the same node type without their claims overlapping first, so the
    error now fires before the second worker even starts."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", name="isopack", presentation=False)
        second = write_iso_manifest(tmp_path / "b", name="isopack2", presentation=False)
        with pytest.raises(
            CompositionError,
            match="'iso' claimed by 'isopack2' overlaps 'iso' claimed by 'isopack'",
        ):
            await compose_serving([first, second], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_rejects_nested_namespace_claims(tmp_path: Path) -> None:
    """Overlap is atom-bounded across separator spellings: iso-extra IS
    iso.extra, which nests inside iso - two owners could otherwise both
    cover an iso.extra.* node type."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", presentation=False)
        second = write_iso_manifest(
            tmp_path / "b",
            name="isopack2",
            presentation=False,
            namespaces=("iso-extra",),
        )
        with pytest.raises(CompositionError, match="overlaps 'iso'"):
            await compose_serving([first, second], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_rejects_separator_equivalent_pack_names(tmp_path: Path) -> None:
    """iso_pack and iso-pack are ONE pack identity (separators are an
    equivalence class) - the drift that made one ComfyUI pack look like
    two packages must fail loudly here."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", name="iso-pack", presentation=False)
        second = write_iso_manifest(tmp_path / "b", name="iso_pack", presentation=False)
        with pytest.raises(CompositionError, match="one identity"):
            await compose_serving([first, second], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_rejects_uncovered_node_type(tmp_path: Path) -> None:
    """A worker announcing a node type outside its manifest's claims is a
    startup error: the loading record, not the schema, is the authority
    for what a pack may provide. Omitting namespaces claims the pack name,
    which does not cover iso.*."""

    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, presentation=False, namespaces=None)
        with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
            await compose_serving([manifest], worker_env=WORKER_ENV)

    asyncio.run(scenario())


def test_compose_gates_reserved_claims_on_host_trust(tmp_path: Path) -> None:
    """A reserved-root claim composes only when the spec vouches for the
    pack (the local analog of the registry's grant table): untrusted is a
    startup error before the worker even launches, trusted composes."""
    from dinkster.compose import PackSpec

    async def scenario() -> None:
        manifest = write_iso_manifest(
            tmp_path, presentation=False, namespaces=("iso", "comfy.extras")
        )
        with pytest.raises(CompositionError, match="explicit host trust"):
            await compose_serving([manifest], worker_env=WORKER_ENV)

        composition = await compose_serving(
            [PackSpec(manifest=manifest, env=WORKER_ENV, trust_reserved=True)]
        )
        try:
            assert composition.node_packs["iso.chatty"] == "isopack"
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_composed_server_end_to_end(tmp_path: Path) -> None:
    """One server: std nodes in-process, the pack isolated, both executing
    through the same queue; /api/nodes attributes each side truthfully and
    carries the manifest's declared presentation."""

    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        composition = await compose_serving([manifest], worker_env=WORKER_ENV)
        published_schemas = {
            node_type: schema
            for node_type, schema in composition.schemas.items()
            if node_type.startswith("iso.") or node_type == "std.math.add_ints"
        }
        published_packs = {
            pack_id: replace(info, comfy_aliases=None, comfy_groups=None)
            for pack_id, info in composition.packs.items()
            if pack_id in {"isopack", "dinkster-nodes-foundation"}
        }
        app = create_app(
            composition.make_engine,
            published_schemas,
            packs=published_packs,
            node_packs={
                node_type: pack_id
                for node_type, pack_id in composition.node_packs.items()
                if node_type in published_schemas
            },
            choices=composition.choices,
            lazy_choices=composition.lazy_choices,
        )

        async def close_composition(_: object) -> None:
            await composition.close()

        app.on_cleanup.append(close_composition)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Provenance: manifest presentation reaches the packs table,
            # pack nodes and standard nodes both attribute to their packs. A
            # plain dev pack exposes where its bytes live but no pin -
            # version/artifactDigest stay omitted (unpinned), never null.
            data = await (await client.get("/api/nodes")).json()
            assert data["packs"]["isopack"] == {
                "displayName": "Iso Pack",
                "abbr": "ISO",
                "mark": "\U0001f9ea",
                "color": "#336699",
                "source": f"local:{Path(manifest).parent.resolve()}",
            }
            assert data["nodes"]["iso.chatty"]["pack"] == "isopack"
            assert data["nodes"]["std.math.add_ints"]["pack"] == "dinkster-nodes-foundation"
            blob_size = next(
                item
                for item in data["nodes"]["iso.blob_out"]["interface"]
                if item["role"] == "input" and item["id"] == "size"
            )
            assert blob_size["widget"] == {
                "type": "isopack.size",
                "min": 1,
                "max": 12,
                "unit": "bytes",
            }

            # The in-process media-io pack's lazy device route serves per
            # fetch: no capture provider configured means an empty list,
            # not a missing route.
            devices = await client.get("/api/choices/dinkster.devices.audio_inputs")
            assert devices.status == 200, await devices.text()
            assert await devices.json() == []

            # One graph spanning both workers: the std add runs
            # in-process, iso.chatty in the pack process, one job.
            graph = Graph(
                nodes={
                    "g": GraphNode("std.math.add_ints", {"a": 2, "b": 3}),
                    "c": GraphNode("iso.chatty", {"value": "hi"}),
                    "b": GraphNode("iso.blob_out", {"size": 3}),
                }
            )
            body = {
                "clientId": "c1",
                "jobId": "j1",
                "graph": graph_to_wire(graph),
                "targets": ["g", "c", "b"],
            }
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202, await resp.text()
            status: dict[str, object] = {}
            for _ in range(200):
                status = await (await client.get("/api/jobs/c1/j1")).json()
                if status["state"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            assert status.get("state") == "completed", status

            value_url = "/api/values?clientId=c1&jobId=j1&nodeId=b&outputId=blob"
            discovery = await client.get(value_url)
            assert discovery.status == 200, await discovery.text()
            value_data = await discovery.json()
            assert value_data["descriptor"]["typeId"] == "iso.blob"
            assert value_data["renditions"] == [
                {
                    "kind": "summary",
                    "mime": "text/plain",
                    "default": True,
                    "cacheKey": "summary/1",
                    "version": "1",
                    "parameters": ["prefix"],
                    "defaults": {"prefix": "blob"},
                    "limits": {"prefixLength": 32},
                }
            ]
            rendered = await client.get(value_url + "&rendition=summary/1&prefix=pack")
            assert rendered.status == 200, await rendered.text()
            assert rendered.content_type == "text/plain"
            assert await rendered.read() == b"pack:3:xxx"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_serving_composer_replaces_and_retracts_pack_renditions(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            manifest = write_iso_manifest(tmp_path)
            await composer.add_pack(manifest)
            registry = composer.composition._registry
            original = registry.renditions_of("iso.blob")
            assert len(original) == 1
            assert original[0].kind == "summary"
            assert original[0].relay_owner == "isopack"

            await composer.reload_pack("isopack")
            reloaded = registry.renditions_of("iso.blob")
            assert len(reloaded) == 1
            assert reloaded[0].relay_owner == "isopack"
            assert reloaded[0].render_async is not original[0].render_async

            await composer.remove_pack("isopack")
            assert registry.renditions_of("iso.blob") == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
@pytest.mark.parametrize("replica_cuda_indices", [(), (0, 1)], ids=["lazy", "replica-pool"])
def test_cold_catalog_pack_renditions_relay_through_composed_registry(
    tmp_path: Path, replica_cuda_indices: tuple[int, ...]
) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        environment = {**os.environ, **WORKER_ENV}
        assert prepare_catalog(manifest, environment=environment).ok
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    manifest,
                    require_catalog=True,
                    replica_cuda_indices=replica_cuda_indices,
                )
            )
            worker = composer._records["isopack"].worker
            assert worker.cold

            registry = composer.composition._registry
            spec = registry.renditions_of("iso.blob")[0]
            metadata = {"size": 3}
            assert await registry.rendition_mime(spec, metadata) == "text/plain"
            mime, parameters = await registry.resolve_rendition(spec, metadata, {"prefix": "pack"})
            assert (mime, parameters) == ("text/plain", {"prefix": "pack"})
            value = Value(
                type_id="iso.blob",
                fingerprint="iso-blob-test",
                meta=ValueMeta(metadata),
                payload=EncodedPayload(
                    "iso.blob",
                    default_encode({"n": 3, "data": "xxx"}),
                    None,
                ),
            )
            rendition = await registry.render_async(
                value,
                "summary",
                parameters,
            )
            assert (rendition.mime, rendition.data) == ("text/plain", b"pack:3:xxx")
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_add_pack_applies_host_types_at_commit(tmp_path: Path) -> None:
    """PackSpec.host_types registers host-side value types when the pack
    composes: applied exactly at the commit point (a failing pack never
    fires it), against the composition's own registry, tolerating the
    idempotent re-registration the hook contract demands."""
    from dinkster.compose import PackSpec, ServingComposer

    calls: list[int] = []

    def host_types(registry: object) -> None:
        from dinkster_values import TypeRegistry

        assert isinstance(registry, TypeRegistry)
        calls.append(1)
        if "hosttest.blob" not in registry:
            registry.register("hosttest.blob")

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            # A pack that fails validation never reaches the commit point.
            bad = PackSpec(
                manifest=write_iso_manifest(tmp_path / "bad", namespaces=("std",)),
                host_types=host_types,
            )
            with pytest.raises(CompositionError):
                await composer.add_pack(bad)
            assert calls == []

            good = PackSpec(
                manifest=write_iso_manifest(tmp_path / "good", name="hostpack"),
                host_types=host_types,
            )
            await composer.add_pack(good)
            assert calls == [1]
            assert "hosttest.blob" in composer.composition._registry
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_comfy_compat_specs_carry_host_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both compat specs (core and legacy quarantine) install the comfy
    host-type hook, so whichever composes first gives the torchless engine
    the comfy.IMAGE decode + PNG rendition."""
    from dinkster import comfy_compose
    from dinkster.comfy_compose import comfy_compat_specs, register_comfy_host_types

    (tmp_path / "legacy-pack").mkdir()
    monkeypatch.setattr(comfy_compose, "_probe_comfy_blake3", lambda _interpreter: None)
    specs = comfy_compat_specs(
        tmp_path,
        legacy_packs=[tmp_path / "legacy-pack"],
        aimdo="on",
        memory_budgets={"vram:cuda:0": 20_000, "ram": 40_000},
        reserve_vram=128,
    )
    assert len(specs) == 3
    assert specs[0].host_types is None
    assert all(spec.host_types is register_comfy_host_types for spec in specs[1:])
    assert all(spec.aimdo == "on" for spec in specs[1:])
    assert all(spec.vram_budgets == {"vram:cuda:0": 20_000} for spec in specs[1:])
    assert all(spec.reserve_vram == 128 for spec in specs[1:])


def test_pack_spec_validates_aimdo_mode(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec

    manifest = write_iso_manifest(tmp_path)
    assert PackSpec(manifest, aimdo="off").aimdo == "off"
    assert PackSpec(manifest, aimdo="auto").aimdo == "auto"
    assert PackSpec(manifest, aimdo="on").aimdo == "on"
    with pytest.raises(ValueError, match="PackSpec aimdo"):
        PackSpec(manifest, aimdo="sticky")
    with pytest.raises(ValueError, match="PackSpec aimdo"):
        PackSpec(manifest, aimdo="sometimes")
    spec = PackSpec(
        manifest,
        vram_budgets={"vram:cuda:0": 20_000},
        reserve_vram=128,
    )
    assert spec.vram_budgets == {"vram:cuda:0": 20_000}
    with pytest.raises(TypeError):
        spec.vram_budgets["vram:cuda:1"] = 10_000  # type: ignore[index]
    with pytest.raises(ValueError, match="vram:cuda:N"):
        PackSpec(manifest, vram_budgets={"ram": 1})
    with pytest.raises(ValueError, match="non-negative"):
        PackSpec(manifest, vram_budgets={"vram:cuda:0": -1})
    with pytest.raises(ValueError, match="non-negative"):
        PackSpec(manifest, reserve_vram=-1)
    assert PackSpec(manifest, replica_cuda_indices=(2, 0, 1)).replica_cuda_indices == (
        2,
        0,
        1,
    )
    with pytest.raises(ValueError, match="at least two"):
        PackSpec(manifest, replica_cuda_indices=(0,))
    with pytest.raises(ValueError, match="unique"):
        PackSpec(manifest, replica_cuda_indices=(0, 0))
    with pytest.raises(ValueError, match="non-negative ints"):
        PackSpec(manifest, replica_cuda_indices=(0, -1))
    with pytest.raises(ValueError, match="execution_config"):
        PackSpec(manifest, execution_config={"": "value"})


def test_pack_spec_validates_single_job_mode(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec

    manifest = write_iso_manifest(tmp_path)
    for mode in ("auto", "guidance", "sequence", "window"):
        assert PackSpec(manifest, single_job_mode=mode).single_job_mode == mode
    with pytest.raises(ValueError, match="single_job_mode is invalid"):
        PackSpec(manifest, single_job_mode="model")


def test_pack_spec_comfy_args_are_frozen_validated_and_allow_unknowns(
    tmp_path: Path,
) -> None:
    from dataclasses import FrozenInstanceError

    from dinkster_server import comfy_dtype_args

    from dinkster.compose import PackSpec

    manifest = write_iso_manifest(tmp_path)
    spec = PackSpec(manifest, comfy_args=("--future-comfy-flag", "value"))
    assert spec.comfy_args == ("--future-comfy-flag", "value")
    with pytest.raises(FrozenInstanceError):
        spec.comfy_args = ()  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="tuple of strings"):
        PackSpec(manifest, comfy_args=["--preview-size"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON array of strings"):
        PackSpec(manifest, comfy_args=("--preview-size", 256))  # type: ignore[arg-type]
    dtype_args = comfy_dtype_args({"diffusion": "bfloat16", "textEncoder": "auto", "vae": "auto"})
    assert PackSpec(manifest, comfy_args=dtype_args).comfy_args == ("--bf16-unet",)


@pytest.mark.parametrize(
    ("argument", "flag", "owner"),
    [
        ("--port", "--port", "Dinkster server"),
        ("--enable-cors-header=*", "--enable-cors-header", "Dinkster server"),
        ("--front-end-root=/tmp", "--front-end-root", "Dinkster server"),
        ("--reserve-vram", "--reserve-vram", "Dinkster memory/aimdo policy"),
        ("--gpu-only=true", "--gpu-only", "Dinkster memory/aimdo policy"),
        ("--novram", "--novram", "Dinkster memory/aimdo policy"),
        ("--vram-headroom=1", "--vram-headroom", "Dinkster memory/aimdo policy"),
        ("--bf16-unet", "--bf16-unet", "Dinkster dtype policy"),
        ("--por=8188", "--por", "Dinkster server"),
        ("--gpu-o", "--gpu-o", "Dinkster memory/aimdo policy"),
    ],
)
def test_pack_spec_comfy_args_deny_dinkster_owned_flags(
    tmp_path: Path, argument: str, flag: str, owner: str
) -> None:
    from dinkster_server import ComfyArgumentError

    from dinkster.compose import PackSpec

    with pytest.raises(ComfyArgumentError) as raised:
        PackSpec(write_iso_manifest(tmp_path), comfy_args=(argument,))
    assert raised.value.flag == flag
    assert raised.value.owner == owner
    assert flag in str(raised.value)
    assert owner in str(raised.value)


def test_serving_composer_incremental(tmp_path: Path) -> None:
    """The diagnostic host starts empty and grows through ordinary pack deltas."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition
            assert composition.schemas == {}
            assert composition.packs == {}
            engine = composition.make_engine(lambda event: None)

            app = create_app(composition.make_engine, composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert await (await client.get("/api/health")).json() == {"ok": True}
                nodes = await (await client.get("/api/nodes")).json()
                assert nodes["nodes"] == {}
            finally:
                await client.close()

            standard_deltas = [
                await composer.add_pack(standard) for standard in default_pack_specs()
            ]
            assert set().union(*(delta.schemas for delta in standard_deltas)) == set(
                build_schemas(DEFAULT_NODES)
            )
            for standard_delta in standard_deltas:
                engine.announce_schemas(standard_delta.schemas)
            result = await engine.run(
                Graph(nodes={"g": GraphNode("std.math.add_ints", {"a": 2, "b": 3})}),
                ["g"],
            )
            assert "g" in result.outputs

            delta = await composer.add_pack(write_iso_manifest(tmp_path))
            assert delta.pack == "isopack"
            assert "iso.chatty" in delta.schemas
            assert delta.node_packs["iso.chatty"] == "isopack"
            assert delta.packs["isopack"].display_name == "Iso Pack"
            # Merged into the shared composition too (the all-at-once path).
            assert composition.schemas["iso.chatty"] is delta.schemas["iso.chatty"]
            assert composition.node_packs["iso.chatty"] == "isopack"

            # The engine made BEFORE the pack composed executes it after
            # announcement - no rebuild, no restart.
            engine.announce_schemas(delta.schemas)
            result = await engine.run(
                Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hi"})}),
                ["c"],
            )
            assert result.outputs["c"]["value"].resolve() == "HI"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_broken_default_pack_leaves_zero_node_host_and_registry_healthy(tmp_path: Path) -> None:
    module = tmp_path / "broken_media.py"
    module.write_text(
        "def register_types(registry):\n"
        "    registry.register('broken.partial')\n"
        "    raise RuntimeError('broken media types')\n"
        "NODES = []\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "dinkster-nodes-media-io"\nnamespaces = ["dinkster"]\n'
        '[pack.entry]\nnodes = "broken_media:NODES"\n'
        'types = "broken_media:register_types"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            with pytest.raises(RuntimeError, match="broken media types"):
                await composer.add_pack(PackSpec(manifest, in_process=True, trust_reserved=True))
            assert composer.composition.schemas == {}
            assert composer.composition.packs == {}
            assert "broken.partial" not in composer.composition._registry

            app = create_app(composer.composition.make_engine, composer.composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert await (await client.get("/api/health")).json() == {"ok": True}
                nodes = await (await client.get("/api/nodes")).json()
                assert nodes["nodes"] == {}
            finally:
                await client.close()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_generation_staging_rebuilds_pack_type_registrations(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        module = tmp_path / "generation_types_nodes.py"
        module.write_text(
            "def register_types(registry):\n"
            "    registry.register('generation.ephemeral')\n"
            "NODES = []\n",
            encoding="utf-8",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "generation-types"\nnamespaces = ["generation"]\n'
            '[pack.entry]\nnodes = "generation_types_nodes:NODES"\n'
            'types = "generation_types_nodes:register_types"\n',
            encoding="utf-8",
        )
        await composer.add_pack(PackSpec(manifest, in_process=True))
        assert "generation.ephemeral" in composer.composition._registry

        staged = composer.spawn_empty()
        old = composer.adopt(staged)
        try:
            assert "generation.ephemeral" not in composer.composition._registry
        finally:
            await old.close()
            await composer.close()

    asyncio.run(scenario())


def test_serving_composer_registers_and_deregisters_armed_headroom_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster.compose as compose_module
    from dinkster.compose import PackSpec, ServingComposer

    class FakeMirror:
        def __init__(self) -> None:
            self.registered: list[object] = []
            self.deregistered: list[object] = []

        async def register(self, session: object, device_map: object) -> None:
            del device_map
            self.registered.append(session)

        def deregister(self, session: object) -> None:
            self.deregistered.append(session)

    mirror = FakeMirror()
    bases: list[int] = []

    def fake_mirror(_governor: object, *, base_bytes: int) -> FakeMirror:
        bases.append(base_bytes)
        return mirror

    monkeypatch.setattr(
        compose_module,
        "HeadroomMirror",
        fake_mirror,
    )

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env=WORKER_ENV,
            governor=MemoryGovernor(),
            headroom_base=123,
        )
        try:
            assert bases == [123]
            await composer.add_pack(PackSpec(write_iso_manifest(tmp_path), aimdo="on"))
            assert len(mirror.registered) == 1
            await composer.remove_pack("isopack")
            assert mirror.deregistered == mirror.registered
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_serving_composer_maintains_asset_catalog(tmp_path: Path) -> None:
    """[[pack.assets]] declarations ride composition onto the live asset
    catalog: add_pack registers the pack's artifact root and needs (and
    the packs-table wire carries the descriptors), reload swaps them for
    the new manifest's declarations, remove retracts everything - the
    SAME catalog object throughout, since ServerLibrary holds the
    reference for the process lifetime."""
    from dinkster_assets import PackagedSource, digest_bytes

    from dinkster.compose import ServingComposer

    model_bytes = b"iso aux model" * 64
    model_digest = digest_bytes(model_bytes)

    def write_manifest_with_assets(directory: Path) -> Path:
        manifest = write_iso_manifest(directory)
        (directory / "assets").mkdir(exist_ok=True)
        (directory / "assets" / "model.bin").write_bytes(model_bytes)
        manifest.write_text(
            manifest.read_text()
            + "\n[[pack.assets]]\n"
            + f'id = "aux-model"\nname = "Aux Model"\ndigest = "{model_digest}"\n'
            + 'kind = "model/auxiliary"\nfile = "assets/model.bin"\n'
            + 'urls = ["https://hub.example/aux.bin"]\nnodes = ["iso.chatty"]\n'
        )
        return manifest

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            catalog = composer.composition.asset_catalog
            pack_dir = tmp_path / "isopack"
            delta = await composer.add_pack(write_manifest_with_assets(pack_dir))

            # Descriptors ride the packs table; bytes never do.
            wire = delta.packs["isopack"].to_wire()
            assets = wire["assets"]
            assert isinstance(assets, list) and len(assets) == 1
            assert assets[0]["id"] == "aux-model"
            assert assets[0]["digest"] == model_digest
            assert assets[0]["nodes"] == ["iso.chatty"]

            # The catalog knows the root and the need, by digest and by
            # requiring node type.
            assert catalog.pack_roots() == {"isopack": pack_dir}
            need = catalog.need_for(model_digest)
            assert need is not None
            assert PackagedSource(pack="isopack", path="assets/model.bin") in (need.sources)
            assert set(catalog.needs_for_nodes({"iso.chatty"})) == {model_digest}

            # Reload against a manifest WITHOUT the declaration: the
            # catalog swaps to the new surface - no stale need survives.
            write_iso_manifest(pack_dir)
            await composer.reload_pack("isopack")
            assert catalog.need_for(model_digest) is None
            assert catalog.needs_for_nodes({"iso.chatty"}) == {}

            # And back: reload with the declaration restores it.
            write_manifest_with_assets(pack_dir)
            await composer.reload_pack("isopack")
            assert catalog.need_for(model_digest) is not None

            # Removal retracts declarations and root alike.
            await composer.remove_pack("isopack")
            assert catalog.need_for(model_digest) is None
            assert catalog.pack_roots() == {}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_serving_composer_failed_add_pack_reaps_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An add_pack that fails AFTER its worker started (node type outside
    the declared namespaces) leaves the worker registered for cleanup:
    close() reaps every started worker, the earlier pack's included."""
    from dinkster_workers import IsolatedWorker

    from dinkster.compose import ServingComposer

    started: list[object] = []
    closed: list[object] = []

    class Recording(IsolatedWorker):
        async def start(self) -> None:
            await super().start()
            started.append(self)

        async def close(self) -> None:
            closed.append(self)
            await super().close()

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(write_iso_manifest(tmp_path / "a"))
            bad = write_iso_manifest(tmp_path / "b", name="isopack2", namespaces=None)
            with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
                await composer.add_pack(bad)
        finally:
            await composer.close()
        assert len(started) == 2  # the bad pack failed only after its hello
        assert set(closed) == set(started)

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
def test_serving_composer_starts_ordered_cuda_replica_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import IsolatedWorker

    from dinkster.compose import PackSpec, ServingComposer, _ReplicaWorkerPool

    started: list[IsolatedWorker] = []

    class Recording(IsolatedWorker):
        async def start(self) -> None:
            await super().start()
            started.append(self)

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    write_iso_manifest(tmp_path),
                    replica_cuda_indices=(2, 0, 1),
                )
            )
            pool = composer._records["isopack"].worker
            assert isinstance(pool, _ReplicaWorkerPool)
            assert tuple(lane.cuda_index for lane in pool.lanes) == (2, 0, 1)
            workers = [lane.worker for lane in pool.lanes]
            assert all(worker in started for worker in workers)
            assert [worker._extra_env["CUDA_VISIBLE_DEVICES"] for worker in workers] == [
                "2",
                "0",
                "1",
            ]
            assert all(worker._device_map is not None for worker in workers)
            assert [worker._device_map.mapping for worker in workers if worker._device_map] == [
                {"cuda:0": "cuda:2"},
                {"cuda:0": "cuda:0"},
                {"cuda:0": "cuda:1"},
            ]
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
def test_serving_composer_sets_pure_ulysses_sequence_geometry(tmp_path: Path) -> None:
    from dinkster_values import TypeRegistry

    from dinkster.compose import PackSpec, ServingComposer, _SingleJobWorkerPool

    composer = ServingComposer(worker_env=WORKER_ENV)
    manifest_path = write_iso_manifest(tmp_path)
    pool = composer._isolated_worker(  # pyright: ignore[reportPrivateUsage]
        PackSpec(
            manifest_path,
            single_job_cuda_indices=(2, 0),
            single_job_mode="sequence",
        ),
        load_manifest(manifest_path),
        TypeRegistry(),
    )
    assert isinstance(pool, _SingleJobWorkerPool)
    for rank, lane in enumerate(pool.lanes):
        environment = lane.worker._extra_env  # pyright: ignore[reportPrivateUsage]
        assert environment["DINKSTER_SINGLE_JOB_RANK"] == str(rank)
        assert environment["DINKSTER_SINGLE_JOB_WORLD_SIZE"] == "2"
        assert environment["DINKSTER_SINGLE_JOB_MULTI_GPU_MODE"] == "sequence"
        assert environment["DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES"] == "2"
        assert environment["DINKSTER_SINGLE_JOB_SEQUENCE_RING"] == "1"
        assert environment["DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE"] == "1"
    asyncio.run(pool.close())


def test_serving_composer_failed_add_pack_merges_nothing(tmp_path: Path) -> None:
    """add_pack is atomic: a pack failing composition AFTER its pack-table
    entry would have merged (node type outside its declared namespaces -
    the last validation) leaves the composition untouched - no schemas, no
    pack entry, no name or claim held. A continue-on-failure host serves
    survivors from an unpolluted table, and a LATER pack may reuse the
    failed pack's name and namespace."""
    from dinkster.compose import ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition
            bad = write_iso_manifest(tmp_path / "a", presentation=False, namespaces=None)
            with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
                await composer.add_pack(bad)
            assert "isopack" not in composition.packs
            assert not any(t.startswith("iso.") for t in composition.schemas)

            delta = await composer.add_pack(write_iso_manifest(tmp_path / "b"))
            assert delta.pack == "isopack"
            assert composition.packs["isopack"].display_name == "Iso Pack"
            assert "iso.chatty" in composition.schemas
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_failed_start_reaps_started_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A composition error after some packs started must not orphan their
    processes: every started worker is closed before the error escapes."""
    from dinkster_workers import IsolatedWorker

    started: list[object] = []
    closed: list[object] = []

    class Recording(IsolatedWorker):
        async def start(self) -> None:
            await super().start()
            started.append(self)

        async def close(self) -> None:
            closed.append(self)
            await super().close()

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        good = write_iso_manifest(tmp_path / "a", presentation=False)
        bad = write_iso_manifest(
            tmp_path / "b",
            name="zzbad",
            presentation=False,
            namespaces=("zzbad",),
        )
        bad.write_text(bad.read_text().replace("isopack_nodes:NODES", "missing_nodes:NODES"))
        with pytest.raises(RuntimeError, match="failed to start"):
            await compose_serving([good, bad], worker_env=WORKER_ENV)
        assert len(started) == 1
        assert started[0] in closed

    asyncio.run(scenario())


def test_group_contract_conflict_refuses_before_launch(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec

    async def scenario() -> None:
        alpha = write_iso_manifest(tmp_path / "a", name="alpha", presentation=False)
        beta = write_iso_manifest(tmp_path / "b", name="beta", presentation=False)
        manifests = (alpha, beta)
        specs = (
            PackSpec(alpha, env=WORKER_ENV, worker_group="models", group_manifests=manifests),
            PackSpec(
                beta,
                env={**WORKER_ENV, "EXTRA": "different"},
                worker_group="models",
                group_manifests=manifests,
            ),
        )
        with pytest.raises(CompositionError, match="incompatible launch contracts"):
            await compose_serving(specs)

    asyncio.run(scenario())


def test_pack_spec_custom_attribution(tmp_path: Path) -> None:
    """A spec can split one worker's nodes across several provenance
    entries (the compat-worker shape: one process, many pack chips)."""
    from dinkster_server import PackInfo

    from dinkster.compose import PackSpec

    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, presentation=False)
        table = {
            "iso-a": PackInfo(display_name="Iso A"),
            "iso-rest": PackInfo(display_name="Iso Rest"),
        }
        spec = PackSpec(
            manifest=manifest,
            env=WORKER_ENV,
            packs=table,
            attribute=lambda nt: "iso-a" if nt == "iso.chatty" else "iso-rest",
        )
        composition = await compose_serving([spec], include_default_packs=False)
        try:
            assert composition.packs == table
            assert composition.node_packs["iso.chatty"] == "iso-a"
            assert composition.node_packs["iso.sleepy"] == "iso-rest"
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_pack_spec_attribution_outside_table_fails(tmp_path: Path) -> None:
    from dinkster_server import PackInfo

    from dinkster.compose import PackSpec

    async def scenario() -> None:
        spec = PackSpec(
            manifest=write_iso_manifest(tmp_path, presentation=False),
            env=WORKER_ENV,
            packs={"only": PackInfo(display_name="Only")},
            attribute=lambda nt: "elsewhere",
        )
        with pytest.raises(CompositionError, match="not among this spec's pack"):
            await compose_serving([spec])

    asyncio.run(scenario())


def test_pack_table_entries_merge_only_when_identical(tmp_path: Path) -> None:
    """Two specs may contribute the same table entry (the shared 'comfy'
    chip) only if the infos agree; a conflicting redefinition fails."""
    from dinkster_server import PackInfo

    from dinkster.compose import PackSpec

    async def scenario() -> None:
        first = PackSpec(
            manifest=write_iso_manifest(tmp_path / "a", name="pack-a", presentation=False),
            env=WORKER_ENV,
            packs={
                "shared": PackInfo(display_name="Shared"),
                "pack-a": PackInfo(display_name="A"),
            },
            attribute=lambda nt: "pack-a",
        )
        # Different manifest name, CONFLICTING "shared" entry.
        second_manifest = tmp_path / "b" / "dinkster-pack.toml"
        (tmp_path / "b").mkdir()
        second_manifest.write_text(
            '[pack]\nname = "pack-b"\n\n[pack.entry]\n'
            'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
        )
        second = PackSpec(
            manifest=second_manifest,
            env=WORKER_ENV,
            packs={
                "shared": PackInfo(display_name="Shared, but different"),
                "pack-b": PackInfo(display_name="B"),
            },
            attribute=lambda nt: "pack-b",
        )
        with pytest.raises(CompositionError, match="different info"):
            await compose_serving([first, second])

    asyncio.run(scenario())


def test_pack_table_rejects_comfy_alias_collisions_without_mutating() -> None:
    from test_comfy_alias_registry import registry

    packs = {"native-a": PackInfo("Native A", comfy_aliases=registry())}
    with pytest.raises(CompositionError, match="collide on comfy alias record id"):
        _merge_pack_entry(
            packs,
            "native-b",
            PackInfo("Native B", comfy_aliases=registry()),
            "test",
        )
    assert set(packs) == {"native-a"}


def test_pack_table_rejects_comfy_group_collisions_without_mutating() -> None:
    from test_comfy_group_registry import registry

    packs = {"native-a": PackInfo("Native A", comfy_groups=registry())}
    with pytest.raises(CompositionError, match="collide on comfy group record id"):
        _merge_pack_entry(
            packs,
            "native-b",
            PackInfo("Native B", comfy_groups=registry()),
            "test",
        )
    assert set(packs) == {"native-a"}


def test_comfy_compat_specs_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure spec construction: no ComfyUI needed. The core worker
    attributes everything to 'comfy'; legacy nodes attribute per pack by
    the same namespace rule legacy.py applies, longest prefix first."""
    from dinkster import comfy_compose
    from dinkster.comfy_compose import comfy_compat_specs

    root = tmp_path / "ComfyUI"
    root.mkdir()
    (tmp_path / "packs" / "rgthree").mkdir(parents=True)
    single = tmp_path / "packs" / "solo.py"
    single.write_text("")
    # A pack that declares its own badge via a presentation-only manifest
    # (no [pack.entry] - not a loadable Dinkster pack, just a badge).
    themed = tmp_path / "packs" / "themed"
    themed.mkdir()
    from dinkster_assets import PackagedSource, digest_bytes
    from icon_bytes import png_bytes

    (themed / "badge.png").write_bytes(png_bytes())
    (themed / "starter.json").write_text('{"graphs": {}}')
    aux_bytes = b"legacy aux model" * 8
    aux_digest = digest_bytes(aux_bytes)
    (themed / "models").mkdir()
    (themed / "models" / "aux.bin").write_bytes(aux_bytes)
    (themed / "dinkster-pack.toml").write_text(
        "[pack]\n[pack.presentation]\n"
        'display_name = "Themed Pack"\nabbr = "TH"\ncolor = "#aa3355"\n'
        'icon = "badge.png"\n'
        "[[pack.blueprints]]\n"
        'id = "starter"\nname = "Starter"\nfile = "starter.json"\n'
        "[[pack.assets]]\n"
        f'id = "aux"\nname = "Aux Model"\ndigest = "{aux_digest}"\n'
        'file = "models/aux.bin"\nnodes = ["comfy.themed.AuxNode"]\n'
        "[[pack.templates]]\n"
        'id = "quickstart"\nname = "Quickstart"\nfile = "starter.json"\n'
        'assets = ["aux"]\n'
    )
    # A pack whose manifest is broken TOML: advisory means it composes
    # anyway, with the shared legacy default.
    broken = tmp_path / "packs" / "broken"
    broken.mkdir()
    (broken / "dinkster-pack.toml").write_text("[pack\n")
    monkeypatch.setattr(comfy_compose, "_probe_comfy_blake3", lambda _interpreter: None)

    specs = comfy_compat_specs(
        root,
        python=sys.executable,
        legacy_packs=[tmp_path / "packs" / "rgthree", single, themed, broken],
        comfy_nodes=["EmptyLatentImage"],
        asset_vault=tmp_path / "library" / "vault",
        mounts_snapshot=tmp_path / "library" / "mounts.json",
    )
    assert len(specs) == 3
    generation, core, legacy = specs
    assert generation.in_process is True
    assert generation.packs is not None and set(generation.packs) == {"dinkster-nodes-generation"}
    assert core.env["DINKSTER_COMFYUI_ROOT"] == str(root)
    assert core.env["DINKSTER_COMFY_NODES"] == "EmptyLatentImage"
    assert core.env["DINKSTER_ASSET_VAULT"] == str(tmp_path / "library" / "vault")
    assert core.env["DINKSTER_MOUNTS_SNAPSHOT"] == str(tmp_path / "library" / "mounts.json")
    assert core.python == sys.executable
    assert core.packs is not None and set(core.packs) == {"comfy"}
    assert core.attribute is not None
    assert core.attribute("comfy.EmptyLatentImage") == "comfy"
    assert core.attribute("dinkster.ksampler") == "comfy"
    # The compat layer's own declared chip: distinguishable from legacy
    # packs (shared color, no puzzle mark).
    assert core.packs["comfy"].display_name == "ComfyUI Compat"
    assert core.packs["comfy"].abbr == "C1"
    assert core.packs["comfy"].color

    multi_gpu = comfy_compat_specs(
        root,
        python=sys.executable,
        multi_device_cuda_indices=(1, 0),
    )[1]
    assert multi_gpu.replica_cuda_indices == (1, 0)

    assert legacy.packs is not None
    assert set(legacy.packs) == {
        "comfy",
        "comfy.rgthree",
        "comfy.solo",
        "comfy.themed",
        "comfy.broken",
    }
    assert "rgthree" in str(legacy.env["DINKSTER_LEGACY_PACKS"])
    assert legacy.env["DINKSTER_ASSET_VAULT"] == str(tmp_path / "library" / "vault")
    assert legacy.env["DINKSTER_MOUNTS_SNAPSHOT"] == str(tmp_path / "library" / "mounts.json")
    assert legacy.attribute is not None
    assert legacy.attribute("comfy.rgthree.FastMuter") == "comfy.rgthree"
    assert legacy.attribute("comfy.solo.Thing") == "comfy.solo"
    # No configured pack matches -> the truthful compat-layer chip.
    assert legacy.attribute("comfy.mystery.Node") == "comfy"
    # Default legacy badge: shared "unported legacy pack" mark + color,
    # per-pack display name. Single-file packs get the same default.
    rgthree = legacy.packs["comfy.rgthree"]
    assert rgthree.display_name == "rgthree (ComfyUI)"
    assert rgthree.mark == "\U0001f9e9"
    assert rgthree.color
    assert legacy.packs["comfy.solo"].mark == "\U0001f9e9"
    # Declared [pack.presentation] overrides the default badge wholesale.
    themed_info = legacy.packs["comfy.themed"]
    assert themed_info.display_name == "Themed Pack"
    assert themed_info.abbr == "TH"
    assert themed_info.color == "#aa3355"
    assert not themed_info.mark
    # A declared icon rides along: legacy packs get raster badges through
    # the same presentation-only manifest, no porting required.
    assert themed_info.icon is not None
    assert themed_info.icon.media_type == "image/png"
    assert themed_info.icon.digest.startswith("sha256:")
    assert themed_info.icon.data == png_bytes()
    # Blueprints ride the same presentation-only on-ramp: a legacy pack
    # ships starter workflows without being a loadable Dinkster pack.
    assert [bp.id for bp in themed_info.blueprints] == ["starter"]
    assert themed_info.blueprints[0].data == b'{"graphs": {}}'
    # Templates too, with asset references validated against the same
    # file's [[pack.assets]] declarations.
    assert [tp.id for tp in themed_info.templates] == ["quickstart"]
    assert themed_info.templates[0].assets == ("aux",)
    assert themed_info.templates[0].data == b'{"graphs": {}}'
    # [[pack.assets]] rides it too: the declaration resolves under the
    # synthetic legacy pack id, whose artifact root the spec maps to the
    # pack directory - so packaged acquisition works for unported packs.
    assert [a.id for a in themed_info.assets] == ["aux"]
    assert themed_info.assets[0].need.digest == aux_digest
    assert themed_info.assets[0].nodes == ("comfy.themed.AuxNode",)
    assert themed_info.assets[0].need.sources == (
        PackagedSource(pack="comfy.themed", path="models/aux.bin"),
    )
    # Only directory packs get roots: single-file packs have no adjacent
    # manifest to declare assets from.
    assert dict(legacy.asset_roots) == {
        "comfy.rgthree": tmp_path / "packs" / "rgthree",
        "comfy.themed": themed,
        "comfy.broken": broken,
    }
    # Packs without a manifest (or with a broken one) simply have none.
    assert rgthree.blueprints == ()
    assert rgthree.assets == () and legacy.packs["comfy.broken"].assets == ()
    # Broken presentation file: advisory, so the pack still composes and
    # wears the shared default.
    assert legacy.packs["comfy.broken"].display_name == "broken (ComfyUI)"
    assert legacy.packs["comfy.broken"].mark == "\U0001f9e9"

    with pytest.raises(CompositionError, match="not found"):
        comfy_compat_specs(tmp_path / "nowhere")
    with pytest.raises(CompositionError, match="legacy pack not found"):
        comfy_compat_specs(root, legacy_packs=[tmp_path / "missing"])


def test_comfy_host_types_register_save_target_runtime_values() -> None:
    from dinkster_assets import SAVE_TARGET_TYPE, AssetError, SaveTarget
    from dinkster_values import TypeRegistry

    from dinkster.comfy_compose import register_comfy_host_types

    registry = TypeRegistry()
    register_comfy_host_types(registry)
    register_comfy_host_types(registry)  # both compat PackSpecs share the hook

    explicit = {"mount": "comfy-output", "prefix": "renders/scene"}
    wrapped = registry.wrap(SAVE_TARGET_TYPE, explicit)
    assert wrapped.resolve() == SaveTarget(**explicit)
    assert wrapped.fingerprint == "dinkster.save_target:comfy-output/renders/scene"
    encoded = registry.spec(SAVE_TARGET_TYPE).encode(explicit)
    assert registry.spec(SAVE_TARGET_TYPE).decode(encoded) == SaveTarget(**explicit)

    from dinkster_nodes_media_io import SaveImage

    advertised_default = SaveImage.schema().inputs[1].default
    assert advertised_default is None

    with pytest.raises(AssetError, match="unknown keys"):
        registry.wrap(
            SAVE_TARGET_TYPE,
            {"mount": "comfy-output", "prefix": "renders/scene", "path": "/tmp"},
        )
    with pytest.raises(AssetError, match="invalid save target prefix"):
        registry.wrap(
            SAVE_TARGET_TYPE,
            {"mount": "comfy-output", "prefix": "../escape"},
        )


def test_composition_close_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, presentation=False)
        composition = await compose_serving([manifest], worker_env=WORKER_ENV)
        assert isinstance(composition, Composition)
        await composition.close()
        await composition.close()  # second close is a no-op, not an error

    asyncio.run(scenario())


def test_composition_close_is_exactly_once_concurrent_and_cancellation_safe() -> None:
    from dinkster.compose import ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        composition = composer.composition
        started = asyncio.Event()
        release = asyncio.Event()
        close_calls = 0

        class SlowWorker:
            async def close(self) -> None:
                nonlocal close_calls
                close_calls += 1
                started.set()
                await release.wait()

        composition._isolated.append(SlowWorker())  # noqa: SLF001
        cancelled_close = asyncio.create_task(composition.close())
        await started.wait()
        other_close = asyncio.create_task(composition.close())
        cancelled_close.cancel()
        await asyncio.sleep(0)
        assert not cancelled_close.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_close
        await other_close
        await composition.close()
        assert close_calls == 1

    asyncio.run(scenario())


def test_composition_closes_component_publishers_after_tenant_registries() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        composition = composer.composition
        events: list[str] = []

        class Registry:
            async def close(self) -> None:
                events.append("registry")

        class Publisher:
            def close(self) -> None:
                events.append("publisher")

        class Worker:
            async def close(self) -> None:
                events.append("worker")

        composition._tenant_registries.append(Registry())  # type: ignore[arg-type]  # noqa: SLF001
        composition._component_publishers.append(Publisher())  # type: ignore[arg-type]  # noqa: SLF001
        composition._isolated.append(Worker())  # noqa: SLF001

        await composition.close()

        assert events == ["registry", "publisher", "worker"]

    asyncio.run(scenario())


# -- memory governance (DESIGN 3.10) ---------------------------------------
#
# The relay, lease, and shedding protocols are proven at the session level
# (test_relay.py, test_reservations.py). These tests prove the COMPOSITION
# wiring: a host-supplied governor and reservation service reach every
# worker the composer starts, so a pack's manifest-declared consumers and
# reservation planner are live on the production path, not just when a test
# constructs IsolatedWorker by hand.


def test_compose_serving_relays_pack_consumers_to_governor(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor()
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        composition = await compose_serving(
            [manifest],
            worker_env=WORKER_ENV,
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        try:
            engine = composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "mem.load",
                        {"name": "sd15.safetensors", "vram": 400, "ram": 300},
                    )
                }
            )
            result = await engine.run(graph, ["load"])
            assert "load" in result.outputs
            # The child's snapshot push races the result frame: poll until
            # the relayed consumer's footprint lands in the parent governor.
            async with asyncio.timeout(8):
                while governor.footprint("vram:cuda:0") != 400:
                    await asyncio.sleep(0.02)
            assert governor.footprint("ram") == 300
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_serving_composer_full_free_releases_live_worker_without_startup_guard(
    tmp_path: Path,
    isolated_full_free_pool: ResidentPool,
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 1000})
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        composer = ServingComposer(
            worker_env=WORKER_ENV,
            governor=governor,
            dev=True,
            cache_mode="layered",
            cache_dir=tmp_path / "execution-cache",
        )
        try:
            await composer.add_pack(DEV_PACK_MANIFEST)
            await composer.add_pack(manifest)
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            gradient = Graph(
                nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})}
            )
            assert (await engine.run(gradient, ["g"])).executed == ("g",)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "mem.load",
                        {"name": "sd15.safetensors", "vram": 0, "ram": 0},
                    )
                }
            )
            await engine.run(graph, ["load"])
            await asyncio.sleep(0.05)

            results = await composer.full_free("maintenance-composed")
            local = results[0]
            worker = next(row for row in results[1:] if row["worker"] == "memorypack")
            assert local["status"] == "complete"
            assert worker["worker"] == "memorypack"
            assert worker["status"] == "complete"
            consumers = cast("list[dict[str, object]]", worker["consumers"])
            assert {item["consumer"] for item in consumers} == {
                "models",
                "zero-cost",
            }
            events.clear()
            assert (await engine.run(gradient, ["g"])).cached == ("g",)
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_hit.detail["cacheLayer"] == "disk"

            use = Graph(
                nodes={
                    "load": GraphNode("mem.load", {"name": "sd15.safetensors"}),
                    "use": GraphNode("mem.use", {"model": Link("load", "model")}),
                }
            )
            rerun = await engine.run(use, ["use"])
            assert "load" in rerun.executed
            assert rerun.outputs["use"]["name"].resolve() == "sd15.safetensors"

            live_worker = composer._records["memorypack"].worker
            await live_worker.close()
            results = await composer.full_free("maintenance-missing")
            local = results[0]
            missing = next(row for row in results[1:] if row["worker"] == "memorypack")
            assert local["status"] == "complete"
            assert missing == {
                "worker": "memorypack",
                "workerInstance": None,
                "deviceMap": {"mapping": {}, "qualifier": None},
                "status": "error",
                "error": "worker is unavailable at the maintenance snapshot",
                "consumers": [],
            }
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_serving_composer_full_free_serializes_group_members_by_process(
    tmp_path: Path,
    isolated_full_free_pool: ResidentPool,
) -> None:
    async def scenario() -> None:
        manifests: list[Path] = []
        for name in ("alpha", "beta"):
            directory = tmp_path / name
            directory.mkdir()
            module = directory / f"{name}_nodes.py"
            module.write_text(
                "from dinkster_schema import Node,NodeSchema\n"
                "class Empty(Node):\n"
                "    @classmethod\n"
                "    def define_schema(cls):\n"
                f"        return NodeSchema(node_type='{name}.empty')\n"
                "    @classmethod\n"
                "    async def execute(cls):\n"
                "        return cls.outputs()\n"
                "NODES=[Empty]\n"
            )
            manifest = directory / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n\n'
                f'[pack.entry]\nnodes = "{name}_nodes:NODES"\n'
                'consumers = "memorypack_nodes:serial_memory_consumers"\n'
            )
            manifests.append(manifest)
        env = {
            "PYTHONPATH": os.pathsep.join(
                (str(TESTS_DIR), *(str(manifest.parent) for manifest in manifests))
            )
        }
        composer = ServingComposer()
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        env=env,
                        worker_group="models",
                        group_manifests=tuple(manifests),
                    )
                )
            workers = await composer.full_free("group-members")
            grouped = [worker for worker in workers if worker["worker"] in {"alpha", "beta"}]
            assert len(grouped) == 2
            assert len({worker["workerInstance"] for worker in grouped}) == 1
            assert all(worker["status"] == "complete" for worker in grouped)
            assert all(
                worker["consumers"] == [{"consumer": "serial", "status": "complete"}]
                for worker in grouped
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_serving_composer_full_free_cancellation_waits_for_settlement(
    monkeypatch: pytest.MonkeyPatch,
    isolated_full_free_pool: ResidentPool,
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def blocked(
            _composer: ServingComposer, _request_id: str
        ) -> tuple[dict[str, object], ...]:
            entered.set()
            await finish.wait()
            return ()

        monkeypatch.setattr(ServingComposer, "_full_free_locked", blocked)
        composer = ServingComposer()
        operation = asyncio.create_task(composer.full_free("cancelled-maintenance"))
        await entered.wait()
        operation.cancel()
        await asyncio.sleep(0)
        assert composer._mutate.locked()
        assert not operation.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert not composer._mutate.locked()
        await composer.close()

    asyncio.run(scenario())


def test_compose_serving_delivers_pack_telemetry(tmp_path: Path) -> None:
    async def scenario() -> None:
        reported = ReportedTelemetry()
        governor = MemoryGovernor(telemetry=reported.probe, telemetry_devices=reported.devices)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
            'telemetry = "memorypack_nodes:memory_telemetry"\n'
        )
        composition = await compose_serving(
            [manifest],
            worker_env=WORKER_ENV,
            governor=governor,
            telemetry=reported,
        )
        try:
            # Measurements rode the worker's hello: no invocation needed.
            assert reported.probe("vram:cuda:0") == MeasuredMemory(
                free_bytes=8_000, total_bytes=10_000
            )
            # The governor reports the measured, unbudgeted device.
            status = governor.status()
            assert status["vram:cuda:0"]["budgetBytes"] is None
            assert status["vram:cuda:0"]["measured"] == {
                "freeBytes": 8_000,
                "totalBytes": 10_000,
            }
        finally:
            await composition.close()
        # Composition close closes every worker: measurements die with it.
        assert reported.devices() == frozenset()

    asyncio.run(scenario())


def test_compose_serving_wires_worker_reservations(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "isopack"\nnamespaces = ["iso"]\n\n[pack.entry]\n'
            'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
            'reservations = "isopack_nodes:plan_reservations"\n'
        )
        composition = await compose_serving(
            [manifest],
            worker_env=WORKER_ENV,
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        try:
            engine = composition.make_engine(lambda event: None)
            graph = Graph(nodes={"hog": GraphNode("iso.hog", {"nbytes": 60, "seconds": 0.3})})
            task = asyncio.create_task(engine.run(graph, ["hog"]))
            # The planned lease is visible in the parent's governor while
            # the child executes...
            async with asyncio.timeout(8):
                while governor.reserved("ram") != 60:
                    await asyncio.sleep(0.005)
            result = await task
            assert "hog" in result.outputs
            # ...and released once the invocation settles.
            async with asyncio.timeout(8):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await composition.close()

    asyncio.run(scenario())
