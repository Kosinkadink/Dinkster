"""BiRefNet execution from source and bundled manifest-provisioned runtimes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from dinkster_assets import AssetVault
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_values import TypeRegistry, register_core_types
from dinkster_vision_birefnet import register_types
from dinkster_workers import IsolatedWorker, ensure_pack_venv, load_manifest

from dinkster.serve import _pack_runtime_sources

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "packages" / "dinkster-vision-birefnet" / "dinkster-pack.toml"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DINKSTER_BIREFNET_TEST_WHEELHOUSE")
    or not os.environ.get("DINKSTER_BIREFNET_TEST_MODEL"),
    reason=(
        "clean provisioning requires DINKSTER_BIREFNET_TEST_WHEELHOUSE and "
        "DINKSTER_BIREFNET_TEST_MODEL"
    ),
)


@pytest.fixture(params=["source", "bundled"])
def provider_runtime(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, dict[str, str]]:
    wheelhouse = Path(os.environ["DINKSTER_BIREFNET_TEST_WHEELHOUSE"]).resolve()
    manifest_path = MANIFEST
    if request.param == "bundled":
        wheels = tuple(wheelhouse.glob("dinkster_vision_birefnet-*.whl"))
        assert len(wheels) == 1
        with ZipFile(wheels[0]) as wheel:
            wheel.extractall(tmp_path / "installed")
        manifest_path = (
            tmp_path / "installed" / "dinkster_vision_birefnet_pack" / "dinkster-pack.toml"
        )
        assert not (manifest_path.parent / "pyproject.toml").exists()
    manifest = load_manifest(manifest_path)
    assert "dinkster-inference-torch==0.0.1" in manifest.requires
    workspace, pythonpath = _pack_runtime_sources(manifest)
    if request.param == "source":
        assert ROOT / "packages" / "dinkster-inference-torch" in workspace
        # Editable installation, not inherited dev paths, must supply source dependencies.
        pythonpath = ""
    else:
        assert workspace == ()
        assert pythonpath == str(manifest_path.parent)
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_NO_CACHE", "1")
    monkeypatch.setenv("UV_FIND_LINKS", str(wheelhouse))
    python = ensure_pack_venv(
        manifest,
        venv_root=tmp_path / "venvs",
        workspace_packages=workspace,
        accelerator="cpu",
    )
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata as m, importlib.util as u, json, pathlib, sys\n"
            "dist = m.distribution('dinkster-inference-torch')\n"
            "assert dist.version == '0.0.1'\n"
            "spec = u.find_spec('dinkster_inference_torch')\n"
            "assert spec is not None and spec.origin is not None\n"
            "assert pathlib.Path(spec.origin).with_name('birefnet.py').is_file()\n"
            "direct = dist.read_text('direct_url.json')\n"
            "if sys.argv[1] == 'source':\n"
            "    assert direct is not None\n"
            "    assert json.loads(direct)['url'] == pathlib.Path(sys.argv[2]).as_uri()\n"
            "else:\n"
            "    assert direct is None\n"
            "    assert pathlib.Path(spec.origin).is_relative_to(sys.prefix)\n"
            "for name in ('dinkster_engine', 'comfy_kitchen', 'tokenizers', 'comfy'):\n"
            "    assert u.find_spec(name) is None, name\n",
            str(request.param),
            str(ROOT / "packages" / "dinkster-inference-torch"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    return (
        manifest_path,
        str(python),
        {
            "PYTHONPATH": pythonpath,
            "PYTHONSAFEPATH": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )


def test_manifest_provisioned_birefnet_matches_pinned_matte(
    tmp_path: Path, provider_runtime: tuple[Path, str, dict[str, str]]
) -> None:
    manifest, python, environment = provider_runtime
    golden = json.loads((ROOT / "tests" / "goldens" / "birefnet_c67885b.json").read_text())
    model = Path(os.environ["DINKSTER_BIREFNET_TEST_MODEL"])
    with model.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == golden["modelSha256"]
    source = np.frombuffer(base64.b64decode(golden["source"]["uint8Base64"]), dtype=np.uint8)
    source = source.reshape(golden["source"]["shape"])[None].astype(np.float32) / 255.0
    expected = np.frombuffer(base64.b64decode(golden["matte"]["float32Base64"]), dtype=np.float32)
    expected = expected.reshape(golden["matte"]["shape"])
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(golden["modelBlake3"]) as writer, model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            writer.write(chunk)
        writer.commit()

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            python=python,
            extra_env={**environment, "DINKSTER_ASSET_VAULT": str(vault.root)},
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
                    "matte": GraphNode(
                        "dinkster.image.matte",
                        {
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "provider": "dinkster-vision-birefnet",
                        },
                    )
                }
            )
            result = await engine.run(graph, ["matte"])
            matte = result.outputs["matte"]["mask"].resolve()
            assert isinstance(matte, np.ndarray) and matte.dtype == np.float32
            np.testing.assert_array_equal(matte, expected[None])
        finally:
            await worker.close()

    asyncio.run(scenario())
