from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.depth_anything_v3 import model as depth_model
from dinkster_nodes_vision.depth_anything_v3 import register_types
from dinkster_nodes_vision.depth_anything_v3.da3 import (
    OUT_LAYERS,
    PATCH_SIZE,
    DepthAnything3MonoLarge,
)
from dinkster_nodes_vision.depth_anything_v3.model import (
    execute_depth_anything_v3,
    normalize_depth,
    prepare_frame,
    relative_depth,
    resize_hint,
    target_size,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest
from safetensors import safe_open

ROOT = Path(__file__).parents[3]
MANIFEST = (
    ROOT
    / "packages/dinkster-nodes-vision/dinkster_vision_depth_anything_v3_pack/dinkster-pack.toml"
)
GOLDEN_PATH = ROOT / "tests" / "goldens" / "depth_anything_3_mono_large_e7051b0.json"
MODEL_DIGEST = "blake3:c7c3ae1883d3ad41d64aa9ce2988f265fa3c437105fadc32ff2949b7e8f18323"
MODEL_SHA256 = "9b44eda5bedba5b4e125686fdb79d1db309c1b9785277576eb930f885b008f96"
TORCH_NUM_THREADS = 1


class _ModelResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == MODEL_DIGEST else None


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_DEPTH_ANYTHING_V3_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_DEPTH_ANYTHING_V3_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_DEPTH_ANYTHING_V3_TEST_MODEL does not exist: {path}")
    return path


def _decode(record: object, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload[f"{dtype.name}Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=dtype).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def _install_model(path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _ModelResolver(path))


def _vault(tmp_path: Path) -> AssetVault:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        with _model_path().open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                writer.write(chunk)
        writer.commit()
    return vault


def test_mono_large_architecture_matches_artifact() -> None:
    assert OUT_LAYERS == (4, 11, 17, 23)
    assert PATCH_SIZE == 14
    with torch.device("meta"):
        model = DepthAnything3MonoLarge()
    actual = model.state_dict()
    with safe_open(_model_path(), framework="pt", device="cpu") as stored:
        expected_keys = {name.removeprefix("model.") for name in stored.keys()}
        assert set(actual) == expected_keys
        for name, parameter in actual.items():
            assert tuple(parameter.shape) == tuple(stored.get_slice(f"model.{name}").get_shape())


def test_preprocessing_matches_comfyui_byte_grid() -> None:
    golden = _golden()
    source = _decode(golden["source"], dtype=np.dtype(np.uint8))
    prepared = prepare_frame(source)
    assert target_size(*source.shape[:2]) == (378, 504)
    assert prepared.shape == (1, 3, 378, 504)
    assert prepared.dtype == torch.float32
    assert hashlib.sha256(prepared.numpy().tobytes()).hexdigest() == golden["preparedSha256"]


def test_depth_output_matches_pinned_comfyui_vector() -> None:
    golden = _golden()
    assert golden["baseline"] == "e7051b03758a1247e3adb84a5b784ffacb9a23bd"
    assert golden["generationCpu"] == "AMD Ryzen 9 5950X 16-Core Processor"
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["opencv"] == "5.0.0"
    assert golden["pillow"] == "12.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    source = _decode(golden["source"], dtype=np.dtype(np.uint8))
    expected_raw = _decode(golden["rawDepth"], dtype=np.dtype(np.float32))
    expected_output = _decode(golden["output"], dtype=np.dtype(np.float32))
    _install_model(_model_path())
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        with use_declared_asset_pack("dinkster-vision-depth-anything-v3"):
            actual_raw = relative_depth(depth_model.load_model(), source)
            actual_output = resize_hint(normalize_depth(actual_raw), 64)
    finally:
        torch.set_num_threads(previous_threads)
    assert torch.get_num_threads() == previous_threads
    np.testing.assert_array_equal(actual_raw, expected_raw)
    # Hosted resizing differed by at most 7.450581e-09; 1.5e-08 is twice that
    # spread, with a 1e-05 relative floor for float32 model output.
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1.5e-8)


def test_batches_channels_and_immutable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(depth_model, "load_model", object)
    seen: list[np.ndarray] = []

    def fake_depth(_model: object, frame: np.ndarray) -> np.ndarray:
        seen.append(frame.copy())
        return np.ascontiguousarray(frame[:, :, 0], dtype=np.float32)

    monkeypatch.setattr(depth_model, "relative_depth", fake_depth)
    image = np.zeros((2, 3, 4, 4), dtype=np.float32)
    image[0, :, :, :3] = (0.2, 0.3, 0.4)
    image[0, :, :, 3] = 0.0
    image[1, :, :, :3] = (0.8, 0.7, 0.6)
    image[1, :, :, 3] = 1.0
    output = execute_depth_anything_v3(image, resolution=64)
    assert len(seen) == 2
    assert all(frame.shape == (3, 4, 3) for frame in seen)
    assert output.shape == (2, 64, 85, 3)
    assert output.dtype == np.float32 and output.flags.c_contiguous
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output.setflags(write=True)

    gray = execute_depth_anything_v3(np.full((1, 2, 3, 1), 0.25), resolution=64)
    assert seen[-1].shape == (2, 3, 3)
    assert gray.shape == (1, 64, 96, 3)


def test_invalid_inputs_fail_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(depth_model, "load_model", unexpected_load)
    with pytest.raises(ValueError, match="resolution must be an integer"):
        execute_depth_anything_v3(np.zeros((1, 4, 4, 3)), resolution=32)
    with pytest.raises(ValueError, match="non-empty BHWC"):
        execute_depth_anything_v3(np.zeros((0, 4, 4, 3)), resolution=64)
    with pytest.raises(ValueError, match="finite pixel"):
        execute_depth_anything_v3(np.full((1, 4, 4, 3), np.nan), resolution=64)
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_depth_anything_v3(np.zeros((1, 4, 4, 2)), resolution=64)


def test_depth_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        golden = _golden()
        source = _decode(golden["source"], dtype=np.dtype(np.uint8))[None].astype(np.float32)
        source /= 255.0
        vault = _vault(tmp_path)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            extra_env={"DINKSTER_ASSET_VAULT": str(vault.root)},
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
                    "depth": GraphNode(
                        "dinkster.preprocess.model_depth",
                        {
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "provider": "dinkster-vision-depth-anything-v3",
                            "resolution": 64,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["depth"])
            output = result.outputs["depth"]["image"].resolve()
            assert isinstance(output, np.ndarray)
            assert output.shape == (1, 64, 85, 3) and output.dtype == np.float32
            assert np.isfinite(output).all()
            assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
        finally:
            await worker.close()

    depth_model._MODEL = None
    gc.collect()
    asyncio.run(scenario())


def test_model_artifact_is_the_expected_bytes() -> None:
    path = _model_path()
    assert path.stat().st_size == 1_336_748_056
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
