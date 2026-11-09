from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.depth_anything_v3 import model as depth_model
from dinkster_nodes_vision.depth_anything_v3 import register_types
from dinkster_nodes_vision.depth_anything_v3.model import (
    execute_depth_anything_v3,
    prepare_frame,
    target_size,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

ROOT = Path(__file__).parents[3]
MANIFEST = (
    ROOT
    / "packages/dinkster-nodes-vision/dinkster_vision_depth_anything_v3_pack/dinkster-pack.toml"
)
GOLDEN_PATH = ROOT / "tests" / "goldens" / "depth_anything_3_mono_large_e7051b0.json"
MODEL_DIGEST = "blake3:c7c3ae1883d3ad41d64aa9ce2988f265fa3c437105fadc32ff2949b7e8f18323"


def _decode(record: object, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload[f"{dtype.name}Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=dtype).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def test_preprocessing_matches_comfyui_byte_grid() -> None:
    golden = _golden()
    source = _decode(golden["source"], dtype=np.dtype(np.uint8))
    prepared = prepare_frame(source)
    assert target_size(*source.shape[:2]) == (378, 504)
    assert prepared.shape == (1, 3, 378, 504)
    assert prepared.dtype == torch.float32
    assert hashlib.sha256(prepared.numpy().tobytes()).hexdigest() == golden["preparedSha256"]


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
        source = np.zeros((1, 2, 3, 3), dtype=np.float32)
        source[0, :, :, 0] = (0.0, 0.5, 1.0)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            launcher=worker_patch_launcher(
                tmp_path,
                """
import numpy as np
from dinkster_nodes_vision.depth_anything_v3 import model

model.load_model = lambda: object()
model.relative_depth = lambda _model, frame: np.ascontiguousarray(
    frame[:, :, 0], dtype=np.float32
)
""",
            ),
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
                            "image": TypedLiteral("IMAGE", source.tolist()),
                            "provider": "dinkster-vision-depth-anything-v3",
                            "resolution": 64,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["depth"])
            output = result.outputs["depth"]["image"].resolve()
            assert isinstance(output, np.ndarray)
            assert output.shape == (1, 64, 96, 3) and output.dtype == np.float32
            assert float(output[0, 0, 0, 0]) == pytest.approx(1.0)
            assert float(output[0, 0, -1, 0]) == pytest.approx(0.0)
            np.testing.assert_array_equal(output[0, :, :, 0], output[0, :, :, 1])
            np.testing.assert_array_equal(output[0, :, :, 1], output[0, :, :, 2])
        finally:
            await worker.close()

    asyncio.run(scenario())
