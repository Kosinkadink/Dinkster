from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
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
from dinkster_inference.cli_args import args
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

args.cpu = True

birefnet_module = importlib.import_module("dinkster_nodes_vision.birefnet")
birefnet_model = importlib.import_module("dinkster_nodes_vision.birefnet.model")
register_types = birefnet_module.register_types
execute_matte = birefnet_model.execute_matte
prepare_frame = birefnet_model.prepare_frame

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_birefnet_pack/dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "birefnet_c67885b.json"
MODEL_DIGEST = "blake3:03f8793ff101fb10981ee700fe276a6f481af00cb607dfafcfee46aeb8e638db"


def _decode(record: object, *, dtype: np.dtype[np.generic], key: str) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload[key])
    return np.frombuffer(base64.b64decode(encoded), dtype=dtype).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def test_preprocessing_matches_comfyui_byte_grid() -> None:
    source = _decode(
        _golden()["source"],
        dtype=np.dtype(np.uint8),
        key="uint8Base64",
    )
    prepared = prepare_frame(source.astype(np.float32) / 255.0)
    assert prepared.shape == (1, 3, 1024, 1024)
    assert prepared.dtype == torch.float32
    assert hashlib.sha256(prepared.numpy().tobytes()).hexdigest() == (
        "cfd1b00e5b29f9756289d13f0559f10108ff5e0c82c0c51959f9a6c257f2faa5"
    )


def test_batches_channels_and_immutable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(birefnet_model, "load_model", object)
    seen: list[np.ndarray] = []

    def matte_frame(_model: object, frame: np.ndarray) -> np.ndarray:
        seen.append(frame.copy())
        return np.ascontiguousarray(frame[:, :, 0], dtype=np.float32)

    monkeypatch.setattr(birefnet_model, "_matte_frame", matte_frame)
    image = np.zeros((2, 3, 4, 4), dtype=np.float32)
    image[0, :, :, :3] = (0.2, 0.3, 0.4)
    image[0, :, :, 3] = 0.0
    image[1, :, :, :3] = (0.8, 0.7, 0.6)
    image[1, :, :, 3] = 1.0
    output = execute_matte(image)
    assert len(seen) == 2
    assert all(frame.shape == (3, 4, 3) for frame in seen)
    np.testing.assert_array_equal(seen[0], image[0, :, :, :3])
    np.testing.assert_array_equal(seen[1], image[1, :, :, :3])
    assert output.shape == (2, 3, 4)
    assert output.dtype == np.float32 and output.flags.c_contiguous
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output.setflags(write=True)
    assert isinstance(output.base, np.ndarray) and not output.base.flags.writeable
    with pytest.raises(ValueError):
        output.base.setflags(write=True)

    gray = execute_matte(np.full((1, 2, 3, 1), 0.25, dtype=np.float32))
    assert seen[-1].shape == (2, 3, 3)
    assert np.count_nonzero(seen[-1] != 0.25) == 0
    assert gray.shape == (1, 2, 3)


def test_invalid_inputs_fail_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(birefnet_model, "load_model", unexpected_load)
    with pytest.raises(ValueError, match="non-empty BHWC"):
        execute_matte(np.zeros((0, 4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="finite pixel"):
        execute_matte(np.full((1, 4, 4, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_matte(np.zeros((1, 4, 4, 2), dtype=np.float32))


def test_birefnet_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = np.full((1, 2, 3, 3), 0.2, dtype=np.float32)
        source[0, :, :, 0] = ((0.0, 0.5, 1.0), (0.75, 0.25, 0.5))
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
from dinkster_nodes_vision.birefnet import model

model.load_model = lambda: object()
model._matte_frame = lambda _model, frame: np.ascontiguousarray(
    frame[:, :, 0] * 0.5 + 0.25, dtype=np.float32
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
                    "matte": GraphNode(
                        "dinkster.image.matte",
                        {
                            "image": TypedLiteral("IMAGE", source.tolist()),
                            "provider": "dinkster-vision-birefnet",
                        },
                    )
                }
            )
            result = await engine.run(graph, ["matte"])
            matte = result.outputs["matte"]["mask"].resolve()
            assert isinstance(matte, np.ndarray)
            assert matte.shape == source.shape[:3] and matte.dtype == np.float32
            expected = np.array([[[0.25, 0.5, 0.75], [0.625, 0.375, 0.5]]], dtype=np.float32)
            np.testing.assert_array_equal(matte, expected)
        finally:
            await worker.close()

    asyncio.run(scenario())
