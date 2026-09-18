from __future__ import annotations

import asyncio
import base64
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
from dinkster_values import TypeRegistry, register_core_types
from dinkster_vision_birefnet import model as birefnet_model
from dinkster_vision_birefnet import register_types
from dinkster_vision_birefnet.model import execute_matte, prepare_frame
from dinkster_workers import IsolatedWorker, load_manifest

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages" / "dinkster-vision-birefnet" / "dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "birefnet_c67885b.json"
MODEL_DIGEST = "blake3:03f8793ff101fb10981ee700fe276a6f481af00cb607dfafcfee46aeb8e638db"
MODEL_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
TORCH_NUM_THREADS = 1


class _ModelResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == MODEL_DIGEST else None


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_BIREFNET_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_BIREFNET_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_BIREFNET_TEST_MODEL does not exist: {path}")
    return path


def _decode(record: object, *, dtype: np.dtype[np.generic], key: str) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload[key])
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


def test_birefnet_output_matches_pinned_comfyui_vector() -> None:
    golden = _golden()
    assert golden["baseline"] == "c67885b14556cf3e4e061862925282d403d09862"
    assert golden["modelBlake3"] == MODEL_DIGEST
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    assert golden["torchvision"] == "0.28.0+cpu"
    source = _decode(golden["source"], dtype=np.dtype(np.uint8), key="uint8Base64")
    expected = _decode(golden["matte"], dtype=np.dtype(np.float32), key="float32Base64")
    _install_model(_model_path())
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        with use_declared_asset_pack("dinkster-vision-birefnet"):
            actual = execute_matte(source[None].astype(np.float32) / 255.0)
    finally:
        torch.set_num_threads(previous_threads)
    assert actual.dtype == np.float32
    assert actual.shape == (1, *expected.shape)
    np.testing.assert_array_equal(actual[0], expected)


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
        golden = _golden()
        source = (
            _decode(
                golden["source"],
                dtype=np.dtype(np.uint8),
                key="uint8Base64",
            )[None].astype(np.float32)
            / 255.0
        )
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
            assert isinstance(matte, np.ndarray)
            assert matte.shape == source.shape[:3] and matte.dtype == np.float32
            assert np.isfinite(matte).all()
            assert 0.0 <= float(matte.min()) <= float(matte.max()) <= 1.0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_model_artifact_is_the_expected_bytes() -> None:
    with _model_path().open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
