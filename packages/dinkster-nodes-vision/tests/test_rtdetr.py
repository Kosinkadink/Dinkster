from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from dinkster_api.v1 import Detection, Region
from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.rtdetr import model as rtdetr_model
from dinkster_nodes_vision.rtdetr import register_types
from dinkster_nodes_vision.rtdetr.model import execute_detect, load_model, prepare_frames
from dinkster_nodes_vision.rtdetr.rtdetr import RTv4
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest
from safetensors import safe_open

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_rtdetr_pack/dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "rtdetr_v4_x_hgnet_c67885b.json"
MODEL_DIGEST = "blake3:5eaa01a6d16d654d9a4991ab1dfe489b580acc4b939cd1963ac6d12ceb9dc7f8"
MODEL_SHA256 = "581f9af9bbabb664d1891cbccd823308b176ecd409146f954dfa39af3bec2476"
MODEL_SIZE = 123_968_978
TORCH_NUM_THREADS = 1


class _ModelResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == MODEL_DIGEST else None


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_RTDETR_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_RTDETR_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_RTDETR_TEST_MODEL does not exist: {path}")
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


@contextmanager
def _torch_threads(count: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_architecture_exactly_matches_fp16_checkpoint_state() -> None:
    model = RTv4(enc_h=384, device=torch.device("cpu"), dtype=torch.float32)
    expected = model.state_dict()
    actual: dict[str, tuple[int, ...]] = {}
    with safe_open(_model_path(), framework="pt", device="cpu") as stored:
        keys = stored.keys()
        assert len(keys) == 1_183
        for key in keys:
            actual[key] = tuple(stored.get_slice(key).get_shape())
            tensor = stored.get_tensor(key)
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float16
    assert set(actual) == set(expected)
    for name, shape in actual.items():
        assert shape == tuple(expected[name].shape)


def test_outputs_match_pinned_comfyui_reference_vector() -> None:
    golden = _golden()
    assert golden["baseline"] == "c67885b14556cf3e4e061862925282d403d09862"
    assert golden["modelBlake3"] == MODEL_DIGEST
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    assert golden["torchvision"] == "0.28.0+cpu"
    source = _decode(golden["source"], dtype=np.dtype(np.uint8))
    frames = source[None].astype(np.float32) / 255.0
    _install_model(_model_path())
    previous_threads = torch.get_num_threads()
    with _torch_threads(TORCH_NUM_THREADS):
        with use_declared_asset_pack("dinkster-vision-rtdetr"):
            model = load_model()
            prepared = prepare_frames([frames[0]])
            with torch.inference_mode():
                outputs = model._forward(prepared)
                result = model.postprocess(outputs, (source.shape[1], source.shape[0]))[0]
            capped = execute_detect(
                np.concatenate((frames, frames)),
                prompt="",
                min_score=0.0,
                max_results=1,
            )
    assert torch.get_num_threads() == previous_threads
    assert len(capped) == 2 and capped[0] == capped[1]
    np.testing.assert_array_equal(
        outputs["pred_logits"].numpy(),
        _decode(golden["predLogits"], dtype=np.dtype(np.float32)),
    )
    np.testing.assert_array_equal(
        outputs["pred_boxes"].numpy(),
        _decode(golden["predBoxes"], dtype=np.dtype(np.float32)),
    )
    np.testing.assert_array_equal(
        result["labels"].numpy(),
        _decode(golden["labels"], dtype=np.dtype(np.int64)),
    )
    np.testing.assert_array_equal(
        result["boxes"].numpy(),
        _decode(golden["boxes"], dtype=np.dtype(np.float32)),
    )
    np.testing.assert_array_equal(
        result["scores"].numpy(),
        _decode(golden["scores"], dtype=np.dtype(np.float32)),
    )


def test_exact_vector_thread_scope_restores_after_failure() -> None:
    previous = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="injected"):
        with _torch_threads(TORCH_NUM_THREADS):
            assert torch.get_num_threads() == TORCH_NUM_THREADS
            raise RuntimeError("injected")
    assert torch.get_num_threads() == previous


def test_channels_batching_filtering_threshold_and_per_frame_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, tuple[int, int]]] = []

    class FakeModel:
        def __call__(
            self,
            batch: torch.Tensor,
            source_size: tuple[int, int],
        ) -> list[dict[str, torch.Tensor]]:
            calls.append((batch.clone(), source_size))
            frame_results = (
                (
                    torch.tensor([1, 2, 1, 1]),
                    torch.tensor([0.7, 0.9, 0.8, 0.5]),
                ),
                (
                    torch.tensor([2, 1, 1]),
                    torch.tensor([0.85, 0.95, 0.5]),
                ),
            )
            return [
                {
                    "labels": labels,
                    "scores": scores,
                    "boxes": torch.tensor(
                        [[index, index + 1, index + 2, index + 3] for index in range(len(labels))]
                    ),
                }
                for labels, scores in frame_results[: batch.shape[0]]
            ]

    monkeypatch.setattr(rtdetr_model, "load_model", FakeModel)
    image = np.zeros((2, 2, 3, 4), dtype=np.float32)
    image[0, ..., :3] = (0.2, 0.3, 0.4)
    image[0, ..., 3] = 0.0
    image[1, ..., :3] = (0.8, 0.7, 0.6)
    image[1, ..., 3] = 1.0

    unlimited = execute_detect(image, prompt=" BICYCLE, ", min_score=0.5, max_results=-1)
    assert [item.score for item in unlimited] == pytest.approx([0.8, 0.7, 0.95])
    assert [item.label for item in unlimited] == ["bicycle", "bicycle", "bicycle"]
    assert calls[0][0].shape == (2, 3, 640, 640)
    assert calls[0][1] == (3, 2)
    np.testing.assert_allclose(
        calls[0][0][0].numpy(),
        1.0,
        rtol=0.0,
        atol=float(np.finfo(np.float32).eps),
    )
    assert tuple(float(value) for value in calls[0][0][1, :, 0, 0]) == pytest.approx(
        (0.8, 0.7, 0.6)
    )
    capped = execute_detect(image, prompt="bicycle", min_score=0.5, max_results=1)
    assert [item.score for item in capped] == pytest.approx([0.8, 0.95])
    assert capped[0].region == Region(2.0, 3.0, 2.0, 2.0)
    assert capped[1].region == Region(1.0, 2.0, 2.0, 2.0)
    sliced = execute_detect(
        image,
        prompt="bicycle",
        min_score=-1.0,
        max_results=-1,
        result_limit_mode="slice-stop",
    )
    assert [item.score for item in sliced] == pytest.approx([0.8, 0.7, 0.95])
    assert (
        execute_detect(
            image,
            prompt="bicycle,person",
            prompt_mode="literal",
            min_score=0.5,
        )
        == []
    )
    assert execute_detect(image, prompt="bicycle", min_score=0.5, max_results=0) == []
    assert len(calls) == 4
    score_field = "score"
    with pytest.raises(FrozenInstanceError):
        setattr(capped[0], score_field, 0.0)
    x_field = "x"
    with pytest.raises(FrozenInstanceError):
        setattr(capped[0].region, x_field, 0.0)

    gray = np.full((1, 2, 3, 1), 0.25, dtype=np.float32)
    execute_detect(gray, prompt="", min_score=0.99)
    assert calls[-1][0].shape == (1, 3, 640, 640)
    np.testing.assert_allclose(
        calls[-1][0].numpy(),
        0.25,
        rtol=0.0,
        atol=float(np.spacing(np.float32(0.25))),
    )


def test_invalid_inputs_fail_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(rtdetr_model, "load_model", unexpected_load)
    image = np.zeros((1, 4, 5, 3), dtype=np.float32)
    assert execute_detect(image, prompt="", min_score=0.5, max_results=0) == []
    with pytest.raises(TypeError, match="prompt must be a string"):
        execute_detect(image, prompt=cast("str", 1), min_score=0.5)
    with pytest.raises(ValueError, match="unknown prompt mode"):
        execute_detect(image, prompt="", prompt_mode="unknown", min_score=0.5)
    with pytest.raises(TypeError, match="min_score must be a number"):
        execute_detect(image, prompt="", min_score=cast("float", "0.5"))
    with pytest.raises(ValueError, match="finite"):
        execute_detect(image, prompt="", min_score=float("nan"))
    with pytest.raises(ValueError, match="at least -1 in count mode"):
        execute_detect(image, prompt="", min_score=0.5, max_results=-2)
    with pytest.raises(ValueError, match="unknown result limit mode"):
        execute_detect(
            image,
            prompt="",
            min_score=0.5,
            max_results=0,
            result_limit_mode="unknown",
        )
    with pytest.raises(ValueError, match="integer"):
        execute_detect(image, prompt="", min_score=0.5, max_results=cast("int", 1.0))
    with pytest.raises(ValueError, match="non-empty BHWC"):
        execute_detect(np.zeros((0, 4, 5, 3)), prompt="", min_score=0.5)
    with pytest.raises(ValueError, match="finite pixel"):
        execute_detect(np.full((1, 4, 5, 3), np.nan), prompt="", min_score=0.5)
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_detect(np.zeros((1, 4, 5, 2)), prompt="", min_score=0.5)


def test_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
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
                    "detect": GraphNode(
                        "dinkster.detection.detect",
                        {
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "provider": "dinkster-vision-rtdetr",
                            "prompt": "",
                            "min_score": 0.0,
                            "max_results": 1,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["detect"])
            detections = result.outputs["detect"]["detections"].resolve()
            assert result.outputs["detect"]["count"].resolve() == 1
            assert isinstance(detections, list)
            assert len(detections) == 1 and isinstance(detections[0], Detection)
            assert detections[0].region.width >= 0.0
            assert detections[0].region.height >= 0.0
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_model_artifact_is_the_expected_bytes() -> None:
    path = _model_path()
    assert path.stat().st_size == MODEL_SIZE
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
