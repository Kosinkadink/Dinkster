from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import json
import os
import sys
from contextlib import contextmanager
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
from dinkster_nodes_vision.sam31 import TrackObjects, register_types
from dinkster_nodes_vision.sam31 import model as sam_model
from dinkster_nodes_vision.sam31.model import (
    execute_detect,
    execute_segment,
    execute_text_segment,
    execute_track,
    prepare_frame,
)
from dinkster_nodes_vision.sam31.tokenizer import CLIP_BOS, CLIP_EOS, load_tokenizer
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest
from safetensors import safe_open
from torch.nn import functional as F

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_sam31_pack/dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "sam31_multiplex_8dc3f3f.json"
TRACK_GOLDEN_PATH = ROOT / "tests" / "goldens" / "sam31_track_8dc3f3f.json"
DETECT_GOLDEN_PATH = ROOT / "tests" / "goldens" / "sam31_detect_8dc3f3f.json"
TRACK_GOLDEN_SHA256 = "1d7ea415a504476475d634cc7a308a4bef0a11f4997fad082c16703b6c2e5521"
DETECT_GOLDEN_SHA256 = "3a686eed763d64873e81a3aaacc101a3ae3f2e68564d1626bdffaf40eb8a2345"
MODEL_DIGEST = "blake3:1c8d5762dbaf238bc9a2f10de07e0c476d1b68e75453566feb8dc9bcf2cb41c5"
MODEL_SHA256 = "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03"
TORCH_NUM_THREADS = 1


class _ModelResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == MODEL_DIGEST else None


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_SAM31_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_SAM31_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_SAM31_TEST_MODEL does not exist: {path}")
    return path


def _decode(record: object, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload[f"{dtype.name}Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=dtype).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def _tracking_golden() -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads(TRACK_GOLDEN_PATH.read_text(encoding="utf-8")),
    )


def _detection_golden() -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads(DETECT_GOLDEN_PATH.read_text(encoding="utf-8")),
    )


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


def test_architecture_exactly_matches_selected_checkpoint_state() -> None:
    with torch.device("meta"):
        model = sam_model.SAM31Model()
    expected = model.state_dict()
    actual: dict[str, tuple[int, ...]] = {}
    with safe_open(_model_path(), framework="pt", device="cpu") as stored:
        for source in stored.keys():
            target = sam_model._target_key(source)
            if target is None:
                continue
            shape = tuple(stored.get_slice(source).get_shape())
            if target.endswith((".in_proj_weight", ".in_proj_bias")):
                base, suffix = target.rsplit(".in_proj_", 1)
                ending = ".weight" if suffix == "weight" else ".bias"
                split_shape = (shape[0] // 3, *shape[1:])
                actual[base + ".q_proj" + ending] = split_shape
                actual[base + ".k_proj" + ending] = split_shape
                actual[base + ".v_proj" + ending] = split_shape
            else:
                target = target.replace(".mlp.lin1.", ".mlp.0.")
                target = target.replace(".mlp.lin2.", ".mlp.2.")
                target = target.replace(".norm_final_attn.", ".norm_final.")
                actual[target] = shape
                if target.startswith("tracker.interactive_sam_") or target == (
                    "tracker.interactivity_no_mem_embed"
                ):
                    actual[target.removeprefix("tracker.")] = shape
    assert set(actual) == set(expected)
    assert len(actual) == 1_030
    assert sum(value.numel() for value in expected.values()) == 483_988_295
    for name, shape in actual.items():
        assert shape == tuple(expected[name].shape)


def test_detection_architecture_exactly_matches_complete_checkpoint_state() -> None:
    with torch.device("meta"):
        model = sam_model.SAM31DetectionModel()
    expected = model.state_dict()
    actual: dict[str, tuple[int, ...]] = {}
    with safe_open(_model_path(), framework="pt", device="cpu") as stored:
        for source in stored.keys():
            target = sam_model._detection_target_key(source)
            assert target is not None
            shape = tuple(stored.get_slice(source).get_shape())
            if target.endswith((".in_proj_weight", ".in_proj_bias")):
                base, suffix = target.rsplit(".in_proj_", 1)
                ending = ".weight" if suffix == "weight" else ".bias"
                split_shape = (shape[0] // 3, *shape[1:])
                actual[base + ".q_proj" + ending] = split_shape
                actual[base + ".k_proj" + ending] = split_shape
                actual[base + ".v_proj" + ending] = split_shape
            else:
                target = target.replace(".mlp.lin1.", ".mlp.0.")
                target = target.replace(".mlp.lin2.", ".mlp.2.")
                target = target.replace(".norm_final_attn.", ".norm_final.")
                actual[target] = shape
                if target.startswith("tracker.interactive_sam_") or target == (
                    "tracker.interactivity_no_mem_embed"
                ):
                    actual[target.removeprefix("tracker.")] = shape
    assert set(actual) == set(expected)
    assert len(actual) == 1_983
    assert sum(value.numel() for value in expected.values()) == 876_883_581
    for name, shape in actual.items():
        assert shape == tuple(expected[name].shape)


def test_offline_tokenizer_matches_pinned_clip_tokens() -> None:
    tokenizer = load_tokenizer()
    assert tokenizer.encode("person") == (2533,)
    assert tokenizer.encode("neon guitarist") == (13919, 13035)
    assert tokenizer.encode("\u4f60\u597d") == (47466, 254, 29290, 377)
    assert tokenizer.encode("\U0001f468\u200d\U0001f469\u200d\U0001f467") == (
        25023,
        26304,
        964,
        356,
    )
    assert tokenizer.batches("red car") == ((CLIP_BOS, 736, 1615, CLIP_EOS, *((0,) * 28)),)
    long = tokenizer.batches(" ".join("person" for _ in range(31)))
    assert len(long) == 2
    assert long[0][0] == long[1][0] == CLIP_BOS
    assert long[0][31] == long[1][2] == CLIP_EOS


def test_sam31_output_matches_pinned_comfyui_vector() -> None:
    golden = _golden()
    assert golden["baseline"] == "8dc3f3f2094121c0a013e21d89136ebc331d2974"
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["pillow"] == "12.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    assert golden["sourceImageBaseline"] == "0b1ef3ec90846bf82eba195ddcc30a1f5b2b6b38"
    assert golden["sourceImageSha256"] == (
        "83e63383d1715a7084afb5cf1e2e47e302869c2c944bfa62d86e769b4ff65ecc"
    )
    source = _decode(golden["source"], dtype=np.dtype(np.uint8)).astype(np.float32) / 255.0
    expected = _decode(golden["refinedLogits"], dtype=np.dtype(np.float32))
    box = cast("tuple[float, float, float, float]", tuple(cast("list[float]", golden["box"])))
    _install_model(_model_path())
    with _torch_threads(TORCH_NUM_THREADS):
        with use_declared_asset_pack("dinkster-vision-sam31"):
            model = sam_model.load_model()
            prepared = prepare_frame(source)
            prepared_sha256 = hashlib.sha256(prepared.numpy().tobytes()).hexdigest()
            assert prepared_sha256 == golden["preparedSha256"]
            with torch.inference_mode():
                features = model.encode_image(prepared)
                first = model.segment(
                    features,
                    box=sam_model._prompt(box, height=source.shape[0], width=source.shape[1]),
                )
                refined = model.segment(features, mask=first)
                actual = F.interpolate(
                    refined,
                    size=source.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
    # Intermediate model tensors vary by CPU kernel, so the deterministic
    # prepared input stays exact while the final float output uses the hosted floor.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1.6e-5)
    sam_model._MODEL = None
    gc.collect()


def test_sam31_tracking_matches_pinned_comfyui_vector() -> None:
    assert hashlib.sha256(TRACK_GOLDEN_PATH.read_bytes()).hexdigest() == TRACK_GOLDEN_SHA256
    golden = _tracking_golden()
    assert golden["baseline"] == "8dc3f3f2094121c0a013e21d89136ebc331d2974"
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["opencv"] == "5.0.0"
    assert golden["pillow"] == "12.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    assert golden["sourceImageBaseline"] == "0b1ef3ec90846bf82eba195ddcc30a1f5b2b6b38"
    assert golden["sourceImageSha256"] == (
        "83e63383d1715a7084afb5cf1e2e47e302869c2c944bfa62d86e769b4ff65ecc"
    )
    frames = _decode(golden["sourceFrames"], dtype=np.dtype(np.uint8)).astype(np.float32)
    frames /= 255.0
    boxes = cast("list[list[float]]", golden["boxes"])
    detections = [
        Detection(
            f"object-{index}",
            1.0,
            Region(box[0], box[1], box[2] - box[0], box[3] - box[1]),
        )
        for index, box in enumerate(boxes)
    ]
    expected = _decode(golden["trackedMasks"], dtype=np.dtype(np.float32))
    expected_combined = _decode(golden["combinedMask"], dtype=np.dtype(np.float32))
    _install_model(_model_path())
    with _torch_threads(TORCH_NUM_THREADS):
        with use_declared_asset_pack("dinkster-vision-sam31"):
            tracked, combined = execute_track(frames, detections)
            actual = np.stack(tracked)
    # The local and hosted tracking runs measured zero drift; the 1e-05
    # relative and absolute floors cover these float32 model masks.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(combined, expected_combined, rtol=1e-5, atol=1e-5)
    sam_model._MODEL = None
    gc.collect()


def test_sam31_detection_and_text_segmentation_match_pinned_comfyui_vector() -> None:
    assert hashlib.sha256(DETECT_GOLDEN_PATH.read_bytes()).hexdigest() == DETECT_GOLDEN_SHA256
    golden = _detection_golden()
    assert golden["baseline"] == "8dc3f3f2094121c0a013e21d89136ebc331d2974"
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["pillow"] == "12.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    assert golden["transformers"] == "5.16.1"
    assert golden["prompt"] == "person"
    assert cast("list[int]", golden["tokenIds"])[:3] == [CLIP_BOS, 2533, CLIP_EOS]
    assert golden["tokenizerCommit"] == "3bee28119e6b28e75b82b811b87b56935314e6a5"
    assert golden["tokenizerSha256"] == (
        "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
    )
    assert golden["sourceImageSha256"] == (
        "83e63383d1715a7084afb5cf1e2e47e302869c2c944bfa62d86e769b4ff65ecc"
    )
    source = _decode(golden["source"], dtype=np.dtype(np.uint8))[None].astype(np.float32)
    source /= 255.0
    expected_text = _decode(golden["textEmbedding"], dtype=np.dtype(np.float32))
    expected_boxes = _decode(golden["boxes"], dtype=np.dtype(np.float32))
    expected_logits = _decode(golden["logits"], dtype=np.dtype(np.float32))
    expected_coarse = _decode(golden["coarseMask"], dtype=np.dtype(np.float32))
    expected_mask = _decode(golden["refinedMask"], dtype=np.dtype(np.float32))
    top_query = cast("int", golden["topQuery"])
    _install_model(_model_path())
    with _torch_threads(TORCH_NUM_THREADS):
        with use_declared_asset_pack("dinkster-vision-sam31"):
            model = sam_model.load_detection_model()
            captured_text: list[torch.Tensor] = []
            captured_detection: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            text_hook = model.text_encoder.register_forward_hook(
                lambda _module, _inputs, output: captured_text.append(output.detach().cpu())
            )
            detector_hook = model.detector.register_forward_hook(
                lambda _module, _inputs, output: captured_detection.append(
                    tuple(value.detach().cpu() for value in output)
                )
            )
            try:
                detections, masks = execute_text_segment(
                    source,
                    prompt="person",
                    min_score=0.5,
                )
            finally:
                text_hook.remove()
                detector_hook.remove()
    assert len(captured_text) == len(captured_detection) == len(detections) == len(masks) == 1
    boxes, logits, coarse = captured_detection[0]
    # Local CPU kernels differed by at most 7.6293945e-06; 1.6e-05 is more
    # than twice that spread, with a 1e-05 relative floor for float32 output.
    np.testing.assert_allclose(captured_text[0].numpy(), expected_text, rtol=1e-5, atol=1.6e-5)
    np.testing.assert_allclose(boxes[0].numpy(), expected_boxes, rtol=1e-5, atol=1.6e-5)
    np.testing.assert_allclose(logits[0].numpy(), expected_logits, rtol=1e-5, atol=1.6e-5)
    np.testing.assert_allclose(
        coarse[0, top_query].numpy(), expected_coarse, rtol=1e-5, atol=1.6e-5
    )
    detection = detections[0]
    assert detection.label == "person"
    expected_score = torch.tensor(expected_logits[top_query].item(), dtype=torch.float32).sigmoid()
    assert detection.score == pytest.approx(float(expected_score), rel=1e-5, abs=1.6e-5)
    raw_box = expected_boxes[top_query] * np.array((256, 256, 256, 256), dtype=np.float32)
    assert detection.region == Region(
        float(np.clip(raw_box[0], 0, 256)),
        float(np.clip(raw_box[1], 0, 256)),
        float(np.clip(raw_box[2], 0, 256) - np.clip(raw_box[0], 0, 256)),
        float(np.clip(raw_box[3], 0, 256) - np.clip(raw_box[1], 0, 256)),
    )
    assert detection.mask is not None
    np.testing.assert_array_equal(detection.mask, expected_mask)
    np.testing.assert_array_equal(masks[0][0], expected_mask)
    assert masks[0].shape == (1, 256, 256)
    assert masks[0].dtype == np.float32 and not masks[0].flags.writeable
    assert not np.shares_memory(detection.mask, masks[0])
    assert detection.mask.dtype == np.float32 and not detection.mask.flags.writeable
    with pytest.raises(ValueError):
        detection.mask.setflags(write=True)
    with pytest.raises(ValueError):
        masks[0].setflags(write=True)
    sam_model._MODEL = None
    gc.collect()


def test_exact_vector_thread_scope_restores_after_failure() -> None:
    previous = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="injected"):
        with _torch_threads(TORCH_NUM_THREADS):
            assert torch.get_num_threads() == TORCH_NUM_THREADS
            raise RuntimeError("injected")
    assert torch.get_num_threads() == previous


def test_detections_channels_clipping_and_immutable_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sam_model, "load_model", object)
    seen: list[tuple[np.ndarray, list[tuple[float, float, float, float]]]] = []

    def fake_logits(
        _model: object,
        frame: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        seen.append((frame.copy(), boxes))
        return [
            np.full(frame.shape[:2], index - 0.5, dtype=np.float32) for index, _ in enumerate(boxes)
        ]

    monkeypatch.setattr(sam_model, "predict_logits", fake_logits)
    image = np.zeros((1, 4, 5, 4), dtype=np.float32)
    image[0, :, :, :3] = (0.2, 0.3, 0.4)
    image[0, :, :, 3] = 0.9
    detections = [
        Detection("clipped", 0.9, Region(-2.0, -1.0, 5.0, 4.0)),
        Detection("inside", 0.7, Region(1.0, 1.0, 3.0, 2.0)),
        Detection("outside", 0.2, Region(10.0, 10.0, 2.0, 2.0)),
    ]
    segmented, masks = execute_segment(image, detections)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], image[0, :, :, :3])
    assert seen[0][1] == [(0.0, 0.0, 3.0, 3.0), (1.0, 1.0, 4.0, 3.0)]
    assert len(segmented) == len(masks) == 3
    for before, after, mask in zip(detections, segmented, masks, strict=True):
        assert (after.label, after.score, after.region) == (
            before.label,
            before.score,
            before.region,
        )
        assert after.mask is not None and after.mask.shape == (4, 5)
        assert after.mask.dtype == np.float32 and not after.mask.flags.writeable
        with pytest.raises(ValueError):
            after.mask.setflags(write=True)
        assert mask.shape == (1, 4, 5) and not mask.flags.writeable
        np.testing.assert_array_equal(mask[0], after.mask)
    assert np.count_nonzero(masks[0]) == 0
    assert np.count_nonzero(masks[1]) == 20
    assert np.count_nonzero(masks[2]) == 0

    gray = np.full((1, 2, 3, 1), 0.25, dtype=np.float32)
    execute_segment(gray, [Detection("gray", 1.0, Region(0.0, 0.0, 3.0, 2.0))])
    assert seen[-1][0].shape == (2, 3, 3)
    assert np.count_nonzero(seen[-1][0] != 0.25) == 0


def test_invalid_inputs_fail_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(sam_model, "load_model", unexpected_load)
    assert execute_segment(np.zeros((1, 4, 4, 3), dtype=np.float32), []) == ([], [])
    with pytest.raises(ValueError, match="exactly one"):
        execute_segment(np.zeros((2, 4, 4, 3), dtype=np.float32), [])
    with pytest.raises(ValueError, match="finite pixel"):
        execute_segment(np.full((1, 4, 4, 3), np.nan), [])
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_segment(np.zeros((1, 4, 4, 2), dtype=np.float32), [])


def test_detection_prompt_batch_order_clipping_and_immutable_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackbone:
        @staticmethod
        def trunk(_image: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 1, 1, 1)

    class FakeDetector:
        calls = 0

        def __call__(
            self,
            _trunk: torch.Tensor,
            _text: torch.Tensor,
            _mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.calls += 1
            high = 2.0 + self.calls
            return (
                torch.tensor(
                    [[[-0.1, 0.25, 0.75, 1.2], [0.2, 0.1, 0.4, 0.5], [0.0, 0.0, 1.0, 1.0]]]
                ),
                torch.tensor([[high, 2.0, -3.0]]),
                torch.zeros(1, 3, 2, 2),
            )

    class FakeModel:
        backbone = FakeBackbone()
        detector = FakeDetector()

        class text_encoder:
            @staticmethod
            def encode(_phrase: str) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(1, 1, 1), torch.ones(1, 1)

    mask_number = 0

    def fake_refine(
        _model: object,
        frame: np.ndarray,
        _coarse: torch.Tensor,
        _box: torch.Tensor,
    ) -> np.ndarray:
        nonlocal mask_number
        mask_number += 1
        return sam_model._immutable_mask(
            np.full(frame.shape[:2], mask_number % 2, dtype=np.float32)
        )

    monkeypatch.setattr(sam_model, "_refine_detection_mask", fake_refine)
    encoded = (
        ("first phrase", torch.zeros(1, 1, 1), torch.ones(1, 1)),
        ("second phrase", torch.ones(1, 1, 1), torch.ones(1, 1)),
    )
    actual = sam_model._detect_frame(
        cast("sam_model.SAM31DetectionModel", FakeModel()),
        np.zeros((4, 5, 3), dtype=np.float32),
        encoded,
        0.5,
    )
    assert [item.label for item in actual] == [
        "second phrase",
        "first phrase",
        "first phrase",
        "second phrase",
    ]
    assert [round(item.score, 6) for item in actual] == [
        round(float(torch.sigmoid(torch.tensor(4.0))), 6),
        round(float(torch.sigmoid(torch.tensor(3.0))), 6),
        round(float(torch.sigmoid(torch.tensor(2.0))), 6),
        round(float(torch.sigmoid(torch.tensor(2.0))), 6),
    ]
    assert actual[0].region == Region(0.0, 1.0, 3.75, 3.0)
    for item in actual:
        assert item.mask is not None and item.mask.shape == (4, 5)
        assert item.mask.dtype == np.float32 and not item.mask.flags.writeable
        with pytest.raises(ValueError):
            item.mask.setflags(write=True)
    for index, item in enumerate(actual):
        for other in actual[index + 1 :]:
            assert item.mask is not None and other.mask is not None
            assert not np.shares_memory(item.mask, other.mask)
    capped = sam_model._detect_frame(
        cast("sam_model.SAM31DetectionModel", FakeModel()),
        np.zeros((4, 5, 3), dtype=np.float32),
        encoded,
        0.5,
        2,
    )
    assert len(capped) == 2
    assert [item.score for item in capped] == sorted([item.score for item in capped], reverse=True)
    sliced = sam_model._detect_frame(
        cast("sam_model.SAM31DetectionModel", FakeModel()),
        np.zeros((4, 5, 3), dtype=np.float32),
        encoded,
        -1.0,
        -1,
        "slice-stop",
    )
    assert len(sliced) == 5
    assert mask_number == 11
    monkeypatch.setattr(sam_model, "load_detection_model", FakeModel)
    batched = execute_detect(
        np.zeros((2, 4, 5, 3), dtype=np.float32),
        prompt="object",
        min_score=0.5,
        max_results=2,
    )
    assert len(batched) == 4


def test_detection_threshold_excludes_equal_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackbone:
        @staticmethod
        def trunk(_image: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 1, 1, 1)

    class FakeDetector:
        @staticmethod
        def __call__(
            _trunk: torch.Tensor,
            _text: torch.Tensor,
            _mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]),
                torch.tensor([[1.0, 0.0]]),
                torch.zeros(1, 2, 2, 2),
            )

    class FakeModel:
        backbone = FakeBackbone()
        detector = FakeDetector()

    monkeypatch.setattr(
        sam_model,
        "_refine_detection_mask",
        lambda _model, frame, _coarse, _box: sam_model._immutable_mask(
            np.zeros(frame.shape[:2], dtype=np.float32)
        ),
    )
    actual = sam_model._detect_frame(
        cast("sam_model.SAM31DetectionModel", FakeModel()),
        np.zeros((4, 5, 3), dtype=np.float32),
        (("object", torch.zeros(1, 1, 1), torch.ones(1, 1)),),
        0.5,
    )

    assert len(actual) == 1
    assert actual[0].score > 0.5


def test_detection_empty_and_invalid_inputs_fail_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(sam_model, "load_detection_model", unexpected_load)
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    assert execute_detect(image, prompt=" , ", min_score=0.5) == []
    assert execute_text_segment(image, prompt=" , ", min_score=0.5) == ([], [])
    with pytest.raises(TypeError, match="prompt must be a string"):
        execute_detect(image, prompt=cast("str", 1), min_score=0.5)
    with pytest.raises(TypeError, match="min_score must be a number"):
        execute_detect(image, prompt="person", min_score=cast("float", "0.5"))
    with pytest.raises(ValueError, match="finite"):
        execute_detect(image, prompt="person", min_score=float("nan"))
    with pytest.raises(ValueError, match="at least -1 in count mode"):
        execute_detect(image, prompt="person", min_score=0.5, max_results=-2)
    with pytest.raises(ValueError, match="unknown result limit mode"):
        execute_detect(
            image,
            prompt="person",
            min_score=0.5,
            max_results=0,
            result_limit_mode="unknown",
        )
    with pytest.raises(ValueError, match="integer"):
        execute_detect(
            image,
            prompt="person",
            min_score=0.5,
            max_results=cast("int", 1.0),
        )
    assert execute_detect(image, prompt="person", min_score=0.5, max_results=0) == []
    with pytest.raises(ValueError, match="non-empty BHWC"):
        execute_detect(np.zeros((0, 4, 5, 3)), prompt="person", min_score=0.5)
    with pytest.raises(ValueError, match="finite pixel"):
        execute_detect(np.full((1, 4, 5, 3), np.nan), prompt="person", min_score=0.5)
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_detect(np.zeros((1, 4, 5, 2)), prompt="person", min_score=0.5)


def test_text_segmentation_executes_prompt_phrases_once_and_frames_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded: list[str] = []
    expected_phrase_count = 2

    class FakeTextEncoder:
        @staticmethod
        def encode(phrase: str) -> tuple[torch.Tensor, torch.Tensor]:
            encoded.append(phrase)
            return torch.zeros(1, 1, 1), torch.ones(1, 1)

    class FakeModel:
        text_encoder = FakeTextEncoder()

    def fake_detect_frame(
        _model: object,
        frame: np.ndarray,
        phrases: object,
        min_score: float,
        max_results: int = -1,
        result_limit_mode: str = "count",
    ) -> list[Detection]:
        assert len(cast("list[object]", phrases)) == expected_phrase_count
        assert min_score == 0.25
        assert max_results == -1
        assert result_limit_mode == "count"
        mask = sam_model._immutable_mask(np.full(frame.shape[:2], frame[0, 0, 0]))
        return [
            Detection(
                str(float(frame[0, 0, 0])),
                1.0,
                Region(0.0, 0.0, 1.0, 1.0),
                mask,
            )
        ]

    monkeypatch.setattr(sam_model, "load_detection_model", FakeModel)
    monkeypatch.setattr(sam_model, "_detect_frame", fake_detect_frame)
    images = np.zeros((2, 2, 2, 3), dtype=np.float32)
    images[0] = 0.2
    images[1] = 0.8
    result, masks = execute_text_segment(images, prompt=" person, guitar ", min_score=0.25)
    assert encoded == ["person", "guitar"]
    assert [item.label for item in result] == [
        str(float(images[0, 0, 0, 0])),
        str(float(images[1, 0, 0, 0])),
    ]
    assert len(masks) == len(result) == 2
    for detection, mask in zip(result, masks, strict=True):
        assert detection.mask is not None
        assert mask.shape == (1, 2, 2)
        np.testing.assert_array_equal(mask[0], detection.mask)
        assert not np.shares_memory(mask, detection.mask)
        assert not mask.flags.writeable
        with pytest.raises(ValueError):
            mask.setflags(write=True)

    encoded.clear()
    expected_phrase_count = 1
    result, _ = execute_text_segment(
        images[:1],
        prompt="person, guitar",
        prompt_mode="literal",
        min_score=0.25,
    )
    assert encoded == ["person, guitar"]
    assert len(result) == 1


def test_tracking_preserves_object_order_soft_edges_and_immutable_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def track(self, frames: torch.Tensor, initial_masks: torch.Tensor) -> torch.Tensor:
            assert frames.shape == (2, 3, 4, 5)
            assert initial_masks.shape == (1, 1, 4, 5)
            assert torch.count_nonzero(initial_masks) == 20
            return torch.full((2, 1, 1008, 1008), 0.25, dtype=torch.float32)

    model = FakeModel()
    monkeypatch.setattr(sam_model, "load_model", lambda: model)
    seen: list[tuple[np.ndarray, list[tuple[float, float, float, float]]]] = []

    def fake_logits(
        actual_model: object,
        frame: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        *,
        bicubic: bool = False,
    ) -> list[np.ndarray]:
        assert actual_model is model
        assert bicubic
        seen.append((frame.copy(), boxes))
        return [np.ones(frame.shape[:2], dtype=np.float32) for _ in boxes]

    monkeypatch.setattr(sam_model, "predict_logits", fake_logits)
    images = np.zeros((2, 4, 5, 4), dtype=np.float32)
    images[..., :3] = (0.2, 0.3, 0.4)
    images[..., 3] = 0.9
    detections = [
        Detection("outside", 0.1, Region(10.0, 10.0, 1.0, 1.0)),
        Detection("person", 0.9, Region(-1.0, -2.0, 7.0, 8.0)),
    ]
    masks, combined = execute_track(images, detections)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], images[0, :, :, :3])
    assert seen[0][1] == [(0.0, 0.0, 5.0, 4.0)]
    assert len(masks) == 2
    assert masks[0].shape == masks[1].shape == (2, 4, 5)
    assert np.count_nonzero(masks[0]) == 0
    np.testing.assert_array_equal(masks[1], 0.25)
    np.testing.assert_array_equal(combined, 0.25)
    for mask in masks:
        assert mask.dtype == np.float32 and not mask.flags.writeable
        with pytest.raises(ValueError):
            mask.setflags(write=True)


def test_tracking_accepts_first_frame_masks_and_preserves_object_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def track(self, frames: torch.Tensor, initial_masks: torch.Tensor) -> torch.Tensor:
            assert frames.shape == (2, 3, 4, 5)
            assert initial_masks.shape == (2, 1, 3, 2)
            np.testing.assert_array_equal(initial_masks[:, 0].numpy(), masks)
            result = torch.empty((2, 2, 1008, 1008), dtype=torch.float32)
            result[:, 0] = 0.25
            result[:, 1] = 0.75
            return result

    monkeypatch.setattr(sam_model, "load_model", FakeModel)
    images = np.zeros((2, 4, 5, 3), dtype=np.float32)
    masks = np.zeros((2, 3, 2), dtype=np.float32)
    masks[0, :, 0] = 1.0
    masks[1, :, 1] = 1.0
    tracked, combined = execute_track(images, initial_masks=masks)
    assert len(tracked) == 2
    np.testing.assert_array_equal(tracked[0], 0.25)
    np.testing.assert_array_equal(tracked[1], 0.75)
    np.testing.assert_array_equal(combined, 0.75)
    for mask in tracked:
        assert mask.shape == (2, 4, 5)
        assert mask.dtype == np.float32 and not mask.flags.writeable
        with pytest.raises(ValueError):
            mask.setflags(write=True)


def test_tracking_node_returns_an_immutable_all_object_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = np.zeros((2, 3, 4), dtype=np.float32)
    second = np.zeros((2, 3, 4), dtype=np.float32)
    first[:, :, :2] = 0.25
    second[:, :, 1:] = 0.75
    source_union = np.full((2, 3, 4), 0.9, dtype=np.float32)
    source_union.setflags(write=False)
    monkeypatch.setattr(
        sam_model,
        "execute_track",
        lambda *_args, **_kwargs: ([first, second], source_union),
    )

    result = TrackObjects.execute(
        image=np.zeros((2, 3, 4, 3), dtype=np.float32),
        provider="dinkster-vision-sam31",
        initial_masks=np.ones((2, 3, 4), dtype=np.float32),
    )
    assert result["masks"] is not None
    assert cast("list[np.ndarray]", result["masks"])[0] is first
    assert cast("list[np.ndarray]", result["masks"])[1] is second
    combined = cast("np.ndarray", result["combined"])
    assert combined is source_union
    assert combined.dtype == np.float32 and not combined.flags.writeable
    assert not np.shares_memory(combined, first)
    assert not np.shares_memory(combined, second)


def test_invalid_tracking_inputs_fail_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(sam_model, "load_model", unexpected_load)
    with pytest.raises(ValueError, match="exactly one"):
        execute_track(np.zeros((2, 4, 4, 3), dtype=np.float32), [])
    with pytest.raises(ValueError, match="exactly one"):
        execute_track(
            np.zeros((2, 4, 4, 3), dtype=np.float32),
            [Detection("object", 1.0, Region(0.0, 0.0, 1.0, 1.0))],
            initial_masks=np.ones((1, 4, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="non-empty BHW"):
        execute_track(
            np.zeros((2, 4, 4, 3), dtype=np.float32),
            initial_masks=np.zeros((0, 4, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="finite values"):
        execute_track(
            np.zeros((2, 4, 4, 3), dtype=np.float32),
            initial_masks=np.full((1, 4, 4), np.nan, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="non-empty BHWC"):
        execute_track(np.zeros((0, 4, 4, 3), dtype=np.float32), [])
    with pytest.raises(ValueError, match="finite pixel"):
        execute_track(np.full((2, 4, 4, 3), np.nan), [])
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_track(np.zeros((2, 4, 4, 2), dtype=np.float32), [])


def test_sam31_provider_nodes_execute_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        golden = _golden()
        source = _decode(golden["source"], dtype=np.dtype(np.uint8))[None].astype(np.float32)
        source /= 255.0
        box = cast("list[float]", golden["box"])
        detection = Detection(
            "performer",
            0.8,
            Region(box[0], box[1], box[2] - box[0], box[3] - box[1]),
        )
        expected = _decode(golden["refinedLogits"], dtype=np.dtype(np.float32)) > 0.0
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
            assert {
                "dinkster.detection.detect",
                "dinkster.detection.segment",
                "dinkster.detection.segment_text",
                "dinkster.detection.track",
            } <= worker.schemas.keys()
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "segment": GraphNode(
                        "dinkster.detection.segment",
                        {
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "detections": TypedLiteral(
                                "list<dinkster.detection>",
                                [detection.to_record()],
                            ),
                            "provider": "dinkster-vision-sam31",
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["segment"])
            detections = cast(
                "list[Detection]",
                result.outputs["segment"]["detections"].resolve(),
            )
            masks = cast(
                "list[np.ndarray]",
                result.outputs["segment"]["masks"].resolve(),
            )
            assert len(detections) == len(masks) == 1
            actual = detections[0]
            assert (actual.label, actual.score, actual.region) == (
                detection.label,
                detection.score,
                detection.region,
            )
            assert actual.mask is not None
            np.testing.assert_array_equal(actual.mask, expected)
            np.testing.assert_array_equal(masks[0][0], expected)
        finally:
            await worker.close()

    sam_model._MODEL = None
    gc.collect()
    asyncio.run(scenario())


def test_model_artifact_is_the_expected_bytes() -> None:
    path = _model_path()
    assert path.stat().st_size == 1_745_546_848
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
