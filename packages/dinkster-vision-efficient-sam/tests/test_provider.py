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

pytest.importorskip("onnxruntime")

from dinkster_api.v1 import Detection, Region
from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_values import TypeRegistry, register_core_types
from dinkster_vision_efficient_sam import register_types
from dinkster_vision_efficient_sam.model import (
    _create_sessions,
    _load_sessions,
    _predict_candidates,
    _sigmoid,
    execute_segment,
)
from dinkster_workers import IsolatedWorker, load_manifest

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages" / "dinkster-vision-efficient-sam" / "dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "efficient_sam_vitt_d525f62.json"
ENCODER_DIGEST = "blake3:106600f3dd645019eff0d6fabe46cc1359654b03bfaec45316dabb1121cc9e6e"
DECODER_DIGEST = "blake3:41cdd8a75918dbef651ba10a244d22084999fc6162a265b9d504faacaef14004"
ENCODER_SHA256 = "84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951"
DECODER_SHA256 = "a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11"
INTRA_OP_NUM_THREADS = 1


def _model_paths() -> dict[str, Path]:
    encoder = os.environ.get("DINKSTER_EFFICIENT_SAM_TEST_ENCODER")
    decoder = os.environ.get("DINKSTER_EFFICIENT_SAM_TEST_DECODER")
    if not encoder or not decoder:
        pytest.skip("EfficientSAM test model paths are not configured")
    paths = {ENCODER_DIGEST: Path(encoder), DECODER_DIGEST: Path(decoder)}
    for path in paths.values():
        if not path.is_file():
            pytest.fail(f"EfficientSAM test model does not exist: {path}")
    return paths


def _decode_uint8(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _decode_float32(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["float32Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def _vault(tmp_path: Path) -> AssetVault:
    vault = AssetVault(tmp_path / "vault")
    paths = _model_paths()
    for digest, path in paths.items():
        with vault.writer(digest) as writer:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    writer.write(chunk)
            writer.commit()
    return vault


def _require_exact_vector_cpu() -> None:
    try:
        fields = {
            key.strip(): value.strip()
            for line in Path("/proc/cpuinfo")
            .read_text(encoding="utf-8")
            .split("\n\n", 1)[0]
            .splitlines()
            if ":" in line
            for key, value in (line.split(":", 1),)
        }
    except OSError:
        pytest.skip("EfficientSAM exact vectors require Linux CPU dispatch metadata")
    flags = set(fields.get("flags", "").split())
    if fields.get("vendor_id") != "AuthenticAMD" or "avx2" not in flags or "avx512f" in flags:
        pytest.skip("EfficientSAM exact vectors require an AMD AVX2 host without AVX-512")


def test_efficient_sam_outputs_match_pinned_reference_vectors(tmp_path: Path) -> None:
    _require_exact_vector_cpu()
    golden = _golden()
    assert golden["baseline"] == "d525f622e6f640acf5a0fc37c7ca1f243da5bde0"
    assert golden["encoderSha256"] == ENCODER_SHA256
    assert golden["decoderSha256"] == DECODER_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["onnxruntime"] == "1.29.0"
    assert golden["intraOpNumThreads"] == INTRA_OP_NUM_THREADS
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _vault(tmp_path))
    frame = _decode_uint8(golden["source"]).astype(np.float32) / 255.0
    boxes = [
        cast("tuple[float, float, float, float]", tuple(cast("list[float]", box)))
        for box in cast("list[object]", golden["boxes"])
    ]
    with use_declared_asset_pack(manifest.name):
        sessions = _create_sessions(intra_op_num_threads=INTRA_OP_NUM_THREADS)
        logits, ious = _predict_candidates(frame, boxes, sessions)
    np.testing.assert_array_equal(logits, _decode_float32(golden["logits"]))
    np.testing.assert_array_equal(ious, _decode_float32(golden["ious"]))


def test_efficient_sam_preserves_detections_and_attaches_soft_masks(tmp_path: Path) -> None:
    golden = _golden()
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _vault(tmp_path))
    image = _decode_uint8(golden["source"])[None].astype(np.float32) / 255.0
    boxes = cast("list[list[float]]", golden["boxes"])
    detections = [
        Detection("disk", 0.9, Region(boxes[0][0], boxes[0][1], 30.0, 28.0)),
        Detection("rectangle", 0.7, Region(boxes[1][0], boxes[1][1], 27.0, 36.0)),
        Detection("outside", 0.1, Region(-20.0, -20.0, 5.0, 5.0)),
    ]
    with use_declared_asset_pack(manifest.name):
        segmented, masks = execute_segment(image, detections)

    assert len(segmented) == len(masks) == 3
    for before, after, mask in zip(detections, segmented, masks, strict=True):
        assert (after.label, after.score, after.region) == (
            before.label,
            before.score,
            before.region,
        )
        assert after.mask is not None and after.mask.shape == image.shape[1:3]
        assert after.mask.dtype == np.float32 and not after.mask.flags.writeable
        assert mask.shape == image.shape[0:3]
        np.testing.assert_array_equal(mask[0], after.mask)
        assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
    assert np.count_nonzero(masks[0]) > 0
    assert np.count_nonzero(masks[1]) > 0
    assert np.count_nonzero(masks[2]) == 0

    with use_declared_asset_pack(manifest.name):
        reference_logits, reference_ious = _predict_candidates(
            image[0],
            [
                (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]),
                (boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]),
            ],
            _load_sessions(),
        )
    for index in range(2):
        candidate = int(np.argmax(reference_ious[index]))
        np.testing.assert_array_equal(masks[index][0], _sigmoid(reference_logits[index, candidate]))


def test_efficient_sam_requires_one_frame_but_accepts_no_detections() -> None:
    image = np.zeros((1, 8, 8, 3), dtype=np.float32)
    assert execute_segment(image, []) == ([], [])
    with pytest.raises(ValueError, match="exactly one"):
        execute_segment(np.concatenate((image, image)), [])


def test_efficient_sam_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        golden = _golden()
        image = _decode_uint8(golden["source"])[None].astype(np.float32) / 255.0
        box = cast("list[list[float]]", golden["boxes"])[0]
        detections = [Detection("object", 0.8, Region(box[0], box[1], 30.0, 28.0))]
        vault = _vault(tmp_path)
        manifest = load_manifest(MANIFEST)
        install_declared_assets(manifest.name, manifest.assets, vault)
        with use_declared_asset_pack(manifest.name):
            expected_detections, expected_masks = execute_segment(image, detections)
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
                    "segment": GraphNode(
                        "dinkster.detection.segment",
                        {
                            "image": TypedLiteral("dinkster.image", image.tolist()),
                            "detections": TypedLiteral(
                                "list<dinkster.detection>",
                                [detection.to_record() for detection in detections],
                            ),
                            "provider": "dinkster-vision-efficient-sam",
                        },
                    )
                }
            )
            result = await engine.run(graph, ["segment"])
            actual_detections = result.outputs["segment"]["detections"].resolve()
            actual_masks = result.outputs["segment"]["masks"].resolve()
            assert actual_detections == expected_detections
            assert isinstance(actual_masks, list)
            assert len(actual_masks) == len(expected_masks)
            for actual, expected in zip(actual_masks, expected_masks, strict=True):
                np.testing.assert_array_equal(actual, expected)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_efficient_sam_artifacts_are_the_expected_bytes() -> None:
    paths = _model_paths()
    assert hashlib.sha256(paths[ENCODER_DIGEST].read_bytes()).hexdigest() == ENCODER_SHA256
    assert hashlib.sha256(paths[DECODER_DIGEST].read_bytes()).hexdigest() == DECODER_SHA256
