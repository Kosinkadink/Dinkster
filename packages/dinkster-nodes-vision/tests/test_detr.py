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
from dinkster_api.v1 import Detection
from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.detr import model as detr_model
from dinkster_nodes_vision.detr import register_types
from dinkster_nodes_vision.detr.model import COCO_CLASSES, execute_detect, load_model, prepare_frame
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_detr_pack/dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "detr_r50_29901c5.json"
MODEL_DIGEST = "blake3:2bb221c9ab83ea68d6a66bdc4cfe7bce4c49a1784287f5923521d34d23d150a2"
MODEL_SHA256 = "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc"
TORCH_NUM_THREADS = 1


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_DETR_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_DETR_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_DETR_TEST_MODEL does not exist: {path}")
    return path


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
    with vault.writer(MODEL_DIGEST) as writer:
        with _model_path().open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                writer.write(chunk)
        writer.commit()
    return vault


def test_detr_outputs_match_pinned_reference_vectors(tmp_path: Path) -> None:
    golden = _golden()
    assert golden["baseline"] == "29901c51d7fe8712168b8d0d64351170bc0f83e0"
    assert golden["generationCpu"] == "AMD Ryzen 9 5950X 16-Core Processor"
    assert golden["modelBlake3"] == MODEL_DIGEST
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["torchNumThreads"] == TORCH_NUM_THREADS
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _vault(tmp_path))
    source = _decode_uint8(golden["source"])[None].astype(np.float32) / 255.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        with use_declared_asset_pack(manifest.name):
            model = load_model()
            with torch.no_grad():
                logits, boxes = model(prepare_frame(source[0]))
    finally:
        detr_model._MODEL = None
        torch.set_num_threads(previous_threads)
    assert logits.dtype == torch.float32 and boxes.dtype == torch.float32
    # Hosted CPU kernels differed by at most 8.392334e-05; 8.5e-05 adds 1.3% headroom.
    np.testing.assert_allclose(
        logits.numpy(), _decode_float32(golden["logits"]), rtol=0, atol=8.5e-5
    )
    np.testing.assert_allclose(boxes.numpy(), _decode_float32(golden["boxes"]), rtol=0, atol=8.5e-5)


def test_detr_detections_are_ordered_filtered_and_clipped(tmp_path: Path) -> None:
    golden = _golden()
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _vault(tmp_path))
    source = _decode_uint8(golden["source"])[None].astype(np.float32) / 255.0
    height, width = source.shape[1:3]
    with use_declared_asset_pack(manifest.name):
        unfiltered = execute_detect(source, prompt="", min_score=0.0)
        assert len(unfiltered) == 100
        scores = [detection.score for detection in unfiltered]
        assert scores == sorted(scores, reverse=True)
        for detection in unfiltered:
            assert detection.label in COCO_CLASSES
            region = detection.region
            assert 0.0 <= region.x <= region.x + region.width <= width
            assert 0.0 <= region.y <= region.y + region.height <= height
        top = unfiltered[0].label
        expected = [detection for detection in unfiltered if detection.label == top]
        filtered = execute_detect(source, prompt=f" {top.upper()} ,", min_score=0.0)
        assert filtered == expected
        assert (
            execute_detect(
                source,
                prompt=f" {top.upper()} ,",
                min_score=0.0,
                max_results=1,
            )
            == expected[:1]
        )
        assert (
            execute_detect(
                source,
                prompt=f"{top},person",
                prompt_mode="literal",
                min_score=0.0,
            )
            == []
        )
        assert execute_detect(source, prompt="not a coco class", min_score=0.0) == []
        thresholded = execute_detect(source, prompt="", min_score=0.5)
        assert thresholded == [d for d in unfiltered if d.score >= 0.5]
        doubled = execute_detect(np.concatenate((source, source)), prompt="", min_score=0.0)
        assert doubled == unfiltered + unfiltered
        capped = execute_detect(
            np.concatenate((source, source)),
            prompt="",
            min_score=0.0,
            max_results=2,
        )
        assert capped == unfiltered[:2] + unfiltered[:2]
        sliced = execute_detect(
            np.concatenate((source, source)),
            prompt="",
            min_score=-1.0,
            max_results=-2,
            result_limit_mode="slice-stop",
        )
        assert sliced == unfiltered[:-2] + unfiltered[:-2]
        assert execute_detect(source, prompt="", min_score=2.0) == []
        assert execute_detect(source, prompt="", min_score=0.0, max_results=0) == []


def test_invalid_max_results_fails_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr("dinkster_nodes_vision.detr.model.load_model", unexpected_load)
    image = np.zeros((1, 2, 3, 3), dtype=np.float32)
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
    with pytest.raises(ValueError, match="finite"):
        execute_detect(image, prompt="", min_score=float("nan"), max_results=0)
    with pytest.raises(ValueError, match="integer"):
        execute_detect(image, prompt="", min_score=0.5, max_results=cast("int", 1.0))
    assert execute_detect(image, prompt="", min_score=0.5, max_results=0) == []


def test_detr_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(TORCH_NUM_THREADS)
        detr_model._MODEL = None
        golden = _golden()
        source = _decode_uint8(golden["source"])[None].astype(np.float32) / 255.0
        vault = _vault(tmp_path)
        manifest = load_manifest(MANIFEST)
        install_declared_assets(manifest.name, manifest.assets, vault)
        with use_declared_asset_pack(manifest.name):
            expected = execute_detect(source, prompt="", min_score=0.1)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            extra_env={
                "DINKSTER_ASSET_VAULT": str(vault.root),
                "MKL_NUM_THREADS": str(TORCH_NUM_THREADS),
                "OMP_NUM_THREADS": str(TORCH_NUM_THREADS),
            },
        )
        try:
            await worker.start()
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
                            "provider": "dinkster-vision-detr",
                            "prompt": "",
                            "min_score": 0.1,
                            "max_results": -1,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["detect"])
            detections = result.outputs["detect"]["detections"].resolve()
            count = result.outputs["detect"]["count"].resolve()
            assert count == len(expected)
            assert isinstance(detections, list)
            assert all(isinstance(detection, Detection) for detection in detections)
            assert detections == expected
        finally:
            await worker.close()
            detr_model._MODEL = None
            torch.set_num_threads(previous_threads)

    asyncio.run(scenario())


def test_model_artifact_is_the_expected_bytes() -> None:
    assert hashlib.sha256(_model_path().read_bytes()).hexdigest() == MODEL_SHA256
