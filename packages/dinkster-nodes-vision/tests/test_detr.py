from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import Detection
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.detr import register_types
from dinkster_nodes_vision.detr.model import execute_detect
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_detr_pack/dinkster-pack.toml"


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
        source = np.zeros((1, 4, 5, 3), dtype=np.float32)
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
import torch
from dinkster_nodes_vision.detr import model

class FakeModel:
    def __call__(self, _image):
        logits = torch.full((2, 92), -10.0)
        logits[0, 1] = 5.0
        logits[1, 3] = 4.0
        return logits, torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.2, 0.3, 0.2, 0.2]])

model.load_model = FakeModel
""",
            ),
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
                            "image": TypedLiteral("IMAGE", source.tolist()),
                            "provider": "dinkster-vision-detr",
                            "prompt": "person,car",
                            "min_score": 0.1,
                            "max_results": -1,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["detect"])
            detections = result.outputs["detect"]["detections"].resolve()
            count = result.outputs["detect"]["count"].resolve()
            assert count == 2
            assert isinstance(detections, list)
            assert all(isinstance(detection, Detection) for detection in detections)
            assert [detection.label for detection in detections] == ["person", "car"]
            assert detections[0].region.x == pytest.approx(1.5)
            assert detections[0].region.width == pytest.approx(2.0)
        finally:
            await worker.close()

    asyncio.run(scenario())
