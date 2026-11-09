from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from dinkster_api.v1 import Detection, Region
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.efficient_sam import register_types
from dinkster_nodes_vision.efficient_sam.model import (
    execute_segment,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

ROOT = Path(__file__).parents[3]
MANIFEST = (
    ROOT / "packages/dinkster-nodes-vision/dinkster_vision_efficient_sam_pack/dinkster-pack.toml"
)


def test_efficient_sam_requires_one_frame_but_accepts_no_detections() -> None:
    image = np.zeros((1, 8, 8, 3), dtype=np.float32)
    assert execute_segment(image, []) == ([], [])
    with pytest.raises(ValueError, match="exactly one"):
        execute_segment(np.concatenate((image, image)), [])


def test_efficient_sam_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = np.zeros((1, 4, 5, 3), dtype=np.float32)
        detections = [
            Detection("left", 0.8, Region(1.0, 1.0, 2.0, 2.0)),
            Detection("right", 0.6, Region(3.0, 0.0, 2.0, 3.0)),
        ]
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
from dinkster_nodes_vision.efficient_sam import model

class Encoder:
    def run(self, _outputs, inputs):
        image = inputs["batched_images"]
        return [np.zeros((1, 1, image.shape[2], image.shape[3]), dtype=np.float32)]

class Decoder:
    def run(self, _outputs, inputs):
        height, width = (int(value) for value in inputs["orig_im_size"])
        left = float(inputs["batched_point_coords"][0, 0, 0, 0])
        logits = np.full((1, 1, 3, height, width), -20.0, dtype=np.float32)
        if left < 2.0:
            logits[0, 0, 1] = 0.0
            ious = np.array([[[0.1, 0.9, 0.2]]], dtype=np.float32)
        else:
            logits[0, 0, 2] = np.log(3.0)
            ious = np.array([[[0.2, 0.1, 0.8]]], dtype=np.float32)
        return [logits, ious]

model._load_sessions = lambda: (Encoder(), Decoder())
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
                    "segment": GraphNode(
                        "dinkster.detection.segment",
                        {
                            "image": TypedLiteral("IMAGE", image.tolist()),
                            "detections": TypedLiteral(
                                "list<DETECTION>",
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
            assert isinstance(actual_detections, list)
            assert [(item.label, item.score, item.region) for item in actual_detections] == [
                (item.label, item.score, item.region) for item in detections
            ]
            assert actual_detections[0].mask is not None
            assert actual_detections[1].mask is not None
            np.testing.assert_allclose(actual_detections[0].mask, 0.5, rtol=0.0, atol=1e-7)
            np.testing.assert_allclose(actual_detections[1].mask, 0.75, rtol=0.0, atol=1e-7)
            assert isinstance(actual_masks, list)
            assert len(actual_masks) == 2
            assert actual_masks[0].shape == actual_masks[1].shape == (1, 4, 5)
            np.testing.assert_allclose(actual_masks[0], 0.5, rtol=0.0, atol=1e-7)
            np.testing.assert_allclose(actual_masks[1], 0.75, rtol=0.0, atol=1e-7)
        finally:
            await worker.close()

    asyncio.run(scenario())
