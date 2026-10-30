from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from dinkster_api.v1 import Region
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.rtdetr import model as rtdetr_model
from dinkster_nodes_vision.rtdetr import register_types
from dinkster_nodes_vision.rtdetr.model import execute_detect
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_rtdetr_pack/dinkster-pack.toml"
TORCH_NUM_THREADS = 1


@contextmanager
def _torch_threads(count: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


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
from dinkster_nodes_vision.rtdetr import model

class FakeModel:
    def __call__(self, _batch, _source_size):
        return [{
            "labels": torch.tensor([0, 2]),
            "scores": torch.tensor([0.9, 0.8]),
            "boxes": torch.tensor([[1.0, 1.0, 4.0, 3.0], [0.0, 0.0, 2.0, 2.0]]),
        }]

model.load_model = FakeModel
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
                    "detect": GraphNode(
                        "dinkster.detection.detect",
                        {
                            "image": TypedLiteral("IMAGE", source.tolist()),
                            "provider": "dinkster-vision-rtdetr",
                            "prompt": "person,car",
                            "min_score": 0.5,
                            "max_results": -1,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["detect"])
            detections = result.outputs["detect"]["detections"].resolve()
            assert result.outputs["detect"]["count"].resolve() == 2
            assert isinstance(detections, list)
            assert [detection.label for detection in detections] == ["person", "car"]
            assert detections[0].region == Region(1.0, 1.0, 3.0, 2.0)
        finally:
            await worker.close()

    asyncio.run(scenario())
