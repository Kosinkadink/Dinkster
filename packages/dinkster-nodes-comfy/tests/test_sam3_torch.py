"""Controlled CPU tracking and packed-mask selection through native nodes."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    SAM3_TRACK_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sam3 import SAM3_TrackToMask, SAM3_VideoTrack

from dinkster.compose import PackSpec, ServingComposer


@pytest.fixture(autouse=True)
def cpu_model_management(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cpu"))


@pytest.mark.parametrize("multi", (False, True))
def test_tracking_preserves_prompts_masks_options_and_result(multi: bool) -> None:
    images = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4) / 120
    original = images.clone()
    initial = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5) / 30
    embedding = torch.tensor([[[2.0, -3.0], [7.0, 11.0], [13.0, 17.0]]])
    attention = torch.tensor([[1, 0, 1]])
    conditioning = [[embedding, {"attention_mask": attention}]]
    if multi:
        conditioning[0][1] = {
            "sam3_multi_cond": [
                {"cond": embedding, "attention_mask": None, "max_detections": 3},
                {"cond": embedding + 5, "attention_mask": attention},
            ]
        }
    scores = [0.875, 0.25]
    result = {"packed_masks": None, "n_frames": 2, "scores": scores, "extra": {"label": "kept"}}

    def forward_video(**kwargs: Any) -> dict[str, Any]:
        torch.testing.assert_close(kwargs["images"], images[..., :3].movedim(-1, 1), rtol=0, atol=0)
        assert kwargs["initial_masks"].dtype == torch.float64
        torch.testing.assert_close(
            kwargs["initial_masks"], initial.unsqueeze(1).double(), rtol=0, atol=0
        )
        assert kwargs["new_det_thresh"] == 0.375
        assert kwargs["max_objects"] == 7 and kwargs["detect_interval"] == 3
        assert kwargs["target_device"] == torch.device("cpu")
        assert kwargs["target_dtype"] == torch.float64
        prompts = kwargs["text_prompts"]
        assert len(prompts) == (2 if multi else 1)
        torch.testing.assert_close(prompts[0][0], embedding.double(), rtol=0, atol=0)
        torch.testing.assert_close(
            prompts[0][1],
            torch.ones(1, 3, dtype=torch.int64) if multi else attention,
            rtol=0,
            atol=0,
        )
        if multi:
            torch.testing.assert_close(prompts[1][0], (embedding + 5).double(), rtol=0, atol=0)
            torch.testing.assert_close(prompts[1][1], attention, rtol=0, atol=0)
        return result

    model = SimpleNamespace(
        model=SimpleNamespace(
            get_dtype=lambda: torch.float64,
            diffusion_model=SimpleNamespace(forward_video=forward_video),
        )
    )
    output = SAM3_VideoTrack.execute(
        images=images.numpy(),
        model=model,
        initial_mask=initial.numpy(),
        conditioning=conditioning,
        detection_threshold=0.375,
        max_objects=7,
        detect_interval=3,
    )["track_data"]
    assert output is result and output["scores"] is scores
    assert output["orig_size"] == (3, 5) and output["extra"] == {"label": "kept"}
    torch.testing.assert_close(images, original, rtol=0, atol=0)


def test_tracking_requires_a_prompt_but_accepts_empty_initial_mask() -> None:
    calls = []

    def forward_video(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"packed_masks": None, "n_frames": 1}

    model = SimpleNamespace(
        model=SimpleNamespace(
            get_dtype=lambda: torch.float32,
            diffusion_model=SimpleNamespace(forward_video=forward_video),
        )
    )
    images = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError, match="Either initial_mask or conditioning"):
        SAM3_VideoTrack.execute(images=images, model=model, conditioning=[])
    SAM3_VideoTrack.execute(images=images, model=model, initial_mask=torch.empty(0, 2, 3))
    assert len(calls) == 1 and calls[0]["text_prompts"] is None
    assert calls[0]["initial_masks"].shape == (0, 1, 2, 3)
    assert calls[0]["max_objects"] == 0


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ("0", [1, 0, 0, 0, 0, 0, 0, 1]),
        ("2, 0, 2, -1, invalid, 99", [1, 0, 1, 0, 0, 0, 0, 1]),
        ("", [1, 1, 1, 0, 0, 0, 0, 1]),
        ("-1, bad, 99", [0] * 8),
    ],
)
def test_mask_selection_unions_packed_bits_without_mutation(
    indices: str, expected: list[int]
) -> None:
    packed = torch.tensor([[[[129]], [[2]], [[4]]], [[[16]], [[32]], [[64]]]], dtype=torch.uint8)
    original = packed.clone()
    track = {"packed_masks": packed, "orig_size": (1, 8), "n_frames": 2}
    output = SAM3_TrackToMask.execute(track_data=track, object_indices=indices)["masks"]
    assert output.shape == (2, 1, 8) and output.dtype == torch.float32
    torch.testing.assert_close(
        output[0, 0], torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
    )
    second = torch.zeros(8)
    if indices == "0":
        second[4] = 1
    elif indices.startswith("2"):
        second[4] = second[6] = 1
    elif indices == "":
        second[4:7] = 1
    torch.testing.assert_close(output[1, 0], second, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)


def test_mask_resize_and_no_tracks() -> None:
    packed = torch.tensor([[[[1]]]], dtype=torch.uint8)
    output = SAM3_TrackToMask.execute(
        track_data={"packed_masks": packed, "orig_size": (3, 16), "n_frames": 1}
    )["masks"]
    expected = torch.tensor([1, 0.75, 0.25] + [0] * 13, dtype=torch.float32).expand(1, 3, 16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    empty = SAM3_TrackToMask.execute(
        track_data={"packed_masks": None, "orig_size": (3, 7), "n_frames": 2}
    )["masks"]
    assert empty.shape == (2, 3, 7) and torch.count_nonzero(empty) == 0


class ControlledTracker:
    def __init__(self) -> None:
        self.model = SimpleNamespace(get_dtype=lambda: torch.float32, diffusion_model=self)

    def forward_video(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["images"].shape == (2, 3, 3, 8)
        assert kwargs["initial_masks"].shape == (1, 1, 3, 8)
        assert kwargs["max_objects"] == 5 and kwargs["detect_interval"] == 3
        assert kwargs["new_det_thresh"] == 0.375
        prompts = kwargs["text_prompts"]
        assert len(prompts) == 1
        torch.testing.assert_close(
            prompts[0][0], torch.tensor([[[2.0, -3.0], [7.0, 11.0]]]), rtol=0, atol=0
        )
        torch.testing.assert_close(prompts[0][1], torch.tensor([[1, 0]]), rtol=0, atol=0)
        return {
            "packed_masks": torch.tensor([[[[129]], [[2]]], [[[16]], [[32]]]], dtype=torch.uint8),
            "n_frames": 2,
            "scores": [0.875, 0.25],
            "extra": {"label": "retained"},
        }


class TrackingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sam3-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("images", IMAGE_TYPE),
                    ("mask", MASK_TYPE),
                    ("conditioning", CONDITIONING_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_model_gpu = lambda model: None
        management.get_torch_device = lambda: torch.device("cpu")
        return cls.outputs(
            model=ControlledTracker(),
            images=torch.ones(2, 3, 8, 3).numpy(),
            mask=torch.zeros(1, 3, 8).numpy(),
            conditioning=[
                [
                    torch.tensor([[[2.0, -3.0], [7.0, 11.0]]]),
                    {"attention_mask": torch.tensor([[1, 0]])},
                ]
            ],
        )


class TrackingInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sam3-fixture.inspect",
            inputs=(
                InputSpec("tracks", TypeExpr.concrete(SAM3_TRACK_DATA_TYPE)),
                InputSpec("masks", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, tracks: Any, masks: Any) -> dict[str, object]:
        tracks = materialize_latent(tracks)
        masks = materialize_image(masks)
        assert tuple(tracks["orig_size"]) == (3, 8) and tracks["n_frames"] == 2
        assert tracks["scores"] == [0.875, 0.25] and tracks["extra"] == {"label": "retained"}
        assert tracks["packed_masks"].dtype == torch.uint8
        expected = (
            torch.tensor(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
            )
            .unsqueeze(1)
            .expand(2, 3, 8)
        )
        torch.testing.assert_close(masks, expected, rtol=0, atol=0)
        return cls.outputs(summary="portable tracks and selected masks")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_tracks_and_selects_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"sam3_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_sam3_torch import TrackingSource, TrackingInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (TrackingSource, TrackingInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="sam3-fixture"\npublisher="dinkster"\n'
            'namespaces=["sam3-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="sam3-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sam3-fixture.source"),
                    "track": GraphNode(
                        "SAM3_VideoTrack",
                        {
                            "model": Link("source", "model"),
                            "images": Link("source", "images"),
                            "initial_mask": Link("source", "mask"),
                            "conditioning": Link("source", "conditioning"),
                            "detection_threshold": 0.375,
                            "max_objects": 5,
                            "detect_interval": 3,
                        },
                    ),
                    "select": GraphNode(
                        "SAM3_TrackToMask",
                        {"track_data": Link("track", "track_data"), "object_indices": "1, 0"},
                    ),
                    "inspect": GraphNode(
                        "sam3-fixture.inspect",
                        {"tracks": Link("track", "track_data"), "masks": Link("select", "masks")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = (
                    (await engine.run(graph, ["inspect"])).outputs["inspect"]["summary"].resolve()
                )
                assert result == "portable tracks and selected masks"
        finally:
            await composer.close()

    asyncio.run(scenario())
