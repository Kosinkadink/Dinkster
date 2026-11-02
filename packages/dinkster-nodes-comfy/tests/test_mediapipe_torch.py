"""Face drawing pixels, source preservation and ordinary-host cache behavior."""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    FACE_LANDMARKS_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize

from dinkster.compose import PackSpec, ServingComposer


def _packet() -> dict[str, Any]:
    def ring(indices: list[int]) -> frozenset[tuple[int, int]]:
        return frozenset(zip(indices, indices[1:] + indices[:1], strict=True))

    landmarks = np.array(
        [[1, 1], [6, 1], [6, 4], [1, 4], [2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32
    )
    return {
        "frames": [
            [{"landmarks_xy": landmarks}, {"landmarks_xy": landmarks + [4, 0]}],
            [],
            [{"landmarks_xy": landmarks}],
        ],
        "image_size": (7, 11),
        "connection_sets": {
            "face_oval": ring([0, 1, 2, 3]),
            **{part: ring([4, 5, 6, 7]) for part in ("lips", "left_eye", "right_eye", "irises")},
            "left_eyebrow": frozenset({(4, 5)}),
            "right_eyebrow": frozenset({(7, 6)}),
            "nose": frozenset({(0, 6)}),
            "tesselation": frozenset({(0, 2), (1, 3), (0, 99)}),
        },
    }


def _visualize(packet: Any, **options: Any) -> Any:
    return MediaPipeFaceMeshVisualize.execute(
        face_landmarks=packet,
        **{
            "connections": {"connections": "fill"},
            "color": "#d02070",
            "thickness": 2,
            "point_size": 3,
            **options,
        },
    )["IMAGE"]


def _assert_pixels(image: Any, mask: Any) -> None:
    image, mask = materialize_image(image), torch.as_tensor(materialize_latent(mask))
    expected = torch.zeros(3, 7, 11)
    expected[0, 1:5, 1:11] = 1
    expected[2, 1:5, 1:7] = 1
    assert torch.equal(mask, expected)
    assert image.dtype == mask.dtype == torch.float32
    assert image.device.type == mask.device.type == "cpu"
    assert torch.equal(image, expected.unsqueeze(-1) * torch.tensor([208, 32, 112]) / 255)


def test_region_union_fill_and_source_preservation() -> None:
    packet = _packet()
    before = copy.deepcopy(packet)
    mask = MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "all"})["MASK"]
    _assert_pixels(_visualize(packet), mask)
    lips = MediaPipeFaceMask.execute(
        face_landmarks=packet, regions={"regions": "custom", "lips": True}
    )["MASK"]
    expected = torch.zeros(3, 7, 11)
    expected[0, 2:4, 2:4] = expected[0, 2:4, 6:8] = expected[2, 2:4, 2:4] = 1
    assert torch.equal(lips, expected)
    assert torch.equal(
        MediaPipeFaceMask.execute(
            face_landmarks=packet, regions="custom", **{"regions.lips": True}
        )["MASK"],
        expected,
    )
    assert (
        torch.count_nonzero(
            MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "custom"})["MASK"]
        )
        == 0
    )
    for original, actual in zip(before["frames"], packet["frames"], strict=True):
        for a, b in zip(original, actual, strict=True):
            np.testing.assert_array_equal(a["landmarks_xy"], b["landmarks_xy"])


def test_edges_points_background_and_empty_batches() -> None:
    packet = _packet()
    background = torch.linspace(-0.2, 1.2, 4 * 7 * 11 * 4).reshape(4, 7, 11, 4)
    before = background.clone()
    output = _visualize(
        packet,
        image=background,
        thickness=0,
        point_size=0,
        connections={"connections": "all"},
        color="invalid-color",
    )
    expected = before[..., :3].mul(255).add(0.5).clamp(0, 255).to(torch.uint8).float().div(255)
    assert torch.equal(output, expected) and torch.equal(background, before)
    output = _visualize(
        packet,
        image=background,
        thickness=1,
        point_size=0,
        connections={"connections": "custom", "left_eyebrow": True},
        color="invalid-color",
    )
    expected[0, 2, 2:4] = expected[0, 2, 6:8] = expected[2, 2, 2:4] = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(output, expected) and torch.equal(background, before)
    for size in (0, 1, 3):
        output = _visualize(
            packet, thickness=0, point_size=size, connections={"connections": "custom"}
        )
        assert torch.count_nonzero(output[1]) == 0
        if size:
            assert torch.equal(output[2, 1, 1], torch.tensor([208, 32, 112]) / 255)
        else:
            assert torch.count_nonzero(output) == 0
    # The out-of-range tessellation edge is ignored, not indexed.
    assert _visualize(packet, connections={"connections": "custom", "tesselation": True}).shape == (
        3,
        7,
        11,
        3,
    )
    empty = {**packet, "frames": []}
    assert MediaPipeFaceMask.execute(face_landmarks=empty, regions={"regions": "all"})[
        "MASK"
    ].shape == (0, 7, 11)
    assert _visualize(empty).shape == (0, 7, 11, 3)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="face-fixture.source",
            inputs=(),
            outputs=(OutputSpec("faces", TypeExpr.concrete(FACE_LANDMARKS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(faces=_packet())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="face-fixture.inspect",
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any, mask: Any) -> Any:
        _assert_pixels(image, mask)
        return cls.outputs(result="face pixels preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_face_drawing_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    module = f"face_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_mediapipe_torch import Source, Inspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="face-fixture"\npublisher="dinkster"\nnamespaces=["face-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="face", group_manifests=(native, fixture)
                    )
                )
            graph = graph_from_wire(
                graph_to_wire(
                    Graph(
                        nodes={
                            "source": GraphNode("face-fixture.source"),
                            "visualize": GraphNode(
                                "MediaPipeFaceMeshVisualize",
                                {
                                    "face_landmarks": Link("source", "faces"),
                                    "color": "#d02070",
                                },
                                slot_variants={"connections": "fill"},
                            ),
                            "mask": GraphNode(
                                "MediaPipeFaceMask",
                                {"face_landmarks": Link("source", "faces")},
                                slot_variants={"regions": "all"},
                            ),
                            "inspect": GraphNode(
                                "face-fixture.inspect",
                                {"image": Link("visualize", "IMAGE"), "mask": Link("mask", "MASK")},
                            ),
                        }
                    )
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "face pixels preserved"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
