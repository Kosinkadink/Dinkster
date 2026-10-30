"""Public CPU image dimension and rebatch behavior."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_engine import EngineEvent
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_comfy.rebatch import RebatchImages, RebatchLatents

from dinkster.compose import PackSpec, ServingComposer


def test_empty_image_initializes_cpu_policy_in_a_cold_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_nodes_comfy.image_core import EmptyImage\n"
            "assert 'dinkster_inference.model_management' not in sys.modules\n"
            "image = EmptyImage.execute(\n"
            "    width=73, height=41, batch_size=2, color=0x32B76D)['IMAGE']\n"
            "assert image.device.type == 'cpu'\n"
            "assert tuple(image.shape) == (2, 41, 73, 3)\n"
            "import torch\n"
            "expected = torch.tensor([50 / 255, 183 / 255, 109 / 255], dtype=image.dtype)\n"
            "torch.testing.assert_close(image, expected.expand_as(image), rtol=0, atol=0)\n",
        ],
        env={**os.environ, "DINKSTER_ACCELERATOR": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rebatch_preserves_frame_order_and_latent_indices() -> None:
    first = torch.arange(2 * 3 * 5 * 3, dtype=torch.float32).reshape(2, 3, 5, 3)
    second = torch.full((1, 3, 5, 3), 123.0)
    images = RebatchImages.execute(images=[first, second], batch_size=2)["IMAGE"]
    assert len(images) == 2
    assert torch.equal(images[0], first)
    assert torch.equal(images[1], second)

    samples = torch.arange(3 * 4 * 2 * 5, dtype=torch.float32).reshape(3, 4, 2, 5)
    latents = RebatchLatents.execute(
        latents=[{"samples": samples, "batch_index": [8, 3, 11]}], batch_size=2
    )["LATENT"]
    assert [item["batch_index"] for item in latents] == [[8, 3], [11]]
    assert torch.equal(latents[0]["samples"], samples[:2])
    assert torch.equal(latents[1]["samples"], samples[2:])
    assert all("noise_mask" not in item for item in latents)

    mask = torch.full((1, 1, 2, 3), 0.25)
    masked = RebatchLatents.execute(
        latents=[{"samples": torch.zeros((2, 4, 2, 3)), "noise_mask": mask}], batch_size=2
    )["LATENT"]
    assert torch.equal(masked[0]["noise_mask"], mask.repeat(2, 1, 1, 1))
    assert mask.shape == (1, 1, 2, 3)


class DimensionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-dimension-fixture.source",
            inputs=(InputSpec("video", TypeExpr.concrete(CORE_BOOLEAN)),),
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, video: bool) -> Any:
        return cls.outputs(
            image=torch.zeros((2, 3, 7, 4)),
            latent={"samples": torch.zeros((3, 4, 5, 6, 9) if video else (3, 4, 6, 9))},
        )


@pytest.mark.parametrize("in_process", (True,))
@pytest.mark.parametrize("video", (False, True))
def test_image_dimensions_ordinary_manifest_twice(
    tmp_path: Path, in_process: bool, video: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = f"dimension_fixture_{in_process}_{video}"
            (tmp_path / f"{module}.py").write_text(
                "from test_image_batch_torch import DimensionSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, DimensionSource)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["image-dimension-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    f"{module}:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("image-dimension-fixture.source", {"video": video}),
                    "image": GraphNode("GetImageSize", {"image": Link("source", "image")}),
                    "latent": GraphNode("GetLatentSize", {"latent": Link("source", "latent")}),
                }
            )
            events: list[EngineEvent] = []
            for _ in range(2):
                events.clear()
                engine = composer.composition.make_engine(events.append)
                result = await engine.run(graph, ["image", "latent"])
                assert {
                    name: {key: value.resolve() for key, value in outputs.items()}
                    for name, outputs in result.outputs.items()
                } == {
                    "image": {"width": 7, "height": 3, "batch_size": 2},
                    "latent": {
                        "width": 9,
                        "height": 6,
                        "frames": 5 if video else 1,
                        "batch_size": 3,
                    },
                }
                reports = [event for event in events if event.kind == "node_event"]
                assert {
                    event.node_id: {
                        "name": event.detail["name"],
                        "data": event.detail["data"],
                    }
                    for event in reports
                } == {
                    "image": {
                        "name": "progress",
                        "data": {
                            "step": 1,
                            "total": 1,
                            "text": "width: 7, height: 3\n batch size: 2",
                        },
                    },
                    "latent": {
                        "name": "progress",
                        "data": {
                            "step": 1,
                            "total": 1,
                            "text": (
                                f"width: 9, height: 6\n frames: {5 if video else 1}, batch size: 3"
                            ),
                        },
                    },
                }
                assert len(reports) == 2
                assert all(event.run_id == result.run_id for event in reports)
        finally:
            await composer.close()

    asyncio.run(scenario())


class RebatchSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "rebatch-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))),
                OutputSpec("latents", TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))),
                OutputSpec("sizes", TypeExpr.list_of(TypeExpr.concrete(CORE_INT))),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(
            images=[torch.zeros((1, 3, 7, 3)), torch.ones((2, 3, 7, 3))],
            latents=[
                {
                    "samples": torch.zeros((1, 4, 2, 5)),
                    "batch_index": [8],
                },
                {
                    "samples": torch.ones((2, 4, 2, 5)),
                    "batch_index": [3, 11],
                },
            ],
            sizes=[2, 1],
        )


@pytest.mark.parametrize("in_process", (True,))
def test_rebatch_ordinary_manifest_with_explicit_selection_and_map(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "rebatch_fixture.py").write_text(
                "from test_image_batch_torch import RebatchSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, RebatchSource)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["rebatch-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "rebatch_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "rebatch_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "rebatch_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(
                PackSpec(
                    manifest=manifest.parent.parent
                    / "dinkster-nodes-foundation/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("rebatch-fixture.source"),
                    "size": GraphNode(
                        "std.list.element", {"list": Link("source", "sizes"), "index": 0}
                    ),
                    "images": GraphNode(
                        "RebatchImages",
                        {"images": Link("source", "images"), "batch_size": Link("size", "item")},
                    ),
                    "latents": GraphNode(
                        "RebatchLatents",
                        {"latents": Link("source", "latents"), "batch_size": Link("size", "item")},
                    ),
                    "per_image": RegionNode(
                        kind="map",
                        body=Graph(
                            nodes={
                                "size": GraphNode(
                                    "GetImageSize", {"image": Link(PORTS_NODE_ID, "image")}
                                )
                            }
                        ),
                        ports={"image": TypeExpr.concrete(IMAGE_TYPE)},
                        inputs={"image": Link("images", "IMAGE")},
                        element_ports=("image",),
                        outputs={"sizes": RegionOutput(Link("size", "batch_size"))},
                    ),
                    "per_latent": RegionNode(
                        kind="map",
                        body=Graph(
                            nodes={
                                "size": GraphNode(
                                    "GetLatentSize", {"latent": Link(PORTS_NODE_ID, "latent")}
                                )
                            }
                        ),
                        ports={"latent": TypeExpr.concrete(LATENT_TYPE)},
                        inputs={"latent": Link("latents", "LATENT")},
                        element_ports=("latent",),
                        outputs={"sizes": RegionOutput(Link("size", "batch_size"))},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["per_image", "per_latent"])
                assert result.outputs["per_image"]["sizes"].resolve() == [2, 1]
                assert result.outputs["per_latent"]["sizes"].resolve() == [2, 1]
        finally:
            await composer.close()

    asyncio.run(scenario())
