"""Observable Gaussian rasterization and portable ordinary-host execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import SPLAT_TYPE, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.splat_render import RenderSplat

from dinkster.compose import PackSpec, ServingComposer


def source() -> SimpleNamespace:
    return SimpleNamespace(
        positions=torch.tensor(
            [[[0, 0, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0]] * 3], dtype=torch.float64
        ),
        scales=torch.tensor([[[0.18, 0.07, 0.03]] * 3] * 2, dtype=torch.float64),
        rotations=torch.tensor([[[1, 0, 0, 0]] * 3] * 2, dtype=torch.float64),
        opacities=torch.tensor([[[0.6], [0.3], [1]], [[1], [1], [1]]], dtype=torch.float64),
        sh=torch.zeros(2, 3, 1, 3, dtype=torch.float64),
        counts=torch.tensor([2, 0], dtype=torch.int32),
    )


def parameters() -> dict:
    return dict(
        width=80,
        height=64,
        frames=1,
        splat_scale=1.0,
        sharpen=2.0,
        headlight_shading=0.0,
        opacity_threshold=0.0,
        render_style="color",
        background="#204080",
        camera_info={
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "cameraType": "perspective",
            "zoom": 1.0,
            "fov": 35.0,
        },
    )


@pytest.mark.parametrize("style", ["color", "clay", "depth", "normal"])
@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_render_splat_counts_coverage_and_shading(style: str, projection: str) -> None:
    value = source()
    before = {key: tensor.clone() for key, tensor in vars(value).items()}
    args = parameters()
    args["render_style"] = style
    args["camera_info"]["cameraType"] = projection
    result = RenderSplat.execute(splat=value, **args)
    image, mask = result["IMAGE"], result["MASK"]
    assert image.shape == (2, 64, 80, 3) and mask.shape == (2, 64, 80)
    assert image.device.type == mask.device.type == "cpu"
    assert image.dtype == mask.dtype == torch.float32
    assert torch.isfinite(image).all() and torch.isfinite(mask).all()
    # Both centered Gaussians have q=0 at this pixel: 1-(1-.6)*(1-.3).
    torch.testing.assert_close(mask[0, 32, 40], torch.tensor(0.72), rtol=0, atol=2e-6)
    assert not mask[1].any()
    if style == "color":
        torch.testing.assert_close(image[1, 0, 0], torch.tensor([32, 64, 128]) / 255)
    else:
        assert not image[1].any()
    if style == "normal":
        torch.testing.assert_close(
            image[0, 32, 40], torch.tensor([0.5, 0.5, 1.0]), rtol=0, atol=1e-6
        )
    assert mask[0, 32, 43] > mask[0, 35, 40]
    for key, tensor in before.items():
        assert torch.equal(getattr(value, key), tensor)


def test_render_splat_signed_frames_background_cycle_and_culling() -> None:
    value = source()
    value.positions[0, 0] = torch.tensor([0.2, -0.1, 0.3])
    args = parameters()
    args.update(
        frames=3, bg_image=torch.tensor([[[[0.1, 0.2, 0.3, 0.9]]], [[[0.7, 0.4, 0.2, 0.1]]]])
    )
    positive = RenderSplat.execute(splat=value, **args)
    args["frames"] = -3
    negative = RenderSplat.execute(splat=value, **args)
    # The first camera is unchanged; subsequent signed orbits visit opposite views.
    torch.testing.assert_close(positive["MASK"][0], negative["MASK"][0], rtol=0, atol=0)
    torch.testing.assert_close(positive["MASK"][1], negative["MASK"][2], rtol=1e-5, atol=1e-5)
    assert not torch.equal(positive["MASK"][1], negative["MASK"][1])
    assert not positive["MASK"][3:].any()
    for index, expected in zip(
        (3, 4, 5), ([0.7, 0.4, 0.2], [0.1, 0.2, 0.3], [0.7, 0.4, 0.2]), strict=True
    ):
        torch.testing.assert_close(positive["IMAGE"][index, 0, 0], torch.tensor(expected))
    args.update(frames=0, opacity_threshold=0.61, camera_info=None, bg_image=None)
    culled = RenderSplat.execute(splat=value, **args)
    assert culled["IMAGE"].shape[0] == 2 and not culled["MASK"].any()


class SplatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-render-source.make",
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict:
        return cls.outputs(splat=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_render_splat_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    module = "splat_render_source"
    (tmp_path / f"{module}.py").write_text(
        "from test_render_splat_torch import SplatSource\n"
        "from dinkster_nodes_comfy import register_inference_types\nNODES=(SplatSource,)\n"
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="splat-render-source"\npublisher="dinkster"\n'
        'namespaces=["splat-render-source"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n'
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "DINKSTER_ACCELERATOR": "cpu",
        "PYTHONPATH": os.pathsep.join(
            (str(tmp_path), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
        ),
    }

    async def scenario() -> None:
        composer = ServingComposer(worker_env=environment)
        try:
            for path in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", manifest):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            args = parameters()
            args.pop("camera_info")
            graph = Graph(
                {
                    "source": GraphNode("splat-render-source.make", {}),
                    "render": GraphNode("RenderSplat", {"splat": Link("source", "splat"), **args}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["render"])
                assert result.outputs["render"]["IMAGE"].resolve().shape == (2, 64, 80, 3)
                mask = torch.as_tensor(result.outputs["render"]["MASK"].resolve())
                assert mask[0].max() > 0 and not mask[1].any()
        finally:
            await composer.close()

    asyncio.run(scenario())
