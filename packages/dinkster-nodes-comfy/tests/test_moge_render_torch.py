"""Asymmetric geometry visualization and portable tensor consumption."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MOGE_GEOMETRY_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moge_render import MoGeRender
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


def portable_geometry(geometry: Any, portable: bool) -> Any:
    return decode_latent(encode_latent(geometry), require_samples=False) if portable else geometry


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("output", ("normal_opengl", "normal_directx", "mask"))
def test_predicted_normals_mask_and_source_preservation(portable: bool, output: str) -> None:
    normals = torch.tensor(
        [[[[2.0, -0.6, 0.8], [-0.2, 0.4, -2.0]]], [[[0.0, 1.0, -1.0], [0.6, -0.8, 0.0]]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[[False, True]], [[True, False]]])
    geometry = {"normal": normals, "mask": mask, "points": torch.full_like(normals, float("nan"))}
    before = normals.clone()
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry(geometry, portable), output=output
    )
    # Hand-derived channel signs: OpenGL flips Y, both conventions flip Z.
    expected = {
        "normal_opengl": [
            [[[1.0, 0.8, 0.1], [0.4, 0.3, 1.0]]],
            [[[0.5, 0.0, 1.0], [0.8, 0.9, 0.5]]],
        ],
        "normal_directx": [
            [[[1.0, 0.2, 0.1], [0.4, 0.7, 1.0]]],
            [[[0.5, 1.0, 1.0], [0.8, 0.1, 0.5]]],
        ],
        "mask": [[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]],
    }
    torch.testing.assert_close(result["IMAGE"], torch.tensor(expected[output]))
    assert result["IMAGE"].dtype == torch.float32 and result["IMAGE"].device.type == "cpu"
    torch.testing.assert_close(normals, before)


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("output", ("depth", "depth_colored"))
def test_depth_percentiles_batch_and_invalid_pixels(portable: bool, output: str) -> None:
    depth = torch.tensor(
        [
            [[1.0, 2.0, 4.0, 0.0, -1.0, float("inf"), float("nan")]],
            [[2.0, 4.0, 8.0, 0.0, -1.0, float("inf"), float("nan")]],
        ],
        dtype=torch.float64,
    )
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry({"depth": depth}, portable), output=output
    )
    if output == "depth":
        # Three valid disparities [1, .5, .25]: q.001=.2505, q.999=.999.
        middle = (0.5 - 0.2505) / (0.999 - 0.2505)
        expected = torch.tensor([1.0, middle, 0.0, 0.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(
            result["IMAGE"], expected[None, None, :, None].expand(2, 1, 7, 3)
        )
    else:
        # Turbo's polynomial constant is the black-end color for invalid/zero disparity.
        torch.testing.assert_close(
            result["IMAGE"][:, :, 2:],
            torch.tensor([0.13572138, 0.09140261, 0.10667330]).expand(2, 1, 5, 3),
        )
        assert torch.isfinite(result["IMAGE"]).all()
        torch.testing.assert_close(result["IMAGE"][0], result["IMAGE"][1])
    torch.testing.assert_close(
        depth[:, :, :5],
        torch.tensor(
            [[[1.0, 2.0, 4.0, 0.0, -1.0]], [[2.0, 4.0, 8.0, 0.0, -1.0]]], dtype=torch.float64
        ),
    )


@pytest.mark.parametrize("portable", (False, True))
def test_all_invalid_depth_renders_black(portable: bool) -> None:
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry(
            {"depth": torch.tensor([[[0.0, -1.0, float("nan")]]])}, portable
        )
    )
    assert torch.count_nonzero(result["IMAGE"]) == 0


@pytest.mark.parametrize("output", ("normal_opengl", "normal_directx"))
def test_point_normal_fallback_cross_orientation_and_invalid_mask(output: str) -> None:
    # Hand-derived plane z=x+2y: dy x dx gives (1,2,-1)/sqrt(6).
    yy, xx = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
    points = torch.stack((xx, yy, xx + 2 * yy), dim=-1)[None]
    points[0, 0, 0] = float("nan")
    result: Any = MoGeRender.execute(moge_geometry={"points": points}, output=output)
    sign = -1 if output == "normal_opengl" else 1
    expected = (torch.tensor([1.0, 2.0 * sign, 1.0]) / (6**0.5)) * 0.5 + 0.5
    torch.testing.assert_close(result["IMAGE"][0, 2, 2], expected)
    torch.testing.assert_close(result["IMAGE"][0, 0, 0], torch.full((3,), 0.5))


@pytest.mark.parametrize(
    "output, message",
    [
        ("depth", "has no depth output"),
        ("depth_colored", "has no depth output"),
        ("normal_opengl", "has neither normals nor points"),
        ("normal_directx", "has neither normals nor points"),
        ("mask", "has no mask output"),
        ("unknown", "Unknown output mode: unknown"),
    ],
)
def test_missing_fields_and_unknown_mode(output: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MoGeRender.execute(moge_geometry={}, output=output)


class RenderGeometrySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-render-fixture.source",
            outputs=(OutputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            geometry={
                "normal": torch.tensor(
                    [
                        [[[2.0, -0.6, 0.8], [-0.2, 0.4, -2.0]]],
                        [[[0.0, 1.0, -1.0], [0.6, -0.8, 0.0]]],
                    ],
                    dtype=torch.float64,
                ),
                "mask": torch.tensor([[[False, True]], [[True, False]]]),
                "depth": torch.tensor([[[1.0, 2.0, 4.0, 0.0]], [[2.0, 4.0, 8.0, 0.0]]]),
            }
        )


class RenderImageObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-render-fixture.observer",
            inputs=(
                InputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mode", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, geometry: Any, image: Any, mode: str) -> dict[str, object]:
        pixels = materialize_image(image)
        assert pixels.dtype == torch.float32 and pixels.device.type == "cpu"
        expected = {
            "normal_opengl": [
                [[[1.0, 0.8, 0.1], [0.4, 0.3, 1.0]]],
                [[[0.5, 0.0, 1.0], [0.8, 0.9, 0.5]]],
            ],
            "normal_directx": [
                [[[1.0, 0.2, 0.1], [0.4, 0.7, 1.0]]],
                [[[0.5, 1.0, 1.0], [0.8, 0.1, 0.5]]],
            ],
            "mask": [[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]],
        }
        if mode in expected:
            torch.testing.assert_close(pixels, torch.tensor(expected[mode]))
        elif mode == "depth":
            middle = (0.5 - 0.2505) / (0.999 - 0.2505)
            values = torch.tensor([1.0, middle, 0.0, 0.0])
            torch.testing.assert_close(pixels, values[None, None, :, None].expand(2, 1, 4, 3))
        else:
            assert mode == "depth_colored" and pixels.shape == (2, 1, 4, 3)
            assert torch.isfinite(pixels).all()
            torch.testing.assert_close(pixels[0], pixels[1])
            torch.testing.assert_close(
                pixels[:, :, 2:],
                torch.tensor([0.13572138, 0.09140261, 0.10667330]).expand(2, 1, 2, 3),
            )
            assert not torch.allclose(pixels[:, :, 0], pixels[:, :, 2])
        original = materialize_latent(geometry)
        torch.testing.assert_close(
            original["normal"],
            torch.tensor(
                [[[[2.0, -0.6, 0.8], [-0.2, 0.4, -2.0]]], [[[0.0, 1.0, -1.0], [0.6, -0.8, 0.0]]]],
                dtype=torch.float64,
            ),
            rtol=0,
            atol=0,
        )
        assert torch.equal(original["mask"], torch.tensor([[[False, True]], [[True, False]]]))
        torch.testing.assert_close(
            original["depth"],
            torch.tensor([[[1.0, 2.0, 4.0, 0.0]], [[2.0, 4.0, 8.0, 0.0]]]),
            rtol=0,
            atol=0,
        )
        return cls.outputs(result=mode)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_geometry_hosts_all_modes_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "render_fixture.py").write_text(
            "from test_moge_render_torch import RenderGeometrySource, RenderImageObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(RenderGeometrySource,RenderImageObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="moge-render-fixture"\npublisher="dinkster"\n'
            'namespaces=["moge-render-fixture"]\n[pack.entry]\nnodes="render_fixture:NODES"\n'
            'types="render_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="moge-render",
                        group_manifests=(native, fixture),
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for mode in ("depth", "depth_colored", "normal_opengl", "normal_directx", "mask"):
                graph = Graph(
                    nodes={
                        "source": GraphNode("moge-render-fixture.source"),
                        "render": GraphNode(
                            "MoGeRender",
                            {
                                "moge_geometry": Link("source", "geometry"),
                                "output": mode,
                            },
                        ),
                        "observe": GraphNode(
                            "moge-render-fixture.observer",
                            {
                                "geometry": Link("source", "geometry"),
                                "image": Link("render", "IMAGE"),
                                "mode": mode,
                            },
                        ),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["observe"])
                    assert result.outputs["observe"]["result"].resolve() == mode
        finally:
            await composer.close()

    asyncio.run(scenario())
