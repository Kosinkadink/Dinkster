"""Camera measurements preserve normalized-intrinsics and geometry selection."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    MOGE_GEOMETRY_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moge import MoGeGeometryToFOV
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("axis", "radians"),
    (
        ("horizontal", math.pi / 2),
        ("vertical", 2 * math.atan(0.5)),
        ("diagonal", 2 * math.atan(math.sqrt(1.25))),
    ),
)
@pytest.mark.parametrize("unit", ("degrees", "radians"))
@pytest.mark.parametrize("portable", (False, True))
def test_fov_uses_first_intrinsics_and_image_height(
    axis: str, radians: float, unit: str, portable: bool
) -> None:
    # Hand-derived: fx=.5 -> tan(fov_x/2)=1; fy=1 -> tan(fov_y/2)=.5.
    # The second camera and other geometry heights must not influence either result.
    intrinsics = torch.tensor(
        [[[0.5, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]], [[4.0, 0, 0.5], [0, 8.0, 0.5], [0, 0, 1.0]]]
    )
    geometry = {
        "intrinsics": intrinsics,
        "image": torch.zeros(2, 7, 11, 3),
        "points": torch.zeros(2, 13, 19, 3),
        "depth": torch.zeros(2, 23, 29),
    }
    before = intrinsics.clone()
    if portable:
        geometry = decode_latent(encode_latent(geometry), require_samples=False)
    result = MoGeGeometryToFOV.execute(moge_geometry=geometry, axis=axis, unit=unit)
    assert result["fov"] == pytest.approx(math.degrees(radians) if unit == "degrees" else radians)
    assert result["focal_pixels"] == 7.0
    assert torch.equal(intrinsics, before)


@pytest.mark.parametrize("source", ("points", "depth"))
def test_unbatched_intrinsics_and_height_fallback(source: str) -> None:
    geometry = {
        "intrinsics": torch.tensor([[1.0, 0, 0.5], [0, 0.25, 0.5], [0, 0, 1.0]]),
        source: torch.zeros(1, 12, 17, 3) if source == "points" else torch.zeros(1, 12, 17),
    }
    result = MoGeGeometryToFOV.execute(moge_geometry=geometry, axis="vertical", unit="radians")
    assert result["fov"] == pytest.approx(2 * math.atan(2))
    assert result["focal_pixels"] == 3.0


@pytest.mark.parametrize("geometry", (None, {}, {"intrinsics": None}))
def test_missing_intrinsics(geometry: Any) -> None:
    with pytest.raises(ValueError, match="no intrinsics"):
        MoGeGeometryToFOV.execute(moge_geometry=geometry, axis="vertical", unit="degrees")


def test_missing_geometry_height() -> None:
    with pytest.raises(ValueError, match="no image/points/depth"):
        MoGeGeometryToFOV.execute(
            moge_geometry={"intrinsics": torch.eye(3)}, axis="horizontal", unit="degrees"
        )


class GeometrySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-fov-fixture.source",
            outputs=(OutputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            geometry={
                "intrinsics": torch.tensor(
                    [
                        [[0.5, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]],
                        [[4.0, 0, 0.5], [0, 8.0, 0.5], [0, 0, 1]],
                    ]
                ),
                "image": torch.full((2, 7, 11, 3), 0.25),
                "points": torch.full((2, 13, 19, 3), 0.5),
                "depth": torch.full((2, 23, 29), 2.0),
            }
        )


class FOVObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-fov-fixture.observer",
            inputs=(
                InputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec("fov", TypeExpr.concrete(CORE_FLOAT)),
                InputSpec("focal_pixels", TypeExpr.concrete(CORE_FLOAT)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, geometry: Any, fov: float, focal_pixels: float) -> dict[str, object]:
        assert fov == pytest.approx(96.37937020844281)
        assert focal_pixels == 7.0
        value = materialize_latent(geometry)
        torch.testing.assert_close(
            value["intrinsics"],
            torch.tensor(
                [[[0.5, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], [[4, 0, 0.5], [0, 8, 0.5], [0, 0, 1]]]
            ),
            rtol=0,
            atol=0,
        )
        for key, shape, fill in (
            ("image", (2, 7, 11, 3), 0.25),
            ("points", (2, 13, 19, 3), 0.5),
            ("depth", (2, 23, 29), 2.0),
        ):
            torch.testing.assert_close(value[key], torch.full(shape, fill), rtol=0, atol=0)
        return cls.outputs(result="first camera and image height preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_geometry_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "fov_fixture.py").write_text(
            "from test_moge_torch import GeometrySource, FOVObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(GeometrySource,FOVObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="moge-fov-fixture"\npublisher="dinkster"\n'
            'namespaces=["moge-fov-fixture"]\n[pack.entry]\nnodes="fov_fixture:NODES"\n'
            'types="fov_fixture:register_inference_types"\n',
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
                        worker_group="moge-fov",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("moge-fov-fixture.source"),
                    "fov": GraphNode(
                        "MoGeGeometryToFOV",
                        {
                            "moge_geometry": Link("source", "geometry"),
                            "axis": "diagonal",
                            "unit": "degrees",
                        },
                    ),
                    "observe": GraphNode(
                        "moge-fov-fixture.observer",
                        {
                            "geometry": Link("source", "geometry"),
                            "fov": Link("fov", "fov"),
                            "focal_pixels": Link("fov", "focal_pixels"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observe"])
                assert result.outputs["observe"]["result"].resolve() == (
                    "first camera and image height preserved"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
