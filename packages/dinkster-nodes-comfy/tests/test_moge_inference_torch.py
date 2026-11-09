"""Chunked geometry output, image boundaries and optional fields."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    IMAGE_TYPE,
    MOGE_GEOMETRY_TYPE,
    MOGE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moge_inference import MoGeInference
from dinkster_values import decode_image_array, encode_image_array

from dinkster.compose import PackSpec, ServingComposer


class GeometryModel:
    def __init__(
        self, expected_options: dict[str, Any], partial: bool, dtype: torch.dtype = torch.float64
    ) -> None:
        self.options = expected_options
        self.partial = partial
        self.dtype = dtype

    def infer(self, pixels: Any, **options: Any) -> Any:
        assert options == self.options
        assert pixels.is_contiguous() and pixels.shape[1:] == (3, 3, 7)
        assert 0 < len(pixels) <= 2
        assert pixels.dtype == self.dtype
        # Each channel carries a distinct offset; alpha must never enter inference.
        red, green, blue = pixels.unbind(1)
        torch.testing.assert_close(green, red + 10)
        torch.testing.assert_close(blue, red + 20)
        ids = red[:, 0, 0]
        packet: dict[str, Any] = {"depth": red + 100, "points": pixels.movedim(1, -1)}
        if not self.partial:
            packet.update(
                intrinsics=ids[:, None, None].expand(-1, 3, 3).clone(),
                mask=red > 2,
                normal=pixels.movedim(1, -1) * -1,
            )
        elif ids[0] < 2:
            packet["mask"] = red > 2
        return packet


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("fov", (-5.0, 0.0, 71.5))
@pytest.mark.parametrize("partial", (False, True))
def test_chunked_rgb_order_options_optional_fields_and_nonmutation(
    portable: bool, fov: float, partial: bool
) -> None:
    # Hand-derived: five image IDs, three different channel offsets, ignored
    # alpha 99. A two-image chunk size leaves an asymmetric one-image tail.
    ids = torch.arange(5.0, dtype=torch.float64)
    pixels = torch.stack((ids, ids + 10, ids + 20, ids + 99), -1)[:, None, None, :]
    pixels = pixels.expand(5, 3, 7, 4).clone()
    before = pixels.clone()
    image: Any = pixels
    if portable:
        image = decode_image_array(encode_image_array(pixels))
    model = GeometryModel(
        {
            "resolution_level": 6,
            "fov_x": None if fov <= 0 else fov,
            "force_projection": False,
            "apply_mask": True,
            "refine_steps": 2,
        },
        partial,
    )
    result: Any = MoGeInference.execute(
        moge_model=model,
        image=image,
        resolution_level=6,
        fov_x_degrees=fov,
        batch_size=2,
        force_projection=False,
        apply_mask=True,
        refine_steps=2,
    )
    geometry = result["moge_geometry"]
    rgb = before[..., :3]
    torch.testing.assert_close(geometry["image"], rgb)
    torch.testing.assert_close(geometry["points"], rgb)
    torch.testing.assert_close(geometry["depth"], rgb[..., 0] + 100)
    assert geometry["image"].device.type == "cpu"
    assert geometry["depth"].dtype == torch.float64
    if partial:
        assert set(geometry) == {"image", "points", "depth", "mask"}
        # F concatenates only chunks that supplied each optional field.
        torch.testing.assert_close(geometry["mask"], rgb[:2, ..., 0] > 2)
    else:
        torch.testing.assert_close(geometry["intrinsics"], ids[:, None, None].expand(5, 3, 3))
        torch.testing.assert_close(geometry["mask"], rgb[..., 0] > 2)
        torch.testing.assert_close(geometry["normal"], -rgb)
    torch.testing.assert_close(pixels, before)


def test_defaults_empty_predictions_and_numpy_input() -> None:
    image = torch.full((1, 3, 7, 4), 0.75, dtype=torch.float32).numpy()

    class NoGeometry:
        def infer(self, pixels: Any, **options: Any) -> Any:
            assert options == {
                "resolution_level": 9,
                "fov_x": None,
                "force_projection": True,
                "apply_mask": True,
                "refine_steps": 3,
            }
            torch.testing.assert_close(pixels, torch.full((1, 3, 3, 7), 0.75))
            return {"ignored": torch.tensor([99])}

    result: Any = MoGeInference.execute(moge_model=NoGeometry(), image=image)
    assert set(result["moge_geometry"]) == {"image"}
    torch.testing.assert_close(result["moge_geometry"]["image"], materialize_image(image)[..., :3])


class InferenceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-inference-fixture.source",
            inputs=(InputSpec("fov", TypeExpr.concrete(CORE_FLOAT)),),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MOGE_MODEL_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, fov: float) -> dict[str, object]:
        ids = torch.arange(5.0)
        pixels = torch.stack((ids, ids + 10, ids + 20, ids + 99), -1)
        return cls.outputs(
            image=pixels[:, None, None, :].expand(5, 3, 7, 4).clone(),
            model=GeometryModel(
                {
                    "resolution_level": 6,
                    "fov_x": None if fov == 0 else fov,
                    "force_projection": False,
                    "apply_mask": True,
                    "refine_steps": 2,
                },
                partial=False,
                dtype=torch.float32,
            ),
        )


class GeometryObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "moge-inference-fixture.observer",
            inputs=(
                InputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec("source", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, geometry: Any, source: Any) -> dict[str, object]:
        value = materialize_latent(geometry)
        original = materialize_image(source)
        ids = torch.arange(5.0)
        rgb = torch.stack((ids, ids + 10, ids + 20), -1)[:, None, None, :].expand(5, 3, 7, 3)
        assert set(value) == {"image", "points", "depth", "intrinsics", "mask", "normal"}
        torch.testing.assert_close(value["image"], rgb, rtol=0, atol=0)
        torch.testing.assert_close(value["points"], rgb, rtol=0, atol=0)
        torch.testing.assert_close(value["normal"], -rgb, rtol=0, atol=0)
        torch.testing.assert_close(value["depth"], rgb[..., 0] + 100, rtol=0, atol=0)
        torch.testing.assert_close(value["intrinsics"], ids[:, None, None].expand(5, 3, 3))
        assert torch.equal(value["mask"], rgb[..., 0] > 2)
        torch.testing.assert_close(original[..., :3], rgb, rtol=0, atol=0)
        torch.testing.assert_close(
            original[..., 3], (ids + 99)[:, None, None].expand(5, 3, 7), rtol=0, atol=0
        )
        return cls.outputs(result="five ordered geometry frames and unchanged RGBA source")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_model_geometry_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "inference_fixture.py").write_text(
            "from test_moge_inference_torch import InferenceSource, GeometryObserver\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(InferenceSource,GeometryObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="moge-inference-fixture"\npublisher="dinkster"\n'
            'namespaces=["moge-inference-fixture"]\n[pack.entry]\nnodes="inference_fixture:NODES"\n'
            'types="inference_fixture:register_inference_types"\n',
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
                        worker_group="moge-inference",
                        group_manifests=(native, fixture),
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for fov in (0.0, 71.5):
                graph = Graph(
                    nodes={
                        "source": GraphNode("moge-inference-fixture.source", {"fov": fov}),
                        "infer": GraphNode(
                            "MoGeInference",
                            {
                                "moge_model": Link("source", "model"),
                                "image": Link("source", "image"),
                                "resolution_level": 6,
                                "fov_x_degrees": fov,
                                "batch_size": 2,
                                "force_projection": False,
                                "apply_mask": True,
                                "refine_steps": 2,
                            },
                        ),
                        "observe": GraphNode(
                            "moge-inference-fixture.observer",
                            {
                                "geometry": Link("infer", "moge_geometry"),
                                "source": Link("source", "image"),
                            },
                        ),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["observe"])
                    assert result.outputs["observe"]["result"].resolve() == (
                        "five ordered geometry frames and unchanged RGBA source"
                    )
        finally:
            await composer.close()

    asyncio.run(scenario())
