"""CPU behavior and manifest hosting for Cosmos image conditioning."""

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
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference.latent_formats import Wan21
from dinkster_nodes_comfy.cosmos_conditioning import (
    CosmosImageToVideoLatent,
    CosmosPredict2ImageToVideoLatent,
)

from dinkster.compose import PackSpec, ServingComposer


class VAE:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.images.append(image.clone())
        frames = ((image.shape[0] - 1) // 8) + 1
        result = torch.empty((1, 16, frames, image.shape[1] // 8, image.shape[2] // 8))
        for frame in range(frames):
            result[:, :, frame] = image[min(frame * 8, image.shape[0] - 1)].mean()
        return result


def _image(frames: int, offset: float = 0.0) -> torch.Tensor:
    pixels = torch.arange(frames * 16 * 24 * 4, dtype=torch.float32).reshape(frames, 16, 24, 4)
    return pixels / 100.0 + offset


def _execute(
    node: type[CosmosImageToVideoLatent],
    *,
    length: int,
    start_image: torch.Tensor | None = None,
    end_image: torch.Tensor | None = None,
    batch_size: int = 1,
) -> tuple[dict[str, Any], VAE]:
    vae = VAE()
    result = node.execute(
        vae=vae,
        width=24,
        height=16,
        length=length,
        batch_size=batch_size,
        start_image=start_image,
        end_image=end_image,
    )
    return result["LATENT"], vae


def test_cosmos_schema_matches_fixed_reference() -> None:
    for node, defaults, step in (
        (CosmosImageToVideoLatent, (1280, 704, 121), 8),
        (CosmosPredict2ImageToVideoLatent, (848, 480, 93), 4),
    ):
        schema = node.define_schema()
        assert schema.node_type == node.__name__
        assert schema.category == "model/conditioning/cosmos"
        assert [input.id for input in schema.inputs] == [
            "vae",
            "width",
            "height",
            "length",
            "batch_size",
            "start_image",
            "end_image",
        ]
        assert [input.default for input in schema.inputs[1:5]] == [*defaults, 1]
        assert schema.inputs[3].widget.step == step
        assert not schema.inputs[5].required and not schema.inputs[6].required
        assert schema.outputs[0].id == "LATENT"


@pytest.mark.parametrize(
    ("node", "length", "frames"),
    ((CosmosImageToVideoLatent, 17, 3), (CosmosPredict2ImageToVideoLatent, 9, 3)),
)
def test_cosmos_no_image_returns_a_single_unmasked_source_batch(
    node: type[CosmosImageToVideoLatent], length: int, frames: int
) -> None:
    latent, vae = _execute(node, length=length, batch_size=3)
    assert not vae.images
    assert set(latent) == {"samples"}
    assert latent["samples"].shape == (1, 16, frames, 2, 3)
    assert latent["samples"].dtype == torch.get_default_dtype()
    assert latent["samples"].device.type == "cpu"
    assert not latent["samples"].count_nonzero()


def test_cosmos_padding_crops_rgb_before_resize_and_fills_non_saturated_frames() -> None:
    source = _image(1, 2.0)
    before = source.clone()
    _, vae = _execute(CosmosImageToVideoLatent, length=17, start_image=source)
    assert vae.images[0].shape == (9, 16, 24, 3)
    torch.testing.assert_close(vae.images[0][0], source[0, :, :, :3])
    torch.testing.assert_close(vae.images[0][1:], torch.full_like(vae.images[0][1:], 0.5))
    torch.testing.assert_close(source, before)


@pytest.mark.parametrize("length", (1, 9, 17))
def test_cosmos_padding_stays_inside_length_boundaries(length: int) -> None:
    _, vae = _execute(CosmosImageToVideoLatent, length=length, start_image=_image(20))
    assert 1 <= vae.images[0].shape[0] <= length


def test_cosmos_assigns_start_end_and_overlap_with_end_winning() -> None:
    start = _image(17, 1.0)
    end = _image(17, 100.0)
    start_only, _ = _execute(CosmosImageToVideoLatent, length=17, start_image=start, batch_size=2)
    end_only, _ = _execute(CosmosImageToVideoLatent, length=17, end_image=end, batch_size=2)
    both, _ = _execute(
        CosmosImageToVideoLatent, length=17, start_image=start, end_image=end, batch_size=2
    )
    assert start_only["samples"].shape == end_only["samples"].shape == (2, 16, 3, 2, 3)
    assert torch.equal(start_only["samples"][0], start_only["samples"][1])
    assert torch.equal(end_only["samples"][0], end_only["samples"][1])
    assert not start_only["noise_mask"][:, :, :3].count_nonzero()
    assert not end_only["noise_mask"][:, :, -3:].count_nonzero()
    torch.testing.assert_close(both["samples"][:, :, -3:], end_only["samples"][:, :, -3:])
    assert not both["noise_mask"].count_nonzero()


def test_cosmos_start_end_masks_and_samples_are_distinct_and_exact() -> None:
    start = torch.full((1, 16, 24, 4), 2.0)
    end = torch.full((1, 16, 24, 4), 9.0)
    latent, _ = _execute(
        CosmosImageToVideoLatent, length=25, start_image=start, end_image=end, batch_size=2
    )
    expected = torch.zeros((1, 16, 4, 2, 3))
    expected[:, :, 0] = 2.0
    expected[:, :, -1] = 9.0
    expected_mask = torch.ones((1, 1, 4, 2, 3))
    expected_mask[:, :, 0] = 0.0
    expected_mask[:, :, -1] = 0.0
    torch.testing.assert_close(latent["samples"], expected.repeat(2, 1, 1, 1, 1))
    torch.testing.assert_close(latent["noise_mask"], expected_mask.repeat(2, 1, 1, 1, 1))


def test_predict2_normalizes_unconditioned_positions_with_wan21() -> None:
    image = _image(1, 3.0)
    latent, _ = _execute(
        CosmosPredict2ImageToVideoLatent, length=9, start_image=image, batch_size=2
    )
    mask = latent["noise_mask"]
    expected = Wan21().process_out(torch.zeros_like(latent["samples"])) * mask
    torch.testing.assert_close(latent["samples"] * mask, expected)
    assert torch.equal(latent["samples"][0], latent["samples"][1])


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cosmos-conditioning-fixture.source",
            outputs=(OutputSpec("image", _IMAGE), OutputSpec("vae", _VAE)),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=_image(1, 3.0), vae=VAE())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cosmos-conditioning-fixture.inspect",
            inputs=(InputSpec("latent", _LATENT),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> dict[str, object]:
        value = materialize_latent(latent)
        samples = materialize_image(value["samples"])
        assert set(value) == {"samples", "noise_mask"}
        assert samples.shape in ((2, 16, 2, 2, 3), (2, 16, 3, 2, 3))
        assert samples.dtype == torch.get_default_dtype()
        assert samples.device.type == "cpu"
        mask = value["noise_mask"]
        assert mask.shape == (2, 1, samples.shape[2], 2, 3)
        assert mask.dtype == torch.get_default_dtype()
        assert mask.device.type == "cpu"
        source_value = _image(1, 3.0)[..., :3].mean()
        expected = torch.zeros((1, 16, samples.shape[2], 2, 3))
        expected[:, :, 0] = source_value
        expected_mask = torch.ones((1, 1, samples.shape[2], 2, 3))
        expected_mask[:, :, 0] = 0.0
        if samples.shape[2] == 3:
            expected = Wan21().process_out(expected) * expected_mask + expected * (
                1.0 - expected_mask
            )
        torch.testing.assert_close(samples, expected.repeat(2, 1, 1, 1, 1))
        torch.testing.assert_close(mask, expected_mask.repeat(2, 1, 1, 1, 1))
        return cls.outputs(summary="cosmos")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_both_cosmos_nodes_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src"))
                ),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "cosmos_conditioning_fixture.py").write_text(
                "from test_cosmos_conditioning_torch import Inspect, Source\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, Source, Inspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["cosmos-conditioning-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "cosmos_conditioning_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "cosmos_conditioning_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for node_type in ("CosmosImageToVideoLatent", "CosmosPredict2ImageToVideoLatent"):
                graph = Graph(
                    nodes={
                        "source": GraphNode("cosmos-conditioning-fixture.source"),
                        "conditioning": GraphNode(
                            node_type,
                            {
                                "vae": Link("source", "vae"),
                                "start_image": Link("source", "image"),
                                "width": 24,
                                "height": 16,
                                "length": 9,
                                "batch_size": 2,
                            },
                        ),
                        "inspect": GraphNode(
                            "cosmos-conditioning-fixture.inspect",
                            {"latent": Link("conditioning", "LATENT")},
                        ),
                    }
                )
                for _ in range(2):
                    assert (await engine.run(graph, ["inspect"])).outputs["inspect"][
                        "summary"
                    ].resolve() == "cosmos"
        finally:
            await composer.close()

    asyncio.run(scenario())
