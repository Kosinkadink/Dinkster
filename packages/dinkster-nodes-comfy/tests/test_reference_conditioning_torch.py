"""CPU image-edit conditioning and ordinary-manifest transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.reference_conditioning import (
    InstructPixToPixConditioning,
    ReferenceLatent,
)

from dinkster.compose import PackSpec, ServingComposer


class ImageVAE:
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.movedim(-1, 1) * 2.0 - 1.0


def test_image_edit_crops_without_mutating_inputs() -> None:
    pixels = torch.arange(11 * 21 * 4, dtype=torch.float64).reshape(1, 11, 21, 4) / 1024
    before = pixels.clone()
    positive = [[torch.tensor([3.0]), {"strength": 0.7}]]
    negative = [[torch.tensor([-2.0]), {"concat_latent_image": "replace", "strength": 0.2}]]
    result: Any = InstructPixToPixConditioning.execute(
        vae=ImageVAE(), pixels=pixels.numpy(), positive=positive, negative=negative
    )
    expected = pixels[:, 1:9, 2:18, :].movedim(-1, 1) * 2 - 1
    assert torch.equal(result["positive"][0][1]["concat_latent_image"], expected)
    assert torch.equal(result["negative"][0][1]["concat_latent_image"], expected)
    assert result["positive"][0][1]["strength"] == 0.7
    assert result["negative"][0][1]["strength"] == 0.2
    assert result["latent"]["samples"].dtype == torch.float64
    assert torch.equal(result["latent"]["samples"], torch.zeros(1, 4, 8, 16, dtype=torch.float64))
    assert torch.equal(pixels, before)
    assert positive[0][1] == {"strength": 0.7}
    assert negative[0][1]["concat_latent_image"] == "replace"


@pytest.mark.parametrize("existing, count", ((None, 0), ([], 0), ([torch.tensor([5.0])], 1)))
def test_reference_latent_appends_without_mutating_existing_references(
    existing: Any, count: int
) -> None:
    metadata = {"reference_latents": existing, "strength": 0.3}
    conditioning = [[torch.tensor([2.0]), metadata]]
    latent = {"samples": np.array([[[[7.0, 11.0]]]], dtype=np.float64)}
    result: Any = ReferenceLatent.execute(conditioning=conditioning, latent=latent)["CONDITIONING"]
    references = result[0][1]["reference_latents"]
    assert len(references) == count + 1
    assert references[-1].dtype == torch.float64
    assert torch.equal(references[-1], torch.tensor([[[[7.0, 11.0]]]], dtype=torch.float64))
    if existing:
        assert torch.equal(references[0], torch.tensor([5.0]))
    assert result[0][1]["strength"] == 0.3
    assert metadata["reference_latents"] is existing
    assert len(existing or []) == count


class ReferenceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="reference-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("pixels", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[[torch.tensor([1.0]), {"strength": 0.7}]],
            negative=[[torch.tensor([-1.0]), {"strength": 0.2}]],
            vae=ImageVAE(),
            pixels=np.arange(11 * 21 * 4, dtype=np.float32).reshape(1, 11, 21, 4) / 1024,
        )


class MediaConditioningInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="reference-fixture.media-inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("positive", "negative", "area", "source_positive", "source_negative")
            ),
            outputs=(OutputSpec("strength", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        negative: Any,
        area: Any,
        source_positive: Any,
        source_negative: Any,
    ) -> dict[str, object]:
        for result, source, strength, value in (
            (positive, source_positive, 0.7, 1.0),
            (negative, source_negative, 0.2, -1.0),
        ):
            assert torch.equal(torch.as_tensor(result[0][0]), torch.tensor([value]))
            assert result[0][1] == {
                "strength": strength,
                "seconds_start": 2.25,
                "seconds_total": 37.5,
            }
            assert source[0][1] == {"strength": strength}
        assert torch.equal(torch.as_tensor(area[0][0]), torch.tensor([1.0]))
        assert area[0][1] == {
            "seconds_start": 2.25,
            "seconds_total": 37.5,
            "strength": 2.5,
            "area": ("percentage", 0.25, 0.5, 0.75, 0.625, 0.375, 0.125),
            "set_area_to_bounds": False,
        }
        return cls.outputs(strength=area[0][1]["strength"])


class ReferenceInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="reference-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("edited", "references", "negative")
            ),
            outputs=(OutputSpec("pixel", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, edited: Any, references: Any, negative: Any) -> dict[str, object]:
        encoded = torch.as_tensor(edited[0][1]["concat_latent_image"])
        assert encoded.shape == (1, 4, 8, 16)
        latents = references[0][1]["reference_latents"]
        assert len(latents) == 1
        assert torch.equal(torch.as_tensor(latents[0]), torch.zeros(1, 4, 8, 16))
        assert negative[0][1]["strength"] == 0.2
        assert edited[0][1]["strength"] == 0.7
        return cls.outputs(pixel=float(encoded[0, 0, 0, 0]))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_links_image_edit_and_reference_nodes(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src")))}
        )
        try:
            (tmp_path / "reference_fixture.py").write_text(
                "from test_reference_conditioning_torch import ReferenceSource, ReferenceInspect\n"
                "from test_reference_conditioning_torch import MediaConditioningInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ReferenceSource, ReferenceInspect,\n"
                "         MediaConditioningInspect)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["reference-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "reference_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "reference_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("reference-fixture.source"),
                    "timed": GraphNode(
                        "ConditioningStableAudio",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "seconds_start": 2.25,
                            "seconds_total": 37.5,
                        },
                    ),
                    "area": GraphNode(
                        "ConditioningSetAreaPercentageVideo",
                        {
                            "conditioning": Link("timed", "positive"),
                            "width": 0.75,
                            "height": 0.5,
                            "temporal": 0.25,
                            "x": 0.125,
                            "y": 0.375,
                            "z": 0.625,
                            "strength": 2.5,
                        },
                    ),
                    "media_inspect": GraphNode(
                        "reference-fixture.media-inspect",
                        {
                            "positive": Link("timed", "positive"),
                            "negative": Link("timed", "negative"),
                            "area": Link("area", "CONDITIONING"),
                            "source_positive": Link("source", "positive"),
                            "source_negative": Link("source", "negative"),
                        },
                    ),
                    "edit": GraphNode(
                        "InstructPixToPixConditioning",
                        {
                            name: Link("source", name)
                            for name in ("positive", "negative", "vae", "pixels")
                        },
                    ),
                    "reference": GraphNode(
                        "ReferenceLatent",
                        {
                            "conditioning": Link("edit", "positive"),
                            "latent": Link("edit", "latent"),
                        },
                    ),
                    "inspect": GraphNode(
                        "reference-fixture.inspect",
                        {
                            "edited": Link("edit", "positive"),
                            "references": Link("reference", "CONDITIONING"),
                            "negative": Link("edit", "negative"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect", "media_inspect"])
                assert result.outputs["inspect"]["pixel"].resolve() == -105 / 128
                assert result.outputs["media_inspect"]["strength"].resolve() == 2.5
        finally:
            await composer.close()

    asyncio.run(scenario())
