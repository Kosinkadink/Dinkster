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
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference.utils import common_upscale
from dinkster_nodes_comfy.reference_conditioning import (
    InstructPixToPixConditioning,
    ReferenceLatent,
    ReferenceMultiLatent,
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


def test_reference_multi_latent_orders_resizes_and_preserves_inputs() -> None:
    class RecordingVAE(ImageVAE):
        def __init__(self) -> None:
            self.images: list[torch.Tensor] = []

        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            self.images.append(pixels.clone())
            return super().encode(pixels)

    image2 = torch.arange(3 * 5 * 3, dtype=torch.float64).reshape(1, 3, 5, 3) / 45
    image10 = torch.arange(5 * 3 * 3, dtype=torch.float64).reshape(1, 5, 3, 3) / 30
    image2_before, image10_before = image2.clone(), image10.clone()
    existing = torch.tensor([7.0])
    positive = [[torch.tensor([2.0]), {"reference_latents": [existing], "strength": 0.3}]]
    negative = [[torch.tensor([-2.0]), {"reference_latents": None, "strength": 0.2}]]
    vae = RecordingVAE()

    result: Any = ReferenceMultiLatent.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        images={"image10": image10.numpy(), "image2": image2.numpy(), "image3": None},
        resolution="resize",
        **{"resolution.megapixels": 0.01},
    )

    assert [image.shape for image in vae.images] == [(1, 79, 132, 3), (1, 132, 79, 3)]
    positive_references = result["positive"][0][1]["reference_latents"]
    negative_references = result["negative"][0][1]["reference_latents"]
    assert len(positive_references) == 3
    assert len(negative_references) == 2
    assert positive_references[0] is existing
    torch.testing.assert_close(positive_references[1], vae.images[0].movedim(-1, 1) * 2 - 1)
    torch.testing.assert_close(positive_references[2], vae.images[1].movedim(-1, 1) * 2 - 1)
    torch.testing.assert_close(
        result["empty_latent"]["samples"], torch.zeros_like(positive_references[1])
    )
    torch.testing.assert_close(image2, image2_before)
    torch.testing.assert_close(image10, image10_before)
    assert positive[0][1] == {"reference_latents": [existing], "strength": 0.3}
    assert negative[0][1] == {"reference_latents": None, "strength": 0.2}


def test_reference_multi_latent_without_vae_or_images_returns_empty_latent() -> None:
    positive = [[torch.tensor([2.0]), {"strength": 0.3}]]
    result: Any = ReferenceMultiLatent.execute(
        positive=positive,
        resolution={"resolution": "resize", "megapixels": 0.01},
    )

    assert result["negative"] is result["positive"]
    assert result["positive"] is positive
    assert result["empty_latent"]["samples"].shape == (1, 4, 12, 12)
    assert positive[0][1] == {"strength": 0.3}


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
                    ("image2", IMAGE_TYPE),
                    ("image10", IMAGE_TYPE),
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
            image2=np.full((1, 3, 5, 4), 0.25, dtype=np.float32),
            image10=np.full((1, 5, 3, 4), 0.75, dtype=np.float32),
        )


class ReferenceMultiSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="reference-fixture.multi-source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("image2", IMAGE_TYPE),
                    ("image10", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[["multi-positive", {"strength": 0.7}]],
            negative=[["multi-negative", {"strength": 0.2}]],
            vae=ImageVAE(),
            image2=np.full((1, 3, 5, 4), 0.25, dtype=np.float32),
            image10=np.full((1, 5, 3, 4), 0.75, dtype=np.float32),
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


class ReferenceMultiInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="reference-fixture.multi-inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("resized", CONDITIONING_TYPE),
                    ("resized_negative", CONDITIONING_TYPE),
                    ("resized_empty", "core.latent"),
                    ("kept", CONDITIONING_TYPE),
                    ("source_positive", CONDITIONING_TYPE),
                    ("source_negative", CONDITIONING_TYPE),
                )
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        resized: Any,
        resized_negative: Any,
        resized_empty: Any,
        kept: Any,
        source_positive: Any,
        source_negative: Any,
    ) -> dict[str, object]:
        resized_references = resized[0][1]["reference_latents"]
        resized_negative_references = resized_negative[0][1]["reference_latents"]
        kept_references = kept[0][1]["reference_latents"]
        assert (
            len(resized_references) == len(resized_negative_references) == len(kept_references) == 2
        )
        assert resized_negative[0][0] == "multi-positive"
        assert source_positive[0][0] == "multi-positive"
        assert source_negative[0][0] == "multi-negative"
        resized_first = materialize_image(resized_references[0])
        resized_second = materialize_image(resized_references[1])
        kept_first = materialize_image(kept_references[0])
        kept_second = materialize_image(kept_references[1])
        image2 = torch.full((1, 4, 3, 5), 63 / 255)
        image10 = torch.full((1, 4, 5, 3), 191 / 255)
        expected_resized_first = common_upscale(image2, 132, 79, "lanczos", "disabled") * 2 - 1
        expected_resized_second = common_upscale(image10, 79, 132, "lanczos", "disabled") * 2 - 1
        torch.testing.assert_close(resized_first, expected_resized_first)
        torch.testing.assert_close(resized_second, expected_resized_second)
        torch.testing.assert_close(kept_first, torch.full((1, 4, 3, 5), -0.5))
        torch.testing.assert_close(kept_second, torch.full((1, 4, 5, 3), 0.5))
        torch.testing.assert_close(
            materialize_image(materialize_latent(resized_empty)["samples"]),
            torch.zeros((1, 4, 79, 132)),
        )
        assert resized[0][1]["strength"] == 2.5
        assert kept[0][1]["strength"] == 3.5
        assert source_positive[0][1] == {"strength": 0.7}
        assert source_negative[0][1] == {"strength": 0.2}
        return cls.outputs(value=float(resized_first[0, 0, 0, 0] + kept_first[0, 0, 0, 0]))


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
                "from test_reference_conditioning_torch import (\n"
                "    ReferenceMultiInspect, ReferenceMultiSource,\n"
                "    ReferenceSource, ReferenceInspect,\n"
                ")\n"
                "from test_reference_conditioning_torch import MediaConditioningInspect\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = (ReferenceSource, ReferenceMultiSource, ReferenceInspect,\n"
                "         ReferenceMultiInspect,\n"
                "         MediaConditioningInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "reference-fixture-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "reference-conditioning-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["reference-fixture"]\n'
                '[pack.entry]\nnodes = "reference_fixture:NODES"\n'
                'types = "reference_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = root / "dinkster-pack.toml"
            if in_process:
                native_spec = PackSpec(manifest=native_manifest, in_process=True)
                fixture_spec = PackSpec(manifest=fixture_manifest, in_process=True)
            else:
                group_manifests = (native_manifest, fixture_manifest)
                native_spec = PackSpec(
                    manifest=native_manifest,
                    worker_group="reference-conditioning",
                    group_manifests=group_manifests,
                )
                fixture_spec = PackSpec(
                    manifest=fixture_manifest,
                    worker_group="reference-conditioning",
                    group_manifests=group_manifests,
                )
            await composer.add_pack(native_spec)
            await composer.add_pack(fixture_spec)
            graph = Graph(
                nodes={
                    "source": GraphNode("reference-fixture.source"),
                    "multi_source": GraphNode("reference-fixture.multi-source"),
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
                    "multi_resize": GraphNode(
                        "ReferenceMultiLatent",
                        {
                            "positive": Link("multi_source", "positive"),
                            "vae": Link("multi_source", "vae"),
                            "images.image10": Link("multi_source", "image10"),
                            "images.image2": Link("multi_source", "image2"),
                            "resolution.megapixels": 0.01,
                        },
                        slot_variants={"resolution": "resize"},
                    ),
                    "multi_keep": GraphNode(
                        "ReferenceMultiLatent",
                        {
                            "positive": Link("multi_source", "positive"),
                            "negative": Link("multi_source", "negative"),
                            "vae": Link("multi_source", "vae"),
                            "images.image10": Link("multi_source", "image10"),
                            "images.image2": Link("multi_source", "image2"),
                        },
                        slot_variants={"resolution": "keep resolution"},
                    ),
                    "resize_strength": GraphNode(
                        "ConditioningSetAreaStrength",
                        {"conditioning": Link("multi_resize", "positive"), "strength": 2.5},
                    ),
                    "keep_strength": GraphNode(
                        "ConditioningSetAreaStrength",
                        {"conditioning": Link("multi_keep", "positive"), "strength": 3.5},
                    ),
                    "multi_inspect": GraphNode(
                        "reference-fixture.multi-inspect",
                        {
                            "resized": Link("resize_strength", "CONDITIONING"),
                            "resized_negative": Link("multi_resize", "negative"),
                            "resized_empty": Link("multi_resize", "empty_latent"),
                            "kept": Link("keep_strength", "CONDITIONING"),
                            "source_positive": Link("multi_source", "positive"),
                            "source_negative": Link("multi_source", "negative"),
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
                result = await engine.run(graph, ["inspect", "media_inspect", "multi_inspect"])
                assert result.outputs["inspect"]["pixel"].resolve() == -105 / 128
                assert result.outputs["media_inspect"]["strength"].resolve() == 2.5
                assert result.outputs["multi_inspect"]["value"].resolve() == pytest.approx(
                    -127 / 255 - 0.5
                )
                if _:
                    assert set(result.cached) == {
                        "source",
                        "multi_source",
                        "timed",
                        "area",
                        "media_inspect",
                        "edit",
                        "reference",
                        "multi_resize",
                        "multi_keep",
                        "resize_strength",
                        "keep_strength",
                        "multi_inspect",
                        "inspect",
                    }
        finally:
            await composer.close()

    asyncio.run(scenario())
