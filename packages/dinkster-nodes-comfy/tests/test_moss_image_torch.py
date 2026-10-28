"""MOSS layout coordinates and edit-format reference semantics on CPU."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moss_image import (
    MOSSImageControl,
    MOSSImageLayoutPrompt,
    TextEncodeMOSSImageEdit,
)

from dinkster.compose import PackSpec, ServingComposer


def test_moss_layout_uses_enclosing_centralized_edges_and_text_precedence() -> None:
    elements = [
        {"bbox": [123.5, 98.2, 789.4, 900.1], "desc": "subject", "color_palette": ["#abcdef"]},
        {"bbox": [0, 0, 1000, 1000], "type": "text", "desc": "caption", "text": "quoted"},
    ]
    result = MOSSImageLayoutPrompt.execute(
        prompt="  scene  ", element=elements, width=1001, height=601, character_boxes="1, 2"
    )["prompt"]
    assert json.loads(result) == {
        "summary": {"caption": "scene"},
        "items": [
            {
                "label": "character",
                "caption": "subject",
                "palette": ["#abcdef"],
                "bbox": [-226, -402, 175, 402],
            },
            {
                "label": "text",
                "caption": "caption",
                "text": "quoted",
                "bbox": [-300, -500, 301, 501],
            },
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    assert elements[0]["bbox"] == [123.5, 98.2, 789.4, 900.1]
    assert (
        MOSSImageLayoutPrompt.execute(prompt="  unchanged  ", element=[], width=1001, height=601)[
            "prompt"
        ]
        == "  unchanged  "
    )


class Clip:
    def __init__(self) -> None:
        self.prompt = ""
        self.images: list[torch.Tensor] = []
        self.conditioning = [[torch.tensor([19.0]), {"keep": 7, "reference_latents": ["old"]}]]

    def tokenize(self, prompt: str, *, images: list[torch.Tensor]) -> str:
        self.prompt = prompt
        self.images = images
        return prompt

    def encode_from_tokens_scheduled(self, tokens: str) -> list[list[Any]]:
        assert tokens == self.prompt
        return self.conditioning


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[-1] == 3
        return image.mean(dim=(1, 2), keepdim=True)


@pytest.mark.parametrize(
    ("edit_format", "method", "vision"),
    [
        ("ai-toolkit", "index_timestep_zero_isolated", True),
        ("ai-toolkit (kv_cache off)", "index_timestep_zero", True),
        ("Moss-Image-Edit", "index_timestep_zero_text_start", False),
    ],
)
def test_moss_edit_orders_references_and_selects_vision_and_latent_layout(
    edit_format: str, method: str, vision: bool
) -> None:
    clip = Clip()
    large = torch.full((1, 400, 600, 4), 10.0)
    small = torch.full((1, 3, 5, 4), 2.0)
    output = TextEncodeMOSSImageEdit.execute(
        clip=clip,
        prompt="edit",
        vae=VAE(),
        edit_format=edit_format,
        images={"image_2": small.numpy(), "image_0": None, "image_10": large.numpy()},
    )["CONDITIONING"]
    assert output[0][0] is clip.conditioning[0][0]
    assert output[0][1]["keep"] == 7
    assert clip.conditioning[0][1] == {"keep": 7, "reference_latents": ["old"]}
    assert output[0][1]["reference_latents_method"] == method
    refs = output[0][1]["reference_latents"]
    assert len(refs) == 2
    torch.testing.assert_close(refs[0], torch.full((1, 1, 1, 3), 10.0), rtol=0, atol=0)
    torch.testing.assert_close(refs[1], torch.full((1, 1, 1, 3), 2.0), rtol=0, atol=0)
    if vision:
        assert clip.prompt == (
            "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
            "Picture 2: <|vision_start|><|image_pad|><|vision_end|>edit"
        )
        assert [tuple(image.shape) for image in clip.images] == [(1, 314, 470, 4), (1, 3, 5, 4)]
        torch.testing.assert_close(
            clip.images[0], torch.full((1, 314, 470, 4), 10.0), rtol=0, atol=0
        )
    else:
        assert clip.prompt == "edit"
        assert clip.images == []
    assert torch.equal(large, torch.full_like(large, 10.0))
    assert torch.equal(small, torch.full_like(small, 2.0))


class ControlVAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[-1] == 3
        return image.movedim(-1, 1)


@pytest.mark.parametrize("missing", ("color", "depth", "neither"))
def test_moss_control_aligns_maps_batches_and_splits_timestep_range(missing: str) -> None:
    color = torch.full((1, 3, 5, 4), 2.0)
    depth = torch.stack((torch.full((5, 3, 3), 3.0), torch.full((5, 3, 3), 7.0)))
    positive = [[torch.tensor([11.0]), {"keep": "positive"}]]
    negative = [
        [torch.tensor([-7.0]), {"keep": "negative", "start_percent": 0.4, "end_percent": 0.9}]
    ]
    output = MOSSImageControl.execute(
        positive=positive,
        negative=negative,
        vae=ControlVAE(),
        color_map=None if missing == "color" else color.numpy(),
        depth_map=None if missing == "depth" else depth.numpy(),
        strength=1.5,
        start_percent=0.25,
        end_percent=0.6,
    )
    height, width = (5, 3) if missing == "color" else (3, 5)
    batch = 1 if missing == "depth" else 2
    for name, source, start in (("positive", positive, 0.25), ("negative", negative, 0.4)):
        result = output[name]
        active = next(entry for entry in result if "concat_latent_image" in entry[1])
        assert active[0] is source[0][0]
        assert active[1]["keep"] == name
        assert active[1]["start_percent"] == start
        assert active[1]["end_percent"] == 0.6
        control = active[1]["concat_latent_image"]
        assert control.shape == (batch, 6, height, width)
        torch.testing.assert_close(
            control[:, :3],
            torch.full((batch, 3, height, width), 0.0 if missing == "color" else 3.0),
        )
        for index in range(batch):
            torch.testing.assert_close(
                control[index, 3:],
                torch.full(
                    (3, height, width), 0.0 if missing == "depth" else (4.5 if index == 0 else 10.5)
                ),
            )
    assert positive[0][1] == {"keep": "positive"}
    assert negative[0][1] == {"keep": "negative", "start_percent": 0.4, "end_percent": 0.9}
    passthrough = MOSSImageControl.execute(positive=positive, negative=negative, vae=ControlVAE())
    assert passthrough["positive"] is positive
    assert passthrough["negative"] is negative


class MossSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="moss-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("clip", CLIP_TYPE),
                    ("vae", VAE_TYPE),
                    ("control_vae", VAE_TYPE),
                    ("conditioning", CONDITIONING_TYPE),
                    ("image2", IMAGE_TYPE),
                    ("image10", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            clip=Clip(),
            vae=VAE(),
            control_vae=ControlVAE(),
            conditioning=[[torch.tensor([11.0]), {"keep": 7}]],
            image2=torch.full((1, 3, 5, 4), 2.0).numpy(),
            image10=torch.full((1, 5, 3, 4), 10.0).numpy(),
        )


class MossInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="moss-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("vision", "text", "control_positive", "control_negative")
            ),
            outputs=(OutputSpec("count", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(
        cls, *, vision: Any, text: Any, control_positive: Any, control_negative: Any
    ) -> dict[str, object]:
        for output in (control_positive, control_negative):
            assert len(output) == 3
            active = output[1][1]
            assert active["start_percent"] == 0.25
            assert active["end_percent"] == 0.6
            control = materialize_image(active["concat_latent_image"])
            torch.testing.assert_close(control[:, :3], torch.full((1, 3, 3, 5), 3.0))
            torch.testing.assert_close(control[:, 3:], torch.full((1, 3, 3, 5), 15.0))
        for output, method in (
            (vision, "index_timestep_zero_isolated"),
            (text, "index_timestep_zero_text_start"),
        ):
            assert output[0][1]["keep"] == 7
            assert output[0][1]["reference_latents_method"] == method
            refs = output[0][1]["reference_latents"]
            assert len(refs) == 2
            for reference, value in zip(refs, (10.0, 2.0), strict=True):
                torch.testing.assert_close(
                    materialize_image(reference), torch.full((1, 1, 1, 3), value), rtol=0, atol=0
                )
        return cls.outputs(count=2)


@pytest.mark.parametrize("in_process", (True,))
def test_moss_edit_ordinary_manifest_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(root / "tests"), str(root / "src"), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            (tmp_path / "moss_fixture.py").write_text(
                "from test_moss_image_torch import MossSource, MossInspect\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = (MossSource, MossInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "moss-fixture-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "moss-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["moss-fixture"]\n'
                '[pack.entry]\nnodes = "moss_fixture:NODES"\n'
                'types = "moss_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = root / "dinkster-pack.toml"
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="moss-edit",
                        group_manifests=(native_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("moss-fixture.source"),
                    **{
                        name: GraphNode(
                            "TextEncodeMOSSImageEdit",
                            inputs={
                                "clip": Link("source", "clip"),
                                "vae": Link("source", "vae"),
                                "prompt": "edit",
                                "edit_format": edit_format,
                                "images.image_2": Link("source", "image2"),
                                "images.image_10": Link("source", "image10"),
                            },
                        )
                        for name, edit_format in (
                            ("vision", "ai-toolkit"),
                            ("text", "Moss-Image-Edit"),
                        )
                    },
                    "control": GraphNode(
                        "MOSSImageControl",
                        inputs={
                            "positive": Link("source", "conditioning"),
                            "negative": Link("source", "conditioning"),
                            "vae": Link("source", "control_vae"),
                            "color_map": Link("source", "image2"),
                            "depth_map": Link("source", "image10"),
                            "strength": 1.5,
                            "start_percent": 0.25,
                            "end_percent": 0.6,
                        },
                    ),
                    "inspect": GraphNode(
                        "moss-fixture.inspect",
                        inputs={
                            "vision": Link("vision", "CONDITIONING"),
                            "text": Link("text", "CONDITIONING"),
                            "control_positive": Link("control", "positive"),
                            "control_negative": Link("control", "negative"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["count"].resolve() == 2
        finally:
            await composer.close()

    asyncio.run(scenario())
