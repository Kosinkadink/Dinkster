"""Multimodal encoder image streams and conditioning metadata on CPU."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mage_image_encoders import TextEncodeMageFlowEdit
from dinkster_nodes_comfy.ming_image_encoders import TextEncodeMingImageEdit
from dinkster_nodes_comfy.qwen_image_encoders import (
    QwenImage21Cache,
    TextEncodeQwenImage21,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
)

from dinkster.compose import PackSpec, ServingComposer


class Clip:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def tokenize(self, text: str, **options: Any) -> dict[str, Any]:
        self.calls.append((text, options))
        return {"text": text, **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        return [[torch.ones((1, 2)), {"tokens": tokens}]]


def test_ming_without_images_preserves_prompt_and_has_no_reference_metadata() -> None:
    clip = Clip()
    result = TextEncodeMingImageEdit.execute(clip=clip, prompt="describe", images={})
    conditioning: Any = result["CONDITIONING"]
    assert len(conditioning) == 1
    torch.testing.assert_close(conditioning[0][0], torch.ones((1, 2)))
    assert conditioning[0][1] == {"tokens": {"text": "describe", "images": []}}


def test_mage_without_images_uses_each_default_axis_and_null_negative_prompt() -> None:
    clip = Clip()
    result = TextEncodeMageFlowEdit.execute(
        clip=clip, prompt="edit", negative_prompt="", width=0, height=512, batch_size=2
    )
    assert result["latent"]["samples"].shape == (2, 128, 32, 64)
    assert clip.calls == [("edit", {"images": []}), (" ", {"images": []})]


def test_qwen_edit_encoders_preserve_no_image_token_contracts() -> None:
    clip = Clip()
    edit = TextEncodeQwenImageEdit.execute(clip=clip, prompt="edit")["CONDITIONING"]
    plus = TextEncodeQwenImageEditPlus.execute(clip=clip, prompt="plus")["CONDITIONING"]
    assert edit[0][1]["tokens"] == {"text": "edit", "images": []}
    assert plus[0][1]["tokens"]["text"] == "plus"
    assert plus[0][1]["tokens"]["images"] == []
    assert plus[0][1]["tokens"]["llama_template"] == TextEncodeQwenImageEditPlus.llama_template


def test_qwen_image21_empty_images_prevent_empty_text_and_create_default_latent() -> None:
    clip = Clip()
    result = TextEncodeQwenImage21.execute(clip=clip, prompt="", negative_prompt="", resolution=0)
    assert result["latent"]["samples"].shape == (1, 64, 64, 64)
    assert [call[1] for call in clip.calls] == [
        {"images": [], "keep_vision": True, "prevent_empty_text": True},
        {"images": [], "keep_vision": True, "prevent_empty_text": True},
    ]


def test_qwen_cache_clones_model_and_replaces_only_its_cache_option() -> None:
    class Model:
        def __init__(self) -> None:
            self.model_options = {"transformer_options": {"other": "keep"}}

        def clone(self) -> Any:
            clone = Model()
            clone.model_options = {
                "transformer_options": self.model_options["transformer_options"].copy()
            }
            return clone

    source = Model()
    patched = QwenImage21Cache.execute(model=source, device="cpu", dtype="int8")["MODEL"]
    assert patched is not source
    assert source.model_options == {"transformer_options": {"other": "keep"}}
    assert patched.model_options["transformer_options"] == {
        "other": "keep",
        "qwen_image21_cache": {"device": "cpu", "dtype": "int8"},
    }


class ReferenceClip:
    def tokenize(self, text: str, **options: Any) -> dict[str, Any]:
        images = options.pop("images", [])
        return {
            "text": text,
            "images": [(tuple(image.shape), float(image[0, 0, 0, 0])) for image in images],
            **options,
        }

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        return [
            [
                torch.tensor([13.0]),
                {
                    "tokens": tokens,
                    "keep": "existing",
                    "reference_latents": [torch.tensor([-7.0])],
                    "reference_latents_text_embeds": [torch.tensor([-11.0])],
                },
            ]
        ]


class ReferenceVAE:
    def encode(self, image: Any) -> Any:
        batch, height, width, channels = image.shape
        return torch.tensor([float(image[0, 0, 0, 0]), batch, height, width, channels])


class ReferenceVision:
    def encode_image(self, image: Any) -> Any:
        return torch.tensor([float(image[0, 0, 0, 0])])


class ReferenceModel:
    def __init__(self) -> None:
        self.model_options: Any = {"transformer_options": {"other": "keep"}}

    def clone(self) -> ReferenceModel:
        return ReferenceModel()


class MultimodalSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="multimodal-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("clip", CLIP_TYPE),
                    ("vae", VAE_TYPE),
                    ("vision", CLIP_VISION_TYPE),
                    ("model", MODEL_TYPE),
                    ("image2", IMAGE_TYPE),
                    ("image10", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            clip=ReferenceClip(),
            vae=ReferenceVAE(),
            vision=ReferenceVision(),
            model=ReferenceModel(),
            image2=torch.full((1, 32, 64, 3), 0.25).numpy(),
            image10=torch.full((1, 64, 32, 3), 0.75).numpy(),
        )


class MultimodalInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="multimodal-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("edit", "plus", "qwen", "ming", "mage", "joy", "boogu", "fibo", "z")
            )
            + (InputSpec("cache", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("count", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, **inputs: Any) -> dict[str, object]:
        cache = inputs.pop("cache")
        assert cache.model_options["transformer_options"] == {
            "other": "keep",
            "qwen_image21_cache": {"device": "cpu", "dtype": "int4"},
        }
        for name, conditioning in inputs.items():
            metadata = conditioning[0][1]
            assert metadata["keep"] == "existing"
            refs = [materialize_image(value) for value in metadata["reference_latents"]]
            torch.testing.assert_close(refs[0], torch.tensor([-7.0]), rtol=0, atol=0)
            expected = [0.25] if name in ("edit", "fibo") else [0.25, 0.75]
            if name == "fibo":
                # The pinned Lanczos path converts through uint8 PIL pixels.
                expected = [63 / 255]
            torch.testing.assert_close(
                torch.stack([ref[0] for ref in refs[1:]]), torch.tensor(expected)
            )
            for ref in refs[1:]:
                assert ref[1] == 1
                assert ref[4] == 3
            if name == "ming":
                assert [ref[2:4].tolist() for ref in refs[1:]] == [[32, 64], [32, 64]]
            elif name in ("mage", "fibo"):
                assert all(ref[2:4].tolist() == [32, 80] for ref in refs[1:])
            elif name == "qwen":
                assert [ref[2:4].tolist() for ref in refs[1:]] == [[32, 64], [64, 32]]
                assert metadata["tokens"]["keep_vision"] is False
            elif name == "joy":
                assert [ref[2:4].tolist() for ref in refs[1:]] == [[704, 1408], [1408, 704]]
            elif name == "z":
                assert [ref[2:4].tolist() for ref in refs[1:]] == [[32, 64], [64, 32]]
                assert [
                    materialize_image(value).item() for value in metadata["clip_vision_outputs"]
                ] == [0.25, 0.75]
                embeds = metadata["reference_latents_text_embeds"]
                assert len(embeds) == 4
                torch.testing.assert_close(materialize_image(embeds[0]), torch.tensor([-11.0]))
            if name == "plus":
                assert metadata["tokens"]["text"] == (
                    "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
                    "Picture 3: <|vision_start|><|image_pad|><|vision_end|>edit"
                )
        return cls.outputs(count=len(inputs) + 1)


@pytest.mark.parametrize("in_process", (True,))
def test_multimodal_reference_streams_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "multimodal_fixture.py").write_text(
                "from test_multimodal_image_encoders_torch import (\n"
                "    MultimodalSource, MultimodalInspect,\n)\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = (MultimodalSource, MultimodalInspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "fixture.toml"
            fixture.write_text(
                '[pack]\nname = "multimodal-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["multimodal-fixture"]\n[pack.entry]\n'
                'nodes = "multimodal_fixture:NODES"\n'
                'types = "multimodal_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native = root / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="multimodal",
                        group_manifests=(native, fixture),
                    )
                )
            single = {"image": Link("source", "image2")}
            triple = {"image1": Link("source", "image2"), "image3": Link("source", "image10")}
            family = {
                "images.image_10": Link("source", "image10"),
                "images.image_2": Link("source", "image2"),
            }
            definitions = (
                ("edit", "TextEncodeQwenImageEdit", single, "CONDITIONING"),
                ("plus", "TextEncodeQwenImageEditPlus", triple, "CONDITIONING"),
                (
                    "qwen",
                    "TextEncodeQwenImage21",
                    {**family, "resolution": 0, "negative_prompt": ""},
                    "positive",
                ),
                (
                    "ming",
                    "TextEncodeMingImageEdit",
                    {
                        "images.image_8": Link("source", "image10"),
                        "images.image_2": Link("source", "image2"),
                    },
                    "CONDITIONING",
                ),
                (
                    "mage",
                    "TextEncodeMageFlowEdit",
                    {**family, "width": 80, "height": 32, "negative_prompt": ""},
                    "positive",
                ),
                (
                    "joy",
                    "TextEncodeJoyImageEdit",
                    {
                        "images.image0": Link("source", "image2"),
                        "images.image1": Link("source", "image10"),
                    },
                    "CONDITIONING",
                ),
                (
                    "boogu",
                    "TextEncodeBooguEdit",
                    {**family, "negative_prompt": "negative"},
                    "positive",
                ),
                (
                    "fibo",
                    "TextEncodeFibo2",
                    {
                        "images.image_2": Link("source", "image2"),
                        "width": 80,
                        "height": 32,
                        "negative_prompt": "negative",
                    },
                    "positive",
                ),
                (
                    "z",
                    "TextEncodeZImageOmni",
                    {
                        **triple,
                        "auto_resize_images": False,
                        "image_encoder": Link("source", "vision"),
                    },
                    "CONDITIONING",
                ),
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("multimodal-fixture.source"),
                    "cache": GraphNode(
                        "QwenImage21Cache",
                        inputs={"model": Link("source", "model"), "device": "cpu", "dtype": "int4"},
                    ),
                    **{
                        name: GraphNode(
                            node_type,
                            inputs={
                                "clip": Link("source", "clip"),
                                "vae": Link("source", "vae"),
                                "prompt": "edit",
                                **extra,
                            },
                        )
                        for name, node_type, extra, _ in definitions
                    },
                    "inspect": GraphNode(
                        "multimodal-fixture.inspect",
                        inputs={
                            **{name: Link(name, output) for name, _, _, output in definitions},
                            "cache": Link("cache", "MODEL"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["count"].resolve() == 10
        finally:
            await composer.close()

    asyncio.run(scenario())
