"""Core CLIP nodes preserve scheduled outputs and isolate layer selection."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.clip import (
    CLIPSetLastLayer,
    CLIPTextEncode,
    CLIPVisionEncode,
    unCLIPConditioning,
)
from dinkster_nodes_comfy.clip_model_specific import (
    CLIPTextEncodeHiDream,
    CLIPTextEncodeLumina2,
    CLIPTextEncodePixArtAlpha,
    T5TokenizerOptions,
)
from dinkster_nodes_comfy.flux_conditioning import (
    CLIPTextEncodeFlux,
    FluxDisableGuidance,
    FluxGuidance,
    FluxKontextMultiReferenceLatentMethod,
)

from dinkster.compose import PackSpec, ServingComposer


class TextEncoder:
    def __init__(self, layer: int = -1) -> None:
        self.layer = layer

    def clone(self) -> TextEncoder:
        return TextEncoder(self.layer)

    def clip_layer(self, layer: int) -> None:
        self.layer = layer

    def tokenize(self, text: str) -> dict[str, str]:
        return {"literal": text}

    def encode_from_tokens_scheduled(self, tokens: dict[str, str]) -> list[list[object]]:
        return [
            [tokens["literal"], {"layer": self.layer, "start_percent": 0.125}],
            [tokens["literal"] + " next", {"layer": self.layer, "start_percent": 0.75}],
        ]


@pytest.mark.parametrize("layer", (-24, -7, -1))
def test_layer_selection_preserves_source_and_scheduled_conditioning(layer: int) -> None:
    source = TextEncoder()
    selected: Any = CLIPSetLastLayer.execute(clip=source, stop_at_clip_layer=layer)["CLIP"]
    text = "  literal {prompt|choices}\nnext\t"
    result = CLIPTextEncode.execute(clip=selected, text=text)["CONDITIONING"]
    assert result == [
        [text, {"layer": layer, "start_percent": 0.125}],
        [text + " next", {"layer": layer, "start_percent": 0.75}],
    ]
    assert selected is not source
    assert source.layer == -1


def test_missing_clip_reports_checkpoint_encoder_error() -> None:
    with pytest.raises(RuntimeError, match="checkpoint does not contain a valid clip"):
        CLIPTextEncode.execute(clip=None, text="prompt")


def test_clip_vision_encode_schema_uses_portable_vision_types() -> None:
    schema = CLIPVisionEncode.define_schema()

    assert schema.node_type == "CLIPVisionEncode"
    assert [input.id for input in schema.inputs] == ["clip_vision", "image", "crop"]
    assert schema.inputs[0].type == TypeExpr.concrete("CLIP_VISION")
    assert schema.inputs[1].type == TypeExpr.concrete("IMAGE")
    assert schema.inputs[2].default == "center"
    assert schema.outputs[0].id == "CLIP_VISION_OUTPUT"
    assert schema.outputs[0].type == TypeExpr.concrete("CLIP_VISION_OUTPUT")


def test_unclip_conditioning_schema_uses_portable_vision_output() -> None:
    schema = unCLIPConditioning.define_schema()

    assert schema.node_type == "unCLIPConditioning"
    assert schema.category == "model/conditioning"
    assert [input.id for input in schema.inputs] == [
        "conditioning",
        "clip_vision_output",
        "strength",
        "noise_augmentation",
    ]
    assert schema.inputs[0].type == TypeExpr.concrete("CONDITIONING")
    assert schema.inputs[1].type == TypeExpr.concrete("CLIP_VISION_OUTPUT")
    assert schema.inputs[2].default == 1.0
    assert schema.inputs[3].default == 0.0
    assert schema.outputs[0].id == "CONDITIONING"
    assert schema.outputs[0].type == TypeExpr.concrete("CONDITIONING")


class ModelTextEncoder:
    def __init__(self) -> None:
        self.options: dict[str, int] = {"unrelated": 17}

    def clone(self) -> ModelTextEncoder:
        result = ModelTextEncoder()
        result.options = self.options.copy()
        return result

    def set_tokenizer_option(self, name: str, value: int) -> None:
        self.options[name] = value

    def tokenize(self, text: str) -> dict[str, str]:
        return {key: f"{key}:{text}" for key in ("l", "g", "t5xxl", "llama", "extra")}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, str], add_dict: dict[str, Any] | None = None
    ) -> list[list[object]]:
        return [[tokens, {"start_percent": 0.25, **(add_dict or {})}]]


def test_flux_prompt_streams_and_guidance_preserve_scheduled_metadata() -> None:
    encoder = ModelTextEncoder()
    result: Any = CLIPTextEncodeFlux.execute(
        clip=encoder, clip_l="short\n{one|two}", t5xxl="different long prompt", guidance=7.25
    )["CONDITIONING"]
    expected_tokens = {
        "l": "l:short\n{one|two}",
        "g": "g:short\n{one|two}",
        "t5xxl": "t5xxl:different long prompt",
        "llama": "llama:short\n{one|two}",
        "extra": "extra:short\n{one|two}",
    }
    assert result == [[expected_tokens, {"start_percent": 0.25, "guidance": 7.25}]]
    result.append(["second embedding", {"end_percent": 0.9, "guidance": 13.0}])
    changed: Any = FluxGuidance.execute(conditioning=result, guidance=0.0)["CONDITIONING"]
    disabled: Any = FluxDisableGuidance.execute(conditioning=changed)["CONDITIONING"]
    assert changed == [
        [expected_tokens, {"start_percent": 0.25, "guidance": 0.0}],
        ["second embedding", {"end_percent": 0.9, "guidance": 0.0}],
    ]
    assert disabled == [
        [expected_tokens, {"start_percent": 0.25, "guidance": None}],
        ["second embedding", {"end_percent": 0.9, "guidance": None}],
    ]
    assert result[0][1] == {"start_percent": 0.25, "guidance": 7.25}
    assert result[1][1] == {"end_percent": 0.9, "guidance": 13.0}
    assert changed[0][0] is result[0][0]


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("offset", "offset"),
        ("index", "index"),
        ("uxo/uno", "uxo"),
        ("uso/uno", "uxo"),
        ("index_timestep_zero", "index_timestep_zero"),
        ("index_timestep_zero_isolated", "index_timestep_zero_isolated"),
        ("index_timestep_zero_text_start", "index_timestep_zero_text_start"),
    ),
)
def test_flux_reference_method_preserves_source(method: str, expected: str) -> None:
    source = [["embedding", {"guidance": 2.5, "reference_latents_method": "prior"}]]
    result = FluxKontextMultiReferenceLatentMethod.execute(
        conditioning=source, reference_latents_method=method
    )["CONDITIONING"]
    assert result == [["embedding", {"guidance": 2.5, "reference_latents_method": expected}]]
    assert source == [["embedding", {"guidance": 2.5, "reference_latents_method": "prior"}]]


def test_model_specific_prompts_and_resolution() -> None:
    encoder = ModelTextEncoder()
    pixart = CLIPTextEncodePixArtAlpha.execute(
        clip=encoder, text="literal\n{one|two}", width=832, height=1216
    )["CONDITIONING"]
    assert pixart == [
        [
            encoder.tokenize("literal\n{one|two}"),
            {"start_percent": 0.25, "width": 832, "height": 1216},
        ]
    ]
    hidream: Any = CLIPTextEncodeHiDream.execute(
        clip=encoder, clip_l="left", clip_g="global", t5xxl="long", llama="instruct"
    )["CONDITIONING"]
    assert hidream == [
        [
            {
                "l": "l:left",
                "g": "g:global",
                "t5xxl": "t5xxl:long",
                "llama": "llama:instruct",
                "extra": "extra:global",
            },
            {"start_percent": 0.25},
        ]
    ]
    for key, system in (
        (
            "superior",
            "You are an assistant designed to generate superior images with the superior "
            "degree of image-text alignment based on textual prompts or user prompts.",
        ),
        (
            "alignment",
            "You are an assistant designed to generate high-quality images with the "
            "highest degree of image-text alignment based on textual prompts.",
        ),
    ):
        result = CLIPTextEncodeLumina2.execute(
            clip=encoder, user_prompt="  user\n", system_prompt=key
        )["CONDITIONING"]
        assert result == [
            [encoder.tokenize(f"{system} <Prompt Start>   user\n"), {"start_percent": 0.25}]
        ]
    with pytest.raises(RuntimeError, match="checkpoint does not contain a valid clip"):
        CLIPTextEncodeLumina2.execute(clip=None, user_prompt="", system_prompt="alignment")


def test_t5_options_preserve_source_and_other_tokenizers() -> None:
    source = ModelTextEncoder()
    selected: Any = T5TokenizerOptions.execute(clip=source, min_padding=7, min_length=43)["CLIP"]
    assert selected is not source
    assert source.options == {"unrelated": 17}
    assert selected.options == {
        "unrelated": 17,
        "t5xxl_min_padding": 7,
        "t5xxl_min_length": 43,
        "pile_t5xl_min_padding": 7,
        "pile_t5xl_min_length": 43,
        "t5base_min_padding": 7,
        "t5base_min_length": 43,
        "mt5xl_min_padding": 7,
        "mt5xl_min_length": 43,
        "umt5xxl_min_padding": 7,
        "umt5xxl_min_length": 43,
    }


class EncoderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="clip-fixture.source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(CLIP=TextEncoder())


class ConditioningText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="clip-fixture.text",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: list[list[Any]]) -> Mapping[str, object]:
        return cls.outputs(text=repr(conditioning))


CLIP_FIXTURES = (EncoderSource, ConditioningText)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_pack_encodes_original_and_selected_layers(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "clip_fixture.py").write_text(
                "from test_comfy_clip_nodes import CLIP_FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *CLIP_FIXTURES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["clip-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "clip_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "clip_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("clip-fixture.source"),
                    "select": GraphNode(
                        "CLIPSetLastLayer",
                        {"clip": Link("source", "CLIP"), "stop_at_clip_layer": -7},
                    ),
                    "original": GraphNode(
                        "CLIPTextEncode", {"clip": Link("source", "CLIP"), "text": "original"}
                    ),
                    "selected": GraphNode(
                        "CLIPTextEncode", {"clip": Link("select", "CLIP"), "text": "selected"}
                    ),
                    "original-text": GraphNode(
                        "clip-fixture.text", {"conditioning": Link("original", "CONDITIONING")}
                    ),
                    "selected-text": GraphNode(
                        "clip-fixture.text", {"conditioning": Link("selected", "CONDITIONING")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["original-text", "selected-text"])
                for name, text, layer in (
                    ("original-text", "original", -1),
                    ("selected-text", "selected", -7),
                ):
                    assert result.outputs[name]["text"].resolve() == repr(
                        [
                            [text, {"layer": layer, "start_percent": 0.125}],
                            [text + " next", {"layer": layer, "start_percent": 0.75}],
                        ]
                    )
        finally:
            await composer.close()

    asyncio.run(scenario())
