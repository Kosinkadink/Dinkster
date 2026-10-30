"""SDXL prompt streams and conditioning metadata through the public nodes."""

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
from dinkster_nodes_comfy.clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner

from dinkster.compose import PackSpec, ServingComposer


class SDXLEncoder:
    def tokenize(self, text: str) -> dict[str, list[str]]:
        chunks = text.split("|")
        return {"g": [f"g:{chunk}" for chunk in chunks], "l": [f"l:{c}" for c in chunks]}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, list[str]], *, add_dict: dict[str, object]
    ) -> list[list[object]]:
        return [
            [tokens, {**add_dict, "start_percent": 0.125}],
            [tokens, {**add_dict, "start_percent": 0.75}],
        ]


_SIZE = {
    "width": 768,
    "height": 1280,
    "crop_w": 13,
    "crop_h": 27,
    "target_width": 1536,
    "target_height": 896,
}


@pytest.mark.parametrize(
    ("text_g", "text_l", "global_tokens", "local_tokens"),
    (
        ("red", "blue", ["g:red"], ["l:blue"]),
        ("red|green|white", "blue", ["g:red", "g:green", "g:white"], ["l:blue", "l:", "l:"]),
        ("red", "blue|black|gold", ["g:red", "g:", "g:"], ["l:blue", "l:black", "l:gold"]),
        ("", "blue|black", ["g:", "g:"], ["l:blue", "l:black"]),
    ),
)
def test_sdxl_pads_only_the_shorter_prompt_stream(
    text_g: str, text_l: str, global_tokens: list[str], local_tokens: list[str]
) -> None:
    result = CLIPTextEncodeSDXL.execute(clip=SDXLEncoder(), text_g=text_g, text_l=text_l, **_SIZE)[
        "CONDITIONING"
    ]
    assert result == [
        [{"g": global_tokens, "l": local_tokens}, {**_SIZE, "start_percent": 0.125}],
        [{"g": global_tokens, "l": local_tokens}, {**_SIZE, "start_percent": 0.75}],
    ]


def test_refiner_preserves_aesthetic_score_and_schedule() -> None:
    result = CLIPTextEncodeSDXLRefiner.execute(
        clip=SDXLEncoder(), text="  literal {prompt|choices}\n", ascore=3.25, width=640, height=1152
    )["CONDITIONING"]
    tokens = {
        "g": ["g:  literal {prompt", "g:choices}\n"],
        "l": ["l:  literal {prompt", "l:choices}\n"],
    }
    metadata = {"aesthetic_score": 3.25, "width": 640, "height": 1152}
    assert result == [
        [tokens, {**metadata, "start_percent": 0.125}],
        [tokens, {**metadata, "start_percent": 0.75}],
    ]


class SDXLSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-fixture.source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(CLIP=SDXLEncoder())


class SDXLInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> Mapping[str, object]:
        return cls.outputs(text=repr(conditioning))


SDXL_FIXTURES = (SDXLSource, SDXLInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_preserves_sdxl_prompt_streams_and_refiner_metadata(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "sdxl_fixture.py").write_text(
                "from test_comfy_clip_sdxl_nodes import SDXL_FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, *SDXL_FIXTURES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["sdxl-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "sdxl_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "sdxl_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "sdxl_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("sdxl-fixture.source"),
                    "base": GraphNode(
                        "CLIPTextEncodeSDXL",
                        {
                            "clip": Link("source", "CLIP"),
                            "text_g": "red",
                            "text_l": "blue|black",
                            **_SIZE,
                        },
                    ),
                    "refiner": GraphNode(
                        "CLIPTextEncodeSDXLRefiner",
                        {
                            "clip": Link("source", "CLIP"),
                            "text": "gold",
                            "ascore": 3.25,
                            "width": 640,
                            "height": 1152,
                        },
                    ),
                    "base-text": GraphNode(
                        "sdxl-fixture.inspect", {"conditioning": Link("base", "CONDITIONING")}
                    ),
                    "refiner-text": GraphNode(
                        "sdxl-fixture.inspect", {"conditioning": Link("refiner", "CONDITIONING")}
                    ),
                }
            )
            expected_tokens = {"g": ["g:red", "g:"], "l": ["l:blue", "l:black"]}
            refiner_tokens = {"g": ["g:gold"], "l": ["l:gold"]}
            refiner_meta = {"aesthetic_score": 3.25, "width": 640, "height": 1152}
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["base-text", "refiner-text"])
                assert result.outputs["base-text"]["text"].resolve() == repr(
                    [
                        [expected_tokens, {**_SIZE, "start_percent": 0.125}],
                        [expected_tokens, {**_SIZE, "start_percent": 0.75}],
                    ]
                )
                assert result.outputs["refiner-text"]["text"].resolve() == repr(
                    [
                        [refiner_tokens, {**refiner_meta, "start_percent": 0.125}],
                        [refiner_tokens, {**refiner_meta, "start_percent": 0.75}],
                    ]
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
