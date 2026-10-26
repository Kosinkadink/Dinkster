"""Controlled CPU identity fusion and PhotoMaker token conditioning."""

from __future__ import annotations

import asyncio
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
    PHOTOMAKER_TYPE,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import photomaker
from dinkster_nodes_comfy.photomaker import PhotoMakerEncode, PhotoMakerLoader

from dinkster.compose import PackSpec, ServingComposer


class IdentityModel:
    load_device = torch.device("cpu")

    def __init__(self) -> None:
        self.pixels: Any = None
        self.mask: Any = None

    def __call__(self, *, id_pixel_values: Any, prompt_embeds: Any, class_tokens_mask: Any) -> Any:
        self.pixels = id_pixel_values
        self.mask = class_tokens_mask
        return prompt_embeds + class_tokens_mask[..., None] * 7


class Clip:
    def __init__(self) -> None:
        self.tokens = {
            "g": [[(11, 1.0, 0), (12, 0.4, 2), (13, 0.9, 3), (99, 1.0, -1)]],
            "l": [[(21, 0.3, 2), (22, 0.7, 2), (23, 0.6, 4), (98, 1.0, -1)]],
        }
        self.cond = torch.arange(77 * 3, dtype=torch.float32).reshape(1, 77, 3) / 13
        self.pooled = torch.tensor([[2.0, -3.0]])
        self.received: Any = None

    def tokenize(self, text: str, *, return_word_ids: bool) -> Any:
        assert return_word_ids
        return self.tokens

    def encode_from_tokens(self, tokens: Any, *, return_pooled: bool) -> Any:
        assert return_pooled
        self.received = tokens
        return self.cond, self.pooled


@pytest.mark.parametrize("text", ("portrait photomaker sunset", "portrait without marker"))
def test_encode_removes_selected_word_and_preserves_padding_and_pooled_identity(text: str) -> None:
    model = IdentityModel()
    clip = Clip()
    original = {key: [list(chunk) for chunk in chunks] for key, chunks in clip.tokens.items()}
    image = torch.linspace(0, 1, 32 * 48 * 3).reshape(1, 32, 48, 3)
    result: Any = PhotoMakerEncode.execute(photomaker=model, image=image, clip=clip, text=text)[
        "CONDITIONING"
    ]
    assert result[0][1]["pooled_output"] is clip.pooled
    assert clip.tokens == original
    if "photomaker" in text:
        assert clip.received == {
            "g": [[(11, 1.0, 0), (13, 0.9, 3), (99, 1.0, -1), (99, 1.0, -1)]],
            "l": [[(23, 0.6, 4), (98, 1.0, -1), (98, 1.0, -1), (98, 1.0, -1)]],
        }
        assert model.mask.shape == (1, 77)
        assert model.mask.nonzero().tolist() == [[0, 1]]
        expected = clip.cond.clone()
        expected[:, 1] += 7
        torch.testing.assert_close(result[0][0], expected, rtol=0, atol=0)
        assert model.pixels.shape == (1, 1, 3, 224, 224)
        assert model.pixels.dtype == torch.float32
    else:
        assert result[0][0] is clip.cond
        assert model.mask is None
        # F uses index=-1 when absent, removing matching end-token word IDs.
        assert clip.received["g"][0] == original["g"][0]
        assert clip.received["l"][0] == original["l"][0]


def test_fusion_preserves_unselected_tokens_and_uses_ordered_valid_ids() -> None:
    fusion: Any = photomaker._fusion_module(torch.nn)(3)
    with torch.no_grad():
        for index, parameter in enumerate(fusion.parameters()):
            parameter.copy_(
                torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                * 0.017
                + (index + 1) * 0.031
            )
    prompt = torch.tensor([[[1.0, -2.0, 3.0], [7.0, 0.5, -4.0], [-1.0, 8.0, 2.0]]])
    ids = torch.tensor([[[[0.2, -0.4, 0.8]], [[1.3, 2.1, -0.7]], [[99.0, 99.0, 99.0]]]])
    mask = torch.tensor([[True, False, True]])
    before = prompt.clone()

    def mlp(x: Any, module: Any) -> Any:
        normalized = torch.nn.functional.layer_norm(
            x, (x.shape[-1],), module.layernorm.weight, module.layernorm.bias
        )
        hidden = torch.nn.functional.gelu(
            torch.nn.functional.linear(normalized, module.fc1.weight, module.fc1.bias)
        )
        value = torch.nn.functional.linear(hidden, module.fc2.weight, module.fc2.bias)
        return value + x if module.use_residual else value

    selected = before[:, [0, 2]].reshape(2, 3)
    combined = mlp(torch.cat((selected, ids[0, :2, 0]), dim=-1), fusion.mlp1) + selected
    expected = torch.nn.functional.layer_norm(
        mlp(combined, fusion.mlp2),
        (3,),
        fusion.layer_norm.weight,
        fusion.layer_norm.bias,
    )
    result = fusion(prompt, ids.to(torch.float64), mask)
    assert result.data_ptr() == prompt.data_ptr()
    torch.testing.assert_close(result[:, 1], before[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(result[:, [0, 2]].reshape(2, 3), expected, rtol=0, atol=0)


@pytest.mark.parametrize("nested", (True, False))
def test_loader_safely_reads_admitted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    initialize_inference_process()
    data = {"visual_projection_2.weight": torch.tensor([[1.0, -3.0], [2.0, 4.0]])}
    path = tmp_path / "identity.pt"
    torch.save({"id_encoder": data} if nested else data, path)

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

    class Model:
        state: Any = None

        def load_state_dict(self, state: Any) -> None:
            self.state = state

    model = Model()
    monkeypatch.setattr(photomaker, "_id_encoder", lambda: model)
    asset = AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        resolver=Resolver(),
    )
    result = PhotoMakerLoader.execute(photomaker_model_name=asset)
    assert result["PHOTOMAKER"] is model
    assert list(model.state) == ["visual_projection_2.weight"]
    torch.testing.assert_close(
        model.state["visual_projection_2.weight"], data["visual_projection_2.weight"]
    )


class PhotoMakerSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="photomaker-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", PHOTOMAKER_TYPE),
                    ("clip", CLIP_TYPE),
                    ("image", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=IdentityModel(), clip=Clip(), image=torch.ones(1, 32, 48, 3) * 0.25
        )


class PhotoMakerInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="photomaker-fixture.inspect",
            inputs=(
                InputSpec("model", TypeExpr.concrete(PHOTOMAKER_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("count", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, model: Any, clip: Any, conditioning: Any) -> dict[str, object]:
        assert model.mask.nonzero().tolist() == [[0, 1]]
        assert model.pixels.shape == (1, 1, 3, 224, 224)
        expected = clip.cond.clone()
        expected[:, 1] += 7
        torch.testing.assert_close(conditioning[0][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(conditioning[0][1]["pooled_output"], clip.pooled, rtol=0, atol=0)
        return cls.outputs(count=1)


@pytest.mark.parametrize("in_process", (True, False))
def test_photo_identity_crosses_ordinary_resident_hosts(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "photomaker_fixture.py").write_text(
                "from test_photomaker_torch import PhotoMakerSource, PhotoMakerInspect\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = (PhotoMakerSource, PhotoMakerInspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "fixture.toml"
            fixture.write_text(
                '[pack]\nname = "photomaker-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["photomaker-fixture"]\n[pack.entry]\n'
                'nodes = "photomaker_fixture:NODES"\n'
                'types = "photomaker_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native = root / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="photomaker",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("photomaker-fixture.source"),
                    "encode": GraphNode(
                        "PhotoMakerEncode",
                        inputs={
                            "photomaker": Link("source", "model"),
                            "clip": Link("source", "clip"),
                            "image": Link("source", "image"),
                            "text": "portrait photomaker sunset",
                        },
                    ),
                    "inspect": GraphNode(
                        "photomaker-fixture.inspect",
                        inputs={
                            "model": Link("source", "model"),
                            "clip": Link("source", "clip"),
                            "conditioning": Link("encode", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["count"].resolve() == 1
        finally:
            await composer.close()

    asyncio.run(scenario())
