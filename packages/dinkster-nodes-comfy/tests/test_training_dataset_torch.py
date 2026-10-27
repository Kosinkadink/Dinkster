"""Training dataset orchestration with controlled CPU encoders."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.dataset_groups import MakeTrainingDataset

from dinkster.compose import PackSpec, ServingComposer


class DatasetVAE:
    def encode(self, pixels: Any) -> Any:
        assert pixels.shape[-1] == 3
        return pixels.movedim(-1, 1) * 3 - 2


class DatasetCLIP:
    def tokenize(self, text: str) -> list[int]:
        return [len(text), sum(map(ord, text))]

    def encode_from_tokens_scheduled(self, tokens: list[int]) -> Any:
        return [[torch.tensor([[tokens]], dtype=torch.float32), {"length": tokens[0]}]]


def _images() -> list[Any]:
    return [
        torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4) / 50,
        torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 20,
    ]


@pytest.mark.parametrize("texts", (None, [], ["shared"], ["ab", "z!"]))
def test_dataset_encoding_preserves_pairs_and_ignores_alpha(texts: list[str] | None) -> None:
    images = _images()
    originals = [image.clone() for image in images]
    result = MakeTrainingDataset.execute(
        images=images, vae=DatasetVAE(), clip=DatasetCLIP(), texts=texts
    )
    captions = ["", ""] if not texts else texts * 2 if len(texts) == 1 else texts
    assert len(result["latents"]) == len(result["conditioning"]) == 2
    for index, caption in enumerate(captions):
        assert set(result["latents"][index]) == {"samples"}
        assert torch.equal(
            result["latents"][index]["samples"], images[index][..., :3].movedim(-1, 1) * 3 - 2
        )
        embedding, metadata = result["conditioning"][index][0]
        assert embedding.tolist() == [[[len(caption), sum(map(ord, caption))]]]
        assert metadata == {"length": len(caption)}
        assert torch.equal(images[index], originals[index])


def test_dataset_encoding_rejects_caption_mismatch_including_empty_images() -> None:
    for images, texts in ((_images(), ["a", "b", "c"]), ([], None), ([], []), ([], ["x"])):
        with pytest.raises(ValueError, match="does not match number of images"):
            MakeTrainingDataset.execute(
                images=images, vae=DatasetVAE(), clip=DatasetCLIP(), texts=texts
            )


class TrainingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training-dataset-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("texts", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(
            images=_images(), vae=DatasetVAE(), clip=DatasetCLIP(), texts=["ab", "z!"]
        )


class TrainingInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training-dataset-fixture.inspect",
            inputs=(
                InputSpec("latents", TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))),
                InputSpec("conditioning", TypeExpr.list_of(TypeExpr.concrete(CONDITIONING_TYPE))),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, latents: Any, conditioning: Any) -> Any:
        assert len(latents) == len(conditioning) == 2
        for index, (shape, divisor, tokens) in enumerate(
            (((2, 2, 3, 4), 50, [2, 195]), ((1, 3, 2, 3), 20, [2, 155]))
        ):
            source = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
            expected = source.reshape(shape)[..., :3].movedim(-1, 1) / divisor * 3 - 2
            assert torch.equal(materialize_latent(latents[index])["samples"], expected)
            embedding, metadata = conditioning[index][0]
            assert embedding.tolist() == [[tokens]]
            assert metadata == {"length": 2}
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_training_dataset_both_ordinary_hosts_cache_exact_outputs(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"training_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_training_dataset_torch import TrainingSource, TrainingInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (TrainingSource, TrainingInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "training-dataset-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["training-dataset-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (checkout / "src", *sorted(checkout.glob("packages/*/src")))
        composer = ServingComposer(
            worker_env={
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        *(str(p) for p in paths),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (package / "dinkster-pack.toml", fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "training-dataset-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("training-dataset-fixture.source"),
                    "encode": GraphNode(
                        "MakeTrainingDataset",
                        {name: Link("source", name) for name in ("images", "vae", "clip", "texts")},
                    ),
                    "inspect": GraphNode(
                        "training-dataset-fixture.inspect",
                        {name: Link("encode", name) for name in ("latents", "conditioning")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for iteration in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
                if iteration:
                    assert set(result.cached) == {
                        "source",
                        "encode",
                        "inspect",
                    }
        finally:
            await composer.close()

    asyncio.run(scenario())
