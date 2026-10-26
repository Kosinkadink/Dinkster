"""CPU behavior for Bernini conditioning."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
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
from dinkster_nodes_comfy.bernini_conditioning import BerniniConditioning

from dinkster.compose import PackSpec, ServingComposer


class ControlledVAE:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(image.clone())
        return image.mean(dim=(0, 1, 2), keepdim=True)


def test_bernini_encodes_ordered_context_and_preserves_sources(monkeypatch: Any) -> None:
    import dinkster_nodes_comfy.bernini_conditioning as module

    monkeypatch.setattr(module, "initialize_inference_process", lambda: None)
    management = type("Management", (), {"intermediate_device": staticmethod(lambda: "cpu")})
    actual_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name, package=None: (
            management
            if name == "dinkster_inference.model_management"
            else actual_import(name, package)
        ),
    )
    vae = ControlledVAE()
    positive = [[torch.tensor([1.0]), {"source": "positive"}]]
    negative = [[torch.tensor([2.0]), {"source": "negative"}]]
    source = torch.arange(3 * 9 * 13 * 4, dtype=torch.float32).reshape(3, 9, 13, 4)
    reference_video = torch.arange(5 * 10 * 20 * 4, dtype=torch.float32).reshape(5, 10, 20, 4)
    references = {
        "reference_image_10": torch.full((1, 12, 6, 4), 10.0),
        "reference_image_2": torch.full((2, 7, 19, 4), 2.0),
        "reference_image_0": None,
    }
    result = BerniniConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=32,
        length=2,
        batch_size=2,
        source_video=source.numpy(),
        reference_video=reference_video.numpy(),
        reference_images=references,
        ref_max_size=16,
    )
    context = result["positive"][0][1]["context_latents"]
    assert positive[0][1] == {"source": "positive"}
    assert negative[0][1] == {"source": "negative"}
    assert result["positive"][0][0] is positive[0][0]
    assert result["negative"][0][0] is negative[0][0]
    assert len(context) == 5
    assert [tuple(image.shape) for image in vae.inputs] == [
        (2, 32, 16, 3),
        (2, 16, 16, 3),
        (1, 16, 16, 3),
        (1, 16, 16, 3),
        (1, 16, 16, 3),
    ]
    assert result["latent"]["samples"].shape == (2, 16, 1, 4, 2)
    no_context = BerniniConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=16,
        length=5,
        batch_size=1,
    )
    assert no_context["positive"] is positive
    assert no_context["negative"] is negative
    assert no_context["latent"]["samples"].shape == (1, 16, 2, 2, 2)


class ContextSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "bernini-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("source", IMAGE_TYPE),
                    ("video", IMAGE_TYPE),
                    ("first", IMAGE_TYPE),
                    ("last", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        source = torch.full((3, 9, 13, 4), 0.25)
        source[..., 3] = 19
        video = torch.stack([torch.full((10, 20, 4), value) for value in (0.5, 1.5, 7, 8, 9)])
        video[..., 3] = 23
        first = torch.full((1, 12, 6, 4), 10.0)
        last = torch.stack([torch.full((7, 19, 4), value) for value in (2.0, 3.0)])
        return cls.outputs(
            positive=[[torch.tensor([1.0]), {"source": "positive"}]],
            negative=[[torch.tensor([-2.0]), {"source": "negative"}]],
            vae=ControlledVAE(),
            source=source.numpy(),
            video=video.numpy(),
            first=first.numpy(),
            last=last.numpy(),
        )


class ContextInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "bernini-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("empty_positive", CONDITIONING_TYPE),
                    ("empty_negative", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                    ("empty_latent", LATENT_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        negative: Any,
        empty_positive: Any,
        empty_negative: Any,
        latent: Any,
        empty_latent: Any,
        vae: Any,
    ) -> dict[str, object]:
        assert positive[0][1]["source"] == "positive"
        assert negative[0][1]["source"] == "negative"
        assert empty_positive[0][1] == {"source": "positive"}
        assert empty_negative[0][1] == {"source": "negative"}
        p = positive[0][1]["context_latents"]
        n = negative[0][1]["context_latents"]
        assert len(p) == len(n) == 5
        for actual, other, expected in zip(p, n, (0.25, 1.0, 10.0, 2.0, 3.0), strict=True):
            assert torch.equal(actual, torch.full((1, 1, 1, 3), expected))
            assert torch.equal(actual, other)
        assert [tuple(image.shape) for image in vae.inputs] == [
            (2, 32, 16, 3),
            (2, 16, 16, 3),
            (1, 16, 16, 3),
            (1, 16, 16, 3),
            (1, 16, 16, 3),
        ]
        samples = materialize_latent(latent)["samples"]
        empty_samples = materialize_latent(empty_latent)["samples"]
        assert samples.shape == (2, 16, 1, 4, 2) and torch.count_nonzero(samples) == 0
        assert empty_samples.shape == (1, 16, 2, 2, 2)
        assert torch.count_nonzero(empty_samples) == 0
        return cls.outputs(value=sum(float(value.sum()) for value in p))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_bernini_context_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"bernini_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_bernini_conditioning_torch import ContextSource, ContextInspect\n"
            "from dinkster_inference_wire import register_inference_types\n"
            "NODES = (ContextSource, ContextInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "bernini-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["bernini-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(__file__).parent),
                        os.environ.get("PYTHONPATH", ""),
                    )
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="bernini-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            base = {
                "positive": Link("source", "positive"),
                "negative": Link("source", "negative"),
                "vae": Link("source", "vae"),
                "width": 16,
                "height": 32,
                "length": 2,
                "batch_size": 2,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("bernini-fixture.source"),
                    "context": GraphNode(
                        "BerniniConditioning",
                        {
                            **base,
                            "source_video": Link("source", "source"),
                            "reference_video": Link("source", "video"),
                            "reference_images.reference_image_2": Link("source", "first"),
                            "reference_images.reference_image_7": Link("source", "last"),
                            "ref_max_size": 16,
                        },
                    ),
                    "empty": GraphNode(
                        "BerniniConditioning", {**base, "height": 16, "length": 5, "batch_size": 1}
                    ),
                    "inspect": GraphNode(
                        "bernini-fixture.inspect",
                        {
                            "positive": Link("context", "positive"),
                            "negative": Link("context", "negative"),
                            "latent": Link("context", "latent"),
                            "empty_positive": Link("empty", "positive"),
                            "empty_negative": Link("empty", "negative"),
                            "empty_latent": Link("empty", "latent"),
                            "vae": Link("source", "vae"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            values = [
                (await engine.run(graph, ["inspect"])).outputs["inspect"]["value"].resolve()
                for _ in range(2)
            ]
            assert values == [48.75, 48.75]
        finally:
            await composer.close()

    asyncio.run(scenario())
