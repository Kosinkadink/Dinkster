"""CPU contracts for HiDream-O1 shifted-grid prediction blending."""

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
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hidream_patches import (
    HiDreamO1PatchSeamSmoothing,
    HiDreamO1ReferenceImages,
)

from dinkster.compose import PackSpec, ServingComposer


class Sampling:
    multiplier = 3.0

    def percent_to_sigma(self, percent: float) -> float:
        return (1.0 - percent) * 10.0


class Model:
    def __init__(self) -> None:
        self.wrappers: dict[tuple[str, str], Any] = {}

    def clone(self) -> Model:
        result = Model()
        result.wrappers = self.wrappers.copy()
        return result

    def get_model_object(self, name: str) -> Sampling:
        assert name == "model_sampling"
        return Sampling()

    def add_wrapper_with_key(self, point: str, name: str, wrapper: Any) -> None:
        self.wrappers[point, name] = wrapper


def test_reference_order_append_and_shallow_identity() -> None:
    token, negative_token, marker = object(), object(), object()
    prior, negative_prior = torch.tensor([-7.0]), torch.tensor([-13.0])
    positive = [[token, {"reference_latents": [prior], "marker": marker}]]
    negative = [[negative_token, {"reference_latents": [negative_prior], "prompt_type": "old"}]]
    images = {
        name: torch.full((1, 2, 3, 3), float(index))
        for name, index in (
            ("extra", 201),
            ("image_100", 100),
            ("image_10", 10),
            ("image_2", 2),
            ("tail", 301),
        )
    }
    images["image_1"] = None
    result = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, images=images)
    for output, original, previous in (
        ("positive", positive, prior),
        ("negative", negative, negative_prior),
    ):
        entry = result[output][0]
        assert entry[0] is original[0][0] and entry[1] is not original[0][1]
        refs = entry[1]["reference_latents"]
        assert refs[0] is previous and refs[1] is None
        assert all(
            actual is images[name]
            for actual, name in zip(
                refs[2:], ("image_2", "image_10", "image_100", "extra", "tail"), strict=True
            )
        )
        assert len(original[0][1]["reference_latents"]) == 1
    assert result["positive"][0][1]["marker"] is marker
    assert result["negative"][0][1]["prompt_type"] == "negative"
    assert negative[0][1]["prompt_type"] == "old"
    for empty in (None, {}):
        unchanged = HiDreamO1ReferenceImages.execute(
            positive=positive, negative=negative, images=empty
        )
        assert unchanged["positive"] is positive and unchanged["negative"] is negative


class ReferenceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.references",
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                *(
                    OutputSpec(f"image{index}", TypeExpr.concrete(IMAGE_TYPE))
                    for index in (2, 10, 100)
                ),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[[torch.tensor([2.0]), {"reference_latents": [torch.tensor([-7.0])]}]],
            negative=[[torch.tensor([-3.0]), {"reference_latents": [torch.tensor([-13.0])]}]],
            **{f"image{index}": torch.full((1, 2, 3, 3), float(index)) for index in (2, 10, 100)},
        )


class ReferenceInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.reference-inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("positive", "negative")
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any) -> dict[str, object]:
        for conditioning, token, prior in ((positive, 2.0, -7.0), (negative, -3.0, -13.0)):
            assert torch.equal(conditioning[0][0], torch.tensor([token]))
            refs = conditioning[0][1]["reference_latents"]
            assert len(refs) == 4 and torch.equal(refs[0], torch.tensor([prior]))
            for reference, value in zip(refs[1:], (2.0, 10.0, 100.0), strict=True):
                assert torch.equal(reference, torch.full((1, 2, 3, 3), value))
        assert negative[0][1]["prompt_type"] == "negative"
        return cls.outputs(value=112.0)


def test_shifted_grid_average_preserves_edges_and_scales_strength() -> None:
    source = Model()
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=source,
        start_percent=0.2,
        end_percent=0.8,
        pattern="single_shift",
        passes="2",
        blend="average",
        strength=0.625,
    )["MODEL"]
    assert isinstance(patched, Model)
    assert patched is not source and source.wrappers == {}
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    bias = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 283

    def predict(value: torch.Tensor, sigma: torch.Tensor, *, offset: float) -> torch.Tensor:
        return value.square() + bias + offset

    sigma = torch.tensor([15.0])
    natural = predict(image, sigma, offset=0.75)
    shifted = image.square() + torch.roll(bias, (-16, -16), (-2, -1)) + 0.75
    expected_center = natural * 0.375 + ((natural + shifted) / 2) * 0.625
    result = wrapper(predict, image, sigma, offset=0.75)
    torch.testing.assert_close(
        result[..., 36:60, 36:92], expected_center[..., 36:60, 36:92], rtol=0, atol=0
    )
    assert torch.equal(result[..., :33, :], natural[..., :33, :])
    assert torch.equal(result[..., -33:, :], natural[..., -33:, :])
    assert torch.equal(result[..., :, :33], natural[..., :, :33])
    assert torch.equal(result[..., :, -33:], natural[..., :, -33:])


@pytest.mark.parametrize("pattern", ("single_shift", "symmetric"))
@pytest.mark.parametrize("blend", ("average", "window", "median"))
def test_ramped_smoothing_window_boundaries_and_pass_counts(pattern: str, blend: str) -> None:
    patched = HiDreamO1PatchSeamSmoothing.execute(
        model=Model(),
        start_percent=0.2,
        end_percent=0.8,
        pattern=pattern,
        passes="ramp_2_4_8",
        blend=blend,
        strength=1.0,
    )["MODEL"]
    assert isinstance(patched, Model)
    wrapper = next(iter(patched.wrappers.values()))
    image = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
    calls: list[torch.Tensor] = []

    def predict(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return value.square()

    for timestep, level in (
        (24.1, 0),
        (24.0, 2),
        (18.1, 2),
        (18.0, 4),
        (12.1, 4),
        (12.0, 8),
        (6.0, 8),
        (5.9, 0),
    ):
        calls.clear()
        actual = wrapper(predict, image, torch.tensor([timestep]))
        expected_calls = max(level, 1) + (pattern == "symmetric" and level > 0)
        assert len(calls) == expected_calls
        torch.testing.assert_close(actual, image.square(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "strength,start,end", ((0.0, 0.2, 0.8), (-0.5, 0.2, 0.8), (0.5, 0.7, 0.7), (0.5, 0.8, 0.2))
)
def test_disabled_smoothing_preserves_model_identity(
    strength: float, start: float, end: float
) -> None:
    model = Model()
    assert (
        HiDreamO1PatchSeamSmoothing.execute(
            model=model,
            start_percent=start,
            end_percent=end,
            pattern="single_shift",
            passes="2",
            blend="average",
            strength=strength,
        )["MODEL"]
        is model
    )


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.source", outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=Model())


class ModelInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hidream-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> dict[str, object]:
        wrapper = next(iter(model.wrappers.values()))
        x = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 997
        bias = torch.arange(96 * 128, dtype=torch.float32).reshape(1, 1, 96, 128) / 283

        def predict(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return value.square() + bias

        sigma = torch.tensor([15.0])
        natural = predict(x, sigma)
        shifted = x.square() + torch.roll(bias, (-16, -16), (-2, -1))
        expected = natural * 0.375 + ((natural + shifted) / 2) * 0.625
        result = wrapper(predict, x, sigma)
        torch.testing.assert_close(
            result[..., 36:60, 36:92], expected[..., 36:60, 36:92], rtol=0, atol=0
        )
        assert torch.equal(result[..., :33, :], natural[..., :33, :])
        assert torch.equal(wrapper(predict, x, torch.tensor([24.1])), natural)
        return cls.outputs(value=float(result.sum()))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_hidream_smoothing_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"hidream_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_hidream_patches_torch import (\n"
            "    ModelSource, ModelInspect, ReferenceSource, ReferenceInspect,\n)\n"
            "from dinkster_inference_wire import register_inference_types\n"
            "NODES = (ModelSource, ModelInspect, ReferenceSource, ReferenceInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "hidream-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["hidream-fixture"]\n[pack.entry]\n'
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
                        worker_group="hidream-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hidream-fixture.source"),
                    "patch": GraphNode(
                        "HiDreamO1PatchSeamSmoothing",
                        {
                            "model": Link("source", "model"),
                            "start_percent": 0.2,
                            "end_percent": 0.8,
                            "pattern": "single_shift",
                            "passes": "2",
                            "blend": "average",
                            "strength": 0.625,
                        },
                    ),
                    "inspect": GraphNode(
                        "hidream-fixture.inspect", {"model": Link("patch", "MODEL")}
                    ),
                    "references": GraphNode("hidream-fixture.references"),
                    "append": GraphNode(
                        "HiDreamO1ReferenceImages",
                        {
                            "positive": Link("references", "positive"),
                            "negative": Link("references", "negative"),
                            **{
                                f"images.image_{index}": Link("references", f"image{index}")
                                for index in (100, 10, 2)
                            },
                        },
                    ),
                    "inspect_references": GraphNode(
                        "hidream-fixture.reference-inspect",
                        {
                            "positive": Link("append", "positive"),
                            "negative": Link("append", "negative"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            results = []
            for _ in range(2):
                outputs = (await engine.run(graph, ["inspect", "inspect_references"])).outputs
                results.append(outputs["inspect"]["value"].resolve())
                assert outputs["inspect_references"]["value"].resolve() == 112.0
            assert results[0] == results[1] and results[0] > 0
        finally:
            await composer.close()

    asyncio.run(scenario())
