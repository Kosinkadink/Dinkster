"""CPU behavior and ordinary-host transport for advanced native samplers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SAMPLER_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.advanced_samplers import (
    ADVANCED_SAMPLER_NODES,
    SamplerEulerCFGpp,
    SamplerLCM,
    SamplerLCMUpscale,
    sample_euler_cfg_pp_alternative,
    sample_lcm_upscale,
)

from dinkster.compose import PackSpec, ServingComposer


def _sampler(node: type[Any], **inputs: Any) -> Any:
    return node.execute(**inputs)["SAMPLER"]


def test_advanced_sampler_schemas_remain_torch_free_until_execution() -> None:
    assert tuple(node.define_schema().node_type for node in ADVANCED_SAMPLER_NODES) == (
        "SamplerLCMUpscale",
        "SamplerLCM",
        "SamplerEulerCFGpp",
    )
    upscale = SamplerLCMUpscale.define_schema()
    assert upscale.category == "model/sampling/samplers"
    assert [(item.id, item.default) for item in upscale.inputs] == [
        ("scale_ratio", 1.0),
        ("scale_steps", -1),
        ("upscale_method", "bislerp"),
    ]
    cfgpp = SamplerEulerCFGpp.define_schema()
    assert cfgpp.category == "experimental"
    assert cfgpp.display_name == "SamplerEulerCFG++"


def test_lcm_and_regular_cfgpp_use_the_pinned_runtime_samplers() -> None:
    initialize_inference_process()
    lcm = _sampler(SamplerLCM, s_noise=0.375, s_noise_end=1.75, noise_clip_std=0.25)
    regular = _sampler(SamplerEulerCFGpp, version="regular")
    alternative = _sampler(SamplerEulerCFGpp, version="alternative")

    assert lcm.sampler_function.__name__ == "sample_lcm"
    assert lcm.extra_options == {
        "s_noise": 0.375,
        "s_noise_end": 1.75,
        "noise_clip_std": 0.25,
    }
    assert regular.sampler_function.__name__ == "sample_euler_cfg_pp"
    assert alternative.sampler_function is sample_euler_cfg_pp_alternative
    assert alternative.extra_options == {}


def test_lcm_upscale_preserves_callback_order_noise_and_source_tensor() -> None:
    source = torch.tensor([[[[0.25, -0.5, 1.0], [1.5, -1.25, 0.75]]]], dtype=torch.float32)
    before = source.clone()
    sigmas = torch.tensor((1.0, 0.5, 0.25, 0.0), dtype=torch.float32)
    callback_records: list[dict[str, Any]] = []
    model_inputs: list[tuple[tuple[int, ...], torch.Tensor]] = []

    def model(value: torch.Tensor, sigma: torch.Tensor, **extra_args: Any) -> torch.Tensor:
        assert extra_args == {"marker": "preserved"}
        model_inputs.append((tuple(value.shape), sigma.clone()))
        return value * 0.5 + sigma.reshape((-1, 1, 1, 1))

    torch.manual_seed(918)
    actual = sample_lcm_upscale(
        model,
        source,
        sigmas,
        extra_args={"marker": "preserved"},
        callback=callback_records.append,
        total_upscale=2.0,
        upscale_method="nearest-exact",
        upscale_steps=2,
    )

    torch.manual_seed(918)
    expected = source.clone()
    expected_shapes = ((1, 1, 2, 3), (1, 1, 3, 4), (1, 1, 4, 6))
    expected_sigmas = ((1.0,), (0.5,), (0.25,))
    for index, size in enumerate(((3, 4), (4, 6), None)):
        denoised = expected * 0.5 + sigmas[index]
        if size is not None:
            height, width = size
            expected = torch.nn.functional.interpolate(
                denoised, size=(height, width), mode="nearest-exact"
            )
        else:
            expected = denoised
        if sigmas[index + 1] > 0:
            expected += sigmas[index + 1] * torch.randn_like(expected)

    assert torch.equal(source, before)
    assert [shape for shape, _ in model_inputs] == list(expected_shapes)
    assert [tuple(sigma.tolist()) for _, sigma in model_inputs] == list(expected_sigmas)
    assert [record["i"] for record in callback_records] == [0, 1, 2]
    assert [tuple(record["x"].shape) for record in callback_records] == list(expected_shapes)
    assert [record["sigma_hat"] for record in callback_records] == [sigmas[0], sigmas[1], sigmas[2]]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("sigmas", "scale_steps", "expected_shapes"),
    (
        (torch.tensor((1.0, 0.0)), -1, ((1, 1, 2, 3),)),
        (torch.tensor((1.0, 0.5, 0.0)), 1000, ((1, 1, 2, 3), (1, 1, 3, 4))),
        (torch.tensor((0.0,)), 0, ()),
    ),
)
def test_lcm_upscale_scale_step_boundaries(
    sigmas: torch.Tensor, scale_steps: int, expected_shapes: tuple[tuple[int, ...], ...]
) -> None:
    sampler = _sampler(
        SamplerLCMUpscale,
        scale_ratio=2.0,
        scale_steps=scale_steps,
        upscale_method="nearest-exact",
    )
    seen: list[tuple[int, ...]] = []

    def model(value: torch.Tensor, _sigma: torch.Tensor, **_extra: Any) -> torch.Tensor:
        seen.append(tuple(value.shape))
        return value

    torch.manual_seed(12)
    result = sampler.sampler_function(
        model,
        torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3),
        sigmas,
        extra_args={},
        disable=True,
        **sampler.extra_options,
    )
    assert seen == list(expected_shapes)
    if len(sigmas) == 1:
        assert tuple(result.shape) == (1, 1, 2, 3)
    elif scale_steps == 1000:
        assert tuple(result.shape) == (1, 1, 3, 5)
    else:
        assert tuple(result.shape) == (1, 1, 4, 6)


def test_alternative_cfgpp_routes_unconditioned_denoised_through_post_cfg_callback() -> None:
    source = torch.tensor([[[[2.0, -4.0], [1.0, 3.0]]]], dtype=torch.float32)
    before = source.clone()
    original_options = {"existing": "kept"}
    callback_records: list[dict[str, Any]] = []
    received_options: list[dict[str, Any]] = []

    def model(value: torch.Tensor, _sigma: torch.Tensor, **extra_args: Any) -> torch.Tensor:
        options = extra_args["model_options"]
        received_options.append(options)
        assert options["existing"] == "kept"
        assert options["disable_cfg1_optimization"] is True
        return options["sampler_post_cfg_function"][0](
            {"uncond_denoised": value * 0.25, "denoised": value * 0.5}
        )

    actual = sample_euler_cfg_pp_alternative(
        model,
        source,
        torch.tensor((2.0, 1.0, 0.0)),
        extra_args={"model_options": original_options},
        callback=callback_records.append,
    )

    assert torch.equal(source, before)
    assert original_options == {"existing": "kept"}
    assert len(received_options) == 2
    assert received_options[0] is received_options[1]
    assert [record["i"] for record in callback_records] == [0, 1]
    assert [record["sigma_hat"] for record in callback_records] == [
        torch.tensor(2.0),
        torch.tensor(1.0),
    ]
    torch.testing.assert_close(actual, source * 0.65625)


class AdvancedSamplerConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        sampler = TypeExpr.concrete(SAMPLER_TYPE)
        return NodeSchema(
            node_type="advanced-sampler.consumer",
            inputs=(
                InputSpec("upscale", sampler),
                InputSpec("lcm", sampler),
                InputSpec("cfgpp", sampler),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, upscale: Any, lcm: Any, cfgpp: Any) -> dict[str, object]:
        assert upscale.sampler_function is sample_lcm_upscale
        assert upscale.extra_options == {
            "total_upscale": 1.5,
            "upscale_steps": None,
            "upscale_method": "nearest-exact",
        }
        assert lcm.sampler_function.__name__ == "sample_lcm"
        assert lcm.extra_options == {
            "s_noise": 0.375,
            "s_noise_end": 1.75,
            "noise_clip_std": 0.25,
        }
        assert cfgpp.sampler_function is sample_euler_cfg_pp_alternative
        return cls.outputs(
            summary="|".join(
                (
                    upscale.sampler_function.__name__,
                    lcm.sampler_function.__name__,
                    cfgpp.sampler_function.__name__,
                )
            )
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_advanced_samplers_cross_ordinary_manifest_hosts_twice_with_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            module = f"advanced_samplers_fixture_{int(in_process)}"
            (tmp_path / f"{module}.py").write_text(
                "from test_advanced_samplers_torch import AdvancedSamplerConsumer\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, AdvancedSamplerConsumer)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            source = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            manifest.write_text(
                source.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["advanced-sampler",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "upscale": GraphNode(
                        "SamplerLCMUpscale",
                        {
                            "scale_ratio": 1.5,
                            "scale_steps": -1,
                            "upscale_method": "nearest-exact",
                        },
                    ),
                    "lcm": GraphNode(
                        "SamplerLCM",
                        {"s_noise": 0.375, "s_noise_end": 1.75, "noise_clip_std": 0.25},
                    ),
                    "cfgpp": GraphNode("SamplerEulerCFGpp", {"version": "alternative"}),
                    "consumer": GraphNode(
                        "advanced-sampler.consumer",
                        {
                            "upscale": Link("upscale", "SAMPLER"),
                            "lcm": Link("lcm", "SAMPLER"),
                            "cfgpp": Link("cfgpp", "SAMPLER"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = "sample_lcm_upscale|sample_lcm|sample_euler_cfg_pp_alternative"
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
                if attempt:
                    assert set(result.cached) == {"upscale", "lcm", "cfgpp", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
