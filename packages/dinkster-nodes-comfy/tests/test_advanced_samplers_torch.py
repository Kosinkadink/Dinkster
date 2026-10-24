"""CPU behavior and ordinary-host transport for advanced native samplers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
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


class _SamplingSpace:
    noise_scale = 1.0
    sigma_max = 2.0

    def get_model_object(self, name: str) -> _SamplingSpace:
        assert name == "model_sampling"
        return self

    def noise_scaling(
        self, sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor, *_: object
    ) -> torch.Tensor:
        return latent + sigma * noise

    def inverse_noise_scaling(self, _sigma: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _RuntimeSamplingModel:
    def __init__(self) -> None:
        self.model_sampling = _SamplingSpace()
        self.model_patcher = SimpleNamespace(
            model=object(), get_model_object=self.model_sampling.get_model_object
        )
        self.inner_model = SimpleNamespace(
            model_sampling=self.model_sampling, model_patcher=self.model_patcher
        )
        self.cfg = 1.5
        self.calls: list[tuple[torch.Tensor, torch.Tensor, dict[str, object], int | None]] = []

    def __call__(
        self,
        value: torch.Tensor,
        sigma: torch.Tensor,
        *,
        model_options: dict[str, object],
        seed: int | None,
    ) -> torch.Tensor:
        self.calls.append((value.clone(), sigma.clone(), model_options, seed))
        denoised = value * 0.5 + sigma.reshape((-1, 1, 1, 1)) * 0.1
        uncond_denoised = value * -0.25 + sigma.reshape((-1, 1, 1, 1)) * 0.05
        for function in model_options.get("sampler_post_cfg_function", []):
            denoised = function({"denoised": denoised, "uncond_denoised": uncond_denoised})
        return denoised


def _sample(
    sampler: Any,
    model: _RuntimeSamplingModel,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    latent: torch.Tensor,
    callback_records: list[tuple[int, torch.Tensor, torch.Tensor, int]],
    *,
    model_options: dict[str, object],
) -> torch.Tensor:
    return sampler.sample(
        model,
        sigmas,
        {"seed": 739, "model_options": model_options},
        lambda index, denoised, value, total: callback_records.append(
            (index, denoised.clone(), value.clone(), total)
        ),
        noise,
        latent_image=latent,
        disable_pbar=True,
    )


def _denoised(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return value * 0.5 + sigma * 0.1


def _uncond_denoised(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return value * -0.25 + sigma * 0.05


class AdvancedSamplerConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        sampler = TypeExpr.concrete(SAMPLER_TYPE)
        return NodeSchema(
            node_type="advanced-sampler.consumer",
            inputs=(
                InputSpec("upscale", sampler),
                InputSpec("lcm", sampler),
                InputSpec("regular", sampler),
                InputSpec("alternative", sampler),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, upscale: Any, lcm: Any, regular: Any, alternative: Any
    ) -> dict[str, object]:
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
        assert regular.sampler_function.__name__ == "sample_euler_cfg_pp"
        assert alternative.sampler_function is sample_euler_cfg_pp_alternative

        noise = torch.tensor(
            [[[[0.2, -0.4, 1.1], [-0.8, 0.6, -1.3]]]], dtype=torch.float32, requires_grad=True
        )
        latent = torch.tensor(
            [[[[1.2, -0.7, 0.5], [0.25, -1.5, 0.9]]]], dtype=torch.float32, requires_grad=True
        )
        original_noise = noise.detach().clone()
        original_latent = latent.detach().clone()

        upscale_records: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        upscale_model = _RuntimeSamplingModel()
        torch.manual_seed(918)
        upscale_result = _sample(
            upscale,
            upscale_model,
            torch.tensor((1.0, 0.5, 0.25, 0.0)),
            noise,
            latent,
            upscale_records,
            model_options={"marker": "upscale"},
        )
        torch.manual_seed(918)
        upscale_expected = original_latent + original_noise
        for index, size in enumerate(((2, 4), (3, 4), None)):
            sigma = (1.0, 0.5, 0.25)[index]
            upscale_expected = _denoised(upscale_expected, torch.tensor(sigma))
            if size is not None:
                upscale_expected = torch.nn.functional.interpolate(
                    upscale_expected, size=size, mode="nearest-exact"
                )
            next_sigma = (0.5, 0.25, 0.0)[index]
            if next_sigma:
                upscale_expected += next_sigma * torch.randn_like(upscale_expected)
        torch.testing.assert_close(upscale_result, upscale_expected)
        assert [record[0] for record in upscale_records] == [0, 1, 2]
        assert [record[3] for record in upscale_records] == [3, 3, 3]
        assert [tuple(record[2].shape) for record in upscale_records] == [
            (1, 1, 2, 3),
            (1, 1, 2, 4),
            (1, 1, 3, 4),
        ]
        torch.testing.assert_close(
            upscale_records[0][1], _denoised(original_latent + original_noise, torch.tensor(1.0))
        )
        assert all(options["marker"] == "upscale" for _, _, options, _ in upscale_model.calls)

        lcm_records: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        lcm_model = _RuntimeSamplingModel()
        lcm_result = _sample(
            lcm,
            lcm_model,
            torch.tensor((1.0, 0.5, 0.0)),
            noise,
            latent,
            lcm_records,
            model_options={"marker": "lcm"},
        )
        lcm_expected = original_latent + original_noise
        lcm_expected = _denoised(lcm_expected, torch.tensor(1.0))
        lcm_noise = torch.randn(original_noise.shape, generator=torch.Generator().manual_seed(740))
        lcm_noise = lcm_noise.clamp(min=-0.25 * lcm_noise.std(), max=0.25 * lcm_noise.std())
        lcm_expected += 0.5 * lcm_noise * 0.375
        lcm_expected = _denoised(lcm_expected, torch.tensor(0.5))
        torch.testing.assert_close(lcm_result, lcm_expected)
        assert [record[0] for record in lcm_records] == [0, 1]
        assert [record[3] for record in lcm_records] == [2, 2]
        torch.testing.assert_close(
            lcm_records[0][1], _denoised(original_latent + original_noise, torch.tensor(1.0))
        )
        assert all(options["marker"] == "lcm" for _, _, options, _ in lcm_model.calls)

        cfg_sigmas = torch.tensor((2.0, 1.0, 0.0))
        regular_options = {"marker": "regular"}
        regular_records: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        regular_model = _RuntimeSamplingModel()
        regular_result = _sample(
            regular,
            regular_model,
            cfg_sigmas,
            noise,
            latent,
            regular_records,
            model_options=regular_options,
        )
        regular_initial = original_latent + 2.0 * original_noise
        regular_first = _denoised(regular_initial, torch.tensor(2.0))
        regular_step = (
            regular_first
            + (regular_initial - _uncond_denoised(regular_initial, torch.tensor(2.0))) / 2.0
        )
        regular_expected = _denoised(regular_step, torch.tensor(1.0))
        torch.testing.assert_close(regular_result, regular_expected)
        assert [record[0] for record in regular_records] == [0, 1]
        assert [record[3] for record in regular_records] == [2, 2]
        torch.testing.assert_close(regular_records[0][1], regular_first)
        assert regular_options == {"marker": "regular"}
        assert all(options["marker"] == "regular" for _, _, options, _ in regular_model.calls)

        alternative_options = {"marker": "alternative"}
        alternative_records: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        alternative_model = _RuntimeSamplingModel()
        alternative_result = _sample(
            alternative,
            alternative_model,
            cfg_sigmas,
            noise,
            latent,
            alternative_records,
            model_options=alternative_options,
        )
        alternative_initial = regular_initial
        alternative_first = _denoised(alternative_initial, torch.tensor(2.0))
        alternative_step = (
            alternative_initial
            - (
                alternative_initial
                + _uncond_denoised(alternative_initial, torch.tensor(2.0))
                - 2.0 * alternative_first
            )
            / 2.0
        )
        alternative_second = _denoised(alternative_step, torch.tensor(1.0))
        alternative_expected = 2.0 * alternative_second - _uncond_denoised(
            alternative_step, torch.tensor(1.0)
        )
        torch.testing.assert_close(alternative_result, alternative_expected)
        assert [record[0] for record in alternative_records] == [0, 1]
        assert [record[3] for record in alternative_records] == [2, 2]
        torch.testing.assert_close(alternative_records[0][1], alternative_first)
        assert alternative_options == {"marker": "alternative"}
        assert all(
            options["marker"] == "alternative" for _, _, options, _ in alternative_model.calls
        )
        assert not torch.equal(regular_result, alternative_result)
        assert not upscale_result.requires_grad
        assert not lcm_result.requires_grad
        assert not regular_result.requires_grad
        assert not alternative_result.requires_grad
        assert torch.equal(noise.detach(), original_noise)
        assert torch.equal(latent.detach(), original_latent)
        return cls.outputs(summary="public-sample-contracts")


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
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (AdvancedSamplerConsumer,)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "fixture-dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "advanced-sampler-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["advanced-sampler"]\n'
                f'[pack.entry]\nnodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            specs = (
                (
                    PackSpec(manifest=native_manifest, in_process=True),
                    PackSpec(manifest=fixture_manifest, in_process=True),
                )
                if in_process
                else (
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="advanced-sampler-fixture",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="advanced-sampler-fixture",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
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
                    "regular": GraphNode("SamplerEulerCFGpp", {"version": "regular"}),
                    "alternative": GraphNode("SamplerEulerCFGpp", {"version": "alternative"}),
                    "consumer": GraphNode(
                        "advanced-sampler.consumer",
                        {
                            "upscale": Link("upscale", "SAMPLER"),
                            "lcm": Link("lcm", "SAMPLER"),
                            "regular": Link("regular", "SAMPLER"),
                            "alternative": Link("alternative", "SAMPLER"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "public-sample-contracts"
                if attempt:
                    assert set(result.cached) == {
                        "upscale",
                        "lcm",
                        "regular",
                        "alternative",
                        "consumer",
                    }
        finally:
            await composer.close()

    asyncio.run(scenario())
