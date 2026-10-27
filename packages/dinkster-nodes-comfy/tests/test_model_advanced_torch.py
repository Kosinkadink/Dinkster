"""CPU behavior coverage for native advanced model patch nodes."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_advanced import (
    MODEL_ADVANCED_NODES,
    ModelAttentionBackend,
    ModelComputeDtype,
    ModelNoiseScale,
    ModelSamplingAuraFlow,
    ModelSamplingContinuousEDM,
    ModelSamplingContinuousV,
    ModelSamplingDiscrete,
    ModelSamplingFlux,
    ModelSamplingSD3,
    ModelSamplingStableCascade,
    RescaleCFG,
)


class Patcher:
    def __init__(self, sampling: Any) -> None:
        self.model = SimpleNamespace(model_config=SimpleNamespace(sampling_settings={}))
        self.sampling = sampling
        self.cfg = None
        self.dtype = None
        self.attention = None

    def clone(self) -> Patcher:
        result = Patcher(self.sampling)
        result.cfg = self.cfg
        result.dtype = self.dtype
        result.attention = self.attention
        return result

    def add_object_patch(self, name: str, value: Any) -> None:
        assert name == "model_sampling"
        self.sampling = value

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return self.sampling

    def set_model_sampler_cfg_function(self, function: Any) -> None:
        self.cfg = function

    def set_model_compute_dtype(self, dtype: Any) -> None:
        self.dtype = dtype

    def set_model_optimized_attention(self, attention: Any) -> None:
        self.attention = attention


def source() -> Patcher:
    from dinkster_inference.model_sampling import ModelSamplingDiscreteFlow

    return Patcher(ModelSamplingDiscreteFlow())


@pytest.mark.parametrize("sampling", ("eps", "v_prediction", "x0", "img_to_img", "img_to_img_flow"))
def test_discrete_sampling_matches_pinned_prediction_math_on_asymmetric_tensors(
    sampling: str,
) -> None:
    original = source()
    patched = ModelSamplingDiscrete.execute(model=original, sampling=sampling, zsnr=False)["MODEL"]
    sigma = torch.tensor([0.25, 1.5])
    model_input = torch.tensor([[[[1.0, -2.0]]], [[[3.0, 5.0]]]])
    model_output = torch.tensor([[[[-4.0, 2.0]]], [[[1.0, -3.0]]]])
    expanded = sigma.view(2, 1, 1, 1)
    expected = {
        "eps": model_input - model_output * expanded,
        "v_prediction": model_input / (expanded**2 + 1.0)
        - model_output * expanded / (expanded**2 + 1.0).sqrt(),
        "x0": model_output,
        "img_to_img": model_output,
        "img_to_img_flow": model_output,
    }[sampling]
    assert patched is not original
    assert original.sampling is not patched.sampling
    assert torch.allclose(
        patched.sampling.calculate_denoised(sigma, model_output, model_input), expected
    )


def test_lcm_sampling_reduces_schedule_and_matches_lcm_output_math() -> None:
    patched = ModelSamplingDiscrete.execute(model=source(), sampling="lcm", zsnr=False)[
        "MODEL"
    ].sampling
    sigma = patched.sigmas[torch.tensor([3, 17])]
    model_input = torch.tensor([[[[1.0, -2.0]]], [[[3.0, 5.0]]]])
    model_output = torch.tensor([[[[-4.0, 2.0]]], [[[1.0, -3.0]]]])
    timestep = patched.timestep(sigma).view(2, 1, 1, 1)
    expanded = sigma.view(2, 1, 1, 1)
    x0 = model_input - model_output * expanded
    scaled = timestep * 10.0
    expected = scaled / (scaled**2 + 0.25).sqrt() * x0 + 0.25 / (scaled**2 + 0.25) * model_input
    assert len(patched.sigmas) == 50
    assert patched.skip_steps == 20
    assert torch.allclose(patched.calculate_denoised(sigma, model_output, model_input), expected)


def test_discrete_zsnr_changes_runtime_schedule_without_mutating_source() -> None:
    original = source()
    regular = ModelSamplingDiscrete.execute(model=original, sampling="eps", zsnr=False)[
        "MODEL"
    ].sampling
    zsnr = ModelSamplingDiscrete.execute(model=original, sampling="eps", zsnr=True)[
        "MODEL"
    ].sampling
    assert zsnr.zsnr is True
    assert regular.zsnr is False
    assert not torch.equal(regular.sigmas, zsnr.sigmas)
    assert original.sampling is not zsnr


@pytest.mark.parametrize(
    ("node", "kwargs", "attribute"),
    (
        (
            ModelSamplingContinuousEDM,
            {"sampling": "edm", "sigma_max": 16.0, "sigma_min": 0.25},
            "sigma_data",
        ),
        (
            ModelSamplingContinuousV,
            {"sampling": "v_prediction", "sigma_max": 16.0, "sigma_min": 0.25},
            "sigma_data",
        ),
        (ModelSamplingStableCascade, {"shift": 2.5}, "shift"),
        (ModelSamplingSD3, {"shift": 3.5}, "shift"),
        (ModelSamplingAuraFlow, {"shift": 1.5, "sampling": "img_to_img_velocity"}, "shift"),
        (
            ModelSamplingFlux,
            {"max_shift": 1.5, "base_shift": 0.5, "width": 1024, "height": 1024},
            "shift",
        ),
    ),
)
def test_sampling_nodes_install_runtime_sampling(
    node: Any, kwargs: dict[str, Any], attribute: str
) -> None:
    result = node.execute(model=source(), **kwargs)["MODEL"]
    assert hasattr(result.sampling, attribute)


@pytest.mark.parametrize("sampling", ("eps", "v_prediction", "edm", "cosmos_rflow"))
def test_continuous_edm_sampling_matches_prediction_seams(sampling: str) -> None:
    patched = ModelSamplingContinuousEDM.execute(
        model=source(), sampling=sampling, sigma_max=16.0, sigma_min=0.25
    )["MODEL"].sampling
    sigma = torch.tensor([0.25, 2.0])
    model_input = torch.tensor([[[[1.0, -2.0]]], [[[3.0, 5.0]]]])
    model_output = torch.tensor([[[[-4.0, 2.0]]], [[[1.0, -3.0]]]])
    expanded = sigma.view(2, 1, 1, 1)
    if sampling == "eps":
        expected = model_input - model_output * expanded
    elif sampling == "cosmos_rflow":
        flow_sigma = expanded / (expanded + 1.0)
        expected = model_input * (1.0 - flow_sigma) - model_output * flow_sigma
    else:
        sigma_data = 0.5 if sampling == "edm" else 1.0
        expected = model_input * sigma_data**2 / (expanded**2 + sigma_data**2)
        signed_output = model_output if sampling == "edm" else -model_output
        expected += signed_output * expanded * sigma_data / (expanded**2 + sigma_data**2).sqrt()
    assert torch.allclose(patched.calculate_denoised(sigma, model_output, model_input), expected)
    assert float(patched.sigma_min) == pytest.approx(0.25)
    assert float(patched.sigma_max) == pytest.approx(16.0)


def test_continuous_v_and_flow_variants_preserve_asymmetric_runtime_math() -> None:
    continuous = ModelSamplingContinuousV.execute(
        model=source(), sampling="v_prediction", sigma_max=16.0, sigma_min=0.25
    )["MODEL"].sampling
    assert torch.allclose(
        continuous.sigma(torch.tensor([0.25, 0.75])),
        torch.tan(torch.tensor([0.25, 0.75]) * math.pi / 2),
    )

    sd3 = ModelSamplingSD3.execute(model=source(), shift=3.5)["MODEL"].sampling
    aura = ModelSamplingAuraFlow.execute(model=source(), shift=1.5, sampling="img_to_img_velocity")[
        "MODEL"
    ].sampling
    time = torch.tensor([100.0, 750.0])
    expected_sd3 = 3.5 * (time / 1000) / (1 + 2.5 * (time / 1000))
    assert torch.allclose(sd3.sigma(time), expected_sd3)
    assert sd3.noise_scale == 1.0
    inputs = torch.tensor([[[[1.0, -2.0]]], [[[3.0, 5.0]]]])
    outputs = torch.tensor([[[[-4.0, 2.0]]], [[[1.0, -3.0]]]])
    assert torch.equal(
        aura.calculate_denoised(torch.tensor([0.1, 0.9]), outputs, inputs), inputs - outputs
    )


def test_flux_shift_uses_image_dimensions_and_stable_cascade_uses_requested_shift() -> None:
    flux = ModelSamplingFlux.execute(
        model=source(), max_shift=1.5, base_shift=0.5, width=2048, height=1024
    )["MODEL"].sampling
    slope = (1.5 - 0.5) / (4096 - 256)
    expected_shift = (2048 * 1024 / 256) * slope + 0.5 - slope * 256
    assert flux.shift == pytest.approx(expected_shift)
    timestep = torch.tensor([0.2, 0.8])
    expected_sigma = math.exp(expected_shift) / (math.exp(expected_shift) + (1 / timestep - 1))
    assert torch.allclose(flux.sigma(timestep), expected_sigma)

    cascade = ModelSamplingStableCascade.execute(model=source(), shift=2.5)["MODEL"].sampling
    assert cascade.shift == 2.5
    assert torch.all(cascade.sigma(torch.tensor([0.1, 0.9])) > 0)


def test_noise_scale_replaces_flow_sampling_and_preserves_shift() -> None:
    original = source()
    original.sampling.set_parameters(shift=2.0, multiplier=7.0)
    result = ModelNoiseScale.execute(model=original, noise_scale=3.0)["MODEL"]
    assert result.sampling.shift == 2.0
    assert result.sampling.multiplier == 7.0
    assert result.sampling.noise_scale == 3.0


def test_rescale_cfg_installs_flow_callback() -> None:
    flow_model = ModelSamplingSD3.execute(model=source(), shift=3.0)["MODEL"]
    result = RescaleCFG.execute(model=flow_model, multiplier=0.5)["MODEL"]
    value = result.cfg(
        {
            "input": torch.ones((1, 1, 2, 2)),
            "cond_scale": 2.0,
            "cond_denoised": torch.full((1, 1, 2, 2), 2.0),
            "uncond_denoised": torch.ones((1, 1, 2, 2)),
        }
    )
    assert torch.equal(value, torch.full((1, 1, 2, 2), -0.5))


def test_compute_dtype_installs_torch_dtype() -> None:
    result = ModelComputeDtype.execute(model=source(), dtype="bf16")["MODEL"]
    assert result.dtype is torch.bfloat16


def test_attention_backend_installs_runtime_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_nodes_comfy.model_advanced as advanced

    def pytorch_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return query + key - value

    module = SimpleNamespace(
        get_attention_function=lambda name, default=None: (
            pytorch_attention if name == "pytorch" else default
        )
    )
    actual_import = advanced.importlib.import_module
    initialized: list[bool] = []
    monkeypatch.setattr(advanced, "initialize_inference_process", lambda: initialized.append(True))
    monkeypatch.setattr(
        advanced.importlib,
        "import_module",
        lambda name: (
            module if name == "dinkster_inference.ldm.modules.attention" else actual_import(name)
        ),
    )
    original = source()
    patched = ModelAttentionBackend.execute(model=original, attention="pytorch attention")["MODEL"]
    query = torch.tensor([1.0, -2.0])
    key = torch.tensor([3.0, 5.0])
    value = torch.tensor([-4.0, 2.0])
    assert patched is not original
    assert patched.attention(query, key, value) is not None
    assert torch.equal(patched.attention(query, key, value), query + key - value)
    assert original.attention is None
    assert initialized == [True]


class AdvancedModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="advanced-model-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=source())


class AdvancedModelResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="advanced-model-fixture.result",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        sigma = torch.tensor([0.25])
        output = model.sampling.calculate_denoised(
            sigma, torch.tensor([[[[-4.0, 2.0]]]]), torch.tensor([[[[1.0, -2.0]]]])
        )
        return cls.outputs(value=float(output[0, 0, 0, 0]))


ADVANCED_MODEL_FIXTURE_NODES = (AdvancedModelSource, AdvancedModelResult)


@pytest.mark.parametrize("in_process", (True,))
def test_manifest_transports_model_sampling_patch_chain_in_both_hosting_modes(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            (tmp_path / "advanced_model_fixture_sources.py").write_text(
                "from test_model_advanced_torch import ADVANCED_MODEL_FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *ADVANCED_MODEL_FIXTURE_NODES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["advanced-model-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "advanced_model_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "advanced_model_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("advanced-model-fixture.source", {}),
                    "patch": GraphNode(
                        "ModelSamplingSD3", {"model": Link("source", "MODEL"), "shift": 3.0}
                    ),
                    "noise": GraphNode(
                        "ModelNoiseScale", {"model": Link("patch", "MODEL"), "noise_scale": 2.5}
                    ),
                    "result": GraphNode(
                        "advanced-model-fixture.result", {"model": Link("noise", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["result"])
                assert result.outputs["result"]["value"].resolve() == pytest.approx(2.0)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_pack_exports_exact_advanced_node_group() -> None:
    assert len(MODEL_ADVANCED_NODES) == 11
