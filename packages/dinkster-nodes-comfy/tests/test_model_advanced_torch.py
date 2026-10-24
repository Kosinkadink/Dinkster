"""CPU behavior coverage for native advanced model patch nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.model_advanced import (
    MODEL_ADVANCED_NODES,
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

    def clone(self) -> Patcher:
        return Patcher(self.sampling)

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


def source() -> Patcher:
    from dinkster_inference.model_sampling import ModelSamplingDiscreteFlow

    return Patcher(ModelSamplingDiscreteFlow())


@pytest.mark.parametrize(
    "sampling", ("eps", "v_prediction", "x0", "img_to_img", "img_to_img_flow", "lcm")
)
def test_discrete_sampling_clones_and_installs_selected_prediction(sampling: str) -> None:
    original = source()
    result = ModelSamplingDiscrete.execute(model=original, sampling=sampling, zsnr=False)["MODEL"]
    assert result is not original
    assert result.sampling is not original.sampling


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


def test_pack_exports_exact_advanced_node_group() -> None:
    assert len(MODEL_ADVANCED_NODES) == 11
