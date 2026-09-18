"""Causal custom nodes preserve the dense seam's noise and output bytes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
import torch
from dinkster_compat_comfy import native_arm, native_residency
from dinkster_inference import (
    WAN21,
    WAN21_CAUSAL_INITIAL_LATENT_KEY,
    ComponentBinding,
    Conditioning,
    CustomSamplingRequest,
    MultiStreamLatent,
    ReconstructionRecipe,
    RuntimeKnobs,
    Wan21Config,
    WeightSourceBinding,
    WeightSourceRef,
    bind_component_conditioning,
    select_builtin_sampler,
)
from dinkster_inference_torch import wan21_causal
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.operations import CastOperations
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.wan21_runtime import (
    Wan21CausalDiffusionRuntime,
    wan21_text_conditioning_to_carrier,
)
from test_ltxv_node_exposure import _cpu_coordinator  # pyright: ignore[reportPrivateUsage]
from unet_fill import fill_state_dict


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("has_initial", (False, True))
@pytest.mark.parametrize("add_noise", (False, True))
@pytest.mark.parametrize("structural", (False, True))
def test_causal_custom_node_matches_dense_sampling(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    has_initial: bool,
    add_noise: bool,
    structural: bool,
) -> None:
    config = Wan21Config(
        hidden_size=8, ffn_hidden_size=16, num_heads=1, num_layers=1, time_freq_dim=4
    )
    monkeypatch.setattr(wan21_causal, "WAN21_CAUSAL_AR_1_3B", config)
    model = wan21_causal.Wan21CausalModel(config, operations=CastOperations(torch.float32))
    model.load_state_dict(
        fill_state_dict(
            [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
        ),
        strict=True,
    )
    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding("diffusion", WeightSourceRef("blake3:" + "1" * 64, "fixture", 1)),
        ),
        family_id=WAN21.id,
        component_identity=("family=dinkster.wan21", "profile=causal-ar-1.3b"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32", text_dtype="unloaded", vae_dtype="unloaded", fp8_matmul=False
        ),
    )
    runtime = Wan21CausalDiffusionRuntime(model, WAN21, runtime_identity=recipe.runtime_identity)
    handle = native_residency.NativeRuntimeHandle(
        runtime, "cpu", recipe=recipe, coordinator=_cpu_coordinator()
    )
    cond = Conditioning(torch.ones((1, 2, config.text_dim)))
    positive = bind_component_conditioning(
        wan21_text_conditioning_to_carrier(cond),
        ComponentBinding("umt5xxl", WAN21.id, recipe.runtime_identity),
    )
    samples = torch.zeros((1, 16, 3, 2, 2), dtype=dtype)
    initial = torch.full((1, 16, 1, 2, 2), 0.25, dtype=dtype) if has_initial else None
    latent: dict[str, object] = {
        "samples": MultiStreamLatent.from_pairs((("video", samples),)) if structural else samples,
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 4,
    }
    if initial is not None:
        latent[WAN21_CAUSAL_INITIAL_LATENT_KEY] = initial
    selection = select_builtin_sampler("ar_video", num_frame_per_block=1)
    sampler = torch_sampler_registry().get(selection.sampler_id)
    assert sampler is not None
    sigmas = (1.0, 0.5, 0.0)
    with torch.inference_mode():
        expected = runtime.sample_custom(
            samples,
            noise=prepare_noise(samples, 23) if add_noise else torch.zeros_like(samples),
            cond=cond,
            cfg=None,
            request=CustomSamplingRequest(sampler, selection.options, sigmas),
            seed=23 if add_noise else 0,
            initial_latent=initial,
        )
    output = native_arm.GenerationSamplerCustom.execute(
        model=handle,
        positive=positive,
        negative=None,
        cfg=1.0,
        latent_image=latent,
        sampler=selection,
        sigmas=native_arm._CustomSigmasValue(sigmas),  # pyright: ignore[reportPrivateUsage]
        add_noise=add_noise,
        noise_seed=23,
    )
    for key, reference in (
        ("output", expected.output),
        ("denoised_output", expected.denoised_output),
    ):
        actual = cast("Mapping[str, object]", output[key])
        tensor = actual["samples"]
        if structural:
            assert type(tensor) is MultiStreamLatent
            tensor = tensor.by_role("video")
        assert type(tensor) is torch.Tensor
        assert type(reference) is torch.Tensor
        assert tensor.dtype == reference.dtype
        assert torch.equal(tensor.view(torch.uint8), reference.view(torch.uint8))
        if not structural:
            assert actual["downscale_ratio_spacial"] == 8
            assert actual["downscale_ratio_temporal"] == 4
        if initial is not None:
            assert actual[WAN21_CAUSAL_INITIAL_LATENT_KEY] is initial
