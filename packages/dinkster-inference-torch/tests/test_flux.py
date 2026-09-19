"""Stage 5 slice 6: the native classic Flux diffusion transformer.

Every golden in goldens/flux_goldens.json was produced by RUNNING the
reference Flux (comfy/ldm/flux/model.py @ the audited baseline,
tools/gen_flux_goldens.py) with attention forced to pytorch SDPA and
RoPE to the reference's pure-torch path. Weights come from the shared
deterministic hash (unet_fill.py - Flux's only rank-1 weights are the
QKNorm RMS scales, so the rank rule holds) and inputs from its
``hashed_input`` namespace, so both sides run bit-identical
parameters and activations without storing megabytes.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import (
    FLUX_DEV_CONFIG,
    FLUX_SCHNELL_CONFIG,
    FluxConfig,
    flux_layout,
    normalize_flux_keys,
)
from dinkster_inference_torch import Flux, flux_timestep_embedding, rope, select_attention
from dinkster_inference_torch.flux import apply_rope, apply_rope1
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "flux_goldens.json")
GATED_GOLDEN = load_platform_golden(Path(__file__).parent / "goldens" / "flux_gated_goldens.json")
VECTOR_FREE_GOLDEN = load_platform_golden(
    Path(__file__).parent / "goldens" / "flux_vector_free_goldens.json"
)

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> FluxConfig:
    spec = dict(GOLDENS["cases"][case]["config"])
    spec["axes_dim"] = tuple(spec["axes_dim"])
    return FluxConfig(**spec)


def build_model(case: str) -> Flux:
    model = Flux(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def gated_config() -> FluxConfig:
    spec = dict(GATED_GOLDEN["case"]["config"])
    spec["axes_dim"] = tuple(spec["axes_dim"])
    return FluxConfig(**spec)


def build_gated_model() -> Flux:
    model = Flux(gated_config())
    model.load_state_dict(fill_state_dict(GATED_GOLDEN["case"]["state_dict"]), strict=True)
    return model


def vector_free_config() -> FluxConfig:
    spec = dict(VECTOR_FREE_GOLDEN["case"]["config"])
    spec["axes_dim"] = tuple(spec["axes_dim"])
    spec["txt_ids_dims"] = tuple(spec["txt_ids_dims"])
    return FluxConfig(**spec)


def build_vector_free_model() -> Flux:
    model = Flux(vector_free_config())
    result = model.load_state_dict(
        fill_state_dict(VECTOR_FREE_GOLDEN["case"]["state_dict"]),
        strict=True,
        assign=True,
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    return model


def vector_free_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    spec = VECTOR_FREE_GOLDEN["case"]
    config = spec["config"]
    batch = spec["batch"]
    return (
        hashed_input(
            f"{spec['name']}:x",
            (batch, config["in_channels"], spec["height"], spec["width"]),
        ),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input(
            f"{spec['name']}:context",
            (batch, spec["context_len"], config["context_in_dim"]),
        ),
        None,
        None,
    )


def gated_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GATED_GOLDEN["case"]
    config = spec["config"]
    batch = spec["batch"]
    return (
        hashed_input(
            f"{spec['name']}:x",
            (batch, config["in_channels"], spec["height"], spec["width"]),
        ),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input(
            f"{spec['name']}:context",
            (batch, spec["context_len"], config["context_in_dim"]),
        ),
        hashed_input(f"{spec['name']}:y", (batch, config["vec_in_dim"])),
        torch.tensor(spec["guidance"], dtype=torch.float32),
    )


def case_inputs(
    case: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """The generator's exact inputs, rebuilt from the shared hash."""
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    batch = spec["batch"]
    x = hashed_input(
        f"{case}:x",
        (batch, config["in_channels"], spec["height"], spec["width"]),
    )
    timesteps = torch.tensor(spec["timesteps"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context",
        (batch, spec["context_len"], config["context_in_dim"]),
    )
    y = hashed_input(f"{case}:y", (batch, config["vec_in_dim"]))
    guidance = None
    if spec["guidance"] is not None:
        guidance = torch.tensor(spec["guidance"], dtype=torch.float32)
    return x, timesteps, context, y, guidance


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in flux_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("flux_dev", FLUX_DEV_CONFIG),
        ("flux_dev_txt_norm", replace(FLUX_DEV_CONFIG, txt_norm=True)),
        ("flux_schnell", FLUX_SCHNELL_CONFIG),
    ],
)
def test_full_size_module_matches_reference_layout(name: str, config: FluxConfig) -> None:
    """The real dev/schnell architectures, constructed on the meta
    device (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = Flux(config)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


def test_full_size_gated_ovis_module_matches_reference_layout() -> None:
    config = dict(GATED_GOLDEN["config"])
    config["axes_dim"] = tuple(config["axes_dim"])
    with torch.device("meta"):
        model = Flux(FluxConfig(**config))
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert ours == [(key, list(shape)) for key, shape in GATED_GOLDEN["layout"]]


def test_full_size_vector_free_ovis_module_matches_reference_layout() -> None:
    config = dict(VECTOR_FREE_GOLDEN["config"])
    config["axes_dim"] = tuple(config["axes_dim"])
    config["txt_ids_dims"] = tuple(config["txt_ids_dims"])
    with torch.device("meta"):
        model = Flux(FluxConfig(**config))
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert len(ours) == VECTOR_FREE_GOLDEN["artifact"]["key_count"] == 397
    assert ours == [(key, list(shape)) for key, shape in VECTOR_FREE_GOLDEN["layout"]]
    assert model.vector_in is None
    assert model.guidance_in is None


def test_gated_ovis_state_dict_strict_loads() -> None:
    model = Flux(gated_config())
    result = model.load_state_dict(fill_state_dict(GATED_GOLDEN["case"]["state_dict"]), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_bare_bfl_scale_spelling_loads_after_normalization() -> None:
    """Bare BFL exports spell the RMSNorm scales ``*_norm.scale``;
    normalize_flux_keys is the reference rename
    (Flux.process_unet_state_dict) and the renamed mapping must
    strict-load."""
    tensors = fill_state_dict(golden_entries("schnell_plain"))
    bare = {
        (
            key[: -len(".weight")] + ".scale"
            if key.endswith(("query_norm.weight", "key_norm.weight"))
            else key
        ): value
        for key, value in tensors.items()
    }
    assert any(key.endswith(".scale") for key in bare)
    model = Flux(case_config("schnell_plain"))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(bare, strict=True)
    model.load_state_dict(normalize_flux_keys(bare), strict=True)


def test_txt_norm_scale_spelling_strict_loads() -> None:
    tensors = fill_state_dict(golden_entries("dev_txt_norm"))
    tensors["txt_norm.scale"] = tensors.pop("txt_norm.weight")
    model = Flux(case_config("dev_txt_norm"))
    model.load_state_dict(normalize_flux_keys(tensors), strict=True)


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_matches_executed_reference(case: str) -> None:
    x, timesteps, context, y, guidance = case_inputs(case)
    got = build_model(case)(x, timesteps, context, y, guidance)
    assert_reference_tensor(got, dec(GOLDENS["cases"][case]["output"]), rtol=1e-4, atol=1e-5)


def test_gated_ovis_forward_and_blocks_match_executed_reference() -> None:
    model = build_gated_model()
    observed: dict[str, torch.Tensor] = {}
    hooks = [
        model.double_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(
                double_img=output[0], double_txt=output[1]
            )
        ),
        model.single_blocks[0].register_forward_hook(
            lambda _module, _inputs, output: observed.update(single=cast(torch.Tensor, output))
        ),
    ]
    try:
        output = model(*gated_inputs())
    finally:
        for hook in hooks:
            hook.remove()
    golden = GATED_GOLDEN["case"]
    assert_reference_tensor(
        observed["double_img"],
        dec(golden["block_outputs"]["double_img"]),
        rtol=1e-4,
        atol=1e-5,
    )
    assert_reference_tensor(
        observed["double_txt"],
        dec(golden["block_outputs"]["double_txt"]),
        rtol=1e-4,
        atol=1e-5,
    )
    assert_reference_tensor(
        observed["single"],
        dec(golden["block_outputs"]["single"]),
        rtol=1e-4,
        atol=1e-5,
    )
    assert_reference_tensor(output, dec(golden["output"]), rtol=1e-4, atol=1e-5)


def test_vector_free_ovis_absent_y_forward_matches_executed_reference() -> None:
    model = build_vector_free_model()
    inputs = vector_free_inputs()
    output = model(*inputs[:3])
    torch.testing.assert_close(
        output,
        dec(VECTOR_FREE_GOLDEN["case"]["output"]),
        rtol=1e-4,
        atol=1e-5,
    )
    incidental_y = torch.full((inputs[0].shape[0], 1), float("nan"))
    assert torch.equal(output, model(*inputs[:3], incidental_y, None))


def test_txt_norm_runs_before_text_projection() -> None:
    model = build_model("dev_txt_norm")
    context = case_inputs("dev_txt_norm")[2]
    assert model.txt_norm is not None
    expected = model.txt_in(model.txt_norm(context))
    observed: list[torch.Tensor] = []
    hook = model.txt_in.register_forward_hook(
        lambda _module, inputs, _output: observed.append(inputs[0])
    )
    try:
        model(*case_inputs("dev_txt_norm"))
    finally:
        hook.remove()
    torch.testing.assert_close(observed[0], model.txt_norm(context))
    torch.testing.assert_close(model.txt_in(observed[0]), expected)


def test_classic_flux_passes_context_directly_to_text_projection() -> None:
    model = build_model("dev_guidance")
    inputs = case_inputs("dev_guidance")
    context = inputs[2]
    assert model.txt_norm is None
    observed: list[torch.Tensor] = []
    hook = model.txt_in.register_forward_hook(
        lambda _module, inputs, _output: observed.append(inputs[0])
    )
    try:
        model(*inputs)
    finally:
        hook.remove()
    assert observed[0] is context


def test_odd_spatial_output_matches_input_extents() -> None:
    """The circular pad_to_patch_size leg: odd extents pad up to
    patch multiples inside and must crop back exactly - executed on
    the Dinkster model, not just read from the golden record."""
    spec = GOLDENS["cases"]["dev_odd_spatial"]
    assert (spec["height"], spec["width"]) == (7, 10)
    x, timesteps, context, y, guidance = case_inputs("dev_odd_spatial")
    got = build_model("dev_odd_spatial")(x, timesteps, context, y, guidance)
    assert tuple(got.shape[2:]) == (7, 10)


def test_reference_latent_offset_positions_match_reference_placement() -> None:
    model = Flux(vector_free_config())
    channels = model.config.in_channels
    first = torch.zeros((1, channels, 3, 5))
    second = torch.zeros((1, channels, 5, 3))

    _, first_ids = model._patchify(first, index=1.0)  # pyright: ignore[reportPrivateUsage]
    _, second_ids = model._patchify(  # pyright: ignore[reportPrivateUsage]
        second,
        index=1.0,
        height_offset=2,
    )

    assert first_ids[0, :, 0].tolist() == [1.0] * 6
    assert first_ids[0, :, 1].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert first_ids[0, :, 2].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    assert second_ids[0, :, 0].tolist() == [1.0] * 6
    assert second_ids[0, :, 1].tolist() == [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert second_ids[0, :, 2].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_reference_latents_preserve_target_shape_and_empty_path_values() -> None:
    model = build_vector_free_model()
    inputs = vector_free_inputs()
    ordinary = model(*inputs)
    assert torch.equal(model(*inputs, ref_latents=()), ordinary)

    batch = inputs[0].shape[0]
    channels = model.config.in_channels
    references = (
        torch.zeros((batch, channels, 3, 5)),
        torch.zeros((batch, channels, 5, 3)),
    )
    edited = model(*inputs, ref_latents=references)

    assert edited.shape == ordinary.shape
    assert not torch.equal(edited, ordinary)


def test_reference_latents_refuse_wrong_geometry_and_declared_positions() -> None:
    model = build_vector_free_model()
    inputs = vector_free_inputs()
    batch = inputs[0].shape[0]
    channels = model.config.in_channels

    with pytest.raises(ValueError, match="reference latents must be"):
        model(*inputs, ref_latents=(torch.zeros((batch, channels + 1, 2, 2)),))
    _, positions = model._patchify(inputs[0])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="declared image positions"):
        model(
            *inputs,
            image_position_ids=positions,
            ref_latents=(torch.zeros((batch, channels, 2, 2)),),
        )


# ------------------------------------------------------ embeddings


def test_flux_timestep_embedding_scales_and_orders_cos_sin() -> None:
    """Flux's embedding scales 0..1 flow timesteps by 1000 and lays
    out cos before sin; t=0 pins the halves so a swapped order or a
    dropped time_factor cannot pass."""
    emb = flux_timestep_embedding(torch.tensor([0.0, 0.999]), 64)
    assert emb.shape == (2, 64)
    assert emb.dtype == torch.float32
    torch.testing.assert_close(emb[0, :32], torch.ones(32))
    torch.testing.assert_close(emb[0, 32:], torch.zeros(32))
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(32, dtype=torch.float32) / 32
    )
    torch.testing.assert_close(emb[1, :32], torch.cos(999.0 * freqs))
    torch.testing.assert_close(emb[1, 32:], torch.sin(999.0 * freqs))


def test_rope_rotation_matrices_pin_position_zero_and_one() -> None:
    """rope() lays out [cos, -sin; sin, cos] pairs: position 0 is the
    identity rotation for every frequency, position 1 the unit-angle
    rotation at frequency omega_0 = 1."""
    out = rope(torch.tensor([[0.0, 1.0]]), 4, 10000)
    assert out.shape == (1, 2, 2, 2, 2)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out[0, 0], torch.eye(2).expand(2, 2, 2))
    expected = torch.tensor(
        [
            torch.cos(torch.tensor(1.0)),
            -torch.sin(torch.tensor(1.0)),
            torch.sin(torch.tensor(1.0)),
            torch.cos(torch.tensor(1.0)),
        ]
    ).reshape(2, 2)
    torch.testing.assert_close(out[0, 1, 0], expected)


def test_rope_refuses_odd_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        rope(torch.tensor([[0.0]]), 3, 10000)


def _rope_reference(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Independent restatement of the reference rotation
    (comfy/ldm/flux/math.py _apply_rope1 @ 947c2749): pairs of channels
    are hit with the [cos, -sin; sin, cos] matrices rope() laid out."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
    return out.reshape(*x.shape).type_as(x)


def test_apply_rope_takes_pure_torch_path_under_autograd() -> None:
    """The kitchen kernel registers no autograd formula; inputs that
    require grad must route to the pure-torch math and produce a
    grad_fn. Values are pinned against an independent restatement of
    the reference rotation."""
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).unsqueeze(1)
    q = torch.randn(1, 2, 3, 8, requires_grad=True)
    k = torch.randn(1, 2, 3, 8)
    got_q, got_k = apply_rope(q, k, freqs)
    assert got_q.grad_fn is not None
    torch.testing.assert_close(got_q, _rope_reference(q, freqs))
    torch.testing.assert_close(got_k, _rope_reference(k, freqs))


def test_apply_rope_dispatches_kitchen_combined_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_apply_rope(
        q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((q, k))
        return _rope_reference(q, freqs_cis), _rope_reference(k, freqs_cis)

    monkeypatch.setattr(flux_module, "_ck_apply_rope", fake_apply_rope)
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).unsqueeze(1)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    with torch.no_grad():
        got_q, got_k = apply_rope(q, k, freqs)
    assert calls == [(q, k)]
    assert torch.equal(got_q, _rope_reference(q, freqs))
    assert torch.equal(got_k, _rope_reference(k, freqs))


def test_apply_rope1_dispatches_kitchen_single_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    calls: list[torch.Tensor] = []

    def fake_apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return _rope_reference(x, freqs_cis)

    monkeypatch.setattr(flux_module, "_ck_apply_rope1", fake_apply_rope1)
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).unsqueeze(1)
    x = torch.randn(1, 2, 3, 8)
    with torch.no_grad():
        got = apply_rope1(x, freqs)
    assert calls == [x]
    assert torch.equal(got, _rope_reference(x, freqs))


def test_apply_rope1_uses_torch_for_seedvr2_rank_three_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    def exploding_kitchen(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        raise AssertionError("rank-three SeedVR2 tensors are outside the kitchen kernel contract")

    monkeypatch.setattr(flux_module, "_ck_apply_rope1", exploding_kitchen)
    x = torch.randn(2, 3, 8)
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).squeeze(0)
    with torch.no_grad():
        got = apply_rope1(x, freqs)
    expected = flux_module._apply_rope1_torch(  # pyright: ignore[reportPrivateUsage]
        x, freqs
    )
    assert torch.equal(got, expected)


def test_apply_rope_falls_back_to_torch_without_kitchen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    monkeypatch.setattr(flux_module, "_ck_apply_rope", None)
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).unsqueeze(1)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    with torch.no_grad():
        got_q, got_k = apply_rope(q, k, freqs)
    expected_q, expected_k = flux_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
        q, k, freqs
    )
    assert torch.equal(got_q, expected_q)
    assert torch.equal(got_k, expected_k)


def test_apply_rope_owned_tier_not_consulted_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dinkster-kernels rotation is CUDA-only; CPU inputs go straight
    to the kitchen tier without probing it."""
    from dinkster_inference_torch import flux as flux_module

    def exploding_probe() -> None:
        raise AssertionError("owned-kernel probe must not run for CPU inputs")

    monkeypatch.setattr(flux_module, "_probe_dinkster_apply_rope", exploding_probe)
    freqs = rope(torch.tensor([[0.0, 1.0, 2.0]]), 8, 10000).unsqueeze(1)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    with torch.no_grad():
        got_q, got_k = apply_rope(q, k, freqs)
    torch.testing.assert_close(got_q, _rope_reference(q, freqs))
    torch.testing.assert_close(got_k, _rope_reference(k, freqs))


def test_dinkster_rope_probe_degrades_without_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dinkster_kernels whose availability probe says no (or raises)
    resolves to None once and stays cached."""
    from dinkster_inference_torch import flux as flux_module

    fake = types.ModuleType("dinkster_kernels")
    fake_module = cast(Any, fake)

    def passthrough(
        q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def supported(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> bool:
        return True

    fake_module.apply_rope_available = lambda: False
    fake_module.apply_rope = passthrough
    fake_module.apply_rope_supported = supported
    monkeypatch.setattr(flux_module, "_dk_rope", None)
    monkeypatch.setattr(flux_module, "_dk_rope_probed", False)
    monkeypatch.setitem(sys.modules, "dinkster_kernels", fake)
    assert flux_module._probe_dinkster_apply_rope() is None  # pyright: ignore[reportPrivateUsage]
    assert flux_module._dk_rope_probed  # pyright: ignore[reportPrivateUsage]
    fake_module.apply_rope_available = lambda: True
    assert flux_module._probe_dinkster_apply_rope() is None  # pyright: ignore[reportPrivateUsage]


def test_dinkster_rope_probe_resolves_available_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    fake = types.ModuleType("dinkster_kernels")
    fake_module = cast(Any, fake)

    def combined(
        q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def supported(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> bool:
        return True

    fake_module.apply_rope_available = lambda: True
    fake_module.apply_rope = combined
    fake_module.apply_rope_supported = supported
    monkeypatch.setattr(flux_module, "_dk_rope", None)
    monkeypatch.setattr(flux_module, "_dk_rope_probed", False)
    monkeypatch.setitem(sys.modules, "dinkster_kernels", fake)
    probed = flux_module._probe_dinkster_apply_rope()  # pyright: ignore[reportPrivateUsage]
    assert probed is not None
    assert probed.kernel is combined


def test_kitchen_rope_probe_requires_combined_operation() -> None:
    from dinkster_inference_torch import flux as flux_module

    fake = types.ModuleType("dinkster_kitchen")

    def combined(
        q: torch.Tensor, k: torch.Tensor, _freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    fake_module = cast(Any, fake)
    fake_module.apply_rope = combined
    real = sys.modules.get("dinkster_kitchen")
    sys.modules["dinkster_kitchen"] = fake
    try:
        probed = flux_module._probe_kitchen_apply_rope()  # pyright: ignore[reportPrivateUsage]
        assert probed is combined
        delattr(fake, "apply_rope")
        assert (
            flux_module._probe_kitchen_apply_rope()  # pyright: ignore[reportPrivateUsage]
            is None
        )
    finally:
        if real is not None:
            sys.modules["dinkster_kitchen"] = real
        else:
            del sys.modules["dinkster_kitchen"]


@pytest.mark.parametrize(
    ("cache_name", "probe_name", "accessor_name"),
    (
        ("_ck_apply_rope", "_probe_kitchen_apply_rope", "_kitchen_apply_rope"),
        ("_ck_apply_rope1", "_probe_kitchen_apply_rope1", "_kitchen_apply_rope1"),
    ),
)
def test_kitchen_rope_probe_runs_once_during_concurrent_first_use(
    cache_name: str,
    probe_name: str,
    accessor_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import flux as flux_module

    calls: list[None] = []
    first_probe = Event()
    second_probe = Event()
    release_probe = Event()

    def probe() -> None:
        calls.append(None)
        (first_probe if len(calls) == 1 else second_probe).set()
        assert release_probe.wait(timeout=5)
        return None

    monkeypatch.setattr(
        flux_module,
        cache_name,
        flux_module._KITCHEN_ROPE_UNPROBED,  # pyright: ignore[reportPrivateUsage]
    )
    monkeypatch.setattr(flux_module, probe_name, probe)
    accessor = cast("Callable[[], object]", getattr(flux_module, accessor_name))
    threads = [Thread(target=accessor) for _ in range(2)]
    threads[0].start()
    assert first_probe.wait(timeout=5)
    threads[1].start()
    try:
        assert not second_probe.wait(timeout=0.2)
    finally:
        release_probe.set()
        for thread in threads:
            thread.join(timeout=5)
    assert calls == [None]
    assert not any(thread.is_alive() for thread in threads)


# ------------------------------------------------- batched attention


def test_flux_injects_one_kernel_without_changing_state_or_output() -> None:
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    baseline = build_model("dev_guidance")
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = Flux(case_config("dev_guidance"), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(
        model(x, timesteps, context, y, guidance),
        baseline(x, timesteps, context, y, guidance),
    )
    assert spy.calls
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)


# ------------------------------------------------- guidance gating


def test_dev_model_accepts_disabled_guidance() -> None:
    x, timesteps, context, y, _ = case_inputs("dev_guidance")
    output = build_model("dev_guidance")(x, timesteps, context, y, None)
    assert output.shape == x.shape


def test_schnell_model_rejects_guidance() -> None:
    x, timesteps, context, y, guidance = case_inputs("schnell_plain")
    assert guidance is None
    with pytest.raises(ValueError, match="no guidance embedder"):
        build_model("schnell_plain")(x, timesteps, context, y, torch.tensor([3.5, 3.5]))


# --------------------------------------------------- input refusal


def test_rejects_batch_mismatched_timesteps() -> None:
    """A (1,)-shaped timestep against a larger batch would broadcast
    silently through the modulation vec."""
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    assert x.shape[0] == 2
    with pytest.raises(ValueError, match="timesteps must be"):
        build_model("dev_guidance")(x, timesteps[:1], context, y, guidance)


def test_rejects_batch_mismatched_guidance() -> None:
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    assert guidance is not None
    with pytest.raises(ValueError, match="guidance must be"):
        build_model("dev_guidance")(x, timesteps, context, y, guidance[:1])


def test_rejects_batch_mismatched_y() -> None:
    """The reference zero-fills a missing y and silently slices a
    wide one; Dinkster refuses both."""
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    with pytest.raises(ValueError, match="y must be"):
        build_model("dev_guidance")(x, timesteps, context, y[:1], guidance)
    wide = torch.cat([y, y], dim=1)
    with pytest.raises(ValueError, match="y must be"):
        build_model("dev_guidance")(x, timesteps, context, wide, guidance)


def test_rejects_malformed_context() -> None:
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    model = build_model("dev_guidance")
    with pytest.raises(ValueError, match="context must be"):
        model(x, timesteps, context[:1], y, guidance)
    with pytest.raises(ValueError, match="context must be"):
        model(x, timesteps, context[..., :-1], y, guidance)


# ------------------------------------------------------ config


def test_module_refuses_non_three_axis_configs() -> None:
    from dataclasses import replace

    config = replace(case_config("dev_guidance"), axes_dim=(8, 8), num_heads=2)
    with pytest.raises(ValueError, match="three|\\(index, h, w\\)"):
        Flux(config)


# -------------------------------------------------------- autograd


def test_gradients_flow_to_every_parameter() -> None:
    """No inference_mode/no_grad anywhere in the module (training
    program): the loss must reach EVERY parameter - projections,
    modulations, RMS scales, and embedder MLPs alike."""
    model = build_model("dev_guidance")
    x, timesteps, context, y, guidance = case_inputs("dev_guidance")
    model(x, timesteps, context, y, guidance).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
