"""Native standard SD1.5 IP-Adapter execution and ownership tests."""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import (
    SD15_IPADAPTER_SITES,
    SD15_UNET_CONFIG,
    DiscreteSigmas,
    PayloadReference,
    PercentRange,
    SD15AttentionContribution,
    sd15_ipadapter_layout,
)
from dinkster_inference_torch import (
    DenoiseError,
    SD15AttentionExecutionContext,
    SD15IPAdapter,
    SD15IPAdapterClipVisionEncoder,
    SD15IPAdapterConditioning,
    SD15IPAdapterExecution,
    SD15IPAdapterResourceError,
    SDDenoiser,
    UNetModel,
    sd15_ipadapter_identity_facts,
    sd15_ipadapter_resource_digest,
    sd15_ipadapter_tensor_digest,
    validate_sd15_ipadapter_resource,
)
from dinkster_inference_torch.attention import AttentionKernel
from dinkster_inference_torch.ipadapter import (
    _bind_sd15_ipadapter_resource,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import ResidencyRouted

ASSET_DIGEST = "blake3:" + "a" * 64


def _model(*, value_projection: float = 0.0) -> SD15IPAdapter:
    model = SD15IPAdapter()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        first = model.ip_adapter[str(SD15_IPADAPTER_SITES[0].adapter_index)]
        first.to_v_ip.weight.fill_(value_projection)  # type: ignore[union-attr]
    digest = sd15_ipadapter_resource_digest(ASSET_DIGEST, torch.float32)
    _bind_sd15_ipadapter_resource(model, digest)
    return model


def _conditioning(
    model: SD15IPAdapter,
    *,
    cond_value: float = 1.0,
    uncond_value: float = 2.0,
    strength: float = 1.0,
    mask: torch.Tensor | None = None,
    site_gains: tuple[float, ...] | None = None,
) -> SD15IPAdapterConditioning:
    cond = torch.full((1, 4, 768), cond_value)
    uncond = torch.full((1, 4, 768), uncond_value)
    tokens_digest = sd15_ipadapter_tensor_digest(
        torch.cat((cond, uncond), dim=0), role="projected-tokens"
    )
    mask_digest = None if mask is None else sd15_ipadapter_tensor_digest(mask, role="output-mask")
    model_digest = model.resource_digest
    assert model_digest is not None
    declaration = SD15AttentionContribution(
        PayloadReference(model_digest),
        PayloadReference(tokens_digest),
        PercentRange(0.0, 1.0),
        strength,
        mask=None if mask_digest is None else PayloadReference(mask_digest),
        site_gains=(1.0,) * 16 if site_gains is None else site_gains,
    )
    return SD15IPAdapterConditioning(
        declaration,
        model,
        cond,
        uncond,
        model_digest,
        tokens_digest,
        mask,
        mask_digest,
    )


def _value_kernel(
    q: torch.Tensor,
    _k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return v.mean(dim=2, keepdim=True).expand_as(q)


def test_full_module_layout_and_unet_sites_are_canonical() -> None:
    with torch.device("meta"):
        adapter = SD15IPAdapter()
        unet = UNetModel(SD15_UNET_CONFIG)

    assert {key: tuple(value.shape) for key, value in adapter.state_dict().items()} == (
        sd15_ipadapter_layout()
    )
    assert set(unet.attention_sites) == {(site.id, site.width) for site in SD15_IPADAPTER_SITES}
    modules = dict(unet.named_modules())
    for site in SD15_IPADAPTER_SITES:
        module = modules[site.id]
        assert module.site_id == site.id  # type: ignore[union-attr]
        assert module.to_q.in_features == site.width  # type: ignore[union-attr]
    assert tuple(adapter.ip_adapter) == tuple(
        str(site.adapter_index) for site in SD15_IPADAPTER_SITES
    )


def test_clip_vision_subclass_returns_projected_pooled_embedding() -> None:
    from dinkster_inference.clip_vision import ClipVisionConfig

    config = ClipVisionConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=4,
        patch_size=2,
        projection_dim=4,
    )
    model = SD15IPAdapterClipVisionEncoder(config)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 1000)
    image = torch.linspace(0.0, 1.0, 48).reshape(1, 4, 4, 3)
    from dinkster_inference_torch.clip_vision import clip_vision_preprocess

    pixels = clip_vision_preprocess(image, size=4)
    _, _, pooled = model.vision_model(pixels)
    expected = model.visual_projection(pooled)

    assert torch.equal(model(image), expected)
    assert expected.shape == (1, 4)


def test_fused_and_separate_guidance_rows_use_actual_lane_ids() -> None:
    conditioning = _conditioning(_model())
    execution = SD15IPAdapterExecution(conditioning, 10.0, 0.0)

    fused = SD15AttentionExecutionContext.for_sigma(
        (execution,),
        5.0,
        ("negative", "positive"),
        2,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert fused is not None
    tokens = fused.contributions[0].tokens
    assert torch.equal(tokens[:2], torch.full((2, 4, 768), 2.0))
    assert torch.equal(tokens[2:], torch.full((2, 4, 768), 1.0))

    separate = SD15AttentionExecutionContext.for_sigma(
        (execution,),
        5.0,
        ("positive",),
        3,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert separate is not None
    assert torch.equal(separate.contributions[0].tokens, torch.ones((3, 4, 768)))

    empty = SD15AttentionExecutionContext.for_sigma(
        (execution,),
        5.0,
        ("empty",),
        1,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert empty is not None
    assert torch.equal(empty.contributions[0].tokens, torch.ones((1, 4, 768)))

    with pytest.raises(ValueError, match="does not define guidance lanes"):
        SD15AttentionExecutionContext.for_sigma(
            (execution,),
            5.0,
            ("custom",),
            1,
            latent_height=1,
            latent_width=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_site_lane_scalar_and_output_mask_gains_multiply_before_to_out() -> None:
    model = _model(value_projection=0.001)
    mask = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]], dtype=torch.float32)
    site_gains = (0.5,) + (1.0,) * 15
    execution = SD15IPAdapterExecution(
        _conditioning(model, strength=2.0, mask=mask, site_gains=site_gains),
        10.0,
        0.0,
    )
    context = SD15AttentionExecutionContext.for_sigma(
        (execution,),
        5.0,
        ("positive", "negative"),
        1,
        latent_height=2,
        latent_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert context is not None
    q = torch.zeros((2, 8, 4, 40))
    out = context.apply(
        SD15_IPADAPTER_SITES[0].id,
        q,
        torch.zeros_like(q),
        height=2,
        width=2,
        kernel=cast("AttentionKernel", _value_kernel),
    )

    tokens = torch.stack((torch.ones((4, 768)), torch.full((4, 768), 2.0)))
    projected = F.linear(tokens, torch.full((320, 768), 0.001))
    expected = projected.view(2, 4, 8, 40).transpose(1, 2).mean(dim=2, keepdim=True)
    expected = expected * mask.reshape(1, 1, 4, 1)
    expected = expected.expand_as(out)
    torch.testing.assert_close(out, expected)


def test_output_mask_matches_source_grid_for_odd_latent_dimensions() -> None:
    model = _model(value_projection=0.001)
    mask = torch.linspace(0.0, 1.0, 35, dtype=torch.float32).reshape(1, 5, 7)
    execution = SD15IPAdapterExecution(_conditioning(model, mask=mask), 10.0, 0.0)
    context = SD15AttentionExecutionContext.for_sigma(
        (execution,),
        5.0,
        ("positive",),
        1,
        latent_height=65,
        latent_width=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert context is not None
    q = torch.zeros((1, 8, 33 * 32, 40))
    out = context.apply(
        SD15_IPADAPTER_SITES[0].id,
        q,
        torch.zeros_like(q),
        height=33,
        width=32,
        kernel=cast("AttentionKernel", _value_kernel),
    )

    source_grid = F.interpolate(mask.unsqueeze(1), size=(32, 33), mode="bilinear")
    expected = 0.768 * source_grid.reshape(1, 1, 33 * 32, 1).expand_as(out)
    actual_site_grid = F.interpolate(mask.unsqueeze(1), size=(33, 32), mode="bilinear")
    assert not torch.equal(source_grid.flatten(), actual_site_grid.flatten())
    torch.testing.assert_close(out, expected)

    padded_q = torch.zeros((1, 8, 25 * 40, 40))
    padded_out = context.apply(
        SD15_IPADAPTER_SITES[0].id,
        padded_q,
        torch.zeros_like(padded_q),
        height=25,
        width=40,
        kernel=cast("AttentionKernel", _value_kernel),
    )
    short_grid = F.interpolate(mask.unsqueeze(1), size=(32, 31), mode="bilinear")
    padded_mask = F.pad(short_grid.reshape(1, 1, 32 * 31, 1), (0, 0, 4, 4))
    torch.testing.assert_close(padded_out, 0.768 * padded_mask.expand_as(padded_out))


def test_multiple_contributions_add_in_declaration_order() -> None:
    model = _model(value_projection=0.001)
    executions = (
        SD15IPAdapterExecution(_conditioning(model, strength=1.0), 10.0, 0.0),
        SD15IPAdapterExecution(_conditioning(model, strength=-0.25), 10.0, 0.0),
    )
    context = SD15AttentionExecutionContext.for_sigma(
        executions,
        5.0,
        ("positive",),
        1,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert context is not None
    q = torch.zeros((1, 8, 1, 40))
    out = context.apply(
        SD15_IPADAPTER_SITES[0].id,
        q,
        torch.zeros_like(q),
        height=1,
        width=1,
        kernel=cast("AttentionKernel", _value_kernel),
    )
    torch.testing.assert_close(out, torch.full_like(out, 0.768 * 0.75))


def test_sigma_window_is_checked_before_resource_or_payload_work() -> None:
    model = _model()
    execution = SD15IPAdapterExecution(_conditioning(model), 10.0, 2.0)
    with torch.no_grad():
        model.image_proj.proj.weight.add_(1.0)

    assert (
        SD15AttentionExecutionContext.for_sigma(
            (execution,),
            11.0,
            ("positive",),
            1,
            latent_height=1,
            latent_width=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )
    with pytest.raises(SD15IPAdapterResourceError, match="parameter"):
        SD15AttentionExecutionContext.for_sigma(
            (execution,),
            5.0,
            ("positive",),
            1,
            latent_height=1,
            latent_width=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_conditioning_snapshots_mutable_tokens_and_mask() -> None:
    model = _model()
    cond = torch.ones((1, 4, 768))
    uncond = torch.full((1, 4, 768), 2.0)
    mask = torch.ones((1, 2, 2), dtype=torch.float32)
    token_digest = sd15_ipadapter_tensor_digest(
        torch.cat((cond, uncond), dim=0), role="projected-tokens"
    )
    mask_digest = sd15_ipadapter_tensor_digest(mask, role="output-mask")
    model_digest = model.resource_digest
    assert model_digest is not None
    declaration = SD15AttentionContribution(
        PayloadReference(model_digest),
        PayloadReference(token_digest),
        PercentRange(0.0, 1.0),
        1.0,
        mask=PayloadReference(mask_digest),
    )
    conditioning = SD15IPAdapterConditioning(
        declaration,
        model,
        cond,
        uncond,
        model_digest,
        token_digest,
        mask,
        mask_digest,
    )
    cond.zero_()
    uncond.zero_()
    mask.zero_()

    assert torch.count_nonzero(conditioning.cond_tokens).item() == conditioning.cond_tokens.numel()
    assert (
        torch.count_nonzero(conditioning.uncond_tokens).item() == conditioning.uncond_tokens.numel()
    )
    assert conditioning.mask is not None
    assert torch.count_nonzero(conditioning.mask).item() == conditioning.mask.numel()


def test_identity_facts_rotate_with_execution_and_placement_inputs() -> None:
    model = _model()
    baseline = _conditioning(model)
    changed_gain = _conditioning(model, strength=2.0)

    baseline_facts = sd15_ipadapter_identity_facts((baseline,))
    changed_facts = sd15_ipadapter_identity_facts((changed_gain,))
    assert baseline_facts != changed_facts
    assert any(".model=" in fact for fact in baseline_facts)
    assert any(".tokens=" in fact for fact in baseline_facts)
    assert any(".window=" in fact for fact in baseline_facts)
    assert any(".sites=" in fact for fact in baseline_facts)
    assert any(".lanes=" in fact for fact in baseline_facts)
    assert any(
        ".lane_sources=positive:conditional,negative:unconditional,empty:conditional" in fact
        for fact in baseline_facts
    )
    assert any(".placement=attn2-pre-to-out" in fact for fact in baseline_facts)


def test_denoiser_snapshots_payloads_again_at_runtime_admission() -> None:
    execution = SD15IPAdapterExecution(_conditioning(_model()), 10.0, 0.0)
    with torch.device("meta"):
        unet = UNetModel(SD15_UNET_CONFIG)
    denoiser = SDDenoiser(unet, DiscreteSigmas.linear_beta(), ipadapter=(execution,))

    execution.conditioning.cond_tokens.zero_()
    execution.conditioning.uncond_tokens.zero_()
    assert torch.count_nonzero(
        denoiser._ipadapter[0].conditioning.cond_tokens  # pyright: ignore[reportPrivateUsage]
    ).item()
    assert torch.count_nonzero(
        denoiser._ipadapter[0].conditioning.uncond_tokens  # pyright: ignore[reportPrivateUsage]
    ).item()


def test_all_projection_parameters_remain_normal_residency_owned_state() -> None:
    model = _model()
    projection_names = {
        name for name, _module in model.named_modules() if name.endswith(("to_k_ip", "to_v_ip"))
    }
    assert len(projection_names) == 32
    assert all(
        isinstance(module, ResidencyRouted)
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    )

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    validate_sd15_ipadapter_resource(model, cast("str", model.resource_digest))
    mechanism.partially_unload(mechanism.loaded_bytes())
    validate_sd15_ipadapter_resource(model, cast("str", model.resource_digest))


def test_denoiser_admits_only_the_canonical_sd15_site_table() -> None:
    execution = SD15IPAdapterExecution(_conditioning(_model()), 10.0, 0.0)
    with torch.device("meta"):
        model = UNetModel(SD15_UNET_CONFIG)
    SDDenoiser(model, DiscreteSigmas.linear_beta(), ipadapter=(execution,))

    model.attention_sites = model.attention_sites[:-1]
    with pytest.raises(DenoiseError, match="canonical 16-site"):
        SDDenoiser(model, DiscreteSigmas.linear_beta(), ipadapter=(execution,))
