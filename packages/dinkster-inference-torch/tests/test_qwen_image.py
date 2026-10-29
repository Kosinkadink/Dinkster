"""Reduced CPU proofs for the unregistered base Qwen Image DiT source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import (
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    QwenImageConfig,
)
from dinkster_inference_torch import (
    EmbedND,
    QwenImage,
    QwenImageAttention,
    QwenImageTransformerBlock,
    qwen_image_timestep_embedding,
    select_attention,
)


@dataclass(frozen=True)
class _ReducedQwenImageConfig:
    transformer_blocks: int = 1
    hidden_width: int = 12
    attention_heads: int = 2
    attention_head_dim: int = 6
    text_width: int = 6
    pooled_width: int = 4
    patchified_input_channels: int = 8
    output_latent_channels: int = 2
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (2, 2, 2)
    default_ref_method: str = "index"
    use_additional_t_cond: bool = False


def reduced_config() -> QwenImageConfig:
    return cast(QwenImageConfig, _ReducedQwenImageConfig())


def reduced_variant_config(
    *, default_ref_method: str, use_additional_t_cond: bool = False
) -> QwenImageConfig:
    return cast(
        QwenImageConfig,
        _ReducedQwenImageConfig(
            default_ref_method=default_ref_method,
            use_additional_t_cond=use_additional_t_cond,
        ),
    )


def fill_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17 - 8) / 64)


def test_timestep_embedding_pins_scale_and_cos_sin_order() -> None:
    timesteps = torch.tensor([0.1234567, 1.0])
    embedding = qwen_image_timestep_embedding(timesteps)
    assert embedding.shape == (2, 256)
    exponent = -torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128
    angles = timesteps[:, None].float() * torch.exp(exponent)[None]
    angles = 1000 * angles
    expected = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    assert torch.equal(embedding, expected)


def test_reduced_state_layout_uses_upstream_checkpoint_names() -> None:
    with torch.device("meta"):
        model = QwenImage(reduced_config())
    state = model.state_dict()
    expected = {
        "time_text_embed.timestep_embedder.linear_1.weight": (12, 256),
        "time_text_embed.timestep_embedder.linear_2.weight": (12, 12),
        "txt_norm.weight": (6,),
        "img_in.weight": (12, 8),
        "txt_in.weight": (12, 6),
        "transformer_blocks.0.img_mod.1.weight": (72, 12),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": (48, 12),
        "transformer_blocks.0.attn.norm_q.weight": (6,),
        "transformer_blocks.0.attn.add_q_proj.weight": (12, 12),
        "transformer_blocks.0.attn.to_out.0.weight": (12, 12),
        "norm_out.linear.weight": (24, 12),
        "proj_out.weight": (8, 12),
    }
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape
    attention = cast(QwenImageTransformerBlock, model.transformer_blocks[0]).attn
    assert attention.norm_q.eps == 1e-6
    assert attention.norm_k.eps == 1e-6
    assert attention.norm_added_q.eps == 1e-6
    assert attention.norm_added_k.eps == 1e-6
    assert all("_attention_kernel" not in key for key in state)


def test_full_profile_consumes_s1_config_without_registering_runtime() -> None:
    with torch.device("meta"):
        model = QwenImage(QWEN_IMAGE_CONFIG)
    assert len(model.state_dict()) == 1933
    assert len(model.transformer_blocks) == 60
    assert model.img_in.in_features == QWEN_IMAGE_CONFIG.patchified_input_channels
    assert model.img_in.out_features == QWEN_IMAGE_CONFIG.hidden_width
    assert model.proj_out.out_features == 64


def test_variant_state_layout_matches_checkpoint_markers() -> None:
    with torch.device("meta"):
        edit = QwenImage(QWEN_IMAGE_EDIT_2511_CONFIG)
        layered = QwenImage(QWEN_IMAGE_LAYERED_CONFIG)
    assert edit.state_dict()["__index_timestep_zero__"].shape == (0,)
    assert "time_text_embed.addition_t_embedding.weight" not in edit.state_dict()
    assert "__index_timestep_zero__" not in layered.state_dict()
    assert layered.state_dict()["time_text_embed.addition_t_embedding.weight"].shape == (
        2,
        3072,
    )


def test_patch_pack_ids_and_final_crop_preserve_odd_target_extent() -> None:
    model = QwenImage(reduced_config())
    fill_parameters(model)
    image = torch.arange(30, dtype=torch.float32).reshape(1, 2, 1, 3, 5)
    packed, ids, padded_shape = model.pack_image(image, index=3)
    assert packed.shape == (1, 6, 8)
    assert padded_shape == (1, 2, 1, 4, 6)
    assert ids.shape == (1, 6, 3)
    assert torch.equal(ids[0, :, 0], torch.full((6,), 3.0))
    assert torch.equal(ids[0, :, 1], torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(ids[0, :, 2], torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]))
    output = model(image, torch.tensor([0.5]), torch.zeros(1, 2, 6))
    assert output.shape == image.shape


def test_spatial_ids_preserve_upstream_low_precision_operation_order() -> None:
    model = QwenImage(reduced_config())
    image = torch.zeros(1, 2, 1, 4098, 2, dtype=torch.bfloat16)
    _, ids, _ = model.pack_image(image)
    low_precision_positions = torch.linspace(0, 2048, steps=2049, dtype=torch.bfloat16)
    expected = torch.zeros(2049, dtype=torch.float32) + low_precision_positions - 1024
    assert torch.equal(ids[0, :, 1], expected)


def test_joint_attention_applies_rope_and_text_prefix_mask() -> None:
    delegate = select_attention("qwen", "sdpa").kernel
    spy = CallableModuleKernel(delegate)
    attention = QwenImageAttention(12, 2, 6, attention_kernel=spy)
    fill_parameters(attention)
    image = torch.randn(1, 3, 12)
    text = torch.randn(1, 2, 12)
    ids = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1).expand(-1, -1, 3)
    frequencies = EmbedND(6, 10000, (2, 2, 2))(ids)
    mask = torch.tensor([[[[0.0, -1000.0, 0.0, 0.0, 0.0]]]])
    image_out, text_out = attention(image, text, mask, frequencies)
    assert image_out.shape == image.shape
    assert text_out.shape == text.shape
    assert spy.calls[0]["q_shape"] == (1, 2, 5, 6)
    assert torch.equal(spy.calls[0]["mask"], mask)
    assert spy.calls[0]["causal"] is False
    assert_kernel_is_not_model_state(attention, spy)


def test_block_preserves_upstream_residual_operation_order() -> None:
    block = QwenImageTransformerBlock(12, 2, 6)
    fill_parameters(block)
    image = torch.randn(1, 3, 12)
    text = torch.randn(1, 2, 12)
    temb = torch.randn(1, 12)
    ids = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1).expand(-1, -1, 3)
    frequencies = EmbedND(6, 10000, (2, 2, 2))(ids)

    def modulate(
        hidden: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = parameters.chunk(3, dim=-1)
        return torch.addcmul(shift[:, None], hidden, 1 + scale[:, None]), gate[:, None]

    img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = block.txt_mod(temb).chunk(2, dim=-1)
    img_norm, img_gate = modulate(block.img_norm1(image), img_mod1)
    txt_norm, txt_gate = modulate(block.txt_norm1(text), txt_mod1)
    img_attn, txt_attn = block.attn(img_norm, txt_norm, None, frequencies)
    expected_image = torch.addcmul(image, img_gate, img_attn)
    expected_text = text + txt_gate * txt_attn
    img_norm, img_gate = modulate(block.img_norm2(expected_image), img_mod2)
    expected_image = torch.addcmul(expected_image, img_gate, block.img_mlp(img_norm))
    txt_norm, txt_gate = modulate(block.txt_norm2(expected_text), txt_mod2)
    expected_text = torch.addcmul(expected_text, txt_gate, block.txt_mlp(txt_norm))

    actual_text, actual_image = block(image, text, temb, frequencies, None)
    assert torch.equal(actual_image, expected_image)
    assert torch.equal(actual_text, expected_text)


def test_forward_orders_text_target_then_one_based_references() -> None:
    spy = CallableModuleKernel(select_attention("qwen", "sdpa").kernel)
    model = QwenImage(reduced_config(), attention_kernel=spy)
    fill_parameters(model)
    seen_ids: list[torch.Tensor] = []
    hook = model.pe_embedder.register_forward_pre_hook(
        lambda _module, inputs: seen_ids.append(inputs[0].detach().clone())
    )
    target = torch.zeros(1, 2, 1, 2, 2)
    first = torch.ones(1, 2, 1, 2, 2)
    second = torch.full((1, 2, 1, 2, 2), 2.0)
    try:
        model(
            target,
            torch.tensor([0.25]),
            torch.zeros(1, 2, 6),
            attention_mask=torch.tensor([[1, 0]]),
            ref_latents=(first, second),
        )
    finally:
        hook.remove()
    ids = seen_ids[0]
    assert ids.shape == (1, 5, 3)
    assert torch.equal(ids[0, :2], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert ids[0, 2, 0].item() == 0
    assert ids[0, 3, 0].item() == 1
    assert ids[0, 4, 0].item() == 2
    assert not hasattr(model, "reference_image_num_tokens")
    expected_mask = torch.tensor([0.0, -torch.finfo(torch.float32).max, 0.0, 0.0, 0.0])
    assert torch.equal(spy.calls[0]["mask"], expected_mask.reshape(1, 1, 1, 5))
    assert_kernel_is_not_model_state(model, spy)


def test_edit_2511_uses_zero_timestep_modulation_for_reference_tokens() -> None:
    model = QwenImage(reduced_variant_config(default_ref_method="index_timestep_zero"))
    fill_parameters(model)
    seen: list[tuple[torch.Tensor, int | None]] = []

    class RecordingBlock(torch.nn.Module):
        def forward(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
            temb: torch.Tensor,
            _frequencies: torch.Tensor,
            _mask: torch.Tensor | None,
            timestep_zero_index: int | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            seen.append((temb.detach().clone(), timestep_zero_index))
            return text, image

    model.transformer_blocks = torch.nn.ModuleList((RecordingBlock(),))
    target = torch.zeros(1, 2, 1, 2, 2)
    reference = torch.ones_like(target)
    output = model(
        target,
        torch.tensor([0.25]),
        torch.zeros(1, 2, 6),
        ref_latents=(reference,),
    )
    assert output.shape == target.shape
    temb, split = seen[0]
    assert temb.shape == (2, 12)
    assert split == 1
    expected = model.time_text_embed(torch.tensor([0.25, 0.0]), torch.zeros(2, 1, 12))
    assert torch.equal(temb, expected)


def test_layered_additional_timestep_and_negative_reference_indices() -> None:
    spy = CallableModuleKernel(select_attention("qwen", "sdpa").kernel)
    model = QwenImage(
        reduced_variant_config(default_ref_method="negative_index", use_additional_t_cond=True),
        attention_kernel=spy,
    )
    fill_parameters(model)
    seen_ids: list[torch.Tensor] = []
    hook = model.pe_embedder.register_forward_pre_hook(
        lambda _module, inputs: seen_ids.append(inputs[0].detach().clone())
    )
    target = torch.zeros(1, 2, 2, 2, 2)
    reference = torch.ones(1, 2, 1, 2, 2)
    try:
        zero = model(
            target,
            torch.tensor([0.25]),
            torch.zeros(1, 2, 6),
            ref_latents=(reference,),
            additional_t_cond=torch.tensor([0]),
        )
        one = model(
            target,
            torch.tensor([0.25]),
            torch.zeros(1, 2, 6),
            ref_latents=(reference,),
            additional_t_cond=torch.tensor([1]),
        )
    finally:
        hook.remove()
    assert zero.shape == one.shape == target.shape
    assert not torch.equal(zero, one)
    ids = seen_ids[0]
    assert ids[0, -1, 0].item() == -1.0


@pytest.mark.parametrize(
    "condition",
    (torch.tensor([2]), torch.tensor([0], dtype=torch.int32), torch.tensor([[0]])),
)
def test_layered_refuses_invalid_additional_timestep_condition(
    condition: torch.Tensor,
) -> None:
    model = QwenImage(
        reduced_variant_config(default_ref_method="negative_index", use_additional_t_cond=True)
    )
    with pytest.raises(ValueError, match="additional timestep condition"):
        model(
            torch.zeros(1, 2, 1, 2, 2),
            torch.zeros(1),
            torch.zeros(1, 2, 6),
            additional_t_cond=condition,
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("rank", "rank 5"),
        ("channels", "channels"),
        ("timestep", "timesteps"),
        ("timestep_device", "timesteps"),
        ("context", "context"),
        ("mask", "attention_mask"),
        ("reference_rank", "reference 0"),
        ("reference_channels", "reference 0"),
    ],
)
def test_malformed_inputs_refuse_before_model_work(change: str, match: str) -> None:
    model = QwenImage(reduced_config())
    called = False

    def mark_called(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal called
        called = True

    hook = model.img_in.register_forward_pre_hook(mark_called)
    x = torch.zeros(1, 2, 1, 2, 2)
    timestep = torch.zeros(1)
    context = torch.zeros(1, 2, 6)
    mask: torch.Tensor | None = torch.ones(1, 2)
    refs: tuple[torch.Tensor, ...] = (torch.zeros_like(x),)
    if change == "rank":
        x = x[:, :, 0]
    elif change == "channels":
        x = torch.zeros(1, 3, 1, 2, 2)
    elif change == "timestep":
        timestep = torch.zeros(2)
    elif change == "timestep_device":
        timestep = torch.empty(1, device="meta")
    elif change == "context":
        context = torch.zeros(1, 2, 7)
    elif change == "mask":
        mask = torch.ones(1, 3)
    elif change == "reference_rank":
        refs = (torch.zeros(1, 2, 2, 2),)
    elif change == "reference_channels":
        refs = (torch.zeros(1, 3, 1, 2, 2),)
    try:
        with pytest.raises(ValueError, match=match):
            model(x, timestep, context, attention_mask=mask, ref_latents=refs)
    finally:
        hook.remove()
    assert called is False


def test_prefetch_temporary_closes_when_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference_torch import qwen_image as module

    model = QwenImage(reduced_config())
    closed: list[object] = []
    queue = object()

    def make_queue(_blocks: torch.nn.ModuleList) -> object:
        return queue

    def pop_queue(_queue: object, _block: torch.nn.Module | None) -> None:
        return None

    monkeypatch.setattr(module, "make_prefetch_queue", make_queue)
    monkeypatch.setattr(module, "prefetch_queue_pop", pop_queue)
    monkeypatch.setattr(module, "close_prefetch_queue", closed.append)

    class RaisingBlock(torch.nn.Module):
        def forward(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("block failed")

    model.transformer_blocks = torch.nn.ModuleList([RaisingBlock()])
    with pytest.raises(RuntimeError, match="block failed"):
        model(
            torch.zeros(1, 2, 1, 2, 2),
            torch.zeros(1),
            torch.zeros(1, 2, 6),
        )
    assert closed == [queue]
