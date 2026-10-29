"""WanDancer model geometry, conditioning, and branch contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from dinkster_inference import WAN22_WANDANCER_14B, Wan21Config, wan21_layout
from dinkster_inference_torch.operations import ResidencyRouted
from dinkster_inference_torch.wan22_dancer import Wan22DancerModel
from unet_fill import fill_state_dict, hashed_input


def _tiny_config() -> Wan21Config:
    config = Wan21Config(
        model_type="i2v",
        in_channels=36,
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
        out_channels=16,
    )
    object.__setattr__(config, "model_variant", "wandancer")
    object.__setattr__(config, "reference_channels", 16)
    return config


def _model() -> Wan22DancerModel:
    model = Wan22DancerModel(_tiny_config())
    state = [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
    model.load_state_dict(fill_state_dict(state), strict=True)
    return model


def _inputs(batch: int = 1) -> tuple[torch.Tensor, ...]:
    x = hashed_input("wandancer:x", (1, 36, 2, 3, 7)).repeat(batch, 1, 1, 1, 1)
    timesteps = torch.full((batch,), 0.5)
    context = hashed_input("wandancer:context", (1, 3, 4)).repeat(batch, 1, 1)
    vision = hashed_input("wandancer:vision", (1, 2, 1280)).repeat(batch, 1, 1)
    reference_vision = hashed_input("wandancer:reference", (1, 4, 1280)).repeat(batch, 1, 1)
    audio = hashed_input("wandancer:audio", (1, 3, 35))
    return x, timesteps, context, vision, reference_vision, audio


def test_official_wandancer_state_and_direct_owners_match_checkpoint_layout() -> None:
    with torch.device("meta"):
        model = Wan22DancerModel()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    expected = wan21_layout(WAN22_WANDANCER_14B)
    for layer in range(2):
        root = f"music_encoder.{layer}.self_attn"
        for suffix in ("weight", "bias"):
            fused_shape = expected.pop(f"{root}.in_proj_{suffix}")
            chunk_shape = (fused_shape[0] // 3, *fused_shape[1:])
            for projection in ("q_proj", "k_proj", "v_proj"):
                expected[f"{root}.{projection}.{suffix}"] = chunk_shape
    assert actual == expected
    assert all(isinstance(module, ResidencyRouted) for module in owners)


def test_wandancer_selects_fps_heads_orders_vision_and_injects_music() -> None:
    model = _model()
    x, timesteps, context, vision, reference_vision, audio = _inputs()
    captured_contexts: list[torch.Tensor] = []

    def capture_context(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        captured_contexts.append(args[3].detach().clone())

    hook = model.blocks[0].register_forward_pre_hook(capture_context)
    local = model(
        x,
        timesteps,
        context,
        vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        fps=30.0,
        audio_inject_scale=1.0,
    )
    hook.remove()
    no_music = model(
        x,
        timesteps,
        context,
        vision,
        reference_vision=reference_vision,
        audio_embed=None,
        fps=30.0,
    )
    zero_music = model(
        x,
        timesteps,
        context,
        vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        fps=30.0,
        audio_inject_scale=0.0,
    )
    global_output = model(
        x,
        timesteps,
        context,
        vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        fps=24.0,
        audio_inject_scale=1.0,
    )

    assert local.shape == global_output.shape == (1, 16, 2, 3, 7)
    assert not torch.equal(local, no_music)
    torch.testing.assert_close(zero_music, no_music, rtol=0.0, atol=0.0)
    assert not torch.equal(local, global_output)
    projected_reference = model.img_emb_refimage(reference_vision)
    assert model.img_emb is not None
    projected_primary = model.img_emb(vision)
    projected_text = model.text_embedding(context)
    torch.testing.assert_close(
        captured_contexts[0],
        torch.cat((projected_reference, projected_primary, projected_text), dim=1),
    )


def test_wandancer_repeats_execution_owned_audio_across_cfg_batch() -> None:
    model = _model()
    x, timesteps, context, vision, reference_vision, audio = _inputs(batch=2)

    output = model(
        x,
        timesteps,
        context,
        vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        fps=24.0,
        audio_inject_scale=0.75,
    )

    assert output.shape == (2, 16, 2, 3, 7)
    torch.testing.assert_close(output[0], output[1], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"reference_vision": torch.empty((1, 2, 1279))}, "reference_vision"),
        ({"audio_embed": torch.empty((1, 2, 34))}, "audio must"),
        ({"fps": 0.0}, "fps"),
        ({"audio_inject_scale": 1}, "audio_inject_scale"),
    ),
)
def test_wandancer_refuses_invalid_execution_inputs(
    changes: dict[str, object],
    match: str,
) -> None:
    model = _model()
    x, timesteps, context, vision, reference_vision, audio = _inputs()
    kwargs: dict[str, object] = {
        "reference_vision": reference_vision,
        "audio_embed": audio,
        "fps": 30.0,
        "audio_inject_scale": 1.0,
    }
    kwargs.update(changes)

    with pytest.raises((TypeError, ValueError), match=match):
        cast(Any, model)(x, timesteps, context, vision, **kwargs)
