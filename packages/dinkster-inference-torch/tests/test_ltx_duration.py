from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import LTXAV_DURATION_HEAD_CONFIG, ltxav_duration_head_layout
from dinkster_inference_torch.ltx_duration import LTXDurationHead, ltx_duration_frames
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import INITLESS
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens/ltx_duration_goldens.json",
    allow_portable_fallback=True,
)


def _model() -> LTXDurationHead:
    model = LTXDurationHead(LTXAV_DURATION_HEAD_CONFIG, operations=INITLESS)
    model.load_state_dict(fill_state_dict(GOLDENS["state_dict"]), strict=True, assign=True)
    return model


class _ReferenceAttentionPooler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = LTXAV_DURATION_HEAD_CONFIG
        self.query_tokens = torch.nn.Parameter(torch.empty(config.num_queries, config.hidden_dim))
        self.cross_attn = torch.nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class _ReferenceDurationHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = LTXAV_DURATION_HEAD_CONFIG
        self.video_input_proj = torch.nn.Linear(config.video_input_dim, config.hidden_dim)
        self.video_modality_emb = torch.nn.Parameter(torch.empty(config.hidden_dim))
        self.audio_input_proj = torch.nn.Linear(config.audio_input_dim, config.hidden_dim)
        self.audio_modality_emb = torch.nn.Parameter(torch.empty(config.hidden_dim))
        self.attention_pooler = _ReferenceAttentionPooler()
        self.mlp_hidden = torch.nn.Linear(
            config.hidden_dim * config.num_queries,
            config.mlp_hidden_dim,
        )
        self.mlp_out = torch.nn.Linear(config.mlp_hidden_dim, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None,
        audio_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        token_groups: list[torch.Tensor] = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        pooled = self.attention_pooler(torch.cat(token_groups, dim=1))
        hidden = torch.nn.functional.gelu(
            self.mlp_hidden(pooled.reshape(pooled.shape[0], -1)),
            approximate="tanh",
        )
        return self.mlp_out(hidden).squeeze(-1).exp()


def _reference_model() -> _ReferenceDurationHead:
    model = _ReferenceDurationHead()
    model.load_state_dict(fill_state_dict(GOLDENS["state_dict"]), strict=True)
    return model


# The stored golden crosses CPU attention and GEMM kernels; the same-process
# stock-torch reference below remains the exact transcription contract. Across
# 1-96 local threads the reference drifted by at most 0.03515625 absolute,
# 1.648645e-5 relative, and 193 ulp, so 5e-5 relative carries 3x headroom.
DURATION_GOLDEN_RTOL = 5e-5


def test_duration_head_layout_matches_executed_reference() -> None:
    model = _model()
    actual = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    expected = [(key, list(shape)) for key, shape in GOLDENS["state_dict"]]
    assert actual == expected
    assert dict(ltxav_duration_head_layout(LTXAV_DURATION_HEAD_CONFIG)) == {
        key: tuple(shape) for key, shape in expected
    }


@pytest.mark.parametrize("name", sorted(GOLDENS["cases"]))
def test_duration_head_matches_executed_reference(name: str) -> None:
    use_video = name != "audio_only"
    use_audio = name != "video_only"
    video = hashed_input(f"{name}:video", (2, 3, 4096)) if use_video else None
    audio = hashed_input(f"{name}:audio", (2, 4, 2048)) if use_audio else None

    with torch.no_grad():
        actual = _model()(video, audio)
        reference = _reference_model()(video, audio)

    assert torch.equal(actual, reference)
    assert_reference_tensor(
        reference,
        torch.tensor(GOLDENS["cases"][name]["output"]),
        rtol=DURATION_GOLDEN_RTOL,
        atol=0.0,
    )


def test_duration_head_is_differentiable_and_requires_tokens() -> None:
    # Unbounded inputs can overflow the exponential head with these synthetic weights.
    video = hashed_input("differentiable:video", (1, 2, 4096)).requires_grad_()
    audio = hashed_input("differentiable:audio", (1, 3, 2048)).requires_grad_()
    reference_video = video.detach().clone().requires_grad_()
    reference_audio = audio.detach().clone().requires_grad_()
    output = _model()(video, audio)
    reference = _reference_model()(reference_video, reference_audio)
    assert torch.isfinite(output).all()
    assert torch.equal(output, reference)
    output.sum().backward()
    reference.sum().backward()

    for actual, expected in ((video, reference_video), (audio, reference_audio)):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        assert torch.count_nonzero(actual.grad) > 0
        assert expected.grad is not None
        assert torch.equal(actual.grad, expected.grad)
    with pytest.raises(ValueError, match="requires video or audio"):
        _model()()


def test_duration_head_enrolls_and_executes_through_component_residency() -> None:
    model = _model()
    video = hashed_input("residency:video", (1, 3, 4096))
    audio = hashed_input("residency:audio", (1, 4, 2048))
    expected = model(video, audio)

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    assert torch.equal(model(video, audio), expected)
    mechanism.unload()
    mechanism.partially_load(None)
    assert torch.equal(model(video, audio), expected)


def test_duration_frame_snapping_matches_executed_reference() -> None:
    assert ltx_duration_frames(0.1, 24.0, 1.0, 20.0) == GOLDENS["frames"]["below_minimum"]
    assert ltx_duration_frames(4.2, 24.0, 1.0, 20.0) == GOLDENS["frames"]["inside"]
    assert ltx_duration_frames(40.0, 24.0, 1.0, 20.0) == GOLDENS["frames"]["above_maximum"]


@pytest.mark.parametrize(
    ("values", "match"),
    (
        ((cast("Any", 1), 24.0, 1.0, 20.0), "finite float"),
        ((1.0, 0.0, 1.0, 20.0), "ordered and nonnegative"),
        ((1.0, 24.0, 2.0, 1.0), "ordered and nonnegative"),
        ((1.0, 24.0, 1.0, 20.0, True), "positive integer"),
    ),
)
def test_duration_frame_snapping_refuses_malformed_values(
    values: tuple[Any, ...], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        if len(values) == 4:
            ltx_duration_frames(*values)
        else:
            ltx_duration_frames(*values[:4], time_scale=values[4])
