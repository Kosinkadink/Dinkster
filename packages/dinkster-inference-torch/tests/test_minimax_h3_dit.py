"""CPU and static proofs for the direct-import MiniMax H3 attention primitive."""

from __future__ import annotations

import importlib.metadata
import math
import os
import tempfile
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import Any, cast

import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_SIGMAS,
    CancellationFlag,
    LatentStream,
    MiniMaxH3Config,
    MiniMaxH3DiTExecutionRefusal,
    MiniMaxH3Sigmas,
    MiniMaxH3Task,
    MiniMaxH3VideoLatentGeometry,
    MultiStreamLatent,
    SamplingGuidance,
    SequenceLayout,
    SequencePartition,
    SequenceShard,
    TimelineGuide,
    UspMesh,
    minimax_h3_dit_layout,
    plan_minimax_h3_token_layout,
    plan_sequence_partition,
)
from dinkster_inference_torch import (
    MiniMaxH3DiTRuntime,
    MiniMaxH3PackedSequenceFacts,
    MiniMaxH3PreparedConditioning,
    MiniMaxH3RuntimeError,
    MiniMaxH3SequenceSharding,
    pack_latent_streams,
)
from dinkster_inference_torch import attention as attention_module
from dinkster_inference_torch import minimax_h3_dit as dit_module
from dinkster_inference_torch.attention import (
    BUILTIN_SDPA_PROVIDER,
    COMFY_KITCHEN_INT8_PROVIDER,
    SAGE2_PROVIDER,
    SOL_ATTENTION_PROVIDER,
    AttentionKernel,
    AttentionSelection,
    AttentionSelectionError,
    AttentionTensorLease,
    builtin_sdpa_kernel,
    dinkster_kitchen_int8_available,
    select_attention,
)
from dinkster_inference_torch.distributed import (
    DistributedSamplingConfig,
    sequence_receipt_identity,
)
from dinkster_inference_torch.minimax_h3_dit import (
    MINIMAX_H3_ATTENTION_GEOMETRY,
    MiniMaxH3Attention,
    MiniMaxH3AttentionGeometry,
    MiniMaxH3AttentionProviderEvidence,
    MiniMaxH3DiT,
    MiniMaxH3DiTConditioning,
    MiniMaxH3KeyframeLatent,
    MiniMaxH3ReferenceKind,
    MiniMaxH3ReferenceLatents,
    assemble_minimax_h3_attention,
    assemble_minimax_h3_dit,
    minimax_h3_attention_provider,
    minimax_h3_guidance_integration_facts,
    minimax_h3_sequence_integration_facts,
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import CastOperations, InitlessOperations
from dinkster_inference_torch.sequence_parallel_attention import SequenceParallelAttentionKernel
from gpu_test_gate import require_gpu_tests_enabled
from manifest_token import minted_consensus_token
from torch.multiprocessing.spawn import spawn
from torch.utils.checkpoint import checkpoint


def _h3(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))


def _distributed_sequence_kernel(
    mesh: UspMesh, layout: SequenceLayout
) -> SequenceParallelAttentionKernel:
    return SequenceParallelAttentionKernel(
        mesh,
        builtin_sdpa_kernel(),
        layout,
        None,
        minted_consensus_token(mesh.ulysses * mesh.ring, layout.shard.index),
        "test-attention-backend",
        "test-exchange-backend",
    )


class _MetaOperations(InitlessOperations):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear:
        self.calls.append(("linear", in_features, out_features, bias))
        return torch.nn.Linear(in_features, out_features, bias=bias, device="meta")

    def rms_norm(
        self,
        normalized_shape: int,
        *,
        eps: float | None = None,
    ) -> torch.nn.RMSNorm:
        self.calls.append(("rms_norm", normalized_shape, eps))
        return torch.nn.RMSNorm(normalized_shape, eps=eps, device="meta")


class _RecordingKernel:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                bool,
                float | None,
                bool,
            ]
        ] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        self.calls.append((q, k, v, mask, causal, scale, enable_gqa))
        return v


def _evidence() -> MiniMaxH3AttentionProviderEvidence:
    return MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, str(torch.__version__))


def _rope_table(sequence: int, rotary_dim: int) -> torch.Tensor:
    angles = torch.linspace(0.1, 0.7, sequence * (rotary_dim // 2)).reshape(
        1, sequence, 1, rotary_dim // 2
    )
    cosine = angles.cos()
    sine = angles.sin()
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, sequence, 1, rotary_dim // 2, 2, 2
    )


def _reference_rope(value: torch.Tensor, table: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    prefix = value[..., :rotary_dim]
    pairs = prefix.reshape(*prefix.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    rotated = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
    return torch.cat((rotated.movedim(-1, -2).reshape_as(prefix), value[..., rotary_dim:]), dim=-1)


def test_builtin_accessor_reuses_selector_kernel_without_h3_role_or_status() -> None:
    kernel = builtin_sdpa_kernel()
    assert kernel is select_attention("flux", "sdpa").kernel
    assert BUILTIN_SDPA_PROVIDER == "torch-sdpa-priority-v1"
    assert not hasattr(kernel, "role")
    assert not hasattr(kernel, "status")
    with pytest.raises(AttentionSelectionError, match="unknown attention role"):
        select_attention(cast(Any, "h3"), "sdpa")


def test_kernel_and_exact_frozen_provider_evidence_are_required() -> None:
    geometry = MiniMaxH3AttentionGeometry(8, 1, 8, 6)
    operations = InitlessOperations()
    kernel = _RecordingKernel()
    model = MiniMaxH3Attention(geometry, kernel, _evidence(), operations=operations)
    assert model.provider_evidence == _evidence()

    with pytest.raises(TypeError):
        MiniMaxH3Attention(
            geometry, cast(AttentionKernel, None), _evidence(), operations=operations
        )
    with pytest.raises(TypeError):
        MiniMaxH3Attention(
            geometry, kernel, cast(MiniMaxH3AttentionProviderEvidence, None), operations=operations
        )
    with pytest.raises(FrozenInstanceError):
        model.provider_evidence.provider = "other"  # type: ignore[misc]


@dataclass
class _MutableEvidence:
    provider: str
    torch_version: str


@pytest.mark.parametrize(
    "evidence",
    (
        _MutableEvidence(BUILTIN_SDPA_PROVIDER, str(torch.__version__)),
        {"provider": BUILTIN_SDPA_PROVIDER, "torch_version": str(torch.__version__)},
    ),
)
def test_mutable_or_foreign_evidence_refuses(evidence: object) -> None:
    with pytest.raises(TypeError, match="exact frozen MiniMaxH3 evidence"):
        MiniMaxH3Attention(
            MiniMaxH3AttentionGeometry(8, 1, 8, 6),
            _RecordingKernel(),
            cast(MiniMaxH3AttentionProviderEvidence, evidence),
            operations=InitlessOperations(),
        )


def test_provider_evidence_refuses_wrong_identity_version_type_and_value() -> None:
    version = str(torch.__version__)
    with pytest.raises(TypeError, match="provider must be a string"):
        MiniMaxH3AttentionProviderEvidence(cast(str, None), version)
    with pytest.raises(ValueError, match="provider must be non-empty"):
        MiniMaxH3AttentionProviderEvidence("", version)
    with pytest.raises(TypeError, match="torch_version"):
        MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, cast(str, 1))
    with pytest.raises(ValueError, match="running torch version"):
        MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, "")
    with pytest.raises(ValueError, match="running torch version"):
        MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, version + ".foreign")
    # Providers beyond the built-ins are accepted with an optional version so
    # H3 consumes whatever provider the generic selector authenticated.
    foreign = MiniMaxH3AttentionProviderEvidence("future-provider-v1", version, "1.2.3")
    assert foreign.provider_version == "1.2.3"
    unversioned = MiniMaxH3AttentionProviderEvidence("future-provider-v1", version)
    assert unversioned.provider_version is None
    with pytest.raises(ValueError, match="None or a non-empty string"):
        MiniMaxH3AttentionProviderEvidence("future-provider-v1", version, "")


def test_production_assembly_requests_exact_geometry_and_state_layout_without_storage() -> None:
    operations = _MetaOperations()
    kernel = _RecordingKernel()
    selection = AttentionSelection(kernel, select_attention("flux", "sdpa").status)
    model = assemble_minimax_h3_attention(
        operations=operations,
        attention_selection=selection,
    )
    config = MINIMAX_H3_CONFIG
    assert model.geometry is MINIMAX_H3_ATTENTION_GEOMETRY
    assert model.geometry == MiniMaxH3AttentionGeometry(
        config.hidden_width,
        config.attention_heads,
        config.attention_head_dim,
        96,
    )
    assert model.attention_kernel is kernel
    assert model.provider_evidence == _evidence()
    assert operations.calls == [
        ("linear", 5376, 21504, False),
        ("rms_norm", 128, 1e-5),
        ("rms_norm", 128, 1e-5),
        ("linear", 7168, 5376, False),
    ]
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == {
        "qkv_proj.weight": (21504, 5376),
        "q_norm.weight": (128,),
        "k_norm.weight": (128,),
        "out_proj.weight": (5376, 7168),
    }
    assert all(value.device.type == "meta" for value in model.state_dict().values())


def test_reduced_cpu_output_matches_independent_norm_rope_attention_reference() -> None:
    torch.manual_seed(712)
    geometry = MiniMaxH3AttentionGeometry(12, 2, 6, 6)
    model = MiniMaxH3Attention(
        geometry, builtin_sdpa_kernel(), _evidence(), operations=InitlessOperations()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    hidden = torch.randn(1, 3, 12)
    table = _rope_table(3, 6)

    query, key, value = F.linear(hidden, model.qkv_proj.weight).chunk(3, dim=-1)
    query = query.view(1, 3, 2, 6)
    key = key.view(1, 3, 2, 6)
    value = value.view(1, 3, 2, 6)
    query = F.rms_norm(query, (6,), model.q_norm.weight, 1e-5)
    key = F.rms_norm(key, (6,), model.k_norm.weight, 1e-5)
    query = _reference_rope(query, table, 6).transpose(1, 2)
    key = _reference_rope(key, table, 6).transpose(1, 2)
    expected = F.scaled_dot_product_attention(query, key, value.transpose(1, 2), dropout_p=0.0)
    expected = F.linear(expected.transpose(1, 2).reshape(1, 3, 12), model.out_proj.weight)

    first = model(hidden, table)
    second = model(hidden, table)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


def test_inference_uses_in_place_kitchen_norm_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = MiniMaxH3AttentionGeometry(12, 2, 6, 6)
    kernel = _RecordingKernel()
    model = MiniMaxH3Attention(
        geometry,
        kernel,
        _evidence(),
        operations=InitlessOperations(),
    )
    hidden = torch.randn(1, 3, 12)
    table = _rope_table(3, 6)
    calls: list[tuple[float, int]] = []

    def fake_fused_norm_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_table: torch.Tensor,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        epsilon: float,
        rot_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((epsilon, rot_dim))
        query.copy_(
            _reference_rope(
                F.rms_norm(query, (geometry.head_dim,), query_weight, epsilon),
                rope_table,
                rot_dim,
            )
        )
        key.copy_(
            _reference_rope(
                F.rms_norm(key, (geometry.head_dim,), key_weight, epsilon),
                rope_table,
                rot_dim,
            )
        )
        return query, key

    monkeypatch.setattr(dinkster_kitchen, "rms_rope_split_half_", fake_fused_norm_rope)
    with torch.no_grad():
        model(hidden, table)

    assert calls == [(geometry.norm_eps, geometry.rotary_dim)]
    assert len(kernel.calls) == 1


def test_norm_precedes_partial_split_half_rope_and_tail_is_untouched() -> None:
    torch.manual_seed(23)
    geometry = MiniMaxH3AttentionGeometry(128, 1, 128, 96)
    kernel = _RecordingKernel()
    model = MiniMaxH3Attention(geometry, kernel, _evidence(), operations=InitlessOperations())
    with torch.no_grad():
        model.qkv_proj.weight.copy_(torch.randn_like(model.qkv_proj.weight))
        model.q_norm.weight.copy_(torch.linspace(0.5, 1.5, 128))
        model.k_norm.weight.copy_(torch.linspace(1.5, 0.5, 128))
        model.out_proj.weight.zero_()
    hidden = torch.randn(1, 2, 128)
    table = _rope_table(2, 96)
    model(hidden, table)

    query, key, _ = F.linear(hidden, model.qkv_proj.weight).chunk(3, dim=-1)
    query = query.view(1, 2, 1, 128)
    key = key.view(1, 2, 1, 128)
    normalized_q = F.rms_norm(query, (128,), model.q_norm.weight, 1e-5)
    normalized_k = F.rms_norm(key, (128,), model.k_norm.weight, 1e-5)
    called_q, called_k, _, _, _, _, _ = kernel.calls[0]
    torch.testing.assert_close(
        called_q.transpose(1, 2)[..., :96], _reference_rope(normalized_q, table, 96)[..., :96]
    )
    torch.testing.assert_close(
        called_k.transpose(1, 2)[..., :96], _reference_rope(normalized_k, table, 96)[..., :96]
    )
    torch.testing.assert_close(called_q.transpose(1, 2)[..., 96:], normalized_q[..., 96:])
    torch.testing.assert_close(called_k.transpose(1, 2)[..., 96:], normalized_k[..., 96:])


def test_norm_and_partial_rope_preserve_autograd() -> None:
    torch.manual_seed(29)
    geometry = MiniMaxH3AttentionGeometry(16, 2, 8, 6)
    model = MiniMaxH3Attention(
        geometry,
        builtin_sdpa_kernel(),
        _evidence(),
        operations=InitlessOperations(),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    hidden = torch.randn(1, 3, 16, requires_grad=True)

    model(hidden, _rope_table(3, 6)).square().mean().backward()

    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    assert bool(torch.count_nonzero(hidden.grad))
    for parameter in model.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))


def test_kernel_call_is_exact_full_unmasked_noncausal_rank_four() -> None:
    kernel = _RecordingKernel()
    model = MiniMaxH3Attention(
        MiniMaxH3AttentionGeometry(12, 2, 6, 6),
        kernel,
        _evidence(),
        operations=InitlessOperations(),
    )
    model(torch.randn(1, 4, 12), _rope_table(4, 6))
    query, key, value, mask, causal, scale, enable_gqa = kernel.calls[0]
    assert query.shape == key.shape == value.shape == (1, 2, 4, 6)
    assert mask is None
    assert causal is False
    assert scale is None
    assert enable_gqa is False


class _FailOnceKernel(_RecordingKernel):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        super().__call__(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        if len(self.calls) == 1:
            raise RuntimeError("delegate failure")
        return v


def test_delegate_exception_propagates_and_immediate_reuse_has_no_stale_state() -> None:
    kernel = _FailOnceKernel()
    model = MiniMaxH3Attention(
        MiniMaxH3AttentionGeometry(8, 1, 8, 6),
        kernel,
        _evidence(),
        operations=InitlessOperations(),
    )
    hidden = torch.randn(1, 2, 8)
    table = _rope_table(2, 6)
    with pytest.raises(RuntimeError, match="delegate failure"):
        model(hidden, table)
    output = model(hidden, table)
    assert output.shape == hidden.shape
    assert len(kernel.calls) == 2


def test_static_boundary_keeps_h3_direct_import_only_and_s4a_refusal() -> None:
    package = Path(__file__).parents[1] / "src" / "dinkster_inference_torch"
    module_source = (package / "minimax_h3_dit.py").read_text()
    attention_source = (package / "attention.py").read_text()
    package_source = (package / "__init__.py").read_text()
    direct_sdpa = {
        path.name: path.read_text().count("scaled_dot_product_attention(")
        for path in package.glob("*.py")
        if "scaled_dot_product_attention(" in path.read_text()
    }
    assert direct_sdpa == {"attention.py": 1}
    assert "select_attention(" not in module_source
    for forbidden in (
        "AttentionRole",
        "AttentionRouteToken",
        "discover_attention_route_token",
        "native_policy",
    ):
        assert forbidden not in module_source
    assert "h3" not in attention_source.split("AttentionRole =", 1)[1].splitlines()[0]
    assert '    "MiniMaxH3Attention",' not in package_source
    assert '    "MiniMaxH3DiT",' not in package_source
    refusal = MiniMaxH3DiTExecutionRefusal("bfloat16")
    assert refusal.runtime_provider is None
    assert MINIMAX_H3_CONFIG.family_id == "dinkster.minimax_h3"


@dataclass(frozen=True)
class _ReducedConfig:
    family_id: str = "dinkster.minimax_h3"
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    depth: int = 1
    hidden_width: int = 12
    attention_heads: int = 2
    attention_head_dim: int = 6
    ffn_width: int = 16
    text_width: int = 8
    patch: tuple[int, int, int] = (1, 2, 2)
    video_spatial_downscale: int = 16
    video_fps: int = 24
    audio_content_channels: int = 2
    audio_latent_rate_hz: int = 40
    batch_size: int = 1
    video_schedule_shift: float = 12.0
    audio_schedule_shift: float = 3.0
    conditioner_id: str = "Qwen3-VL-32B"
    conditioner_layer: int = 50
    video_codec_id: str = "MiniMaxH3VideoVAE"
    audio_codec_id: str = "MiniMaxH3AudioVAE"


@dataclass(frozen=True)
class _GpuGeometryConfig(_ReducedConfig):
    hidden_width: int = MINIMAX_H3_CONFIG.hidden_width
    attention_heads: int = MINIMAX_H3_CONFIG.attention_heads
    attention_head_dim: int = MINIMAX_H3_CONFIG.attention_head_dim
    text_width: int = MINIMAX_H3_CONFIG.hidden_width


def _reduced_model(kernel: AttentionKernel | None = None) -> MiniMaxH3DiT:
    return MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel() if kernel is None else kernel,
        _evidence(),
        operations=InitlessOperations(),
    )


def _fill_reduced_model(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 19 - 9) / 128)
        state = model.state_dict()
        if "adaln_t_table" in state:
            state["adaln_t_table"].copy_(
                torch.linspace(-0.01, 0.01, state["adaln_t_table"].numel()).reshape(
                    state["adaln_t_table"].shape
                )
            )
        state["rope.inv_freq"].fill_(0.5)


def _inputs() -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    video = torch.linspace(-0.5, 0.5, 1 * 24 * 2 * 3 * 5).reshape(1, 24, 2, 3, 5)
    audio = torch.linspace(-0.25, 0.25, 1 * 32 * 2 * 3).reshape(1, 32, 2, 3)
    context = torch.linspace(-0.1, 0.1, 1 * 3 * 8).reshape(1, 3, 8)
    return _h3(video, audio), context


def _fractional_masks(
    value: MultiStreamLatent[torch.Tensor],
) -> MultiStreamLatent[torch.Tensor]:
    video_mask = torch.ones_like(value.by_role("video"))
    video_mask[:, :, :, :2, :2] = 0.25
    video_mask[:, :, :, :2, 2:4] = 0.5
    audio_mask = torch.ones_like(value.by_role("audio"))
    audio_mask[:, :, 0, 0] = 0.2
    audio_mask[:, :, 1, 1] = 0.6
    return _h3(video_mask, audio_mask)


def test_grad_mode_segment_updates_match_inference_values_without_mutation() -> None:
    torch.manual_seed(41)
    hidden = torch.randn(1, 6, 12)
    shift = torch.randn(3, 12)
    scale = torch.randn(3, 12)
    gate = torch.randn(3, 12)
    update = torch.randn(1, 6, 12)
    segments = cast(Any, ((0, 3, torch.tensor((0, 1, 2))), (4, 6, 2)))

    with torch.no_grad():
        modulated_inplace = dit_module._modulate(  # pyright: ignore[reportPrivateUsage]
            hidden.clone(), shift, scale, segments
        )
        gated_inplace = dit_module._gated_residual(  # pyright: ignore[reportPrivateUsage]
            hidden.clone(), gate, update, segments
        )

    with torch.enable_grad():
        source = hidden.clone()
        modulated = dit_module._modulate(  # pyright: ignore[reportPrivateUsage]
            source, shift, scale, segments
        )
        assert torch.equal(source, hidden)
        assert torch.equal(modulated, modulated_inplace)

        source = hidden.clone()
        gated = dit_module._gated_residual(  # pyright: ignore[reportPrivateUsage]
            source, gate, update, segments
        )
        assert torch.equal(source, hidden)
        assert torch.equal(gated, gated_inplace)


def test_checkpointed_block_backward_accepts_view_hidden() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    block = model.blocks[0]
    sequence = 4
    base = torch.linspace(-0.5, 0.5, 2 * sequence * 12).reshape(1, 2 * sequence, 12)
    base.requires_grad_(True)
    hidden = base.chunk(2, dim=1)[0]
    time = torch.linspace(-0.2, 0.2, dit_module._TIME_EMBED_DIM).reshape(  # pyright: ignore[reportPrivateUsage]
        1, -1
    )
    segments = ((0, sequence, 0),)
    table = _rope_table(sequence, 6)

    output = checkpoint(block, hidden, time, segments, table, use_reentrant=False)
    assert type(output) is torch.Tensor
    output.square().mean().backward()

    assert base.grad is not None and bool(torch.isfinite(base.grad).all())
    assert bool(torch.count_nonzero(base.grad))


def test_reduced_dit_forward_backward_under_grad_produces_finite_grads() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()

    output = model(value, 0.5, context)
    loss = output.by_role("video").square().mean() + output.by_role("audio").square().mean()
    assert bool(torch.isfinite(loss))
    loss.backward()

    parameters = dict(model.named_parameters())
    for name in (
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.0.adaln_proj.linear.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "token_refiner.blocks.0.mlp.fc1.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert bool(torch.count_nonzero(gradient)), name


def _packed_facts(
    value: MultiStreamLatent[torch.Tensor], context: torch.Tensor, model: MiniMaxH3DiT
) -> MiniMaxH3PackedSequenceFacts:
    video = dit_module._pad_video(  # pyright: ignore[reportPrivateUsage]
        value.by_role("video"), model.config.patch
    )
    layout = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        context.shape[1],
        video,
        value.by_role("audio"),
        MiniMaxH3DiTConditioning(),
    )
    return MiniMaxH3PackedSequenceFacts(layout.sequence_length, cast(Any, layout.segments))


def _gather_sequence_hidden(
    hidden: torch.Tensor,
    partition: SequencePartition,
    _shard: SequenceShard,
) -> torch.Tensor:
    gathered = [torch.empty_like(hidden) for _ in partition.shards]
    dist.all_gather(gathered, hidden)
    return torch.cat(
        tuple(
            value[:, : shard.valid_rows]
            for value, shard in zip(gathered, partition.shards, strict=True)
        ),
        dim=1,
    )


@dataclass(frozen=True, slots=True)
class _DiTSequenceCase:
    name: str
    ulysses: int
    ring: int

    @property
    def world_size(self) -> int:
        return self.ulysses * self.ring


def _run_sharded_dit_case(rank: int, case: _DiTSequenceCase, rendezvous: str, queue: Any) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        torch.set_num_threads(1)
        model = _reduced_model()
        _fill_reduced_model(model)
        value, context = _inputs()
        facts = _packed_facts(value, context, model)
        mesh = UspMesh.build(guidance=1, ulysses=case.ulysses, ring=case.ring)
        coordinate = mesh.coordinates(rank)
        partition = plan_sequence_partition(facts.sequence_length, case.world_size)
        shard_index = coordinate.ulysses * case.ring + coordinate.ring
        shard = partition.shards[shard_index]
        heads_per_rank = model.config.attention_heads // case.ulysses

        def factory(factory_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
            assert factory_facts == facts
            layout = SequenceLayout(
                facts.sequence_length,
                shard,
                facts.layout_identity,
                model.config.attention_heads,
                coordinate.ulysses * heads_per_rank,
                (coordinate.ulysses + 1) * heads_per_rank,
                shard.start,
                mesh.digest,
            )
            return _distributed_sequence_kernel(mesh, layout)

        sharding = MiniMaxH3SequenceSharding(facts, partition, shard, _gather_sequence_hidden)
        denoise_mask = _fractional_masks(value)
        with torch.no_grad():
            expected = model(value, 0.5, context, denoise_mask=denoise_mask)
            actual = model(
                value,
                0.5,
                context,
                denoise_mask=denoise_mask,
                attention_kernel_factory=factory,
                sequence_sharding=sharding,
            )
        delta = max(
            float((actual.by_role(role) - expected.by_role(role)).abs().max().item())
            for role in ("video", "audio")
        )
        if rank == 0:
            queue.put((case.name, facts.sequence_length, shard.padded_rows, delta))
    finally:
        dist.destroy_process_group()


def _execute_sharded_dit_case(case: _DiTSequenceCase) -> tuple[str, int, int, float]:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-dit-rendezvous")
        spawn(
            _run_sharded_dit_case,
            args=(case, rendezvous, queue),
            nprocs=case.world_size,
            join=True,
        )
    return cast("tuple[str, int, int, float]", queue.get())


def _gpu_geometry_model(
    device: torch.device,
    kernel: AttentionKernel | None = None,
    evidence: MiniMaxH3AttentionProviderEvidence | None = None,
) -> MiniMaxH3DiT:
    with device:
        model = MiniMaxH3DiT(
            cast(MiniMaxH3Config, _GpuGeometryConfig()),
            builtin_sdpa_kernel() if kernel is None else kernel,
            _evidence() if evidence is None else evidence,
            operations=InitlessOperations(),
        )
    model.token_refiner = cast(Any, torch.nn.Identity())
    model.condition_proj = cast(Any, torch.nn.Identity())
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(((index % 7) + 1) / 1024.0)
        table = cast(torch.Tensor, model.adaln_t_table)
        table.copy_(
            torch.linspace(
                -0.01,
                0.01,
                table.numel(),
                device=device,
            ).reshape_as(table)
        )
        cast(torch.Tensor, model.rope.inv_freq).fill_(0.5)
    return model


def _gpu_geometry_inputs(
    device: torch.device,
) -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    value, _ = _inputs()
    context = torch.linspace(
        -0.1,
        0.1,
        3 * MINIMAX_H3_CONFIG.hidden_width,
        device=device,
    ).reshape(1, 3, MINIMAX_H3_CONFIG.hidden_width)
    return (
        _h3(
            value.by_role("video").to(device),
            value.by_role("audio").to(device),
        ),
        context,
    )


def _run_cuda_sharded_dit_case(
    rank: int, case: _DiTSequenceCase, rendezvous: str, queue: Any
) -> None:
    require_gpu_tests_enabled()
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        device = torch.device("cuda", rank)
        model = _gpu_geometry_model(device)
        value, context = _gpu_geometry_inputs(device)
        facts = _packed_facts(value, context, model)
        mesh = UspMesh.build(guidance=1, ulysses=case.ulysses, ring=case.ring)
        coordinate = mesh.coordinates(rank)
        partition = plan_sequence_partition(facts.sequence_length, case.world_size)
        shard_index = coordinate.ulysses * case.ring + coordinate.ring
        shard = partition.shards[shard_index]
        heads_per_rank = model.config.attention_heads // case.ulysses

        def factory(factory_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
            assert factory_facts == facts
            layout = SequenceLayout(
                facts.sequence_length,
                shard,
                "fixed-packed-h3-gpu-order",
                model.config.attention_heads,
                coordinate.ulysses * heads_per_rank,
                (coordinate.ulysses + 1) * heads_per_rank,
                shard.start,
                mesh.digest,
            )
            return _distributed_sequence_kernel(mesh, layout)

        sharding = MiniMaxH3SequenceSharding(facts, partition, shard, _gather_sequence_hidden)
        denoise_mask = _fractional_masks(value)
        with torch.no_grad():
            expected = model(value, 0.5, context, denoise_mask=denoise_mask)
            actual = model(
                value,
                0.5,
                context,
                denoise_mask=denoise_mask,
                attention_kernel_factory=factory,
                sequence_sharding=sharding,
            )
        delta = max(
            float((actual.by_role(role) - expected.by_role(role)).abs().max().item())
            for role in ("video", "audio")
        )
        if rank == 0:
            queue.put((case.name, delta))
    finally:
        dist.destroy_process_group()


def _execute_cuda_sharded_dit_case(case: _DiTSequenceCase) -> tuple[str, float]:
    require_gpu_tests_enabled()
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-dit-cuda-rendezvous")
        spawn(
            _run_cuda_sharded_dit_case,
            args=(case, rendezvous, queue),
            nprocs=case.world_size,
            join=True,
        )
    return cast("tuple[str, float]", queue.get())


@pytest.mark.parametrize(
    ("sequence_length", "segments", "error", "match"),
    (
        (1, (), ValueError, "nonempty"),
        (1, ((-1, 1, "text"),), ValueError, "contiguous from zero"),
        (1, ((0.0, 1, "text"),), TypeError, "boundaries"),
        (2, ((1, 2, "text"),), ValueError, "contiguous from zero"),
        (3, ((0, 1, "text"), (2, 3, "video")), ValueError, "contiguous from zero"),
        (3, ((0, 2, "text"),), ValueError, "final segment stop"),
        (1, ((0, 1, "unknown"),), ValueError, "unknown segment kind"),
        (1, ((0, 0, "text"),), ValueError, "greater than start"),
    ),
)
def test_packed_sequence_facts_reject_invalid_boundaries(
    sequence_length: int,
    segments: tuple[tuple[object, object, object], ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        MiniMaxH3PackedSequenceFacts(sequence_length, cast(Any, segments))


def test_packed_sequence_facts_require_exact_sequence_length_int() -> None:
    with pytest.raises(TypeError, match="sequence_length"):
        MiniMaxH3PackedSequenceFacts(cast(Any, 1.0), ((0, 1, "text"),))


def test_packed_sequence_facts_identify_conditioning_prefix_before_target_video() -> None:
    facts = MiniMaxH3PackedSequenceFacts(
        12,
        ((0, 2, "text"), (2, 5, "audio"), (5, 12, "video")),
    )
    assert facts.conditioning_prefix_length == 5
    assert MiniMaxH3PackedSequenceFacts(4, ((0, 4, "text"),)).conditioning_prefix_length is None
    assert (
        MiniMaxH3PackedSequenceFacts(
            6, ((0, 2, "video"), (2, 6, "audio"))
        ).conditioning_prefix_length
        is None
    )


def test_sequence_sharding_record_is_frozen_and_rejects_invalid_construction() -> None:
    facts = MiniMaxH3PackedSequenceFacts(4, ((0, 4, "text"),))
    partition = plan_sequence_partition(4, 2)

    def gather(
        hidden: torch.Tensor, _partition: SequencePartition, _shard: SequenceShard
    ) -> torch.Tensor:
        return hidden

    sharding = MiniMaxH3SequenceSharding(facts, partition, partition.shards[0], gather)
    with pytest.raises(FrozenInstanceError):
        sharding.shard = partition.shards[1]  # type: ignore[misc]
    with pytest.raises(TypeError, match="facts"):
        MiniMaxH3SequenceSharding(cast(Any, object()), partition, partition.shards[0], gather)
    with pytest.raises(TypeError, match="partition"):
        MiniMaxH3SequenceSharding(facts, cast(Any, object()), partition.shards[0], gather)
    with pytest.raises(TypeError, match="shard"):
        MiniMaxH3SequenceSharding(facts, partition, cast(Any, object()), gather)
    with pytest.raises(ValueError, match="canonical"):
        foreign_partition = plan_sequence_partition(5, 2)
        MiniMaxH3SequenceSharding(facts, foreign_partition, foreign_partition.shards[0], gather)
    with pytest.raises(ValueError, match="belong"):
        MiniMaxH3SequenceSharding(facts, partition, SequenceShard(2, 0, 2, 0), gather)
    with pytest.raises(TypeError, match="gather"):
        MiniMaxH3SequenceSharding(facts, partition, partition.shards[0], cast(Any, None))


def test_sequence_sharding_requires_factory_and_matching_parallel_kernel() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    facts = _packed_facts(value, context, model)
    partition = plan_sequence_partition(facts.sequence_length, 2)
    shard = partition.shards[0]
    sharding = MiniMaxH3SequenceSharding(facts, partition, shard, _gather_sequence_hidden)

    with pytest.raises(ValueError, match="requires an attention kernel factory"):
        model(value, 0.5, context, sequence_sharding=sharding)

    def local_factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        return builtin_sdpa_kernel()

    with pytest.raises(TypeError, match="SequenceParallelAttentionKernel"):
        model(
            value,
            0.5,
            context,
            attention_kernel_factory=local_factory,
            sequence_sharding=sharding,
        )

    mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
    wrong_identity_layout = SequenceLayout(
        facts.sequence_length,
        shard,
        "another-model-packed-sequence.v1",
        model.config.attention_heads,
        0,
        1,
        shard.start,
        mesh.digest,
    )
    wrong_identity_kernel = _distributed_sequence_kernel(mesh, wrong_identity_layout)
    with pytest.raises(ValueError, match="layout identity"):
        sharding.validate_kernel(wrong_identity_kernel, model.config.attention_heads)

    wrong_layout = SequenceLayout(
        facts.sequence_length,
        partition.shards[1],
        facts.layout_identity,
        model.config.attention_heads,
        1,
        2,
        partition.shards[1].start,
        mesh.digest,
    )
    wrong_kernel = _distributed_sequence_kernel(mesh, wrong_layout)
    with pytest.raises(ValueError, match="requested shard"):
        sharding.validate_kernel(wrong_kernel, model.config.attention_heads)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
@pytest.mark.parametrize(
    "case",
    (
        _DiTSequenceCase("R2", 1, 2),
        _DiTSequenceCase("U2", 2, 1),
        _DiTSequenceCase("U2R2-padded", 2, 2),
    ),
    ids=lambda case: case.name,
)
def test_sharded_dit_matches_unsharded_cpu_reference(case: _DiTSequenceCase) -> None:
    name, sequence_length, rank_zero_padding, delta = _execute_sharded_dit_case(case)

    assert name == case.name
    assert sequence_length == 21
    if case.name.endswith("padded"):
        assert plan_sequence_partition(sequence_length, case.world_size).shards[-1].padded_rows
    assert rank_zero_padding == 0
    assert delta <= 2.0e-6


def _sampling_runtime(
    model: MiniMaxH3DiT,
    value: MultiStreamLatent[torch.Tensor],
    context: torch.Tensor,
) -> tuple[MiniMaxH3DiTRuntime, MiniMaxH3PreparedConditioning]:
    _, layout = pack_latent_streams(value)
    prepared = MiniMaxH3PreparedConditioning(
        MiniMaxH3Task.T2VA,
        5,
        context,
        MiniMaxH3DiTConditioning(),
        layout,
    )
    return (
        MiniMaxH3DiTRuntime(
            model,
            model_role="fl2va_dit",
            runtime_identity=f"native:minimax-h3:{'0' * 64}",
            compute_dtype=torch.float32,
        ),
        prepared,
    )


def _sample_reduced_runtime(
    runtime: MiniMaxH3DiTRuntime,
    value: MultiStreamLatent[torch.Tensor],
    prepared: MiniMaxH3PreparedConditioning,
    negative: MiniMaxH3PreparedConditioning | None,
) -> MultiStreamLatent[torch.Tensor]:
    return runtime.sample_multistream(
        value,
        conditioning=prepared,
        cfg=SamplingGuidance(negative, 2.0 if negative is not None else 1.0),
        sampler_id="res_multistep",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=185,
        cancelled=CancellationFlag(),
    )


def _join_spawned_processes(processes: list[Any], timeout: float) -> None:
    try:
        for process in processes:
            process.join(timeout=timeout)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join()


def _run_sequence_sampling_cpu(
    rank: int,
    with_cfg: bool,
    rendezvous: str,
    queue: Any,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    torch.set_num_threads(1)
    model = _reduced_model(
        attention_module.schedule_aware_attention_kernel("sdpa", builtin_sdpa_kernel())
    )
    _fill_reduced_model(model)
    value, context = _inputs()
    runtime, prepared = _sampling_runtime(model, value, context)
    negative = replace(prepared, context=context + 0.025) if with_cfg else None
    baseline = _sample_reduced_runtime(runtime, value, prepared, negative)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            2,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=2,
            sequence_ring=1,
        )
        runtime_module.distributed_sampling_config = lambda: config
        runtime_module.ensure_process_group = lambda: config
        distributed_module.ensure_process_group = lambda: config
        new_group_calls = 0
        original_new_group = dist.new_group

        def counted_new_group(*args: Any, **kwargs: Any) -> Any:
            nonlocal new_group_calls
            new_group_calls += 1
            return original_new_group(*args, **kwargs)

        dist.new_group = counted_new_group
        actual = _sample_reduced_runtime(runtime, value, prepared, negative)
        expected_packed, _ = pack_latent_streams(baseline)
        actual_packed, _ = pack_latent_streams(actual)
        delta = float((actual_packed - expected_packed).abs().max().item())
        if rank == 0:
            queue.put((delta, new_group_calls))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("with_cfg", (False, True), ids=("without-cfg", "with-cfg"))
def test_sequence_parallel_sample_multistream_matches_cpu_reference(with_cfg: bool) -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-cpu-rendezvous")
        spawn(
            _run_sequence_sampling_cpu,
            args=(with_cfg, rendezvous, queue),
            nprocs=2,
            join=True,
        )
    delta, new_group_calls = cast("tuple[float, int]", queue.get())
    print(f"CPU U2R1 cfg={with_cfg} max_abs_delta={delta:.9e}")
    assert delta <= 2.0e-6
    assert new_group_calls == 2


def _run_sequence_guidance_sampling_cpu(
    rank: int,
    case: _DiTSequenceCase,
    rendezvous: str,
    queue: Any,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    torch.set_num_threads(1)
    model = _reduced_model(
        attention_module.schedule_aware_attention_kernel("sdpa", builtin_sdpa_kernel())
    )
    _fill_reduced_model(model)
    value, context = _inputs()
    runtime, prepared = _sampling_runtime(model, value, context)
    negative = replace(prepared, context=context + 0.025)
    baseline = _sample_reduced_runtime(runtime, value, prepared, negative)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            4,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=case.ulysses,
            sequence_ring=case.ring,
            sequence_guidance=2,
        )
        runtime_module.distributed_sampling_config = lambda: config
        runtime_module.ensure_process_group = lambda: config
        distributed_module.ensure_process_group = lambda: config
        new_group_calls = 0
        original_new_group = dist.new_group

        def counted_new_group(*args: Any, **kwargs: Any) -> Any:
            nonlocal new_group_calls
            new_group_calls += 1
            return original_new_group(*args, **kwargs)

        dist.new_group = counted_new_group
        actual = _sample_reduced_runtime(runtime, value, prepared, negative)
        expected_packed, _ = pack_latent_streams(baseline)
        actual_packed, _ = pack_latent_streams(actual)
        delta = float((actual_packed - expected_packed).abs().max().item())
        assert delta <= 2.0e-6
        assert new_group_calls == 6
        queue.put((rank, case.name, delta, new_group_calls))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case",
    (_DiTSequenceCase("U2R1", 2, 1),),
    ids=lambda case: f"CFG2x{case.name}",
)
def test_sequence_guidance_sample_multistream_matches_cpu_reference(
    case: _DiTSequenceCase,
) -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-guidance-cpu-rendezvous")
        processes = [
            context.Process(
                target=_run_sequence_guidance_sampling_cpu,
                args=(rank, case, rendezvous, queue),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        _join_spawned_processes(processes, 30)
    results = cast("list[tuple[int, str, float, int]]", [queue.get() for _ in range(4)])
    assert {rank for rank, _name, _delta, _calls in results} == set(range(4))
    assert {name for _rank, name, _delta, _calls in results} == {case.name}
    print(
        f"CPU CFG2x{case.name} max_abs_delta="
        f"{max(delta for _rank, _name, delta, _calls in results):.9e}"
    )


def _run_sequence_guidance_refusal(rank: int, queue: Any) -> None:
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    torch.set_num_threads(1)
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    runtime, prepared = _sampling_runtime(model, value, context)
    config = DistributedSamplingConfig(
        rank,
        4,
        "sequence",
        "file:///unused",
        "1" * 32,
        sequence_ulysses=2,
        sequence_ring=1,
        sequence_guidance=2,
    )
    runtime_module.distributed_sampling_config = lambda: config
    runtime_module.ensure_process_group = lambda: config
    try:
        _sample_reduced_runtime(runtime, value, prepared, None)
    except MiniMaxH3RuntimeError as error:
        queue.put((rank, str(error)))
    else:
        raise AssertionError("sequence guidance without an unconditional lane must fail")


def test_sequence_guidance_refuses_missing_unconditional_lane_on_every_rank() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    spawn(_run_sequence_guidance_refusal, args=(queue,), nprocs=4, join=True)

    errors = dict(cast("list[tuple[int, str]]", [queue.get() for _ in range(4)]))
    assert errors == {
        rank: "single-job sequence guidance requires conditional and unconditional lanes"
        for rank in range(4)
    }


def _run_sequence_sampling_preflight_failure(
    rank: int,
    world_size: int,
    sequence_guidance: int,
    rendezvous: str,
    queue: Any,
    ring: int = 1,
    cuda: bool = False,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    torch.set_num_threads(1)
    if cuda:
        require_gpu_tests_enabled()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        model = _gpu_geometry_model(device)
        value, context = _gpu_geometry_inputs(device)
    else:
        model = _reduced_model(
            attention_module.schedule_aware_attention_kernel("sdpa", builtin_sdpa_kernel())
        )
        _fill_reduced_model(model)
        value, context = _inputs()
    runtime, prepared = _sampling_runtime(model, value, context)
    negative = replace(prepared, context=context + 0.025) if sequence_guidance == 2 else None
    if rank == 0 and ring == 1:
        prepared = replace(
            prepared,
            dit=replace(
                prepared.dit,
                text_token_tags=torch.full(context.shape[:2], 2, dtype=torch.int64),
            ),
        )
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90 if cuda else 20),
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            world_size,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=world_size // sequence_guidance // ring,
            sequence_ring=ring,
            sequence_guidance=sequence_guidance,
        )
        runtime_module.distributed_sampling_config = lambda: config
        runtime_module.ensure_process_group = lambda: config
        distributed_module.ensure_process_group = lambda: config

        def unexpected_consensus(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("invalid preflight must not reach manifest consensus")

        runtime_module.prove_manifest_consensus = unexpected_consensus
        try:
            _sample_reduced_runtime(runtime, value, prepared, negative)
        except (ValueError, RuntimeError) as error:
            queue.put((rank, str(error)))
        else:
            raise AssertionError("invalid sequence preflight must fail")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "guidance", "ring", "mode"),
    (
        (2, 1, 2, "RingSequenceShard"),
        (4, 1, 2, "UlyssesRingHybrid"),
        (4, 1, 4, "RingSequenceShard"),
        (4, 2, 2, "RingSequenceShard"),
    ),
    ids=("Ring2", "hybrid", "Ring4", "CFG-Ring"),
)
@pytest.mark.parametrize(
    "cuda",
    (
        pytest.param(False, id="cpu"),
        pytest.param(
            True,
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or torch.cuda.device_count() < 4
                or not dist.is_nccl_available(),
                reason="four CUDA GPUs with NCCL are required",
            ),
        ),
    ),
)
def test_tensor_only_runtime_refuses_ring_before_consensus(
    world_size: int,
    guidance: int,
    ring: int,
    mode: str,
    cuda: bool,
) -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-ring-capability-refusal")
        processes = [
            context.Process(
                target=_run_sequence_sampling_preflight_failure,
                args=(rank, world_size, guidance, rendezvous, queue, ring, cuda),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        _join_spawned_processes(processes, 120 if cuda else 30)
    errors = dict(cast("list[tuple[int, str]]", [queue.get() for _ in range(world_size)]))
    assert errors == {rank: f"partition mode {mode} is not declared" for rank in range(world_size)}


def test_sequence_preflight_propagates_rank_local_model_validation_failure() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-preflight-rendezvous")
        processes = [
            context.Process(
                target=_run_sequence_sampling_preflight_failure,
                args=(rank, 2, 1, rendezvous, queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        _join_spawned_processes(processes, 30)
    errors = dict(cast("list[tuple[int, str]]", [queue.get(), queue.get()]))
    assert "text token tags" in errors[0]
    assert errors[1] == "peer H3 sequence preflight failed"


def test_sequence_guidance_preflight_failure_reaches_every_rank() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-guidance-preflight-rendezvous")
        processes = [
            context.Process(
                target=_run_sequence_sampling_preflight_failure,
                args=(rank, 4, 2, rendezvous, queue),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        _join_spawned_processes(processes, 30)
    errors = dict(cast("list[tuple[int, str]]", [queue.get() for _ in range(4)]))
    assert "text token tags" in errors[0]
    assert errors == {
        0: errors[0],
        1: "peer H3 sequence preflight failed",
        2: "peer H3 sequence preflight failed",
        3: "peer H3 sequence preflight failed",
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not dist.is_nccl_available(),
    reason="four CUDA GPUs with NCCL are required",
)
@pytest.mark.parametrize(
    "case",
    (
        _DiTSequenceCase("U2R1", 2, 1),
        _DiTSequenceCase("U1R2", 1, 2),
        _DiTSequenceCase("U2R2", 2, 2),
        _DiTSequenceCase("U4R1", 4, 1),
        _DiTSequenceCase("U1R4", 1, 4),
    ),
    ids=lambda case: f"gpu-{case.name}",
)
def test_cuda_sharded_h3_dit_geometry_parity(case: _DiTSequenceCase) -> None:
    name, delta = _execute_cuda_sharded_dit_case(case)

    print(f"{name} max_abs_delta={delta:.9e}")
    assert name == case.name
    assert delta <= 2.0e-6


def _run_sequence_sampling_cuda(
    rank: int,
    case: _DiTSequenceCase,
    rendezvous: str,
    queue: Any,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    require_gpu_tests_enabled()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = _gpu_geometry_model(device)
    value, context = _gpu_geometry_inputs(device)
    runtime, prepared = _sampling_runtime(model, value, context)
    baseline = _sample_reduced_runtime(runtime, value, prepared, None)

    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            case.world_size,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=case.ulysses,
            sequence_ring=case.ring,
        )
        runtime_module.distributed_sampling_config = lambda: config
        runtime_module.ensure_process_group = lambda: config
        distributed_module.ensure_process_group = lambda: config
        actual = _sample_reduced_runtime(runtime, value, prepared, None)
        expected_packed, _ = pack_latent_streams(baseline)
        actual_packed, _ = pack_latent_streams(actual)
        delta = float((actual_packed - expected_packed).abs().max().item())
        if rank == 0:
            queue.put((case.name, delta))
    finally:
        dist.destroy_process_group()


def _execute_sequence_sampling_cuda(case: _DiTSequenceCase) -> tuple[str, float]:
    require_gpu_tests_enabled()
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-cuda-rendezvous")
        spawn(
            _run_sequence_sampling_cuda,
            args=(case, rendezvous, queue),
            nprocs=case.world_size,
            join=True,
        )
    return cast("tuple[str, float]", queue.get())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not dist.is_nccl_available(),
    reason="four CUDA GPUs with NCCL are required",
)
@pytest.mark.parametrize(
    "case",
    (
        _DiTSequenceCase("U2R1", 2, 1),
        _DiTSequenceCase("U4R1", 4, 1),
    ),
    ids=lambda case: f"gpu-runtime-{case.name}",
)
def test_cuda_sequence_parallel_sample_multistream_parity(case: _DiTSequenceCase) -> None:
    name, delta = _execute_sequence_sampling_cuda(case)

    print(f"runtime {name} max_abs_delta={delta:.9e}")
    assert name == case.name
    assert delta <= 2.0e-6


def _run_sequence_guidance_sampling_cuda(
    rank: int,
    case: _DiTSequenceCase,
    rendezvous: str,
    queue: Any,
) -> None:
    import dinkster_inference_torch.distributed as distributed_module
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    require_gpu_tests_enabled()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = _gpu_geometry_model(device)
    value, context = _gpu_geometry_inputs(device)
    runtime, prepared = _sampling_runtime(model, value, context)
    negative = replace(prepared, context=context + 0.025)
    baseline = _sample_reduced_runtime(runtime, value, prepared, negative)

    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            4,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=case.ulysses,
            sequence_ring=case.ring,
            sequence_guidance=2,
        )
        runtime_module.distributed_sampling_config = lambda: config
        runtime_module.ensure_process_group = lambda: config
        distributed_module.ensure_process_group = lambda: config
        actual = _sample_reduced_runtime(runtime, value, prepared, negative)
        expected_packed, _ = pack_latent_streams(baseline)
        actual_packed, _ = pack_latent_streams(actual)
        delta = float((actual_packed - expected_packed).abs().max().item())
        assert delta <= 2.0e-6
        queue.put((rank, case.name, delta))
    finally:
        dist.destroy_process_group()


def _execute_sequence_guidance_sampling_cuda(
    case: _DiTSequenceCase,
) -> list[tuple[int, str, float]]:
    require_gpu_tests_enabled()
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "h3-runtime-guidance-cuda-rendezvous")
        processes = [
            context.Process(
                target=_run_sequence_guidance_sampling_cuda,
                args=(rank, case, rendezvous, queue),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        _join_spawned_processes(processes, 120)
    return cast("list[tuple[int, str, float]]", [queue.get() for _ in range(4)])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not dist.is_nccl_available(),
    reason="four CUDA GPUs with NCCL are required",
)
@pytest.mark.parametrize(
    "case",
    (_DiTSequenceCase("U2R1", 2, 1),),
    ids=lambda case: f"gpu-runtime-CFG2x{case.name}",
)
def test_cuda_sequence_guidance_sample_multistream_parity(
    case: _DiTSequenceCase,
) -> None:
    results = _execute_sequence_guidance_sampling_cuda(case)

    assert {rank for rank, _name, _delta in results} == set(range(4))
    assert {name for _rank, name, _delta in results} == {case.name}
    print(
        f"runtime CFG2x{case.name} max_abs_delta="
        f"{max(delta for _rank, _name, delta in results):.9e}"
    )


def _declared_segment_kind(identity: str, modality: str, role: str) -> str:
    if identity == "text":
        return "text"
    if role == "condition":
        return "condition_audio" if modality == "audio" else "condition"
    if role == "reference":
        return f"reference_{modality}"
    return modality


def test_attention_factory_receives_exact_declared_packed_sequence_facts() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    value = _h3(value.by_role("video"), torch.zeros(1, 32, 2, 8))
    captured: list[MiniMaxH3PackedSequenceFacts] = []
    construction_kernel = cast(AttentionKernel, cast(Any, model.blocks[0].attn).attention_kernel)

    def factory(facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        captured.append(facts)
        return construction_kernel

    model(value, 0.5, context, attention_kernel_factory=factory)
    declared = plan_minimax_h3_token_layout(
        text_tokens=context.shape[1],
        target_video=MiniMaxH3VideoLatentGeometry(2, 3, 5),
        target_audio_temporal=8,
    )
    expected_segments = tuple(
        (
            segment.start,
            segment.stop,
            _declared_segment_kind(segment.identity, segment.modality, segment.role),
        )
        for segment in declared.layout.segments
    )

    assert len(captured) == 1
    assert captured[0].sequence_length == declared.layout.valid_rows
    assert captured[0].segments == expected_segments


def test_direct_attention_binds_exact_packed_sequence_facts_centrally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    captured: list[MiniMaxH3PackedSequenceFacts] = []

    def bind(kernel: AttentionKernel, facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        captured.append(facts)
        return kernel

    monkeypatch.setattr(dit_module, "bind_packed_attention_kernel", bind)
    model(value, 0.5, context)

    assert len(captured) == 1
    assert captured[0] == _packed_facts(value, context, model)


def test_h3_conditioning_sinks_apply_only_to_the_packed_dit_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import (
        AttentionModifierSchedule,
        SamplingTimelineSchedule,
        realize_sampling_timeline,
    )
    from dinkster_inference.sampling_timeline import use_realized_sampling_timeline

    calls: list[tuple[int, tuple[int, int]]] = []

    def sol_call(
        _kernel: object,
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **options: object,
    ) -> torch.Tensor:
        calls.append(
            (
                q.shape[2],
                cast("tuple[int, int]", options["sink_blocks"]),
            )
        )
        return q

    monkeypatch.setattr(type(attention_module._SOL), "call_with_tau", sol_call)  # pyright: ignore[reportPrivateUsage]
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sol",
        attention_module._SOL,  # pyright: ignore[reportPrivateUsage]
    )
    model = _reduced_model(scheduled)
    _fill_reduced_model(model)
    value, context = _inputs()
    longer_context = torch.linspace(-0.1, 0.1, 70 * context.shape[2]).reshape(
        1, 70, context.shape[2]
    )
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule(
            "sol",
            0.0,
            1.0,
            attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.0, 1.0),),
        ),
        (1.0, 0.0),
    )

    with torch.no_grad(), use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        model(value, 0.5, context)
        model(value, 0.5, longer_context)

    assert calls[:2] == [(context.shape[1], (0, 0))] * 2
    assert calls[2][0] > context.shape[1]
    assert calls[2][1] == (0, 1)
    assert calls[3:5] == [(longer_context.shape[1], (0, 0))] * 2
    assert calls[5][0] > longer_context.shape[1]
    assert calls[5][1] == (0, 2)


@pytest.mark.parametrize("conditioning_kind", ("keyframes", "references", "guides"))
def test_runtime_sequence_facts_match_complex_dit_invocations(conditioning_kind: str) -> None:
    import dinkster_inference_torch.minimax_h3_runtime as runtime_module

    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    if conditioning_kind == "keyframes":
        conditioning = MiniMaxH3DiTConditioning(
            keyframes=(
                MiniMaxH3KeyframeLatent(0, value.by_role("video")[:, :, :1]),
                MiniMaxH3KeyframeLatent(4, value.by_role("video")[:, :, :1]),
            ),
            frame_count=5,
        )
    elif conditioning_kind == "references":
        conditioning = MiniMaxH3DiTConditioning(
            references=(
                MiniMaxH3ReferenceLatents(
                    MiniMaxH3ReferenceKind.IMAGE,
                    video=value.by_role("video")[:, :, :1],
                ),
                MiniMaxH3ReferenceLatents(
                    MiniMaxH3ReferenceKind.VIDEO,
                    video=value.by_role("video"),
                    audio=value.by_role("audio"),
                ),
            )
        )
    else:
        conditioning = MiniMaxH3DiTConditioning(
            references=(
                MiniMaxH3ReferenceLatents(
                    MiniMaxH3ReferenceKind.IMAGE,
                    video=value.by_role("video")[:, :, :1],
                ),
            ),
            frame_count=5,
            guides=(
                TimelineGuide(
                    0,
                    1,
                    _h3(
                        value.by_role("video")[:, :, :1],
                        value.by_role("audio")[..., :2],
                    ),
                ),
            ),
        )
    captured: list[MiniMaxH3PackedSequenceFacts] = []

    def factory(facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        captured.append(facts)
        return builtin_sdpa_kernel()

    model(value, 0.5, context, conditioning=conditioning, attention_kernel_factory=factory)
    runtime_facts = runtime_module._packed_sequence_facts(  # pyright: ignore[reportPrivateUsage]
        value, context, conditioning, model.config.patch
    )

    assert captured == [runtime_facts]


def test_attention_factory_preserves_byte_identity_for_local_and_degenerate_kernels() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    construction_kernel = cast(AttentionKernel, cast(Any, model.blocks[0].attn).attention_kernel)

    def direct_factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        return construction_kernel

    def wrapped_factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        return SequenceParallelAttentionKernel(
            UspMesh.build(guidance=1, ulysses=1, ring=1), construction_kernel
        )

    expected = model(value, 0.5, context)
    direct = model(
        value,
        0.5,
        context,
        attention_kernel_factory=direct_factory,
    )
    wrapped = model(
        value,
        0.5,
        context,
        attention_kernel_factory=wrapped_factory,
    )

    for actual in (direct, wrapped):
        assert torch.equal(actual.by_role("video"), expected.by_role("video"))
        assert torch.equal(actual.by_role("audio"), expected.by_role("audio"))


def test_injected_attention_kernel_does_not_mutate_resident_modules_or_token_refiner() -> None:
    construction_kernel = _RecordingKernel()
    injected_kernel = _RecordingKernel()
    model = _reduced_model(construction_kernel)
    _fill_reduced_model(model)
    value, context = _inputs()
    resident = tuple(
        module.attention_kernel
        for module in model.modules()
        if isinstance(module, MiniMaxH3Attention)
    )

    before = model(value, 0.5, context)

    def injected_factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        return injected_kernel

    model(value, 0.5, context, attention_kernel_factory=injected_factory)
    after = model(value, 0.5, context)

    assert all(
        module.attention_kernel is expected
        for module, expected in zip(
            (module for module in model.modules() if isinstance(module, MiniMaxH3Attention)),
            resident,
            strict=True,
        )
    )
    assert all(kernel is construction_kernel for kernel in resident)
    assert len(injected_kernel.calls) == 1
    assert torch.equal(after.by_role("video"), before.by_role("video"))
    assert torch.equal(after.by_role("audio"), before.by_role("audio"))


def test_attention_factory_failure_and_invalid_return_do_not_poison_forward() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    expected = model(value, 0.5, context)
    calls = 0

    def raising_factory(_facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        nonlocal calls
        calls += 1
        raise RuntimeError("factory failure")

    with pytest.raises(RuntimeError, match="factory failure"):
        model(value, 0.5, context, attention_kernel_factory=raising_factory)
    assert calls == 1

    def invalid_factory(_facts: MiniMaxH3PackedSequenceFacts) -> object:
        return object()

    with pytest.raises(TypeError, match="factory must return an AttentionKernel"):
        model(value, 0.5, context, attention_kernel_factory=cast(Any, invalid_factory))
    actual = model(value, 0.5, context)
    assert torch.equal(actual.by_role("video"), expected.by_role("video"))
    assert torch.equal(actual.by_role("audio"), expected.by_role("audio"))


def test_full_profile_state_layout_exactly_matches_all_532_planned_keys() -> None:
    with torch.device("meta"):
        model = assemble_minimax_h3_dit(attention_selection=select_attention("flux", "sdpa"))
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == dict(minimax_h3_dit_layout().keys)
    assert len(model.blocks) == 50
    assert len(model.token_refiner.blocks) == 2
    assert model.video_patch_proj.in_features == 96
    assert model.audio_patch_proj.in_features == 32


def test_full_profile_mlp_time_embedding_matches_official_535_key_layout() -> None:
    with torch.device("meta"):
        model = assemble_minimax_h3_dit(
            time_embedding_kind="mlp",
            attention_selection=select_attention("flux", "sdpa"),
        )
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == dict(minimax_h3_dit_layout(time_embedding_kind="mlp").keys)
    assert "adaln_t_table" not in actual


@pytest.mark.parametrize("time_embedding_kind", ("curve", "mlp"))
def test_reduced_dit_routes_every_persistent_state_for_residency(
    time_embedding_kind: str,
) -> None:
    model = MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel(),
        _evidence(),
        operations=InitlessOperations(),
        time_embedding_kind=cast(Any, time_embedding_kind),
    )
    _fill_reduced_model(model)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    value, context = _inputs()
    resident = model(value, 0.5, context)
    mechanism.unload()
    assert mechanism.loaded_bytes() == 0
    offloaded = model(value, 0.5, context)
    assert torch.equal(offloaded.by_role("video"), resident.by_role("video"))
    assert torch.equal(offloaded.by_role("audio"), resident.by_role("audio"))


def test_mlp_time_embedder_matches_official_fp32_cos_before_sin_formula() -> None:
    module = dit_module._MiniMaxH3TimeEmbedder(  # pyright: ignore[reportPrivateUsage]
        3, operations=InitlessOperations()
    )
    with torch.no_grad():
        module.proj_in.weight.copy_(
            torch.linspace(-0.02, 0.02, module.proj_in.weight.numel()).reshape_as(
                module.proj_in.weight
            )
        )
        module.proj_in.bias.copy_(torch.tensor((-0.1, 0.0, 0.1)))
        module.proj_out.weight.copy_(
            torch.linspace(-0.01, 0.01, module.proj_out.weight.numel()).reshape_as(
                module.proj_out.weight
            )
        )
        module.proj_out.bias.zero_()
    time = torch.tensor((0.0, 0.25, 1.0), dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(128, dtype=torch.float32) / 128)
    angles = time[:, None] * frequencies[None]
    sinusoidal = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    expected = F.linear(
        F.silu(F.linear(sinusoidal, module.proj_in.weight, module.proj_in.bias)),
        module.proj_out.weight,
        module.proj_out.bias,
    )
    torch.testing.assert_close(module(time), expected, rtol=0.0, atol=0.0)


def test_mlp_final_layer_keeps_bf16_adaln_and_fp32_output_heads() -> None:
    config = cast(MiniMaxH3Config, _ReducedConfig())
    layer = dit_module._MiniMaxH3FinalLayer(  # pyright: ignore[reportPrivateUsage]
        config,
        apply_silu=True,
        operations=CastOperations(torch.bfloat16),
        fp32_operations=CastOperations(torch.float32),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.01)
    hidden = torch.ones(4, config.hidden_width, dtype=torch.bfloat16)
    time = torch.ones(2, 2688, dtype=torch.bfloat16)
    video, audio = layer(hidden, time, (0, 2, 0), (2, 4, 1))
    assert video.dtype is torch.float32
    assert audio.dtype is torch.float32
    assert video.shape == (2, 96)
    assert audio.shape == (2, 32)


def test_packed_layout_orders_text_conditions_references_audio_then_video() -> None:
    target_video = torch.empty(1, 24, 2, 4, 4)
    target_audio = torch.empty(1, 32, 2, 3)
    keyframe = torch.empty(1, 24, 1, 4, 4)
    keyframes = MiniMaxH3DiTConditioning(
        keyframes=(MiniMaxH3KeyframeLatent(0, keyframe), MiniMaxH3KeyframeLatent(4, keyframe)),
        frame_count=5,
    )
    keyframe_layout = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        2, target_video, target_audio, keyframes
    )
    assert keyframe_layout.segments == (
        (0, 2, "text"),
        (2, 6, "condition"),
        (6, 10, "condition"),
        (10, 16, "audio"),
        (16, 24, "video"),
    )
    assert keyframe_layout.video_update.tolist() == [False] * 8 + [True] * 8
    assert keyframe_layout.audio_update.tolist() == [True] * 6

    references = MiniMaxH3DiTConditioning(
        references=(
            MiniMaxH3ReferenceLatents(
                MiniMaxH3ReferenceKind.IMAGE, video=torch.empty(1, 24, 1, 2, 4)
            ),
            MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.AUDIO, audio=torch.empty(1, 32, 2, 2)),
            MiniMaxH3ReferenceLatents(
                MiniMaxH3ReferenceKind.VIDEO,
                video=torch.empty(1, 24, 2, 2, 2),
                audio=torch.empty(1, 32, 2, 3),
            ),
        )
    )
    reference_layout = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        2, target_video, target_audio, references
    )
    assert tuple(kind for _, _, kind in reference_layout.segments) == (
        "text",
        "reference_video",
        "reference_audio",
        "reference_audio",
        "reference_video",
        "audio",
        "video",
    )
    first_reference = reference_layout.segments[1]
    standalone_audio = reference_layout.segments[2]
    paired_audio = reference_layout.segments[3]
    paired_video = reference_layout.segments[4]
    assert first_reference[1] - first_reference[0] == 2
    assert standalone_audio[1] - standalone_audio[0] == 4
    assert paired_audio[1] - paired_audio[0] == 6
    assert paired_video[1] - paired_video[0] == 2


def test_timeline_guide_positions_share_the_target_origin_after_references() -> None:
    target_video = torch.empty(1, 24, 2, 4, 6)
    target_audio = torch.empty(1, 32, 2, 8)
    guide = TimelineGuide(
        2,
        1,
        _h3(torch.empty(1, 24, 1, 4, 6), torch.empty(1, 32, 2, 3)),
    )
    conditioning = MiniMaxH3DiTConditioning(
        references=(
            MiniMaxH3ReferenceLatents(
                MiniMaxH3ReferenceKind.IMAGE,
                video=torch.empty(1, 24, 1, 2, 4),
            ),
        ),
        frame_count=5,
        guides=(guide,),
    )

    layout = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        2, target_video, target_audio, conditioning
    )

    assert layout.segments == (
        (0, 2, "text"),
        (2, 8, "condition"),
        (8, 14, "condition_audio"),
        (14, 16, "reference_video"),
        (16, 32, "audio"),
        (32, 44, "video"),
    )
    anchor = 2.0 + 1.0 + 2 * 5.0 / 3.0
    assert torch.equal(layout.position_ids[2:8, 0], torch.full((6,), anchor, dtype=torch.float64))
    assert torch.equal(
        layout.position_ids[8:14, 0],
        torch.tensor((anchor, anchor + 1, anchor + 2) * 2, dtype=torch.float64),
    )
    assert layout.video_update.tolist() == [False] * 8 + [True] * 12
    assert layout.audio_update.tolist() == [False] * 6 + [True] * 16


def test_dit_consumer_refuses_overlapping_cross_modal_guide_ranges() -> None:
    model = _reduced_model()
    value, context = _inputs()
    guides = (
        TimelineGuide(
            0,
            1,
            MultiStreamLatent.from_pairs((("audio", value.by_role("audio")[..., :3].clone()),)),
        ),
        TimelineGuide(
            1,
            1,
            MultiStreamLatent.from_pairs((("video", value.by_role("video")[:, :, :1].clone()),)),
        ),
    )

    with pytest.raises(ValueError, match="guide 2 overlaps guide 1"):
        model(
            value,
            0.5,
            context,
            conditioning=MiniMaxH3DiTConditioning(frame_count=5, guides=guides),
        )


def test_last_keyframe_position_matches_pinned_reference_expression() -> None:
    layout = dit_module._PackedLayout(  # pyright: ignore[reportPrivateUsage]
        2,
        torch.empty(1, 24, 7, 4, 4),
        torch.empty(1, 32, 2, 37),
        MiniMaxH3DiTConditioning(
            keyframes=(MiniMaxH3KeyframeLatent(21, torch.empty(1, 24, 1, 4, 4)),),
            frame_count=22,
        ),
    )

    assert layout.position_ids[2, 0].item() == 37.0


def test_reduced_full_forward_crops_video_preserves_audio_and_routes_every_attention() -> None:
    spy = _RecordingKernel()
    model = _reduced_model(spy)
    _fill_reduced_model(model)
    value, context = _inputs()
    output = model(value, 0.5, context)
    assert output.by_role("video").shape == value.by_role("video").shape
    assert output.by_role("audio").shape == value.by_role("audio").shape
    assert torch.isfinite(output.by_role("video")).all()
    assert torch.isfinite(output.by_role("audio")).all()
    assert len(spy.calls) == 3
    assert spy.calls[-1][0].shape[-2] == 3 + 6 + 2 * 2 * 3


def test_reduced_dit_is_byte_identical_with_degenerate_sequence_attention() -> None:
    local = _reduced_model()
    wrapped = _reduced_model(
        SequenceParallelAttentionKernel(
            UspMesh.build(guidance=1, ulysses=1, ring=1),
            builtin_sdpa_kernel(),
        )
    )
    _fill_reduced_model(local)
    wrapped.load_state_dict(local.state_dict())
    value, context = _inputs()

    expected = local(value, 0.5, context)
    actual = wrapped(value, 0.5, context)

    assert torch.equal(actual.by_role("video"), expected.by_role("video"))
    assert torch.equal(actual.by_role("audio"), expected.by_role("audio"))


def test_reduced_mixed_bf16_fp32_storage_executes_each_dtype_island() -> None:
    model = MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel(),
        _evidence(),
        operations=CastOperations(torch.bfloat16),
        fp32_operations=CastOperations(torch.float32),
    )
    model.to(dtype=torch.bfloat16)
    model.adaln_t_table = model.adaln_t_table.float()
    model.rope.inv_freq = model.rope.inv_freq.float()
    for name, parameter in model.named_parameters():
        if name in {
            "video_patch_proj.weight",
            "video_patch_proj.bias",
            "audio_patch_proj.weight",
            "audio_patch_proj.bias",
            "final_layer.adaln_proj.linear.weight",
            "final_layer.adaln_proj.linear.bias",
            "final_layer.video_out.weight",
            "final_layer.video_out.bias",
            "final_layer.audio_out.weight",
            "final_layer.audio_out.bias",
        } or name.startswith("blocks.0.adaln_proj.linear."):
            parameter.data = parameter.data.float()
    _fill_reduced_model(model)
    value, context = _inputs()

    output = model(
        _h3(value.by_role("video").bfloat16(), value.by_role("audio").bfloat16()),
        0.5,
        context.bfloat16(),
    )

    assert output.by_role("video").dtype is torch.bfloat16
    assert output.by_role("audio").dtype is torch.bfloat16
    assert torch.isfinite(output.by_role("video")).all()
    assert torch.isfinite(output.by_role("audio")).all()


def test_preprocessed_text_embeddings_preserve_forward_output() -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()

    prepared = model.preprocess_text_embeddings(context)
    direct = model(value, 0.5, context)
    hoisted = model(value, 0.5, prepared)

    assert prepared.shape == (context.shape[0], context.shape[1], model.config.hidden_width)
    assert torch.equal(hoisted.by_role("video"), direct.by_role("video"))
    assert torch.equal(hoisted.by_role("audio"), direct.by_role("audio"))


def test_reduced_keyframe_and_reference_forwards_pack_every_realized_condition() -> None:
    spy = _RecordingKernel()
    model = _reduced_model(spy)
    _fill_reduced_model(model)
    value, context = _inputs()
    keyframe = value.by_role("video")[:, :, :1].clone()
    tags = torch.tensor(((1, 0, 1),))
    keyframe_output = model(
        value,
        0.5,
        context,
        conditioning=MiniMaxH3DiTConditioning(
            text_token_tags=tags,
            keyframes=(
                MiniMaxH3KeyframeLatent(0, keyframe),
                MiniMaxH3KeyframeLatent(4, keyframe),
            ),
            frame_count=5,
        ),
    )
    assert keyframe_output.by_role("video").shape == value.by_role("video").shape
    assert spy.calls[-1][0].shape[-2] == 3 + 6 + 2 * 6 + 12

    references = (
        MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.IMAGE, video=torch.zeros(1, 24, 1, 3, 3)),
        MiniMaxH3ReferenceLatents(MiniMaxH3ReferenceKind.AUDIO, audio=torch.zeros(1, 32, 2, 2)),
        MiniMaxH3ReferenceLatents(
            MiniMaxH3ReferenceKind.VIDEO,
            video=torch.zeros(1, 24, 2, 2, 2),
            audio=torch.zeros(1, 32, 2, 2),
        ),
    )
    reference_output = model(
        value,
        0.5,
        context,
        conditioning=MiniMaxH3DiTConditioning(
            text_token_tags=tags,
            references=references,
        ),
    )
    assert reference_output.by_role("audio").shape == value.by_role("audio").shape
    assert spy.calls[-1][0].shape[-2] == 3 + 4 + 4 + 4 + 2 + 6 + 12


def test_reduced_odd_spatial_guide_and_reference_forward_packs_every_condition() -> None:
    spy = _RecordingKernel()
    model = _reduced_model(spy)
    _fill_reduced_model(model)
    value, context = _inputs()
    guide = TimelineGuide(
        0,
        1,
        _h3(
            value.by_role("video")[:, :, :1].clone(),
            value.by_role("audio")[..., :2].clone(),
        ),
    )
    conditioning = MiniMaxH3DiTConditioning(
        references=(
            MiniMaxH3ReferenceLatents(
                MiniMaxH3ReferenceKind.IMAGE,
                video=torch.zeros(1, 24, 1, 3, 3),
            ),
        ),
        frame_count=5,
        guides=(guide,),
    )

    output = model(value, 0.5, context, conditioning=conditioning)

    assert output.by_role("video").shape == value.by_role("video").shape
    assert output.by_role("audio").shape == value.by_role("audio").shape
    assert torch.isfinite(output.by_role("video")).all()
    assert torch.isfinite(output.by_role("audio")).all()
    assert spy.calls[-1][0].shape[-2] == 3 + 6 + 4 + 4 + 6 + 12


def test_fractional_masks_drive_block_and_final_row_timesteps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    masks = _fractional_masks(value)
    video_mask = masks.by_role("video")
    audio_mask = masks.by_role("audio")
    embedded_times: list[tuple[float, ...]] = []
    modulation_segments: list[tuple[Any, ...]] = []
    final_segments: list[tuple[Any, Any]] = []
    original_curve = model._curve_time_embedding  # pyright: ignore[reportPrivateUsage]
    original_modulate = dit_module._modulate  # pyright: ignore[reportPrivateUsage]
    original_final = model.final_layer.forward

    def curve(_self: MiniMaxH3DiT, values: tuple[float, ...], device: torch.device) -> torch.Tensor:
        embedded_times.append(values)
        return original_curve(values, device)

    def modulate(
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        segments: tuple[Any, ...],
    ) -> torch.Tensor:
        modulation_segments.append(segments)
        return original_modulate(hidden, shift, scale, segments)

    def final(
        _self: torch.nn.Module,
        hidden: torch.Tensor,
        time: torch.Tensor,
        video_segment: Any,
        audio_segment: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_segments.append((video_segment, audio_segment))
        return original_final(hidden, time, video_segment, audio_segment)

    monkeypatch.setattr(model, "_curve_time_embedding", MethodType(curve, model))
    monkeypatch.setattr(dit_module, "_modulate", modulate)
    monkeypatch.setattr(model.final_layer, "forward", MethodType(final, model.final_layer))

    with torch.no_grad():
        output = model(value, 0.5, context, denoise_mask=masks)

    padded_shape = (2, 4, 6)
    video_values = dit_module._video_mask_row_values(  # pyright: ignore[reportPrivateUsage]
        video_mask, padded_shape, model.config.patch
    )
    audio_values = dit_module._audio_mask_row_values(  # pyright: ignore[reportPrivateUsage]
        audio_mask
    )
    assert video_values is not None and audio_values is not None
    torch.testing.assert_close(
        video_values,
        torch.tensor((0.25, 0.5, 1.0, 1.0, 1.0, 1.0) * 2),
    )
    torch.testing.assert_close(audio_values, torch.tensor((0.2, 1.0, 1.0, 1.0, 0.6, 1.0)))
    video_time = 0.5
    audio_sigma = MINIMAX_H3_SIGMAS.audio_sigma(0.5)
    audio_time = 1.0 - audio_sigma
    video_rows = (1.0 - video_values * 0.5).clamp(max=0.999)
    audio_rows = (1.0 - audio_values * audio_sigma).clamp(max=1.0)
    unique_times = tuple(
        sorted({video_time, audio_time} | set(video_rows.tolist()) | set(audio_rows.tolist()))
    )
    time_row = {time: index for index, time in enumerate(unique_times)}
    expected_video_time_rows = dit_module._row_time_indices(  # pyright: ignore[reportPrivateUsage]
        video_rows, time_row
    )
    expected_audio_time_rows = dit_module._row_time_indices(  # pyright: ignore[reportPrivateUsage]
        audio_rows, time_row
    )

    assert embedded_times == [unique_times]
    assert any(
        any(
            type(segment[2]) is torch.Tensor
            and torch.equal(segment[2], expected_video_time_rows * 3)
            for segment in segments
        )
        for segments in modulation_segments
    )
    assert any(
        any(
            type(segment[2]) is torch.Tensor
            and torch.equal(segment[2], expected_audio_time_rows * 3 + 2)
            for segment in segments
        )
        for segments in modulation_segments
    )
    final_video, final_audio = final_segments[0]
    assert torch.equal(final_video[2], expected_video_time_rows)
    assert torch.equal(final_audio[2], expected_audio_time_rows)
    assert bool(torch.isfinite(output.by_role("video")).all())
    assert bool(torch.isfinite(output.by_role("audio")).all())


def test_uniform_fractional_masks_use_scalar_block_and_final_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    masks = _h3(
        torch.full_like(value.by_role("video"), 0.25),
        torch.full_like(value.by_role("audio"), 0.5),
    )
    embedded_times: list[tuple[float, ...]] = []
    modulation_segments: list[tuple[Any, ...]] = []
    final_segments: list[tuple[Any, Any]] = []
    original_curve = model._curve_time_embedding  # pyright: ignore[reportPrivateUsage]
    original_modulate = dit_module._modulate  # pyright: ignore[reportPrivateUsage]
    original_final = model.final_layer.forward

    def curve(
        _self: MiniMaxH3DiT,
        values: tuple[float, ...],
        device: torch.device,
    ) -> torch.Tensor:
        embedded_times.append(values)
        return original_curve(values, device)

    def modulate(
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        segments: tuple[Any, ...],
    ) -> torch.Tensor:
        modulation_segments.append(segments)
        return original_modulate(hidden, shift, scale, segments)

    def final(
        _self: torch.nn.Module,
        hidden: torch.Tensor,
        time: torch.Tensor,
        video_segment: Any,
        audio_segment: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_segments.append((video_segment, audio_segment))
        return original_final(hidden, time, video_segment, audio_segment)

    monkeypatch.setattr(model, "_curve_time_embedding", MethodType(curve, model))
    monkeypatch.setattr(dit_module, "_modulate", modulate)
    monkeypatch.setattr(model.final_layer, "forward", MethodType(final, model.final_layer))

    with torch.no_grad():
        model(value, 0.5, context, denoise_mask=masks)

    video_values = dit_module._video_mask_row_values(  # pyright: ignore[reportPrivateUsage]
        masks.by_role("video"), (2, 4, 6), model.config.patch
    )
    audio_values = dit_module._audio_mask_row_values(  # pyright: ignore[reportPrivateUsage]
        masks.by_role("audio")
    )
    assert video_values is not None and audio_values is not None
    video_row_time = float((1.0 - video_values * 0.5).clamp(max=0.999)[0])
    audio_sigma = MINIMAX_H3_SIGMAS.audio_sigma(0.5)
    audio_row_time = float((1.0 - audio_values * audio_sigma).clamp(max=1.0)[0])
    expected_times = tuple(sorted((0.5, 1.0 - audio_sigma, video_row_time, audio_row_time)))
    time_row = {time: index for index, time in enumerate(expected_times)}

    assert embedded_times == [expected_times]
    assert all(type(segment[2]) is int for segments in modulation_segments for segment in segments)
    assert any(
        segment[2] == time_row[video_row_time] * 3
        for segments in modulation_segments
        for segment in segments
    )
    assert any(
        segment[2] == time_row[audio_row_time] * 3 + 2
        for segments in modulation_segments
        for segment in segments
    )
    final_video, final_audio = final_segments[0]
    assert type(final_video[2]) is int and final_video[2] == time_row[video_row_time]
    assert type(final_audio[2]) is int and final_audio[2] == time_row[audio_row_time]


@pytest.mark.parametrize("mask_dtype", (torch.float16, torch.float64))
def test_direct_dit_normalizes_fractional_mask_rows_to_float32(
    mask_dtype: torch.dtype,
) -> None:
    model = _reduced_model()
    _fill_reduced_model(model)
    value, context = _inputs()
    masks = _fractional_masks(value)
    typed_masks = _h3(
        masks.by_role("video").to(mask_dtype),
        masks.by_role("audio").to(mask_dtype),
    )
    float32_masks = _h3(
        typed_masks.by_role("video").float(),
        typed_masks.by_role("audio").float(),
    )

    with torch.no_grad():
        expected = model(value, 0.5, context, denoise_mask=float32_masks)
        actual = model(value, 0.5, context, denoise_mask=typed_masks)

    assert torch.equal(actual.by_role("video"), expected.by_role("video"))
    assert torch.equal(actual.by_role("audio"), expected.by_role("audio"))


def test_audio_carry_and_velocity_conversion_consume_exact_s2_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    value, context = _inputs()
    seen: list[MultiStreamLatent[torch.Tensor]] = []

    def network(
        _self: MiniMaxH3DiT,
        carried: MultiStreamLatent[torch.Tensor],
        _sigma: float,
        _context: torch.Tensor,
        _conditioning: MiniMaxH3DiTConditioning,
        _sigmas: MiniMaxH3Sigmas,
        *,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        assert denoise_mask is None
        seen.append(carried)
        return _h3(
            torch.full_like(carried.by_role("video"), 2.0),
            torch.full_like(carried.by_role("audio"), 3.0),
        )

    monkeypatch.setattr(model, "_forward_network", MethodType(network, model))
    output = model(value, 0.5, context)
    carry = MINIMAX_H3_SIGMAS.audio_state_factor(0.5)
    first, second = MINIMAX_H3_SIGMAS.audio_velocity_factors(0.5)
    torch.testing.assert_close(seen[0].by_role("audio"), value.by_role("audio") * carry)
    torch.testing.assert_close(
        output.by_role("audio"), first * (value.by_role("audio") * carry) + second * 3.0
    )
    assert torch.equal(output.by_role("video"), torch.full_like(value.by_role("video"), 2.0))


def test_keyframe_reference_tags_and_generic_control_refuse_before_projection() -> None:
    model = _reduced_model()
    value, context = _inputs()
    projected = False

    def mark_projection(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal projected
        projected = True

    hook = model.video_patch_proj.register_forward_pre_hook(mark_projection)
    try:
        with pytest.raises(ValueError, match="generic control"):
            model(value, 0.5, context, control=object())
        with pytest.raises(ValueError, match="first or declared last"):
            model(
                value,
                0.5,
                context,
                conditioning=MiniMaxH3DiTConditioning(
                    keyframes=(MiniMaxH3KeyframeLatent(4, value.by_role("video")[:, :, :1]),)
                ),
            )
        with pytest.raises(ValueError, match="vision/text tags"):
            model(
                value,
                0.5,
                context,
                conditioning=MiniMaxH3DiTConditioning(text_token_tags=torch.tensor(((0, 1, 2),))),
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            model(
                value,
                0.5,
                context,
                conditioning=MiniMaxH3DiTConditioning(
                    keyframes=(MiniMaxH3KeyframeLatent(0, value.by_role("video")[:, :, :1]),),
                    references=(
                        MiniMaxH3ReferenceLatents(
                            MiniMaxH3ReferenceKind.AUDIO, audio=value.by_role("audio")
                        ),
                    ),
                ),
            )
    finally:
        hook.remove()
    assert projected is False


def test_dit_rejects_malformed_fractional_masks_before_projection() -> None:
    model = _reduced_model()
    value, context = _inputs()
    video = torch.ones_like(value.by_role("video"))
    audio = torch.ones_like(value.by_role("audio"))

    with pytest.raises(ValueError, match="exact ordered roles"):
        model(
            value,
            0.5,
            context,
            denoise_mask=MultiStreamLatent.from_pairs((("video", video),)),
        )
    with pytest.raises(ValueError, match="match its latent shape and device"):
        model(value, 0.5, context, denoise_mask=_h3(video[..., :-1], audio))
    video[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="values must be finite"):
        model(value, 0.5, context, denoise_mask=_h3(video, audio))
    video[0, 0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"values must be within \[0, 1\]"):
        model(value, 0.5, context, denoise_mask=_h3(video, audio))


def test_modulation_and_gated_residual_apply_rows_across_sequence_segments() -> None:
    segments = ((0, 2, 0), (2, 4, 1))
    hidden = torch.ones(2, 4, 3)
    shift = torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    scale = torch.tensor(((0.5, 1.0, 1.5), (2.0, 2.5, 3.0)))
    modulated = dit_module._modulate(  # pyright: ignore[reportPrivateUsage]
        hidden.clone(), shift, scale, segments
    )
    for batch in range(2):
        torch.testing.assert_close(
            modulated[batch, :2], ((1.0 + scale[0]) + shift[0]).expand(2, -1)
        )
        torch.testing.assert_close(
            modulated[batch, 2:], ((1.0 + scale[1]) + shift[1]).expand(2, -1)
        )

    update = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    gate = torch.tensor(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
    residual = dit_module._gated_residual(  # pyright: ignore[reportPrivateUsage]
        torch.ones_like(update), gate, update, segments
    )
    torch.testing.assert_close(residual[:, :2], 1.0 + update[:, :2] * gate[0])
    torch.testing.assert_close(residual[:, 2:], 1.0 + update[:, 2:] * gate[1])

    tensor_segments = ((0, 4, torch.tensor((0, 1, 1, 0))),)
    tensor_modulated = dit_module._modulate(  # pyright: ignore[reportPrivateUsage]
        hidden.clone(), shift, scale, tensor_segments
    )
    row = tensor_segments[0][2]
    torch.testing.assert_close(
        tensor_modulated,
        hidden * (1.0 + scale[row]).unsqueeze(0) + shift[row].unsqueeze(0),
    )
    tensor_residual = dit_module._gated_residual(  # pyright: ignore[reportPrivateUsage]
        torch.ones_like(update), gate, update, tensor_segments
    )
    torch.testing.assert_close(
        tensor_residual,
        1.0 + update * gate[row].unsqueeze(0),
    )


def test_modulation_segment_translation_slices_per_row_indices() -> None:
    rows = torch.tensor((3, 4, 5, 6))
    translated = dit_module._translate_modulation_segments(  # pyright: ignore[reportPrivateUsage]
        ((0, 4, rows), (4, 8, 2)), 2, 6
    )

    assert translated[0][:2] == (0, 2)
    translated_rows = translated[0][2]
    assert type(translated_rows) is torch.Tensor
    assert torch.equal(translated_rows, torch.tensor((5, 6)))
    assert translated[1] == (2, 4, 2)


def test_block_failure_closes_prefetch_and_immediate_reuse_has_no_model_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    value, _ = _inputs()
    context = torch.zeros(1, 2, 12)
    queue = object()
    closed: list[object] = []

    def make_queue(_blocks: object) -> object:
        return queue

    def pop_queue(_queue: object, _block: object) -> None:
        return None

    monkeypatch.setattr(dit_module, "make_prefetch_queue", make_queue)
    monkeypatch.setattr(dit_module, "prefetch_queue_pop", pop_queue)
    monkeypatch.setattr(dit_module, "close_prefetch_queue", closed.append)

    class RaisingBlock(torch.nn.Module):
        def forward(self, *_args: object) -> torch.Tensor:
            raise RuntimeError("block failed")

    model.blocks = torch.nn.ModuleList((RaisingBlock(),))
    with pytest.raises(RuntimeError, match="block failed"):
        model(value, 0.5, context)
    with pytest.raises(RuntimeError, match="block failed"):
        model(value, 0.5, context)
    assert closed == [queue, queue]


class _SdpaConsumingKernel:
    """Consuming kernel that records the taken q/k/v and answers with SDPA."""

    def __init__(self) -> None:
        self.taken: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        raise AssertionError("consuming kernels must be handed leases, not borrowed tensors")

    def consume(
        self,
        q: AttentionTensorLease,
        k: AttentionTensorLease,
        v: AttentionTensorLease,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        query, key, value = q.take(), k.take(), v.take()
        self.taken.append((query, key, value))
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)


def _fill_attention(model: MiniMaxH3Attention) -> None:
    torch.manual_seed(415)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))


def _in_place_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rope_table: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    *,
    epsilon: float,
    rot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    query.copy_(
        _reference_rope(F.rms_norm(query, (head_dim,), query_weight, epsilon), rope_table, rot_dim)
    )
    key.copy_(
        _reference_rope(F.rms_norm(key, (head_dim,), key_weight, epsilon), rope_table, rot_dim)
    )
    return query, key


def test_consuming_kernel_takes_shared_projection_views_without_v_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = MiniMaxH3AttentionGeometry(12, 2, 6, 6)
    consuming = _SdpaConsumingKernel()
    model = MiniMaxH3Attention(
        geometry, cast(AttentionKernel, consuming), _evidence(), operations=InitlessOperations()
    )
    _fill_attention(model)
    monkeypatch.setattr(dinkster_kitchen, "rms_rope_split_half_", _in_place_norm_rope)
    hidden = torch.randn(1, 3, 12)
    table = _rope_table(3, 6)

    with torch.no_grad():
        actual = model(hidden, table)
        expected = model(hidden, table, attention_kernel=builtin_sdpa_kernel())

    assert len(consuming.taken) == 1
    query, key, value = consuming.taken[0]
    pointers = {t.untyped_storage().data_ptr() for t in (query, key, value)}
    assert len(pointers) == 1
    fused_bytes = 3 * value.numel() * value.element_size()
    assert value.untyped_storage().nbytes() == fused_bytes
    torch.testing.assert_close(actual, expected)


def test_consuming_kernel_receives_unaliased_v_copy_outside_the_fused_path() -> None:
    geometry = MiniMaxH3AttentionGeometry(12, 2, 6, 6)
    consuming = _SdpaConsumingKernel()
    model = MiniMaxH3Attention(
        geometry, cast(AttentionKernel, consuming), _evidence(), operations=InitlessOperations()
    )
    _fill_attention(model)
    hidden = torch.randn(1, 3, 12)

    actual = model(hidden)
    expected = model(hidden, attention_kernel=builtin_sdpa_kernel())

    assert len(consuming.taken) == 1
    _, _, value = consuming.taken[0]
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    torch.testing.assert_close(actual, expected)


def test_attention_provider_maps_selection_to_kernel_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def any_sol_device(device: torch.device | None = None) -> bool:
        del device
        return True

    sdpa_kernel, sdpa_evidence = minimax_h3_attention_provider(select_attention("flux", "sdpa"))
    assert sdpa_kernel is builtin_sdpa_kernel()
    assert sdpa_evidence == _evidence()

    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    selection = select_attention("flux", "dinkster_kitchen_int8")
    kitchen_kernel, kitchen_evidence = minimax_h3_attention_provider(selection)
    assert kitchen_kernel is selection.kernel
    assert kitchen_evidence.provider == COMFY_KITCHEN_INT8_PROVIDER
    assert kitchen_evidence.provider_version == importlib.metadata.version("dinkster-kitchen")

    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", object())
    monkeypatch.setattr(
        attention_module,
        "_KITCHEN_LIST_BACKENDS",
        lambda: {
            "cuda": {
                "available": True,
                "disabled": False,
                "capabilities": ("sol_attn",),
            }
        },
    )
    monkeypatch.setattr(attention_module, "_sol_device_supported", any_sol_device)
    sol = select_attention("flux", "sol")
    sol_kernel, sol_evidence = minimax_h3_attention_provider(sol)
    assert sol_kernel is sol.kernel
    assert sol_evidence.provider == SOL_ATTENTION_PROVIDER
    assert sol_evidence.provider_version == importlib.metadata.version("dinkster-kitchen")

    with pytest.raises(TypeError, match="exact AttentionSelection"):
        minimax_h3_attention_provider(cast(Any, object()))
    with pytest.raises(TypeError, match="exact AttentionSelection"):
        minimax_h3_attention_provider(cast(Any, None))


def test_kitchen_provider_evidence_requires_exact_installed_version() -> None:
    version = str(torch.__version__)
    kitchen_version = importlib.metadata.version("dinkster-kitchen")
    evidence = MiniMaxH3AttentionProviderEvidence(
        COMFY_KITCHEN_INT8_PROVIDER, version, kitchen_version
    )
    assert evidence.provider_version == kitchen_version
    with pytest.raises(ValueError, match="installed dinkster-kitchen version"):
        MiniMaxH3AttentionProviderEvidence(COMFY_KITCHEN_INT8_PROVIDER, version, None)
    with pytest.raises(ValueError, match="installed dinkster-kitchen version"):
        MiniMaxH3AttentionProviderEvidence(COMFY_KITCHEN_INT8_PROVIDER, version, "0.0.0")
    with pytest.raises(ValueError, match="carries no provider version"):
        MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, version, kitchen_version)


def test_sol_provider_evidence_requires_exact_installed_version() -> None:
    version = str(torch.__version__)
    kitchen_version = importlib.metadata.version("dinkster-kitchen")
    evidence = MiniMaxH3AttentionProviderEvidence(SOL_ATTENTION_PROVIDER, version, kitchen_version)
    assert evidence.provider_version == kitchen_version
    with pytest.raises(ValueError, match="installed dinkster-kitchen version"):
        MiniMaxH3AttentionProviderEvidence(SOL_ATTENTION_PROVIDER, version, None)
    with pytest.raises(ValueError, match="installed dinkster-kitchen version"):
        MiniMaxH3AttentionProviderEvidence(SOL_ATTENTION_PROVIDER, version, "0.0.0")


def test_sage_provider_evidence_requires_exact_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = str(torch.__version__)
    monkeypatch.setattr(dit_module, "sage2_distribution_version", lambda: "2.2.0.post1")
    evidence = MiniMaxH3AttentionProviderEvidence(SAGE2_PROVIDER, version, "2.2.0.post1")
    assert evidence.provider_version == "2.2.0.post1"
    with pytest.raises(ValueError, match="installed SageAttention version"):
        MiniMaxH3AttentionProviderEvidence(SAGE2_PROVIDER, version, None)
    with pytest.raises(ValueError, match="installed SageAttention version"):
        MiniMaxH3AttentionProviderEvidence(SAGE2_PROVIDER, version, "0.0.0")
    monkeypatch.setattr(dit_module, "sage2_distribution_version", lambda: None)
    with pytest.raises(ValueError, match="installed SageAttention distribution"):
        MiniMaxH3AttentionProviderEvidence(SAGE2_PROVIDER, version, "2.2.0")


def test_attention_factory_works_with_any_authenticated_provider() -> None:
    kitchen_evidence = MiniMaxH3AttentionProviderEvidence(
        COMFY_KITCHEN_INT8_PROVIDER,
        str(torch.__version__),
        importlib.metadata.version("dinkster-kitchen"),
    )
    model = MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel(),
        kitchen_evidence,
        operations=InitlessOperations(),
    )
    _fill_reduced_model(model)
    value, context = _inputs()
    facts_seen: list[MiniMaxH3PackedSequenceFacts] = []
    injected = _RecordingKernel()

    def factory(facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        facts_seen.append(facts)
        return injected

    output = model(value, 0.5, context, attention_kernel_factory=factory)
    assert len(facts_seen) == 1
    # The factory's kernel must be the one the model actually dispatches;
    # the CUDA-gated dispatch test below proves it with the real provider.
    assert len(injected.calls) == 1
    assert output.by_role("video").shape == value.by_role("video").shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dinkster_kitchen_int8_available(),
    reason="dinkster-kitchen INT8 attention on a CUDA GPU is required",
)
def test_attention_factory_dispatches_kitchen_provider_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    selection = select_attention("flux", "dinkster_kitchen_int8")
    assert selection.status.primary == "dinkster_kitchen_int8"
    kitchen_evidence = MiniMaxH3AttentionProviderEvidence(
        COMFY_KITCHEN_INT8_PROVIDER,
        str(torch.__version__),
        importlib.metadata.version("dinkster-kitchen"),
    )
    model = _gpu_geometry_model(device, kernel=selection.kernel, evidence=kitchen_evidence)
    value, context = _gpu_geometry_inputs(device)

    executed: list[str] = []
    real_attention = attention_module._KITCHEN_ATTENTION  # pyright: ignore[reportPrivateUsage]
    real_from_prequantized = (
        attention_module._KITCHEN_FROM_PREQUANTIZED  # pyright: ignore[reportPrivateUsage]
    )

    def counting_attention(*args: Any, **kwargs: Any) -> torch.Tensor:
        executed.append("attention")
        return real_attention(*args, **kwargs)

    def counting_from_prequantized(*args: Any, **kwargs: Any) -> torch.Tensor:
        executed.append("prequantized")
        return real_from_prequantized(*args, **kwargs)

    monkeypatch.setattr(attention_module, "_KITCHEN_ATTENTION", counting_attention)
    monkeypatch.setattr(attention_module, "_KITCHEN_FROM_PREQUANTIZED", counting_from_prequantized)

    facts_seen: list[MiniMaxH3PackedSequenceFacts] = []

    def factory(facts: MiniMaxH3PackedSequenceFacts) -> AttentionKernel:
        facts_seen.append(facts)
        return selection.kernel

    sdpa_model = _gpu_geometry_model(device)
    with torch.no_grad():
        expected = sdpa_model(value, 0.5, context)
        actual = model(value, 0.5, context, attention_kernel_factory=factory)

    assert len(facts_seen) == 1
    # The claimed provider's INT8 entry point genuinely ran (the fused-QKV
    # path consumes prequantized tensors), not a silent SDPA fallback.
    assert executed
    for role in ("video", "audio"):
        difference = (actual.by_role(role).float() - expected.by_role(role).float()).abs()
        # This model fills each parameter tensor with one constant, so the
        # attention output is nearly seed-free; measured max drift is 1e-8
        # (RTX PRO 6000 Blackwell, torch 2.13.0+cu130, dinkster-kitchen 0.2.31).
        # The 1e-6 cap is a gross-defect trip with 100x headroom.
        assert difference.max().item() <= 1e-6, role


def test_sequence_integration_facts_bind_provider_and_rank_geometry() -> None:
    model = _reduced_model()

    facts = minimax_h3_sequence_integration_facts(
        model, sequence_ulysses=2, sequence_ring=1, sequence_guidance=1
    )

    assert facts == (
        "topology=sequence",
        "execution_provider=bf16-linear",
        f"attention_provider={BUILTIN_SDPA_PROVIDER}",
        "sequence_ulysses=2",
        "sequence_ring=1",
        "sequence_guidance=1",
    )
    assert minimax_h3_guidance_integration_facts(model) == (
        "topology=guidance",
        "execution_provider=bf16-linear",
    )


def test_sequence_receipt_identity_changes_with_attention_provider() -> None:
    builtin = _reduced_model()
    kitchen = MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel(),
        MiniMaxH3AttentionProviderEvidence(
            COMFY_KITCHEN_INT8_PROVIDER,
            str(torch.__version__),
            importlib.metadata.version("dinkster-kitchen"),
        ),
        operations=InitlessOperations(),
    )

    def identity(model: MiniMaxH3DiT) -> str:
        return sequence_receipt_identity(
            MINIMAX_H3_CONFIG.family_id,
            minimax_h3_sequence_integration_facts(
                model, sequence_ulysses=2, sequence_ring=1, sequence_guidance=1
            ),
            torch.bfloat16,
            2,
        )

    assert identity(builtin) != identity(kitchen)


def test_sequence_integration_facts_refuse_invalid_degrees() -> None:
    model = _reduced_model()

    with pytest.raises(ValueError, match="sequence_ring degree"):
        minimax_h3_sequence_integration_facts(
            model, sequence_ulysses=2, sequence_ring=0, sequence_guidance=1
        )
    with pytest.raises(ValueError, match="sequence_ulysses degree"):
        minimax_h3_sequence_integration_facts(
            model, sequence_ulysses=True, sequence_ring=1, sequence_guidance=1
        )
