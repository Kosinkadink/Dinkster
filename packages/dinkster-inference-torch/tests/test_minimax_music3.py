from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from dinkster_inference import (
    MINIMAX_MUSIC3_CONFIG,
    Conditioning,
    FlowSigmas,
    MiniMaxMusic3DavConfig,
    MiniMaxMusic3TextConfig,
    QwenTextConfig,
    SamplingGuidance,
    minimax_music3_dav_layout,
    minimax_music3_diffusion_layout,
    minimax_music3_latent_length,
    minimax_music3_text_layout,
    split_component_conditioning,
)
from dinkster_inference_torch import (
    MiniMaxMusic3Dav,
    MiniMaxMusic3Denoiser,
    MiniMaxMusic3DiffusionRuntime,
    MiniMaxMusic3DiT,
    MiniMaxMusic3RuntimeError,
    MiniMaxMusic3TextModel,
    QwenTextModel,
    materialize_minimax_music3_conditioning,
    minimax_music3_conditioning_to_carrier,
    sample_top_k,
)
from dinkster_inference_torch import minimax_music3_model as music3_model
from dinkster_inference_torch import minimax_music3_text as music3_text
from dinkster_inference_torch.attention import builtin_sdpa_kernel
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry


def _tiny_qwen(*, merged: bool = False) -> QwenTextConfig:
    return QwenTextConfig(
        architecture="tiny_music_qwen",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        qkv_bias=False,
        qk_norm=True,
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=0,
        attention_head_dim=4,
        merged_qkv=merged,
        merged_mlp=merged,
    )


def _fill(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.linspace(-0.08, 0.08, parameter.numel(), dtype=torch.float32)
            parameter.copy_(values.reshape(parameter.shape) + index * 0.0001)


def test_full_native_modules_match_canonical_state_layouts() -> None:
    full = MiniMaxMusic3TextConfig(False, False, False, False, False, "floating")
    pruned = MiniMaxMusic3TextConfig(True, True, True, True, True, "floating")
    with torch.device("meta"):
        diffusion = MiniMaxMusic3DiT()
        dav = MiniMaxMusic3Dav()
        full_text = MiniMaxMusic3TextModel(full)
        pruned_text = MiniMaxMusic3TextModel(pruned)
    assert {key: tuple(value.shape) for key, value in diffusion.state_dict().items()} == dict(
        minimax_music3_diffusion_layout()
    )
    assert {key: tuple(value.shape) for key, value in dav.state_dict().items()} == dict(
        minimax_music3_dav_layout()
    )
    for model, config in ((full_text, full), (pruned_text, pruned)):
        expected = dict(minimax_music3_text_layout(config))
        del expected["tokenizer_json"]
        assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected


def test_merged_qwen_causal_cache_matches_one_shot_execution() -> None:
    torch.manual_seed(7)
    model = QwenTextModel(_tiny_qwen(merged=True))
    _fill(model)
    ids = torch.tensor([[1, 3, 5, 7]])
    expected = model(ids)
    cache = model.allocate_causal_cache(1, 4, dtype=torch.float32)
    prefill, _ = model.forward_causal(ids[:, :3], cache)
    final, _ = model.forward_causal(ids[:, 3:], cache, cache_position=3)
    torch.testing.assert_close(prefill, expected[:, :3], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final, expected[:, 3:], rtol=1e-5, atol=1e-6)

    embeds = model.embed_tokens(ids[:, :3])
    external, _ = model.forward_causal(None, embeds=embeds)
    torch.testing.assert_close(external, expected[:, :3], rtol=1e-5, atol=1e-6)


def test_top_k_sampling_is_source_deterministic_and_sanitizes_nonfinite() -> None:
    logits = torch.tensor([[float("nan"), float("inf"), 3.0, 2.0, float("-inf")]])
    first = sample_top_k(logits, 2, torch.Generator().manual_seed(11))
    second = sample_top_k(logits, 2, torch.Generator().manual_seed(11))
    assert torch.equal(first, second)
    assert first.item() in (1, 2)


def test_tiny_autoregressive_path_is_seed_deterministic_and_frame_aligned() -> None:
    config = MiniMaxMusic3TextConfig(
        True,
        True,
        True,
        True,
        True,
        "floating",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        audio_vocab_size=8,
        audio_num_codebooks=3,
        decoder_num_heads=2,
        decoder_intermediate_size=16,
        decoder_num_layers=1,
    )
    model = MiniMaxMusic3TextModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    ids = torch.tensor([[1, 2, 3, 4]])
    first = model.generate(ids, 19, 2, compute_dtype=torch.float32, top_k=1)
    second = model.generate(ids, 19, 2, compute_dtype=torch.float32, top_k=1)
    assert first.shape == (2, config.hidden_size * config.audio_num_codebooks)
    assert first.device.type == "cpu"
    assert torch.equal(first, second)


def test_rvq_decoder_skips_inner_prefetch_when_outer_scope_holds_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MiniMaxMusic3TextConfig(
        True,
        True,
        True,
        True,
        True,
        "floating",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        audio_vocab_size=8,
        audio_num_codebooks=3,
        decoder_num_heads=2,
        decoder_intermediate_size=16,
        decoder_num_layers=1,
    )
    decoder = MiniMaxMusic3TextModel(config).model.audio_decoder
    _fill(decoder)
    calls = 0

    def make_prefetch_queue(_blocks: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(music3_text, "make_prefetch_queue", make_prefetch_queue)
    sequence = torch.randn(2, 3, config.hidden_size)
    prefetched = decoder(sequence, prefetched=True)
    assert calls == 0
    regular = decoder(sequence)
    assert calls == 1
    torch.testing.assert_close(prefetched, regular)


class _RecordingDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.widths: list[int] = []

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        del timestep_embedding, rotary_table
        assert condition.shape[-1] == latent.shape[-1]
        self.widths.append(latent.shape[-1])
        return latent

    @staticmethod
    def prepare_timestep(timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return timestep.to(dtype=dtype)


class _WindowHarness(MiniMaxMusic3DiT):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.config = MINIMAX_MUSIC3_CONFIG
        object.__setattr__(self, "_attention_kernel", builtin_sdpa_kernel())
        self.diffusion_transformer = _RecordingDiffusion()

    def aligned_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden.shape[0],
            self.config.hidden_width,
            minimax_music3_latent_length(hidden.shape[1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )

    def prepare_rotary(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.new_empty(0)


def test_diffusion_uses_source_window_and_overlap_average() -> None:
    model = _WindowHarness()
    width = minimax_music3_latent_length(200) + 11
    latent = torch.linspace(-1, 1, width).reshape(1, 1, width).expand(1, 128, width)
    context = torch.zeros(1, 203, 8 * 4096)
    actual = model(latent, torch.ones(1), context, torch.ones(1, 1, 1))
    torch.testing.assert_close(actual, -latent)
    assert model.diffusion_transformer.widths == [
        minimax_music3_latent_length(200),
        width - minimax_music3_latent_length(100),
    ]


def test_diffusion_attention_uses_fused_split_rope_during_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = music3_model.MiniMaxMusic3Attention.__new__(music3_model.MiniMaxMusic3Attention)
    torch.nn.Module.__init__(attention)
    attention.heads = 2
    attention.head_dim = 4
    attention.to_qkv = torch.nn.Linear(8, 24, bias=False)
    attention.to_out = torch.nn.Linear(8, 8, bias=False)

    def attention_kernel(
        _query: torch.Tensor, _key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return value

    object.__setattr__(attention, "_attention_kernel", attention_kernel)
    calls = 0

    def fused(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        def rotate(value: torch.Tensor) -> torch.Tensor:
            pairs = value.reshape(*value.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
            output = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
            return output.movedim(-1, -2).flatten(-2)

        nonlocal calls
        calls += 1
        value.copy_(rotate(value))
        return value

    monkeypatch.setattr(dinkster_kitchen, "apply_rope_split_half1_", fused)
    hidden = torch.randn(1, 3, 8)
    table = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).expand(1, 1, 3, 1, 2, 2)
    with torch.inference_mode():
        output = attention(hidden, table)
    assert output.shape == hidden.shape
    assert calls == 2


def test_diffusion_glu_inference_reuses_activation_without_moving_values() -> None:
    glu = music3_model.MiniMaxMusic3Glu.__new__(music3_model.MiniMaxMusic3Glu)
    torch.nn.Module.__init__(glu)
    glu.proj = torch.nn.Linear(4, 12)
    hidden = torch.randn(2, 3, 4)
    expected = glu(hidden)
    with torch.inference_mode():
        actual = glu(hidden)
    assert torch.equal(actual, expected)


def test_diffusion_block_inference_reuses_residual_without_moving_values() -> None:
    class Attention(torch.nn.Module):
        def forward(self, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            return hidden * table

    block = music3_model.MiniMaxMusic3Block.__new__(music3_model.MiniMaxMusic3Block)
    torch.nn.Module.__init__(block)
    harness = cast("Any", block)
    harness.pre_norm = torch.nn.Identity()
    harness.self_attn = Attention()
    harness.ff_norm = torch.nn.Identity()
    harness.ff = torch.nn.Linear(4, 4)
    hidden = torch.randn(2, 3, 4)
    table = torch.tensor(0.25)
    expected = block(hidden.clone(), table)
    with torch.inference_mode():
        actual = block(hidden.clone(), table)
    assert torch.equal(actual, expected)


class _ConditionScaleMusicDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.scales: list[torch.Tensor] = []

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, context
        self.scales.append(conditioning_scale.clone())
        return conditioning_scale.expand_as(latent) * 3.0

    @staticmethod
    def aligned_condition(context: torch.Tensor) -> torch.Tensor:
        return context

    def prepare_condition(
        self,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        self.scales.append(conditioning_scale.clone())
        return conditioning_scale

    @staticmethod
    def prepare_rotary(latent: torch.Tensor) -> torch.Tensor:
        return latent.new_empty(0)

    @staticmethod
    def forward_prepared(
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, rotary_table
        return condition.expand_as(latent) * 3.0


def test_zeroed_music_conditioning_removes_condition_projection_bias() -> None:
    from dinkster_native import native_arm

    encoded = Conditioning(torch.ones((1, 2, 8 * 4096)), None)
    carrier = minimax_music3_conditioning_to_carrier(encoded)
    materialized = materialize_minimax_music3_conditioning(carrier, device="cpu")
    assert materialized.conditioning_scale is not None
    assert torch.equal(materialized.conditioning_scale, torch.ones(1))

    zeroed = cast(
        "Any",
        native_arm.GenerationConditioningZeroOut.execute(conditioning=carrier)["conditioning"],
    )
    materialized_zero = materialize_minimax_music3_conditioning(zeroed, device="cpu")
    assert torch.count_nonzero(materialized_zero.embeddings) == 0
    assert materialized_zero.conditioning_scale is not None
    assert torch.count_nonzero(materialized_zero.conditioning_scale) == 0

    model = _ConditionScaleMusicDiT()
    evaluator = MiniMaxMusic3Denoiser(cast("MiniMaxMusic3DiT", model), compute_dtype=torch.float32)
    latent = torch.zeros((1, 128, 2))
    denoised = evaluator.evaluate_conditioning(
        latent,
        1.0,
        evaluator.prepare_conditioning(materialized_zero, lane_id="negative"),
    )
    assert torch.equal(model.scales[-1], torch.zeros((1, 1, 1)))
    assert torch.equal(denoised, latent)


def test_dav_direct_decode_is_stereo_and_decode_only() -> None:
    config = replace(
        MiniMaxMusic3DavConfig(),
        latent_channels=4,
        hidden_channels=2,
        decoder_channels=4,
        strides=(2,),
    )
    model = MiniMaxMusic3Dav(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.fill_(1.0 if name.endswith("weight_g") else 0.01)
    latent = torch.linspace(-0.2, 0.2, 12).reshape(1, 4, 3)
    decoded = model.decode(latent)
    assert decoded.shape == (1, 2, 6)
    assert torch.isfinite(decoded).all()


class _ArithmeticMusicDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.aligned_calls = 0
        self.rotary_calls = 0
        self.prepared_scale = torch.ones(1, 1, 1)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1)
        return latent * 0.5 + timestep.reshape(-1, 1, 1) * conditioning_scale + scale

    def aligned_condition(self, context: torch.Tensor) -> torch.Tensor:
        self.aligned_calls += 1
        return context

    def prepare_condition(
        self,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        self.prepared_scale = conditioning_scale
        return self.aligned_condition(context)

    def prepare_rotary(self, latent: torch.Tensor) -> torch.Tensor:
        self.rotary_calls += 1
        return latent.new_empty(0)

    def forward_prepared(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        del rotary_table
        return self.forward(latent, timestep, condition, self.prepared_scale)


@torch.inference_mode()
def test_ksampler_is_bit_equal_sugar_over_custom_sampling() -> None:
    model = _ArithmeticMusicDiT()
    runtime = MiniMaxMusic3DiffusionRuntime(
        cast("MiniMaxMusic3DiT", model),
        runtime_identity="native:dinkster.minimax_music3:" + "0" * 64,
        compute_dtype=torch.float32,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )
    positive = Conditioning(torch.full((1, 2, 8 * 4096), 2.0), None)
    negative = Conditioning(torch.full((1, 2, 8 * 4096), 5.0), None)
    guidance = SamplingGuidance(negative, 1.7)
    latent = torch.rand((1, 128, 3), generator=torch.Generator().manual_seed(11))
    expected = runtime.sample(
        latent,
        cond=positive,
        cfg=guidance,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=185,
        compute_dtype=torch.float32,
    )
    assert model.aligned_calls == 1
    assert model.rotary_calls == 1
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=FlowSigmas(),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=185,
        cond=positive,
        cfg=guidance,
        error=MiniMaxMusic3RuntimeError,
    )
    assert type(result.output) is torch.Tensor
    assert torch.equal(result.output, expected)
    assert model.aligned_calls == 2
    assert model.rotary_calls == 2


def test_official_workflow_native_surface_is_complete() -> None:
    from dinkster_native.native_arm import NATIVE_ARM_NODES

    aliases: set[str] = set()
    for node in NATIVE_ARM_NODES:
        schema = node.schema()
        aliases.add(schema.node_type)
        aliases.update(schema.aliases)
    assert {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxMusic3TextEncode",
        "EmptyMiniMaxMusic3LatentAudio",
        "KSampler",
        "VAEDecodeAudio",
        "VAEDecodeAudioTiled",
    } <= aliases


def test_native_text_node_binds_conditioning_to_the_text_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch as inference_torch
    from dinkster_native import native_arm

    identity = "native:dinkster.minimax_music3:" + "1" * 64
    component = SimpleNamespace(_dinkster_minimax_music3_tokenizer=object())
    handle = SimpleNamespace(
        component=component,
        recipe=SimpleNamespace(knobs=SimpleNamespace(text_dtype="float32")),
        resource_identity=identity,
        stage=nullcontext,
    )

    class TextRuntime:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def encode_text(self, *_args: object, **_kwargs: object) -> Conditioning[torch.Tensor]:
            return Conditioning(torch.ones((1, 2, 8 * 4096)), None)

    def component_handle(*_args: object, **_kwargs: object) -> Any:
        return handle

    monkeypatch.setattr(native_arm, "_torch", lambda: torch)
    monkeypatch.setattr(native_arm, "load_registered_component", component_handle)
    monkeypatch.setattr(inference_torch, "MiniMaxMusic3TextRuntime", TextRuntime)

    output = native_arm.NativeMiniMaxMusic3TextEncode.execute(
        clip=object(),
        caption="warm ambient",
        lyrics="[verse] slow rain",
        seed=7,
        max_duration=1.0,
        cfg_scale=1.7,
        top_k=50,
    )

    _carrier, binding = split_component_conditioning(cast("Any", output["conditioning"]))
    assert binding is not None
    assert (binding.role, binding.family_id, binding.identity) == (
        "text",
        "dinkster.minimax_music3",
        identity,
    )
    assert output["seconds"] == 2 / 25


def test_native_empty_latent_and_direct_tiled_audio_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_native import native_arm

    monkeypatch.setattr(native_arm, "_torch", lambda: torch)
    latent = cast(
        "Any",
        native_arm.NativeEmptyMiniMaxMusic3LatentAudio.execute(
            seconds=1.0,
            batch_size=2,
        )["latent"],
    )
    assert latent["samples"].shape == (2, 128, minimax_music3_latent_length(25))
    assert latent["type"] == "audio"
    assert latent["downscale_ratio_temporal"] == 512

    decoded = torch.tensor([[[0.0, 2.0], [-2.0, 0.0]]])

    class Codec:
        load_device = torch.device("cpu")
        tiled: tuple[tuple[int, ...], tuple[int, ...]] | None = None

        @staticmethod
        def stage() -> Any:
            return nullcontext()

        @staticmethod
        def decode_latent(_latent: torch.Tensor) -> torch.Tensor:
            return decoded

        def decode_latent_tiled(
            self,
            _latent: torch.Tensor,
            *,
            tile: tuple[int, ...],
            overlap: tuple[int, ...],
        ) -> torch.Tensor:
            self.tiled = (tile, overlap)
            return decoded

    codec = Codec()

    def component_codec(_vae: object) -> Codec:
        return codec

    monkeypatch.setattr(native_arm, "_native_component_codec", component_codec)
    samples = {"samples": torch.zeros((1, 128, 3)), "sample_rate": 44100}
    direct = cast(
        "Any",
        native_arm.NativeVAEDecodeAudio.execute(samples=samples, vae=object())["audio"],
    )
    tiled = cast(
        "Any",
        native_arm.NativeVAEDecodeAudioTiled.execute(
            samples=samples,
            vae=object(),
            tile_size=1536,
            overlap=64,
        )["audio"],
    )
    expected = decoded / (torch.std(decoded, dim=(1, 2), keepdim=True) * 5.0)
    torch.testing.assert_close(direct["waveform"], expected)
    torch.testing.assert_close(tiled["waveform"], expected)
    assert direct["sample_rate"] == tiled["sample_rate"] == 44100
    assert codec.tiled == ((1536,), (64,))
