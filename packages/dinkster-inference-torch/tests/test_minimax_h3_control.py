"""CPU and static proofs for the MiniMax H3 Fun ControlNet-Union patch."""

from __future__ import annotations

import json
from typing import cast

import pytest
import torch
from dinkster_inference import (
    ControlApplication,
    MiniMaxH3Config,
    MultiStreamLatent,
    PayloadReference,
    PercentRange,
)
from dinkster_inference.latents import LatentStream
from dinkster_inference.quantization import LayerQuant
from dinkster_inference_torch import (
    MiniMaxH3FunControl,
    MiniMaxH3FunControlConditioning,
    MiniMaxH3FunControlPatch,
    MiniMaxH3FunControlShapes,
    is_minimax_h3_fun_state_dict,
    load_minimax_h3_fun_control,
    minimax_h3_fun_injection_layers,
    minimax_h3_fun_inpaint_post_norm,
    minimax_h3_inpaint_mask_fill,
    prepare_minimax_h3_fun_control_hint,
)
from dinkster_inference_torch.attention import BUILTIN_SDPA_PROVIDER, builtin_sdpa_kernel
from dinkster_inference_torch.minimax_h3_dit import (
    MiniMaxH3AttentionProviderEvidence,
    MiniMaxH3ControlBlockContext,
    MiniMaxH3DiT,
)
from dinkster_inference_torch.operations import InitlessOperations, bound_compute_dtype
from dinkster_inference_torch.quant_linear import Int8Linear

# Expected placements derived from the official ComfyUI implementation
# (commit 95539f563449): control blocks spread evenly over the fifty base
# blocks, so v1 carries five (every tenth) and Union 2.0 ten (every fifth).
V1_INJECTION_LAYERS = (0, 10, 20, 30, 40)
V2_INJECTION_LAYERS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)


class _ReducedConfig:
    family_id: str = "dinkster.minimax_h3"
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    depth: int = 50
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


def _evidence() -> MiniMaxH3AttentionProviderEvidence:
    return MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, str(torch.__version__))


def _fill_weights(model: torch.nn.Module) -> None:
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
        if "rope.inv_freq" in state:
            state["rope.inv_freq"].fill_(0.5)


def _reduced_dit() -> MiniMaxH3DiT:
    model = MiniMaxH3DiT(
        cast(MiniMaxH3Config, _ReducedConfig()),
        builtin_sdpa_kernel(),
        _evidence(),
        operations=InitlessOperations(),
    )
    _fill_weights(model)
    return model


def _control_shapes() -> MiniMaxH3FunControlShapes:
    return MiniMaxH3FunControlShapes(
        hidden_width=12,
        attention_heads=2,
        attention_head_dim=6,
        ffn_width=16,
    )


def _control_model(injection_layers: tuple[int, ...], **kwargs: object) -> MiniMaxH3FunControl:
    model = MiniMaxH3FunControl(
        _control_shapes(),
        builtin_sdpa_kernel(),
        _evidence(),
        injection_layers=injection_layers,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )
    _fill_weights(model)
    return model


def _dit_inputs() -> tuple[MultiStreamLatent[torch.Tensor], torch.Tensor]:
    video = torch.linspace(-0.5, 0.5, 1 * 24 * 2 * 3 * 5).reshape(1, 24, 2, 3, 5)
    audio = torch.linspace(-0.25, 0.25, 1 * 32 * 2 * 3).reshape(1, 32, 2, 3)
    context = torch.linspace(-0.1, 0.1, 1 * 3 * 8).reshape(1, 3, 8)
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio))), context


def _control_latent() -> torch.Tensor:
    # Matches the padded target video extents (2, 4, 6) with 49 control channels.
    return torch.linspace(-0.3, 0.3, 1 * 49 * 2 * 4 * 6).reshape(1, 49, 2, 4, 6)


def _control_video_latent() -> torch.Tensor:
    return torch.linspace(-0.3, 0.3, 1 * 24 * 2 * 4 * 6).reshape(1, 24, 2, 4, 6)


def test_v1_keeps_five_injections_every_tenth_block() -> None:
    assert minimax_h3_fun_injection_layers(5) == V1_INJECTION_LAYERS


def test_v2_spreads_ten_injections_every_fifth_block() -> None:
    assert minimax_h3_fun_injection_layers(10) == V2_INJECTION_LAYERS


def test_injection_layers_reject_checkpoint_without_blocks() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        minimax_h3_fun_injection_layers(0)


def test_metadata_places_override_and_must_match_checkpoint() -> None:
    places = json.dumps([0, 7, 14, 21, 28])
    assert minimax_h3_fun_injection_layers(5, places) == (0, 7, 14, 21, 28)
    with pytest.raises(ValueError, match="control_blocks_places metadata does not match"):
        minimax_h3_fun_injection_layers(10, places)
    with pytest.raises(ValueError, match="unique increasing"):
        minimax_h3_fun_injection_layers(3, json.dumps([0, 50, 20]))
    with pytest.raises(ValueError, match="divide the fifty base blocks evenly"):
        minimax_h3_fun_injection_layers(51)


def test_inpaint_post_norm_requires_exact_metadata_value() -> None:
    assert not minimax_h3_fun_inpaint_post_norm(None)
    assert not minimax_h3_fun_inpaint_post_norm({})
    assert not minimax_h3_fun_inpaint_post_norm({"inpaint_masked_pixel_mode": "pre_norm"})
    assert minimax_h3_fun_inpaint_post_norm({"inpaint_masked_pixel_mode": "post_norm"})


def test_state_dict_detection_distinguishes_fun_checkpoints() -> None:
    state_dict = _control_model(V1_INJECTION_LAYERS).state_dict()
    assert is_minimax_h3_fun_state_dict(state_dict)
    assert not is_minimax_h3_fun_state_dict({})
    assert not is_minimax_h3_fun_state_dict({"control_proj_in.weight": torch.zeros(1)})


def test_mask_fill_stays_black_for_v1_and_fills_mid_gray_for_v2() -> None:
    source = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    visibility = torch.ones(2, 1, 4, 5)
    visibility[:, :, :2, :2] = 0.0
    visibility[:, :, 2:3, 3:4] = 0.5

    black = minimax_h3_inpaint_mask_fill(source, visibility, post_norm=False)
    assert torch.equal(black, source * visibility)

    gray = minimax_h3_inpaint_mask_fill(source, visibility, post_norm=True)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    assert torch.equal(gray, source * visibility + (1.0 - visibility) * mean)
    hole = gray[:, :, 0, 0]
    assert torch.allclose(hole, mean[:, :, 0, 0].expand_as(hole))


def test_mask_fill_boundary_masks_keep_source_or_fill_everywhere() -> None:
    source = torch.randn(2, 3, 4, 5)
    full = torch.ones(2, 1, 4, 5)
    assert torch.equal(minimax_h3_inpaint_mask_fill(source, full, post_norm=True), source)
    empty = torch.zeros(2, 1, 4, 5)
    gray = minimax_h3_inpaint_mask_fill(source, empty, post_norm=True)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    assert torch.allclose(gray, mean.expand_as(source))


def test_control_hint_repeats_short_video_and_ignores_source_without_mask() -> None:
    target_shape = (1, 24, 2, 2, 3)
    encoded: list[torch.Tensor] = []

    def encode(frames: torch.Tensor) -> torch.Tensor:
        encoded.append(frames.clone())
        return torch.ones(target_shape)

    control = torch.stack((torch.ones(3, 8, 8), torch.full((3, 8, 8), 2.0)))
    hint = prepare_minimax_h3_fun_control_hint(
        target_shape,
        encode=encode,
        control_video=control,
        mask=None,
        source_video=torch.full((7, 3, 8, 8), 9.0),
        inpaint_post_norm=False,
    )

    assert hint.shape == target_shape
    assert len(encoded) == 1
    assert encoded[0].shape == (5, 3, 32, 48)
    assert torch.allclose(
        encoded[0][:2], torch.tensor((1.0, 2.0))[:, None, None, None].expand(2, 3, 32, 48)
    )
    assert torch.allclose(encoded[0][2:], torch.full((3, 3, 32, 48), 2.0))


def test_control_hint_uses_initial_frames_for_long_video() -> None:
    target_shape = (1, 24, 2, 1, 1)
    encoded: list[torch.Tensor] = []

    def encode(frames: torch.Tensor) -> torch.Tensor:
        encoded.append(frames.clone())
        return torch.zeros(target_shape)

    control = torch.arange(7, dtype=torch.float32)[:, None, None, None].expand(7, 3, 4, 4)
    prepare_minimax_h3_fun_control_hint(
        target_shape,
        encode=encode,
        control_video=control,
        mask=None,
        source_video=None,
        inpaint_post_norm=False,
    )

    assert torch.equal(encoded[0][:, 0, 0, 0], torch.arange(5, dtype=torch.float32))


def test_mask_only_control_hint_has_zero_control_and_49_channels() -> None:
    target_shape = (1, 24, 2, 2, 3)
    encoded: list[torch.Tensor] = []

    def encode(frames: torch.Tensor) -> torch.Tensor:
        encoded.append(frames.clone())
        return torch.full(target_shape, 3.0)

    hint = prepare_minimax_h3_fun_control_hint(
        target_shape,
        encode=encode,
        control_video=None,
        mask=torch.ones(2, 8, 8),
        source_video=None,
        inpaint_post_norm=False,
    )

    assert hint.shape == (1, 49, 2, 2, 3)
    assert torch.count_nonzero(hint[:, :25]) == 0
    assert torch.equal(hint[:, 25:], torch.full(target_shape, 3.0))
    assert len(encoded) == 1
    assert torch.count_nonzero(encoded[0]) == 0


def test_control_hint_rejects_malformed_target_and_encoded_shapes() -> None:
    with pytest.raises(ValueError, match="target video"):
        prepare_minimax_h3_fun_control_hint(
            (1, 16, 2, 2, 2),
            encode=lambda frames: frames,
            control_video=torch.zeros(1, 3, 4, 4),
            mask=None,
            source_video=None,
            inpaint_post_norm=False,
        )
    with pytest.raises(ValueError, match="VAE output shape"):
        prepare_minimax_h3_fun_control_hint(
            (1, 24, 2, 2, 2),
            encode=lambda frames: torch.zeros(1, 24, 1, 1, 1),
            control_video=torch.zeros(1, 3, 4, 4),
            mask=None,
            source_video=None,
            inpaint_post_norm=False,
        )


def test_loader_preserves_v1_defaults_and_propagates_v2_metadata() -> None:
    v1 = load_minimax_h3_fun_control(
        _control_model(V1_INJECTION_LAYERS).state_dict(),
        None,
        attention_kernel=builtin_sdpa_kernel(),
        evidence=_evidence(),
    )
    assert v1.injection_layers == V1_INJECTION_LAYERS
    assert not v1.inpaint_post_norm

    state_dict = _control_model(V2_INJECTION_LAYERS).state_dict()
    v2 = load_minimax_h3_fun_control(
        state_dict,
        {
            "control_blocks_places": json.dumps(list(V2_INJECTION_LAYERS)),
            "inpaint_masked_pixel_mode": "post_norm",
        },
        attention_kernel=builtin_sdpa_kernel(),
        evidence=_evidence(),
    )
    assert v2.injection_layers == V2_INJECTION_LAYERS
    assert v2.inpaint_post_norm

    derived = load_minimax_h3_fun_control(
        state_dict,
        {"inpaint_masked_pixel_mode": "post_norm"},
        attention_kernel=builtin_sdpa_kernel(),
        evidence=_evidence(),
    )
    assert derived.injection_layers == V2_INJECTION_LAYERS
    assert derived.inpaint_post_norm


def test_loader_strict_rejects_foreign_state_and_metadata_mismatch() -> None:
    state_dict = _control_model(V1_INJECTION_LAYERS).state_dict()
    with pytest.raises(ValueError, match="not a MiniMax H3 Fun control checkpoint"):
        load_minimax_h3_fun_control(
            {}, None, attention_kernel=builtin_sdpa_kernel(), evidence=_evidence()
        )
    extra = dict(state_dict)
    extra["control_blocks.9.after_proj.weight"] = torch.zeros(12, 12)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_minimax_h3_fun_control(
            extra, None, attention_kernel=builtin_sdpa_kernel(), evidence=_evidence()
        )


def test_loader_round_trips_checkpoint_and_metadata() -> None:
    original = _control_model(V2_INJECTION_LAYERS, inpaint_post_norm=True)
    metadata = {"inpaint_masked_pixel_mode": "post_norm"}
    loaded = load_minimax_h3_fun_control(
        original.state_dict(),
        metadata,
        attention_kernel=builtin_sdpa_kernel(),
        evidence=_evidence(),
    )
    assert loaded.injection_layers == original.injection_layers
    assert loaded.inpaint_post_norm == original.inpaint_post_norm
    original_state = original.state_dict()
    loaded_state = loaded.state_dict()
    assert set(original_state) == set(loaded_state)
    for key, value in original_state.items():
        assert torch.equal(value, loaded_state[key]), key


def test_loader_strict_loads_comfy_int8_convrot_control_projections() -> None:
    state = dict(_control_model(V1_INJECTION_LAYERS).state_dict())
    layers = (
        "control_blocks.0.before_proj",
        "control_blocks.3.after_proj",
    )
    config = torch.tensor(
        list(
            json.dumps(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 4,
                }
            ).encode("utf-8")
        ),
        dtype=torch.uint8,
    )
    quant: dict[str, LayerQuant] = {}
    for index, layer in enumerate(layers):
        weight_key = f"{layer}.weight"
        scale_key = f"{layer}.weight_scale"
        config_key = f"{layer}.comfy_quant"
        weight = state[weight_key]
        state[weight_key] = torch.arange(weight.numel(), dtype=torch.int8).reshape(weight.shape)
        state[scale_key] = torch.full((weight.shape[0], 1), 0.125 + index, dtype=torch.float32)
        state[config_key] = config.clone()
        quant[layer] = LayerQuant(
            layer=layer,
            format="int8_tensorwise",
            weight=weight_key,
            weight_scale=scale_key,
            config=config_key,
            parameters={"convrot": True, "convrot_groupsize": 4},
        )

    loaded = load_minimax_h3_fun_control(
        state,
        None,
        attention_kernel=builtin_sdpa_kernel(),
        evidence=_evidence(),
        quant=quant,
    )

    assert bound_compute_dtype(loaded.control_proj_in) is torch.float32
    ordinary_projection = cast("torch.nn.Module", loaded.control_blocks[1].after_proj)
    assert bound_compute_dtype(ordinary_projection) is torch.bfloat16
    loaded_state = loaded.state_dict()
    for layer in layers:
        projection = loaded.get_submodule(layer)
        assert type(projection) is Int8Linear
        assert projection.convrot
        assert projection.convrot_groupsize == 4
        assert torch.equal(loaded_state[f"{layer}.weight"], state[f"{layer}.weight"])
        assert torch.equal(loaded_state[f"{layer}.weight_scale"], state[f"{layer}.weight_scale"])
        assert f"{layer}.comfy_quant" not in loaded_state


class _RecordingPatch:
    """Control patch recording the block indices the DiT invokes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def before_base_block(self, hidden: torch.Tensor, block_index: int) -> None:
        self.calls.append(("before", block_index))

    def after_base_block(
        self, hidden: torch.Tensor, block_index: int, context: MiniMaxH3ControlBlockContext
    ) -> torch.Tensor:
        self.calls.append(("after", block_index))
        return hidden


def test_dit_control_seam_visits_every_block_in_order() -> None:
    model = _reduced_dit()
    value, context = _dit_inputs()
    recording = _RecordingPatch()
    with torch.no_grad():
        model(value, 0.5, context, control=recording)  # type: ignore[arg-type]
    before = [index for kind, index in recording.calls if kind == "before"]
    after = [index for kind, index in recording.calls if kind == "after"]
    assert before == list(range(model.config.depth))
    assert after == list(range(model.config.depth))


class _StepRecordingPatch(MiniMaxH3FunControlPatch):
    """Patch recording the base-block position of every executed control step."""

    def __init__(self, model: MiniMaxH3FunControl, control_latent: torch.Tensor) -> None:
        super().__init__(model, control_latent, 1.0)
        self.steps: list[tuple[int, int]] = []

    def after_base_block(
        self, hidden: torch.Tensor, block_index: int, context: MiniMaxH3ControlBlockContext
    ) -> torch.Tensor:
        layers = self.model.injection_layers
        if block_index in layers:
            self.steps.append((block_index, layers.index(block_index)))
        return super().after_base_block(hidden, block_index, context)


def test_control_patch_runs_exactly_ten_injections_at_v2_positions() -> None:
    dit = _reduced_dit()
    control = _control_model(V2_INJECTION_LAYERS)
    value, context = _dit_inputs()
    patch = _StepRecordingPatch(control, _control_latent())
    executions: list[int] = []
    handles = [
        block.register_forward_hook(
            lambda module, inputs, output, control_index=control_index: executions.append(
                control_index
            )
        )
        for control_index, block in enumerate(control.control_blocks)
    ]
    try:
        with torch.no_grad():
            uncontrolled = dit(value, 0.5, context)
            controlled = dit(value, 0.5, context, control=patch)
    finally:
        for handle in handles:
            handle.remove()
    assert patch.steps == list(
        zip(V2_INJECTION_LAYERS, range(len(V2_INJECTION_LAYERS)), strict=True)
    )
    assert executions == list(range(len(V2_INJECTION_LAYERS)))
    assert not torch.equal(uncontrolled.by_role("video"), controlled.by_role("video"))


def test_control_patch_runs_exactly_five_injections_at_v1_positions() -> None:
    dit = _reduced_dit()
    control = _control_model(V1_INJECTION_LAYERS)
    value, context = _dit_inputs()
    patch = _StepRecordingPatch(control, _control_latent())
    with torch.no_grad():
        dit(value, 0.5, context, control=patch)
    assert patch.steps == list(
        zip(V1_INJECTION_LAYERS, range(len(V1_INJECTION_LAYERS)), strict=True)
    )


def test_control_patch_zero_strength_matches_uncontrolled_output() -> None:
    dit = _reduced_dit()
    control = _control_model(V1_INJECTION_LAYERS)
    value, context = _dit_inputs()
    patch = MiniMaxH3FunControlPatch(control, _control_latent(), 0.0)
    with torch.no_grad():
        uncontrolled = dit(value, 0.5, context)
        controlled = dit(value, 0.5, context, control=patch)
    assert torch.equal(uncontrolled.by_role("video"), controlled.by_role("video"))
    assert torch.equal(uncontrolled.by_role("audio"), controlled.by_role("audio"))


def test_control_video_only_zero_pads_24_channels_to_the_49_channel_patch() -> None:
    dit = _reduced_dit()
    control = _control_model(V1_INJECTION_LAYERS)
    value, context = _dit_inputs()
    patch = MiniMaxH3FunControlPatch(control, _control_video_latent(), 1.0)
    with torch.no_grad():
        output = dit(value, 0.5, context, control=patch)
    assert output.roles == ("video", "audio")


def test_control_conditioning_applies_only_inside_percent_window() -> None:
    model = _control_model(V1_INJECTION_LAYERS)
    conditioning = MiniMaxH3FunControlConditioning(
        ControlApplication(
            "minimax-h3-fun",
            PayloadReference("1" * 64),
            0.75,
            PercentRange(0.25, 0.75),
        ),
        model,
        _control_latent(),
    )

    def percent_to_sigma(percent: float) -> float:
        return 1.0 - percent

    assert conditioning.patch_for_sigma(0.76, percent_to_sigma) is None
    active = conditioning.patch_for_sigma(0.5, percent_to_sigma)
    assert type(active) is MiniMaxH3FunControlPatch
    assert active.model is model
    assert active.strength == 0.75
    assert conditioning.patch_for_sigma(0.24, percent_to_sigma) is None


def test_control_rejects_invalid_strength_and_generic_control() -> None:
    control = _control_model(V1_INJECTION_LAYERS)
    with pytest.raises(ValueError, match="non-negative finite float"):
        MiniMaxH3FunControlPatch(control, _control_latent(), -1.0)
    dit = _reduced_dit()
    value, context = _dit_inputs()
    with pytest.raises(ValueError, match="does not support generic control"):
        dit(value, 0.5, context, control=object())  # type: ignore[arg-type]


def test_control_stream_refuses_adaln_width_mismatch() -> None:
    control = _control_model(V1_INJECTION_LAYERS)
    hidden = torch.zeros(1, 6, 12)
    layout = type("_Layout", (), {"video_update": torch.ones(12, dtype=torch.bool)})()
    with pytest.raises(RuntimeError, match="different adaln forms"):
        control.init_stream(hidden, _control_latent(), layout, torch.zeros(1, 4))  # type: ignore[arg-type]


def test_control_patch_fails_without_stashed_base_input() -> None:
    control = _control_model(V1_INJECTION_LAYERS)
    patch = MiniMaxH3FunControlPatch(control, _control_latent(), 1.0)
    context = type("_Context", (), {})()
    with pytest.raises(RuntimeError, match="without its base input"):
        patch.after_base_block(torch.zeros(1, 6, 12), 0, context)  # type: ignore[arg-type]
