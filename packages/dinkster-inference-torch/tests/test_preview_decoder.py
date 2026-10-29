from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    LATENT2RGB_ANIMATION_PROVIDER,
    LATENT2RGB_PROVIDER,
    LATENT2RGB_WEBP_PROVIDER,
    LATENT2WAVEFORM_PROVIDER,
    EncodedPreviewAnimation,
    LatentDescriptor,
    PreviewClip,
    PreviewFrame,
    TAEHVConfig,
    load_safetensors_header,
    plan_taehv_decoder,
    plan_taesd_decoder,
    taehv_decoder_layout,
)
from dinkster_inference.taesd import taesd_layout
from dinkster_inference_torch import (
    WAVEFORM_HEIGHT,
    WAVEFORM_WIDTH,
    FramePacer,
    SplatTensors,
    TAEHVDecoder,
    TAEHVMemBlock,
    TAEHVTGrow,
    assemble_taehv_decoder,
    assemble_taesd_decoder,
    latent2rgb_animation_decoder,
    latent2rgb_decoder,
    latent2rgb_encoded_frames_decoder,
    latent2waveform_decoder,
    preview_decoder_builders,
    taehv_preview_decoder,
    taesd_preview_decoder,
    triposplat_preview_decoder,
)
from dinkster_inference_torch import preview as preview_module
from test_autoencoder_kl import write_safetensors

DESCRIPTOR = LatentDescriptor(
    channels=2,
    rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    rgb_bias=(0.0, 0.0, -1.0),
)


def _pixels(frame: PreviewFrame) -> torch.Tensor:
    return torch.from_numpy(cast("Any", frame.rgb))


def _still(item: PreviewFrame | PreviewClip | EncodedPreviewAnimation) -> PreviewFrame:
    assert type(item) is PreviewFrame
    return item


def _clip(item: PreviewFrame | PreviewClip | EncodedPreviewAnimation) -> PreviewClip:
    assert type(item) is PreviewClip
    return item


def test_builders_are_keyed_by_provider_spec_id() -> None:
    builders = preview_decoder_builders()
    assert builders[LATENT2RGB_PROVIDER.id] is latent2rgb_decoder
    assert builders[LATENT2RGB_ANIMATION_PROVIDER.id] is latent2rgb_animation_decoder
    assert builders[LATENT2RGB_WEBP_PROVIDER.id] is latent2rgb_encoded_frames_decoder
    assert builders[LATENT2WAVEFORM_PROVIDER.id] is latent2waveform_decoder


def test_latent2rgb_requires_declared_factors() -> None:
    with pytest.raises(ValueError, match="rgb_factors"):
        latent2rgb_decoder(LatentDescriptor(channels=4))


def test_latent2rgb_projects_factors_and_bias_to_uint8() -> None:
    decode = latent2rgb_decoder(DESCRIPTOR)
    state = torch.empty(1, 2, 4, 6)
    state[0, 0] = 1.0  # red channel maps to (1+1)/2*255 = 255
    state[0, 1] = -1.0  # green channel maps to (-1+1)/2*255 = 0
    frame = _still(decode(state))
    assert (frame.width, frame.height) == (6, 4)
    pixels = _pixels(frame)
    assert pixels.dtype == torch.uint8 and tuple(pixels.shape) == (4, 6, 3)
    # Blue has no factor, only the -1 bias: it clamps at 0 as well.
    assert torch.equal(pixels[..., 0], torch.full((4, 6), 255, dtype=torch.uint8))
    assert torch.equal(pixels[..., 1], torch.zeros(4, 6, dtype=torch.uint8))
    assert torch.equal(pixels[..., 2], torch.zeros(4, 6, dtype=torch.uint8))


def test_latent2rgb_previews_the_first_frame_of_video_latents() -> None:
    decode = latent2rgb_decoder(DESCRIPTOR)
    state = torch.full((1, 2, 3, 4, 4), -1.0)
    state[0, 0, 0] = 1.0  # only temporal frame 0 lights up red
    frame = _still(decode(state))
    pixels = _pixels(frame)
    assert torch.equal(pixels[..., 0], torch.full((4, 4), 255, dtype=torch.uint8))


def test_latent2rgb_downscales_to_the_preview_edge_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preview_module, "PREVIEW_MAX_EDGE", 4)
    decode = latent2rgb_decoder(DESCRIPTOR)
    frame = _still(decode(torch.zeros(1, 2, 8, 16)))
    assert (frame.width, frame.height) == (4, 2)


def test_latent2rgb_rejects_unexpected_shapes() -> None:
    decode = latent2rgb_decoder(DESCRIPTOR)
    for shape in ((2, 4, 4), (1, 3, 4, 4), (1, 2, 2, 3, 4, 4)):
        with pytest.raises(ValueError, match="preview state"):
            decode(torch.zeros(*shape))


# -- latent2waveform audio preview ------------------------------------------------

AUDIO_DESCRIPTOR = LatentDescriptor(channels=2, dimensions=1)


def test_latent2waveform_renders_the_channel_rms_envelope() -> None:
    decode = latent2waveform_decoder(AUDIO_DESCRIPTOR)
    state = torch.zeros(1, 2, 8)
    state[0, :, 4] = 2.0  # one loud timestep
    frame = _still(decode(state))
    assert (frame.width, frame.height) == (WAVEFORM_WIDTH, WAVEFORM_HEIGHT)
    pixels = _pixels(frame)
    assert pixels.dtype == torch.uint8
    assert tuple(pixels.shape) == (WAVEFORM_HEIGHT, WAVEFORM_WIDTH, 3)
    # Greyscale waveform, symmetric around the midline.
    assert torch.equal(pixels[..., 0], pixels[..., 1])
    assert torch.equal(pixels[..., 0], pixels[..., 2])
    assert torch.equal(pixels, pixels.flip(0))
    lit = (pixels[..., 0] > 0).sum(dim=0)
    # Every column keeps at least a midline; the loud timestep's column
    # (timestep 4 of 8, resampled onto the fixed width) spans the tallest band.
    assert bool((lit >= 2).all())
    peak_column = int(lit.argmax())
    expected = int((4 + 0.5) / 8 * WAVEFORM_WIDTH)
    assert abs(peak_column - expected) <= WAVEFORM_WIDTH // 8
    assert int(lit[peak_column]) >= WAVEFORM_HEIGHT - 2


def test_latent2waveform_shows_only_the_midline_for_silence() -> None:
    decode = latent2waveform_decoder(AUDIO_DESCRIPTOR)
    pixels = _pixels(_still(decode(torch.zeros(1, 2, 16))))
    lit = (pixels[..., 0] > 0).sum(dim=0)
    assert bool((lit >= 1).all()) and bool((lit <= 2).all())


def test_latent2waveform_folds_a_stereo_axis_into_the_envelope() -> None:
    # H3's real audio geometry: [B, C=32, S=2, T].
    decode = latent2waveform_decoder(LatentDescriptor(channels=32, dimensions=1))
    state = torch.zeros(1, 32, 2, 37)
    state[0, :, 0, 20] = 3.0  # loud on the left channel only
    frame = _still(decode(state))
    assert (frame.width, frame.height) == (WAVEFORM_WIDTH, WAVEFORM_HEIGHT)
    lit = (_pixels(frame)[..., 0] > 0).sum(dim=0)
    peak_column = int(lit.argmax())
    expected = int((20 + 0.5) / 37 * WAVEFORM_WIDTH)
    assert abs(peak_column - expected) <= WAVEFORM_WIDTH // 8
    assert int(lit[peak_column]) >= WAVEFORM_HEIGHT - 2


def test_latent2waveform_rejects_unexpected_shapes() -> None:
    decode = latent2waveform_decoder(AUDIO_DESCRIPTOR)
    for shape in ((2, 8), (1, 3, 8), (1, 3, 4, 4)):
        with pytest.raises(ValueError, match="waveform preview state"):
            decode(torch.zeros(*shape))


# -- TAESD quality decoder ------------------------------------------------------


def _taesd_decoder_file(directory: Path, *, prefix: str = "") -> Path:
    torch.manual_seed(0)
    tensors = {
        prefix + key: torch.randn(shape).mul(0.05) for key, shape in taesd_layout("decoder").items()
    }
    return write_safetensors(directory / "taesd_decoder.safetensors", tensors)


@pytest.fixture(scope="module")
def taesd(tmp_path_factory: pytest.TempPathFactory) -> Any:
    path = _taesd_decoder_file(tmp_path_factory.mktemp("taesd"))
    plan = plan_taesd_decoder(load_safetensors_header(path), family="sd15")
    decoder = assemble_taesd_decoder(plan)
    return decoder, plan, taesd_preview_decoder(decoder, plan.config)


def test_taesd_plan_assemble_decode_round_trip(taesd: Any, tmp_path: Path) -> None:
    _, plan, decode = taesd
    assert plan.config.family == "sd15" and plan.config.role == "decoder"
    frame = _still(decode(torch.randn(1, 4, 8, 8)))
    assert (frame.width, frame.height) == (64, 64)
    pixels = _pixels(frame)
    assert pixels.dtype == torch.uint8 and tuple(pixels.shape) == (64, 64, 3)

    # The decoder half of a combined artifact (prefixed keys) plans and
    # assembles identically.
    prefixed = _taesd_decoder_file(tmp_path, prefix="taesd_decoder.")
    plan = plan_taesd_decoder(load_safetensors_header(prefixed), family="sd15")
    decoder = assemble_taesd_decoder(plan)
    other = _still(taesd_preview_decoder(decoder, plan.config)(torch.zeros(1, 4, 8, 8)))
    assert (other.width, other.height) == (64, 64)


def test_taesd_decode_matches_the_reference_value_convention(taesd: Any) -> None:
    decoder, plan, decode = taesd
    scale = plan.config.vae_scale
    assert scale == 0.18215 and plan.config.vae_shift == 0.0
    torch.manual_seed(1)
    latent = torch.randn(1, 4, 8, 8)
    with torch.inference_mode():
        rgb = decoder.decode(latent.mul(scale))[0]
    expected = rgb.add(1.0).div(2.0).clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0)
    assert torch.equal(_pixels(_still(decode(latent))), expected)


def test_taesd_previews_the_first_frame_of_video_latents(taesd: Any) -> None:
    _, _, decode = taesd
    torch.manual_seed(2)
    video = torch.randn(1, 4, 3, 8, 8)
    assert torch.equal(_pixels(_still(decode(video))), _pixels(_still(decode(video[:, :, 0]))))


def test_taesd_decode_rejects_unexpected_shapes(taesd: Any) -> None:
    _, _, decode = taesd
    for shape in ((4, 8, 8), (1, 3, 8, 8), (1, 4, 2, 3, 8, 8)):
        with pytest.raises(ValueError, match="preview state"):
            decode(torch.zeros(*shape))


def test_assemble_taesd_decoder_requires_a_floating_dtype(taesd: Any) -> None:
    _, plan, _ = taesd
    with pytest.raises(TypeError, match="floating"):
        assemble_taesd_decoder(plan, decoder_dtype=torch.int32)


def test_taesd_decode_leases_through_an_enrolled_offloaded_decoder(tmp_path: Path) -> None:
    from dinkster_inference_torch import enroll_component
    from dinkster_inference_torch.preview import (
        _routed_device,  # pyright: ignore[reportPrivateUsage]
    )

    path = _taesd_decoder_file(tmp_path)
    plan = plan_taesd_decoder(load_safetensors_header(path), family="sd15")
    decoder = assemble_taesd_decoder(plan)
    assert _routed_device(decoder) is None  # unenrolled: parameter device rules
    mechanism = enroll_component(
        decoder, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    # Enrolled: the latent targets the mechanism's load device, where routed
    # layers lease their weights, not wherever the stored parameters sit.
    assert _routed_device(decoder) == torch.device("cpu")
    mechanism.unload()
    frame = _still(taesd_preview_decoder(decoder, plan.config)(torch.randn(1, 4, 8, 8)))
    assert (frame.width, frame.height) == (64, 64)


def test_taesd_decoder_is_not_a_descriptor_only_builder() -> None:
    # Model-cost decoders need loaded weights; the sampling arm builds them.
    assert "dinkster.taesd" not in preview_decoder_builders()
    assert "dinkster.taehv" not in preview_decoder_builders()


# -- Frame pacing ----------------------------------------------------------------


def test_frame_pacer_rotates_full_cap_windows_without_a_rate() -> None:
    pacer = FramePacer(None, cap=3, clock=lambda: 0.0)
    assert pacer.window(8) == (0, 3)
    assert pacer.window(8) == (3, 6)
    assert pacer.window(8) == (6, 8)  # truncates at the ring end
    assert pacer.window(8) == (0, 3)  # then wraps


def test_frame_pacer_budgets_by_elapsed_display_time() -> None:
    times = iter([0.0, 0.25, 1.25, 1.26])
    pacer = FramePacer(4.0, cap=3, clock=lambda: next(times))
    assert pacer.window(100) == (0, 3)  # first call spends the full cap
    assert pacer.window(100) == (3, 4)  # 0.25s at 4 fps sustains 1 frame
    assert pacer.window(100) == (4, 7)  # 1s at 4 fps wants 4, capped at 3
    assert pacer.window(100) == (7, 8)  # instant recall still yields 1


def test_frame_pacer_restarts_when_the_ring_shrinks() -> None:
    pacer = FramePacer(None, cap=3, clock=lambda: 0.0)
    assert pacer.window(4) == (0, 3)
    assert pacer.window(2) == (0, 2)


def test_frame_pacer_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FramePacer(None, cap=0)
    with pytest.raises(ValueError, match="at least 1"):
        FramePacer(None).window(0)


# -- latent2rgb animation --------------------------------------------------------

VIDEO_DESCRIPTOR = LatentDescriptor(
    channels=2,
    dimensions=3,
    temporal_downscale=4,
    rgb_factors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    rgb_bias=(0.0, 0.0, -1.0),
)


def test_latent2rgb_animation_projects_the_pacer_window_per_timestep() -> None:
    decode = latent2rgb_animation_decoder(VIDEO_DESCRIPTOR)
    still = latent2rgb_decoder(DESCRIPTOR)
    torch.manual_seed(3)
    state = torch.randn(1, 2, 5, 4, 4)
    clip = _clip(decode(state))
    # No content rate: every call spends the full frame budget.
    assert clip.frame_indices == (0, 1, 2)
    assert clip.frame_count == 5 and clip.fps is None
    for frame, index in zip(clip.frames, clip.frame_indices, strict=True):
        assert torch.equal(_pixels(frame), _pixels(_still(still(state[:, :, index]))))
    assert _clip(decode(state)).frame_indices == (3, 4)
    assert _clip(decode(state)).frame_indices == (0, 1, 2)


def test_latent2rgb_animation_derives_fps_from_the_content_rate() -> None:
    from dataclasses import replace

    decode = latent2rgb_animation_decoder(replace(VIDEO_DESCRIPTOR, content_fps=16.0))
    clip = _clip(decode(torch.zeros(1, 2, 3, 4, 4)))
    assert clip.fps == 4.0 and clip.frame_indices == (0, 1, 2)


def test_latent2rgb_animation_rejects_still_shapes() -> None:
    decode = latent2rgb_animation_decoder(VIDEO_DESCRIPTOR)
    with pytest.raises(ValueError, match="animated preview state"):
        decode(torch.zeros(1, 2, 4, 4))


# -- latent2rgb encoded-animation frames ------------------------------------------


def test_latent2rgb_encoded_frames_decode_the_full_span_every_call() -> None:
    decode = latent2rgb_encoded_frames_decoder(VIDEO_DESCRIPTOR)
    still = latent2rgb_decoder(DESCRIPTOR)
    torch.manual_seed(3)
    state = torch.randn(1, 2, 5, 4, 4)
    for _ in range(2):  # every call re-decodes the whole span, no rotation
        clip = _clip(decode(state))
        assert clip.frame_indices == (0, 1, 2, 3, 4)
        assert clip.frame_count == 5 and len(clip.frames) == 5
        assert clip.fps is None  # no content rate on this descriptor
        for frame, index in zip(clip.frames, clip.frame_indices, strict=True):
            assert torch.equal(_pixels(frame), _pixels(_still(still(state[:, :, index]))))


def test_latent2rgb_encoded_frames_stride_to_the_cap_and_scale_fps() -> None:
    from dataclasses import replace

    monkey_cap = preview_module.PREVIEW_ENCODED_FRAME_CAP
    assert monkey_cap == 24
    decode = latent2rgb_encoded_frames_decoder(replace(VIDEO_DESCRIPTOR, content_fps=16.0))
    # 50 timesteps stride-sample at ceil(50/24)=3 to 17 frames; the display
    # rate scales down by the stride so wall-clock duration is preserved.
    clip = _clip(decode(torch.zeros(1, 2, 50, 4, 4)))
    assert clip.frame_count == 17 and len(clip.frames) == 17
    assert clip.frame_indices == tuple(range(17))
    assert clip.fps == (16.0 / 4) / 3
    # At or under the cap there is no stride and no fps scaling.
    clip = _clip(decode(torch.zeros(1, 2, 24, 4, 4)))
    assert clip.frame_count == 24 and clip.fps == 4.0


def test_latent2rgb_encoded_frames_require_factors_and_video_shapes() -> None:
    with pytest.raises(ValueError, match="rgb_factors"):
        latent2rgb_encoded_frames_decoder(LatentDescriptor(channels=4, dimensions=3))
    decode = latent2rgb_encoded_frames_decoder(VIDEO_DESCRIPTOR)
    for shape in ((1, 2, 4, 4), (1, 3, 2, 4, 4), (2, 4, 4)):
        with pytest.raises(ValueError, match="encoded animation preview state"):
            decode(torch.zeros(*shape))


# -- TAEHV video decoder ---------------------------------------------------------

TAEHV_16 = TAEHVConfig(16, 1)


def _taehv_decoder_file(
    directory: Path, *, config: TAEHVConfig = TAEHV_16, prefix: str = ""
) -> Path:
    torch.manual_seed(0)
    tensors = {
        prefix + key: torch.randn(shape).mul(0.05)
        for key, shape in taehv_decoder_layout(config).items()
    }
    return write_safetensors(directory / "taehv_decoder.safetensors", tensors)


def _sequential_reference_decode(decoder: TAEHVDecoder, latent: torch.Tensor) -> torch.Tensor:
    """The reference's sequential mode (apply_model_with_memblocks with
    parallel=False, comfy/taehv/taehv.py @ 783545f6): one timestep rides
    depth-first through the blocks, each MemBlock remembering its previous
    input. An independent path to the same math as the parallel decode."""
    config = decoder.config
    batch = latent.shape[0]
    modules = list(decoder)
    mem: list[torch.Tensor | None] = [None] * len(modules)
    out: list[torch.Tensor] = []
    work = deque(
        (timestep.squeeze(1), 0) for timestep in latent.movedim(1, 2).chunk(latent.shape[2], dim=1)
    )
    while work:
        xt, i = work.popleft()
        if i == len(modules):
            if config.patch_size > 1:
                xt = torch.nn.functional.pixel_shuffle(xt, config.patch_size)
            out.append(xt)
            continue
        module = modules[i]
        if isinstance(module, TAEHVMemBlock):
            past = torch.zeros_like(xt) if mem[i] is None else mem[i]
            mem[i], xt = xt.detach().clone(), module(xt, past)
            work.appendleft((xt, i + 1))
        elif isinstance(module, TAEHVTGrow):
            channels = xt.shape[1]
            grown = module(xt).view(batch, module.stride * channels, *xt.shape[2:])
            for chunk in reversed(grown.chunk(module.stride, 1)):
                work.appendleft((chunk, i + 1))
        else:
            work.appendleft((module(xt), i + 1))
    return torch.stack(out, 1)[:, config.frames_to_trim :]


@pytest.fixture(scope="module")
def taehv(tmp_path_factory: pytest.TempPathFactory) -> Any:
    path = _taehv_decoder_file(tmp_path_factory.mktemp("taehv"))
    plan = plan_taehv_decoder(load_safetensors_header(path))
    return assemble_taehv_decoder(plan), plan


def test_taehv_plan_assemble_decode_round_trip(taehv: Any, tmp_path: Path) -> None:
    decoder, plan = taehv
    assert plan.config == TAEHV_16
    with torch.inference_mode():
        content = decoder.decode(torch.randn(1, 16, 3, 4, 4))
    assert tuple(content.shape) == (1, 9, 3, 32, 32)

    # The decoder half of a combined artifact (prefixed keys, encoder half
    # present) plans and assembles identically.
    torch.manual_seed(0)
    tensors = {
        "decoder." + key: torch.randn(shape).mul(0.05)
        for key, shape in taehv_decoder_layout(TAEHV_16).items()
    }
    tensors["encoder.1.weight"] = torch.zeros(64, 3, 3, 3)
    combined = write_safetensors(tmp_path / "combined.safetensors", tensors)
    plan = plan_taehv_decoder(load_safetensors_header(combined))
    assert plan.config == TAEHV_16
    with torch.inference_mode():
        other = assemble_taehv_decoder(plan).decode(torch.randn(1, 16, 2, 4, 4))
    assert tuple(other.shape) == (1, 5, 3, 32, 32)


def test_taehv_parallel_decode_matches_the_sequential_reference(taehv: Any) -> None:
    decoder, _ = taehv
    torch.manual_seed(4)
    latent = torch.randn(1, 16, 3, 4, 4)
    with torch.inference_mode():
        parallel = decoder.decode(latent)
        sequential = _sequential_reference_decode(decoder, latent)
    # Batched-vs-per-timestep conv only reorders float accumulation, so the
    # error stays tiny relative to the activation scale; a semantic
    # difference (wrong past frame, ordering, trim) is O(1) of that scale.
    scale = sequential.abs().max()
    assert (parallel - sequential).abs().max() <= 1e-4 * scale


def test_taehv_patch2_variant_pixel_shuffles_to_16x_spatial(tmp_path: Path) -> None:
    path = _taehv_decoder_file(tmp_path, config=TAEHVConfig(48, 2))
    plan = plan_taehv_decoder(load_safetensors_header(path))
    assert plan.config == TAEHVConfig(48, 2)
    decoder = assemble_taehv_decoder(plan)
    torch.manual_seed(5)
    latent = torch.randn(1, 48, 2, 3, 5)
    with torch.inference_mode():
        content = decoder.decode(latent)
        sequential = _sequential_reference_decode(decoder, latent)
    assert tuple(content.shape) == (1, 5, 3, 48, 80)
    scale = sequential.abs().max()
    assert (content - sequential).abs().max() <= 1e-4 * scale


def test_taehv_decode_rejects_unexpected_shapes(taehv: Any) -> None:
    decoder, _ = taehv
    for shape in ((16, 3, 4, 4), (1, 8, 3, 4, 4)):
        with pytest.raises(ValueError, match="TAEHV decode"):
            decoder.decode(torch.zeros(*shape))


def test_assemble_taehv_decoder_requires_a_floating_dtype(taehv: Any) -> None:
    _, plan = taehv
    with pytest.raises(TypeError, match="floating"):
        assemble_taehv_decoder(plan, decoder_dtype=torch.int32)


def test_taehv_preview_decoder_yields_one_frame_per_latent_timestep(taehv: Any) -> None:
    decoder, plan = taehv
    descriptor = LatentDescriptor(channels=16, dimensions=3, temporal_downscale=4, content_fps=16.0)
    decode = taehv_preview_decoder(decoder, plan.config, descriptor)
    torch.manual_seed(6)
    state = torch.randn(2, 16, 3, 4, 4)  # batch entries beyond 0 are ignored
    clip = _clip(decode(state))
    assert clip.frame_indices == (0, 1, 2)
    assert clip.frame_count == 3 and clip.fps == 4.0
    with torch.inference_mode():
        content = decoder.decode(state[:1])[0, ::4]
    for frame, expected in zip(clip.frames, content, strict=True):
        # The light video TAEs output [0, 1]: clamp and truncate, no rescale.
        pixels = expected.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0)
        assert torch.equal(_pixels(frame), pixels)


def test_taehv_preview_decoder_rejects_still_shapes(taehv: Any) -> None:
    decoder, plan = taehv
    decode = taehv_preview_decoder(decoder, plan.config, LatentDescriptor(channels=16))
    with pytest.raises(ValueError, match="animated preview state"):
        decode(torch.zeros(1, 16, 4, 4))


def _splat(
    view_points: list[tuple[float, float, float]],
    colors: list[tuple[float, float, float]],
    *,
    opacity: float = 1.0,
) -> SplatTensors:
    """Splat tensors whose positions project to the given viewer-space
    points under the preview camera (the view matrix is orthonormal, so
    its transpose inverts the rotation)."""
    import numpy as np

    inverse = preview_module._splat_view_matrix(  # pyright: ignore[reportPrivateUsage]
        preview_module.PREVIEW_SPLAT_YAW, preview_module.PREVIEW_SPLAT_PITCH
    ).T
    positions = torch.tensor(np.array(view_points, dtype=np.float32) @ inverse.T.astype(np.float32))
    count = len(view_points)
    c0 = preview_module._SPLAT_SH_C0  # pyright: ignore[reportPrivateUsage]
    sh0 = (torch.tensor(colors, dtype=torch.float32) - 0.5) / c0
    return SplatTensors(
        positions=positions,
        scales=torch.full((count, 3), 1e-4),
        rotations=torch.zeros(count, 4),
        opacities=torch.full((count, 1), opacity),
        sh=sh0.unsqueeze(1),
    )


def test_splat_frame_zbuffers_the_nearest_gaussian() -> None:
    size = preview_module.PREVIEW_SPLAT_SIZE
    # Both project to the frame center; the red one sits nearer the camera.
    frame = preview_module._splat_frame(  # pyright: ignore[reportPrivateUsage]
        _splat(
            [(0.0, 0.0, -1.2), (0.0, 0.0, 0.0)],
            [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        )
    )
    pixels = _pixels(frame)
    assert frame.width == frame.height == size
    assert pixels.shape == (size, size, 3) and pixels.dtype == torch.uint8
    assert pixels[size // 2, size // 2].tolist() == [255, 0, 0]


def test_splat_frame_blanks_when_nothing_survives_the_cull() -> None:
    size = preview_module.PREVIEW_SPLAT_SIZE
    transparent = preview_module._splat_frame(  # pyright: ignore[reportPrivateUsage]
        _splat([(0.0, 0.0, 0.0)], [(1.0, 1.0, 1.0)], opacity=0.0)
    )
    behind = preview_module._splat_frame(  # pyright: ignore[reportPrivateUsage]
        # Depth is view z plus the camera distance; far enough behind the
        # camera fails the depth cull.
        _splat([(0.0, 0.0, -5.0)], [(1.0, 1.0, 1.0)])
    )
    for frame in (transparent, behind):
        pixels = _pixels(frame)
        assert pixels.shape == (size, size, 3)
        assert int(pixels.sum()) == 0


def test_triposplat_preview_decoder_renders_a_deterministic_frame() -> None:
    from test_triposplat_decoder import build_combined

    decoder = build_combined()
    decode = triposplat_preview_decoder(decoder)
    channels = decoder.config.latent_channels
    torch.manual_seed(5)
    state = torch.randn(2, 7, channels)  # batch entries beyond 0 are ignored
    size = preview_module.PREVIEW_SPLAT_SIZE
    frame = _still(decode(state))
    pixels = _pixels(frame)
    assert frame.width == frame.height == size
    assert pixels.shape == (size, size, 3) and pixels.dtype == torch.uint8
    # The fixed point-sampling seed keeps consecutive previews showing
    # denoising progress, not resampling noise.
    assert torch.equal(pixels, _pixels(_still(decode(state))))


def test_triposplat_preview_decoder_rejects_unexpected_shapes() -> None:
    from test_triposplat_decoder import build_combined

    decoder = build_combined()
    decode = triposplat_preview_decoder(decoder)
    channels = decoder.config.latent_channels
    with pytest.raises(ValueError, match="preview state"):
        decode(torch.zeros(7, channels))
    with pytest.raises(ValueError, match="preview state"):
        decode(torch.zeros(1, 7, channels + 1))
