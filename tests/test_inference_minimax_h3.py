"""MiniMax H3 profile, paired-family, and conditioning contract tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT16,
    INT8,
    MINIMAX_H3,
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_FAMILY,
    MINIMAX_H3_SIGMAS,
    NATIVE_WIRED_FAMILY_IDS,
    DType,
    FlowSigmas,
    LatentDescriptor,
    MiniMaxH3AudioReference,
    MiniMaxH3ConditionerPlan,
    MiniMaxH3Config,
    MiniMaxH3DiTPayloadKind,
    MiniMaxH3DiTReferencePayload,
    MiniMaxH3FL2VARequest,
    MiniMaxH3ImageReference,
    MiniMaxH3Keyframe,
    MiniMaxH3KeyframeRole,
    MiniMaxH3PresentationKind,
    MiniMaxH3PresentationSegment,
    MiniMaxH3REF2VARequest,
    MiniMaxH3Sigmas,
    MiniMaxH3T2VARequest,
    MiniMaxH3Task,
    MiniMaxH3TokenTag,
    MiniMaxH3VideoReference,
    MultiStreamLatentDescriptor,
    Parameterization,
    PayloadDescriptor,
    PayloadReference,
    SamplingDescriptor,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    builtin_family_registry,
    detect_minimax_h3,
    normalize_minimax_h3_conditioning,
    probe_native,
)


class HeaderSource:
    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        *,
        packed_keys: frozenset[str] = frozenset(),
        packed_dtype: DType = FLOAT8_E4M3,
    ) -> None:
        self.shapes = dict(shapes)
        self.packed_keys = packed_keys
        self.packed_dtype = packed_dtype
        self.entries_read: list[str] = []

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        self.entries_read.append(key)
        dtype = self.packed_dtype if key in self.packed_keys else FLOAT16
        geometry = TensorGeometry(self.shapes[key], dtype)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("MiniMax H3 detection must not read metadata or payloads")


class GenericHeaderSource(HeaderSource):
    path = Path("minimax-h3.safetensors")

    def metadata(self) -> Mapping[str, str]:
        return {}


def h3_shapes(prefix: str = "", *, mlp_time_embedding: bool = False) -> dict[str, tuple[int, ...]]:
    shapes = {
        "video_patch_proj.weight": (5376, 96),
        "audio_patch_proj.weight": (5376, 32),
        "condition_proj.weight": (5376, 5120),
        "blocks.0.attn.qkv_proj.weight": (21504, 5376),
        "blocks.0.mlp.fc1.weight": (28672, 5376),
        "final_layer.video_out.weight": (96, 5376),
        "final_layer.audio_out.weight": (32, 5376),
        "rope.inv_freq": (16,),
    }
    if mlp_time_embedding:
        shapes["time_embedder.proj_in.weight"] = (5376, 256)
        shapes["time_embedder.proj_out.weight"] = (2688, 5376)
    else:
        shapes["adaln_t_table"] = (1000, 2688)
    for index in range(50):
        shapes[f"blocks.{index}.attn.q_norm.weight"] = (128,)
    return {prefix + key: shape for key, shape in shapes.items()}


def test_minimax_h3_config_is_exact_and_immutable() -> None:
    config = MINIMAX_H3_CONFIG
    assert config == MiniMaxH3Config()
    assert (
        config.video_latent_channels,
        config.audio_latent_channels,
        config.depth,
        config.hidden_width,
        config.attention_heads,
        config.attention_head_dim,
        config.ffn_width,
        config.text_width,
    ) == (24, 32, 50, 5376, 56, 128, 14336, 5120)
    assert config.patch == (1, 2, 2)
    assert (
        config.video_spatial_downscale,
        config.video_fps,
        config.audio_content_channels,
        config.audio_latent_rate_hz,
        config.batch_size,
    ) == (16, 24, 2, 40, 1)
    assert (config.video_schedule_shift, config.audio_schedule_shift) == (12.0, 3.0)
    assert config.conditioner_id == "Qwen3-VL-32B"
    assert config.conditioner_layer == 50
    assert config.video_codec_id == "MiniMaxH3VideoVAE"
    assert config.audio_codec_id == "MiniMaxH3AudioVAE"
    with pytest.raises(FrozenInstanceError):
        config.depth = 49  # type: ignore[misc]
    with pytest.raises(ValueError, match="exact staged profile"):
        MiniMaxH3Config(depth=49)


@pytest.mark.parametrize(
    "field",
    (
        "video_latent_channels",
        "audio_latent_channels",
        "depth",
        "hidden_width",
        "attention_heads",
        "attention_head_dim",
        "ffn_width",
        "text_width",
        "video_spatial_downscale",
        "video_fps",
        "audio_content_channels",
        "audio_latent_rate_hz",
        "batch_size",
        "conditioner_layer",
    ),
)
def test_minimax_h3_config_refuses_float_at_every_integer_field(field: str) -> None:
    with pytest.raises(ValueError, match="exact staged profile"):
        replace(MINIMAX_H3_CONFIG, **{field: float(getattr(MINIMAX_H3_CONFIG, field))})


@pytest.mark.parametrize("index", range(3))
def test_minimax_h3_config_refuses_float_at_every_patch_member(index: int) -> None:
    patch = list(MINIMAX_H3_CONFIG.patch)
    patch[index] = float(patch[index])  # type: ignore[assignment]
    with pytest.raises(ValueError, match="exact staged profile"):
        MiniMaxH3Config(patch=tuple(patch))  # type: ignore[arg-type]


@pytest.mark.parametrize("index", range(3))
def test_minimax_h3_config_refuses_bool_at_every_patch_member(index: int) -> None:
    patch = list(MINIMAX_H3_CONFIG.patch)
    patch[index] = True
    with pytest.raises(ValueError, match="exact staged profile"):
        MiniMaxH3Config(patch=tuple(patch))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"batch_size": True},
        {"video_fps": 24.0},
        {"audio_content_channels": 2.0},
        {"video_schedule_shift": 12},
        {"video_schedule_shift": True},
        {"audio_schedule_shift": 3},
        {"audio_schedule_shift": True},
    ),
)
def test_minimax_h3_config_refuses_equal_numeric_type_substitutions(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exact staged profile"):
        replace(MINIMAX_H3_CONFIG, **changes)


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_minimax_h3_exact_bare_and_prefixed_headers_are_deterministic(prefix: str) -> None:
    source = HeaderSource(h3_shapes(prefix))
    first = detect_minimax_h3(source)
    second = detect_minimax_h3(source)
    assert first == second
    assert first is not None
    assert first.config is MINIMAX_H3_CONFIG
    assert first.key_prefix == prefix
    assert first.fields == {
        "audio_latent_channels": 32,
        "attention_head_dim": 128,
        "attention_heads": 56,
        "depth": 50,
        "ffn_width": 14336,
        "hidden_width": 5376,
        "key_prefix": prefix,
        "patch": "1x2x2",
        "text_width": 5120,
        "video_latent_channels": 24,
    }
    with pytest.raises(TypeError):
        first.fields["depth"] = 49  # type: ignore[index]
    expected_reads = {
        prefix + key
        for key in (
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.attn.qkv_proj.weight",
            "blocks.0.mlp.fc1.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
            "rope.inv_freq",
            "adaln_t_table",
        )
    }
    assert set(source.entries_read) == expected_reads


def test_minimax_h3_detects_official_mlp_time_embedding_header() -> None:
    source = HeaderSource(h3_shapes(mlp_time_embedding=True))
    evidence = detect_minimax_h3(source)
    assert evidence is not None
    assert evidence.config is MINIMAX_H3_CONFIG
    assert {
        "time_embedder.proj_in.weight",
        "time_embedder.proj_out.weight",
    }.issubset(evidence.matched_keys)
    assert "adaln_t_table" not in evidence.matched_keys


@pytest.mark.parametrize(
    "missing",
    (
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "rope.inv_freq",
        "adaln_t_table",
    ),
)
def test_minimax_h3_every_dereferenced_key_is_independently_guarded(missing: str) -> None:
    shapes = h3_shapes()
    del shapes[missing]
    assert detect_minimax_h3(HeaderSource(shapes)) is None


def test_minimax_h3_inconsistent_source_fails_closed_without_key_error() -> None:
    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    assert detect_minimax_h3(InconsistentSource(h3_shapes())) is None


@pytest.mark.parametrize(
    ("key", "shape"),
    (
        ("video_patch_proj.weight", (5376,)),
        ("video_patch_proj.weight", (5375, 96)),
        ("video_patch_proj.weight", (5376, 92)),
        ("audio_patch_proj.weight", (5376, 31)),
        ("condition_proj.weight", (5376, 4096)),
        ("blocks.0.attn.q_norm.weight", (127,)),
        ("blocks.0.attn.qkv_proj.weight", (21503, 5376)),
        ("blocks.0.mlp.fc1.weight", (28670, 5376)),
        ("final_layer.video_out.weight", (92, 5376)),
        ("final_layer.audio_out.weight", (31, 5376)),
        ("adaln_t_table", (1000,)),
    ),
)
def test_minimax_h3_wrong_rank_or_shape_fails_closed(key: str, shape: tuple[int, ...]) -> None:
    shapes = h3_shapes()
    shapes[key] = shape
    assert detect_minimax_h3(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    ("key", "old_shape"),
    (
        ("video_patch_proj.weight", (7168, 96)),
        ("audio_patch_proj.weight", (7168, 32)),
        ("condition_proj.weight", (7168, 5120)),
        ("blocks.0.attn.qkv_proj.weight", (21504, 7168)),
        ("blocks.0.mlp.fc1.weight", (37888, 7168)),
        ("final_layer.video_out.weight", (96, 7168)),
        ("final_layer.audio_out.weight", (32, 7168)),
    ),
)
def test_minimax_h3_rejects_every_old_outer_geometry(key: str, old_shape: tuple[int, ...]) -> None:
    shapes = h3_shapes()
    shapes[key] = old_shape
    assert detect_minimax_h3(HeaderSource(shapes)) is None


@pytest.mark.parametrize("index", (17, 49))
def test_minimax_h3_missing_any_block_index_fails_closed(index: int) -> None:
    shapes = h3_shapes()
    del shapes[f"blocks.{index}.attn.q_norm.weight"]
    assert detect_minimax_h3(HeaderSource(shapes)) is None


@pytest.mark.parametrize("index", (50, 51, 100))
def test_minimax_h3_extra_block_index_fails_closed(index: int) -> None:
    shapes = h3_shapes()
    shapes[f"blocks.{index}.mlp.fc1.weight"] = (28672, 5376)
    assert detect_minimax_h3(HeaderSource(shapes)) is None


@pytest.mark.parametrize("namespace", ("foo", "01", "-1", "1x", ""))
def test_minimax_h3_nonnumeric_or_ambiguous_block_namespace_fails_closed(
    namespace: str,
) -> None:
    shapes = h3_shapes()
    shapes[f"blocks.{namespace}.attn.q_norm.weight"] = (128,)
    assert detect_minimax_h3(HeaderSource(shapes)) is None


def test_minimax_h3_unbounded_numeric_block_namespace_fails_closed() -> None:
    shapes = h3_shapes()
    shapes[f"blocks.{'9' * 5000}.attn.q_norm.weight"] = (128,)
    assert detect_minimax_h3(HeaderSource(shapes)) is None


@pytest.mark.parametrize("marker", ("video_patch_proj.weight", "audio_patch_proj.weight"))
def test_minimax_h3_one_marker_only_and_near_family_fail_closed(marker: str) -> None:
    assert detect_minimax_h3(HeaderSource({marker: (5376, 96)})) is None


@pytest.mark.parametrize(
    "packed_key",
    (
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "adaln_t_table",
    ),
)
def test_minimax_h3_second_axis_packed_geometry_without_authority_fails_closed(
    packed_key: str,
) -> None:
    source = HeaderSource(h3_shapes(), packed_keys=frozenset({packed_key}))
    assert detect_minimax_h3(source) is None


def test_minimax_h3_int8_matrix_geometry_requires_artifact_authority() -> None:
    key = "blocks.0.attn.qkv_proj.weight"
    source = HeaderSource(
        h3_shapes(),
        packed_keys=frozenset({key}),
        packed_dtype=INT8,
    )

    assert detect_minimax_h3(source) is None
    assert detect_minimax_h3(source, allow_int8_geometry=True) is not None


def test_minimax_h3_registers_in_the_family_catalog_without_a_bundle_planner() -> None:
    ids = tuple(family.id for family in builtin_families())
    assert ids == (
        "dinkster.sd15",
        "dinkster.sdxl",
        "dinkster.sdxl_refiner",
        "dinkster.chroma",
        "dinkster.chroma_radiance",
        "dinkster.flux_dev",
        "dinkster.flux_schnell",
        "dinkster.flux2_dev",
        "dinkster.flux2_klein_9b",
        "dinkster.flux2_klein_4b",
        "dinkster.wan21",
        "dinkster.wan22",
        "dinkster.ltxv",
        "dinkster.ltxav",
        "dinkster.qwen_image",
        "dinkster.z_image",
        "dinkster.z_image_pixel_space",
        "dinkster.minimax_h3",
        "dinkster.minimax_music3",
        "dinkster.krea2",
        "dinkster.ideogram4",
        "dinkster.seedvr2",
        "dinkster.anima",
        "dinkster.lumina2",
        "dinkster.triposplat",
        "dinkster.trellis2",
    )
    assert MINIMAX_H3_CONFIG.family_id in ids
    registry = builtin_family_registry()
    assert MINIMAX_H3_CONFIG.family_id in registry.ids()
    assert MINIMAX_H3_CONFIG.family_id not in NATIVE_WIRED_FAMILY_IDS
    assert MINIMAX_H3_FAMILY.config is MINIMAX_H3_CONFIG
    assert not hasattr(MINIMAX_H3_FAMILY, "descriptor")
    assert MINIMAX_H3_FAMILY.sigmas is MINIMAX_H3_SIGMAS
    assert MINIMAX_H3_FAMILY.component_roles == (
        "fl2va-dit",
        "ref2va-dit",
        "qwen3vl-32b-conditioner",
        "video-vae",
        "audio-vae",
    )
    assert MINIMAX_H3_FAMILY.detect(HeaderSource(h3_shapes())) is not None
    assert not hasattr(MINIMAX_H3_CONFIG, "runtime")
    assert not hasattr(MINIMAX_H3_CONFIG, "conditioner")
    assert not hasattr(MINIMAX_H3_CONFIG, "video_codec")
    assert not hasattr(MINIMAX_H3_CONFIG, "audio_codec")


def test_minimax_h3_generic_detection_has_no_bundle_runtime_wiring() -> None:
    source = GenericHeaderSource(h3_shapes())
    registry = builtin_family_registry()

    detection = registry.detect(source)
    assert detection.best is not None
    assert detection.best.family_id == MINIMAX_H3_CONFIG.family_id
    assert detection.best.matched_keys
    assert registry.detect(GenericHeaderSource({})).best is None
    assert (
        registry.detect(GenericHeaderSource({"video_patch_proj.weight": (5376, 96)})).best is None
    )

    capability = probe_native(diffusion=source)
    assert capability.family_id == MINIMAX_H3_CONFIG.family_id
    assert capability.native is False
    reason = "; ".join(capability.reasons)
    assert "checkpoint components do not match an executable architecture" in reason
    assert f"detected labels=({MINIMAX_H3_CONFIG.family_id!r},)" in reason
    assert "has no native runtime wiring" not in reason


def test_minimax_h3_generic_family_has_multistream_flow_facts() -> None:
    assert MINIMAX_H3.supported_dtypes == frozenset({BFLOAT16})
    assert isinstance(MINIMAX_H3.sampling, SamplingDescriptor)
    assert MINIMAX_H3.sampling.parameterization is Parameterization.FLOW
    assert MINIMAX_H3.sampling.sigma_min == MINIMAX_H3_SIGMAS.sigma_min
    assert MINIMAX_H3.sampling.sigma_max == MINIMAX_H3_SIGMAS.sigma_max
    assert isinstance(MINIMAX_H3.latent, MultiStreamLatentDescriptor)
    assert MINIMAX_H3.latent.streams == (
        (
            "video",
            LatentDescriptor(
                channels=MINIMAX_H3_CONFIG.video_latent_channels,
                dimensions=3,
                spatial_downscale=MINIMAX_H3_CONFIG.video_spatial_downscale,
            ),
        ),
        (
            "audio",
            LatentDescriptor(channels=MINIMAX_H3_CONFIG.audio_latent_channels, dimensions=1),
        ),
    )


def test_single_stream_latent_refuses_multistream_families() -> None:
    families = {family.id: family for family in builtin_families()}
    sd15 = families["dinkster.sd15"]
    assert sd15.single_stream_latent() is sd15.latent
    with pytest.raises(ValueError, match="multistream latent"):
        MINIMAX_H3.single_stream_latent()


def test_multistream_latent_descriptor_validates_streams() -> None:
    video = LatentDescriptor(channels=24, dimensions=3)
    audio = LatentDescriptor(channels=32, dimensions=1)
    value = MultiStreamLatentDescriptor((("video", video), ("audio", audio)))
    assert value.streams == (("video", video), ("audio", audio))

    with pytest.raises(ValueError, match="at least two"):
        MultiStreamLatentDescriptor((("video", video),))
    with pytest.raises(ValueError, match="unique"):
        MultiStreamLatentDescriptor((("video", video), ("video", audio)))
    with pytest.raises(ValueError, match="nonempty"):
        MultiStreamLatentDescriptor((("", video), ("audio", audio)))
    with pytest.raises(TypeError, match="exact tuple"):
        MultiStreamLatentDescriptor([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact strings"):
        MultiStreamLatentDescriptor(((1, video), ("audio", audio)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact LatentDescriptor"):
        MultiStreamLatentDescriptor((("video", object()), ("audio", audio)))  # type: ignore[arg-type]


class ShapePayload:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_generic_multistream_latent_requires_ordered_unique_nonempty_roles() -> None:
    from dinkster_inference import LatentStream, MultiStreamLatent

    video = ShapePayload((1, 24, 2, 3, 5))
    audio = ShapePayload((1, 32, 2, 7))
    value = MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))
    assert value.roles == ("video", "audio")
    assert value.by_role("video") is video
    assert value.replace("audio", video).by_role("audio") is video
    assert value.map(lambda payload: payload.shape).roles == value.roles
    with pytest.raises(ValueError, match="at least one"):
        MultiStreamLatent(())
    with pytest.raises(ValueError, match="unique"):
        MultiStreamLatent((LatentStream("video", video), LatentStream("video", audio)))
    with pytest.raises(ValueError, match="nonempty"):
        LatentStream("", video)


@pytest.mark.parametrize(
    ("video_sigma", "audio_sigma", "state_factor", "velocity_factors"),
    (
        (Fraction(0), Fraction(0), Fraction(1, 4), (Fraction(-3), Fraction(1))),
        (
            Fraction(1, 4),
            Fraction(1, 13),
            Fraction(4, 13),
            (Fraction(-3), Fraction(16, 13)),
        ),
        (
            Fraction(1, 2),
            Fraction(1, 5),
            Fraction(2, 5),
            (Fraction(-3), Fraction(8, 5)),
        ),
        (
            Fraction(3, 4),
            Fraction(3, 7),
            Fraction(4, 7),
            (Fraction(-3), Fraction(16, 7)),
        ),
        (Fraction(1), Fraction(1), Fraction(1), (Fraction(-3), Fraction(4))),
    ),
)
def test_minimax_h3_av_sigma_exact_rational_matrix(
    video_sigma: Fraction,
    audio_sigma: Fraction,
    state_factor: Fraction,
    velocity_factors: tuple[Fraction, Fraction],
) -> None:
    sigmas = MINIMAX_H3_SIGMAS
    sigma = float(video_sigma)
    assert sigmas.audio_scale == 4.0
    assert sigmas.audio_sigma(sigma) == float(audio_sigma)
    assert sigmas.audio_state_factor(sigma) == float(state_factor)
    assert sigmas.audio_velocity_factors(sigma) == tuple(map(float, velocity_factors))


def test_minimax_h3_av_sigmas_delegate_the_video_space() -> None:
    sigmas = MINIMAX_H3_SIGMAS
    assert sigmas.video == FlowSigmas(shift=12.0)
    assert sigmas.audio_shift == 3.0
    assert sigmas.sigma_min == sigmas.video.sigma_min
    assert sigmas.sigma_max == sigmas.video.sigma_max
    assert sigmas.table == sigmas.video.table
    assert sigmas.sigma(500.0) == sigmas.video.sigma(500.0)
    assert sigmas.timestep(0.5) == sigmas.video.timestep(0.5)
    assert sigmas.percent_to_sigma(0.25) == sigmas.video.percent_to_sigma(0.25)


@pytest.mark.parametrize("sigma", (float("nan"), float("inf"), -0.01, 1.01))
def test_minimax_h3_av_sigmas_refuse_invalid_video_sigma(sigma: float) -> None:
    for operation in (
        MINIMAX_H3_SIGMAS.audio_sigma,
        MINIMAX_H3_SIGMAS.audio_state_factor,
        MINIMAX_H3_SIGMAS.audio_velocity_factors,
    ):
        with pytest.raises(ValueError, match="within"):
            operation(sigma)


@pytest.mark.parametrize(
    ("video_shift", "audio_shift"),
    (
        (0.0, 3.0),
        (-1.0, 3.0),
        (float("nan"), 3.0),
        (float("inf"), 3.0),
        (12.0, 0.0),
        (12.0, -1.0),
        (12.0, float("nan")),
        (12.0, float("inf")),
    ),
)
def test_minimax_h3_av_sigmas_refuse_invalid_shifts(video_shift: float, audio_shift: float) -> None:
    with pytest.raises(ValueError, match="shift"):
        MiniMaxH3Sigmas(FlowSigmas(shift=video_shift), audio_shift)


def h3_payload(name: str, shape: tuple[int, ...] = (1, 32, 32, 3)) -> PayloadDescriptor:
    return PayloadDescriptor(PayloadReference(name), shape, "float32", "worker:minimax-h3")


def test_minimax_h3_t2va_conditioning_is_raw_prompt_only() -> None:
    request = MiniMaxH3T2VARequest("  raw prompt <|not-a-template|>  ")
    plan = normalize_minimax_h3_conditioning(request)

    assert request.task is MiniMaxH3Task.T2VA
    assert plan.task is MiniMaxH3Task.T2VA
    assert plan.conditioner_id == "Qwen3-VL-32B"
    assert plan.conditioner_layer == 50
    assert plan.chat_templated is False
    assert [(segment.kind, segment.text, segment.token_tag) for segment in plan.presentation] == [
        (
            MiniMaxH3PresentationKind.TEXT,
            "  raw prompt <|not-a-template|>  ",
            MiniMaxH3TokenTag.TEXT,
        )
    ]
    assert plan.dit_keyframes == ()
    assert plan.dit_references == ()


def test_minimax_h3_fl2va_orders_first_then_last_for_qwen_and_dit() -> None:
    first = h3_payload("first")
    last = h3_payload("last")
    request = MiniMaxH3FL2VARequest(
        "move",
        (
            MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.LAST, last),
            MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, first),
        ),
    )
    plan = normalize_minimax_h3_conditioning(request)

    assert request.task is MiniMaxH3Task.FL2VA
    assert [segment.text for segment in plan.presentation if segment.text is not None] == [
        "<Picture 1>: ",
        "<|vision_start|>",
        "<|vision_end|>",
        "<Picture 2>: ",
        "<|vision_start|>",
        "<|vision_end|>",
        "move",
    ]
    contents = [
        segment.payloads
        for segment in plan.presentation
        if segment.kind is MiniMaxH3PresentationKind.IMAGE_CONTENT
    ]
    assert contents == [(first,), (last,)]
    assert tuple((item.role, item.payload) for item in plan.dit_keyframes) == (
        (MiniMaxH3KeyframeRole.FIRST, first),
        (MiniMaxH3KeyframeRole.LAST, last),
    )
    assert [segment.token_tag for segment in plan.presentation] == [
        MiniMaxH3TokenTag.TEXT,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.TEXT,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.VISION,
        MiniMaxH3TokenTag.TEXT,
    ]


@pytest.mark.parametrize("role", (MiniMaxH3KeyframeRole.FIRST, MiniMaxH3KeyframeRole.LAST))
def test_minimax_h3_fl2va_single_keyframe_is_picture_one(
    role: MiniMaxH3KeyframeRole,
) -> None:
    payload = h3_payload(role.value)
    request = MiniMaxH3FL2VARequest("prompt", (MiniMaxH3Keyframe(role, payload),))
    first = normalize_minimax_h3_conditioning(request)
    second = normalize_minimax_h3_conditioning(request)

    assert first == second
    assert hash(first) == hash(second)
    assert first.presentation[0].text == "<Picture 1>: "
    assert len(first.dit_keyframes) == 1
    assert first.dit_keyframes[0].role is role
    assert first.dit_keyframes[0].payload is payload


def test_minimax_h3_ref2va_preserves_order_ordinals_and_video_blocks() -> None:
    image = h3_payload("image")
    standalone_audio = h3_payload("standalone-audio", (1, 2, 16000))
    video_audio = h3_payload("video-audio", (1, 2, 8000))
    frame_1 = h3_payload("frame-1")
    frame_2 = h3_payload("frame-2")
    video_frames = (
        frame_1,
        *(h3_payload(f"video-filler-{index}") for index in range(1, 12)),
        frame_2,
    )
    trailing_image = h3_payload("image-2")
    request = MiniMaxH3REF2VARequest(
        "finish",
        (
            MiniMaxH3ImageReference(image),
            MiniMaxH3AudioReference(standalone_audio, 32_000),
            MiniMaxH3VideoReference(
                video_frames,
                (0, 12),
                (0.0, 0.5),
                MiniMaxH3AudioReference(video_audio, 32_000),
            ),
            MiniMaxH3ImageReference(trailing_image),
        ),
    )
    plan = normalize_minimax_h3_conditioning(request)

    assert request.task is MiniMaxH3Task.REF2VA
    assert [segment.text for segment in plan.presentation if segment.text is not None] == [
        "<Picture 1>: ",
        "<|vision_start|>",
        "<|vision_end|>",
        "<Audio 1>: ",
        "<Audio 2>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<Picture 2>: ",
        "<|vision_start|>",
        "<|vision_end|>",
        "finish",
    ]
    video_contents = [
        segment.payloads
        for segment in plan.presentation
        if segment.kind is MiniMaxH3PresentationKind.VIDEO_CONTENT
    ]
    assert video_contents == [(frame_1, frame_2)]
    assert tuple((item.kind, item.ordinal, item.payloads) for item in plan.dit_references) == (
        (MiniMaxH3DiTPayloadKind.IMAGE, 1, (image,)),
        (MiniMaxH3DiTPayloadKind.AUDIO, 1, (standalone_audio,)),
        (MiniMaxH3DiTPayloadKind.AUDIO, 2, (video_audio,)),
        (MiniMaxH3DiTPayloadKind.VIDEO, 1, video_frames),
        (MiniMaxH3DiTPayloadKind.IMAGE, 2, (trailing_image,)),
    )


def test_minimax_h3_conditioning_values_are_deeply_immutable() -> None:
    frame = h3_payload("frame")
    request = MiniMaxH3REF2VARequest("prompt", (MiniMaxH3VideoReference((frame,), (0,), (0.0,)),))
    plan = normalize_minimax_h3_conditioning(request)

    for value, field, replacement in (
        (request, "prompt", "changed"),
        (request.references[0], "frames", ()),
        (plan, "presentation", ()),
        (plan.presentation[0], "text", "changed"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_minimax_h3_conditioner_plan_refuses_direct_construction() -> None:
    with pytest.raises(TypeError, match="only by normalization"):
        MiniMaxH3ConditionerPlan(  # type: ignore[call-arg]
            MiniMaxH3Task.T2VA,
            (),
            (),
            (),
        )


def test_minimax_h3_reference_declarations_refuse_multiple_image_or_audio_payloads() -> None:
    first = h3_payload("first")
    second = h3_payload("second")
    for kind in (MiniMaxH3DiTPayloadKind.IMAGE, MiniMaxH3DiTPayloadKind.AUDIO):
        with pytest.raises(ValueError, match="exactly one"):
            MiniMaxH3DiTReferencePayload(kind, 1, (first, second))


def test_minimax_h3_video_presentation_selects_from_full_rate_frames() -> None:
    frames = tuple(h3_payload(f"frame-{index}") for index in range(13))
    reference = MiniMaxH3VideoReference(frames, (0, 12), (0.0, 0.5))
    plan = normalize_minimax_h3_conditioning(MiniMaxH3REF2VARequest("prompt", (reference,)))

    video_segments = tuple(
        segment.payloads
        for segment in plan.presentation
        if segment.kind is MiniMaxH3PresentationKind.VIDEO_CONTENT
    )
    assert video_segments == ((frames[0], frames[12]),)
    assert plan.dit_references[0].payloads == frames


@pytest.mark.parametrize(
    "factory",
    (
        lambda: MiniMaxH3FL2VARequest("prompt", ()),
        lambda: MiniMaxH3FL2VARequest(
            "prompt",
            (
                MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, h3_payload("a")),
                MiniMaxH3Keyframe(MiniMaxH3KeyframeRole.FIRST, h3_payload("b")),
            ),
        ),
        lambda: MiniMaxH3REF2VARequest("prompt", ()),
        lambda: MiniMaxH3AudioReference(h3_payload("audio"), 0),
        lambda: MiniMaxH3AudioReference(h3_payload("audio"), True),  # type: ignore[arg-type]
        lambda: MiniMaxH3VideoReference((h3_payload("frame"),), (0,), ()),
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame-a"), h3_payload("frame-b")), (0, 1), (0.5, 0.5)
        ),
        lambda: MiniMaxH3VideoReference((h3_payload("frame"),), (0,), (float("nan"),)),
        lambda: MiniMaxH3VideoReference((h3_payload("frame"),), (0,), (-0.1,)),
        lambda: MiniMaxH3VideoReference((h3_payload("frame"),), (), ()),
        lambda: MiniMaxH3VideoReference((h3_payload("frame"),), (1,), (0.0,)),
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame-a"), h3_payload("frame-b")), (1, 0), (0.0, 0.5)
        ),
        lambda: MiniMaxH3ImageReference(h3_payload("zero", (1, 0, 3))),
    ),
)
def test_minimax_h3_conditioning_refuses_missing_duplicate_or_malformed_values(
    factory: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: MiniMaxH3FL2VARequest("prompt", []),  # type: ignore[arg-type]
        lambda: MiniMaxH3REF2VARequest("prompt", []),  # type: ignore[arg-type]
        lambda: MiniMaxH3VideoReference([], (0,), (0.0,)),  # type: ignore[arg-type]
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame"),),
            [0],  # type: ignore[arg-type]
            (0.0,),
        ),
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame"),),
            (True,),
            (0.0,),  # type: ignore[arg-type]
        ),
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame"),),
            (0,),
            [0.0],  # type: ignore[arg-type]
        ),
        lambda: MiniMaxH3ImageReference(object()),  # type: ignore[arg-type]
        lambda: MiniMaxH3VideoReference(
            (h3_payload("frame"),),
            (0,),
            (0.0,),
            h3_payload("audio"),  # type: ignore[arg-type]
        ),
        lambda: MiniMaxH3REF2VARequest("prompt", (object(),)),  # type: ignore[arg-type]
        lambda: normalize_minimax_h3_conditioning(object()),  # type: ignore[arg-type]
    ),
)
def test_minimax_h3_conditioning_refuses_mutable_or_foreign_values(factory: object) -> None:
    with pytest.raises(TypeError):
        factory()  # type: ignore[operator]


def test_minimax_h3_presentation_markers_refuse_foreign_equal_values() -> None:
    class StringSubclass(str):
        pass

    class EqualMarker:
        def __eq__(self, other: object) -> bool:
            return other == "<|vision_start|>"

    for marker in (StringSubclass("<|vision_start|>"), EqualMarker()):
        with pytest.raises(ValueError, match="exact marker"):
            MiniMaxH3PresentationSegment(
                MiniMaxH3PresentationKind.VISION_START,
                marker,  # type: ignore[arg-type]
            )


def test_minimax_h3_payloads_refuse_nested_foreign_subclasses() -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple[int, ...]):
        pass

    payloads = (
        PayloadDescriptor(
            PayloadReference(StringSubclass("reference")),
            (1,),
            "float32",
            "worker:minimax-h3",
        ),
        PayloadDescriptor(
            PayloadReference("reference"),
            TupleSubclass((1,)),
            "float32",
            "worker:minimax-h3",
        ),
        PayloadDescriptor(
            PayloadReference("reference"),
            (1,),
            StringSubclass("float32"),
            "worker:minimax-h3",
        ),
        PayloadDescriptor(
            PayloadReference("reference"),
            (1,),
            "float32",
            StringSubclass("worker:minimax-h3"),
        ),
    )
    for payload in payloads:
        with pytest.raises((TypeError, ValueError)):
            MiniMaxH3ImageReference(payload)
