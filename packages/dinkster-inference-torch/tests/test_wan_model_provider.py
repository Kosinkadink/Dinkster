"""Wan Animate model-pack provider contracts."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from dinkster_inference import (
    WAN21,
    WAN21_ANIMATE2_SETTINGS_KEY,
    WAN21_CAUSAL_INITIAL_LATENT_KEY,
    WAN21_CODEC,
    WAN21_I2V_14B,
    WAN21_T2V_14B,
    WAN22_DANCER_SETTINGS_KEY,
    WAN22_WANDANCER_14B,
    WAV2VEC2_CHINESE_BASE,
    WAV2VEC2_LARGE,
    BuiltinSamplerSelection,
    ComponentApplication,
    ComponentBinding,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    MultiStreamLatent,
    PercentRange,
    ReconstructionRecipe,
    RuntimeKnobs,
    Wan22DancerSettings,
    WeightSourceBinding,
    WeightSourceRef,
    bind_component_conditioning,
    split_component_conditioning,
)
from dinkster_inference_torch import Wan21Runtime, basic_conditioning_to_carrier
from dinkster_inference_torch.payloads import payload_binding_to_tensor

MODEL_PACKAGE_SOURCE = Path(__file__).parents[2] / "dinkster-model-wan" / "src"
PROVIDER_PATH = MODEL_PACKAGE_SOURCE / "dinkster_model_wan" / "provider.py"
PROVIDER_PACKAGE = ModuleType("_dinkster_model_wan")
cast("Any", PROVIDER_PACKAGE).__path__ = [str(PROVIDER_PATH.parent)]
sys.modules[PROVIDER_PACKAGE.__name__] = PROVIDER_PACKAGE
PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "_dinkster_model_wan.provider", PROVIDER_PATH
)
assert PROVIDER_SPEC is not None and PROVIDER_SPEC.loader is not None
PROVIDER_MODULE = importlib.util.module_from_spec(PROVIDER_SPEC)
sys.modules[PROVIDER_SPEC.name] = PROVIDER_MODULE
PROVIDER_SPEC.loader.exec_module(PROVIDER_MODULE)
execute_ar_video_i2v = PROVIDER_MODULE.execute_ar_video_i2v
execute_apply_wan21_uni3c = PROVIDER_MODULE.execute_apply_wan21_uni3c
execute_encode_wandancer_audio = PROVIDER_MODULE.execute_encode_wandancer_audio
execute_encode_wav2vec2_audio = PROVIDER_MODULE.execute_encode_wav2vec2_audio
execute_empty_ar_video_latent = PROVIDER_MODULE.execute_empty_ar_video_latent
execute_load_wav2vec2_audio_encoder = PROVIDER_MODULE.execute_load_wav2vec2_audio_encoder
execute_load_wan21_uni3c = PROVIDER_MODULE.execute_load_wan21_uni3c
execute_sampler_ar_video = PROVIDER_MODULE.execute_sampler_ar_video
execute_wan21_animate2_to_video = PROVIDER_MODULE.execute_wan21_animate2_to_video
execute_wan21_humo = PROVIDER_MODULE.execute_wan21_humo
execute_wan21_scail_to_video = PROVIDER_MODULE.execute_wan21_scail_to_video
execute_wan22_animate_to_video = PROVIDER_MODULE.execute_wan22_animate_to_video
execute_wan22_dancer_video = PROVIDER_MODULE.execute_wan22_dancer_video
execute_wan22_s2v = PROVIDER_MODULE.execute_wan22_s2v
execute_wan22_s2v_extend = PROVIDER_MODULE.execute_wan22_s2v_extend
execute_wan_infinite_talk_to_video = PROVIDER_MODULE.execute_wan_infinite_talk_to_video
execute_wandancer_pad_keyframe_list = PROVIDER_MODULE.execute_wandancer_pad_keyframe_list
execute_wandancer_pad_keyframes = PROVIDER_MODULE.execute_wandancer_pad_keyframes
WanDancerAudioOutput = PROVIDER_MODULE.WanDancerAudioOutput
WanHumoAudioOutput = PROVIDER_MODULE.WanHumoAudioOutput
WanInfiniteTalkAudioOutput = PROVIDER_MODULE.WanInfiniteTalkAudioOutput
WanS2VAudioOutput = PROVIDER_MODULE.WanS2VAudioOutput


def _recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef(
                    digest="blake3:" + "0" * 64,
                    name="animate.safetensors",
                    size=1,
                ),
            ),
        ),
        family_id="dinkster.wan21",
        component_identity=("family=dinkster.wan21", "variant=animate"),
        knobs=RuntimeKnobs(
            diffusion_dtype="bfloat16",
            text_dtype="float32",
            vae_dtype="bfloat16",
            fp8_matmul=False,
        ),
    )


class _Runtime(Wan21Runtime):
    def __init__(
        self,
        identity: str,
        *,
        with_vision: bool = True,
        model_variant: str = "animate",
    ) -> None:
        self._identity = identity
        self.family_value = WAN21
        self._fake_assembled = SimpleNamespace(
            diffusion=SimpleNamespace(
                config=SimpleNamespace(
                    model_variant=model_variant,
                    text_dim=8,
                    in_channels=20 if model_variant in ("scail", "scail2") else 36,
                    out_channels=16,
                )
            ),
            clip_vision=object() if with_vision else None,
        )
        self.vision_inputs: list[torch.Tensor] = []
        self.vision_crop_modes: list[bool] = []

    @property
    def runtime_identity(self) -> str:
        return self._identity

    @property
    def assembled(self) -> Any:
        return self._fake_assembled

    @property
    def family(self) -> Any:
        return self.family_value

    def encode_vision(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
        self.vision_inputs.append(image.clone())
        self.vision_crop_modes.append(crop)
        return torch.full((image.shape[0], 257, 1280), 7.0)


class _ModelHandle:
    def __init__(self, *, with_vision: bool = True, model_variant: str = "animate") -> None:
        self.recipe = _recipe()
        self._runtime = _Runtime(
            self.recipe.runtime_identity,
            with_vision=with_vision,
            model_variant=model_variant,
        )
        self.load_device = "cpu"
        self.stages: list[str] = []
        self.runtime_reads = 0
        self.active_checks = 0

    @property
    def runtime(self) -> _Runtime:
        self.runtime_reads += 1
        return self._runtime

    def require_active(self) -> None:
        self.active_checks += 1

    @contextmanager
    def stage(self, role: str) -> Generator[None]:
        self.stages.append(role)
        yield

    @property
    def fake_runtime(self) -> _Runtime:
        return self._runtime


class _CodecHandle:
    def __init__(self) -> None:
        self.descriptor = WAN21_CODEC
        self.resource_identity = "native:dinkster.wan21:" + "f" * 64
        self.load_device = "cpu"
        self.stages = 0
        self.active_checks = 0
        self.encoded: list[torch.Tensor] = []

    def require_active(self) -> None:
        self.active_checks += 1

    @contextmanager
    def stage(self) -> Generator[None]:
        self.stages += 1
        yield

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        self.encoded.append(content.clone())
        temporal = ((content.shape[2] - 1) // 4) + 1
        marker = float(len(self.encoded))
        return torch.full(
            (1, 16, temporal, content.shape[-2] // 8, content.shape[-1] // 8),
            marker,
        )

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Animate conditioning must not decode")


def _text(value: float) -> ConditioningCarrier:
    return basic_conditioning_to_carrier(Conditioning(torch.full((1, 2, 8), value), None))


@pytest.fixture(params=("dinkster.wan21", "example.renamed-video"))
def family_id(request: pytest.FixtureRequest) -> str:
    return cast("str", request.param)


@pytest.mark.parametrize(
    "variant, gate",
    (
        ("animate", "_animate_runtime"),
        ("animate2", "_animate2_runtime"),
        ("scail", "_scail_runtime"),
        ("scail2", "_scail_runtime"),
    ),
)
def test_feature_runtime_gates_use_config_not_family_label(
    variant: str, gate: str, family_id: str
) -> None:
    model = _ModelHandle(model_variant=variant)
    runtime = model.fake_runtime
    runtime.family_value = replace(WAN21, id=family_id)
    require_runtime = getattr(PROVIDER_MODULE, gate)
    assert require_runtime(model) is runtime
    runtime.assembled.diffusion.config.model_variant = "base"
    with pytest.raises(ValueError, match="profile"):
        require_runtime(model)
    with pytest.raises(TypeError, match="Wan21Runtime"):
        require_runtime(SimpleNamespace(runtime=object()))


@pytest.mark.parametrize("provider", ("humo", "infinite_talk", "dancer"))
@pytest.mark.parametrize("invalid", ("role", "source"))
def test_renamed_text_binding_retains_role_and_source_agreement(
    monkeypatch: pytest.MonkeyPatch, provider: str, invalid: str, family_id: str
) -> None:
    role = "clip_l" if invalid == "role" else "umt5xxl"
    binding = ComponentBinding(role, family_id, f"native:{family_id}:" + "a" * 64)
    negative_binding = (
        replace(binding, identity=f"native:{family_id}:" + "b" * 64)
        if invalid == "source"
        else binding
    )
    model = _ModelHandle(model_variant="wandancer")
    codec = _CodecHandle()

    def dancer_runtime(_model: object) -> tuple[_ModelHandle, _Runtime]:
        return model, model.runtime

    monkeypatch.setattr(PROVIDER_MODULE, "_dancer_runtime", dancer_runtime)
    args: dict[str, object] = {
        "positive": bind_component_conditioning(_text(1.0), binding),
        "negative": bind_component_conditioning(_text(-1.0), negative_binding),
        "vae": codec,
        "width": 16,
        "height": 16,
        "length": 1,
    }
    if provider == "infinite_talk":
        execute = execute_wan_infinite_talk_to_video
        args.update(
            mode="single_speaker",
            model=model,
            model_patch=object(),
            audio_encoder_output_1=object(),
            motion_frame_count=1,
            audio_scale=1.0,
            start_image=torch.zeros((1, 16, 16, 3)),
        )
    else:
        args["batch_size"] = 1
        execute = execute_wan21_humo if provider == "humo" else execute_wan22_dancer_video
        if provider == "dancer":
            args["model"] = model
    with pytest.raises(ValueError, match="UMT5-XXL component|share one component binding"):
        execute(**args)
    assert codec.stages == 0


def _channel(carrier: object, channel: ConditioningChannel) -> torch.Tensor:
    assert type(carrier) is ConditioningCarrier
    descriptor = next(
        descriptor
        for record in carrier.conditioning.records
        for record_channel, descriptor in record.channels
        if record_channel == channel
    )
    binding = next(
        item for item in carrier.bindings if item.reference_id == descriptor.reference.id
    )
    return payload_binding_to_tensor(binding)


def test_causal_ar_provider_builds_latent_sampler_and_i2v_seed() -> None:
    empty = cast(
        "Mapping[str, torch.Tensor]",
        execute_empty_ar_video_latent(width=32, height=16, length=9, batch_size=2)["latent"],
    )
    assert empty["samples"].shape == (2, 16, 3, 2, 4)
    assert not torch.count_nonzero(empty["samples"])

    selection = execute_sampler_ar_video(num_frame_per_block=3)["sampler"]
    assert selection == BuiltinSamplerSelection("dinkster.ar_video", (("num_frame_per_block", 3),))

    codec = _CodecHandle()
    model = _ModelHandle(model_variant="causal_ar")
    image = torch.stack((torch.zeros((12, 24, 3)), torch.ones((12, 24, 3))))
    result = execute_ar_video_i2v(
        model=model,
        vae=codec,
        start_image=image,
        width=32,
        height=16,
        length=9,
        batch_size=2,
    )
    latent = cast("Mapping[str, object]", result["latent"])
    assert result["model"] is model
    assert cast("torch.Tensor", latent["samples"]).shape == (2, 16, 3, 2, 4)
    initial = cast("torch.Tensor", latent[WAN21_CAUSAL_INITIAL_LATENT_KEY])
    assert initial.shape == (1, 16, 1, 2, 4)
    assert initial.device.type == "cpu" and initial.is_contiguous()
    assert torch.equal(initial, torch.ones_like(initial))
    assert codec.stages == 1
    assert len(codec.encoded) == 1
    assert codec.encoded[0].shape == (1, 3, 1, 16, 32)
    assert not torch.count_nonzero(codec.encoded[0])

    with pytest.raises(ValueError, match="multiples of 16"):
        execute_empty_ar_video_latent(width=31, height=16, length=9, batch_size=1)
    with pytest.raises(ValueError, match="num_frame_per_block must be <= 64"):
        execute_sampler_ar_video(num_frame_per_block=65)
    with pytest.raises(ValueError, match="CausalAR runtime"):
        execute_ar_video_i2v(
            model=_ModelHandle(model_variant="animate"),
            vae=codec,
            start_image=image,
            width=32,
            height=16,
            length=9,
            batch_size=2,
        )


def test_provider_builds_all_media_channels_and_uses_public_resource_stages() -> None:
    model = _ModelHandle()
    codec = _CodecHandle()
    reference = np.full((1, 12, 20, 3), 0.75, dtype=np.float32)
    continuation = np.stack(
        (
            np.full((12, 20, 3), 0.1, dtype=np.float32),
            np.full((12, 20, 3), 0.2, dtype=np.float32),
        )
    )
    pose = np.stack(
        tuple(np.full((12, 20, 3), index / 10, dtype=np.float32) for index in (1, 2, 3))
    )
    face = pose.copy()
    background = np.full((5, 12, 20, 3), 0.8, dtype=np.float32)
    mask = np.full((12, 20), 0.25, dtype=np.float32)

    output = execute_wan22_animate_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=5,
        batch_size=2,
        continue_motion_max_frames=5,
        video_frame_offset=1,
        reference_image=reference,
        face_video=face,
        pose_video=pose,
        background_video=background,
        character_mask=mask,
        continue_motion=continuation,
    )

    assert codec.active_checks == 1
    assert codec.stages == 1
    assert model.active_checks == 1
    assert model.runtime_reads == 2
    assert model.stages == ["vision"]
    assert len(codec.encoded) == 3
    assert [tuple(value.shape) for value in codec.encoded] == [
        (1, 3, 1, 16, 16),
        (1, 3, 5, 16, 16),
        (1, 3, 5, 16, 16),
    ]
    torch.testing.assert_close(codec.encoded[2][:, :, :1], torch.full((1, 3, 1, 16, 16), 0.1))
    torch.testing.assert_close(codec.encoded[2][:, :, 1:], torch.full((1, 3, 4, 16, 16), 0.8))
    assert len(model.fake_runtime.vision_inputs) == 1
    assert tuple(model.fake_runtime.vision_inputs[0].shape) == (1, 12, 20, 3)
    assert model.fake_runtime.vision_crop_modes == [True]

    positive = cast("ConditioningCarrier", output["positive"])
    negative = cast("ConditioningCarrier", output["negative"])
    assert [channel for channel, _ in positive.conditioning.records[0].channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
        ConditioningChannel.POSE_LATENT,
        ConditioningChannel.FACE_PIXELS,
    ]
    concat = _channel(positive, ConditioningChannel.CONCAT_LATENT)
    assert concat.shape == (1, 20, 3, 2, 2)
    expected_mask = torch.tensor(
        [[1.0, 1.0, 0.75], [1.0, 0.75, 0.0], [1.0, 0.75, 0.0], [1.0, 0.75, 0.0]]
    ).reshape(1, 4, 3, 1, 1)
    torch.testing.assert_close(concat[:, :4], expected_mask.expand(1, 4, 3, 2, 2))
    torch.testing.assert_close(concat[:, 4:, :1], torch.ones((1, 16, 1, 2, 2)))
    torch.testing.assert_close(concat[:, 4:, 1:], torch.full((1, 16, 2, 2, 2), 3.0))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.POSE_LATENT),
        torch.full((1, 16, 2, 2, 2), 2.0),
    )
    positive_face = _channel(positive, ConditioningChannel.FACE_PIXELS)
    negative_face = _channel(negative, ConditioningChannel.FACE_PIXELS)
    assert positive_face.shape == (1, 3, 3, 512, 512)
    torch.testing.assert_close(negative_face, torch.full_like(negative_face, -1.0))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.VISION_EMBEDDING),
        torch.full((1, 257, 1280), 7.0),
    )
    assert output["trim_latent"] == 1
    assert output["trim_image"] == 1
    assert output["video_frame_offset"] == 5
    latent = cast("dict[str, torch.Tensor]", output["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 2)


def test_scail_provider_uses_reference_stretch_vision_preprocessing() -> None:
    model = _ModelHandle(model_variant="scail")
    codec = _CodecHandle()
    reference = np.full((1, 12, 20, 3), 0.75, dtype=np.float32)

    execute_wan21_scail_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=32,
        height=32,
        length=1,
        batch_size=1,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        video_frame_offset=0,
        previous_frame_count=1,
        replacement_mode=False,
        reference_image=reference,
    )

    assert len(model.fake_runtime.vision_inputs) == 1
    assert tuple(model.fake_runtime.vision_inputs[0].shape) == (1, 12, 20, 3)
    assert model.fake_runtime.vision_crop_modes == [False]


def test_scail2_provider_bicubic_resizes_continuation_frames() -> None:
    model = _ModelHandle(with_vision=False, model_variant="scail2")
    codec = _CodecHandle()
    previous = np.linspace(0.0, 1.0, 12 * 20 * 3, dtype=np.float32).reshape(1, 12, 20, 3)

    execute_wan21_scail_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=32,
        height=32,
        length=1,
        batch_size=1,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        video_frame_offset=0,
        previous_frame_count=1,
        replacement_mode=False,
        reference_image=None,
        previous_frames=previous,
    )

    source = torch.from_numpy(previous).movedim(-1, 1)
    cropped = source[..., 4:16]
    expected = torch.nn.functional.interpolate(cropped, size=(32, 32), mode="bicubic")
    expected = expected.permute(1, 0, 2, 3).unsqueeze(0)
    area = torch.nn.functional.interpolate(cropped, size=(32, 32), mode="area")
    area = area.permute(1, 0, 2, 3).unsqueeze(0)
    assert len(codec.encoded) == 1
    assert torch.equal(codec.encoded[0], expected)
    assert not torch.equal(codec.encoded[0], area)


def test_provider_keeps_optional_channels_absent_without_media() -> None:
    model = _ModelHandle(with_vision=False)
    codec = _CodecHandle()

    output = execute_wan22_animate_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=1,
        batch_size=1,
        continue_motion_max_frames=1,
        video_frame_offset=0,
    )

    positive = cast("ConditioningCarrier", output["positive"])
    channels = [channel for channel, _ in positive.conditioning.records[0].channels]
    assert channels == [ConditioningChannel.TEXT, ConditioningChannel.CONCAT_LATENT]
    assert model.stages == []
    assert len(codec.encoded) == 2
    concat = _channel(positive, ConditioningChannel.CONCAT_LATENT)
    torch.testing.assert_close(concat[:, :4, :1], torch.ones((1, 4, 1, 2, 2)))
    torch.testing.assert_close(concat[:, :4, 1:], torch.zeros((1, 4, 1, 2, 2)))
    latent = cast("dict[str, torch.Tensor]", output["latent"])["samples"]
    assert latent.shape == (1, 16, 2, 2, 2)


def test_provider_normalizes_native_grayscale_and_rgba_images_to_rgb() -> None:
    model = _ModelHandle(with_vision=False)
    codec = _CodecHandle()
    face = np.full((1, 12, 20, 4), 0.25, dtype=np.float32)
    face[..., 3] = 1.0
    continuation = np.full((1, 12, 20, 1), 0.1, dtype=np.float32)
    background = np.full((5, 12, 20, 4), 0.8, dtype=np.float32)
    background[..., 3] = 0.0

    output = execute_wan22_animate_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=5,
        batch_size=1,
        continue_motion_max_frames=5,
        video_frame_offset=0,
        face_video=face,
        background_video=background,
        continue_motion=continuation,
    )

    assert codec.encoded[1].shape == (1, 3, 5, 16, 16)
    torch.testing.assert_close(codec.encoded[1][:, :, :1], torch.full((1, 3, 1, 16, 16), 0.1))
    torch.testing.assert_close(codec.encoded[1][:, :, 1:], torch.full((1, 3, 4, 16, 16), 0.8))
    face_pixels = _channel(output["positive"], ConditioningChannel.FACE_PIXELS)
    assert face_pixels.shape == (1, 3, 1, 512, 512)
    torch.testing.assert_close(face_pixels, torch.full_like(face_pixels, -0.5))


def test_animate2_provider_builds_pose_record_continuation_and_vision() -> None:
    model = _ModelHandle(model_variant="animate2")
    codec = _CodecHandle()
    reference = np.full((1, 12, 20, 4), 0.75, dtype=np.float32)
    reference[..., 3] = 0.0
    pose = np.stack(
        tuple(np.full((12, 20, 3), index / 10, dtype=np.float32) for index in range(1, 6))
    )
    continuation = np.stack(
        (
            np.full((12, 20, 1), 0.1, dtype=np.float32),
            np.full((12, 20, 1), 0.2, dtype=np.float32),
        )
    )

    output = execute_wan21_animate2_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        positive_pose=_text(2.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=5,
        batch_size=2,
        video_frame_offset=1,
        pose_strength=0.625,
        pose_start_percent=0.2,
        pose_end_percent=0.8,
        reference_image_strength=0.75,
        reference_image=reference,
        pose_video=pose,
        continue_motion=continuation,
    )

    assert codec.active_checks == 1
    assert codec.stages == 1
    assert model.active_checks == 1
    assert model.runtime_reads == 2
    assert model.stages == ["vision"]
    assert [tuple(value.shape) for value in codec.encoded] == [
        (1, 3, 1, 16, 16),
        (1, 3, 5, 16, 16),
        (1, 3, 5, 16, 16),
    ]
    torch.testing.assert_close(codec.encoded[1][:, :, :1], torch.full((1, 3, 1, 16, 16), 0.2))
    torch.testing.assert_close(codec.encoded[1][:, :, 1:], torch.full((1, 3, 4, 16, 16), 0.5))
    assert len(model.fake_runtime.vision_inputs) == 2

    positive = cast("ConditioningCarrier", output["positive"])
    negative = cast("ConditioningCarrier", output["negative"])
    assert len(positive.conditioning.records) == 2
    main, pose_record = positive.conditioning.records
    assert [channel for channel, _ in main.channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
    ]
    assert [channel for channel, _ in pose_record.channels] == [
        ConditioningChannel.POSE_TEXT,
        ConditioningChannel.POSE_VISION_EMBEDDING,
        ConditioningChannel.POSE_LATENT,
    ]
    assert type(pose_record.schedule) is PercentRange
    assert pose_record.schedule.start_percent == 0.2
    assert pose_record.schedule.end_percent == 0.8
    assert dict(pose_record.extension_metadata) == {
        WAN21_ANIMATE2_SETTINGS_KEY: {
            "pose_strength": 0.625,
            "reference_strength": 0.75,
        }
    }
    concat = _channel(positive, ConditioningChannel.CONCAT_LATENT)
    assert concat.shape == (1, 20, 3, 2, 2)
    torch.testing.assert_close(concat[:, :4, :2], torch.ones((1, 4, 2, 2, 2)))
    torch.testing.assert_close(concat[:, :4, 2:], torch.zeros((1, 4, 1, 2, 2)))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.POSE_TEXT),
        torch.full((1, 2, 8), 2.0),
    )
    torch.testing.assert_close(
        _channel(negative, ConditioningChannel.POSE_TEXT),
        torch.full((1, 2, 8), 2.0),
    )
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.POSE_LATENT),
        torch.full((1, 16, 2, 2, 2), 3.0),
    )
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.POSE_VISION_EMBEDDING),
        torch.full((1, 257, 1280), 7.0),
    )
    assert output["trim_latent"] == 1
    assert output["trim_image"] == 1
    assert output["video_frame_offset"] == 5
    latent = cast("dict[str, torch.Tensor]", output["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 2)


def test_animate2_provider_encodes_original_pose_frame_for_vision() -> None:
    model = _ModelHandle(model_variant="animate2")
    codec = _CodecHandle()
    pose = np.stack(
        tuple(np.full((12, 20, 3), index / 10, dtype=np.float32) for index in range(1, 6))
    )

    execute_wan21_animate2_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=1,
        batch_size=1,
        video_frame_offset=1,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        reference_image_strength=1.0,
        pose_video=pose,
    )

    assert codec.encoded[2].shape == (1, 3, 1, 16, 16)
    torch.testing.assert_close(codec.encoded[2], torch.full((1, 3, 1, 16, 16), 0.2))
    assert len(model.fake_runtime.vision_inputs) == 1
    assert model.fake_runtime.vision_inputs[0].shape == (1, 12, 20, 3)
    torch.testing.assert_close(model.fake_runtime.vision_inputs[0], torch.full((1, 12, 20, 3), 0.1))


def test_animate2_provider_keeps_pose_record_without_optional_media() -> None:
    model = _ModelHandle(with_vision=False, model_variant="animate2")
    codec = _CodecHandle()

    output = execute_wan21_animate2_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=1,
        batch_size=1,
        video_frame_offset=0,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        reference_image_strength=1.0,
    )

    positive = cast("ConditioningCarrier", output["positive"])
    assert [channel for channel, _ in positive.conditioning.records[0].channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
    ]
    assert [channel for channel, _ in positive.conditioning.records[1].channels] == [
        ConditioningChannel.POSE_TEXT,
    ]
    assert model.stages == []
    assert len(codec.encoded) == 2
    assert output["trim_image"] == 0
    assert output["video_frame_offset"] == 1


def test_animate2_provider_refuses_invalid_window_and_exhausted_pose_before_encoding() -> None:
    model = _ModelHandle(model_variant="animate2")
    codec = _CodecHandle()
    arguments: dict[str, object] = {
        "positive": _text(1.0),
        "negative": _text(0.0),
        "model": model,
        "vae": codec,
        "width": 16,
        "height": 16,
        "length": 1,
        "batch_size": 1,
        "video_frame_offset": 0,
        "pose_strength": 1.0,
        "pose_start_percent": 0.8,
        "pose_end_percent": 0.2,
        "reference_image_strength": 1.0,
    }
    with pytest.raises(ValueError, match="start <= end"):
        execute_wan21_animate2_to_video(**arguments)
    arguments["pose_start_percent"] = 0.0
    arguments["pose_end_percent"] = 1.0
    arguments["video_frame_offset"] = 1
    arguments["pose_video"] = np.zeros((1, 16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="nothing remains"):
        execute_wan21_animate2_to_video(**arguments)
    assert codec.stages == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("width", 18, "multiples of 16"),
        ("height", 18, "multiples of 16"),
        ("length", 2, "1 plus a multiple of 4"),
        ("video_frame_offset", -1, "integer >= 0"),
        ("width", 16400, "must be <= 16384"),
        ("height", 16400, "must be <= 16384"),
        ("length", 16385, "must be <= 16384"),
        ("batch_size", 4097, "must be <= 4096"),
        ("video_frame_offset", 16385, "must be <= 16384"),
        ("pose_strength", 10.01, "must be <= 10.0"),
        ("reference_image_strength", 10.01, "must be <= 10.0"),
        ("pose_start_percent", -0.01, "0 <= start <= end <= 1"),
        ("pose_end_percent", 1.01, "0 <= start <= end <= 1"),
    ),
)
def test_animate2_provider_refuses_widget_bounds_before_resource_use(
    field: str, value: int | float, message: str
) -> None:
    arguments: dict[str, object] = {
        "positive": _text(1.0),
        "negative": _text(0.0),
        "model": object(),
        "vae": object(),
        "width": 16,
        "height": 16,
        "length": 1,
        "batch_size": 1,
        "video_frame_offset": 0,
        "pose_strength": 1.0,
        "pose_start_percent": 0.0,
        "pose_end_percent": 1.0,
        "reference_image_strength": 1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        execute_wan21_animate2_to_video(**arguments)


def test_animate2_provider_refuses_shared_wrong_codec_and_wrong_profile_resources() -> None:
    model = _ModelHandle(model_variant="animate2")
    arguments: dict[str, object] = {
        "positive": _text(1.0),
        "negative": _text(0.0),
        "model": model,
        "vae": model,
        "width": 16,
        "height": 16,
        "length": 1,
        "batch_size": 1,
        "video_frame_offset": 0,
        "pose_strength": 1.0,
        "pose_start_percent": 0.0,
        "pose_end_percent": 1.0,
        "reference_image_strength": 1.0,
    }
    with pytest.raises(ValueError, match="distinct resources"):
        execute_wan21_animate2_to_video(**arguments)

    codec = _CodecHandle()
    codec.descriptor = replace(WAN21_CODEC, id="dinkster.other_vae")
    arguments["vae"] = codec
    with pytest.raises(ValueError, match="Wan 2.1 codec descriptor"):
        execute_wan21_animate2_to_video(**arguments)
    assert codec.stages == 0

    codec.descriptor = WAN21_CODEC
    arguments["model"] = _ModelHandle(model_variant="animate")
    with pytest.raises(ValueError, match="Wan 2.1 Animate2 profile"):
        execute_wan21_animate2_to_video(**arguments)
    assert codec.stages == 0


def test_provider_refuses_shared_or_wrong_codec_resources_before_staging() -> None:
    model = _ModelHandle()
    with pytest.raises(ValueError, match="distinct resources"):
        execute_wan22_animate_to_video(
            positive=_text(1.0),
            negative=_text(0.0),
            model=model,
            vae=model,
            width=16,
            height=16,
            length=1,
            batch_size=1,
            continue_motion_max_frames=1,
            video_frame_offset=0,
        )

    codec = _CodecHandle()
    codec.descriptor = replace(WAN21_CODEC, id="dinkster.other_vae")
    with pytest.raises(ValueError, match="Wan 2.1 codec descriptor"):
        execute_wan22_animate_to_video(
            positive=_text(1.0),
            negative=_text(0.0),
            model=model,
            vae=codec,
            width=16,
            height=16,
            length=1,
            batch_size=1,
            continue_motion_max_frames=1,
            video_frame_offset=0,
        )
    assert codec.stages == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("width", 18, "multiples of 16"),
        ("height", 18, "multiples of 16"),
        ("length", 2, "1 plus a multiple of 4"),
        ("continue_motion_max_frames", 2, "1 plus a multiple of 4"),
        ("video_frame_offset", -1, "integer >= 0"),
        ("width", 16400, "must be <= 16384"),
        ("height", 16400, "must be <= 16384"),
        ("length", 16385, "must be <= 16384"),
        ("continue_motion_max_frames", 16385, "must be <= 16384"),
        ("batch_size", 4097, "must be <= 4096"),
        ("video_frame_offset", 16385, "must be <= 16384"),
    ),
)
def test_provider_refuses_invalid_geometry_before_resource_use(
    field: str, value: int, message: str
) -> None:
    arguments: dict[str, object] = {
        "positive": _text(1.0),
        "negative": _text(0.0),
        "model": object(),
        "vae": object(),
        "width": 16,
        "height": 16,
        "length": 1,
        "batch_size": 1,
        "continue_motion_max_frames": 1,
        "video_frame_offset": 0,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        execute_wan22_animate_to_video(**arguments)


def test_wav2vec2_loader_plans_loads_and_publishes_owned_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Asset:
        digest = "blake3:" + "1" * 64
        size = 630_997_322

        def local_path(self) -> Path:
            return Path("wav2vec2.safetensors")

    source = SimpleNamespace(keys=lambda: ("wav2vec2.feature_extractor.conv_layers.0.conv.weight",))
    plan = object()
    module = torch.nn.Identity()
    loaded = SimpleNamespace(module=module)
    published = object()
    calls: list[tuple[object, ...]] = []

    def load_header(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return source

    def plan_component(candidate: object, *, path: Path) -> object:
        calls.append(("plan", candidate, path))
        return plan

    def identity(candidate: object, dtype: object) -> str:
        calls.append(("identity", candidate, dtype))
        return "native:dinkster.wav2vec2:" + "2" * 64

    def load_component(
        path: Path,
        *,
        asset: object,
        expected_identity: str,
        compute_dtype: torch.dtype,
    ) -> object:
        calls.append(("load", path, asset, expected_identity, compute_dtype))
        return loaded

    class Publisher:
        def publish(self, candidate: object, *, resource_identity: str) -> object:
            calls.append(("publish", candidate, resource_identity))
            return published

    monkeypatch.setattr(PROVIDER_MODULE, "AssetRef", Asset)
    monkeypatch.setattr(PROVIDER_MODULE, "load_safetensors_header", load_header)
    monkeypatch.setattr(PROVIDER_MODULE, "plan_wav2vec2_component", plan_component)
    monkeypatch.setattr(PROVIDER_MODULE, "wav2vec2_component_runtime_identity", identity)
    monkeypatch.setattr(PROVIDER_MODULE, "load_wav2vec2_component", load_component)
    monkeypatch.setattr(PROVIDER_MODULE, "component_publisher", lambda: Publisher())

    asset = Asset()
    result = execute_load_wav2vec2_audio_encoder(audio_encoder=asset)

    identity_value = "native:dinkster.wav2vec2:" + "2" * 64
    assert result == {"audio_encoder": published}
    assert calls == [
        ("header", Path("wav2vec2.safetensors"), Asset.digest, Asset.size),
        ("plan", source, Path("wav2vec2.safetensors")),
        ("identity", plan, PROVIDER_MODULE.FLOAT16),
        ("load", Path("wav2vec2.safetensors"), asset, identity_value, torch.float16),
        ("publish", module, identity_value),
    ]


def test_whisper_loader_plans_loads_and_publishes_float32_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Asset:
        digest = "blake3:" + "4" * 64
        size = 3_087_130_976

        def local_path(self) -> Path:
            return Path("whisper_large_v3_fp16.safetensors")

    source = SimpleNamespace(keys=lambda: ("model.encoder.conv1.weight",))
    plan = object()
    module = torch.nn.Identity()
    loaded = SimpleNamespace(module=module)
    published = object()
    calls: list[tuple[object, ...]] = []

    def load_header(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return source

    def plan_component(candidate: object, *, path: Path) -> object:
        calls.append(("plan", candidate, path))
        return plan

    def identity(candidate: object, dtype: object) -> str:
        calls.append(("identity", candidate, dtype))
        return "native:dinkster.whisper-large-v3:" + "5" * 64

    def load_component(
        path: Path,
        *,
        asset: object,
        expected_identity: str,
        compute_dtype: torch.dtype,
    ) -> object:
        calls.append(("load", path, asset, expected_identity, compute_dtype))
        return loaded

    class Publisher:
        def publish(self, candidate: object, *, resource_identity: str) -> object:
            calls.append(("publish", candidate, resource_identity))
            return published

    monkeypatch.setattr(PROVIDER_MODULE, "AssetRef", Asset)
    monkeypatch.setattr(PROVIDER_MODULE, "load_safetensors_header", load_header)
    monkeypatch.setattr(PROVIDER_MODULE, "plan_whisper_large_v3_component", plan_component)
    monkeypatch.setattr(PROVIDER_MODULE, "whisper_large_v3_component_runtime_identity", identity)
    monkeypatch.setattr(PROVIDER_MODULE, "load_whisper_large_v3_component", load_component)
    monkeypatch.setattr(PROVIDER_MODULE, "component_publisher", lambda: Publisher())

    asset = Asset()
    result = execute_load_wav2vec2_audio_encoder(audio_encoder=asset)

    identity_value = "native:dinkster.whisper-large-v3:" + "5" * 64
    assert result == {"audio_encoder": published}
    assert calls == [
        ("header", Path("whisper_large_v3_fp16.safetensors"), Asset.digest, Asset.size),
        ("plan", source, Path("whisper_large_v3_fp16.safetensors")),
        ("identity", plan, PROVIDER_MODULE.FLOAT32),
        (
            "load",
            Path("whisper_large_v3_fp16.safetensors"),
            asset,
            identity_value,
            torch.float32,
        ),
        ("publish", module, identity_value),
    ]


@pytest.mark.parametrize(
    "keys",
    (
        (),
        (
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
            "model.encoder.conv1.weight",
        ),
    ),
)
def test_audio_loader_refuses_unknown_or_ambiguous_component_metadata(
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[str, ...],
) -> None:
    class Asset:
        digest = "blake3:" + "6" * 64
        size = 1

        def local_path(self) -> Path:
            return Path("audio.safetensors")

    def load_header(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(keys=lambda: keys)

    monkeypatch.setattr(PROVIDER_MODULE, "AssetRef", Asset)
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "load_safetensors_header",
        load_header,
    )

    with pytest.raises(ValueError, match="exact supported Wav2Vec2 or Whisper Large v3"):
        execute_load_wav2vec2_audio_encoder(audio_encoder=Asset())


def test_wav2vec2_encode_resamples_stages_and_returns_cpu_owned_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        config = WAV2VEC2_LARGE

        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, waveform: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            self.inputs.append(waveform)
            layer = torch.arange(3 * 1024, dtype=torch.float16).view(1, 3, 1024)
            return layer, (layer,) * 25

    model = Model()

    class Handle:
        component = model
        load_device = torch.device("cpu")
        resource_identity = "native:dinkster.wav2vec2:" + "3" * 64

        def __init__(self) -> None:
            self.stages = 0

        @contextmanager
        def stage(self) -> Generator[None]:
            self.stages += 1
            yield

    handle = Handle()
    resamples: list[tuple[tuple[int, ...], int, int]] = []

    def resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        resamples.append((tuple(waveform.shape), source_rate, target_rate))
        return torch.linspace(-1.0, 1.0, 6_400).reshape(1, 1, -1)

    def require_handle(*_args: object) -> Handle:
        return handle

    monkeypatch.setattr(PROVIDER_MODULE, "require_inference_component_handle", require_handle)
    monkeypatch.setattr(PROVIDER_MODULE, "Wav2Vec2Model", Model)
    monkeypatch.setattr(PROVIDER_MODULE, "ltx_audio_resample", resample)

    result = execute_encode_wav2vec2_audio(
        audio_encoder=object(),
        audio={"waveform": np.zeros((1, 1, 3_200), dtype=np.float32), "sample_rate": 8_000},
    )

    output = result["audio_encoder_output"]
    assert type(output) is WanS2VAudioOutput
    assert output.audio_samples == 6_400
    assert len(output.layers) == 25
    assert all(layer.device.type == "cpu" and layer.is_contiguous() for layer in output.layers)
    assert handle.stages == 1
    assert resamples == [((1, 1, 3_200), 8_000, 16_000)]
    assert len(model.inputs) == 1
    assert model.inputs[0].shape == (1, 1, 6_400)
    assert model.inputs[0].dtype is torch.float16


def test_wav2vec2_chinese_base_encode_returns_infinite_talk_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        config = WAV2VEC2_CHINESE_BASE

        def __call__(self, waveform: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            layer = torch.arange(4 * 768, dtype=torch.float16).view(1, 4, 768)
            return layer, (layer,) * 13

    class Handle:
        component = Model()
        load_device = torch.device("cpu")
        resource_identity = "native:dinkster.wav2vec2:" + "4" * 64

        @contextmanager
        def stage(self) -> Generator[None]:
            yield

    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_component_handle",
        lambda *_args: Handle(),  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(PROVIDER_MODULE, "Wav2Vec2Model", Model)

    result = execute_encode_wav2vec2_audio(
        audio_encoder=object(),
        audio={"waveform": torch.zeros((1, 1, 8_000)), "sample_rate": 16_000},
    )

    output = result["audio_encoder_output"]
    assert type(output) is WanInfiniteTalkAudioOutput
    assert output.audio_samples == 8_000
    assert len(output.layers) == 13
    assert all(layer.shape == (1, 4, 768) and layer.device.type == "cpu" for layer in output.layers)


def test_whisper_encode_stages_and_returns_humo_cpu_owned_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, waveform: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            self.inputs.append(waveform)
            layer = torch.arange(4 * 1280, dtype=torch.float32).view(1, 4, 1280)
            return layer, (layer,) * 33

    model = Model()

    class Handle:
        component = model
        load_device = torch.device("cpu")
        resource_identity = "native:dinkster.whisper-large-v3:" + "7" * 64

        def __init__(self) -> None:
            self.stages = 0

        @contextmanager
        def stage(self) -> Generator[None]:
            self.stages += 1
            yield

    handle = Handle()

    def require_handle(*_args: object) -> Handle:
        return handle

    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_component_handle",
        require_handle,
    )
    monkeypatch.setattr(PROVIDER_MODULE, "WhisperLargeV3Model", Model)

    result = execute_encode_wav2vec2_audio(
        audio_encoder=object(),
        audio={"waveform": torch.zeros((1, 2, 8_000)), "sample_rate": 16_000},
    )

    output = result["audio_encoder_output"]
    assert type(output) is WanHumoAudioOutput
    assert output.audio_samples == 8_000
    assert len(output.layers) == 33
    assert all(layer.device.type == "cpu" and layer.is_contiguous() for layer in output.layers)
    assert handle.stages == 1
    assert len(model.inputs) == 1
    assert model.inputs[0].shape == (1, 2, 8_000)
    assert model.inputs[0].dtype is torch.float32


def test_audio_encode_refuses_wrong_component_identity_before_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        resource_identity = "native:dinkster.clip:" + "8" * 64
        load_device = torch.device("cpu")
        component = torch.nn.Identity()

        @contextmanager
        def stage(self) -> Generator[None]:
            pytest.fail("wrong component must be refused before staging")
            yield

    def require_handle(*_args: object) -> Handle:
        return Handle()

    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_component_handle",
        require_handle,
    )

    with pytest.raises(TypeError, match="native Wav2Vec2 or Whisper Large v3"):
        execute_encode_wav2vec2_audio(
            audio_encoder=object(),
            audio={"waveform": torch.zeros((1, 1, 1_600)), "sample_rate": 16_000},
        )


def test_audio_encode_refuses_model_that_disagrees_with_component_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        resource_identity = "native:dinkster.whisper-large-v3:" + "9" * 64
        load_device = torch.device("cpu")
        component = torch.nn.Identity()

        @contextmanager
        def stage(self) -> Generator[None]:
            yield

    def require_handle(*_args: object) -> Handle:
        return Handle()

    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_component_handle",
        require_handle,
    )

    with pytest.raises(TypeError, match="does not contain native"):
        execute_encode_wav2vec2_audio(
            audio_encoder=object(),
            audio={"waveform": torch.zeros((1, 1, 1_600)), "sample_rate": 16_000},
        )


def test_uni3c_loader_plans_assembles_and_publishes_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Asset:
        digest = "blake3:" + "1" * 64
        size = 123

        def local_path(self) -> Path:
            return Path("uni3c.safetensors")

    source = object()
    component_plan = object()
    plan = SimpleNamespace(patch=component_plan)
    patch = object()
    published = object()
    calls: list[tuple[object, ...]] = []

    def load(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return source

    def plan_patch(value: object, *, asset_digest: str) -> object:
        calls.append(("plan", value, asset_digest))
        return plan

    def assemble(value: object, *, compute_dtype: torch.dtype) -> object:
        calls.append(("assemble", value, compute_dtype))
        return SimpleNamespace(patch=patch, resource_digest="2" * 64)

    class Publisher:
        def publish(self, module: object, *, resource_identity: str) -> object:
            calls.append(("publish", module, resource_identity))
            return published

    def identity(_value: object, model_digest: str) -> str:
        assert model_digest == "2" * 64
        return "id"

    monkeypatch.setattr(PROVIDER_MODULE, "load_safetensors_header", load)
    monkeypatch.setattr(PROVIDER_MODULE, "plan_wan21_uni3c", plan_patch)
    monkeypatch.setattr(PROVIDER_MODULE, "assemble_wan21_uni3c", assemble)
    monkeypatch.setattr(PROVIDER_MODULE, "_wan21_uni3c_resource_identity", identity)
    monkeypatch.setattr(PROVIDER_MODULE, "component_publisher", lambda: Publisher())

    assert execute_load_wan21_uni3c(model_patch=Asset()) == {"patch": published}
    assert calls == [
        ("header", Path("uni3c.safetensors"), Asset.digest, Asset.size),
        ("plan", source, Asset.digest),
        ("assemble", plan, torch.bfloat16),
        ("publish", patch, "id"),
    ]


def test_uni3c_apply_resizes_encodes_and_binds_target_geometry(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    class FakeWanModel:
        def __init__(self) -> None:
            self.config = WAN21_T2V_14B

    class TargetVAE:
        def process_in(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 3.0

    class FakeRuntime:
        family = replace(WAN21, id=family_id)

        def __init__(self) -> None:
            self.assembled = SimpleNamespace(diffusion=FakeWanModel(), vae=TargetVAE())

    class FakePatch:
        resource_digest = "a" * 64

    class ComponentHandle:
        resource_identity = "native:dinkster.wan21:" + "b" * 64
        load_device = torch.device("cpu")

        @property
        def component(self) -> object:
            return patch

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Generator[None]:
            yield

        @contextmanager
        def stage_with(self, _runtime_handle: object, _role: str) -> Generator[None]:
            yield

    class Codec:
        descriptor = WAN21_CODEC
        load_device = torch.device("cpu")

        def __init__(self, resource_identity: str) -> None:
            self.resource_identity = resource_identity
            self.owner = object()
            self.encoded: list[torch.Tensor] = []

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

        @contextmanager
        def stage(self) -> Generator[None]:
            yield

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            self.encoded.append(content.clone())
            return torch.full((1, 16, 3, 2, 3), 2.0)

    patch = FakePatch()
    handle = ComponentHandle()
    codec = Codec("native:dinkster.wan21:" + "c" * 64)
    captured: list[ComponentApplication] = []

    def append(_model: object, application: ComponentApplication) -> object:
        captured.append(application)
        return "applied"

    def execution(
        model: object,
        render_latent: torch.Tensor,
        strength: float,
        window: PercentRange,
        model_digest: str,
        render_digest: str,
    ) -> object:
        return SimpleNamespace(
            model=model,
            render_latent=render_latent,
            strength=strength,
            window=window,
            model_digest=model_digest,
            render_digest=render_digest,
        )

    def require_runtime(_model: object) -> FakeRuntime:
        return FakeRuntime()

    def require_component(_patch: object) -> ComponentHandle:
        return handle

    def require_codec(value: object, _name: str) -> Codec:
        assert isinstance(value, Codec)
        return value

    monkeypatch.setattr(PROVIDER_MODULE, "_wan21_uni3c_runtime", require_runtime)
    monkeypatch.setattr(PROVIDER_MODULE, "_wan21_uni3c_component", require_component)
    monkeypatch.setattr(PROVIDER_MODULE, "require_inference_codec_handle", require_codec)
    monkeypatch.setattr(PROVIDER_MODULE, "_append_application", append)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Runtime", FakeRuntime)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Model", FakeWanModel)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Uni3C", FakePatch)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Uni3CExecution", execution)

    render = torch.stack((torch.zeros((10, 30, 3)), torch.ones((10, 30, 3))))
    assert execute_apply_wan21_uni3c(
        model=object(),
        patch=handle,
        vae=codec,
        render_video=render,
        strength=-1.5,
        start_percent=0.25,
        end_percent=0.75,
    ) == {"model": "applied"}

    application = captured[0]
    other_codec = Codec("native:dinkster.wan21:" + "d" * 64)
    assert execute_apply_wan21_uni3c(
        model=object(),
        patch=handle,
        vae=other_codec,
        render_video=render,
        strength=-1.5,
        start_percent=0.25,
        end_percent=0.75,
    ) == {"model": "applied"}
    assert captured[1].application_identity != application.application_identity
    assert application.resident_dependencies == (codec,)
    assert captured[1].resident_dependencies == (other_codec,)
    render.fill_(0.5)
    assert application.family_id == "dinkster.wan21"
    assert application.role == "diffusion"
    runtime = FakeRuntime()
    latent = MultiStreamLatent.from_pairs((("video", torch.zeros((2, 16, 3, 2, 3))),))
    prepared = cast(
        "SimpleNamespace",
        application.materialize_application_kwargs(runtime, patch, latent)["uni3c"],
    )
    assert len(codec.encoded) == 1
    assert codec.encoded[0].shape == (1, 3, 9, 16, 24)
    torch.testing.assert_close(codec.encoded[0][:, :, :1], torch.zeros((1, 3, 1, 16, 24)))
    torch.testing.assert_close(codec.encoded[0][:, :, 1:], torch.ones((1, 3, 8, 16, 24)))
    assert prepared.model is patch
    assert prepared.render_latent.device.type == "cpu"
    torch.testing.assert_close(prepared.render_latent, torch.full((1, 16, 3, 2, 3), 5.0))
    assert prepared.strength == -1.5
    assert prepared.window == PercentRange(0.25, 0.75)


def test_s2v_provider_stages_and_encodes_each_pixel_stream_serially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LifecycleCodec(_CodecHandle):
        def __init__(self) -> None:
            super().__init__()
            self.active = False
            self.transferred = 0
            self.events: list[str] = []

        @contextmanager
        def stage(self) -> Generator[None]:
            assert not self.active
            self.active = True
            self.stages += 1
            self.events.append("stage-enter")
            try:
                yield
            finally:
                assert self.transferred == 0
                self.events.append("stage-exit")
                self.active = False

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            assert self.active
            assert self.transferred == 1
            self.events.append(f"encode-{content.shape[2]}")
            result = super().encode_content(content)
            self.transferred = 0
            return result

    codec = LifecycleCodec()
    original_codec_content = PROVIDER_MODULE._codec_content

    def tracked_codec_content(frames: torch.Tensor, device: object) -> torch.Tensor:
        assert codec.active
        assert codec.transferred == 0
        codec.transferred = 1
        codec.events.append(f"transfer-{frames.shape[0]}")
        return original_codec_content(frames, device)

    monkeypatch.setattr(PROVIDER_MODULE, "_codec_content", tracked_codec_content)
    layer = torch.arange(120, dtype=torch.float32).view(1, 120, 1).expand(1, 120, 1024)
    audio = WanS2VAudioOutput((layer,) * 25, 38_400)
    result = execute_wan22_s2v(
        positive=_text(1.0),
        negative=_text(-1.0),
        vae=codec,
        width=32,
        height=16,
        length=9,
        batch_size=2,
        audio_encoder_output=audio,
        ref_image=torch.zeros((2, 12, 20, 3)),
        ref_motion=torch.full((3, 12, 20, 3), 0.25),
        control_video=torch.ones((5, 12, 20, 3)),
    )

    assert codec.events == [
        "stage-enter",
        "transfer-1",
        "encode-1",
        "transfer-73",
        "encode-73",
        "transfer-5",
        "encode-5",
        "stage-exit",
    ]
    assert [tuple(value.shape) for value in codec.encoded] == [
        (1, 3, 1, 16, 32),
        (1, 3, 73, 16, 32),
        (1, 3, 5, 16, 32),
    ]
    positive = result["positive"]
    negative = result["negative"]
    assert _channel(positive, ConditioningChannel.AUDIO_EMBEDDING).shape == (1, 25, 1024, 12)
    assert not torch.count_nonzero(_channel(negative, ConditioningChannel.AUDIO_EMBEDDING))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.REFERENCE_LATENT),
        torch.ones((1, 16, 1, 2, 4)),
    )
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.REFERENCE_MOTION),
        torch.full((1, 16, 19, 2, 4), 2.0),
    )
    control = _channel(positive, ConditioningChannel.CONTROL_VIDEO)
    torch.testing.assert_close(control[:, :, :2], torch.full((1, 16, 2, 2, 4), 3.0))
    assert control.shape == (1, 16, 3, 2, 4)
    latent = cast("Mapping[str, torch.Tensor]", result["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 4)


def test_s2v_extend_reuses_prior_geometry_and_motion_without_staging_pixels() -> None:
    codec = _CodecHandle()
    previous = torch.arange(2 * 16 * 4 * 2 * 4, dtype=torch.float32).view(2, 16, 4, 2, 4)
    layer = torch.ones((1, 200, 1024), dtype=torch.float32)
    result = execute_wan22_s2v_extend(
        positive=_text(1.0),
        negative=_text(-1.0),
        vae=codec,
        length=9,
        video_latent={"samples": previous},
        audio_encoder_output=WanS2VAudioOutput((layer,) * 25, 64_000),
    )

    assert codec.stages == 0
    assert not codec.encoded
    assert _channel(result["positive"], ConditioningChannel.AUDIO_EMBEDDING).shape == (
        1,
        25,
        1024,
        12,
    )
    torch.testing.assert_close(
        _channel(result["positive"], ConditioningChannel.REFERENCE_MOTION), previous
    )
    control = _channel(result["positive"], ConditioningChannel.CONTROL_VIDEO)
    assert control.shape == (1, 16, 3, 2, 4)
    latent = cast("Mapping[str, torch.Tensor]", result["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 4)


def test_humo_provider_stages_reference_and_preserves_component_bound_audio_lanes(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    class LifecycleCodec(_CodecHandle):
        def __init__(self) -> None:
            super().__init__()
            self.active = False
            self.events: list[str] = []

        @contextmanager
        def stage(self) -> Generator[None]:
            assert not self.active
            self.active = True
            self.stages += 1
            self.events.append("stage-enter")
            try:
                yield
            finally:
                self.events.append("stage-exit")
                self.active = False

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            assert self.active
            self.events.append("encode")
            return super().encode_content(content)

    codec = LifecycleCodec()
    original_upscale = PROVIDER_MODULE._common_upscale
    original_codec_content = PROVIDER_MODULE._codec_content

    def tracked_upscale(samples: torch.Tensor, width: int, height: int, mode: str) -> torch.Tensor:
        assert codec.active
        codec.events.append("resize")
        return original_upscale(samples, width, height, mode)

    def tracked_codec_content(frames: torch.Tensor, device: object) -> torch.Tensor:
        assert codec.active
        codec.events.append("transfer")
        return original_codec_content(frames, device)

    monkeypatch.setattr(PROVIDER_MODULE, "_common_upscale", tracked_upscale)
    monkeypatch.setattr(PROVIDER_MODULE, "_codec_content", tracked_codec_content)
    layers = tuple(
        (torch.arange(20, dtype=torch.float32).view(1, 20, 1) + layer_index * 100.0).expand(
            1, 20, 1280
        )
        for layer_index in range(33)
    )
    binding = ComponentBinding(
        "umt5xxl",
        family_id,
        f"native:{family_id}:" + "a" * 64,
    )
    result = execute_wan21_humo(
        positive=bind_component_conditioning(_text(1.0), binding),
        negative=bind_component_conditioning(_text(-1.0), binding),
        vae=codec,
        width=32,
        height=16,
        length=9,
        batch_size=2,
        audio_encoder_output=WanHumoAudioOutput(layers, 6_400),
        ref_image=torch.full((2, 12, 20, 3), 0.25),
    )

    assert codec.events == ["stage-enter", "resize", "transfer", "encode", "stage-exit"]
    assert [tuple(value.shape) for value in codec.encoded] == [(1, 3, 1, 16, 32)]
    positive, positive_binding = split_component_conditioning(result["positive"])
    negative, negative_binding = split_component_conditioning(result["negative"])
    assert positive_binding == negative_binding == binding
    audio = _channel(positive, ConditioningChannel.AUDIO_EMBEDDING)
    assert audio.shape == (1, 3, 8, 5, 1280)
    expected_time = torch.linspace(0.0, 19.0, 10)
    assert torch.equal(audio[0, 0, :5], torch.zeros_like(audio[0, 0, :5]))
    torch.testing.assert_close(audio[0, 0, 5:, 0, 0], 350.0 + expected_time[:3])
    torch.testing.assert_close(audio[0, 1, 1:, 4, 0], 3200.0 + expected_time[:7])
    torch.testing.assert_close(audio[0, 2, :7, 2, 0], 1950.0 + expected_time[3:10])
    assert not torch.count_nonzero(audio[0, 2, 7])
    assert not torch.count_nonzero(_channel(negative, ConditioningChannel.AUDIO_EMBEDDING))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.REFERENCE_LATENT),
        torch.ones((1, 16, 1, 2, 4)),
    )
    assert not torch.count_nonzero(_channel(negative, ConditioningChannel.REFERENCE_LATENT))
    latent = cast("Mapping[str, torch.Tensor]", result["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 4)


def test_humo_provider_builds_97_frame_whisper_windows() -> None:
    layers = tuple(
        (torch.arange(20, dtype=torch.float32).view(1, 20, 1) + layer_index * 100.0).expand(
            1, 20, 1280
        )
        for layer_index in range(33)
    )

    result = execute_wan21_humo(
        positive=_text(1.0),
        negative=_text(-1.0),
        vae=_CodecHandle(),
        width=32,
        height=16,
        length=97,
        batch_size=1,
        audio_encoder_output=WanHumoAudioOutput(layers, 6_400),
    )

    audio = _channel(result["positive"], ConditioningChannel.AUDIO_EMBEDDING)
    assert audio.shape == (1, 25, 8, 5, 1280)
    expected_time = torch.linspace(0.0, 19.0, 10)
    assert torch.equal(audio[0, 0, :5], torch.zeros_like(audio[0, 0, :5]))
    torch.testing.assert_close(audio[0, 0, 5:, 0, 0], 350.0 + expected_time[:3])
    torch.testing.assert_close(audio[0, 1, 1:, 4, 0], 3200.0 + expected_time[:7])
    assert not torch.count_nonzero(audio[0, 4:])
    negative_audio = _channel(result["negative"], ConditioningChannel.AUDIO_EMBEDDING)
    assert not torch.count_nonzero(negative_audio)


def test_humo_provider_builds_exact_zero_audio_and_reference_geometry() -> None:
    codec = _CodecHandle()
    result = execute_wan21_humo(
        positive=_text(1.0),
        negative=_text(-1.0),
        vae=codec,
        width=32,
        height=16,
        length=9,
        batch_size=3,
    )

    assert codec.stages == 0
    assert not codec.encoded
    positive_audio = _channel(result["positive"], ConditioningChannel.AUDIO_EMBEDDING)
    positive_reference = _channel(result["positive"], ConditioningChannel.REFERENCE_LATENT)
    assert positive_audio.shape == (1, 3, 8, 5, 1280)
    assert positive_reference.shape == (1, 16, 1, 2, 4)
    assert not torch.count_nonzero(positive_audio)
    assert not torch.count_nonzero(positive_reference)


def test_dancer_provider_builds_masked_media_audio_and_default_conditioning(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    model = _ModelHandle(model_variant="wandancer")
    model.fake_runtime.family_value = replace(WAN21, id=family_id)
    model.fake_runtime.assembled.diffusion.config = WAN22_WANDANCER_14B
    codec = _CodecHandle()
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Runtime", _Runtime)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan22DancerModel", SimpleNamespace)
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_runtime_handle",
        lambda *_args: model,  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_codec_handle",
        lambda *_args: codec,  # pyright: ignore[reportUnknownLambdaType]
    )
    binding = ComponentBinding(
        "umt5xxl",
        family_id,
        f"native:{family_id}:" + "a" * 64,
    )
    start = torch.stack((torch.zeros((12, 20, 3)), torch.ones((12, 20, 3))))
    result = execute_wan22_dancer_video(
        model=model,
        vae=codec,
        positive=bind_component_conditioning(_text(1.0), binding),
        negative=bind_component_conditioning(_text(-1.0), binding),
        width=32,
        height=16,
        length=9,
        batch_size=2,
        start_image=start,
        mask=torch.full((1, 12, 20), 0.75),
        reference_image=torch.full((1, 10, 18, 3), 0.5),
        audio_encoder_output=WanDancerAudioOutput(
            torch.full((1, 5, 35), 9.0),
            Wan22DancerSettings(24.0, 0.6),
        ),
    )

    assert model.stages == ["vision"]
    assert [tuple(value.shape) for value in model.fake_runtime.vision_inputs] == [
        (1, 12, 20, 3),
        (1, 10, 18, 3),
    ]
    assert [tuple(value.shape) for value in codec.encoded] == [(1, 3, 9, 16, 32)]
    positive, positive_binding = split_component_conditioning(result["positive"])
    negative, negative_binding = split_component_conditioning(result["negative"])
    assert positive_binding == negative_binding == binding
    assert [channel for channel, _ in negative.conditioning.records[0].channels] == [
        channel for channel, _ in positive.conditioning.records[0].channels
    ]
    concat = _channel(positive, ConditioningChannel.CONCAT_LATENT)
    assert concat.shape == (1, 20, 3, 2, 4)
    torch.testing.assert_close(concat[:, :4], torch.full_like(concat[:, :4], 0.25))
    torch.testing.assert_close(concat[:, 4:], torch.ones_like(concat[:, 4:]))
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.VISION_EMBEDDING),
        torch.full((1, 257, 1280), 7.0),
    )
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.REFERENCE_VISION_EMBEDDING),
        torch.full((1, 257, 1280), 7.0),
    )
    torch.testing.assert_close(
        _channel(positive, ConditioningChannel.AUDIO_EMBEDDING),
        torch.full((1, 5, 35), 9.0),
    )
    metadata = dict(positive.conditioning.records[0].extension_metadata)
    assert dict(cast("Mapping[str, float]", metadata[WAN22_DANCER_SETTINGS_KEY])) == {
        "fps": 24.0,
        "audio_inject_scale": 0.6,
    }
    latent = cast("Mapping[str, torch.Tensor]", result["latent"])["samples"]
    assert latent.shape == (2, 16, 3, 2, 4)

    unmasked = execute_wan22_dancer_video(
        model=model,
        vae=codec,
        positive=_text(1.0),
        negative=_text(-1.0),
        width=32,
        height=16,
        length=9,
        batch_size=1,
        start_image=start[:1],
    )
    unmasked_positive = cast("ConditioningCarrier", unmasked["positive"])
    unmasked_concat = _channel(unmasked_positive, ConditioningChannel.CONCAT_LATENT)
    assert not torch.count_nonzero(unmasked_concat[:, :4, :1])
    torch.testing.assert_close(
        unmasked_concat[:, :4, 1:],
        torch.ones_like(unmasked_concat[:, :4, 1:]),
    )

    default = execute_wan22_dancer_video(
        model=model,
        vae=codec,
        positive=_text(1.0),
        negative=_text(-1.0),
        width=32,
        height=16,
        length=9,
        batch_size=3,
    )
    default_positive = cast("ConditioningCarrier", default["positive"])
    assert [channel for channel, _ in default_positive.conditioning.records[0].channels] == [
        ConditioningChannel.TEXT
    ]
    default_metadata = dict(default_positive.conditioning.records[0].extension_metadata)
    assert dict(cast("Mapping[str, float]", default_metadata[WAN22_DANCER_SETTINGS_KEY])) == {
        "fps": 30.0,
        "audio_inject_scale": 1.0,
    }
    assert model.stages == ["vision", "vision"]
    assert len(codec.encoded) == 2


def test_dancer_provider_refuses_mask_without_start_image_and_shared_resources() -> None:
    with pytest.raises(ValueError, match="mask requires start_image"):
        execute_wan22_dancer_video(
            model=object(),
            vae=object(),
            positive=_text(1.0),
            negative=_text(-1.0),
            width=32,
            height=16,
            length=9,
            batch_size=1,
            mask=torch.ones((1, 16, 32)),
        )
    shared = object()
    with pytest.raises(ValueError, match="model and vae must be distinct"):
        execute_wan22_dancer_video(
            model=shared,
            vae=shared,
            positive=_text(1.0),
            negative=_text(-1.0),
            width=32,
            height=16,
            length=9,
            batch_size=1,
        )


def test_dancer_audio_and_keyframe_providers_publish_owned_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[np.ndarray, int, np.ndarray, int, float]] = []

    def encode(
        original: np.ndarray,
        sample_rate: int,
        resampled: np.ndarray,
        video_frames: int,
        audio_scale: float,
    ) -> SimpleNamespace:
        captured.append((original.copy(), sample_rate, resampled.copy(), video_frames, audio_scale))
        return SimpleNamespace(
            audio_feature=np.full((1, 5, 35), 3.0, np.float32),
            fps=30.0 if video_frames == 31 else 24.0,
            audio_inject_scale=audio_scale,
        )

    monkeypatch.setattr(PROVIDER_MODULE, "encode_wandancer_audio_features", encode)
    waveform = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    audio = {"waveform": waveform, "sample_rate": 15_360}
    encoded = execute_encode_wandancer_audio(
        audio=audio,
        video_frames=9,
        audio_inject_scale=0.4,
    )
    output = encoded["audio_encoder_output"]
    assert type(output) is WanDancerAudioOutput
    torch.testing.assert_close(output.audio_feature, torch.full((1, 5, 35), 3.0))
    assert output.settings == Wan22DancerSettings(24.0, 0.4)
    assert encoded["fps_string"] == " \u5e27\u7387\u662f24.0000"
    assert len(captured) == 1
    original, sample_rate, resampled, video_frames, audio_scale = captured[0]
    assert sample_rate == 15_360 and video_frames == 9 and audio_scale == 0.4
    np.testing.assert_array_equal(original, waveform.numpy())
    np.testing.assert_array_equal(resampled, waveform.numpy())
    assert (
        execute_encode_wandancer_audio(
            audio=audio,
            video_frames=31,
            audio_inject_scale=0.4,
        )["fps_string"]
        == ", \u5e27\u7387\u662f30fps\u3002"
    )

    images = torch.linspace(0.0, 1.0, 4).reshape(4, 1, 1, 1)
    segment = execute_wandancer_pad_keyframes(
        images=images,
        segment_length=4,
        segment_index=1,
        audio={"waveform": waveform, "sample_rate": 30},
    )
    assert torch.count_nonzero(cast("torch.Tensor", segment["keyframes_mask"])) == 2
    segment_audio = cast("Mapping[str, object]", segment["audio_segment"])
    torch.testing.assert_close(cast("torch.Tensor", segment_audio["waveform"]), waveform[:, :, 4:8])
    listed = execute_wandancer_pad_keyframe_list(
        images=images,
        segment_length=4,
        num_segments=3,
        audio={"waveform": waveform, "sample_rate": 30},
    )
    assert len(cast("list[torch.Tensor]", listed["keyframes_sequence"])) == 3
    assert len(cast("list[Mapping[str, object]]", listed["audio_segment"])) == 3


def test_infinite_talk_provider_builds_start_and_continuation_applications(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    class FakeWanModel:
        config = WAN21_I2V_14B

    class FakeRuntime:
        family = replace(WAN21, id=family_id)

        def __init__(self) -> None:
            self.assembled = SimpleNamespace(diffusion=FakeWanModel(), clip_vision=object())
            self.vision_inputs: list[torch.Tensor] = []

        def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
            self.vision_inputs.append(image.clone())
            return torch.full((1, 257, 1280), 7.0)

    runtime = FakeRuntime()

    class ModelHandle:
        load_device = torch.device("cpu")

        def __init__(self) -> None:
            self.stages: list[str] = []

        @property
        def runtime(self) -> FakeRuntime:
            return runtime

        @contextmanager
        def stage(self, role: str) -> Generator[None]:
            self.stages.append(role)
            yield

    class FakePatch:
        resource_digest = "3" * 64

        def __init__(self) -> None:
            self.projected: list[tuple[tuple[torch.Tensor, ...], int, int]] = []

        def project_audio(
            self,
            streams: tuple[torch.Tensor, ...],
            audio_start: int,
            audio_end: int,
        ) -> torch.Tensor:
            self.projected.append(
                (tuple(stream.detach().clone() for stream in streams), audio_start, audio_end)
            )
            tokens = 64 if len(streams) == 2 else 32
            return torch.full((1, 3, tokens, 768), 5.0)

    patch = FakePatch()

    class PatchHandle:
        resource_identity = "native:dinkster.wan21:" + "4" * 64
        load_device = torch.device("cpu")
        component = patch

        def __init__(self) -> None:
            self.active = False

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Generator[None]:
            self.active = True
            try:
                yield
            finally:
                self.active = False

        @contextmanager
        def stage_with(self, _runtime_handle: object, _role: str) -> Generator[None]:
            yield

    class FakePatchExecution(SimpleNamespace):
        pass

    class FakeInfiniteExecution(SimpleNamespace):
        pass

    model_handle = ModelHandle()
    patch_handle = PatchHandle()
    codec = _CodecHandle()
    applications: list[ComponentApplication] = []
    real_interpolate = PROVIDER_MODULE.functional.interpolate

    def interpolate(input: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if input.ndim == 3:
            assert patch_handle.active
        return real_interpolate(input, *args, **kwargs)

    monkeypatch.setattr(PROVIDER_MODULE.functional, "interpolate", interpolate)

    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_runtime_handle",
        lambda *_args: model_handle,  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_component_handle",
        lambda *_args: patch_handle,  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_codec_handle",
        lambda *_args: codec,  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Runtime", FakeRuntime)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21Model", FakeWanModel)
    monkeypatch.setattr(PROVIDER_MODULE, "Wan21MultiTalk", FakePatch)
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "Wan21MultiTalkExecution",
        lambda model, audio, masks, strength, model_digest, audio_digest, masks_digest: (  # pyright: ignore[reportUnknownLambdaType]
            FakePatchExecution(
                model=model,
                audio_context=audio,
                target_masks=masks,
                strength=strength,
                model_digest=model_digest,
                audio_digest=audio_digest,
                target_masks_digest=masks_digest,
            )
        ),
    )
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "Wan21InfiniteTalkExecution",
        lambda patch_execution, motion, extend, motion_digest: FakeInfiniteExecution(  # pyright: ignore[reportUnknownLambdaType]
            patch=patch_execution,
            motion_latent=motion,
            extend=extend,
            motion_digest=motion_digest,
        ),
    )

    def append(_model: object, application: ComponentApplication) -> ComponentApplication:
        applications.append(application)
        return application

    monkeypatch.setattr(PROVIDER_MODULE, "_append_application", append)
    binding = ComponentBinding(
        "umt5xxl",
        family_id,
        f"native:{family_id}:" + "a" * 64,
    )
    layers_1 = tuple(torch.full((1, 20, 768), float(index)) for index in range(13))
    layers_2 = tuple(torch.full((1, 20, 768), float(index + 100)) for index in range(13))
    audio_1 = WanInfiniteTalkAudioOutput(layers_1, 6_400)
    audio_2 = WanInfiniteTalkAudioOutput(layers_2, 6_400)
    start = torch.full((1, 12, 20, 3), 0.25)
    previous = torch.stack(tuple(torch.full((12, 20, 3), index / 10.0) for index in range(9)))
    result = execute_wan_infinite_talk_to_video(
        mode="two_speakers",
        model=object(),
        model_patch=object(),
        positive=bind_component_conditioning(_text(1.0), binding),
        negative=bind_component_conditioning(_text(-1.0), binding),
        vae=object(),
        width=32,
        height=16,
        length=9,
        audio_encoder_output_1=audio_1,
        motion_frame_count=5,
        audio_scale=0.75,
        start_image=start,
        previous_frames=previous,
        audio_encoder_output_2=audio_2,
        mask_1=torch.ones((1, 16, 32)),
        mask_2=torch.zeros((1, 16, 32)),
    )

    assert result["model"] is applications[0]
    assert result["trim_image"] == 5
    assert model_handle.stages == ["vision"]
    assert codec.stages == 1
    assert [tuple(value.shape) for value in codec.encoded] == [
        (1, 3, 9, 16, 32),
        (1, 3, 5, 16, 32),
    ]
    concat = _channel(result["positive"], ConditioningChannel.CONCAT_LATENT)
    assert concat.shape == (1, 20, 3, 2, 4)
    assert torch.equal(concat[:, :4, :1], torch.ones_like(concat[:, :4, :1]))
    assert not torch.count_nonzero(concat[:, :4, 1:])
    assert torch.equal(concat[:, 4:], torch.ones_like(concat[:, 4:]))
    assert _channel(result["positive"], ConditioningChannel.VISION_EMBEDDING).shape == (
        1,
        257,
        1280,
    )
    _, positive_binding = split_component_conditioning(result["positive"])
    _, negative_binding = split_component_conditioning(result["negative"])
    assert positive_binding == negative_binding == binding
    streams, audio_start, audio_end = patch.projected[0]
    assert (audio_start, audio_end) == (4, 13)
    assert streams[0].shape == streams[1].shape == (20, 12, 768)
    assert torch.count_nonzero(streams[0][:10]) and not torch.count_nonzero(streams[0][10:])
    assert not torch.count_nonzero(streams[1][:10]) and torch.count_nonzero(streams[1][10:])
    latent = MultiStreamLatent.from_pairs(
        (("video", cast("Mapping[str, torch.Tensor]", result["latent"])["samples"]),)
    )
    prepared = cast(
        "FakeInfiniteExecution",
        applications[0].materialize_application_kwargs(runtime, patch, latent)["multitalk"],
    )
    assert prepared.extend is True
    assert torch.equal(prepared.motion_latent, torch.full((1, 16, 2, 2, 4), 2.0))
    assert prepared.patch.strength == 0.75
    assert prepared.patch.audio_context.shape == (1, 3, 64, 768)
    assert torch.equal(
        prepared.patch.target_masks,
        torch.tensor(((True, True), (False, False))),
    )

    start_only = execute_wan_infinite_talk_to_video(
        mode="single_speaker",
        model=object(),
        model_patch=object(),
        positive=_text(1.0),
        negative=_text(-1.0),
        vae=object(),
        width=32,
        height=16,
        length=9,
        audio_encoder_output_1=audio_1,
        motion_frame_count=5,
        audio_scale=1.0,
        start_image=start,
    )
    start_latent = MultiStreamLatent.from_pairs(
        (("video", cast("Mapping[str, torch.Tensor]", start_only["latent"])["samples"]),)
    )
    start_execution = cast(
        "FakeInfiniteExecution",
        applications[1].materialize_application_kwargs(runtime, patch, start_latent)["multitalk"],
    )
    assert start_execution.extend is False
    assert torch.equal(start_execution.motion_latent, torch.full((1, 16, 1, 2, 4), 3.0))
    assert start_execution.patch.target_masks is None
    assert patch.projected[1][1:] == (0, 9)


def test_infinite_talk_provider_requires_start_or_previous_frames() -> None:
    layers = tuple(torch.zeros((1, 4, 768)) for _ in range(13))
    with pytest.raises(ValueError, match="requires start_image or previous_frames"):
        execute_wan_infinite_talk_to_video(
            mode="single_speaker",
            model=object(),
            model_patch=object(),
            positive=_text(1.0),
            negative=_text(-1.0),
            vae=object(),
            width=32,
            height=16,
            length=9,
            audio_encoder_output_1=WanInfiniteTalkAudioOutput(layers, 1_600),
            motion_frame_count=5,
            audio_scale=1.0,
        )


def test_uni3c_apply_refuses_non_rgb_render_before_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def require_runtime(_model: object) -> object:
        return object()

    def require_component(_patch: object) -> object:
        return object()

    def require_codec(_value: object, _name: str) -> object:
        return SimpleNamespace(descriptor=WAN21_CODEC)

    monkeypatch.setattr(PROVIDER_MODULE, "_wan21_uni3c_runtime", require_runtime)
    monkeypatch.setattr(PROVIDER_MODULE, "_wan21_uni3c_component", require_component)
    monkeypatch.setattr(
        PROVIDER_MODULE,
        "require_inference_codec_handle",
        require_codec,
    )
    with pytest.raises(ValueError, match="must be RGB"):
        execute_apply_wan21_uni3c(
            model=object(),
            patch=object(),
            vae=object(),
            render_video=torch.zeros((1, 16, 16, 4)),
            strength=1.0,
            start_percent=0.0,
            end_percent=1.0,
        )
