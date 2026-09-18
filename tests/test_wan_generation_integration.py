from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch", reason="Wan generation integration requires torch")

from dinkster_compat_comfy import native_arm  # noqa: E402
from dinkster_inference import (  # noqa: E402
    FLOAT16,
    FLOAT32,
    WAN21_CODEC,
    Conditioning,
    ConditioningCarrier,
    MultiStreamLatent,
    PercentRange,
    ReconstructionRecipe,
    RuntimeKnobs,
    SamplingSegment,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    Wan21Animate2Settings,
    WeightSourceBinding,
    WeightSourceRef,
)
from dinkster_inference_torch import (  # noqa: E402
    Wan21PreparedConditioning,
    Wan21Runtime,
    basic_conditioning_to_carrier,
)
from dinkster_model_wan.provider import (  # noqa: E402
    execute_wan21_animate2_to_video,
    execute_wan21_scail_to_video,
    execute_wan22_animate_to_video,
)


def _recipe(variant: str = "animate") -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef("blake3:" + "0" * 64, "animate.safetensors", 1),
            ),
        ),
        family_id="dinkster.wan21",
        component_identity=("family=dinkster.wan21", f"variant={variant}"),
        knobs=RuntimeKnobs(
            diffusion_dtype=FLOAT16.name,
            text_dtype=FLOAT32.name,
            vae_dtype=FLOAT16.name,
            fp8_matmul=False,
        ),
    )


class _Runtime(Wan21Runtime):
    def __init__(self, identity: str, variant: str) -> None:
        self._identity = identity
        self._fake_assembled = SimpleNamespace(
            diffusion=SimpleNamespace(
                config=SimpleNamespace(
                    model_variant=variant,
                    text_dim=8,
                    in_channels=20 if variant in ("scail", "scail2") else 36,
                    out_channels=16,
                )
            ),
            clip_vision=None,
        )
        self.sample_calls: list[dict[str, object]] = []

    @property
    def runtime_identity(self) -> str:
        return self._identity

    @property
    def conditioning_identity(self) -> str:
        return (
            f"dinkster.wan.conditioning:test:{self._fake_assembled.diffusion.config.model_variant}"
        )

    @property
    def supports_denoise_mask(self) -> bool:
        return self._fake_assembled.diffusion.config.model_variant == "scail2"

    @property
    def assembled(self) -> Any:
        return self._fake_assembled

    @property
    def family(self) -> Any:
        return SimpleNamespace(id="dinkster.wan21")

    def sample_multistream(
        self, latent: MultiStreamLatent[Any], **kwargs: object
    ) -> MultiStreamLatent[Any]:
        assert type(kwargs["conditioning"]) is Wan21PreparedConditioning
        guidance = cast("Any", kwargs["cfg"])
        assert type(guidance.uncond) is Wan21PreparedConditioning
        self.sample_calls.append(dict(kwargs))
        return latent.replace("video", latent.by_role("video") + 1.0)


class _ModelHandle:
    def __init__(self, variant: str = "animate") -> None:
        self.recipe = _recipe(variant)
        self._runtime = _Runtime(self.recipe.runtime_identity, variant)
        self.load_device = torch.device("cpu")
        self.stages: list[str] = []
        self.active_checks = 0

    @property
    def runtime(self) -> _Runtime:
        return self._runtime

    def require_active(self) -> None:
        self.active_checks += 1

    @contextmanager
    def stage(self, role: str, **_kwargs: object):  # noqa: ANN201
        self.stages.append(role)
        yield


class _CodecHandle:
    descriptor = WAN21_CODEC
    resource_identity = "native:dinkster.wan21:" + "f" * 64
    load_device = torch.device("cpu")

    def __init__(self) -> None:
        self.active_checks = 0
        self.stages = 0
        self.encoded_content: list[Any] = []

    def require_active(self) -> None:
        self.active_checks += 1

    @contextmanager
    def stage(self):  # noqa: ANN201
        self.stages += 1
        yield

    def encode_content(self, content: Any) -> Any:
        self.encoded_content.append(content.clone())
        temporal = ((content.shape[2] - 1) // 4) + 1
        return torch.zeros(
            (content.shape[0], 16, temporal, content.shape[-2] // 8, content.shape[-1] // 8),
            dtype=torch.float32,
        )

    def decode_latent(self, _latent: Any) -> Any:
        raise AssertionError("Animate conditioning must not decode")


def _text(value: float):  # noqa: ANN202
    layout = TokenLayoutDescriptor(
        "dinkster.wan21",
        1,
        ("umt5",),
        (TokenSegmentDescriptor("umt5", "umt5", 0, 2),),
    )
    return basic_conditioning_to_carrier(
        Conditioning(torch.full((1, 2, 8), value), None),
        token_layout=layout,
    )


def test_wan_animate_provider_executes_through_universal_samplers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModelHandle()
    codec = _CodecHandle()
    pose = torch.zeros((5, 16, 16, 3), dtype=torch.float32)
    provider_output = execute_wan22_animate_to_video(
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
        pose_video=pose,
    )

    def native_handle(value: object, input_id: str) -> _ModelHandle:
        assert input_id == "model"
        assert value is model
        model.require_active()
        return model

    monkeypatch.setattr(native_arm, "_native_handle", native_handle)
    sampler_output = native_arm.GenerationKSampler.execute(
        model=model,
        seed=7,
        steps=2,
        cfg=5.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.simple",
        positive=provider_output["positive"],
        negative=provider_output["negative"],
        latent_image=provider_output["latent"],
        denoise=1.0,
    )

    original = cast("dict[str, Any]", provider_output["latent"])["samples"]
    sampled = cast("dict[str, Any]", sampler_output["latent"])["samples"]
    torch.testing.assert_close(sampled, original + 1.0)
    first = model.runtime.sample_calls[0]
    assert cast("Wan21PreparedConditioning", first["conditioning"]).pose_latents is not None
    assert cast("Any", first["cfg"]).uncond.pose_latents is not None
    assert first["sampler_id"] == "dinkster.euler"
    assert first["scheduler_id"] == "dinkster.simple"

    advanced_output = native_arm.GenerationKSamplerAdvanced.execute(
        model=model,
        add_noise="enable",
        noise_seed=9,
        steps=4,
        cfg=6.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.simple",
        positive=provider_output["positive"],
        negative=provider_output["negative"],
        latent_image=provider_output["latent"],
        start_at_step=1,
        end_at_step=3,
        return_with_leftover_noise="enable",
    )

    advanced = cast("dict[str, Any]", advanced_output["latent"])["samples"]
    torch.testing.assert_close(advanced, original + 1.0)
    segment = model.runtime.sample_calls[1]["segment"]
    assert segment == SamplingSegment(4, 1, 3, True, True)
    assert model.stages == ["diffusion", "diffusion"]


def test_wan_animate2_provider_executes_through_universal_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModelHandle("animate2")
    codec = _CodecHandle()
    pose = torch.zeros((5, 16, 16, 3), dtype=torch.float32)
    provider_output = execute_wan21_animate2_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=16,
        height=16,
        length=5,
        batch_size=1,
        video_frame_offset=0,
        pose_strength=0.75,
        pose_start_percent=0.25,
        pose_end_percent=0.75,
        reference_image_strength=0.5,
        pose_video=pose,
    )

    def native_handle(value: object, input_id: str) -> _ModelHandle:
        assert input_id == "model"
        assert value is model
        model.require_active()
        return model

    monkeypatch.setattr(native_arm, "_native_handle", native_handle)
    sampler_output = native_arm.GenerationKSampler.execute(
        model=model,
        seed=7,
        steps=2,
        cfg=5.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.simple",
        positive=provider_output["positive"],
        negative=provider_output["negative"],
        latent_image=provider_output["latent"],
        denoise=1.0,
    )

    original = cast("dict[str, Any]", provider_output["latent"])["samples"]
    sampled = cast("dict[str, Any]", sampler_output["latent"])["samples"]
    torch.testing.assert_close(sampled, original + 1.0)
    prepared = cast("Wan21PreparedConditioning", model.runtime.sample_calls[0]["conditioning"])
    assert prepared.pose_latents is not None
    assert prepared.pose_schedule == PercentRange(0.25, 0.75)
    assert prepared.animate2_settings == Wan21Animate2Settings(0.75, 0.5)


def test_wan_scail2_provider_executes_masks_and_history_through_universal_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModelHandle("scail2")
    codec = _CodecHandle()
    reference = torch.ones((2, 32, 32, 3), dtype=torch.float32)
    pose = torch.zeros((5, 32, 32, 3), dtype=torch.float32)
    pose_mask = torch.zeros_like(pose)
    pose_mask[..., 0] = 1.0
    reference_mask = torch.zeros((2, 32, 32, 3), dtype=torch.float32)
    reference_mask[0, :, :16, 0] = 1.0
    reference_mask[1, ..., 2] = 1.0
    previous = torch.full((1, 32, 32, 3), 0.25, dtype=torch.float32)
    provider_output = execute_wan21_scail_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=32,
        height=32,
        length=5,
        batch_size=1,
        pose_strength=0.75,
        pose_start_percent=0.25,
        pose_end_percent=0.75,
        video_frame_offset=1,
        previous_frame_count=1,
        replacement_mode=True,
        reference_image=reference,
        pose_video=pose,
        pose_video_mask=pose_mask,
        reference_image_mask=reference_mask,
        previous_frames=previous,
    )

    latent = cast("dict[str, Any]", provider_output["latent"])
    assert latent["samples"].shape == (1, 16, 2, 4, 4)
    assert torch.equal(
        latent["noise_mask"],
        torch.tensor([0.0, 1.0]).view(1, 1, 2, 1, 1).expand(1, 1, 2, 4, 4),
    )
    assert provider_output["video_frame_offset"] == 5
    assert torch.count_nonzero(codec.encoded_content[0][..., 16:]) == 0
    assert torch.count_nonzero(codec.encoded_content[0][..., :16]) > 0

    def native_handle(value: object, input_id: str) -> _ModelHandle:
        assert input_id == "model"
        assert value is model
        model.require_active()
        return model

    monkeypatch.setattr(native_arm, "_native_handle", native_handle)
    sampler_output = native_arm.GenerationKSampler.execute(
        model=model,
        seed=7,
        steps=2,
        cfg=5.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.simple",
        positive=provider_output["positive"],
        negative=provider_output["negative"],
        latent_image=provider_output["latent"],
        denoise=1.0,
    )

    sampled = cast("dict[str, Any]", sampler_output["latent"])["samples"]
    torch.testing.assert_close(sampled, latent["samples"] + 1.0)
    call = model.runtime.sample_calls[0]
    prepared = cast("Wan21PreparedConditioning", call["conditioning"])
    assert prepared.pose_latents is not None and prepared.pose_latents.shape == (1, 16, 2, 2, 2)
    assert prepared.pose_schedule == PercentRange(0.25, 0.75)
    assert prepared.scail_reference_latent is not None
    assert prepared.scail_reference_latent.shape == (1, 16, 2, 4, 4)
    assert prepared.scail_reference_mask is not None
    assert prepared.scail_reference_mask.shape == (1, 28, 4, 4, 4)
    assert torch.count_nonzero(prepared.scail_reference_mask[:, :, 2:]) == 0
    blue_channels = (3, 10, 17, 24)
    red_channels = (1, 8, 15, 22)
    non_blue_channels = tuple(index for index in range(28) if index not in blue_channels)
    non_red_channels = tuple(index for index in range(28) if index not in red_channels)
    assert torch.equal(
        prepared.scail_reference_mask[:, blue_channels, 0],
        torch.ones((1, 4, 4, 4)),
    )
    assert torch.count_nonzero(prepared.scail_reference_mask[:, non_blue_channels, 0]) == 0
    assert torch.equal(
        prepared.scail_reference_mask[:, red_channels, 1, :, :2],
        torch.ones((1, 4, 4, 2)),
    )
    assert torch.count_nonzero(prepared.scail_reference_mask[:, :, 1, :, 2:]) == 0
    assert prepared.scail_driving_mask is not None
    assert prepared.scail_driving_mask.shape == (1, 28, 2, 2, 2)
    assert torch.equal(
        prepared.scail_driving_mask[:, red_channels],
        torch.ones((1, 4, 2, 2, 2)),
    )
    assert torch.count_nonzero(prepared.scail_driving_mask[:, non_red_channels]) == 0
    assert prepared.scail_replacement is True
    assert torch.equal(cast("Any", call["denoise_mask"]), latent["noise_mask"])


def test_wan_scail_provider_rejects_scail2_only_inputs() -> None:
    with pytest.raises(ValueError, match="colored identity masks require"):
        execute_wan21_scail_to_video(
            positive=_text(1.0),
            negative=_text(0.0),
            model=_ModelHandle("scail"),
            vae=_CodecHandle(),
            width=32,
            height=32,
            length=5,
            batch_size=1,
            pose_strength=1.0,
            pose_start_percent=0.0,
            pose_end_percent=1.0,
            video_frame_offset=0,
            previous_frame_count=5,
            replacement_mode=False,
            reference_image=torch.zeros((1, 32, 32, 3)),
            reference_image_mask=torch.zeros((1, 32, 32, 3)),
        )


def test_wan_scail_provider_allows_pose_only_conditioning() -> None:
    model = _ModelHandle("scail")
    codec = _CodecHandle()

    output = execute_wan21_scail_to_video(
        positive=_text(1.0),
        negative=_text(0.0),
        model=model,
        vae=codec,
        width=32,
        height=32,
        length=5,
        batch_size=1,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        video_frame_offset=0,
        previous_frame_count=5,
        replacement_mode=False,
        reference_image=None,
        pose_video=torch.zeros((5, 32, 32, 3)),
    )

    prepared = model.runtime.prepare_conditioning(cast("ConditioningCarrier", output["positive"]))
    assert prepared.scail_reference_latent is None
    assert prepared.pose_latents is not None
    assert len(codec.encoded_content) == 1
