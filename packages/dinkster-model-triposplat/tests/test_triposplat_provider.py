from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn.functional as functional
from dinkster_inference import (
    BFLOAT16,
    TRIPOSPLAT_CONFIG,
    CodecDescriptor,
    ComponentBinding,
    ConditioningCarrier,
    LatentDescriptor,
    MultiStreamLatent,
    split_component_conditioning,
)
from dinkster_inference_torch import materialize_triposplat_conditioning
from dinkster_model_triposplat import provider

_GRAY = 100.0 / 255.0


class _ComponentHandle:
    load_device = torch.device("cpu")

    def __init__(self, digest: str, component: object, events: list[str] | None = None) -> None:
        self.resource_identity = "native:dinkster.triposplat:" + digest * 64
        self.module = component
        self._events = [] if events is None else events
        self._staged = False

    @property
    def component(self) -> object:
        if not self._staged:
            raise RuntimeError("component read outside its lease")
        return self.module

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self):
        self._events.append("stage")
        self._staged = True
        try:
            yield
        finally:
            self._staged = False

    @contextmanager
    def stage_with(self, _runtime_handle: object, _role: str):
        yield


class _CodecHandle:
    load_device = torch.device("cpu")
    resource_identity = "codec"

    def __init__(self, channels: int, encoded: torch.Tensor) -> None:
        self.descriptor = CodecDescriptor(
            id="test.codec",
            display_name="Test codec",
            kind="image",
            latent=LatentDescriptor(channels=channels, dimensions=2, spatial_downscale=16),
            supported_dtypes=frozenset({BFLOAT16}),
        )
        self.encoded = encoded
        self.contents: list[torch.Tensor] = []

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self):
        yield

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raise AssertionError("decode_latent must not run")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        self.contents.append(content)
        return self.encoded


@pytest.mark.parametrize(
    ("execute", "input_name", "output_name", "role"),
    (
        (
            provider.execute_load_triposplat_vision_encoder,
            "vision_encoder",
            "vision",
            "dinov3-vision-conditioner",
        ),
        (provider.execute_load_triposplat_decoder, "decoder", "decoder", "gaussian-decoder"),
    ),
)
@pytest.mark.parametrize("explicit_publisher", (False, True))
def test_loaders_plan_load_and_publish(
    monkeypatch: pytest.MonkeyPatch,
    execute: Callable[..., dict[str, object]],
    input_name: str,
    output_name: str,
    role: str,
    explicit_publisher: bool,
) -> None:
    class Asset:
        digest = "blake3:" + "1" * 64
        size = 123

        def local_path(self) -> Path:
            return Path("component.safetensors")

    source = object()
    planned = object()
    loaded = SimpleNamespace(module=object())
    published = object()
    calls: list[tuple[object, ...]] = []

    def load_header(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return source

    def plan(value: object, *, role: str, path: Path) -> object:
        calls.append(("plan", value, role, path))
        return planned

    def identity(value: object, compute_dtype: object) -> str:
        calls.append(("identity", value, compute_dtype))
        return "identity"

    def load(
        path: Path,
        *,
        asset: object,
        expected_role: str,
        expected_identity: str,
        compute_dtype: torch.dtype,
    ) -> object:
        calls.append(("load", path, type(asset), expected_role, expected_identity, compute_dtype))
        return loaded

    class Publisher:
        def publish(self, module: object, *, resource_identity: str) -> object:
            calls.append(("publish", module, resource_identity))
            return published

    monkeypatch.setattr(provider, "AssetRef", Asset)
    monkeypatch.setattr(provider, "load_safetensors_header", load_header)
    monkeypatch.setattr(provider, "plan_triposplat_split_component", plan)
    monkeypatch.setattr(provider, "triposplat_component_runtime_identity", identity)
    monkeypatch.setattr(provider, "load_triposplat_component", load)
    publisher = Publisher()
    if explicit_publisher:
        monkeypatch.setattr(
            provider,
            "component_publisher",
            lambda: (_ for _ in ()).throw(AssertionError("context publisher must not be used")),
        )
    else:
        monkeypatch.setattr(provider, "component_publisher", lambda: publisher)

    kwargs: dict[str, object] = {input_name: Asset()}
    if explicit_publisher:
        kwargs["publisher"] = publisher
    result = execute(**kwargs)
    assert result == {output_name: published}
    assert calls == [
        ("header", Path("component.safetensors"), Asset.digest, 123),
        ("plan", source, role, Path("component.safetensors")),
        ("identity", planned, BFLOAT16),
        ("load", Path("component.safetensors"), Asset, role, "identity", torch.bfloat16),
        ("publish", loaded.module, "identity"),
    ]


def test_loaders_reject_unresolved_assets() -> None:
    with pytest.raises(TypeError, match="vision_encoder must be a resolved asset reference"):
        provider.execute_load_triposplat_vision_encoder(vision_encoder=object())
    with pytest.raises(TypeError, match="decoder must be a resolved asset reference"):
        provider.execute_load_triposplat_decoder(decoder=object())


def test_lanczos_resize_preserves_uniform_content() -> None:
    samples = torch.full((1, 4, 8, 8), _GRAY)
    resized = provider._lanczos_resize(samples, 4, 2)  # pyright: ignore[reportPrivateUsage]
    assert resized.shape == (1, 4, 2, 4)
    assert torch.allclose(resized, torch.full((1, 4, 2, 4), _GRAY), atol=1.0 / 255.0)


def test_preprocess_centers_subject_on_black_square() -> None:
    image = torch.full((1, 32, 48, 3), _GRAY)
    mask = torch.ones((1, 32, 48))
    result = provider.execute_triposplat_preprocess_image(
        image=image, mask=mask, erode_radius=0, size=256
    )
    prepared = cast("torch.Tensor", result["image"])
    assert prepared.shape == (1, 256, 256, 3)
    assert torch.all(prepared >= 0.0) and torch.all(prepared <= 1.0)
    center = prepared[0, 128, 128]
    assert torch.allclose(center, torch.full((3,), _GRAY), atol=2.0 / 255.0)
    assert torch.allclose(prepared[0, 0, 0], torch.zeros(3), atol=2.0 / 255.0)
    assert torch.allclose(prepared[0, 255, 255], torch.zeros(3), atol=2.0 / 255.0)


def test_preprocess_repeats_and_resizes_mask_and_rounds_size() -> None:
    image = torch.full((2, 16, 16, 3), _GRAY)
    mask = torch.ones((1, 8, 8))
    result = provider.execute_triposplat_preprocess_image(
        image=image, mask=mask, erode_radius=0, size=270
    )
    prepared = cast("torch.Tensor", result["image"])
    assert prepared.shape == (2, 256, 256, 3)


def test_preprocess_rejects_empty_mask() -> None:
    image = torch.full((1, 32, 32, 3), _GRAY)
    mask = torch.zeros((1, 32, 32))
    with pytest.raises(ValueError, match="mask is empty"):
        provider.execute_triposplat_preprocess_image(
            image=image, mask=mask, erode_radius=0, size=256
        )


def test_preprocess_erode_can_empty_a_small_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resize(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
        if samples.shape[-2:] == (height, width):
            return samples
        return functional.interpolate(samples, size=(height, width), mode="nearest")

    monkeypatch.setattr(provider, "_lanczos_resize", fake_resize)
    image = torch.zeros((1, 256, 256, 3))
    image[0, 124:132, 124:132] = 1.0
    mask = torch.zeros((1, 256, 256))
    mask[0, 124:132, 124:132] = 1.0

    with pytest.raises(ValueError, match="mask is empty"):
        provider.execute_triposplat_preprocess_image(
            image=image, mask=mask, erode_radius=4, size=256
        )

    result = provider.execute_triposplat_preprocess_image(
        image=image, mask=mask, erode_radius=1, size=256
    )
    prepared = cast("torch.Tensor", result["image"])
    assert prepared.shape == (1, 256, 256, 3)
    assert prepared[0, 128, 128].max() > 0.5


class _VisionModule:
    config = SimpleNamespace(image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))

    def __init__(
        self, sequence: torch.Tensor, parameter_dtype: torch.dtype = torch.float32
    ) -> None:
        self.sequence = sequence
        self.pixel_values: list[torch.Tensor] = []
        self._parameter = torch.zeros((1,), dtype=parameter_dtype)

    def parameters(self):
        return iter((self._parameter,))

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.pixel_values.append(pixel_values)
        return self.sequence


def test_conditioning_builds_component_bound_lanes_and_empty_latent() -> None:
    torch.manual_seed(7)
    sequence = torch.randn((1, 6, TRIPOSPLAT_CONFIG.cond_channels))
    module = _VisionModule(sequence, parameter_dtype=torch.bfloat16)
    vision = _ComponentHandle("1", module)
    encoded = torch.randn((1, TRIPOSPLAT_CONFIG.cond2_channels, 2, 2))
    codec = _CodecHandle(TRIPOSPLAT_CONFIG.cond2_channels, encoded)
    image = torch.full((1, 32, 32, 3), _GRAY)

    result = provider.execute_triposplat_conditioning(vision=vision, vae=codec, image=image)

    binding = ComponentBinding(
        "dinov3-vision-conditioner", TRIPOSPLAT_CONFIG.family_id, vision.resource_identity
    )
    positive = result["positive"]
    negative = result["negative"]
    assert type(positive) is ConditioningCarrier
    assert type(negative) is ConditioningCarrier
    positive_carrier, positive_binding = split_component_conditioning(positive)
    negative_carrier, negative_binding = split_component_conditioning(negative)
    assert positive_binding == binding
    assert negative_binding == binding

    expected_pixel = (torch.full((1, 3, 32, 32), _GRAY) - 0.5) / 0.5
    assert len(module.pixel_values) == 1
    assert module.pixel_values[0].dtype is torch.bfloat16
    assert torch.allclose(module.pixel_values[0], expected_pixel.to(torch.bfloat16))
    assert len(codec.contents) == 1
    assert torch.allclose(codec.contents[0], torch.full((1, 3, 32, 32), _GRAY))

    expected_features = functional.layer_norm(sequence, sequence.shape[-1:])
    materialized = materialize_triposplat_conditioning(positive_carrier, device="cpu")
    assert torch.allclose(materialized.features, expected_features)
    assert materialized.reference_latent is not None
    assert torch.allclose(materialized.reference_latent, encoded)
    uncond = materialize_triposplat_conditioning(negative_carrier, device="cpu")
    assert torch.equal(uncond.features, torch.zeros_like(expected_features))
    assert uncond.reference_latent is not None
    assert torch.equal(uncond.reference_latent, torch.zeros_like(encoded))

    latent = cast("dict[str, object]", result["latent"])
    streams = cast("MultiStreamLatent[torch.Tensor]", latent["samples"])
    assert type(streams) is MultiStreamLatent
    assert streams.roles == ("latent", "camera")
    assert streams.by_role("latent").shape == (
        1,
        TRIPOSPLAT_CONFIG.q_token_length,
        TRIPOSPLAT_CONFIG.latent_channels,
    )
    assert streams.by_role("camera").shape == (1, 1, TRIPOSPLAT_CONFIG.cam_channels)
    assert not streams.by_role("latent").any()
    assert not streams.by_role("camera").any()


def test_conditioning_rejects_foreign_vision_component() -> None:
    module = _VisionModule(torch.zeros((1, 6, TRIPOSPLAT_CONFIG.cond_channels)))
    vision = _ComponentHandle("1", module)
    vision.resource_identity = "native:dinkster.qwen_image:" + "1" * 64
    codec = _CodecHandle(TRIPOSPLAT_CONFIG.cond2_channels, torch.zeros((1, 128, 2, 2)))
    image = torch.full((1, 32, 32, 3), _GRAY)
    with pytest.raises(TypeError, match="vision must be a native TripoSplat"):
        provider.execute_triposplat_conditioning(vision=vision, vae=codec, image=image)


def test_conditioning_rejects_wrong_reference_channels() -> None:
    module = _VisionModule(torch.zeros((1, 6, TRIPOSPLAT_CONFIG.cond_channels)))
    vision = _ComponentHandle("1", module)
    codec = _CodecHandle(16, torch.zeros((1, 16, 2, 2)))
    image = torch.full((1, 32, 32, 3), _GRAY)
    with pytest.raises(ValueError, match="128-channel reference latents"):
        provider.execute_triposplat_conditioning(vision=vision, vae=codec, image=image)


class _DecoderModule:
    gaussians_per_point = 4

    def __init__(self, parameter_dtype: torch.dtype = torch.float32) -> None:
        self.calls: list[tuple[torch.Tensor, int, torch.Generator]] = []
        self._parameter = torch.zeros((1,), dtype=parameter_dtype)

    def parameters(self):
        return iter((self._parameter,))

    def decode(
        self, latent: torch.Tensor, *, num_gaussians: int, generator: torch.Generator
    ) -> tuple[SimpleNamespace, ...]:
        self.calls.append((latent, num_gaussians, generator))
        return tuple(
            SimpleNamespace(
                positions=torch.zeros((8, 3)),
                scales=torch.zeros((8, 3)),
                rotations=torch.zeros((8, 4)),
                opacities=torch.zeros((8, 1)),
                sh=torch.zeros((8, 1, 3)),
            )
            for _ in range(latent.shape[0])
        )


def test_decode_stacks_splat_tensors_and_seeds_the_generator() -> None:
    module = _DecoderModule(torch.float16)
    decoder = _ComponentHandle("2", module)
    latent_input = torch.randn((2, 16, 16), dtype=torch.float32)
    streams = MultiStreamLatent.from_pairs(
        (("latent", latent_input), ("camera", torch.zeros((2, 1, 5))))
    )
    result = provider.execute_triposplat_decode(
        samples={"samples": streams},
        decoder=decoder,
        num_gaussians=32770,
        seed=123,
    )
    splat = cast("dict[str, torch.Tensor]", result["splat"])
    assert set(splat) == {"positions", "scales", "rotations", "opacities", "sh"}
    assert splat["positions"].shape == (2, 8, 3)
    assert splat["scales"].shape == (2, 8, 3)
    assert splat["rotations"].shape == (2, 8, 4)
    assert splat["opacities"].shape == (2, 8, 1)
    assert splat["sh"].shape == (2, 8, 1, 3)
    assert len(module.calls) == 1
    latent, count, generator = module.calls[0]
    assert torch.equal(latent, latent_input)
    assert latent.dtype is torch.float32
    assert count == 32768
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 123


def test_decode_rejects_malformed_latents() -> None:
    module = _DecoderModule()
    decoder = _ComponentHandle("2", module)
    with pytest.raises(TypeError, match="latent mapping"):
        provider.execute_triposplat_decode(
            samples=object(), decoder=decoder, num_gaussians=32768, seed=0
        )
    with pytest.raises(TypeError, match="multi-stream"):
        provider.execute_triposplat_decode(
            samples={"samples": torch.zeros((1, 16, 16))},
            decoder=decoder,
            num_gaussians=32768,
            seed=0,
        )
    wrong_roles = MultiStreamLatent.from_pairs(
        (("video", torch.zeros((1, 16, 16))), ("camera", torch.zeros((1, 1, 5))))
    )
    with pytest.raises(ValueError, match="'latent', 'camera'"):
        provider.execute_triposplat_decode(
            samples={"samples": wrong_roles}, decoder=decoder, num_gaussians=32768, seed=0
        )


def test_decode_validates_count_and_seed_ranges() -> None:
    module = _DecoderModule()
    decoder = _ComponentHandle("2", module)
    streams = MultiStreamLatent.from_pairs(
        (("latent", torch.zeros((1, 16, 16))), ("camera", torch.zeros((1, 1, 5))))
    )
    with pytest.raises(ValueError, match="num_gaussians"):
        provider.execute_triposplat_decode(
            samples={"samples": streams}, decoder=decoder, num_gaussians=1, seed=0
        )
    with pytest.raises(ValueError, match="seed"):
        provider.execute_triposplat_decode(
            samples={"samples": streams}, decoder=decoder, num_gaussians=32768, seed=-1
        )
    with pytest.raises(ValueError, match="seed"):
        provider.execute_triposplat_decode(
            samples={"samples": streams},
            decoder=decoder,
            num_gaussians=32768,
            seed=1 << 64,
        )
