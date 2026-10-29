"""Native Whisper Large v3 model, residency, and strict-loader tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    FLOAT32,
    WHISPER_LARGE_V3,
    WhisperLargeV3ComponentAssemblyError,
    whisper_large_v3_layout,
)
from dinkster_inference_torch import WhisperLargeV3Model, enroll_component
from dinkster_inference_torch import whisper as whisper_model
from dinkster_inference_torch import whisper_component as component
from dinkster_inference_torch.operations import INITLESS


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class _MissingResolver:
    def resolve(self, digest: str) -> None:
        del digest
        return None


class _AttentionKernel:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

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
        self.shapes.append(tuple(q.shape))
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=_FixedResolver(path),
    )


def _tiny_encoder(
    layers: int,
    kernel: _AttentionKernel,
) -> Any:
    encoder = whisper_model._AudioEncoder(  # pyright: ignore[reportPrivateUsage]
        4,
        3,
        8,
        2,
        layers,
        operations=INITLESS,
        attention_kernel=kernel,
    )
    generator = torch.Generator().manual_seed(295)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.025)
    return encoder


def test_full_model_layout_and_depth_match_the_torch_free_contract() -> None:
    with torch.device("meta"):
        model = WhisperLargeV3Model()

    assert {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    } == whisper_large_v3_layout()
    assert len(model.encoder.layers) == 32
    assert model.feature_extractor.window.device.type == "cpu"
    assert model.feature_extractor.mel_filters.shape == (201, 128)
    assert set(model.state_dict()).isdisjoint(
        {"feature_extractor.window", "feature_extractor.mel_filters"}
    )


def test_feature_extractor_matches_pinned_chunking_and_log_mel_math() -> None:
    extractor = whisper_model._WhisperFeatureExtractor(  # pyright: ignore[reportPrivateUsage]
        WHISPER_LARGE_V3
    )
    silence = torch.zeros(1, 2, 320, dtype=torch.float32)
    actual = extractor(silence)
    assert actual.shape == (1, 128, 3000)
    assert torch.equal(actual, torch.full_like(actual, -1.5))

    audio = torch.linspace(-0.75, 0.75, 2 * 480_321, dtype=torch.float32).view(1, 2, -1)
    actual = extractor(audio)
    trimmed = audio.mean(dim=1)[..., :480_000]
    window = torch.hann_window(400)
    spectrum = torch.stft(
        trimmed,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spectrum.abs().pow(2.0)
    mel_filters = cast(torch.Tensor, extractor.mel_filters)
    expected = torch.matmul(power.transpose(-1, -2), mel_filters).transpose(-1, -2)
    expected = expected[:, :, :-1].clamp(min=1e-10).log10()
    expected = torch.maximum(expected, expected.max() - 8.0)
    expected = (expected + 4.0) / 4.0
    assert torch.equal(actual, expected)


def test_feature_extractor_is_bit_equal_to_live_torchaudio() -> None:
    torchaudio = pytest.importorskip("torchaudio")
    extractor = whisper_model._WhisperFeatureExtractor(  # pyright: ignore[reportPrivateUsage]
        WHISPER_LARGE_V3
    )
    audio = torch.linspace(-0.5, 0.5, 2 * 2048, dtype=torch.float32).reshape(1, 2, -1)

    actual = extractor(audio)

    waveform = torch.nn.functional.pad(audio.mean(dim=1), (0, 480_000 - 2048))
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=400,
        hop_length=160,
        n_mels=128,
        f_min=0,
        f_max=8000,
        norm="slaney",
        mel_scale="slaney",
    )
    expected = transform(waveform)[:, :, :-1]
    expected = expected.clamp(min=1e-10).log10()
    expected = torch.maximum(expected, expected.max() - 8.0)
    expected = (expected + 4.0) / 4.0
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "audio",
    (
        torch.zeros(1, 1, 4, dtype=torch.float16),
        torch.zeros(1, 4, dtype=torch.float32),
        torch.zeros(1, 0, 4, dtype=torch.float32),
    ),
)
def test_feature_extractor_refuses_wrong_dtype_or_geometry(audio: torch.Tensor) -> None:
    extractor = whisper_model._WhisperFeatureExtractor(  # pyright: ignore[reportPrivateUsage]
        WHISPER_LARGE_V3
    )
    with pytest.raises((TypeError, ValueError)):
        extractor(audio)


def test_encoder_returns_32_pre_layer_states_plus_final_normalized_state() -> None:
    kernel = _AttentionKernel()
    encoder = _tiny_encoder(32, kernel)
    features = torch.linspace(-1.0, 1.0, 2 * 4 * 6).reshape(2, 4, 6)

    output, layers = encoder(features)

    assert output.shape == (2, 3, 8)
    assert len(layers) == 33
    assert layers[-1] is output
    assert all(layer.shape == output.shape for layer in layers)
    assert kernel.shapes == [(2, 2, 3, 4)] * 32
    with pytest.raises(ValueError, match="Whisper features"):
        encoder(torch.zeros(2, 4, 5))


def test_encoder_position_embedding_and_layers_route_through_residency() -> None:
    kernel = _AttentionKernel()
    encoder = _tiny_encoder(2, kernel)
    mechanism = enroll_component(encoder, load_device="cpu", offload_device="cpu")
    features = torch.linspace(-0.5, 0.5, 4 * 6).reshape(1, 4, 6)
    mechanism.partially_load(None)
    resident = encoder(features)
    mechanism.unload()
    offloaded = encoder(features)

    assert torch.equal(resident[0], offloaded[0])
    assert all(
        torch.equal(left, right) for left, right in zip(resident[1], offloaded[1], strict=True)
    )
    assert encoder.embed_positions.residency_binding() is not None  # type: ignore[attr-defined]


def test_component_loader_preserves_identity_descriptor_and_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path)
    expected_identity = "native:dinkster.whisper-large-v3:" + "a" * 64
    seen: dict[str, object] = {}
    planned = SimpleNamespace(config=WHISPER_LARGE_V3)
    source = SimpleNamespace()

    def header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(handle=handle, path=path)
        return source

    def plan(candidate: object, *, path: Path) -> object:
        seen.update(planner_source=candidate, planner_path=path)
        return planned

    def identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return expected_identity

    model = SimpleNamespace(config=WHISPER_LARGE_V3)

    def load(candidate: object, builder: object, **kwargs: object) -> object:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_plan=candidate, builder=builder, kwargs=kwargs)
        assert kwargs["source_file"] is seen["handle"]
        assert kwargs["source"] is source
        assert pinned.file is seen["handle"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return model

    monkeypatch.setattr(component, "load_safetensors_header_from_file", header)
    monkeypatch.setattr(component, "plan_whisper_large_v3_component", plan)
    monkeypatch.setattr(component, "whisper_large_v3_component_runtime_identity", identity)
    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_whisper_large_v3_component(
        path,
        asset=asset,
        expected_identity=expected_identity,
        compute_dtype=torch.float32,
    )

    assert loaded.module is model
    assert loaded.plan is planned
    assert loaded.runtime_identity == expected_identity
    assert seen["path"] == path
    assert seen["planner_path"] == path
    assert seen["identity_plan"] is planned
    assert seen["identity_dtype"] is FLOAT32
    assert seen["load_plan"] is planned
    assert seen["builder"] is WhisperLargeV3Model
    kwargs = cast("dict[str, object]", seen["kwargs"])
    assert kwargs["compute_dtype"] is torch.float32
    assert kwargs["fp8_matmul"] is False


def test_component_loader_refuses_size_unavailable_dtype_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path)
    identity = "native:dinkster.whisper-large-v3:" + "b" * 64
    mismatched = AssetRef(
        asset.digest,
        asset.name,
        asset.size + 1,
        resolver=_FixedResolver(path),
    )
    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="byte size differs"):
        component.load_whisper_large_v3_component(
            path,
            asset=mismatched,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )
    unavailable = AssetRef(
        asset.digest,
        asset.name,
        asset.size,
        resolver=_MissingResolver(),
    )
    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="artifact is unavailable"):
        component.load_whisper_large_v3_component(
            path,
            asset=unavailable,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="compute dtype must be float32"):
        component.load_whisper_large_v3_component(
            path,
            asset=asset,
            expected_identity=identity,
            compute_dtype=torch.float16,
        )

    source = SimpleNamespace()
    plan = SimpleNamespace(config=WHISPER_LARGE_V3)

    def header(_handle: BinaryIO, *, path: Path) -> object:
        del path
        return source

    def planner(_source: object, *, path: Path) -> object:
        del path
        return plan

    def changed_identity(_plan: object, _dtype: object) -> str:
        return "native:dinkster.whisper-large-v3:" + "c" * 64

    monkeypatch.setattr(
        component,
        "load_safetensors_header_from_file",
        header,
    )
    monkeypatch.setattr(
        component,
        "plan_whisper_large_v3_component",
        planner,
    )
    monkeypatch.setattr(
        component,
        "whisper_large_v3_component_runtime_identity",
        changed_identity,
    )

    def fail_load(*_args: object, **_kwargs: object) -> object:
        pytest.fail("payload loading must not run on identity mismatch")

    monkeypatch.setattr(component, "_load_component", fail_load)
    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="expected component identity"):
        component.load_whisper_large_v3_component(
            path,
            asset=asset,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )
