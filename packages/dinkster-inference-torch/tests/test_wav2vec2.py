"""Native Wav2Vec2 strict-loader, profile, and executed-reference tests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    FLOAT32,
    WAV2VEC2_CHINESE_BASE,
    WAV2VEC2_LARGE,
    Wav2Vec2ComponentAssemblyError,
    Wav2Vec2Config,
    plan_wav2vec2_component,
    wav2vec2_component_runtime_identity,
    wav2vec2_layout,
)
from dinkster_inference.sources import load_safetensors_header_from_file
from dinkster_inference_torch import Wav2Vec2Model
from dinkster_inference_torch import wav2vec2_component as component

GOLDENS = Path(__file__).parent / "goldens" / "wav2vec2_goldens.json"
ARTIFACT_SIZE = 630_997_322
ARTIFACT_SHA256 = "f0017a43ea57ef6b3d4866be607844bbd8cada6d30966f7d70044ed0d63d3f9e"
ARTIFACT_BLAKE3 = "blake3:37197b76dfe7c066f6fdd94c62a5c3b79568756b02bf6e48704d4490dc687d4b"


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


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=_FixedResolver(path),
    )


def _artifact() -> Path:
    configured = os.environ.get("DINKSTER_WAN22_S2V_AUDIO_ENCODER")
    if configured is None:
        pytest.skip("set DINKSTER_WAN22_S2V_AUDIO_ENCODER for artifact parity")
    path = Path(configured)
    if not path.is_file():
        pytest.fail("configured Wav2Vec2 artifact does not exist")
    assert path.stat().st_size == ARTIFACT_SIZE
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == ARTIFACT_SHA256
    return path


def _real_plan(path: Path) -> object:
    with path.open("rb") as handle:
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = component._PinnedSource(  # pyright: ignore[reportPrivateUsage]
            source,
            handle,
            ARTIFACT_BLAKE3,
            ARTIFACT_SIZE,
        )
        return plan_wav2vec2_component(pinned, path=path)


def _passthrough_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    del k, v, mask, causal, scale, enable_gqa
    return q


@pytest.mark.parametrize("config", [WAV2VEC2_LARGE, WAV2VEC2_CHINESE_BASE])
def test_wav2vec2_profiles_build_exact_operations_geometry(config: Wav2Vec2Config) -> None:
    with torch.device("meta"):
        model = Wav2Vec2Model(config, attention_kernel=_passthrough_attention)

    assert set(model.state_dict()) == set(wav2vec2_layout(config))
    assert model.encoder.pos_conv_embed.conv.weight_v.shape == (
        config.embed_dim,
        config.embed_dim // 16,
        128,
    )
    layers = model.feature_extractor.conv_layers
    assert len(layers) == 7
    if config == WAV2VEC2_CHINESE_BASE:
        assert isinstance(layers[0].layer_norm, torch.nn.GroupNorm)
        assert layers[0].layer_norm.num_groups == 512
        assert layers[0].layer_norm.eps == 1e-5
        assert all(not hasattr(layer, "layer_norm") for layer in layers[1:])
    else:
        assert all(isinstance(layer.layer_norm, torch.nn.LayerNorm) for layer in layers)

    with pytest.raises(ValueError, match="exact large and Chinese base"):
        Wav2Vec2Model(replace(config, do_normalize=not config.do_normalize))


class _FeatureRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: torch.Tensor | None = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        self.observed = audio.detach().clone()
        return audio.unsqueeze(-1)


class _ProbeEncoder(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return features, (features,)


def _preprocessing_probe(config: Wav2Vec2Config) -> tuple[Wav2Vec2Model, _FeatureRecorder]:
    model = cast("Any", Wav2Vec2Model.__new__(Wav2Vec2Model))
    torch.nn.Module.__init__(model)
    recorder = _FeatureRecorder()
    model.config = config
    model.feature_extractor = recorder
    model.feature_projection = torch.nn.Identity()
    model.encoder = _ProbeEncoder()
    return cast("Wav2Vec2Model", model), recorder


def test_wav2vec2_profile_preprocessing_preserves_chinese_base_audio() -> None:
    audio = torch.tensor(
        [[[-2.0, 0.0, 2.0, 4.0], [0.0, 2.0, 4.0, 6.0]]],
        dtype=torch.float32,
    )
    averaged = audio.mean(dim=1)
    base, base_recorder = _preprocessing_probe(WAV2VEC2_CHINESE_BASE)
    large, large_recorder = _preprocessing_probe(WAV2VEC2_LARGE)

    base(audio)
    large(audio)

    assert base_recorder.observed is not None
    assert large_recorder.observed is not None
    torch.testing.assert_close(base_recorder.observed, averaged, rtol=0.0, atol=0.0)
    expected_large = (averaged - averaged.mean()) / torch.sqrt(averaged.var() + 1e-7)
    torch.testing.assert_close(large_recorder.observed, expected_large, rtol=0.0, atol=0.0)


def test_wav2vec2_chinese_base_cpu_forward_returns_thirteen_layer_tensors() -> None:
    with torch.device("meta"):
        model = Wav2Vec2Model(
            WAV2VEC2_CHINESE_BASE,
            attention_kernel=_passthrough_attention,
        ).half()
    model.to_empty(device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.encoder.pos_conv_embed.conv.weight_v.fill_(1.0)
    audio = torch.linspace(-1.0, 1.0, 720, dtype=torch.float16).reshape(1, 1, -1)

    with torch.inference_mode():
        output, layers = model(audio)

    assert output.shape == (1, 2, 768)
    assert len(layers) == 13
    assert all(layer.shape == output.shape for layer in layers)
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("profile", [WAV2VEC2_LARGE, WAV2VEC2_CHINESE_BASE])
def test_component_loader_preserves_identity_open_descriptor_and_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: Wav2Vec2Config,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path)
    expected_identity = "native:dinkster.wav2vec2:" + "a" * 64
    seen: dict[str, object] = {}
    planned = SimpleNamespace(config=profile)
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

    model = SimpleNamespace(config=profile)

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
    monkeypatch.setattr(component, "plan_wav2vec2_component", plan)
    monkeypatch.setattr(component, "wav2vec2_component_runtime_identity", identity)
    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_wav2vec2_component(
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
    assert seen["builder"] is Wav2Vec2Model
    kwargs = cast("dict[str, object]", seen["kwargs"])
    assert kwargs["compute_dtype"] is torch.float32
    assert kwargs["fp8_matmul"] is False


def test_component_loader_refuses_identity_size_unavailable_and_invalid_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path)
    identity = "native:dinkster.wav2vec2:" + "b" * 64
    mismatched = AssetRef(
        asset.digest,
        asset.name,
        asset.size + 1,
        resolver=_FixedResolver(path),
    )
    with pytest.raises(Wav2Vec2ComponentAssemblyError, match="byte size differs"):
        component.load_wav2vec2_component(
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
    with pytest.raises(Wav2Vec2ComponentAssemblyError, match="artifact is unavailable"):
        component.load_wav2vec2_component(
            path,
            asset=unavailable,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_wav2vec2_component(
            path,
            asset=asset,
            expected_identity=identity,
            compute_dtype=torch.float64,
        )

    source = SimpleNamespace()
    plan = SimpleNamespace(config=WAV2VEC2_LARGE)

    def load_header(_handle: object, *, path: Path) -> object:
        del path
        return source

    def plan_component(*_args: object, **_kwargs: object) -> object:
        return plan

    def runtime_identity(*_args: object) -> str:
        return "native:dinkster.wav2vec2:" + "c" * 64

    monkeypatch.setattr(
        component,
        "load_safetensors_header_from_file",
        load_header,
    )
    monkeypatch.setattr(component, "plan_wav2vec2_component", plan_component)
    monkeypatch.setattr(
        component,
        "wav2vec2_component_runtime_identity",
        runtime_identity,
    )

    def fail_load(*_args: object, **_kwargs: object) -> object:
        pytest.fail("payload loading must not run on identity mismatch")

    monkeypatch.setattr(component, "_load_component", fail_load)
    with pytest.raises(Wav2Vec2ComponentAssemblyError, match="expected component identity"):
        component.load_wav2vec2_component(
            path,
            asset=asset,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )


def test_official_artifact_strict_load_matches_pinned_comfyui_reference() -> None:
    path = _artifact()
    plan = _real_plan(path)
    expected_identity = wav2vec2_component_runtime_identity(cast("Any", plan), FLOAT32)
    asset = AssetRef(
        ARTIFACT_BLAKE3,
        path.name,
        ARTIFACT_SIZE,
        resolver=_FixedResolver(path),
    )

    loaded = component.load_wav2vec2_component(
        path,
        asset=asset,
        expected_identity=expected_identity,
        compute_dtype=torch.float32,
    )
    state = loaded.module.state_dict()
    assert set(state) == set(wav2vec2_layout())
    assert all(value.device.type == "cpu" for value in state.values())
    assert all(value.dtype is torch.float16 for value in state.values())
    assert len(loaded.plan.keys) == 422
    assert loaded.plan.ignored == ("lm_head.bias", "lm_head.weight")

    golden = json.loads(GOLDENS.read_text())
    assert golden["_meta"]["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    waveform = torch.linspace(-1.0, 1.0, 6_400, dtype=torch.float32).reshape(1, 1, -1)
    with torch.inference_mode():
        output, layers = loaded.module(waveform)
    assert len(layers) == 25
    assert list(layers[0].shape) == golden["layer_shape"]
    indices = torch.tensor(golden["sample_indices"], dtype=torch.int64)
    for index, layer in enumerate(layers):
        expected = torch.tensor(golden["layer_samples"][index], dtype=torch.float32)
        torch.testing.assert_close(
            layer.flatten().index_select(0, indices),
            expected,
            rtol=0.0,
            atol=0.0,
        )
    expected_output = torch.tensor(golden["output"]["data"], dtype=torch.float32).reshape(
        golden["output"]["shape"]
    )
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
