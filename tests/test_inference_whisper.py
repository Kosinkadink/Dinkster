"""Torch-free Whisper Large v3 layout, planning, and identity tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    INT64,
    WHISPER_LARGE_V3,
    ComponentBinding,
    TensorGeometry,
    WeightEntry,
    WhisperLargeV3ComponentAssemblyError,
    plan_whisper_large_v3_component,
    whisper_large_v3_component_runtime_identity,
    whisper_large_v3_layout,
)
from dinkster_inference import whisper_component as component
from dinkster_inference.weights import WeightSource


@dataclass(frozen=True)
class _Source:
    path: Path
    geometries: Mapping[str, TensorGeometry]
    asset_digest: str | None = "blake3:" + "1" * 64
    asset_size: int | None = 3_087_130_976

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def _geometries() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, FLOAT16)
        for key, shape in component._official_artifact_layout().items()  # pyright: ignore[reportPrivateUsage]
    }


def test_whisper_large_v3_config_and_encoder_layout_are_exact() -> None:
    config = WHISPER_LARGE_V3
    layout = whisper_large_v3_layout()

    assert (
        config.n_mels,
        config.n_audio_ctx,
        config.n_audio_state,
        config.n_audio_head,
        config.n_audio_layer,
    ) == (128, 1500, 1280, 20, 32)
    assert (config.sample_rate, config.n_fft, config.hop_length, config.chunk_samples) == (
        16_000,
        400,
        160,
        480_000,
    )
    assert len(layout) == 487
    assert layout["encoder.conv1.weight"] == (1280, 128, 3)
    assert layout["encoder.embed_positions.weight"] == (1500, 1280)
    assert layout["encoder.layers.31.self_attn.k_proj.weight"] == (1280, 1280)
    assert layout["encoder.layers.31.fc1.weight"] == (5120, 1280)
    assert layout["encoder.layer_norm.weight"] == (1280,)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_mels", 0),
        ("n_audio_head", True),
        ("n_audio_head", 24),
        ("n_fft", 401),
        ("chunk_samples", 479_999),
    ),
)
def test_whisper_config_refuses_invalid_geometry(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        replace(WHISPER_LARGE_V3, **{field: value})


def test_whisper_planner_accepts_only_the_official_full_artifact(tmp_path: Path) -> None:
    path = tmp_path / "whisper_large_v3_fp16.safetensors"
    source = _Source(path, _geometries())

    plan = plan_whisper_large_v3_component(cast("WeightSource", source), path=path)

    assert plan.component == "whisper-large-v3"
    assert len(plan.keys) == 487
    assert len(plan.ignored) == 772
    assert plan.keys["encoder.embed_positions.weight"] == ("model.encoder.embed_positions.weight")
    assert plan.keys["encoder.layers.31.self_attn.q_proj.weight"] == (
        "model.encoder.layers.31.self_attn.q_proj.weight"
    )
    assert all(key.startswith("model.decoder.") for key in plan.ignored)
    assert plan.identity_facts == (
        f"asset_digest={source.asset_digest}",
        f"asset_size={source.asset_size}",
        "artifact_layout=official-full-whisper-large-v3",
        "sample_rate=16000",
        "chunk_samples=480000",
        "audio_context=1500",
        "layer_outputs=33",
    )
    identity = whisper_large_v3_component_runtime_identity(plan, FLOAT32)
    ComponentBinding("whisper-large-v3", "dinkster.whisper-large-v3", identity)
    assert identity.startswith("native:dinkster.whisper-large-v3:")
    with pytest.raises(TypeError):
        plan.keys["encoder.conv1.weight"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_encoder", "missing"),
        ("missing_decoder", "missing"),
        ("unexpected", "unexpected"),
        ("encoder_shape", "expected shape"),
        ("decoder_shape", "expected shape"),
        ("decoder_dtype", "floating-point storage"),
    ),
)
def test_whisper_planner_refuses_every_artifact_discrepancy(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / "whisper.safetensors"
    geometries = _geometries()
    encoder = "model.encoder.layers.31.self_attn.q_proj.weight"
    decoder = "model.decoder.layers.31.encoder_attn.q_proj.weight"
    if mutation == "missing_encoder":
        del geometries[encoder]
    elif mutation == "missing_decoder":
        del geometries[decoder]
    elif mutation == "unexpected":
        geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    elif mutation == "encoder_shape":
        geometries[encoder] = TensorGeometry((1, 1280), FLOAT16)
    elif mutation == "decoder_shape":
        geometries[decoder] = TensorGeometry((1, 1280), FLOAT16)
    else:
        geometries[decoder] = TensorGeometry((1280, 1280), INT64)

    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match=message):
        plan_whisper_large_v3_component(
            cast("WeightSource", _Source(path, geometries)),
            path=path,
        )


def test_whisper_planner_refuses_encoder_only_artifact(tmp_path: Path) -> None:
    path = tmp_path / "encoder-only.safetensors"
    geometries = {
        key: value for key, value in _geometries().items() if key.startswith("model.encoder.")
    }

    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="model.decoder"):
        plan_whisper_large_v3_component(
            cast("WeightSource", _Source(path, geometries)),
            path=path,
        )


def test_whisper_identity_changes_with_artifact_and_compute_dtype(tmp_path: Path) -> None:
    path = tmp_path / "whisper.safetensors"
    geometries = _geometries()
    first = plan_whisper_large_v3_component(
        cast("WeightSource", _Source(path, geometries)),
        path=path,
    )
    second = plan_whisper_large_v3_component(
        cast(
            "WeightSource",
            _Source(path, geometries, asset_digest="blake3:" + "2" * 64),
        ),
        path=path,
    )

    assert whisper_large_v3_component_runtime_identity(
        first, FLOAT32
    ) != whisper_large_v3_component_runtime_identity(second, FLOAT32)
    assert whisper_large_v3_component_runtime_identity(
        first, FLOAT32
    ) != whisper_large_v3_component_runtime_identity(first, FLOAT16)


def test_whisper_planner_refuses_path_and_identity_ambiguity(tmp_path: Path) -> None:
    path = tmp_path / "whisper.safetensors"
    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="path differs"):
        plan_whisper_large_v3_component(
            cast("WeightSource", _Source(path, _geometries())),
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(WhisperLargeV3ComponentAssemblyError, match="must carry asset identity"):
        plan_whisper_large_v3_component(
            cast(
                "WeightSource",
                _Source(path, _geometries(), asset_digest=None),
            ),
            path=path,
        )
