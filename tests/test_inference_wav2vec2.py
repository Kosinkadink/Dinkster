"""Torch-free Wav2Vec2 profile, layout, planning, and identity tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    INT64,
    WAV2VEC2_CHINESE_BASE,
    WAV2VEC2_LARGE,
    ComponentBinding,
    TensorGeometry,
    Wav2Vec2ComponentAssemblyError,
    Wav2Vec2Config,
    WeightEntry,
    plan_wav2vec2_component,
    wav2vec2_component_runtime_identity,
    wav2vec2_layout,
)
from dinkster_inference.weights import WeightSource


@dataclass(frozen=True)
class _Source:
    path: Path
    geometries: Mapping[str, TensorGeometry]
    asset_digest: str | None = "blake3:" + "1" * 64
    asset_size: int | None = 630_997_322

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


_LARGE_MARKERS = {
    "lm_head.bias": (32,),
    "lm_head.weight": (32, 1024),
}
_CHINESE_BASE_MARKERS = {
    "project_hid.bias": (256,),
    "project_hid.weight": (256, 768),
    "project_q.bias": (256,),
    "project_q.weight": (256, 256),
    "quantizer.codevectors": (1, 640, 128),
    "quantizer.weight_proj.bias": (640,),
    "quantizer.weight_proj.weight": (640, 512),
}


def _geometries(config: Wav2Vec2Config = WAV2VEC2_LARGE) -> dict[str, TensorGeometry]:
    geometries = {
        f"wav2vec2.{key}": TensorGeometry(shape, FLOAT16)
        for key, shape in wav2vec2_layout(config).items()
    }
    markers = _LARGE_MARKERS if config == WAV2VEC2_LARGE else _CHINESE_BASE_MARKERS
    geometries.update({key: TensorGeometry(shape, FLOAT16) for key, shape in markers.items()})
    return geometries


def test_wav2vec2_large_layout_is_exact_and_complete() -> None:
    layout = wav2vec2_layout(WAV2VEC2_LARGE)

    assert len(layout) == 422
    assert layout["feature_extractor.conv_layers.0.conv.weight"] == (512, 1, 10)
    assert layout["encoder.pos_conv_embed.conv.weight_v"] == (1024, 64, 128)
    assert layout["encoder.layers.23.feed_forward.intermediate_dense.weight"] == (
        4096,
        1024,
    )
    assert layout["encoder.layer_norm.weight"] == (1024,)


def test_wav2vec2_chinese_base_profile_and_layout_are_exact_and_immutable() -> None:
    config = WAV2VEC2_CHINESE_BASE
    layout = wav2vec2_layout(config)

    assert (
        config.embed_dim,
        config.num_heads,
        config.num_layers,
        config.sample_rate,
    ) == (768, 12, 12, 16_000)
    assert not config.conv_norm
    assert not config.conv_bias
    assert not config.do_normalize
    assert not config.do_stable_layer_norm
    assert len(layout) == 211
    assert layout["feature_extractor.conv_layers.0.layer_norm.weight"] == (512,)
    assert "feature_extractor.conv_layers.0.conv.bias" not in layout
    assert "feature_extractor.conv_layers.1.layer_norm.weight" not in layout
    assert layout["encoder.pos_conv_embed.conv.weight_v"] == (768, 48, 128)
    assert layout["encoder.layers.11.feed_forward.intermediate_dense.weight"] == (3072, 768)
    frozen = cast("Any", config)
    with pytest.raises(FrozenInstanceError):
        frozen.embed_dim = 1024


@pytest.mark.parametrize(
    "values",
    (
        {"embed_dim": 0, "num_heads": 1, "num_layers": 1},
        {"embed_dim": 8, "num_heads": 0, "num_layers": 1},
        {"embed_dim": 8, "num_heads": 1, "num_layers": 0},
        {"embed_dim": 8, "num_heads": True, "num_layers": 1},
        {"embed_dim": 7, "num_heads": 2, "num_layers": 1},
    ),
)
def test_wav2vec2_config_refuses_invalid_dimensions(values: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Wav2Vec2Config(**cast("Any", values))


def test_wav2vec2_config_refuses_non_bool_behavior_flags() -> None:
    invalid = cast("Any", 1)
    with pytest.raises(ValueError, match="exact bools"):
        Wav2Vec2Config(768, 12, 12, conv_bias=invalid)


def test_wav2vec2_layout_refuses_nearby_profiles() -> None:
    with pytest.raises(ValueError, match="exact Wav2Vec2"):
        wav2vec2_layout(replace(WAV2VEC2_LARGE, do_normalize=False))


@pytest.mark.parametrize(
    "config,component,markers,layer_count",
    (
        (WAV2VEC2_LARGE, "wav2vec2-large", _LARGE_MARKERS, 25),
        (
            WAV2VEC2_CHINESE_BASE,
            "wav2vec2-chinese-base",
            _CHINESE_BASE_MARKERS,
            13,
        ),
    ),
)
def test_wav2vec2_planner_accepts_only_exact_official_artifact_profiles(
    tmp_path: Path,
    config: Wav2Vec2Config,
    component: str,
    markers: Mapping[str, tuple[int, ...]],
    layer_count: int,
) -> None:
    path = tmp_path / "wav2vec2.safetensors"
    source = _Source(path, _geometries(config))

    plan = plan_wav2vec2_component(cast("WeightSource", source), path=path)

    assert plan.component == component
    assert plan.config == config
    assert len(plan.keys) == len(wav2vec2_layout(config))
    last_layer = config.num_layers - 1
    assert plan.keys[f"encoder.layers.{last_layer}.attention.q_proj.weight"] == (
        f"wav2vec2.encoder.layers.{last_layer}.attention.q_proj.weight"
    )
    assert plan.ignored == tuple(markers)
    assert plan.identity_facts == (
        f"asset_digest={source.asset_digest}",
        f"asset_size={source.asset_size}",
        "sample_rate=16000",
        f"layer_outputs={layer_count}",
    )
    identity = wav2vec2_component_runtime_identity(plan, FLOAT16)
    ComponentBinding(component, "dinkster.wav2vec2", identity)
    assert identity.startswith("native:dinkster.wav2vec2:")
    assert identity != wav2vec2_component_runtime_identity(plan, FLOAT32)


def test_wav2vec2_profile_identities_cannot_collide(tmp_path: Path) -> None:
    path = tmp_path / "wav2vec2.safetensors"
    large = plan_wav2vec2_component(
        cast("WeightSource", _Source(path, _geometries(WAV2VEC2_LARGE))),
        path=path,
    )
    base = plan_wav2vec2_component(
        cast("WeightSource", _Source(path, _geometries(WAV2VEC2_CHINESE_BASE))),
        path=path,
    )

    large_identity = wav2vec2_component_runtime_identity(large, FLOAT16)
    base_identity = wav2vec2_component_runtime_identity(base, FLOAT16)
    assert large_identity == (
        "native:dinkster.wav2vec2:4a78e23892c052c359676046d0e4be8ccc212358d6387ed2288f8c0f463482d1"
    )
    assert large_identity != base_identity


@pytest.mark.parametrize("config", [WAV2VEC2_LARGE, WAV2VEC2_CHINESE_BASE])
@pytest.mark.parametrize(
    "mutation",
    ["missing", "unexpected", "shape", "dtype", "model_shape", "model_dtype", "prefix"],
)
def test_wav2vec2_planner_refuses_every_artifact_discrepancy(
    tmp_path: Path,
    config: Wav2Vec2Config,
    mutation: str,
) -> None:
    path = tmp_path / "wav2vec2.safetensors"
    geometries = _geometries(config)
    target = next(iter(_LARGE_MARKERS if config == WAV2VEC2_LARGE else _CHINESE_BASE_MARKERS))
    if mutation == "missing":
        del geometries[target]
        message = "missing"
    elif mutation == "unexpected":
        geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
        message = "unexpected"
    elif mutation == "shape":
        geometries[target] = TensorGeometry((1,), FLOAT16)
        message = "expected shape"
    elif mutation == "dtype":
        geometries[target] = TensorGeometry(geometries[target].shape, INT64)
        message = "floating-point storage"
    elif mutation in ("model_shape", "model_dtype"):
        model_target = f"wav2vec2.encoder.layers.{config.num_layers - 1}.attention.q_proj.weight"
        if mutation == "model_shape":
            geometries[model_target] = TensorGeometry((1,), FLOAT16)
            message = "expected shape"
        else:
            geometries[model_target] = TensorGeometry(geometries[model_target].shape, INT64)
            message = "floating-point storage"
    else:
        prefixed = "wav2vec2.feature_projection.layer_norm.weight"
        geometries[prefixed.removeprefix("wav2vec2.")] = geometries.pop(prefixed)
        message = "missing"

    with pytest.raises(Wav2Vec2ComponentAssemblyError, match=message):
        plan_wav2vec2_component(
            cast("WeightSource", _Source(path, geometries)),
            path=path,
        )


def test_wav2vec2_identity_changes_with_artifact_content(tmp_path: Path) -> None:
    path = tmp_path / "wav2vec2.safetensors"
    geometries = _geometries()
    first = plan_wav2vec2_component(
        cast("WeightSource", _Source(path, geometries)),
        path=path,
    )
    second = plan_wav2vec2_component(
        cast(
            "WeightSource",
            _Source(path, geometries, asset_digest="blake3:" + "2" * 64),
        ),
        path=path,
    )

    assert wav2vec2_component_runtime_identity(
        first, FLOAT16
    ) != wav2vec2_component_runtime_identity(second, FLOAT16)


def test_wav2vec2_planner_refuses_path_and_identity_ambiguity(tmp_path: Path) -> None:
    path = tmp_path / "wav2vec2.safetensors"
    with pytest.raises(Wav2Vec2ComponentAssemblyError, match="path differs"):
        plan_wav2vec2_component(
            cast("WeightSource", _Source(path, _geometries())),
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(Wav2Vec2ComponentAssemblyError, match="must carry asset identity"):
        plan_wav2vec2_component(
            cast(
                "WeightSource",
                _Source(path, _geometries(), asset_digest=None),
            ),
            path=path,
        )
