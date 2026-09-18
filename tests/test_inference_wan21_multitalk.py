"""Torch-free Wan 2.1 InfiniteTalk/MultiTalk admission and planning proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT16,
    FLOAT32,
    INT8,
    WAN21_MULTITALK,
    DType,
    TensorGeometry,
    Wan21MultiTalkConfig,
    WeightEntry,
    plan_wan21_multitalk,
)
from dinkster_inference.wan21_multitalk import (
    detect_wan21_multitalk,
    require_wan21_multitalk_layout,
    wan21_multitalk_layout,
    wan21_multitalk_model_layout,
)


@dataclass
class HeaderSource:
    geometries: dict[str, TensorGeometry]
    path: Path = Path("infinitetalk_single.safetensors")

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def _official_dtype(key: str) -> DType:
    if key.startswith("audio_proj.") or key.endswith("audio_cross_attn.q_linear.bias"):
        return FLOAT32
    if ".audio_cross_attn." in key:
        return FLOAT8_E4M3 if key.endswith(".weight") else FLOAT32
    return BFLOAT16


def source_for() -> HeaderSource:
    return HeaderSource(
        {
            key: TensorGeometry(shape, _official_dtype(key))
            for key, shape in wan21_multitalk_layout().items()
        }
    )


def test_exact_layout_admits_the_immutable_official_mixed_float_geometry() -> None:
    source = source_for()
    layout = wan21_multitalk_layout()

    assert detect_wan21_multitalk(source) is WAN21_MULTITALK
    assert require_wan21_multitalk_layout(source) == (WAN21_MULTITALK, layout)
    assert layout == wan21_multitalk_model_layout()
    assert len(layout) == 330
    assert layout["audio_proj.proj1.weight"] == (512, 46080)
    assert layout["audio_proj.proj1_vf.weight"] == (512, 73728)
    assert layout["audio_proj.proj3.weight"] == (24576, 512)
    assert layout["blocks.39.audio_cross_attn.q_linear.weight"] == (5120, 5120)
    assert layout["blocks.39.audio_cross_attn.kv_linear.weight"] == (10240, 768)
    assert layout["blocks.39.norm_x.weight"] == (5120,)
    assert source.geometries["blocks.0.audio_cross_attn.proj.weight"].dtype == FLOAT8_E4M3
    assert source.geometries["blocks.0.norm_x.weight"].dtype == BFLOAT16


@pytest.mark.parametrize("mutation", ("missing", "foreign", "shape", "integer"))
def test_detection_fails_closed_on_any_layout_or_nonfloating_drift(mutation: str) -> None:
    source = source_for()
    if mutation == "missing":
        source.geometries.pop("blocks.39.norm_x.bias")
    elif mutation == "foreign":
        source.geometries["foreign.weight"] = TensorGeometry((1,), FLOAT16)
    elif mutation == "shape":
        source.geometries["audio_proj.proj1_vf.weight"] = TensorGeometry((512, 73727), FLOAT32)
    else:
        source.geometries["blocks.17.norm_x.weight"] = TensorGeometry((5120,), INT8)
    assert detect_wan21_multitalk(source) is None


def test_all_loadable_floating_storage_dtypes_are_admitted_and_preserved_in_plan() -> None:
    source = source_for()
    source.geometries["blocks.3.audio_cross_attn.proj.weight"] = TensorGeometry(
        (5120, 5120), FLOAT16
    )

    plan = plan_wan21_multitalk(source, asset_digest="blake3:" + "a" * 64)

    assert plan.source_role == "wan21_multitalk"
    assert plan.patch.config is WAN21_MULTITALK
    assert set(plan.patch.keys) == set(wan21_multitalk_model_layout())
    assert plan.patch.dtypes["blocks.3.audio_cross_attn.proj.weight"] == FLOAT16
    assert plan.identity_components == (plan.patch,)


def test_plan_and_profile_are_immutable_and_refuse_forgery() -> None:
    plan = plan_wan21_multitalk(source_for(), asset_digest="blake3:" + "0" * 64)
    with pytest.raises(TypeError):
        plan.patch.keys["audio_proj.proj1.bias"] = "forged"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.asset_digest = "blake3:" + "1" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonical blake3"):
        plan_wan21_multitalk(source_for(), asset_digest="sha256:" + "0" * 64)
    source = source_for()
    source.geometries.pop("audio_proj.norm.bias")
    with pytest.raises(ValueError, match="exact maintained"):
        plan_wan21_multitalk(source, asset_digest="blake3:" + "0" * 64)
    forged = Wan21MultiTalkConfig()
    with pytest.raises(ValueError, match="exact maintained"):
        wan21_multitalk_layout(forged)
