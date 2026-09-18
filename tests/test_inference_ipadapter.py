"""Torch-free standard SD1.5 IP-Adapter contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    INT64,
    SD15_IPADAPTER,
    SD15_IPADAPTER_CLIP_VISION,
    SD15_IPADAPTER_SITES,
    PayloadReference,
    PercentRange,
    SD15AttentionContribution,
    SD15IPAdapterDetectError,
    TensorGeometry,
    WeightEntry,
    detect_sd15_ipadapter,
    detect_sd15_ipadapter_clip_vision,
    plan_sd15_ipadapter,
    runtime_component_identity,
    sd15_ipadapter_layout,
)
from dinkster_inference.clip_vision import clip_vision_layout

ASSET_DIGEST = "blake3:" + "a" * 64


@dataclass
class HeaderSource:
    path: Path
    geometries: dict[str, TensorGeometry]
    payload_reads: int = 0

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}

    def read_float_scalar(self, key: str) -> float:
        del key
        self.payload_reads += 1
        raise AssertionError("IP-Adapter planning must not read payloads")


def adapter_geometries() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT16) for key, shape in sd15_ipadapter_layout().items()}


def clip_vision_geometries() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, INT64 if key.endswith("position_ids") else FLOAT32)
        for key, shape in clip_vision_layout(SD15_IPADAPTER_CLIP_VISION).items()
    }


def test_standard_sd15_layout_pins_all_canonical_attn2_sites() -> None:
    assert len(SD15_IPADAPTER_SITES) == 16
    assert tuple(site.adapter_index for site in SD15_IPADAPTER_SITES) == tuple(range(1, 32, 2))
    assert tuple(site.width for site in SD15_IPADAPTER_SITES) == (
        320,
        320,
        640,
        640,
        1280,
        1280,
        1280,
        1280,
        1280,
        640,
        640,
        640,
        320,
        320,
        320,
        1280,
    )
    assert all(site.id.endswith(".attn2") for site in SD15_IPADAPTER_SITES)
    assert len(sd15_ipadapter_layout()) == 36


def test_detectors_admit_only_exact_standard_artifact_layouts() -> None:
    assert detect_sd15_ipadapter(adapter_geometries()) is SD15_IPADAPTER
    assert detect_sd15_ipadapter_clip_vision(clip_vision_geometries()) is SD15_IPADAPTER_CLIP_VISION

    missing = adapter_geometries()
    del missing["ip_adapter.31.to_v_ip.weight"]
    with pytest.raises(SD15IPAdapterDetectError, match="missing ip_adapter.31.to_v_ip.weight"):
        detect_sd15_ipadapter(missing)

    wrong_dtype = clip_vision_geometries()
    wrong_dtype["visual_projection.weight"] = TensorGeometry((1024, 1280), FLOAT16)
    with pytest.raises(SD15IPAdapterDetectError, match="visual_projection.weight"):
        detect_sd15_ipadapter_clip_vision(wrong_dtype)

    extra = adapter_geometries()
    extra["perceiver_resampler.weight"] = TensorGeometry((1,), FLOAT16)
    with pytest.raises(SD15IPAdapterDetectError, match="unexpected key"):
        detect_sd15_ipadapter(extra)


def test_planner_claims_both_complete_headers_without_payload_reads() -> None:
    adapter = HeaderSource(Path("ip-adapter_sd15.safetensors"), adapter_geometries())
    vision = HeaderSource(Path("image_encoder.safetensors"), clip_vision_geometries())

    plan = plan_sd15_ipadapter(
        adapter,
        vision,
        adapter_asset_digest=ASSET_DIGEST,
        clip_vision_asset_digest="blake3:" + "b" * 64,
    )

    assert plan.family_id == "dinkster.sd15"
    assert plan.adapter.config is SD15_IPADAPTER
    assert plan.clip_vision.config is SD15_IPADAPTER_CLIP_VISION
    assert set(plan.adapter.keys.values()) == set(adapter.geometries)
    assert set(plan.clip_vision.keys.values()) == set(vision.geometries)
    assert runtime_component_identity(plan.family_id, plan.identity_components)[0] == (
        "family=dinkster.sd15"
    )
    assert adapter.payload_reads == vision.payload_reads == 0


def test_portable_contribution_is_frozen_and_identity_complete() -> None:
    contribution = SD15AttentionContribution(
        PayloadReference("model-digest"),
        PayloadReference("token-digest"),
        PercentRange(0.25, 0.75),
        1.5,
    )

    assert contribution.sites is SD15_IPADAPTER_SITES
    assert contribution.placement == "attn2-pre-to-out"
    assert contribution.lane_gains == (
        ("positive", 1.0),
        ("negative", 1.0),
        ("empty", 1.0),
    )
    assert contribution.lane_sources == (
        ("positive", "conditional"),
        ("negative", "unconditional"),
        ("empty", "conditional"),
    )
    with pytest.raises(FrozenInstanceError):
        contribution.scalar_gain = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonical 16"):
        replace(contribution, sites=SD15_IPADAPTER_SITES[:-1])
    with pytest.raises(ValueError, match="lane gains are immutable"):
        replace(contribution, lane_gains=(("positive", 1.0),))
    with pytest.raises(ValueError, match="lane sources are immutable"):
        replace(contribution, lane_sources=(("positive", "conditional"),))
