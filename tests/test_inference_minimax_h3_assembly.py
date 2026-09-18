"""Torch-free MiniMax H3 split-planning and codec-contract tests."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest
from dinkster_inference.assembly import NativePlanningContext
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.minimax_h3_assembly import (
    component_plan_claims,
    minimax_h3_component_runtime_identity,
    plan_minimax_h3_common_component,
    plan_minimax_h3_model_assembly,
)
from dinkster_inference.minimax_h3_codecs import (
    minimax_h3_audio_vae_layout,
    minimax_h3_video_vae_layout,
)
from dinkster_inference.minimax_h3_conditioner import minimax_h3_conditioner_layout
from dinkster_inference.minimax_h3_dit import minimax_h3_dit_layout
from dinkster_inference.sources import (
    load_safetensors_header,
    load_safetensors_header_from_file,
)
from dinkster_inference.weights import AssetIdentifiedSource, TensorGeometry, WeightEntry

_CONTEXT = NativePlanningContext(
    torch_version="2.99.0-test",
    comfy_kitchen_version="9.9.9-test",
)
_TEST_IDENTITIES = {
    role: (index, "blake3:" + f"{index:x}" * 64)
    for index, role in enumerate(
        ("fl2va-dit", "ref2va-dit", "qwen3vl-32b-conditioner", "video-vae", "audio-vae"),
        1,
    )
}


class HeaderSource:
    """Header-only fake without asset identity."""

    def __init__(
        self,
        path: Path,
        geometries: Mapping[str, TensorGeometry],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.path = path
        self.geometries = dict(geometries)
        self._metadata = {} if metadata is None else dict(metadata)

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return self._metadata


class IdentifiedHeaderSource(HeaderSource):
    """Header-only fake that carries provider-artifact identity."""

    def __init__(
        self,
        path: Path,
        geometries: Mapping[str, TensorGeometry],
        *,
        asset_digest: str | None,
        asset_size: int | None,
    ) -> None:
        super().__init__(path, geometries)
        self.asset_digest = asset_digest
        self.asset_size = asset_size


def _authority(role: str) -> tuple[int, str]:
    return _TEST_IDENTITIES[role]


def _dit_geometries() -> dict[str, TensorGeometry]:
    layout = minimax_h3_dit_layout()
    return {
        key: TensorGeometry(shape, FLOAT32 if key in layout.fp32_storage_keys else BFLOAT16)
        for key, shape in layout.keys.items()
    }


def _component_geometries(
    layout: Mapping[str, tuple[int, ...]], dtype: DType
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}


def _identified(tmp_path: Path, role: str, name: str) -> IdentifiedHeaderSource:
    geometries: dict[str, TensorGeometry]
    if role in ("fl2va-dit", "ref2va-dit"):
        geometries = _dit_geometries()
    elif role == "qwen3vl-32b-conditioner":
        geometries = _component_geometries(minimax_h3_conditioner_layout().keys, BFLOAT16)
    elif role == "video-vae":
        geometries = _component_geometries(minimax_h3_video_vae_layout(), FLOAT16)
    else:
        geometries = _component_geometries(minimax_h3_audio_vae_layout(), FLOAT32)
    size, digest = _authority(role)
    return IdentifiedHeaderSource(tmp_path / name, geometries, asset_digest=digest, asset_size=size)


@pytest.mark.parametrize(
    ("role", "component", "dtype"),
    (
        ("qwen3vl-32b-conditioner", "conditioner", BFLOAT16),
        ("video-vae", "video_vae", FLOAT16),
        ("audio-vae", "audio_vae", FLOAT16),
    ),
)
def test_standalone_common_component_plan_binds_asset_identity(
    tmp_path: Path,
    role: str,
    component: str,
    dtype: DType,
) -> None:
    source = _identified(tmp_path, role, f"{role}.safetensors")
    plan = plan_minimax_h3_common_component(
        source,
        role=role,  # type: ignore[arg-type]
        path=source.path,
        context=_CONTEXT,
    )
    size, digest = _authority(role)

    assert plan.component == component
    assert f"asset_digest={digest}" in plan.identity_facts
    assert f"asset_size={size}" in plan.identity_facts
    assert minimax_h3_component_runtime_identity(
        plan,
        role,  # type: ignore[arg-type]
        dtype,
    ).startswith("native:dinkster.minimax_h3:")


def test_planning_context_requires_non_empty_versions() -> None:
    with pytest.raises(ValueError, match="torch_version"):
        NativePlanningContext(torch_version="", comfy_kitchen_version="1.0")
    with pytest.raises(ValueError, match="comfy_kitchen_version"):
        NativePlanningContext(torch_version="2.13.0", comfy_kitchen_version="")


def test_video_and_audio_vae_layouts_are_pinned() -> None:
    video = minimax_h3_video_vae_layout()
    audio = minimax_h3_audio_vae_layout()
    assert len(video) == 562
    assert len(audio) == 917
    for layout in (video, audio):
        assert isinstance(layout, MappingProxyType)
        for key, shape in layout.items():
            assert key
            assert type(shape) is tuple
            assert all(type(dim) is int and dim > 0 for dim in shape)
    assert minimax_h3_video_vae_layout() is video
    assert minimax_h3_audio_vae_layout() is audio


def _minimal_safetensors(path: Path) -> None:
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


def test_safetensors_source_carries_asset_identity(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _minimal_safetensors(path)
    digest = "blake3:" + "0" * 64

    unidentified = load_safetensors_header(path)
    assert unidentified.asset_digest is None
    assert unidentified.asset_size is None

    identified = load_safetensors_header(path, asset_digest=digest, asset_size=17)
    assert identified.asset_digest == digest
    assert identified.asset_size == 17
    assert isinstance(identified, AssetIdentifiedSource)

    with path.open("rb") as handle:
        from_file = load_safetensors_header_from_file(
            handle, path=path, asset_digest=digest, asset_size=17
        )
    assert from_file.asset_digest == digest
    assert from_file.asset_size == 17


def test_safetensors_source_rejects_partial_or_malformed_identity(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _minimal_safetensors(path)
    digest = "blake3:" + "0" * 64

    with pytest.raises(ValueError, match="asset_digest and asset_size"):
        load_safetensors_header(path, asset_digest=digest)
    with pytest.raises(ValueError, match="asset_digest and asset_size"):
        load_safetensors_header(path, asset_size=17)
    with pytest.raises(ValueError, match="canonical blake3"):
        load_safetensors_header(path, asset_digest="0" * 71, asset_size=17)
    with pytest.raises(ValueError, match="canonical blake3"):
        load_safetensors_header(path, asset_digest="blake3:" + "0" * 63, asset_size=17)
    with pytest.raises(ValueError, match="canonical blake3"):
        load_safetensors_header(path, asset_digest="blake3:" + "G" * 64, asset_size=17)
    with pytest.raises(ValueError, match="nonnegative"):
        load_safetensors_header(path, asset_digest=digest, asset_size=-1)


def test_model_assembly_plans_one_dit_without_shared_assets(tmp_path: Path) -> None:
    path = tmp_path / "ref2va.safetensors"
    source = HeaderSource(path, _dit_geometries())
    plan = plan_minimax_h3_model_assembly(source, role="ref2va-dit", path=path, context=_CONTEXT)
    assert plan.diffusion_role == "ref2va-dit"
    assert plan.diffusion.component == "diffusion"
    assert "torch_version=2.99.0-test" in plan.diffusion.identity_facts
    assert "artifact_role=ref2va-dit" in plan.diffusion.identity_facts
    assert plan.claims == component_plan_claims(
        plan.diffusion  # pyright: ignore[reportArgumentType]
    )
