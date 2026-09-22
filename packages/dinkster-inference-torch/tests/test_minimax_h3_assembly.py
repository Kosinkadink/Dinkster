from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    INT8,
    MINIMAX_H3_CONFIG,
    UINT8,
    AttentionCapabilityEvidence,
    ComponentBinding,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.minimax_h3_codecs import (
    minimax_h3_audio_vae_layout,
    minimax_h3_video_vae_layout,
)
from dinkster_inference.minimax_h3_conditioner import minimax_h3_conditioner_layout
from dinkster_inference.minimax_h3_dit import minimax_h3_dit_layout
from dinkster_inference.quantization import LayerQuant
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.weights import TensorGeometry, WeightEntry, WeightSource
from dinkster_inference_torch import attention as attention_module
from dinkster_inference_torch import minimax_h3_assembly as assembly
from dinkster_inference_torch.attention import (
    ATTENTION_ADAPTER_CONTRACT,
    AttentionPolicy,
    AttentionSelection,
    select_attention,
)
from dinkster_inference_torch.minimax_h3_assembly import (
    MiniMaxH3ArtifactPaths,
    MiniMaxH3DiTRole,
    MiniMaxH3SplitAssemblyError,
    MiniMaxH3SplitAssemblyPlan,
    load_minimax_h3_component,
    minimax_h3_audio_vae_runtime_identity,
    minimax_h3_conditioner_runtime_identity,
    minimax_h3_guidance_receipt_identity,
    minimax_h3_video_vae_runtime_identity,
    plan_minimax_h3_model_assembly,
    plan_minimax_h3_split_assembly,
    verify_minimax_h3_artifacts,
)
from dinkster_inference_torch.minimax_h3_audio import MiniMaxH3AudioVAE
from dinkster_inference_torch.minimax_h3_video_vae import MiniMaxH3VideoVAE
from dinkster_inference_torch.module_residency import ModuleStateStore
from dinkster_inference_torch.operations import INITLESS, CastOperations
from dinkster_inference_torch.quant_linear import Int8Linear

_TEST_IDENTITIES = {
    role: ((index, "blake3:" + f"{index:x}" * 64),)
    for index, role in enumerate(
        ("fl2va-dit", "ref2va-dit", "qwen3vl-32b-conditioner", "video-vae", "audio-vae"),
        1,
    )
}


class HeaderSource:
    def __init__(
        self,
        path: Path,
        geometries: Mapping[str, TensorGeometry],
        *,
        keys: Sequence[str] | None = None,
        metadata: Mapping[str, str] | None = None,
        configurations: Mapping[str, bytes] | None = None,
    ) -> None:
        self.path = path
        self.geometries = dict(geometries)
        self._keys = tuple(self.geometries if keys is None else keys)
        self._metadata = {} if metadata is None else dict(metadata)
        self._configurations = {} if configurations is None else dict(configurations)

    def keys(self) -> Sequence[str]:
        return self._keys

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    def read_uint8_configuration(self, key: str) -> bytes:
        return self._configurations[key]


@dataclass(frozen=True)
class FixedResolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _asset(path: Path, digest: str, size: int) -> AssetRef:
    return AssetRef(digest, path.name, size, resolver=FixedResolver(path))


def _skip_asset_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_unverified(asset: AssetRef) -> BinaryIO:
        assert asset.resolver is not None
        path = asset.resolver.resolve(asset.digest)
        assert path is not None
        return path.open("rb")

    monkeypatch.setattr(AssetRef, "open", open_unverified)


def _paths_from_authorities(
    tmp_path: Path,
    role: str,
    authorities: Mapping[str, tuple[tuple[int, str], ...]],
) -> MiniMaxH3ArtifactPaths:
    roles = (role, "qwen3vl-32b-conditioner", "video-vae", "audio-vae")
    selected = {name: authorities[name][0] for name in roles}
    paths = {
        role: tmp_path / "dit.safetensors",
        "qwen3vl-32b-conditioner": tmp_path / "conditioner.safetensors",
        "video-vae": tmp_path / "video.safetensors",
        "audio-vae": tmp_path / "audio.safetensors",
    }
    return MiniMaxH3ArtifactPaths(
        role,  # type: ignore[arg-type]
        paths,
        {
            name: _asset(paths[name], authority[1], authority[0])
            for name, authority in selected.items()
        },
    )


def _paths(tmp_path: Path, role: str = "fl2va-dit") -> MiniMaxH3ArtifactPaths:
    return _paths_from_authorities(
        tmp_path,
        role,
        _TEST_IDENTITIES,
    )


def _sources(
    artifacts: MiniMaxH3ArtifactPaths,
) -> tuple[HeaderSource, HeaderSource, HeaderSource, HeaderSource]:
    dit_layout = minimax_h3_dit_layout()
    diffusion = HeaderSource(
        artifacts.paths[artifacts.diffusion_role],
        {
            key: TensorGeometry(
                shape,
                FLOAT32 if key in dit_layout.fp32_storage_keys else BFLOAT16,
            )
            for key, shape in dit_layout.keys.items()
        },
    )
    conditioner = HeaderSource(
        artifacts.paths["qwen3vl-32b-conditioner"],
        {
            key: TensorGeometry(shape, BFLOAT16)
            for key, shape in minimax_h3_conditioner_layout().keys.items()
        },
    )
    with torch.device("meta"):
        video_state = MiniMaxH3VideoVAE().state_dict()
        audio_state = MiniMaxH3AudioVAE().state_dict()
    video = HeaderSource(
        artifacts.paths["video-vae"],
        {key: TensorGeometry(tuple(value.shape), FLOAT16) for key, value in video_state.items()},
    )
    audio = HeaderSource(
        artifacts.paths["audio-vae"],
        {key: TensorGeometry(tuple(value.shape), FLOAT32) for key, value in audio_state.items()},
    )
    return diffusion, conditioner, video, audio


def _plan(artifacts: MiniMaxH3ArtifactPaths):
    diffusion, conditioner, video, audio = _sources(artifacts)
    return plan_minimax_h3_split_assembly(
        diffusion=diffusion,
        conditioner=conditioner,
        video_vae=video,
        audio_vae=audio,
        artifacts=artifacts,
    )


def test_vendored_codec_layouts_match_live_modules() -> None:
    with torch.device("meta"):
        video = assembly._module_layout(  # pyright: ignore[reportPrivateUsage]
            MiniMaxH3VideoVAE()
        )
        audio = assembly._module_layout(  # pyright: ignore[reportPrivateUsage]
            MiniMaxH3AudioVAE()
        )
    assert tuple(minimax_h3_video_vae_layout()) == tuple(video)
    assert tuple(minimax_h3_audio_vae_layout()) == tuple(audio)
    assert dict(minimax_h3_video_vae_layout()) == video
    assert dict(minimax_h3_audio_vae_layout()) == audio


def test_split_plan_preserves_int8_convrot_dit_provider_contract(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)
    diffusion, conditioner, video, audio = _sources(artifacts)
    layer = "blocks.0.attn.qkv_proj"
    geometries = dict(diffusion.geometries)
    rows, columns = geometries[layer + ".weight"].shape
    assert columns % 256 == 0
    geometries[layer + ".weight"] = TensorGeometry((rows, columns), INT8)
    geometries[layer + ".weight_scale"] = TensorGeometry((rows, 1), FLOAT32)
    configuration = json.dumps(
        {
            "format": "int8_tensorwise",
            "params": {"convrot": True, "convrot_groupsize": 256},
        }
    ).encode()
    config_key = layer + ".comfy_quant"
    geometries[config_key] = TensorGeometry((len(configuration),), UINT8)
    diffusion = HeaderSource(
        diffusion.path,
        geometries,
        configurations={config_key: configuration},
    )

    plan = plan_minimax_h3_split_assembly(
        diffusion=diffusion,
        conditioner=conditioner,
        video_vae=video,
        audio_vae=audio,
        artifacts=artifacts,
    )

    quant = plan.diffusion.quant[layer]
    assert quant.format == "int8_tensorwise"
    assert quant.parameters == {"convrot": True, "convrot_groupsize": 256}
    assert plan.diffusion.dtypes[layer + ".weight"] is INT8
    assert layer + ".weight_scale" in plan.claims["fl2va-dit"]
    assert "int8_provider=dinkster-kitchen.int8_linear" in plan.diffusion.identity_facts
    assert any(
        fact.startswith("dinkster_kitchen_version=") for fact in plan.diffusion.identity_facts
    )


def test_split_plan_preserves_int8_convrot_video_vae_provider_contract(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)
    diffusion, conditioner, video, audio = _sources(artifacts)
    layer = "decoder.transformer_blocks.0.attn.to_qkv"
    geometries = {
        key: TensorGeometry(geometry.shape, FLOAT32) for key, geometry in video.geometries.items()
    }
    rows, columns = geometries[layer + ".weight"].shape
    assert columns % 256 == 0
    geometries[layer + ".weight"] = TensorGeometry((rows, columns), INT8)
    geometries[layer + ".weight_scale"] = TensorGeometry((rows, 1), FLOAT32)
    configuration = json.dumps(
        {
            "format": "int8_tensorwise",
            "params": {"convrot": True, "convrot_groupsize": 256},
        }
    ).encode()
    config_key = layer + ".comfy_quant"
    geometries[config_key] = TensorGeometry((len(configuration),), UINT8)
    video = HeaderSource(
        video.path,
        geometries,
        configurations={config_key: configuration},
    )

    plan = plan_minimax_h3_split_assembly(
        diffusion=diffusion,
        conditioner=conditioner,
        video_vae=video,
        audio_vae=audio,
        artifacts=artifacts,
    )

    quant = plan.video_vae.quant[layer]
    assert quant.format == "int8_tensorwise"
    assert quant.parameters == {"convrot": True, "convrot_groupsize": 256}
    assert plan.video_vae.dtypes[layer + ".weight"] is INT8
    assert layer + ".weight_scale" in plan.claims["video-vae"]
    assert (
        "artifact_provider=Kijai/MiniMax-H3-experimental@7f5705937cc106963a9dd77c322f7631e3610e89"
    ) in plan.video_vae.identity_facts
    assert (
        "artifact_sha256=9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410"
    ) in plan.video_vae.identity_facts
    assert "int8_provider=dinkster-kitchen.int8_linear" in plan.video_vae.identity_facts
    assert any(
        fact.startswith("dinkster_kitchen_version=") for fact in plan.video_vae.identity_facts
    )


def test_split_plan_claims_exact_selected_graph_and_is_immutable(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)
    plan = _plan(artifacts)
    assert set(plan.claims) == {
        "fl2va-dit",
        "qwen3vl-32b-conditioner",
        "video-vae",
        "audio-vae",
    }
    assert len(plan.claims["fl2va-dit"]) == 532
    assert len(plan.claims["qwen3vl-32b-conditioner"]) == 902
    assert len(plan.claims["video-vae"]) == 562
    assert len(plan.claims["audio-vae"]) == 917
    assert "int8_provider=dinkster-kitchen.int8_linear" not in plan.diffusion.identity_facts
    assert not any(
        fact.startswith("dinkster_kitchen_version=") for fact in plan.diffusion.identity_facts
    )
    assert "int8_provider=dinkster-kitchen.int8_linear" not in plan.video_vae.identity_facts
    assert not any(
        fact.startswith("dinkster_kitchen_version=") for fact in plan.video_vae.identity_facts
    )
    assert "artifact_role=fl2va-dit" in plan.diffusion.identity_facts
    with pytest.raises(TypeError):
        plan.claims["extra"] = ()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        artifacts.diffusion_role = "ref2va-dit"  # type: ignore[misc]


def test_split_plan_accepts_ref2va_as_the_alternative_exact_dit(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path, "ref2va-dit")
    plan = _plan(artifacts)
    assert set(plan.claims) == {
        "ref2va-dit",
        "qwen3vl-32b-conditioner",
        "video-vae",
        "audio-vae",
    }
    assert plan.artifacts.diffusion_role == "ref2va-dit"


@pytest.mark.parametrize("role", ["fl2va-dit", "ref2va-dit"])
def test_single_model_plan_loads_only_the_selected_dit(tmp_path: Path, role: str) -> None:
    artifacts = _paths(tmp_path, role)
    diffusion, _conditioner, _video, _audio = _sources(artifacts)

    plan = plan_minimax_h3_model_assembly(
        diffusion,
        role=role,  # type: ignore[arg-type]
        path=artifacts.paths[role],
    )

    assert plan.diffusion_role == role
    assert plan.diffusion.component == "diffusion"
    assert plan.diffusion.path == artifacts.paths[role]
    assert plan.claims == tuple(sorted(diffusion.geometries))
    assert plan.diffusion.identity_facts[-1] == f"artifact_role={role}"


def test_single_model_load_reads_only_one_role_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fl2va.safetensors"
    payload = b"official-fl2va"
    path.write_bytes(payload)
    asset_digest = digest_file(path)
    artifacts = _paths(tmp_path)
    source = _sources(artifacts)[0]
    source.path = path

    def read_header(_handle: BinaryIO, *, path: Path) -> HeaderSource:
        assert path == source.path
        return source

    monkeypatch.setattr(
        assembly,
        "load_safetensors_header_from_file",
        read_header,
    )
    loaded: list[str] = []

    def fake_load(
        component: ComponentPlan[object],
        _build: object,
        **_kwargs: object,
    ) -> object:
        loaded.append(component.component)
        return torch.nn.Identity()

    monkeypatch.setattr(assembly, "_load_component", fake_load)
    source_plan = assembly.plan_minimax_h3_model_assembly(
        source, role="fl2va-dit", path=path
    ).diffusion
    expected_identity = assembly.minimax_h3_dit_runtime_identity(
        asset_digest=asset_digest,
        asset_size=len(payload),
        role="fl2va-dit",
        diffusion_dtype=BFLOAT16.name,
        runtime_facts=source_plan.identity_facts,
    )
    real_plan = assembly.plan_minimax_h3_model_assembly
    planned_sources: list[object] = []

    def plan_pinned_source(
        source: WeightSource,
        *,
        role: MiniMaxH3DiTRole,
        path: Path,
        attention_policy: AttentionPolicy = "auto",
    ) -> object:
        planned_sources.append(source)
        return real_plan(source, role=role, path=path, attention_policy=attention_policy)

    monkeypatch.setattr(assembly, "plan_minimax_h3_model_assembly", plan_pinned_source)
    with pytest.raises(MiniMaxH3SplitAssemblyError, match="constructed"):
        assembly.load_minimax_h3_model(
            path,
            asset=_asset(path, asset_digest, len(payload)),
            role="fl2va-dit",
            expected_identity="wrong",
            attention_backend="flux",
        )
    assert loaded == []
    model = assembly.load_minimax_h3_model(
        path,
        asset=_asset(path, asset_digest, len(payload)),
        role="fl2va-dit",
        expected_identity=expected_identity,
        attention_backend="flux",
    )

    assert model.model_role == "fl2va-dit"
    assert model.runtime_identity == expected_identity
    assert model.receipt_identity is None
    assert loaded == ["diffusion"]
    assert all(
        isinstance(source, assembly._DescriptorPinnedSource)  # pyright: ignore[reportPrivateUsage]
        for source in planned_sources
    )


def test_single_model_load_rejects_unverified_asset_protocol(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"

    @dataclass(frozen=True)
    class UnverifiedAsset:
        digest = "blake3:" + "1" * 64
        size = 1

        def open(self) -> BinaryIO:
            return path.open("rb")

    with pytest.raises(TypeError, match="AssetRef"):
        assembly.load_minimax_h3_model(
            path,
            asset=cast("AssetRef", UnverifiedAsset()),
            role="fl2va-dit",
            expected_identity="identity",
            attention_backend="flux",
        )


def test_split_plan_refuses_missing_foreign_geometry_and_path(
    tmp_path: Path,
) -> None:
    artifacts = _paths(tmp_path)
    diffusion, conditioner, video, audio = _sources(artifacts)
    cases = []
    missing = dict(conditioner.geometries)
    missing.pop(next(iter(missing)))
    cases.append((diffusion, HeaderSource(conditioner.path, missing), video, audio))
    foreign = dict(video.geometries)
    foreign["foreign"] = TensorGeometry((1,), FLOAT16)
    cases.append((diffusion, conditioner, HeaderSource(video.path, foreign), audio))
    geometry = dict(audio.geometries)
    key = next(iter(geometry))
    geometry[key] = TensorGeometry((1,), FLOAT32)
    cases.append((diffusion, conditioner, video, HeaderSource(audio.path, geometry)))
    cases.append(
        (
            HeaderSource(tmp_path / "wrong.safetensors", diffusion.geometries),
            conditioner,
            video,
            audio,
        )
    )
    for bad_diffusion, bad_conditioner, bad_video, bad_audio in cases:
        with pytest.raises((MiniMaxH3SplitAssemblyError, ValueError)):
            plan_minimax_h3_split_assembly(
                diffusion=bad_diffusion,
                conditioner=bad_conditioner,
                video_vae=bad_video,
                audio_vae=bad_audio,
                artifacts=artifacts,
            )


def test_split_plan_accepts_float32_conditioner_storage(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)
    diffusion, conditioner, video, audio = _sources(artifacts)
    float32_conditioner = HeaderSource(
        conditioner.path,
        {
            key: TensorGeometry(geometry.shape, FLOAT32)
            for key, geometry in conditioner.geometries.items()
        },
    )
    plan = plan_minimax_h3_split_assembly(
        diffusion=diffusion,
        conditioner=float32_conditioner,
        video_vae=video,
        audio_vae=audio,
        artifacts=artifacts,
    )
    assert set(plan.conditioner.dtypes.values()) == {FLOAT32}


def test_artifact_verification_fails_closed_on_wrong_payload_size(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)
    artifacts.paths["fl2va-dit"].write_bytes(b"not a model")
    with pytest.raises(MiniMaxH3SplitAssemblyError, match="integrity failure"):
        verify_minimax_h3_artifacts(artifacts)


def test_artifact_verification_checks_every_selected_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    expected: dict[str, tuple[tuple[int, str], ...]] = {}
    for index, (role, path) in enumerate(paths.paths.items()):
        payload = bytes((index + 1,)) * (index + 1)
        path.write_bytes(payload)
        expected[role] = ((len(payload), digest_file(path)),)
    artifacts = _paths_from_authorities(tmp_path, "fl2va-dit", expected)
    assert verify_minimax_h3_artifacts(artifacts) is None
    wrong = dict(artifacts.assets)
    audio = wrong["audio-vae"]
    wrong["audio-vae"] = _asset(artifacts.paths["audio-vae"], "blake3:" + "0" * 64, audio.size)
    with pytest.raises(MiniMaxH3SplitAssemblyError, match="integrity failure"):
        verify_minimax_h3_artifacts(replace(artifacts, assets=wrong))


def test_artifact_paths_reject_unverified_asset_protocols(tmp_path: Path) -> None:
    artifacts = _paths(tmp_path)

    @dataclass(frozen=True)
    class UnverifiedAsset:
        digest: str
        size: int

        def open(self) -> BinaryIO:
            return (tmp_path / "unverified").open("rb")

    assets = dict(artifacts.assets)
    selected = assets[artifacts.diffusion_role]
    assets[artifacts.diffusion_role] = cast(
        "AssetRef", UnverifiedAsset(selected.digest, selected.size)
    )
    with pytest.raises(TypeError, match="AssetRef"):
        replace(artifacts, assets=assets)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows prevents replacing an open file")
def test_configuration_payload_stays_pinned_to_verified_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "configuration.safetensors"
    path.write_bytes(b"new")
    verified_path = tmp_path / "verified.safetensors"
    verified_path.write_bytes(b"old")
    geometry = TensorGeometry((3,), UINT8)
    source = SafetensorsSource(
        path,
        {"config": WeightEntry("config", geometry, 0, geometry.nbytes)},
        {},
    )
    with verified_path.open("rb") as handle:
        pinned = assembly._DescriptorPinnedSource(  # pyright: ignore[reportPrivateUsage]
            source,
            handle,
        )
        assert pinned.entries == source.entries
        assert source.read_uint8_configuration("config") == b"new"
        assert pinned.read_uint8_configuration("config") == b"old"


def test_artifact_verification_accepts_arbitrary_identity_and_rejects_wrong_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dit.safetensors"
    primary = b"int8"
    alternate = b"bf16-alternate"
    for index, payload in enumerate((primary, alternate), 1):
        path.write_bytes(payload)
        with path.open("rb") as handle:
            assembly._verify_artifact_file(  # pyright: ignore[reportPrivateUsage]
                "fl2va-dit", path, handle, "blake3:" + f"{index:x}" * 64, len(payload)
            )

    path.write_bytes(alternate)
    with path.open("rb") as handle:
        with pytest.raises(MiniMaxH3SplitAssemblyError, match="byte size"):
            assembly._verify_artifact_file(  # pyright: ignore[reportPrivateUsage]
                "fl2va-dit", path, handle, "blake3:" + "3" * 64, len(alternate) + 1
            )


def test_artifact_verification_does_not_read_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _paths(tmp_path)
    role = artifacts.diffusion_role
    path = artifacts.paths[role]
    path.write_bytes(b"payload")

    class NoRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, *_args: object) -> bytes:
            raise AssertionError("provider identity verification must not reread payload bytes")

    digest = "blake3:" + "1" * 64
    with path.open("rb") as handle:
        assembly._verify_artifact_file(  # pyright: ignore[reportPrivateUsage]
            role, path, cast("BinaryIO", NoRead(handle)), digest, len(b"payload")
        )


def test_diffusion_builder_binds_reference_split_precision_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-owned float32 islands do not inherit the loader's
    bfloat16 diffusion operations."""
    selected: tuple[object, object, object] | None = None

    def capture(
        *,
        operations: object,
        fp32_operations: object,
        text_operations: object,
        time_embedding_kind: str,
        attention_selection: object,
    ) -> object:
        nonlocal selected
        selected = operations, fp32_operations, text_operations
        assert time_embedding_kind == "curve"
        assert attention_selection is selected_attention
        return torch.nn.Identity()

    monkeypatch.setattr(assembly, "assemble_minimax_h3_dit", capture)
    selected_attention = select_attention("flux", "sdpa")
    for operations in (CastOperations(torch.bfloat16), INITLESS):
        selected = None
        assembly._build_diffusion(  # pyright: ignore[reportPrivateUsage]
            minimax_h3_dit_layout(),
            operations=operations,
            attention_selection=selected_attention,
        )

        assert selected is not None
        assert selected[0] is operations
        assert isinstance(selected[1], CastOperations)
        assert selected[1].dtype is torch.float32
        assert selected[2] is operations


def test_verified_diffusion_declares_route_materialization_ceilings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = CastOperations(torch.bfloat16).linear(2, 2)
    component.load_state_dict(
        {
            "weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "bias": torch.ones(2, dtype=torch.bfloat16),
        },
        strict=True,
        assign=True,
    )

    def load_component(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return component

    monkeypatch.setattr(assembly, "_load_verified_component", load_component)
    loaded = assembly._load_verified_diffusion(  # pyright: ignore[reportPrivateUsage]
        cast("Any", object()),
        cast("Any", object()),
        compute_dtype=torch.bfloat16,
        attention_selection=select_attention("flux", "sdpa"),
    )
    store = ModuleStateStore(loaded)

    assert store.max_materialized_itemsize("weight") == 2
    assert store.max_materialized_itemsize("bias") == 2


@pytest.mark.parametrize("diffusion_dtype", [torch.bfloat16, torch.float32])
def test_h3_projection_storage_matches_comfyui_model_dtype(
    diffusion_dtype: torch.dtype,
) -> None:
    keys = assembly._COMFYUI_MODEL_DTYPE_PROJECTION_KEYS | {  # pyright: ignore[reportPrivateUsage]
        "blocks.0.adaln_proj.linear.weight",
        "blocks.49.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
    }
    source = {key: torch.tensor([1.001], dtype=torch.float32) for key in keys}
    source["blocks.0.attn.q_proj.weight_scale"] = torch.tensor(0.125)

    rounded = assembly._round_h3_projection_storage(  # pyright: ignore[reportPrivateUsage]
        cast("Any", object()), source, diffusion_dtype=diffusion_dtype
    )

    expected = torch.tensor([1.001], dtype=torch.float32).to(diffusion_dtype)
    assert keys == {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
        "blocks.0.adaln_proj.linear.weight",
        "blocks.49.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
    }
    for key in keys:
        assert rounded[key].dtype is diffusion_dtype
        assert torch.equal(rounded[key], expected)
    assert rounded["blocks.0.attn.q_proj.weight_scale"].dtype is torch.float32


def test_artifact_paths_refuse_incomplete_duplicate_and_mutable_authority(
    tmp_path: Path,
) -> None:
    artifacts = _paths(tmp_path)
    with pytest.raises(ValueError, match="complete split graph"):
        replace(artifacts, paths={"fl2va-dit": tmp_path / "dit"})
    duplicate = dict(artifacts.paths)
    duplicate["audio-vae"] = duplicate["video-vae"]
    with pytest.raises(ValueError, match="distinct"):
        replace(artifacts, paths=duplicate)
    with pytest.raises(ValueError, match="immutable authority"):
        replace(artifacts, provider_revision="main")


def _runtime_plans(
    tmp_path: Path,
) -> tuple[MiniMaxH3SplitAssemblyPlan, MiniMaxH3SplitAssemblyPlan, dict[Path, HeaderSource]]:
    fl2va_artifacts = _paths(tmp_path, "fl2va-dit")
    ref2va_artifacts = _paths(tmp_path, "ref2va-dit")
    ref2va_paths = dict(ref2va_artifacts.paths)
    ref2va_paths["ref2va-dit"] = tmp_path / "ref2va.safetensors"
    ref2va_assets = dict(ref2va_artifacts.assets)
    ref2va_asset = ref2va_assets["ref2va-dit"]
    ref2va_assets["ref2va-dit"] = _asset(
        ref2va_paths["ref2va-dit"], ref2va_asset.digest, ref2va_asset.size
    )
    ref2va_artifacts = replace(
        ref2va_artifacts,
        paths=ref2va_paths,
        assets=ref2va_assets,
    )
    fl2va_sources = _sources(fl2va_artifacts)
    ref2va_sources = _sources(ref2va_artifacts)
    fl2va_plan = plan_minimax_h3_split_assembly(
        diffusion=fl2va_sources[0],
        conditioner=fl2va_sources[1],
        video_vae=fl2va_sources[2],
        audio_vae=fl2va_sources[3],
        artifacts=fl2va_artifacts,
    )
    ref2va_plan = plan_minimax_h3_split_assembly(
        diffusion=ref2va_sources[0],
        conditioner=ref2va_sources[1],
        video_vae=ref2va_sources[2],
        audio_vae=ref2va_sources[3],
        artifacts=ref2va_artifacts,
    )
    sources = {source.path: source for source in (*fl2va_sources, *ref2va_sources)}
    return fl2va_plan, ref2va_plan, sources


def _int8_convrot_plan(plan: MiniMaxH3SplitAssemblyPlan) -> MiniMaxH3SplitAssemblyPlan:
    """Requant a BF16 fl2va plan as INT8 ConvRot so its provider fact changes."""
    quant = LayerQuant(
        "blocks.0.attn.qkv_proj",
        "int8_tensorwise",
        "qkv.weight",
        "qkv.weight_scale",
        parameters={"convrot": True, "convrot_groupsize": 256},
    )
    int8_diffusion = replace(plan.diffusion, quant={quant.layer: quant})
    int8_claims = dict(plan.claims)
    int8_claims[plan.artifacts.diffusion_role] = assembly._plan_claims(  # pyright: ignore[reportPrivateUsage]
        cast("ComponentPlan[object]", int8_diffusion)
    )
    return replace(plan, diffusion=int8_diffusion, claims=int8_claims)


def test_guidance_receipt_identity_derives_provider_from_plan(tmp_path: Path) -> None:
    fl2va, _ref2va, _sources_by_path = _runtime_plans(tmp_path)
    assert minimax_h3_guidance_receipt_identity(fl2va) == (
        "distributed:dinkster.minimax_h3:"
        "1647c460cf045d3d77c40711760f8f58fe1feb7c6f84c4b1de28621e7428d0fa"
    )
    int8_fl2va = _int8_convrot_plan(fl2va)
    assert minimax_h3_guidance_receipt_identity(int8_fl2va) == (
        "distributed:dinkster.minimax_h3:"
        "66834bed2771b87b5c186cb0546194e3e90b49bac081c159fcd727873ca1be59"
    )


def test_component_runtime_identities_are_stable_native_bindings(tmp_path: Path) -> None:
    plan = _plan(_paths(tmp_path))
    cases = (
        (
            "conditioner",
            minimax_h3_conditioner_runtime_identity(plan.conditioner),
            minimax_h3_conditioner_runtime_identity(plan.conditioner),
            minimax_h3_conditioner_runtime_identity(
                plan.conditioner, conditioner_dtype=torch.float32
            ),
        ),
        (
            "video-vae",
            minimax_h3_video_vae_runtime_identity(plan.video_vae),
            minimax_h3_video_vae_runtime_identity(plan.video_vae),
            minimax_h3_video_vae_runtime_identity(plan.video_vae, video_vae_dtype=torch.float32),
        ),
        (
            "audio-vae",
            minimax_h3_audio_vae_runtime_identity(plan.audio_vae),
            minimax_h3_audio_vae_runtime_identity(plan.audio_vae),
            minimax_h3_audio_vae_runtime_identity(plan.audio_vae, audio_vae_dtype=torch.float32),
        ),
    )

    identities: set[str] = set()
    for role, identity, repeated, different_dtype in cases:
        assert identity == repeated
        assert identity != different_dtype
        prefix, family, digest = identity.split(":")
        assert (prefix, family) == ("native", MINIMAX_H3_CONFIG.family_id)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        ComponentBinding(role, MINIMAX_H3_CONFIG.family_id, identity)
        identities.add(identity)
    assert len(identities) == 3


@pytest.mark.parametrize(
    ("role", "plan_name", "dtype", "identity"),
    (
        (
            "qwen3vl-32b-conditioner",
            "conditioner",
            torch.bfloat16,
            minimax_h3_conditioner_runtime_identity,
        ),
        ("video-vae", "video_vae", torch.float16, minimax_h3_video_vae_runtime_identity),
        ("audio-vae", "audio_vae", torch.float16, minimax_h3_audio_vae_runtime_identity),
    ),
)
def test_standalone_component_load_preserves_plan_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    plan_name: str,
    dtype: torch.dtype,
    identity: Any,
) -> None:
    _skip_asset_integrity(monkeypatch)
    path = tmp_path / f"{role}.safetensors"
    path.write_bytes(role.encode())
    asset = _asset(path, digest_file(path), path.stat().st_size)
    plan = getattr(_plan(_paths(tmp_path)), plan_name)
    expected_identity = identity(plan)
    module = torch.nn.Sequential(
        Int8Linear(
            16,
            16,
            bias=False,
            compute_dtype=dtype,
            convrot=False,
            convrot_groupsize=256,
        )
    )

    def read_header(_handle: BinaryIO, *, path: Path) -> HeaderSource:
        return HeaderSource(path, {})

    def retain_plan(*_args: object, **_kwargs: object) -> object:
        return plan

    loaded_compute_dtype: torch.dtype | None = None

    def load_component(*_args: object, **kwargs: object) -> torch.nn.Module:
        nonlocal loaded_compute_dtype
        loaded_compute_dtype = cast(torch.dtype, kwargs["compute_dtype"])
        return module

    monkeypatch.setattr(
        assembly,
        "load_safetensors_header_from_file",
        read_header,
    )
    monkeypatch.setattr(assembly, "plan_minimax_h3_common_component", retain_plan)
    monkeypatch.setattr(
        assembly,
        "_load_verified_component",
        load_component,
    )

    loaded = load_minimax_h3_component(
        path,
        asset=asset,
        expected_role=role,  # type: ignore[arg-type]
        expected_identity=expected_identity,
        compute_dtype=dtype,
    )

    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is plan
    assert loaded.runtime_identity == expected_identity
    assert loaded_compute_dtype is (torch.float32 if role == "qwen3vl-32b-conditioner" else dtype)
    quantized = cast(torch.nn.Sequential, loaded.module)[0]
    assert isinstance(quantized, Int8Linear)
    assert quantized.compute_dtype is (
        torch.float32 if role == "qwen3vl-32b-conditioner" else dtype
    )
    assert quantized.full_precision_matmul is (role == "qwen3vl-32b-conditioner")


def test_standalone_component_load_refuses_wrong_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_asset_integrity(monkeypatch)
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path, digest_file(path), path.stat().st_size)

    def load_header(_handle: BinaryIO, *, path: Path) -> HeaderSource:
        return HeaderSource(path, {})

    monkeypatch.setattr(
        assembly,
        "load_safetensors_header_from_file",
        load_header,
    )
    with pytest.raises(
        MiniMaxH3SplitAssemblyError,
        match="conditioner",
    ):
        load_minimax_h3_component(
            path,
            asset=asset,
            expected_role="qwen3vl-32b-conditioner",
            expected_identity="expected",
            compute_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("override_policy", ("dinkster_kitchen_int8", "sol"))
def test_single_model_load_threads_attention_selection_and_identity(
    override_policy: AttentionPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fl2va.safetensors"
    payload = b"official-fl2va"
    path.write_bytes(payload)
    asset_digest = digest_file(path)
    artifacts = _paths(tmp_path)
    source = _sources(artifacts)[0]
    source.path = path

    def fake_load_header(_handle: object, *, path: Path) -> HeaderSource:
        return source

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", fake_load_header)

    selections: list[object] = []

    def fake_load(
        component: ComponentPlan[object],
        build: Any,
        **_kwargs: object,
    ) -> object:
        return build(None, operations=None)

    def fake_build_diffusion(
        _layout: object, *, operations: object, attention_selection: object = None
    ) -> object:
        selections.append(attention_selection)
        return torch.nn.Identity()

    monkeypatch.setattr(assembly, "_load_component", fake_load)
    monkeypatch.setattr(assembly, "_build_diffusion", fake_build_diffusion)

    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", object())
    monkeypatch.setattr(
        attention_module,
        "_KITCHEN_LIST_BACKENDS",
        lambda: {
            "cuda": {
                "available": True,
                "disabled": False,
                "capabilities": ("sol_attn",),
            }
        },
    )

    def sol_device_supported(_device: torch.device | None = None) -> bool:
        return True

    monkeypatch.setattr(
        attention_module,
        "_sol_device_supported",
        sol_device_supported,
    )
    fake_cuda_torch = SimpleNamespace(
        __version__=str(torch.__version__), version=SimpleNamespace(hip=None)
    )
    overrides: tuple[tuple[str, AttentionPolicy], ...] = (("flux", override_policy),)
    token = attention_module.discover_attention_route_token(
        "sdpa",
        requested_role_policies=overrides,
        torch_module=fake_cuda_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert token.version == 2
    assert token.adapter_contract_revision == ATTENTION_ADAPTER_CONTRACT

    capabilities = AttentionCapabilityEvidence(
        version=1,
        available_policies=("sdpa", "sol", "dinkster_kitchen_int8"),
        provider_versions=token.provider_versions,
        adapter_contract_revision=token.adapter_contract_revision,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime=token.sdpa_torch_runtime,
    )
    monkeypatch.setattr(
        attention_module,
        "discover_attention_capabilities",
        lambda: capabilities,
    )

    selected_plan = assembly.plan_minimax_h3_model_assembly(
        source, role="fl2va-dit", path=path, attention_policy=override_policy
    ).diffusion
    source_plan = assembly.plan_minimax_h3_model_assembly(
        source, role="fl2va-dit", path=path
    ).diffusion
    assert selected_plan.identity_facts != source_plan.identity_facts
    expected_provider = {
        "dinkster_kitchen_int8": "attention_provider=dinkster-kitchen.int8_attention",
        "sol": "attention_provider=dinkster-kitchen.sol_attn",
    }[override_policy]
    assert expected_provider in selected_plan.identity_facts
    assert "dinkster_kitchen_version=0.2.35.post1" in selected_plan.identity_facts
    bound_identity = assembly.minimax_h3_dit_runtime_identity(
        asset_digest=asset_digest,
        asset_size=len(payload),
        role="fl2va-dit",
        diffusion_dtype=BFLOAT16.name,
        attention_policy="sdpa",
        attention_route_token=token,
        runtime_facts=selected_plan.identity_facts,
    )
    unbound_identity = assembly.minimax_h3_dit_runtime_identity(
        asset_digest=asset_digest,
        asset_size=len(payload),
        role="fl2va-dit",
        diffusion_dtype=BFLOAT16.name,
        runtime_facts=source_plan.identity_facts,
    )
    assert bound_identity != unbound_identity

    with pytest.raises(MiniMaxH3SplitAssemblyError, match="constructed"):
        assembly.load_minimax_h3_model(
            path,
            asset=_asset(path, asset_digest, len(payload)),
            role="fl2va-dit",
            expected_identity=unbound_identity,
            attention_policy="sdpa",
            attention_route_token=token,
            attention_backend="flux",
        )
    model = assembly.load_minimax_h3_model(
        path,
        asset=_asset(path, asset_digest, len(payload)),
        role="fl2va-dit",
        expected_identity=bound_identity,
        attention_policy="sdpa",
        attention_route_token=token,
        attention_backend="flux",
    )
    assert model.runtime_identity == bound_identity
    assert len(selections) == 1
    selection = selections[0]
    assert isinstance(selection, AttentionSelection)
    assert selection.status.requested_policy == override_policy
    assert selection.status.primary == override_policy
    assert selection.status.authenticated is True
    assert model.assembled.attention_status == {"flux": selection.status}
