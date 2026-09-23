"""Strict native loading for official TRELLIS.2 flow artifacts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    ComponentPlan,
    DINOv3ViTConfig,
    NAFConfig,
    SparseLatent,
    Trellis2ArtifactRole,
    Trellis2AssemblyError,
    Trellis2ComponentRole,
    Trellis2DecoderConfig,
    Trellis2FlowConfig,
    Trellis2FlowRole,
    Trellis2ModelPlan,
    Trellis2SplitModelPlan,
    Trellis2VisionPlan,
    load_safetensors_header,
    plan_trellis2_artifact,
    plan_trellis2_decoder_component,
    plan_trellis2_flow_artifact,
    plan_trellis2_model,
    plan_trellis2_vision,
    trellis2_artifact_runtime_identity,
)
from dinkster_inference.devices import DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .dinov3 import DINOv3ViTModel
from .naf import NAF
from .operations import CastOperations, Operations
from .trellis2_flow import Trellis2FlowModel
from .trellis2_vae import (
    Trellis2ShapeVae,
    Trellis2SparseDecoder,
    Trellis2StructureDecoder,
    Trellis2TextureVae,
)


class Trellis2FlowBundle(torch.nn.Module):
    """The fused four-flow profile or Microsoft's five split flows."""

    def __init__(
        self,
        structure: Trellis2FlowModel,
        shape: Trellis2FlowModel,
        shape_512: Trellis2FlowModel,
        texture: Trellis2FlowModel,
        texture_512: Trellis2FlowModel | None = None,
    ) -> None:
        super().__init__()
        self.structure = structure
        self.shape = shape
        self.shape_512 = shape_512
        self.texture = texture
        self.texture_512 = texture_512

    def forward(
        self,
        stage: str,
        latent: torch.Tensor | SparseLatent[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        projected: torch.Tensor | None = None,
        first_shape_pass: bool = False,
        low_resolution_texture: bool = False,
    ) -> torch.Tensor | SparseLatent[torch.Tensor]:
        if stage == "structure":
            model = self.structure
        elif stage == "shape":
            model = self.shape_512 if first_shape_pass else self.shape
        elif stage == "texture":
            model = (
                self.texture_512
                if low_resolution_texture and self.texture_512 is not None
                else self.texture
            )
        else:
            raise ValueError(f"unknown TRELLIS.2 flow stage {stage!r}")
        return model(latent, timestep, context, projected=projected)


@dataclass(frozen=True)
class AssembledTrellis2:
    diffusion: Trellis2FlowBundle
    plan: Trellis2ModelPlan | Trellis2SplitModelPlan
    _compute_dtype: torch.dtype = field(repr=False)
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)

    def compute_dtype(self, role: str) -> torch.dtype | None:
        return self._compute_dtype if role == "diffusion" else None


@dataclass(frozen=True)
class AssembledTrellis2Decoder:
    module: torch.nn.Module
    plan: ComponentPlan[Trellis2DecoderConfig]


@dataclass(frozen=True)
class AssembledTrellis2Vision:
    dino: DINOv3ViTModel
    naf: NAF | None
    plan: Trellis2VisionPlan


@dataclass(frozen=True)
class Trellis2LoadedArtifact:
    """One independently verified and strict-loaded TRELLIS.2 artifact."""

    role: Trellis2ArtifactRole
    family_id: str
    module: torch.nn.Module
    plan: object
    identity_components: tuple[ComponentPlan[object], ...]
    runtime_identity: str


@dataclass(frozen=True)
class Trellis2LoadedFlow:
    """One independently verified split flow artifact."""

    role: Trellis2FlowRole
    module: Trellis2FlowModel
    plan: ComponentPlan[Trellis2FlowConfig]


@dataclass(frozen=True)
class _PinnedSource:
    source: SafetensorsSource
    file: BinaryIO
    asset_digest: str
    asset_size: int

    @property
    def path(self) -> Path:
        return self.source.path

    @property
    def entries(self) -> Mapping[str, WeightEntry]:
        return self.source.entries

    def keys(self) -> tuple[str, ...]:
        return tuple(self.source.keys())

    def entry(self, key: str) -> WeightEntry:
        return self.source.entry(key)

    def metadata(self) -> Mapping[str, str]:
        return self.source.metadata()

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key)


_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


def _microsoft_split_state(
    module: Trellis2FlowModel, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    block_linear_parameters = {
        f"{layer}.{name}"
        for layer, owner in module.named_modules()
        if layer.startswith("blocks.") and isinstance(owner, torch.nn.Linear)
        for name, _parameter in owner.named_parameters(recurse=False)
    }
    return {
        key: value
        if key in block_linear_parameters or not value.is_floating_point()
        else value.float()
        for key, value in state.items()
    }


def _load_flow(
    plan: ComponentPlan[Trellis2FlowConfig],
    *,
    compute_dtype: torch.dtype,
    fp8_matmul: bool,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    microsoft_split_precision: bool = False,
) -> Trellis2FlowModel:
    mixed_precision = microsoft_split_precision and compute_dtype is torch.bfloat16

    def build(config: Trellis2FlowConfig, *, operations: Operations) -> Trellis2FlowModel:
        return Trellis2FlowModel(
            config,
            operations=operations,
            compute_dtype=compute_dtype,
            microsoft_split_precision=mixed_precision,
        )

    def transform(
        module: Trellis2FlowModel, state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not mixed_precision:
            return state
        return _microsoft_split_state(module, state)

    return _load_component(
        plan,
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
        source_file=source_file,
        source=source,
        transform=transform,
    )


def _load_vision_dino(
    plan: ComponentPlan[DINOv3ViTConfig],
    *,
    storage_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
) -> DINOv3ViTModel:
    def build(config: DINOv3ViTConfig, *, operations: Operations) -> DINOv3ViTModel:
        del operations
        return DINOv3ViTModel(
            config,
            operations=CastOperations(torch.float32),
        )

    def transform(
        _module: DINOv3ViTModel, state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(storage_dtype) if value.is_floating_point() else value
            for key, value in state.items()
        }

    return _load_component(
        plan,
        build,
        compute_dtype=storage_dtype,
        fp8_matmul=False,
        source_file=source_file,
        source=source,
        transform=transform,
    )


def _load_vision_naf(
    plan: ComponentPlan[NAFConfig],
    *,
    storage_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
) -> NAF:
    def build(config: NAFConfig, *, operations: Operations) -> NAF:
        del operations
        return NAF(
            config,
            operations=CastOperations(torch.float32),
        )

    def transform(_module: NAF, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(storage_dtype) if value.is_floating_point() else value
            for key, value in state.items()
        }

    return _load_component(
        plan,
        build,
        compute_dtype=storage_dtype,
        fp8_matmul=False,
        source_file=source_file,
        source=source,
        transform=transform,
    )


def _decoder_builder(
    plan: ComponentPlan[Trellis2DecoderConfig],
    *,
    compute_dtype: torch.dtype,
) -> Callable[..., torch.nn.Module]:
    def build(config: Trellis2DecoderConfig, *, operations: Operations) -> torch.nn.Module:
        microsoft_split_precision = compute_dtype is torch.float16
        if config.kind == "structure":
            return Trellis2StructureDecoder(
                operations=operations,
                microsoft_split_precision=microsoft_split_precision,
            )
        fused_prefix = {"shape": "shape_dec.", "texture": "txt_dec."}[config.kind]
        fused = any(key.startswith(fused_prefix) for key in plan.keys)
        if config.kind == "shape" and fused:
            return Trellis2ShapeVae(operations=operations)
        if config.kind == "texture" and fused:
            return Trellis2TextureVae(operations=operations)
        return Trellis2SparseDecoder(
            out_channels=config.out_channels,
            predict_subdivision=config.kind == "shape",
            operations=operations,
            microsoft_split_precision=microsoft_split_precision,
        )

    return build


def assemble_trellis2(
    path: Path,
    *,
    compute_dtype: torch.dtype,
    fp8_matmul: bool = False,
) -> AssembledTrellis2:
    """Plan and strict-load all four flows from one fused checkpoint."""
    plan = plan_trellis2_model(load_safetensors_header(path))
    model = Trellis2FlowBundle(
        _load_flow(plan.structure, compute_dtype=compute_dtype, fp8_matmul=fp8_matmul),
        _load_flow(plan.shape, compute_dtype=compute_dtype, fp8_matmul=fp8_matmul),
        _load_flow(plan.shape_512, compute_dtype=compute_dtype, fp8_matmul=fp8_matmul),
        _load_flow(plan.texture, compute_dtype=compute_dtype, fp8_matmul=fp8_matmul),
    )
    return AssembledTrellis2(model, plan, compute_dtype)


def assemble_trellis2_decoder(
    path: Path,
    role: Trellis2ComponentRole,
    *,
    compute_dtype: torch.dtype,
) -> AssembledTrellis2Decoder:
    """Plan and strict-load one fused or split decoder artifact."""
    if role not in ("structure-decoder", "shape-decoder", "texture-decoder"):
        raise ValueError(f"{role!r} is not a TRELLIS.2 decoder role")
    plan = plan_trellis2_decoder_component(load_safetensors_header(path), role)

    module = _load_component(
        plan,
        _decoder_builder(plan, compute_dtype=compute_dtype),
        compute_dtype=compute_dtype,
        fp8_matmul=False,
    )
    return AssembledTrellis2Decoder(module, plan)


def assemble_trellis2_vision(
    path: Path,
    *,
    compute_dtype: torch.dtype,
) -> AssembledTrellis2Vision:
    """Plan and strict-load DINOv3-L and the optional Pixal3D NAF subtree."""
    plan = plan_trellis2_vision(load_safetensors_header(path))
    dino = _load_vision_dino(
        plan.dino,
        storage_dtype=compute_dtype,
    )
    naf = (
        None
        if plan.naf is None
        else _load_vision_naf(
            plan.naf,
            storage_dtype=compute_dtype,
        )
    )
    return AssembledTrellis2Vision(dino, naf, plan)


def load_trellis2_artifact(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Trellis2ArtifactRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Trellis2LoadedArtifact:
    """Verify, plan, identity-check, and strict-load one TRELLIS.2 artifact."""

    if type(asset) is not AssetRef:
        raise TypeError("TRELLIS.2 artifact asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("TRELLIS.2 artifact requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("TRELLIS.2 compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Trellis2AssemblyError(
            f"TRELLIS.2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Trellis2AssemblyError(
                f"TRELLIS.2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Trellis2AssemblyError(
                f"TRELLIS.2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_trellis2_artifact(
            cast("SafetensorsSource", pinned), role=expected_role, path=path
        )
        runtime_identity = trellis2_artifact_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise Trellis2AssemblyError(
                f"expected artifact identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        source_view = cast("SafetensorsSource", pinned)
        if expected_role == "diffusion":
            plan = cast("Trellis2ModelPlan", planned.plan)
            module: torch.nn.Module = Trellis2FlowBundle(
                *(
                    _load_flow(
                        component,
                        compute_dtype=compute_dtype,
                        fp8_matmul=False,
                        source_file=handle,
                        source=source_view,
                    )
                    for component in plan.identity_components
                )
            )
        elif expected_role == "vision":
            vision = cast("Trellis2VisionPlan", planned.plan)
            dino = _load_vision_dino(
                vision.dino,
                storage_dtype=compute_dtype,
                source_file=handle,
                source=source_view,
            )
            naf = (
                None
                if vision.naf is None
                else _load_vision_naf(
                    vision.naf,
                    storage_dtype=compute_dtype,
                    source_file=handle,
                    source=source_view,
                )
            )
            module = Trellis2VisionModule(dino, naf)
            plan = vision
        else:
            decoder = cast("ComponentPlan[Trellis2DecoderConfig]", planned.plan)
            module = _load_component(
                decoder,
                _decoder_builder(decoder, compute_dtype=compute_dtype),
                compute_dtype=compute_dtype,
                fp8_matmul=False,
                source_file=handle,
                source=source_view,
            )
            plan = decoder
    return Trellis2LoadedArtifact(
        expected_role,
        planned.family_id,
        module,
        plan,
        planned.identity_components,
        runtime_identity,
    )


def load_trellis2_flow_artifact(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Trellis2FlowRole,
    compute_dtype: torch.dtype,
) -> Trellis2LoadedFlow:
    """Verify and strict-load one Microsoft split flow artifact."""

    if type(asset) is not AssetRef:
        raise TypeError("TRELLIS.2 flow artifact asset must be an AssetRef")
    if compute_dtype not in _IDENTITY_DTYPES:
        raise TypeError("TRELLIS.2 compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Trellis2AssemblyError(
            f"TRELLIS.2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Trellis2AssemblyError(
                f"TRELLIS.2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Trellis2AssemblyError(
                f"TRELLIS.2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        plan = cast(
            "ComponentPlan[Trellis2FlowConfig]",
            plan_trellis2_flow_artifact(
                cast("SafetensorsSource", pinned),
                role=expected_role,
                path=path,
            ),
        )
        module = _load_flow(
            plan,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=cast("SafetensorsSource", pinned),
            microsoft_split_precision=True,
        )
    return Trellis2LoadedFlow(expected_role, module, plan)


class Trellis2VisionModule(torch.nn.Module):
    def __init__(self, dino: DINOv3ViTModel, naf: NAF | None) -> None:
        super().__init__()
        self.dino = dino
        self.naf = naf


__all__ = [
    "AssembledTrellis2",
    "AssembledTrellis2Decoder",
    "AssembledTrellis2Vision",
    "Trellis2FlowBundle",
    "Trellis2LoadedArtifact",
    "Trellis2LoadedFlow",
    "Trellis2VisionModule",
    "assemble_trellis2",
    "assemble_trellis2_decoder",
    "assemble_trellis2_vision",
    "load_trellis2_artifact",
    "load_trellis2_flow_artifact",
]
