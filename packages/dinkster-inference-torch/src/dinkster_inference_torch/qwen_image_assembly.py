"""Strict direct-import assembly for the official split base Qwen Image artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    QwenImageComponentRole,
    plan_qwen_image_official_component,
    qwen_image_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.catalog import QWEN_IMAGE
from dinkster_inference.devices import BFLOAT16, FLOAT8_E4M3, FLOAT16, FLOAT32, DType
from dinkster_inference.quantization import LayerQuant
from dinkster_inference.qwen_image import QWEN_IMAGE_CONFIG, QwenImageConfig
from dinkster_inference.qwen_image_layout import (
    QwenImageDiTAssemblyError,
    plan_qwen_image_dit_assembly,
    qwen_image_dit_layout,
)
from dinkster_inference.qwen_image_text import (
    QWEN_IMAGE_TEXT_CONFIG,
    QwenImageTextConfig,
    qwen_image_text_layout,
)
from dinkster_inference.sources import (
    SafetensorsSource,
    load_safetensors_header,
    load_safetensors_header_from_file,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry, WeightSource

from .assemble import AssembledQwenImage, _load_component  # pyright: ignore[reportPrivateUsage]
from .operations import Operations
from .qwen_image import QwenImage
from .qwen_image_text import QwenImageTextModel
from .wan21_vae import WanVAE, WanVAEConfig

_PROVIDER_REVISION = "46839d338df81ce625d5fae27d7e370314c0fbc9"
_ROLES = ("qwen-image-dit", "qwen2.5-vl-7b-text", "wan21-vae")
_HEADER_SHA256 = MappingProxyType(
    {
        "qwen-image-dit": "9356eb06d3b193fa894c2921ad8f61b19bb87823d54b0f57e22bdd76bc3a5b9f",
        "qwen2.5-vl-7b-text": "c4e6e0abbd46c2216857d21eaef3e85ed56553e9ddd3103dbbeff243b8a38d73",
        "wan21-vae": "5fcff35e07ec3899a69d23ddec25bc5ec29c092d2902a66c0135bf36f3ec7cc5",
    }
)
_TEXT_EXTRAS = ("lm_head.weight", "scaled_fp8")
_SUPPORTED_COMPUTE_DTYPES = (torch.bfloat16, torch.float32)
_VAE_COMPUTE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_IDENTITY_DTYPES = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


class QwenImageSplitAssemblyError(ValueError):
    """An input does not match the reviewed official split assembly contract."""


@dataclass(frozen=True)
class QwenImageSplitArtifactReceipt:
    """The immutable header identities and local paths verified by the receipt gate."""

    provider_revision: str
    paths: Mapping[str, Path]
    header_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        paths_obj = cast("object", self.paths)
        digests_obj = cast("object", self.header_sha256)
        if not isinstance(paths_obj, Mapping) or not isinstance(digests_obj, Mapping):
            raise TypeError("receipt paths and header digests must be mappings")
        paths = dict(cast("Mapping[str, object]", paths_obj))
        digests = dict(cast("Mapping[str, object]", digests_obj))
        if self.provider_revision != _PROVIDER_REVISION:
            raise ValueError("receipt provider revision is not the reviewed official revision")
        if tuple(sorted(paths)) != tuple(sorted(_ROLES)):
            raise ValueError("receipt paths must name exactly the three official split roles")
        if any(not isinstance(path, Path) for path in paths.values()):
            raise TypeError("receipt paths must be pathlib.Path values")
        if digests != dict(_HEADER_SHA256):
            raise ValueError("receipt header digest is not the reviewed official identity")
        if len(set(paths.values())) != len(_ROLES):
            raise ValueError("receipt paths must be distinct")
        object.__setattr__(self, "paths", MappingProxyType(cast("dict[str, Path]", paths)))
        object.__setattr__(
            self,
            "header_sha256",
            MappingProxyType(cast("dict[str, str]", digests)),
        )


@dataclass(frozen=True)
class QwenImageSplitAssemblyPlan:
    """Complete deterministic plan for the three official split components."""

    diffusion: ComponentPlan[QwenImageConfig]
    text: ComponentPlan[QwenImageTextConfig]
    vae: ComponentPlan[WanVAEConfig]
    receipt: QwenImageSplitArtifactReceipt
    claims: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        claims_obj = cast("object", self.claims)
        if not isinstance(claims_obj, Mapping):
            raise TypeError("Qwen Image split claims must be a mapping")
        claims = dict(cast("Mapping[str, object]", claims_obj))
        if tuple(sorted(claims)) != tuple(sorted(_ROLES)):
            raise ValueError("Qwen Image split claims must name exactly three roles")
        for role, values in claims.items():
            if (
                not isinstance(values, tuple)
                or any(not isinstance(value, str) for value in values)
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
            ):
                raise ValueError(f"Qwen Image {role} claims must be sorted and unique")
        components = {
            "qwen-image-dit": self.diffusion,
            "qwen2.5-vl-7b-text": self.text,
            "wan21-vae": self.vae,
        }
        expected_components = {
            "qwen-image-dit": "diffusion",
            "qwen2.5-vl-7b-text": "text",
            "wan21-vae": "vae",
        }
        for role, component in components.items():
            if component.component != expected_components[role]:
                raise ValueError(f"Qwen Image {role} component name is inconsistent")
            if component.path != self.receipt.paths[role]:
                raise QwenImageSplitAssemblyError(
                    f"Qwen Image {role} receipt path does not match its planned source"
                )
        _validate_component_plans(self.diffusion, self.text, self.vae)
        expected_claims = {
            "qwen-image-dit": tuple(
                sorted((*self.diffusion.keys.values(), *self.diffusion.ignored))
            ),
            "qwen2.5-vl-7b-text": tuple(
                sorted(
                    (
                        *self.text.keys.values(),
                        *(
                            scale
                            for quant in self.text.quant.values()
                            for scale in (quant.weight_scale, quant.input_scale)
                            if scale is not None
                        ),
                        *self.text.ignored,
                    )
                )
            ),
            "wan21-vae": tuple(sorted((*self.vae.keys.values(), *self.vae.ignored))),
        }
        if claims != expected_claims:
            raise ValueError("Qwen Image split claims do not exactly cover planned source keys")
        object.__setattr__(
            self,
            "claims",
            MappingProxyType(cast("dict[str, tuple[str, ...]]", claims)),
        )


def _source_path(source: WeightSource, role: str) -> Path:
    path = getattr(source, "path", None)
    if not isinstance(path, Path):
        raise QwenImageSplitAssemblyError(f"Qwen Image {role} source has no file path")
    return path


def _source_geometries(
    source: WeightSource, role: str
) -> tuple[tuple[str, ...], dict[str, TensorGeometry]]:
    keys = tuple(source.keys())
    if len(keys) != len(set(keys)):
        raise QwenImageSplitAssemblyError(f"duplicate Qwen Image {role} source keys")
    geometries: dict[str, TensorGeometry] = {}
    for key in keys:
        try:
            entry = source.entry(key)
        except KeyError as error:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {role} source omitted advertised key {key!r}"
            ) from error
        if entry.key != key:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {role} source returned {entry.key!r} for {key!r}"
            )
        geometries[key] = entry.geometry
    return keys, geometries


def _require_exact_keys(role: str, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    foreign = sorted(actual - expected)
    if missing:
        raise QwenImageSplitAssemblyError(
            f"missing Qwen Image {role} keys: " + ", ".join(missing[:3])
        )
    if foreign:
        raise QwenImageSplitAssemblyError(
            f"foreign Qwen Image {role} keys: " + ", ".join(foreign[:3])
        )


def _require_geometry(
    role: str,
    key: str,
    geometry: TensorGeometry,
    shape: tuple[int, ...],
    dtypes: tuple[DType, ...],
) -> None:
    if geometry.shape != shape:
        raise QwenImageSplitAssemblyError(
            f"Qwen Image {role} shape mismatch for {key}: found {geometry.shape}, expected {shape}"
        )
    if geometry.dtype not in dtypes:
        expected = ", ".join(dtype.name for dtype in dtypes)
        raise QwenImageSplitAssemblyError(
            f"Qwen Image {role} dtype mismatch for {key}: "
            f"found {geometry.dtype.name}, expected {expected}"
        )


@lru_cache(maxsize=1)
def _text_source_contract() -> tuple[Mapping[str, tuple[int, ...]], frozenset[str]]:
    with torch.device("meta"):
        module = QwenImageTextModel()
    shapes = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    public_layout = dict(qwen_image_text_layout())
    if shapes != public_layout:
        raise RuntimeError("Qwen Image text source state disagrees with the text layout")
    linear_weights = frozenset(
        f"{name}.weight"
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear)
    )
    if not linear_weights.issubset(shapes):
        raise RuntimeError("Qwen Image text source Linear state is incomplete")
    return MappingProxyType(shapes), linear_weights


@lru_cache(maxsize=1)
def _vae_source_contract() -> Mapping[str, tuple[int, ...]]:
    with torch.device("meta"):
        module = WanVAE()
    shapes = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    if len(shapes) != 194:
        raise RuntimeError("Wan21 source state must contain exactly 194 tensors")
    return MappingProxyType(shapes)


def _validate_component_plans(
    diffusion: ComponentPlan[QwenImageConfig],
    text: ComponentPlan[QwenImageTextConfig],
    vae: ComponentPlan[WanVAEConfig],
) -> None:
    diffusion_layout = qwen_image_dit_layout().keys
    if (
        diffusion.config != QWEN_IMAGE_CONFIG
        or dict(diffusion.keys) != {key: key for key in diffusion_layout}
        or set(diffusion.dtypes) != set(diffusion_layout)
        or any(dtype.kind != "float" for dtype in diffusion.dtypes.values())
        or diffusion.quant
        or diffusion.ignored
        or diffusion.absent
        or diffusion.transforms
    ):
        raise QwenImageSplitAssemblyError("diffusion ComponentPlan is not the exact S2 plan")

    text_layout, linear_weights = _text_source_contract()
    fp8_weights = {key for key, dtype in text.dtypes.items() if dtype == FLOAT8_E4M3}
    expected_quant = {
        key.removesuffix(".weight"): LayerQuant(
            layer=key.removesuffix(".weight"),
            format="float8_e4m3fn",
            weight=key,
            weight_scale=f"{key.removesuffix('.weight')}.scale_weight",
            input_scale=f"{key.removesuffix('.weight')}.scale_input",
        )
        for key in fp8_weights
    }
    if (
        text.config != QWEN_IMAGE_TEXT_CONFIG
        or dict(text.keys) != {key: key for key in text_layout}
        or set(text.dtypes) != set(text_layout)
        or any(dtype not in (BFLOAT16, FLOAT8_E4M3) for dtype in text.dtypes.values())
        or len(fp8_weights) != 358
        or not fp8_weights.issubset(linear_weights)
        or dict(text.quant) != expected_quant
        or text.ignored != _TEXT_EXTRAS
        or text.absent
        or text.transforms
    ):
        raise QwenImageSplitAssemblyError(
            "text ComponentPlan is not the exact scaled-FP8 text plan"
        )

    vae_layout = _vae_source_contract()
    if (
        vae.config != WanVAEConfig()
        or dict(vae.keys) != {key: key for key in vae_layout}
        or set(vae.dtypes) != set(vae_layout)
        or any(dtype.kind != "float" for dtype in vae.dtypes.values())
        or vae.quant
        or vae.ignored
        or vae.absent
        or vae.transforms
    ):
        raise QwenImageSplitAssemblyError("VAE ComponentPlan is not the exact S5A plan")


def _plan_diffusion(source: WeightSource) -> tuple[ComponentPlan[QwenImageConfig], tuple[str, ...]]:
    try:
        source_plan = plan_qwen_image_dit_assembly(source)
    except QwenImageDiTAssemblyError as error:
        raise QwenImageSplitAssemblyError(f"Qwen Image diffusion: {error}") from error
    if source_plan.source_prefix:
        raise QwenImageSplitAssemblyError("official Qwen Image diffusion split must use bare keys")
    path = _source_path(source, "diffusion")
    return (
        ComponentPlan(
            component="diffusion",
            path=path,
            config=QWEN_IMAGE_CONFIG,
            keys=source_plan.keys,
            dtypes=source_plan.dtypes,
            quant={},
            identity_facts=(
                f"provider_revision={_PROVIDER_REVISION}",
                f"header_sha256={_HEADER_SHA256['qwen-image-dit']}",
            ),
        ),
        source_plan.claims,
    )


def _plan_text(source: WeightSource) -> tuple[ComponentPlan[QwenImageTextConfig], tuple[str, ...]]:
    source_keys, geometries = _source_geometries(source, "text")
    layout, linear_weights = _text_source_contract()
    scale_suffixes = (".scale_input", ".scale_weight")
    logical_keys = {key for key in source_keys if not key.endswith(scale_suffixes)}
    _require_exact_keys("text logical", logical_keys, set(layout) | set(_TEXT_EXTRAS))

    _require_geometry(
        "text", "lm_head.weight", geometries["lm_head.weight"], (152064, 3584), (BFLOAT16,)
    )
    _require_geometry("text", "scaled_fp8", geometries["scaled_fp8"], (0,), (FLOAT8_E4M3,))
    fp8_weights: list[str] = []
    dtypes: dict[str, DType] = {}
    for key, shape in layout.items():
        geometry = geometries[key]
        _require_geometry("text", key, geometry, shape, (BFLOAT16, FLOAT8_E4M3))
        if geometry.dtype == FLOAT8_E4M3:
            if key not in linear_weights:
                raise QwenImageSplitAssemblyError(
                    f"Qwen Image text FP8 tensor {key!r} is not a Linear weight"
                )
            fp8_weights.append(key)
        dtypes[key] = geometry.dtype
    if len(fp8_weights) != 358:
        raise QwenImageSplitAssemblyError(
            f"official Qwen Image text must carry 358 scaled FP8 Linear weights, "
            f"found {len(fp8_weights)}"
        )

    expected_scales = {
        f"{key.removesuffix('.weight')}.{suffix}"
        for key in fp8_weights
        for suffix in ("scale_input", "scale_weight")
    }
    actual_scales = set(source_keys) - logical_keys
    missing_scales = sorted(expected_scales - actual_scales)
    foreign_scales = sorted(actual_scales - expected_scales)
    if missing_scales:
        raise QwenImageSplitAssemblyError(
            "missing Qwen Image text scale companions: " + ", ".join(missing_scales[:3])
        )
    if foreign_scales:
        raise QwenImageSplitAssemblyError(
            "foreign Qwen Image text scale companions: " + ", ".join(foreign_scales[:3])
        )
    for key in expected_scales:
        geometry = geometries[key]
        if geometry.shape != () or geometry.dtype != FLOAT32:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image text scale {key!r} must be scalar float32"
            )

    quant = {
        key.removesuffix(".weight"): LayerQuant(
            layer=key.removesuffix(".weight"),
            format="float8_e4m3fn",
            weight=key,
            weight_scale=f"{key.removesuffix('.weight')}.scale_weight",
            input_scale=f"{key.removesuffix('.weight')}.scale_input",
        )
        for key in fp8_weights
    }
    return (
        ComponentPlan(
            component="text",
            path=_source_path(source, "text"),
            config=QWEN_IMAGE_TEXT_CONFIG,
            keys={key: key for key in layout},
            dtypes=dtypes,
            quant=quant,
            ignored=_TEXT_EXTRAS,
            identity_facts=(
                f"provider_revision={_PROVIDER_REVISION}",
                f"header_sha256={_HEADER_SHA256['qwen2.5-vl-7b-text']}",
            ),
        ),
        tuple(sorted(source_keys)),
    )


def _plan_vae(source: WeightSource) -> tuple[ComponentPlan[WanVAEConfig], tuple[str, ...]]:
    source_keys, geometries = _source_geometries(source, "vae")
    layout = _vae_source_contract()
    _require_exact_keys("VAE", set(source_keys), set(layout))
    for key, shape in layout.items():
        if geometries[key].shape != shape or geometries[key].dtype.kind != "float":
            raise QwenImageSplitAssemblyError(
                f"Qwen Image VAE geometry mismatch for {key}: found {geometries[key]}"
            )
    return (
        ComponentPlan(
            component="vae",
            path=_source_path(source, "VAE"),
            config=WanVAEConfig(),
            keys={key: key for key in layout},
            dtypes={key: geometries[key].dtype for key in layout},
            quant={},
            identity_facts=(
                f"provider_revision={_PROVIDER_REVISION}",
                f"header_sha256={_HEADER_SHA256['wan21-vae']}",
            ),
        ),
        tuple(sorted(source_keys)),
    )


def plan_qwen_image_split_assembly(
    *,
    diffusion: WeightSource,
    text: WeightSource,
    vae: WeightSource,
    receipt: QwenImageSplitArtifactReceipt,
) -> QwenImageSplitAssemblyPlan:
    """Plan the exact official split set from headers without reading payloads."""

    if not isinstance(cast("object", receipt), QwenImageSplitArtifactReceipt):
        raise TypeError("receipt must be QwenImageSplitArtifactReceipt")
    source_paths = {
        "qwen-image-dit": _source_path(diffusion, "diffusion"),
        "qwen2.5-vl-7b-text": _source_path(text, "text"),
        "wan21-vae": _source_path(vae, "VAE"),
    }
    for role, path in source_paths.items():
        if receipt.paths[role] != path:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {role} receipt path does not match the source path"
            )
    diffusion_plan, diffusion_claims = _plan_diffusion(diffusion)
    text_plan, text_claims = _plan_text(text)
    vae_plan, vae_claims = _plan_vae(vae)
    for role, source in (
        ("qwen-image-dit", diffusion),
        ("qwen2.5-vl-7b-text", text),
        ("wan21-vae", vae),
    ):
        if _canonical_header_digest(source) != receipt.header_sha256[role]:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {role} header digest does not match the receipt"
            )
    return QwenImageSplitAssemblyPlan(
        diffusion=diffusion_plan,
        text=text_plan,
        vae=vae_plan,
        receipt=receipt,
        claims={
            "qwen-image-dit": diffusion_claims,
            "qwen2.5-vl-7b-text": text_claims,
            "wan21-vae": vae_claims,
        },
    )


def _require_compute_dtype(name: str, dtype: torch.dtype) -> None:
    if dtype not in _SUPPORTED_COMPUTE_DTYPES:
        raise TypeError(f"Qwen Image {name} compute dtype must be torch.bfloat16 or torch.float32")


def _require_vae_compute_dtype(dtype: torch.dtype) -> None:
    if dtype not in _VAE_COMPUTE_DTYPES:
        raise TypeError("Qwen Image VAE compute dtype must be bfloat16, float16, or float32")


def _build_qwen_image_text(
    config: QwenImageTextConfig, *, operations: Operations
) -> QwenImageTextModel:
    if config != QWEN_IMAGE_TEXT_CONFIG:
        raise QwenImageSplitAssemblyError("text builder requires the exact text config")
    return QwenImageTextModel(operations=operations)


def _canonical_header_digest(source: WeightSource) -> str:
    dtype_codes = {
        BFLOAT16: "BF16",
        FLOAT32: "F32",
        FLOAT8_E4M3: "F8_E4M3",
    }
    canonical: list[tuple[str, str, tuple[int, ...]]] = []
    for key in sorted(source.keys()):
        geometry = source.entry(key).geometry
        try:
            dtype = dtype_codes[geometry.dtype]
        except KeyError as error:
            raise QwenImageSplitAssemblyError(
                f"unsupported dtype in Qwen Image receipt header: {geometry.dtype.name}"
            ) from error
        canonical.append((key, dtype, geometry.shape))
    encoded = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _preflight_current_headers(plan: QwenImageSplitAssemblyPlan) -> None:
    try:
        sources = {
            "qwen-image-dit": load_safetensors_header(plan.diffusion.path),
            "qwen2.5-vl-7b-text": load_safetensors_header(plan.text.path),
            "wan21-vae": load_safetensors_header(plan.vae.path),
        }
        for role, source in sources.items():
            if _canonical_header_digest(source) != plan.receipt.header_sha256[role]:
                raise QwenImageSplitAssemblyError(
                    f"current Qwen Image {role} header digest does not match the receipt"
                )
        current = plan_qwen_image_split_assembly(
            diffusion=sources["qwen-image-dit"],
            text=sources["qwen2.5-vl-7b-text"],
            vae=sources["wan21-vae"],
            receipt=plan.receipt,
        )
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, QwenImageSplitAssemblyError):
            raise
        raise QwenImageSplitAssemblyError(
            f"current Qwen Image split headers failed preflight: {error}"
        ) from error
    if current != plan:
        raise QwenImageSplitAssemblyError(
            "current Qwen Image split headers do not match the immutable plan"
        )


def assemble_qwen_image_split(
    plan: QwenImageSplitAssemblyPlan,
    *,
    diffusion_dtype: torch.dtype = torch.bfloat16,
    text_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    fp8_matmul: bool = False,
) -> AssembledQwenImage:
    """Strict-load the planned split components without registering a runtime."""

    if not isinstance(cast("object", plan), QwenImageSplitAssemblyPlan):
        raise TypeError("plan must be QwenImageSplitAssemblyPlan")
    for name, dtype in (
        ("diffusion", diffusion_dtype),
        ("text", text_dtype),
    ):
        _require_compute_dtype(name, dtype)
    _require_vae_compute_dtype(vae_dtype)
    if type(fp8_matmul) is not bool:
        raise TypeError("fp8_matmul must be a bool")
    _preflight_current_headers(plan)

    diffusion = _load_component(
        plan.diffusion,
        QwenImage,
        compute_dtype=diffusion_dtype,
        fp8_matmul=fp8_matmul,
    )
    text = _load_component(
        plan.text,
        _build_qwen_image_text,
        compute_dtype=text_dtype,
        fp8_matmul=fp8_matmul,
    )
    vae = _load_component(
        plan.vae,
        WanVAE,
        compute_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
    )
    return AssembledQwenImage(
        diffusion=diffusion,
        text=text,
        vae=vae,
        family=QWEN_IMAGE,
        _component_compute_dtypes={
            "diffusion": diffusion_dtype,
            "text": text_dtype,
            "vae": vae_dtype,
        },
    )


@dataclass(frozen=True)
class QwenImageLoadedComponent:
    """One independently verified and strict-loaded Qwen Image component."""

    role: QwenImageComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
    runtime_identity: str


@dataclass(frozen=True)
class _DescriptorPinnedSource:
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


def load_qwen_image_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: QwenImageComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> QwenImageLoadedComponent:
    """Verify, plan, identity-check, and strict-load one official component."""

    if type(asset) is not AssetRef:
        raise TypeError("Qwen Image component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Qwen Image component requires an expected identity")
    if expected_role == "vae":
        _require_vae_compute_dtype(compute_dtype)
    else:
        _require_compute_dtype(expected_role, compute_dtype)
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise QwenImageSplitAssemblyError(
            f"Qwen Image {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise QwenImageSplitAssemblyError(
                f"Qwen Image {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _DescriptorPinnedSource(source, handle, asset.digest, asset.size)
        plan = plan_qwen_image_official_component(pinned, role=expected_role, path=path)
        identity_dtype = _IDENTITY_DTYPES[compute_dtype]
        runtime_identity = qwen_image_component_runtime_identity(
            plan, expected_role, identity_dtype
        )
        if runtime_identity != expected_identity:
            raise QwenImageSplitAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        builder = {
            "diffusion": QwenImage,
            "qwen2_5_vl_7b": _build_qwen_image_text,
            "vae": WanVAE,
        }[expected_role]
        module = _load_component(
            plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=cast("SafetensorsSource", pinned),
        )
    return QwenImageLoadedComponent(expected_role, module, plan, runtime_identity)


__all__ = [
    "AssembledQwenImage",
    "QwenImageSplitArtifactReceipt",
    "QwenImageSplitAssemblyError",
    "QwenImageSplitAssemblyPlan",
    "QwenImageLoadedComponent",
    "assemble_qwen_image_split",
    "load_qwen_image_component",
    "plan_qwen_image_split_assembly",
]
