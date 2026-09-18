"""Tensor execution for the standard SD1.5 IP-Adapter contribution."""

from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import (
    SD15_IPADAPTER,
    SD15_IPADAPTER_SITES,
    SD15AttentionContribution,
    SD15IPAdapterConfig,
)

from .attention import AttentionKernel
from .operations import INITLESS, Operations


class SD15IPAdapterResourceError(RuntimeError):
    """An adapter resource no longer matches its immutable binding."""


class _ImageProjection(torch.nn.Module):
    def __init__(self, config: SD15IPAdapterConfig, *, operations: Operations) -> None:
        super().__init__()
        self.config = config
        self.proj = operations.linear(
            config.clip_embedding_dim,
            config.token_count * config.token_dim,
        )
        self.norm = operations.layer_norm(config.token_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.config.clip_embedding_dim:
            raise ValueError(
                f"IP-Adapter image embedding must be [batch x {self.config.clip_embedding_dim}]"
            )
        tokens = self.proj(embedding).reshape(
            embedding.shape[0], self.config.token_count, self.config.token_dim
        )
        return self.norm(tokens)


class _SiteProjection(torch.nn.Module):
    def __init__(self, width: int, token_dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.to_k_ip = operations.linear(token_dim, width, bias=False)
        self.to_v_ip = operations.linear(token_dim, width, bias=False)


class SD15IPAdapter(torch.nn.Module):
    """Image projection and all 16 normal residency-owned K/V projections."""

    def __init__(
        self,
        config: SD15IPAdapterConfig = SD15_IPADAPTER,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        self._resource_seal: _Seal | None = None
        self.image_proj = _ImageProjection(config, operations=operations)
        self.ip_adapter = torch.nn.ModuleDict(
            {
                str(site.adapter_index): _SiteProjection(
                    site.width,
                    config.token_dim,
                    operations=operations,
                )
                for site in config.sites
            }
        )

    @property
    def resource_digest(self) -> str | None:
        seal = self._resource_seal
        return None if seal is None else seal.digest

    def project_image_embedding(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape != (1, self.config.clip_embedding_dim):
            raise ValueError("standard SD1.5 IP-Adapter accepts exactly one reference image")
        parameter = self.image_proj.proj.weight
        cond_source = embedding.to(device=parameter.device, dtype=parameter.dtype)
        cond = self.image_proj(cond_source)
        uncond = self.image_proj(torch.zeros_like(cond_source))
        return cond, uncond

    def attention(
        self,
        site_id: str,
        q: torch.Tensor,
        tokens: torch.Tensor,
        kernel: AttentionKernel,
    ) -> torch.Tensor:
        by_site = {site.id: site for site in self.config.sites}
        try:
            site = by_site[site_id]
        except KeyError:
            raise ValueError(f"unknown SD1.5 IP-Adapter site {site_id!r}") from None
        if q.ndim != 4 or q.shape[1] * q.shape[3] != site.width:
            raise ValueError(
                f"IP-Adapter site {site_id!r} query width "
                f"{q.shape[1] * q.shape[3]} does not match {site.width}"
            )
        if tokens.ndim != 3 or tokens.shape[0] != q.shape[0] or tokens.shape[1:] != (4, 768):
            raise ValueError("IP-Adapter projected tokens do not match the fused query batch")
        projection = cast("_SiteProjection", self.ip_adapter[str(site.adapter_index)])
        keys = projection.to_k_ip(tokens)
        values = projection.to_v_ip(tokens)
        batch, token_count, _ = keys.shape

        def heads_first(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, token_count, q.shape[1], q.shape[3]).transpose(1, 2)

        return kernel(q, heads_first(keys), heads_first(values))


@dataclass(frozen=True)
class _SealedTensor:
    name: str
    tensor: torch.Tensor
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    storage: torch.UntypedStorage
    stride: tuple[int, ...]
    storage_offset: int
    conjugated: bool
    negated: bool


@dataclass(frozen=True)
class _Seal:
    digest: str
    tensors: tuple[_SealedTensor, ...]


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("IP-Adapter resource tensors must track mutation versions") from error


def _sealed_tensor(name: str, tensor: torch.Tensor) -> _SealedTensor:
    return _SealedTensor(
        name,
        tensor,
        _tensor_version(tensor),
        tensor.dtype,
        tuple(tensor.shape),
        tensor.untyped_storage(),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _matches_seal(
    model: SD15IPAdapter,
    name: str,
    tensor: torch.Tensor,
    expected: _SealedTensor,
) -> bool:
    from .module_residency import (
        _residency_assignment_generation,  # pyright: ignore[reportPrivateUsage]
        _residency_assignment_version,  # pyright: ignore[reportPrivateUsage]
    )

    if (
        name != expected.name
        or tensor.dtype != expected.dtype
        or tuple(tensor.shape) != expected.shape
        or tuple(tensor.stride()) != expected.stride
        or tensor.storage_offset() != expected.storage_offset
        or tensor.is_conj() != expected.conjugated
        or tensor.is_neg() != expected.negated
    ):
        return False
    if (
        tensor is expected.tensor
        and tensor.untyped_storage() is expected.storage
        and _tensor_version(tensor) == expected.version
    ):
        return True
    if _residency_assignment_generation(model, name, tensor) is None:
        return False
    authorized_version = _residency_assignment_version(model, name, tensor)
    try:
        current_version = _tensor_version(tensor)
    except ValueError:
        return tensor.is_inference() and authorized_version is None
    return authorized_version == current_version


def sd15_ipadapter_resource_digest(asset_digest: str, dtype: torch.dtype) -> str:
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("IP-Adapter source digest must be a canonical blake3 asset digest")
    if not dtype.is_floating_point:
        raise TypeError("IP-Adapter compute dtype must be floating")
    return blake3(f"dinkster.sd15_ipadapter.v1\0{asset_digest}\0{dtype}".encode()).hexdigest()


def _bind_sd15_ipadapter_resource(  # pyright: ignore[reportUnusedFunction]
    model: SD15IPAdapter, digest: str
) -> None:
    if type(model) is not SD15IPAdapter:
        raise TypeError("IP-Adapter binding requires an exact SD15IPAdapter")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("IP-Adapter resource digest must be a lowercase BLAKE3 digest")
    if model._resource_seal is not None:  # pyright: ignore[reportPrivateUsage]
        raise ValueError("IP-Adapter resource is already bound")
    model._resource_seal = _Seal(  # pyright: ignore[reportPrivateUsage]
        digest,
        tuple(_sealed_tensor(name, tensor) for name, tensor in model.named_parameters()),
    )


def validate_sd15_ipadapter_resource(model: SD15IPAdapter, digest: str) -> None:
    seal = model._resource_seal  # pyright: ignore[reportPrivateUsage]
    if seal is None or seal.digest != digest:
        raise SD15IPAdapterResourceError("IP-Adapter resource binding does not match")
    current = tuple(model.named_parameters())
    if len(current) != len(seal.tensors):
        raise SD15IPAdapterResourceError("IP-Adapter parameter inventory changed")
    for (name, tensor), expected in zip(current, seal.tensors, strict=True):
        if not _matches_seal(model, name, tensor, expected):
            raise SD15IPAdapterResourceError(f"IP-Adapter parameter {expected.name!r} changed")


def sd15_ipadapter_tensor_digest(value: torch.Tensor, *, role: str) -> str:
    if type(value) is not torch.Tensor or value.layout is not torch.strided:
        raise TypeError("IP-Adapter payload must be an exact strided torch.Tensor")
    tensor = value.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(
        tensor.data_ptr(), tensor.numel() * tensor.element_size()
    )
    width = tensor.element_size()
    if sys.byteorder == "big" and width > 1:
        raw = bytearray(raw)
        for start in range(0, len(raw), width):
            raw[start : start + width] = reversed(raw[start : start + width])
    hasher = blake3()
    hasher.update(b"dinkster.sd15-ipadapter-payload.v1\n")
    hasher.update(f"role={role}\n".encode("ascii"))
    hasher.update(f"shape={','.join(str(dim) for dim in tensor.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(tensor.dtype).removeprefix('torch.')}\n\n".encode("ascii"))
    hasher.update(raw)
    return hasher.hexdigest()


@dataclass(frozen=True)
class SD15IPAdapterConditioning:
    """One validated tensor binding for a portable contribution declaration."""

    declaration: SD15AttentionContribution
    model: SD15IPAdapter
    cond_tokens: torch.Tensor
    uncond_tokens: torch.Tensor
    model_digest: str
    token_digest: str
    mask: torch.Tensor | None = None
    mask_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.declaration) is not SD15AttentionContribution:
            raise TypeError("IP-Adapter declaration must be exact")
        if type(self.model) is not SD15IPAdapter:
            raise TypeError("IP-Adapter conditioning requires an exact SD15IPAdapter")
        validate_sd15_ipadapter_resource(self.model, self.model_digest)
        cond = self.cond_tokens.detach().clone()
        uncond = self.uncond_tokens.detach().clone()
        if cond.shape != (1, 4, 768) or uncond.shape != (1, 4, 768):
            raise ValueError("standard IP-Adapter projected tokens must be [1 x 4 x 768]")
        if not cond.is_floating_point() or uncond.dtype != cond.dtype:
            raise TypeError("IP-Adapter projected tokens must share one floating dtype")
        joined = torch.cat((cond, uncond), dim=0)
        if sd15_ipadapter_tensor_digest(joined, role="projected-tokens") != self.token_digest:
            raise SD15IPAdapterResourceError("IP-Adapter projected-token digest does not match")
        if self.declaration.model.id != self.model_digest:
            raise SD15IPAdapterResourceError("IP-Adapter declaration model digest does not match")
        if self.declaration.projected_tokens.id != self.token_digest:
            raise SD15IPAdapterResourceError("IP-Adapter declaration token digest does not match")
        object.__setattr__(self, "cond_tokens", cond)
        object.__setattr__(self, "uncond_tokens", uncond)
        if self.mask is None:
            if self.mask_digest is not None or self.declaration.mask is not None:
                raise SD15IPAdapterResourceError("IP-Adapter mask declaration is unresolved")
        else:
            mask = self.mask.detach().clone()
            if (
                mask.ndim != 3
                or min(mask.shape) < 1
                or mask.dtype is not torch.float32
                or not bool(torch.isfinite(mask).all())
                or bool(torch.any((mask < 0.0) | (mask > 1.0)))
            ):
                raise ValueError("IP-Adapter mask must be float32 [batch x H x W] in [0, 1]")
            actual = sd15_ipadapter_tensor_digest(mask, role="output-mask")
            if self.mask_digest != actual:
                raise SD15IPAdapterResourceError("IP-Adapter mask digest does not match")
            if self.declaration.mask is None or self.declaration.mask.id != actual:
                raise SD15IPAdapterResourceError("IP-Adapter mask declaration does not match")
            object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class SD15IPAdapterExecution:
    conditioning: SD15IPAdapterConditioning
    sigma_start: float
    sigma_end: float

    def __post_init__(self) -> None:
        if type(self.conditioning) is not SD15IPAdapterConditioning:
            raise TypeError("IP-Adapter execution requires exact conditioning")
        if (
            type(self.sigma_start) is not float
            or type(self.sigma_end) is not float
            or not math.isfinite(self.sigma_start)
            or not math.isfinite(self.sigma_end)
            or self.sigma_start < self.sigma_end
        ):
            raise ValueError("IP-Adapter sigma bounds must be finite and descending")

    def active(self, sigma: float) -> bool:
        return sigma <= self.sigma_start and sigma >= self.sigma_end


def sd15_ipadapter_identity_facts(
    contributions: tuple[SD15IPAdapterConditioning, ...],
) -> tuple[str, ...]:
    facts: list[str] = []
    for index, contribution in enumerate(contributions):
        declaration = contribution.declaration
        prefix = f"ipadapter[{index}]"
        facts.extend(
            (
                f"{prefix}.model={contribution.model_digest}",
                f"{prefix}.tokens={contribution.token_digest}",
                f"{prefix}.window={declaration.window.start_percent.hex()},"
                f"{declaration.window.end_percent.hex()}",
                f"{prefix}.scalar={declaration.scalar_gain.hex()}",
                f"{prefix}.sites="
                + ",".join(
                    f"{site.id}:{site.width}:{site.adapter_index}:"
                    f"{declaration.site_gains[site_index].hex()}"
                    for site_index, site in enumerate(declaration.sites)
                ),
                f"{prefix}.lanes="
                + ",".join(f"{lane}:{gain.hex()}" for lane, gain in declaration.lane_gains),
                f"{prefix}.lane_sources="
                + ",".join(f"{lane}:{source}" for lane, source in declaration.lane_sources),
                f"{prefix}.mask={contribution.mask_digest or '-'}",
                f"{prefix}.placement={declaration.placement}",
            )
        )
    return tuple(facts)


@dataclass(frozen=True)
class _ActiveContribution:
    conditioning: SD15IPAdapterConditioning
    tokens: torch.Tensor


@dataclass(frozen=True)
class SD15AttentionExecutionContext:
    """Active contributions expanded to the denoiser's actual fused lane rows."""

    contributions: tuple[_ActiveContribution, ...]
    lane_ids: tuple[str, ...]
    batch: int
    latent_height: int
    latent_width: int

    @classmethod
    def for_sigma(
        cls,
        executions: tuple[SD15IPAdapterExecution, ...],
        sigma: float,
        lane_ids: tuple[str, ...],
        batch: int,
        *,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SD15AttentionExecutionContext | None:
        active = tuple(execution for execution in executions if execution.active(sigma))
        if not active:
            return None
        if batch <= 0 or not lane_ids or len(lane_ids) != len(set(lane_ids)):
            raise ValueError("IP-Adapter fused lane layout must be non-empty and unique")
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError("IP-Adapter latent spatial dimensions must be positive")
        allowed = {"positive", "negative", "empty"}
        unknown = set(lane_ids) - allowed
        if unknown:
            raise ValueError(
                "IP-Adapter does not define guidance lanes: " + ", ".join(sorted(unknown))
            )
        contributions: list[_ActiveContribution] = []
        for execution in active:
            conditioning = execution.conditioning
            validate_sd15_ipadapter_resource(conditioning.model, conditioning.model_digest)
            parts = []
            lane_sources = dict(conditioning.declaration.lane_sources)
            for lane_id in lane_ids:
                source = (
                    conditioning.cond_tokens
                    if lane_sources[lane_id] == "conditional"
                    else conditioning.uncond_tokens
                )
                parts.append(source.to(device=device, dtype=dtype).repeat(batch, 1, 1))
            contributions.append(_ActiveContribution(conditioning, torch.cat(parts, dim=0)))
        return cls(tuple(contributions), lane_ids, batch, latent_height, latent_width)

    def apply(
        self,
        site_id: str,
        q: torch.Tensor,
        out: torch.Tensor,
        *,
        height: int,
        width: int,
        kernel: AttentionKernel,
    ) -> torch.Tensor:
        site_index = next(
            (index for index, site in enumerate(SD15_IPADAPTER_SITES) if site.id == site_id),
            None,
        )
        if site_index is None:
            raise ValueError(f"unknown SD1.5 IP-Adapter site {site_id!r}")
        if q.shape != out.shape or q.shape[0] != self.batch * len(self.lane_ids):
            raise ValueError("IP-Adapter attention rows do not match the fused denoiser layout")
        for active in self.contributions:
            conditioning = active.conditioning
            addition = conditioning.model.attention(site_id, q, active.tokens, kernel)
            lane_gains = dict(conditioning.declaration.lane_gains)
            gains = torch.tensor(
                [lane_gains[lane] for lane in self.lane_ids],
                device=q.device,
                dtype=q.dtype,
            ).repeat_interleave(self.batch)
            gain = (
                conditioning.declaration.scalar_gain
                * conditioning.declaration.site_gains[site_index]
            )
            addition = addition * gains.view(-1, 1, 1, 1) * gain
            if conditioning.mask is not None:
                sequence_length = q.shape[2]
                mask_height_float = self.latent_height / math.sqrt(
                    self.latent_height * self.latent_width / sequence_length
                )
                mask_height = int(mask_height_float)
                mask_height += int(sequence_length % mask_height != 0)
                mask_width = sequence_length // mask_height
                mask = F.interpolate(
                    conditioning.mask.unsqueeze(1).to(device=q.device, dtype=q.dtype),
                    size=(mask_height, mask_width),
                    mode="bilinear",
                ).squeeze(1)
                if mask.shape[0] < self.batch:
                    mask = torch.cat(
                        (mask, mask[-1:].repeat(self.batch - mask.shape[0], 1, 1)), dim=0
                    )
                elif mask.shape[0] > self.batch:
                    mask = mask[: self.batch]
                mask = mask.repeat(len(self.lane_ids), 1, 1).reshape(q.shape[0], 1, -1, 1)
                mask_length = mask_height * mask_width
                if mask_length < sequence_length:
                    padding = sequence_length - mask_length
                    mask = F.pad(mask, (0, 0, padding // 2, padding - padding // 2))
                elif mask_length > sequence_length:
                    crop_start = (mask_length - sequence_length) // 2
                    mask = mask[:, :, crop_start : crop_start + sequence_length, :]
                addition = addition * mask
            out = out + addition
        return out


__all__ = [
    "SD15AttentionExecutionContext",
    "SD15IPAdapter",
    "SD15IPAdapterConditioning",
    "SD15IPAdapterExecution",
    "SD15IPAdapterResourceError",
    "sd15_ipadapter_resource_digest",
    "sd15_ipadapter_identity_facts",
    "sd15_ipadapter_tensor_digest",
    "validate_sd15_ipadapter_resource",
]
