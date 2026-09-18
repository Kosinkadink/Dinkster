"""Native Qwen Image InstantX and Fun ControlNet execution."""

from __future__ import annotations

import ctypes
import math
import sys
import weakref
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import ControlApplication
from dinkster_inference.qwen_image import QWEN_IMAGE_CONFIG, QwenImageConfig
from dinkster_inference.qwen_image_control import QwenImageControlKind, QwenImageDiffSynthKind

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations
from .qwen_image import (
    QwenImage,
    QwenImageTransformerBlock,
)

_DEFAULT_QWEN_IMAGE_CONTROL_ATTENTION = select_attention("qwen").kernel


def _target_transformer_inputs(
    model: QwenImage,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    image, image_ids, padded_shape = model.pack_image(x)
    batch = x.shape[0]
    text_start = round(max(padded_shape[-1] // 4, padded_shape[-2] // 4))
    text_ids = (
        torch.arange(text_start, text_start + context.shape[1], device=x.device)
        .reshape(1, -1, 1)
        .expand(batch, -1, 3)
    )
    frequencies = (
        model.pe_embedder(torch.cat((text_ids, image_ids), dim=1)).to(dtype=x.dtype).contiguous()
    )
    image = model.img_in(image)
    text = model.txt_in(model.txt_norm(context))
    temb = model.time_text_embed(timesteps, image)
    mask = model._joint_mask(  # pyright: ignore[reportPrivateUsage]
        attention_mask,
        image.shape[1],
        dtype=image.dtype,
        device=image.device,
    )
    return image, text, temb, frequencies, mask


class QwenImageInstantXControlNet(QwenImage):
    """InstantX Qwen ControlNet producing one residual per base block."""

    def __init__(
        self,
        config: QwenImageConfig = QWEN_IMAGE_CONFIG,
        *,
        extra_condition_channels: int = 0,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_IMAGE_CONTROL_ATTENTION,
    ) -> None:
        super().__init__(config, operations=operations, attention_kernel=attention_kernel)
        if extra_condition_channels < 0 or extra_condition_channels % 4:
            raise ValueError("InstantX extra condition channels must be non-negative and packed")
        del self.norm_out
        del self.proj_out
        self.controlnet_blocks = torch.nn.ModuleList(
            operations.linear(config.hidden_width, config.hidden_width)
            for _ in range(config.transformer_blocks)
        )
        self.controlnet_x_embedder = operations.linear(
            config.patchified_input_channels + extra_condition_channels,
            config.hidden_width,
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        hint: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        self._validate_inputs(x, timesteps, context, attention_mask, None)
        expected_hint_channels = self.controlnet_x_embedder.in_features // 4
        if (
            hint.ndim != 5
            or hint.shape[0] != x.shape[0]
            or hint.shape[1] != expected_hint_channels
            or hint.shape[2:] != x.shape[2:]
            or hint.dtype != x.dtype
            or hint.device != x.device
        ):
            raise ValueError(
                "InstantX control hint must match the latent with the checkpoint's"
                " condition channels"
            )
        image, text, temb, frequencies, mask = _target_transformer_inputs(
            self, x, timesteps, context, attention_mask
        )
        hint_tokens, _, _ = self.pack_image(hint)
        image = image + self.controlnet_x_embedder(hint_tokens)
        repeat = math.ceil(self.config.transformer_blocks / len(self.controlnet_blocks))
        residuals: list[torch.Tensor] = []
        prefetch = make_prefetch_queue(self.transformer_blocks)
        try:
            for block, projection in zip(
                self.transformer_blocks, self.controlnet_blocks, strict=True
            ):
                prefetch_queue_pop(prefetch, block)
                text, image = block(image, text, temb, frequencies, mask)
                residuals.extend((projection(image),) * repeat)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        return tuple(residuals[: self.config.transformer_blocks])


class QwenImageFunControlBlock(QwenImageTransformerBlock):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        has_before_projection: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            dim,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.before_proj: torch.nn.Linear | None = (
            operations.linear(dim, dim) if has_before_projection else None
        )
        self.after_proj = operations.linear(dim, dim)


class QwenImageFunControlNet(torch.nn.Module):
    """Five-block Qwen Fun control path using the active base model projections."""

    injection_layers = (0, 12, 24, 36, 48)

    def __init__(
        self,
        config: QwenImageConfig = QWEN_IMAGE_CONFIG,
        *,
        control_in_features: int = 132,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_IMAGE_CONTROL_ATTENTION,
    ) -> None:
        super().__init__()
        if control_in_features < 1:
            raise ValueError("Qwen Fun control input width must be positive")
        self.config = config
        self.control_img_in = operations.linear(control_in_features, config.hidden_width)
        self.control_blocks = torch.nn.ModuleList(
            QwenImageFunControlBlock(
                config.hidden_width,
                config.attention_heads,
                config.attention_head_dim,
                has_before_projection=index == 0,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for index in range(len(self.injection_layers))
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def _pack_hint(self, hint: torch.Tensor) -> torch.Tensor:
        if hint.ndim == 4:
            hint = hint.unsqueeze(2)
        expected_channels = self.control_img_in.in_features // 4
        if hint.ndim != 5 or hint.shape[1] < 1:
            raise ValueError("Qwen Fun control hint must have rank 4 or 5")
        if hint.shape[1] == 16 and expected_channels == 33:
            hint = torch.cat((hint, torch.zeros_like(hint[:, :1]), torch.zeros_like(hint)), dim=1)
        batch, channels, temporal, height, width = hint.shape
        hint = F.pad(hint, (0, width % 2, 0, height % 2))
        packed = (
            hint.view(batch, channels, temporal, hint.shape[-2] // 2, 2, hint.shape[-1] // 2, 2)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(batch, -1, channels * 4)
        )
        expected = self.control_img_in.in_features
        if packed.shape[-1] < expected:
            packed = F.pad(packed, (0, expected - packed.shape[-1]))
        return packed[..., :expected]

    def forward(
        self,
        base_model: QwenImage,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        hint: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if type(base_model) is not QwenImage or base_model.config != self.config:
            raise TypeError("Qwen Fun control requires the matching exact QwenImage base model")
        base_model._validate_inputs(  # pyright: ignore[reportPrivateUsage]
            x, timesteps, context, None, None
        )
        hint_tokens = self._pack_hint(hint)
        image_tokens, _, _ = base_model.pack_image(x)
        token_count = min(image_tokens.shape[1], hint_tokens.shape[1])
        if token_count < 1:
            raise ValueError("Qwen Fun control and target must have at least one token")
        image_tokens = image_tokens[:, :token_count]
        hint_tokens = hint_tokens[:, :token_count]

        image, text, temb, frequencies, _ = _target_transformer_inputs(
            base_model, x, timesteps, context, None
        )
        image = image[:, :token_count]
        frequencies = torch.cat(
            (
                frequencies[:, :, : context.shape[1]],
                frequencies[:, :, context.shape[1] : context.shape[1] + token_count],
            ),
            dim=2,
        )
        control = self.control_img_in(hint_tokens)
        residuals: list[torch.Tensor | None] = [None] * self.config.transformer_blocks
        for index, module in enumerate(self.control_blocks):
            block = cast(QwenImageFunControlBlock, module)
            before = block.before_proj
            control_input = before(control) + image if before is not None else control
            text, control = block(control_input, text, temb, frequencies, None)
            residuals[self.injection_layers[index]] = block.after_proj(control)
        return tuple(residuals)


class QwenImageDiffSynthBlock(torch.nn.Module):
    """One exact DiffSynth blockwise image-token correction."""

    def __init__(self, dim: int = 3072, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.x_rms = operations.rms_norm(dim, eps=1e-6)
        self.y_rms = operations.rms_norm(dim, eps=1e-6)
        self.input_proj = operations.linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.linear(dim, dim)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.output_proj(
            self.act(self.input_proj(self.x_rms(image) + self.y_rms(condition)))
        )


class QwenImageDiffSynthPatch(torch.nn.Module):
    """All 60 blockwise DiffSynth corrections for Qwen Image."""

    def __init__(
        self,
        *,
        input_features: int = 64,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        if input_features not in (64, 68):
            raise ValueError("Qwen Image DiffSynth input width must be 64 or 68")
        self.img_in = operations.linear(input_features, 3072)
        self.controlnet_blocks = torch.nn.ModuleList(
            QwenImageDiffSynthBlock(operations=operations) for _ in range(60)
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def prepare_condition(self, latent: torch.Tensor) -> torch.Tensor:
        expected_channels = self.img_in.in_features // 4
        if (
            latent.ndim != 5
            or latent.shape[1] != expected_channels
            or latent.shape[2] != 1
            or not latent.is_floating_point()
            or min(latent.shape) < 1
        ):
            raise ValueError(
                "Qwen Image DiffSynth latent must be floating [batch,channels,1,height,width]"
            )
        height_pad = latent.shape[-2] % 2
        width_pad = latent.shape[-1] % 2
        if height_pad or width_pad:
            latent = F.pad(latent, (0, width_pad, 0, height_pad, 0, 0), mode="circular")
        batch, channels, _, height, width = latent.shape
        packed = (
            latent.view(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // 2) * (width // 2), channels * 4)
        )
        return self.img_in(packed)

    def apply_block(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        block_index: int,
        strength: float,
    ) -> torch.Tensor:
        if type(block_index) is not int or not 0 <= block_index < len(self.controlnet_blocks):
            raise ValueError("Qwen Image DiffSynth block index is out of range")
        if (
            condition.ndim != 3
            or condition.shape[0] != image.shape[0]
            or condition.shape[1] > image.shape[1]
            or condition.shape[2] != image.shape[2]
            or condition.dtype != image.dtype
            or condition.device != image.device
        ):
            raise ValueError("Qwen Image DiffSynth condition must match the image-token prefix")
        count = condition.shape[1]
        correction = self.controlnet_blocks[block_index](image[:, :count], condition) * strength
        return torch.cat((image[:, :count] + correction, image[:, count:]), dim=1)


QwenImageControlModel = QwenImageInstantXControlNet | QwenImageFunControlNet


class QwenImageControlBindingError(ValueError):
    """A Qwen Image control resource does not match its declared identity."""


@dataclass(frozen=True, slots=True)
class _TensorSeal:
    name: str
    tensor: torch.Tensor
    storage: torch.UntypedStorage
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    conjugated: bool
    negated: bool


@dataclass(frozen=True, slots=True)
class _ResourceSeal:
    digest: str
    tensors: tuple[_TensorSeal, ...]


_RESOURCE_SEALS: weakref.WeakKeyDictionary[torch.nn.Module, _ResourceSeal] = (
    weakref.WeakKeyDictionary()
)


def _resource_tensors(model: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", tensor) for name, tensor in model.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in model.named_buffers()),
    )


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("Qwen Image control tensors must track mutation versions") from error


def qwen_image_control_resource_digest(
    asset_digest: str,
    kind: QwenImageControlKind,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for one assembled maintained Qwen Image control artifact."""
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("Qwen Image control source digest must be canonical blake3 identity")
    if kind not in ("instantx", "instantx_inpaint", "fun"):
        raise ValueError("Qwen Image control kind must be maintained")
    if not compute_dtype.is_floating_point:
        raise TypeError("Qwen Image control compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.qwen-image-control-resource.v1\n")
    hasher.update(f"asset={asset_digest}\n".encode("ascii"))
    hasher.update(f"kind={kind}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def qwen_image_diffsynth_resource_digest(
    asset_digest: str,
    kind: QwenImageDiffSynthKind,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for one assembled maintained Qwen Image DiffSynth patch."""
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("Qwen Image DiffSynth source digest must be canonical blake3 identity")
    if kind not in ("diffsynth", "diffsynth_inpaint"):
        raise ValueError("Qwen Image DiffSynth kind must be maintained")
    if not compute_dtype.is_floating_point:
        raise TypeError("Qwen Image DiffSynth compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.qwen-image-diffsynth-resource.v1\n")
    hasher.update(f"asset={asset_digest}\n".encode("ascii"))
    hasher.update(f"kind={kind}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def qwen_image_control_hint_digest(hint: torch.Tensor) -> str:
    """Content identity for one materialized Qwen Image control hint."""
    if type(hint) is not torch.Tensor or hint.layout is not torch.strided:
        raise TypeError("Qwen Image control hint must be an exact strided tensor")
    value = hint.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(
        value.data_ptr(), value.numel() * value.element_size()
    )
    width = value.element_size()
    if sys.byteorder == "big" and width > 1:
        raw = bytearray(raw)
        for start in range(0, len(raw), width):
            raw[start : start + width] = reversed(raw[start : start + width])
    hasher = blake3()
    hasher.update(b"dinkster.qwen-image-control-hint.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in value.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(value.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


def _bind_qwen_image_control_resource(  # pyright: ignore[reportUnusedFunction]
    model: QwenImageControlModel,
    digest: str,
) -> None:
    if type(model) not in (QwenImageInstantXControlNet, QwenImageFunControlNet):
        raise TypeError("Qwen Image control resource must be an exact maintained control model")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Qwen Image control model digest must be lowercase BLAKE3")
    if model in _RESOURCE_SEALS:
        raise ValueError("Qwen Image control resource is already bound")
    _bind_resource(model, digest)


def _bind_qwen_image_diffsynth_resource(  # pyright: ignore[reportUnusedFunction]
    model: QwenImageDiffSynthPatch,
    digest: str,
) -> None:
    if type(model) is not QwenImageDiffSynthPatch:
        raise TypeError("Qwen Image DiffSynth resource must be an exact maintained patch")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Qwen Image DiffSynth model digest must be lowercase BLAKE3")
    if model in _RESOURCE_SEALS:
        raise ValueError("Qwen Image DiffSynth resource is already bound")
    _bind_resource(model, digest)


def _bind_resource(model: torch.nn.Module, digest: str) -> None:
    _RESOURCE_SEALS[model] = _ResourceSeal(
        digest,
        tuple(
            _TensorSeal(
                name,
                tensor,
                tensor.untyped_storage(),
                _tensor_version(tensor),
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.is_conj(),
                tensor.is_neg(),
            )
            for name, tensor in _resource_tensors(model)
        ),
    )


def validate_qwen_image_control_resource(model: QwenImageControlModel, digest: str) -> None:
    """Require a Qwen Image control model to retain assembly-proven state."""
    _validate_resource(model, digest, "control-resource-binding-mismatch")


def validate_qwen_image_diffsynth_resource(model: QwenImageDiffSynthPatch, digest: str) -> None:
    """Require a DiffSynth patch to retain its assembly-proven state."""
    if type(model) is not QwenImageDiffSynthPatch:
        raise TypeError("Qwen Image DiffSynth resource must be an exact maintained patch")
    _validate_resource(model, digest, "diffsynth-resource-binding-mismatch")


def _validate_resource(model: torch.nn.Module, digest: str, error_code: str) -> None:
    from .module_residency import (
        _residency_assignment_generation,  # pyright: ignore[reportPrivateUsage]
    )

    seal = _RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            name != expected.name
            or tensor.dtype != expected.dtype
            or tuple(tensor.shape) != expected.shape
            or tuple(tensor.stride()) != expected.stride
            or tensor.storage_offset() != expected.storage_offset
            or tensor.is_conj() != expected.conjugated
            or tensor.is_neg() != expected.negated
            or not (
                (
                    tensor is expected.tensor
                    and tensor.untyped_storage() is expected.storage
                    and _tensor_version(tensor) == expected.version
                )
                or (
                    _residency_assignment_generation(model, name.partition(":")[2], tensor)
                    is not None
                    and _tensor_version(tensor) == expected.version + 1
                )
            )
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise QwenImageControlBindingError(
            f"{error_code}: Qwen Image patch provenance is absent or changed"
        )


@dataclass(frozen=True, slots=True)
class QwenImageDiffSynthExecution:
    """One prepared, identity-bound DiffSynth patch application."""

    model: QwenImageDiffSynthPatch
    condition: torch.Tensor
    strength: float
    model_digest: str

    def __post_init__(self) -> None:
        if type(self.model) is not QwenImageDiffSynthPatch:
            raise TypeError("Qwen Image DiffSynth model must be an exact maintained patch")
        if (
            type(self.condition) is not torch.Tensor
            or self.condition.layout is not torch.strided
            or self.condition.ndim != 3
            or self.condition.shape[-1] != 3072
            or not self.condition.is_floating_point()
            or min(self.condition.shape) < 1
        ):
            raise ValueError("Qwen Image DiffSynth condition must be floating [batch,tokens,3072]")
        if not math.isfinite(self.strength) or not -10.0 <= self.strength <= 10.0:
            raise ValueError("Qwen Image DiffSynth strength must be finite and within [-10, 10]")
        validate_qwen_image_diffsynth_resource(self.model, self.model_digest)

    def __call__(self, image: torch.Tensor, block_index: int) -> torch.Tensor:
        return self.model.apply_block(image, self.condition, block_index, self.strength)


@dataclass(frozen=True, slots=True)
class QwenImageDiffSynthConditioning:
    """One identity-bound source image for a DiffSynth block patch."""

    model: QwenImageDiffSynthPatch
    kind: QwenImageDiffSynthKind
    content: torch.Tensor
    mask: torch.Tensor | None
    strength: float
    model_digest: str
    content_digest: str
    mask_digest: str | None

    def __post_init__(self) -> None:
        if type(self.model) is not QwenImageDiffSynthPatch:
            raise TypeError("Qwen Image DiffSynth model must be an exact maintained patch")
        if self.kind not in ("diffsynth", "diffsynth_inpaint"):
            raise ValueError("Qwen Image DiffSynth kind must be maintained")
        expected_features = 64 if self.kind == "diffsynth" else 68
        if self.model.img_in.in_features != expected_features:
            raise ValueError("Qwen Image DiffSynth model input width does not match its kind")
        if (
            type(self.content) is not torch.Tensor
            or self.content.layout is not torch.strided
            or self.content.ndim != 4
            or self.content.shape[1] != 3
            or not self.content.is_floating_point()
            or min(self.content.shape) < 1
        ):
            raise ValueError("Qwen Image DiffSynth content must be floating [batch,3,height,width]")
        if self.mask is not None and (
            type(self.mask) is not torch.Tensor
            or self.mask.layout is not torch.strided
            or self.mask.ndim not in (3, 4)
            or (self.mask.ndim == 4 and self.mask.shape[1] != 1)
            or self.mask.shape[0] != self.content.shape[0]
            or not self.mask.is_floating_point()
            or min(self.mask.shape) < 1
        ):
            raise ValueError("Qwen Image DiffSynth mask must be floating [batch,height,width]")
        if not math.isfinite(self.strength) or not -10.0 <= self.strength <= 10.0:
            raise ValueError("Qwen Image DiffSynth strength must be finite and within [-10, 10]")
        validate_qwen_image_diffsynth_resource(self.model, self.model_digest)
        if qwen_image_control_hint_digest(self.content) != self.content_digest:
            raise QwenImageControlBindingError("Qwen Image DiffSynth content identity changed")
        expected_mask_digest = (
            None if self.mask is None else qwen_image_control_hint_digest(self.mask)
        )
        if self.mask_digest != expected_mask_digest:
            raise QwenImageControlBindingError("Qwen Image DiffSynth mask identity changed")


def snapshot_qwen_image_diffsynth_conditioning(
    conditioning: QwenImageDiffSynthConditioning,
) -> QwenImageDiffSynthConditioning:
    """Validate then own DiffSynth image inputs for one sampling execution."""
    if type(conditioning) is not QwenImageDiffSynthConditioning:
        raise TypeError("Qwen Image DiffSynth input must be exact conditioning")
    return QwenImageDiffSynthConditioning(
        conditioning.model,
        conditioning.kind,
        conditioning.content.detach().clone(),
        None if conditioning.mask is None else conditioning.mask.detach().clone(),
        conditioning.strength,
        conditioning.model_digest,
        conditioning.content_digest,
        conditioning.mask_digest,
    )


@dataclass(frozen=True, slots=True)
class QwenImageControlConditioning:
    """One identity-bound maintained Qwen Image control application."""

    application: ControlApplication
    model: QwenImageControlModel
    kind: QwenImageControlKind
    hint: torch.Tensor
    model_digest: str
    hint_digest: str

    def __post_init__(self) -> None:
        if type(self.application) is not ControlApplication:
            raise TypeError("control application must be an exact ControlApplication")
        if self.kind not in ("instantx", "instantx_inpaint", "fun"):
            raise ValueError("Qwen Image control kind must be maintained")
        expected_type = (
            QwenImageFunControlNet if self.kind == "fun" else QwenImageInstantXControlNet
        )
        if type(self.model) is not expected_type:
            raise TypeError("Qwen Image control model must match its declared kind")
        if type(self.hint) is not torch.Tensor or self.hint.layout is not torch.strided:
            raise TypeError("Qwen Image control hint must be an exact strided tensor")
        if self.hint.ndim not in ((4, 5) if self.kind == "fun" else (5,)):
            raise ValueError("Qwen Image control hint rank does not match its control kind")
        expected_channels = {"instantx": (16,), "instantx_inpaint": (17,), "fun": (16, 33)}[
            self.kind
        ]
        if self.hint.shape[1] not in expected_channels:
            raise ValueError("Qwen Image control hint channels do not match its control kind")
        if self.kind != "fun":
            assert type(self.model) is QwenImageInstantXControlNet
            expected_features = 64 if self.kind == "instantx" else 68
            if self.model.controlnet_x_embedder.in_features != expected_features:
                raise ValueError("InstantX control model input width does not match its kind")
        if not self.hint.is_floating_point() or min(self.hint.shape) < 1:
            raise ValueError("Qwen Image control hint must be positive and floating")
        if self.application.previous is not None:
            raise ValueError("Qwen Image runtime currently admits one control application")
        if self.application.mode is not None:
            raise ValueError("Qwen Image control does not accept an SD control mode")
        validate_qwen_image_control_resource(self.model, self.model_digest)
        if self.application.hint.id != self.hint_digest:
            raise QwenImageControlBindingError("Qwen Image control hint reference identity changed")
        if qwen_image_control_hint_digest(self.hint) != self.hint_digest:
            raise QwenImageControlBindingError("Qwen Image control hint content changed")


def snapshot_qwen_image_control_conditioning(
    conditioning: QwenImageControlConditioning,
) -> QwenImageControlConditioning:
    """Validate then own the hint tensor for one sampling execution."""
    if type(conditioning) is not QwenImageControlConditioning:
        raise TypeError("Qwen Image control must be exact QwenImageControlConditioning")
    return QwenImageControlConditioning(
        conditioning.application,
        conditioning.model,
        conditioning.kind,
        conditioning.hint.detach().clone(),
        conditioning.model_digest,
        conditioning.hint_digest,
    )


__all__ = [
    "QwenImageControlBindingError",
    "QwenImageControlConditioning",
    "QwenImageControlModel",
    "QwenImageDiffSynthBlock",
    "QwenImageDiffSynthConditioning",
    "QwenImageDiffSynthExecution",
    "QwenImageDiffSynthPatch",
    "QwenImageFunControlBlock",
    "QwenImageFunControlNet",
    "QwenImageInstantXControlNet",
    "qwen_image_control_hint_digest",
    "qwen_image_control_resource_digest",
    "qwen_image_diffsynth_resource_digest",
    "snapshot_qwen_image_control_conditioning",
    "snapshot_qwen_image_diffsynth_conditioning",
    "validate_qwen_image_control_resource",
    "validate_qwen_image_diffsynth_resource",
]
