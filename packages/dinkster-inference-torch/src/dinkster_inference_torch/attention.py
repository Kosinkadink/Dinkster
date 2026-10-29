"""Typed attention-kernel selection over built-in SDPA and provider adapters.

This module owns the rank-4 kernel contract, the provider adapters, and the
selection evidence consumed by model call sites.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from threading import RLock
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from dinkster_inference import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    AttentionRuntimeStatus,
    automatic_attention_route,
    canonical_attention_route_token_bytes,
    current_realized_sampling_row,
    derive_attention_route_token,
    resolve_attention_runtime_status,
    resolve_role_policy,
)
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32
from dinkster_inference.partition_compatibility import (
    ContiguousShard,
    PartitionCompatibility,
    Replicated,
    UlyssesHeadScatter,
)

from .dtype_policy import force_fp16_attention_upcast
from .memory import get_free_memory, soft_empty_cache

logger = logging.getLogger(__name__)

AttentionRole = Literal["unet", "flux", "vae", "clip", "t5", "qwen"]
AttentionPolicy = Literal[
    "auto", "sdpa", "flash", "xformers", "sage", "sage3", "sol", "dinkster_kitchen_int8"
]

_ROLES: frozenset[str] = frozenset(("unet", "flux", "vae", "clip", "t5", "qwen"))
_POLICIES: frozenset[str] = frozenset(
    ("auto", "sdpa", "flash", "xformers", "sage", "sage3", "sol", "dinkster_kitchen_int8")
)
_OPTIONAL_POLICIES: frozenset[str] = frozenset(("flash", "xformers", "sage3"))

# PyTorch SDPA has an NVIDIA batch-axis limit at 2^15 in the audited
# ComfyUI path. Chunking every device is semantically equivalent and avoids
# import-time device probing.
SDP_BATCH_LIMIT = 2**15
SDP_PRIORITY_MIN_ELEMENTS = 1024 * 128
ATTENTION_ADAPTER_CONTRACT = "dinkster.attention-kernel.v2"
BUILTIN_SDPA_PROVIDER = "torch-sdpa-priority-v1"
BOUNDED_ATTENTION_PROVIDER = "torch-bounded-attention-v1"
COMFY_KITCHEN_INT8_PROVIDER = "dinkster-kitchen-int8-attention-v1"
SOL_ATTENTION_PROVIDER = "dinkster-kitchen-sol-attention-v1"
SAGE2_PROVIDER = "sageattention2-int8-v1"
_SOL_BLOCK_SIZE = 64
_SOL_CONDITIONING_MODIFIERS = frozenset(("sol_conditioning_exact_kv",))


def _sdpa_priority_support() -> tuple[Any | None, tuple[Any, ...]]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
            raise TypeError("torch SDPA priority is unavailable")
        return sdpa_kernel, (
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        )
    except (AttributeError, ImportError, TypeError, ValueError):
        return None, ()


_SDPA_KERNEL_FACTORY, _SDPA_BACKEND_PRIORITY = _sdpa_priority_support()
_SDPA_PRIORITY_ACTIVE: ContextVar[bool] = ContextVar("dinkster_sdpa_priority_active", default=False)
_SDPA_PRIORITY_LOCK = RLock()
_sdpa_cuda_priority_initialized = False


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _get_sdpa_state() -> tuple[list[int], tuple[bool, bool, bool, bool]]:
    return (
        torch._C._get_sdp_priority_order(),  # pyright: ignore[reportPrivateUsage]
        (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        ),
    )


def _set_sdpa_state(
    priority: list[int],
    enabled: tuple[bool, bool, bool, bool],
) -> None:
    torch.backends.cuda.enable_flash_sdp(enabled[0])
    torch.backends.cuda.enable_cudnn_sdp(enabled[1])
    torch.backends.cuda.enable_mem_efficient_sdp(enabled[2])
    torch.backends.cuda.enable_math_sdp(enabled[3])
    torch._C._set_sdp_priority_order(priority)  # pyright: ignore[reportPrivateUsage]


def _encode_sdpa_state() -> torch.Tensor:
    priority, enabled = _get_sdpa_state()
    padded_priority = [*priority, *([-1] * (5 - len(priority)))]
    return torch.tensor(
        [len(priority), *padded_priority, *(int(value) for value in enabled)],
        dtype=torch.int8,
    )


def _decode_sdpa_state(
    encoded: torch.Tensor,
) -> tuple[list[int], tuple[bool, bool, bool, bool]]:
    values = encoded.tolist()
    length = values[0]
    enabled = cast(tuple[bool, bool, bool, bool], tuple(bool(value) for value in values[6:10]))
    return values[1 : length + 1], enabled


@torch.library.custom_op("dinkster_inference_torch::locked_sdpa_v2", mutates_args=())
def _locked_sdpa_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a compiled small SDPA call under the process-global priority lock."""
    with _SDPA_PRIORITY_LOCK:
        state = _encode_sdpa_state()
        output = _run_sdpa(q, k, v, mask, causal, scale, enable_gqa).contiguous()
        return output, state


@_locked_sdpa_v2.register_fake
def _locked_sdpa_v2_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = _run_sdpa(q, k, v, mask, causal, scale, enable_gqa).contiguous()
    return output, torch.empty((10,), dtype=torch.int8, device="cpu")


@torch.library.custom_op("dinkster_inference_torch::locked_sdpa_backward_v2", mutates_args=())
def _locked_sdpa_backward_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    state: torch.Tensor,
    grad_output: torch.Tensor,
    differentiate_mask: bool,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> list[torch.Tensor]:
    with _SDPA_PRIORITY_LOCK:
        current_state = _get_sdpa_state()
        _set_sdpa_state(*_decode_sdpa_state(state))
        try:
            if differentiate_mask:
                assert mask is not None and mask.is_floating_point()

                def run_with_mask(
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    attention_mask: torch.Tensor,
                ) -> torch.Tensor:
                    return _run_sdpa(
                        query,
                        key,
                        value,
                        attention_mask,
                        causal,
                        scale,
                        enable_gqa,
                    )

                _, vjp = cast(Any, torch.func.vjp)(
                    run_with_mask,
                    q,
                    k,
                    v,
                    mask,
                )
                gradients = vjp(grad_output)
            else:

                def run_without_mask(
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                ) -> torch.Tensor:
                    return _run_sdpa(
                        query,
                        key,
                        value,
                        mask,
                        causal,
                        scale,
                        enable_gqa,
                    )

                _, vjp = cast(Any, torch.func.vjp)(
                    run_without_mask,
                    q,
                    k,
                    v,
                )
                gradients = (*vjp(grad_output), q.new_empty((0,)))
        finally:
            _set_sdpa_state(*current_state)
    return [gradient.contiguous() for gradient in gradients]


@_locked_sdpa_backward_v2.register_fake
def _locked_sdpa_backward_v2_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    state: torch.Tensor,
    grad_output: torch.Tensor,
    differentiate_mask: bool,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> list[torch.Tensor]:
    del state, grad_output, causal, scale, enable_gqa
    mask_gradient = (
        torch.empty_like(mask, memory_format=torch.contiguous_format)
        if differentiate_mask and mask is not None and mask.is_floating_point()
        else q.new_empty((0,))
    )
    return [
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
        mask_gradient,
    ]


def _locked_sdpa_v2_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k, v, mask, causal, scale, enable_gqa = inputs
    ctx.has_mask = mask is not None
    ctx.differentiable_mask = mask is not None and mask.requires_grad
    ctx.save_for_backward(q, k, v, *((mask,) if mask is not None else ()), output[1])
    ctx.causal = causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa


def _locked_sdpa_v2_autograd(
    context: Any,
    grad_output: torch.Tensor,
    grad_state: torch.Tensor | None,
) -> tuple[Any, ...]:
    del grad_state
    saved = context.saved_tensors
    q, k, v = saved[:3]
    mask = saved[3] if context.has_mask else None
    state = saved[4] if context.has_mask else saved[3]
    gradients = _locked_sdpa_backward_v2(
        q,
        k,
        v,
        mask,
        state,
        grad_output,
        context.differentiable_mask,
        context.causal,
        context.scale,
        context.enable_gqa,
    )
    return (
        gradients[0],
        gradients[1],
        gradients[2],
        gradients[3] if context.differentiable_mask else None,
        None,
        None,
        None,
    )


_locked_sdpa_v2.register_autograd(
    _locked_sdpa_v2_autograd,
    setup_context=_locked_sdpa_v2_setup_context,
)


def _apply_sdpa_autocast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device_type = q.device.type
    if not torch.is_autocast_enabled(device_type):
        return q, k, v, mask
    dtype = torch.get_autocast_dtype(device_type)

    def cast_input(value: torch.Tensor | None) -> torch.Tensor | None:
        if (
            value is None
            or not value.is_floating_point()
            or value.device.type != device_type
            or value.dtype is torch.float64
        ):
            return value
        return value.to(dtype)

    return (
        cast(torch.Tensor, cast_input(q)),
        cast(torch.Tensor, cast_input(k)),
        cast(torch.Tensor, cast_input(v)),
        cast_input(mask),
    )


def _initialize_cuda_sdpa_priority(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> None:
    """Initialize Torch's CUDA chooser and reapply Dinkster's active priority."""
    global _sdpa_cuda_priority_initialized  # noqa: PLW0603
    if q.device.type != "cuda" or _sdpa_cuda_priority_initialized:
        return
    with _SDPA_PRIORITY_LOCK:
        if _sdpa_cuda_priority_initialized:
            return
        torch._fused_sdp_choice(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        priority = [int(backend) for backend in _SDPA_BACKEND_PRIORITY]
        priority.extend(
            backend
            for backend in torch._C._get_sdp_priority_order()  # pyright: ignore[reportPrivateUsage]
            if backend not in priority
        )
        torch._C._set_sdp_priority_order(priority)  # pyright: ignore[reportPrivateUsage]
        _sdpa_cuda_priority_initialized = True


@contextmanager
def _sdpa_priority_context() -> Generator[None, None, None]:
    assert _SDPA_KERNEL_FACTORY is not None
    # Torch stores the priority order process-wide, so snapshot, selection,
    # dispatch, and restoration must not overlap across worker threads.
    with _SDPA_PRIORITY_LOCK:
        priority_order = torch._C._get_sdp_priority_order()  # pyright: ignore[reportPrivateUsage]
        try:
            with _SDPA_KERNEL_FACTORY(list(_SDPA_BACKEND_PRIORITY), set_priority=True):
                yield
        finally:
            # sdpa_kernel cannot reconstruct an order when the caller's enabled
            # backend list is empty, so restore the complete snapshot directly.
            torch._C._set_sdp_priority_order(  # pyright: ignore[reportPrivateUsage]
                priority_order
            )


def _enable_fp16_bf16_reduction_math_sdp(setter: Callable[[bool], object] | None = None) -> bool:
    """Let SDPA's math backend accumulate fp16/bf16 in reduced precision.

    The reference enables this global-context flag at import
    (model_management @ b78cec87), so its math-backend fallbacks - the
    invocations every fused backend refuses - run faster reduced-precision
    reductions. Mirroring it keeps those fallback values and speed aligned
    with the reference's executed arithmetic; invocations served by a fused
    backend are unaffected. Tolerates torch builds without the knob, like
    the reference. Returns whether the flag was applied."""
    try:
        apply = torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp if setter is None else setter
        apply(True)
        return True
    except (AttributeError, RuntimeError, TypeError):
        return False


_FP16_BF16_REDUCTION_MATH_SDP = _enable_fp16_bf16_reduction_math_sdp()
# Host fact probed once at import, like the reference's module-level
# FORCE_UPCAST_ATTENTION_DTYPE; tests substitute it to exercise both sides.
_FORCE_FP16_ATTENTION_UPCAST = force_fp16_attention_upcast()


def _hip_runtime_version(torch_module: object = torch) -> str | None:
    """The HIP runtime version of a ROCm torch build, or None on other builds."""
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    return hip if isinstance(hip, str) and hip else None


def _priority_context_eligible(device: torch.device, query_elements: int) -> bool:
    """Whether the built-in SDPA backend priority list may wrap this call.

    XPU is excluded: the priority list names the CUDA-oriented backends and
    omits OVERRIDEABLE, the backend XPU SDPA dispatches through, so wrapping
    an XPU call would force the math backend. The reference ComfyUI never
    installs the priority wrapper on non-CUDA torch builds.
    """
    return (
        _SDPA_KERNEL_FACTORY is not None
        and query_elements >= SDP_PRIORITY_MIN_ELEMENTS
        and device.type != "xpu"
    )


def _supports_native_masked_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    causal: bool,
) -> bool:
    if q.device.type != "cuda":
        return False
    if _hip_runtime_version() is not None:
        # ROCm masked GQA always repeats k/v: the reference ComfyUI does the
        # same on non-NVIDIA, and the HIP eligibility queries have reported
        # kernels that are not actually present (ComfyUI issue #15647).
        return False
    try:
        params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, causal, True)
        return bool(
            torch.backends.cuda.can_use_flash_attention(params)
            or torch.backends.cuda.can_use_cudnn_attention(params)
            or torch.backends.cuda.can_use_efficient_attention(params)
        )
    except (AttributeError, RuntimeError, TypeError):
        return False


class AttentionError(ValueError):
    """Base class for selector construction and kernel validation errors."""


class AttentionSelectionError(AttentionError):
    """The requested role or policy cannot construct an attention kernel."""


class AttentionValidationError(AttentionError):
    """An attention invocation violates the rank-4 adapter contract."""


@runtime_checkable
class AttentionKernel(Protocol):
    """Opaque attention callable over q/k/v shaped ``[B, H, S, D]``."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class AttentionBlockResult:
    """Float32 normalized output and statistics from the same provider scores.

    Output is [B,H,Q,Dv]; maximum and exponential_sum are [B,H,Q,1].
    exponential_sum is sum(exp(score - maximum)) over valid keys. An
    entirely masked block returns zero output/mass and maximum -inf.
    Statistics reconstructed from another provider's scores do not qualify.
    """

    output: torch.Tensor
    maximum: torch.Tensor
    exponential_sum: torch.Tensor


@runtime_checkable
class AttentionBlockKernel(Protocol):
    """Explicit inference capability for normalized attention blocks.

    Inputs are rank-4 tensors with nonempty extents within the provider's
    documented numerical domain. Mask is a boolean key mask broadcastable
    to [B,H,1,K]. Output stays float32
    until blocks are merged. Implementing this interface is not numerical
    qualification or a declaration of partition compatibility.
    """

    def attention_block(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> AttentionBlockResult: ...


@runtime_checkable
class PackedAttentionFacts(Protocol):
    """Generic packed-sequence facts consumed by central attention adapters."""

    @property
    def sequence_length(self) -> int: ...

    @property
    def conditioning_prefix_length(self) -> int | None: ...


class AttentionTensorLease:
    """Single-owner handoff of one tensor into a consuming attention kernel.

    The producer wraps a tensor and drops every other reference; the kernel
    calls :meth:`take` exactly once, after which the lease holds nothing and
    the kernel owns the tensor's lifetime. This lets a kernel release the
    floating-point inputs before allocating its output.
    """

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        if not isinstance(cast("object", tensor), torch.Tensor):
            raise AttentionValidationError(
                f"attention tensor lease requires a torch.Tensor, got {type(tensor).__name__}"
            )
        self._tensor: torch.Tensor | None = tensor

    def take(self) -> torch.Tensor:
        tensor = self._tensor
        if tensor is None:
            raise AttentionValidationError("attention tensor lease was already taken")
        self._tensor = None
        return tensor


@runtime_checkable
class QkvConsumingAttentionKernel(Protocol):
    """Attention kernel that takes ownership of leased q/k/v tensors.

    ``consume`` releases the floating-point q/k/v before allocating its
    output, so callers must transfer their only references through the
    leases. Consuming kernels are inference-only.
    """

    def consume(
        self,
        q: AttentionTensorLease,
        k: AttentionTensorLease,
        v: AttentionTensorLease,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class AttentionStatus:
    """Stable evidence for one selected attention implementation."""

    requested_policy: AttentionPolicy
    role: AttentionRole
    primary: Literal["sdpa", "bounded", "sage", "sol", "dinkster_kitchen_int8"]
    fallback: Literal["sdpa", "bounded"] | None
    reason: str
    authenticated: bool
    provider_versions: tuple[tuple[str, str], ...]
    adapter_contract: str
    device_kind: str
    device_sm: int | None
    sdpa_torch_runtime: str


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """An opaque kernel paired with immutable selection evidence."""

    kernel: AttentionKernel
    status: AttentionStatus


def _validate_invocation(
    q: object,
    k: object,
    v: object,
    *,
    mask: object,
    causal: object,
    scale: object,
    enable_gqa: object,
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise AttentionValidationError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if tensor.ndim != 4:
            raise AttentionValidationError(
                f"{name} must have rank 4 [B, H, S, D], got rank {tensor.ndim}"
            )

    if mask is not None and not isinstance(mask, torch.Tensor):
        raise AttentionValidationError(
            f"mask must be a torch.Tensor or None, got {type(mask).__name__}"
        )
    if not isinstance(causal, bool):
        raise AttentionValidationError("causal must be a bool")
    if not isinstance(enable_gqa, bool):
        raise AttentionValidationError("enable_gqa must be a bool")
    if scale is not None and (
        isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)
    ):
        raise AttentionValidationError(
            "scale must be None or a finite int or float, excluding bool"
        )

    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    if not q.is_floating_point() or not k.is_floating_point() or not v.is_floating_point():
        raise AttentionValidationError("q, k, and v must use floating-point dtypes")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AttentionValidationError(
            f"q, k, and v must share one dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise AttentionValidationError(
            f"q, k, and v must share one device, got {q.device}, {k.device}, {v.device}"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AttentionValidationError("q, k, and v must share one batch size")
    if q.shape[3] != k.shape[3]:
        raise AttentionValidationError("q and k head dimensions must match")
    if k.shape[2] != v.shape[2]:
        raise AttentionValidationError("k and v sequence lengths must match")
    if k.shape[1] != v.shape[1]:
        raise AttentionValidationError("k and v head counts must match")

    query_heads = q.shape[1]
    key_heads = k.shape[1]
    if enable_gqa:
        if key_heads == 0 or query_heads % key_heads != 0:
            raise AttentionValidationError(
                "GQA requires a nonzero k/v head count that evenly divides q heads"
            )
    elif query_heads != key_heads:
        raise AttentionValidationError("q and k/v head counts must match when GQA is disabled")

    if causal and mask is not None:
        raise AttentionValidationError("causal attention cannot also receive a mask")
    if mask is None:
        return
    if mask.device != q.device:
        raise AttentionValidationError(
            f"mask must be on the q/k/v device {q.device}, got {mask.device}"
        )
    if mask.dtype != torch.bool and mask.dtype != q.dtype:
        raise AttentionValidationError(
            f"mask must be boolean or additive with dtype {q.dtype}, got {mask.dtype}"
        )
    target = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    try:
        broadcast = torch.broadcast_shapes(mask.shape, target)
    except RuntimeError as error:
        raise AttentionValidationError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {target}"
        ) from error
    if tuple(broadcast) != target:
        raise AttentionValidationError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {target}"
        )


class _SDPAKernel:
    __slots__ = ()

    partition_compatibility: ClassVar[PartitionCompatibility] = PartitionCompatibility(
        revision=1,
        modes=(Replicated(), UlyssesHeadScatter()),
        tensor_expectation=ContiguousShard(2),
        supported_dtypes=(FLOAT32, FLOAT16, BFLOAT16),
        device_kinds=("cpu", "cuda"),
        provides_matching_block_normalization=False,
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        _validate_invocation(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

        # FP16 attention produces black images on macOS 14.5+, so the
        # reference forces that math to float32 there
        # (force_upcast_attention_dtype @ b78cec87). SDPA cannot upcast
        # only the softmax stage, so the whole call runs in float32 and
        # the output returns in float16.
        upcast = _FORCE_FP16_ATTENTION_UPCAST and q.dtype == torch.float16
        if upcast:
            q = q.float()
            k = k.float()
            v = v.float()
            if mask is not None and mask.dtype == torch.float16:
                mask = mask.float()

        def run(
            chunk_q: torch.Tensor,
            chunk_k: torch.Tensor,
            chunk_v: torch.Tensor,
            chunk_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            prioritized = _priority_context_eligible(chunk_q.device, chunk_q.numel())
            run_k = chunk_k
            run_v = chunk_v
            run_gqa = enable_gqa
            if (
                prioritized
                and enable_gqa
                and chunk_mask is not None
                and chunk_q.shape[1] != chunk_k.shape[1]
                and not _supports_native_masked_gqa(
                    chunk_q,
                    chunk_k,
                    chunk_v,
                    chunk_mask,
                    causal=causal,
                )
            ):
                repeats = chunk_q.shape[1] // chunk_k.shape[1]
                run_k = chunk_k.repeat_interleave(repeats, dim=1)
                run_v = chunk_v.repeat_interleave(repeats, dim=1)
                run_gqa = False
            if not prioritized or _SDPA_PRIORITY_ACTIVE.get():
                context = nullcontext()
            else:
                context = _sdpa_priority_context()
            if torch.compiler.is_compiling() and not prioritized:
                cast_q, cast_k, cast_v, cast_mask = _apply_sdpa_autocast(
                    chunk_q, run_k, run_v, chunk_mask
                )
                return _locked_sdpa_v2(
                    cast_q,
                    cast_k,
                    cast_v,
                    cast_mask,
                    causal,
                    scale,
                    run_gqa,
                )[0]
            with _SDPA_PRIORITY_LOCK:
                with context:
                    if prioritized:
                        _initialize_cuda_sdpa_priority(
                            chunk_q,
                            run_k,
                            run_v,
                            mask=chunk_mask,
                            causal=causal,
                            scale=scale,
                            enable_gqa=run_gqa,
                        )
                    return _run_sdpa(
                        chunk_q,
                        run_k,
                        run_v,
                        chunk_mask,
                        causal,
                        scale,
                        run_gqa,
                    )

        batch = q.shape[0]
        if batch <= SDP_BATCH_LIMIT:
            out = run(q, k, v, mask)
        else:
            chunks: list[torch.Tensor] = []
            for start in range(0, batch, SDP_BATCH_LIMIT):
                stop = start + SDP_BATCH_LIMIT
                chunk_mask = mask
                if mask is not None and mask.ndim == 4 and mask.shape[0] == batch:
                    chunk_mask = mask[start:stop]
                chunks.append(run(q[start:stop], k[start:stop], v[start:stop], chunk_mask))
            out = torch.cat(chunks, dim=0)
        return out.to(torch.float16) if upcast else out


_SDPA = _SDPAKernel()

MAX_BOUNDED_ATTENTION_STEPS = 128


def _is_attention_oom(error: Exception) -> bool:
    return isinstance(error, torch.OutOfMemoryError)


def _initial_bounded_attention_steps(q: torch.Tensor, k: torch.Tensor) -> int:
    if q.shape[2] == 0 or k.shape[2] == 0:
        return 1
    score_bytes = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * max(q.element_size(), 4)
    free = max(get_free_memory(q.device).free_total, 1)
    steps = 1
    while score_bytes // steps > free and steps < MAX_BOUNDED_ATTENTION_STEPS:
        steps *= 2
    return steps


class _BoundedAttentionKernel:
    __slots__ = ()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        _validate_invocation(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        if q.shape[2] == 0:
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)

        steps = _initial_bounded_attention_steps(q, k)
        retry_same_size = True
        while True:
            chunks: list[torch.Tensor] = []
            chunk_size = (q.shape[2] + steps - 1) // steps
            try:
                for start in range(0, q.shape[2], chunk_size):
                    stop = min(start + chunk_size, q.shape[2])
                    chunk_mask = mask
                    chunk_causal = causal
                    if causal:
                        query_positions = torch.arange(start, stop, device=q.device)
                        key_positions = torch.arange(k.shape[2], device=q.device)
                        chunk_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                        chunk_causal = False
                    elif mask is not None and mask.ndim >= 2 and mask.shape[-2] == q.shape[2]:
                        chunk_mask = mask[..., start:stop, :]
                    chunks.append(
                        _SDPA(
                            q[:, :, start:stop],
                            k,
                            v,
                            mask=chunk_mask,
                            causal=chunk_causal,
                            scale=scale,
                            enable_gqa=enable_gqa,
                        )
                    )
                return torch.cat(chunks, dim=2)
            except Exception as error:
                if not _is_attention_oom(error) or chunks:
                    raise
                soft_empty_cache(q.device)
                if retry_same_size:
                    retry_same_size = False
                    continue
                steps *= 2
                if steps > MAX_BOUNDED_ATTENTION_STEPS:
                    raise


class _VaeSDPAKernel:
    __slots__ = ()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        try:
            return _SDPA(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        except Exception as error:
            if not _is_attention_oom(error):
                raise
            soft_empty_cache(q.device)
            return _BOUNDED(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )


_BOUNDED = _BoundedAttentionKernel()
_VAE_SDPA = _VaeSDPAKernel()

# Indirection points so CPU tests can substitute the capability probe and the
# kernel entry points without touching the dinkster_kitchen module itself.
_KITCHEN_UNPROBED = object()
_KITCHEN_AVAILABLE: Any = _KITCHEN_UNPROBED
_KITCHEN_ATTENTION: Any = _KITCHEN_UNPROBED
_KITCHEN_PREQUANTIZE: Any = _KITCHEN_UNPROBED
_KITCHEN_FROM_PREQUANTIZED: Any = _KITCHEN_UNPROBED
_KITCHEN_SOL_ATTENTION: Any = _KITCHEN_UNPROBED
_KITCHEN_LIST_BACKENDS: Any = _KITCHEN_UNPROBED

# dinkster-kitchen's INT8 kernel rejects head_dim > 256. The KL and Wan VAE
# families run single-head attention whose head dim is the channel axis
# (384-512), so those invocations must take the SDPA fallback.
_KITCHEN_MAX_HEAD_DIM = 256


def _load_kitchen_apis() -> None:
    global _KITCHEN_AVAILABLE
    global _KITCHEN_ATTENTION
    global _KITCHEN_PREQUANTIZE
    global _KITCHEN_FROM_PREQUANTIZED
    global _KITCHEN_SOL_ATTENTION
    global _KITCHEN_LIST_BACKENDS
    kitchen = importlib.import_module("dinkster_kitchen")
    _disable_kitchen_triton(kitchen)
    for name, attribute in (
        ("_KITCHEN_AVAILABLE", "int8_attention_is_available"),
        ("_KITCHEN_ATTENTION", "int8_attention"),
        ("_KITCHEN_PREQUANTIZE", "prequantize_int8_attention"),
        ("_KITCHEN_FROM_PREQUANTIZED", "int8_attention_from_prequantized"),
        ("_KITCHEN_SOL_ATTENTION", "sol_attn"),
        ("_KITCHEN_LIST_BACKENDS", "list_backends"),
    ):
        if globals()[name] is _KITCHEN_UNPROBED:
            globals()[name] = getattr(kitchen, attribute, None)


def _disable_kitchen_triton(kitchen: Any) -> None:
    """Keep Kitchen's Triton backend opt-in, matching ComfyUI startup."""
    disable = getattr(getattr(kitchen, "registry", None), "disable", None)
    if callable(disable):
        disable("triton")


def _missing_kitchen_int8_apis() -> tuple[str, ...]:
    _load_kitchen_apis()
    return tuple(
        name
        for name, api in (
            ("int8_attention_is_available", _KITCHEN_AVAILABLE),
            ("int8_attention", _KITCHEN_ATTENTION),
            ("prequantize_int8_attention", _KITCHEN_PREQUANTIZE),
            ("int8_attention_from_prequantized", _KITCHEN_FROM_PREQUANTIZED),
        )
        if api is None
    )


def dinkster_kitchen_int8_available() -> bool:
    """Whether the dinkster-kitchen INT8 attention kernel supports this machine.

    The attention entry points bypass dinkster-kitchen's backend registry, so
    this direct capability probe governs their availability.
    """
    if _missing_kitchen_int8_apis():
        return False
    assert _KITCHEN_AVAILABLE is not None
    return bool(_KITCHEN_AVAILABLE())


def _kitchen_invocation_supported(
    q: torch.Tensor,
    *,
    causal: bool,
) -> bool:
    if causal:
        raise AttentionValidationError("dinkster-kitchen INT8 attention has no causal path")
    if q.device.type != "cuda":
        logger.warning("dinkster-kitchen INT8 attention unavailable on %s; using SDPA", q.device)
        return False
    if torch.is_grad_enabled() and q.requires_grad:
        raise AttentionValidationError(
            "dinkster-kitchen INT8 attention is inference-only and cannot record gradients"
        )
    return True


class _ComfyKitchenInt8Kernel:
    """INT8 tensor-core attention through dinkster-kitchen.

    The borrowing call quantizes and runs in one step. ``consume`` mirrors the
    audited ComfyUI container path: prequantize q/k/v to INT8, drop the
    floating-point inputs, then run attention, so the peak allocation excludes
    the original q/k/v.

    The INT8 kernel computes bidirectional same-head-count attention over
    head dims up to 256, so causal, grouped-query, and wider-head invocations
    execute through the built-in SDPA kernel; the selection routes record
    that fallback.
    """

    __slots__ = ()

    partition_compatibility: ClassVar[PartitionCompatibility] = PartitionCompatibility(
        revision=1,
        modes=(Replicated(), UlyssesHeadScatter()),
        tensor_expectation=ContiguousShard(2),
        supported_dtypes=(FLOAT32, FLOAT16, BFLOAT16),
        device_kinds=("cuda",),
        provides_matching_block_normalization=False,
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if causal or enable_gqa:
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        _validate_invocation(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        if q.shape[-1] > _KITCHEN_MAX_HEAD_DIM:
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        if not _kitchen_invocation_supported(q, causal=causal):
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        kitchen_scale = None if scale is None else float(scale)
        assert _KITCHEN_ATTENTION is not None
        return _KITCHEN_ATTENTION(q, k, v, scale=kitchen_scale, attn_mask=mask)

    def consume(
        self,
        q: AttentionTensorLease,
        k: AttentionTensorLease,
        v: AttentionTensorLease,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # A propagating exception's traceback must not keep the leased
        # floating-point q/k/v alive: the finally block clears this frame's
        # tensor references and the except block clears the finished callee
        # frames (validation, prequantization) captured in the traceback.
        query = key = value = None
        try:
            query = q.take()
            key = k.take()
            value = v.take()
            if causal or enable_gqa:
                return _SDPA(
                    query,
                    key,
                    value,
                    mask=mask,
                    causal=causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            _validate_invocation(
                query, key, value, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa
            )
            if query.shape[-1] > _KITCHEN_MAX_HEAD_DIM:
                return _SDPA(
                    query,
                    key,
                    value,
                    mask=mask,
                    causal=causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            if not _kitchen_invocation_supported(query, causal=causal):
                return _SDPA(
                    query, key, value, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa
                )
            assert _KITCHEN_PREQUANTIZE is not None
            prequantized = _KITCHEN_PREQUANTIZE(
                query,
                key,
                value,
                scale=None if scale is None else float(scale),
                attn_mask=mask,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            del query, key, value
        assert _KITCHEN_FROM_PREQUANTIZED is not None
        return _KITCHEN_FROM_PREQUANTIZED(prequantized)


_COMFY_KITCHEN_INT8 = _ComfyKitchenInt8Kernel()


def _sol_backend_available() -> bool:
    _load_kitchen_apis()
    if _KITCHEN_SOL_ATTENTION is None or _KITCHEN_LIST_BACKENDS is None:
        return False
    backends = _KITCHEN_LIST_BACKENDS()
    if not isinstance(backends, Mapping):
        return False
    cuda = backends.get("cuda")
    if not isinstance(cuda, Mapping):
        return False
    capabilities = cuda.get("capabilities")
    return (
        cuda.get("available") is True
        and cuda.get("disabled") is False
        and isinstance(capabilities, Sequence)
        and not isinstance(capabilities, (str, bytes))
        and "sol_attn" in capabilities
    )


def _sol_device_supported(device: torch.device | None = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    cuda = cast("Any", getattr(torch, "cuda", None))
    if not _backend_available(cuda) or _hip_runtime_version() is not None:
        return False
    capability = cuda.get_device_capability(device)
    return tuple(capability) >= (8, 0)


def sol_attention_available() -> bool:
    """Whether the fused dinkster-kitchen Sol backend supports this machine."""
    return _sol_backend_available() and _sol_device_supported()


def _sol_can_execute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    causal: bool,
    enable_gqa: bool,
) -> bool:
    return (
        mask is None
        and not causal
        and not enable_gqa
        and q.numel() > 0
        and q.dtype is torch.bfloat16
        and q.shape == k.shape == v.shape
        and q.shape[-1] == 128
        and _sol_device_supported(q.device)
    )


def _validate_sol_invocation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise AttentionValidationError(
            "Sol sparse attention is inference-only and cannot record gradients"
        )


def _sol_layout(value: torch.Tensor) -> torch.Tensor:
    transposed = value.transpose(1, 2)
    if (
        transposed.stride(-1) != 1
        or transposed.data_ptr() % 16
        or any(
            transposed.shape[dimension] > 1 and transposed.stride(dimension) % 8
            for dimension in range(3)
        )
    ):
        return transposed.contiguous()
    return transposed


def _validate_sol_sink_range(value: tuple[int, int], name: str, sequence_length: int) -> None:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(boundary) is not int for boundary in value)
    ):
        raise AttentionValidationError(f"Sol {name} must be an exact integer pair")
    start, stop = value
    block_count = (sequence_length + _SOL_BLOCK_SIZE - 1) // _SOL_BLOCK_SIZE
    if not 0 <= start <= stop <= block_count:
        raise AttentionValidationError(
            f"Sol {name} must stay inside the invocation's {block_count} blocks"
        )


class _SolAttentionKernel:
    """Training-free Sol sparse attention through dinkster-kitchen.

    Sol executes equal-shape BF16 self-attention with 128-wide heads on
    NVIDIA SM80+ devices. Calls outside that contract retain their exact
    mask, causal, and GQA semantics through the built-in SDPA fallback.
    """

    __slots__ = ()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        return self.call_with_tau(
            q,
            k,
            v,
            tau=1.0,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    def call_with_tau(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tau: float,
        sink_blocks: tuple[int, int] = (0, 0),
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if type(tau) is not float or not math.isfinite(tau) or tau <= 0.0:
            raise AttentionValidationError("Sol tau must be a finite exact float greater than zero")
        _validate_invocation(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        _validate_sol_sink_range(sink_blocks, "exact-KV range", q.shape[2])
        if not _sol_can_execute(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            enable_gqa=enable_gqa,
        ):
            logger.debug("Sol attention invocation unsupported on %s; using SDPA", q.device)
            return _SDPA(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        _validate_sol_invocation(q, k, v)
        assert _KITCHEN_SOL_ATTENTION is not None
        options: dict[str, object] = {
            "tau": tau,
            "scale": None if scale is None else float(scale),
            "tail": True,
        }
        if sink_blocks != (0, 0):
            options["sink_blocks"] = list(sink_blocks)
        output = _KITCHEN_SOL_ATTENTION(_sol_layout(q), _sol_layout(k), _sol_layout(v), **options)
        return output.transpose(1, 2)


_SOL = _SolAttentionKernel()


def _import_sage_package() -> Any | None:
    try:
        import sageattention  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return sageattention


def _import_sageattn() -> Any | None:
    try:
        from sageattention import sageattn  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return sageattn


def _import_sage_core() -> Any | None:
    try:
        from sageattention import core  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return core


# Indirection points so CPU tests can substitute the optional sageattention
# distribution's kernel entry point and the device capability probe.
_SAGE_PACKAGE: Any = _import_sage_package()
_SAGE_ATTENTION: Any = _import_sageattn()
_SAGE_CORE: Any = _import_sage_core()

# Upstream sageattn dispatches one quantized kernel arm per SM and raises for
# architectures outside this set (SageAttention 2.2.0, core.py sageattn).
_SAGE2_SUPPORTED_SMS = frozenset((80, 86, 89, 90, 120))
# Upstream pads narrower heads to 64 or 128 and refuses anything wider.
_SAGE2_MAX_HEAD_DIM = 128

# Upstream guards each compiled kernel arm behind a try/except import flag,
# so sageattn can import with an arm missing and assert at call time. sm86
# runs the triton arm, which the package import itself already requires;
# sm120 dispatches into the SM89-compiled fp8 arm.
_SAGE2_REQUIRED_ARM_FLAGS: dict[int, tuple[str, ...]] = {
    80: ("SM80_ENABLED",),
    86: (),
    89: ("SM89_ENABLED",),
    90: ("SM90_ENABLED",),
    120: ("SM89_ENABLED",),
}
_MANAGED_SAGE2_DISTRIBUTION = "dinkster-kitchen"


def _sage2_arm_enabled(sm: int) -> bool:
    if _SAGE_CORE is None:
        return False
    flags = _SAGE2_REQUIRED_ARM_FLAGS.get(sm)
    if flags is None:
        return False
    return all(bool(getattr(_SAGE_CORE, flag, False)) for flag in flags)


def _sage2_device_supported() -> bool:
    cuda = cast("Any", getattr(torch, "cuda", None))
    if not _backend_available(cuda) or _hip_runtime_version() is not None:
        return False
    capability = cuda.get_device_capability()
    sm = int(capability[0]) * 10 + int(capability[1])
    return sm in _SAGE2_SUPPORTED_SMS and _sage2_arm_enabled(sm)


_SAGE_DEVICE_SUPPORTED: Callable[[], bool] = _sage2_device_supported


def sage2_distribution_version() -> str | None:
    """Return the version of the imported managed or upstream distribution."""
    distribution = (
        _MANAGED_SAGE2_DISTRIBUTION
        if getattr(_SAGE_PACKAGE, "__distribution__", None) == _MANAGED_SAGE2_DISTRIBUTION
        else "sageattention"
    )
    try:
        return _distribution_version(distribution)
    except PackageNotFoundError:
        return None


def sage2_attention_available() -> bool:
    """Whether the SageAttention 2 INT8 kernel supports this machine.

    Availability requires the optional versioned sageattention distribution
    plus a CUDA device whose SM has an upstream kernel arm.
    """
    return (
        _SAGE_ATTENTION is not None
        and sage2_distribution_version() is not None
        and _SAGE_DEVICE_SUPPORTED()
    )


def _sage2_can_execute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    causal: bool,
) -> bool:
    """Whether upstream sageattn executes this invocation exactly as asked.

    The public sageattn entry point accepts no mask, quantizes only fp16 and
    bf16 inputs, pads q/k/v jointly so v must share q's head dim, and applies
    its causal mask only to equal query/key lengths. Zero-size dims stay on
    the built-in SDPA path, which defines the adapter contract for them.
    """
    if mask is not None:
        return False
    if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] > _SAGE2_MAX_HEAD_DIM or v.shape[-1] != q.shape[-1]:
        return False
    if causal and q.shape[2] != k.shape[2]:
        return False
    return True


def _sage2_invocation_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise AttentionValidationError(
            "SageAttention 2 is inference-only and cannot record gradients"
        )
    if q.device.type != "cuda":
        logger.warning("SageAttention 2 unavailable on %s; using SDPA", q.device)
        return False
    return True


class _SageAttention2Kernel:
    """INT8-quantized attention through the optional SageAttention 2 kernels.

    Masked, non-fp16/bf16, wider-than-128-head-dim, mismatched-v-head-dim,
    and causal cross-length invocations execute through the built-in SDPA
    kernel, as do CUDA devices without an upstream kernel arm; the selection
    routes record that fallback.
    """

    __slots__ = ()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        _validate_invocation(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        if not _sage2_can_execute(q, k, v, mask=mask, causal=causal):
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        if not _sage2_invocation_supported(q, k, v):
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        capability = torch.cuda.get_device_capability(q.device)
        sm = int(capability[0]) * 10 + int(capability[1])
        if sm not in _SAGE2_SUPPORTED_SMS or not _sage2_arm_enabled(sm):
            logger.warning("SageAttention 2 kernel unavailable for capability %s; using SDPA", sm)
            return _SDPA(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
        # Upstream asserts a contiguous last dim on q, k, and v.
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()
        assert _SAGE_ATTENTION is not None
        return _SAGE_ATTENTION(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=causal,
            sm_scale=None if scale is None else float(scale),
        )


_SAGE2 = _SageAttention2Kernel()


class _ScheduledAttentionKernel:
    """Read one immutable outer-step decision at the central provider seam."""

    __slots__ = (
        "_conditioning_prefix_length",
        "_kernel",
        "_packed_modifier_scope",
        "_provider",
        "_sequence_length",
    )

    def __init__(
        self,
        provider: str,
        kernel: AttentionKernel,
        *,
        packed_modifier_scope: bool | None = None,
        sequence_length: int | None = None,
        conditioning_prefix_length: int | None = None,
    ) -> None:
        self._provider = provider
        self._kernel = kernel
        self._packed_modifier_scope = packed_modifier_scope
        self._sequence_length = sequence_length
        self._conditioning_prefix_length = conditioning_prefix_length

    def bind_packed_sequence(self, facts: PackedAttentionFacts) -> _ScheduledAttentionKernel:
        if not isinstance(cast("object", facts), PackedAttentionFacts):
            raise AttentionSelectionError("conditioning sinks require packed attention facts")
        sequence_length = facts.sequence_length
        prefix_length = facts.conditioning_prefix_length
        if (
            type(sequence_length) is not int
            or type(prefix_length) is not int
            or not 0 < prefix_length < sequence_length
        ):
            raise AttentionSelectionError(
                "conditioning sinks require a nonempty conditioning prefix before target tokens"
            )
        return type(self)(
            self._provider,
            self._kernel,
            packed_modifier_scope=True,
            sequence_length=sequence_length,
            conditioning_prefix_length=prefix_length,
        )

    def exclude_packed_modifiers(self) -> _ScheduledAttentionKernel:
        return type(self)(self._provider, self._kernel, packed_modifier_scope=False)

    def active_kernel(self) -> AttentionKernel:
        row = current_realized_sampling_row()
        if row is None:
            return self._kernel
        plan = row.attention_plan
        unsupported = tuple(
            modifier for modifier in plan.modifiers if modifier not in _SOL_CONDITIONING_MODIFIERS
        )
        if unsupported:
            raise AttentionSelectionError(
                "sampling timeline selected unsupported attention modifiers: "
                + ", ".join(unsupported)
            )
        if plan.provider == self._provider:
            return self._kernel
        if plan.provider == "sdpa":
            return _SDPA
        raise AttentionSelectionError(
            f"sampling timeline selected {plan.provider!r}, but this model's "
            f"authenticated attention provider is {self._provider!r}"
        )

    @property
    def partition_compatibility(self) -> PartitionCompatibility | None:
        return cast(
            PartitionCompatibility | None,
            getattr(self.active_kernel(), "partition_compatibility", None),
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        kernel = self.active_kernel()
        if kernel is _SOL:
            row = current_realized_sampling_row()
            tau = 1.0 if row is None else row.attention_plan.parameters.get("sol.tau", 1.0)
            sink_blocks = (0, 0)
            modifiers = frozenset() if row is None else frozenset(row.attention_plan.modifiers)
            conditioning_modifiers = modifiers.intersection(_SOL_CONDITIONING_MODIFIERS)
            if conditioning_modifiers and self._packed_modifier_scope is None:
                raise AttentionSelectionError(
                    "scheduled Sol conditioning sinks require authenticated packed layout facts"
                )
            if conditioning_modifiers and self._packed_modifier_scope:
                assert self._sequence_length is not None
                assert self._conditioning_prefix_length is not None
                if q.shape[2] != self._sequence_length:
                    raise AttentionSelectionError(
                        "scheduled Sol conditioning sink layout does not match the "
                        "attention invocation"
                    )
                sink_stop = (
                    self._conditioning_prefix_length + _SOL_BLOCK_SIZE - 1
                ) // _SOL_BLOCK_SIZE
                sink_blocks = (0, sink_stop)
            return _SOL.call_with_tau(
                q,
                k,
                v,
                tau=tau,
                sink_blocks=sink_blocks,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return kernel(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


class _ScheduledConsumingAttentionKernel(_ScheduledAttentionKernel):
    __slots__ = ()

    def consume(
        self,
        q: AttentionTensorLease,
        k: AttentionTensorLease,
        v: AttentionTensorLease,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        kernel = self.active_kernel()
        if isinstance(kernel, QkvConsumingAttentionKernel):
            return kernel.consume(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return kernel(
            q.take(),
            k.take(),
            v.take(),
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def schedule_aware_attention_kernel(provider: str, kernel: AttentionKernel) -> AttentionKernel:
    """Apply realized attention plans without changing the static provider."""
    if provider not in ("sdpa", "dinkster_kitchen_int8", "sage", "sol"):
        return kernel
    if isinstance(kernel, QkvConsumingAttentionKernel):
        return _ScheduledConsumingAttentionKernel(provider, kernel)
    return _ScheduledAttentionKernel(provider, kernel)


def exclude_packed_attention_modifiers(kernel: AttentionKernel) -> AttentionKernel:
    """Mark an attention call as outside an enclosing packed invocation."""
    if isinstance(kernel, _ScheduledAttentionKernel):
        return kernel.exclude_packed_modifiers()
    return kernel


def bind_packed_attention_kernel(
    kernel: AttentionKernel, facts: PackedAttentionFacts
) -> AttentionKernel:
    """Bind invocation-local packed facts when the realized plan needs them."""
    row = current_realized_sampling_row()
    if row is None or not _SOL_CONDITIONING_MODIFIERS.intersection(row.attention_plan.modifiers):
        return kernel
    if not isinstance(kernel, _ScheduledAttentionKernel):
        raise AttentionSelectionError(
            "scheduled conditioning sinks require the central schedule-aware attention selector"
        )
    return kernel.bind_packed_sequence(facts)


@contextmanager
def attention_kernel_context(
    kernel: AttentionKernel, query_elements: int, *, device: torch.device
) -> Generator[None, None, None]:
    """Keep built-in SDPA backend priority active across related large calls.

    The execution device is required: once this outer context activates the
    priority list, inner calls cannot undo it, so ineligible devices (XPU)
    must be excluded here.
    """
    active_kernel = (
        kernel.active_kernel() if isinstance(kernel, _ScheduledAttentionKernel) else kernel
    )
    if (
        not isinstance(active_kernel, _SDPAKernel)
        or not _priority_context_eligible(device, query_elements)
        or _SDPA_PRIORITY_ACTIVE.get()
    ):
        yield
        return

    token = _SDPA_PRIORITY_ACTIVE.set(True)
    try:
        with _sdpa_priority_context():
            yield
    finally:
        _SDPA_PRIORITY_ACTIVE.reset(token)


def builtin_sdpa_kernel() -> AttentionKernel:
    """Return the roleless built-in SDPA kernel singleton."""
    return _SDPA


def _portable_attention_selection(
    role: AttentionRole, policy: AttentionPolicy, reason: str
) -> AttentionSelection:
    reason = f"attention policy {policy!r} unavailable for role {role!r}: {reason}; using SDPA"
    logger.warning(reason)
    return AttentionSelection(
        kernel=_SDPA,
        status=replace(
            select_attention(role, "sdpa").status, requested_policy=policy, reason=reason
        ),
    )


def select_attention(role: AttentionRole, policy: AttentionPolicy = "sdpa") -> AttentionSelection:
    """Select an attention kernel for one role under one policy.

    ``auto`` resolves from local capability evidence. The named
    kitchen and Sage policies use built-in SDPA for invocations their provider
    cannot execute. Unavailable providers resolve to SDPA with a diagnostic.
    """
    if role not in _ROLES:
        raise AttentionSelectionError(
            f"unknown attention role {role!r}; expected one of {sorted(_ROLES)}"
        )
    if policy not in _POLICIES:
        raise AttentionSelectionError(
            f"unknown attention policy {policy!r}; expected one of {sorted(_POLICIES)}"
        )
    if policy == "auto":
        evidence = discover_attention_capabilities()
        route = automatic_attention_route(role, evidence.available_policies, evidence.device_kind)
        if route.primary == "bounded":
            return AttentionSelection(
                kernel=_BOUNDED,
                status=AttentionStatus(
                    requested_policy="auto",
                    role=role,
                    primary="bounded",
                    fallback=None,
                    reason="auto avoids VAE SDPA on ROCm and uses bounded attention",
                    authenticated=False,
                    provider_versions=(),
                    adapter_contract=ATTENTION_ADAPTER_CONTRACT,
                    device_kind=evidence.device_kind,
                    device_sm=evidence.device_sm,
                    sdpa_torch_runtime=evidence.sdpa_torch_runtime,
                ),
            )
        return AttentionSelection(
            kernel=_VAE_SDPA if route.fallback == "bounded" else _SDPA,
            status=AttentionStatus(
                requested_policy="auto",
                role=role,
                primary="sdpa",
                fallback=cast("Literal['bounded'] | None", route.fallback),
                reason=(
                    "auto selected built-in SDPA with bounded VAE OOM fallback"
                    if route.fallback == "bounded"
                    else "auto selected the built-in SDPA kernel"
                ),
                authenticated=False,
                provider_versions=(),
                adapter_contract=ATTENTION_ADAPTER_CONTRACT,
                device_kind=evidence.device_kind,
                device_sm=evidence.device_sm,
                sdpa_torch_runtime=evidence.sdpa_torch_runtime,
            ),
        )
    if policy in _OPTIONAL_POLICIES:
        return _portable_attention_selection(
            role, policy, f"no {policy} provider adapter is implemented"
        )

    typed_role: AttentionRole = role
    typed_policy: AttentionPolicy = policy
    if policy == "sol":
        _load_kitchen_apis()
        if _KITCHEN_SOL_ATTENTION is None or _KITCHEN_LIST_BACKENDS is None:
            return _portable_attention_selection(
                role, policy, "installed dinkster-kitchen is missing the Sol sparse attention API"
            )
        if not sol_attention_available():
            return _portable_attention_selection(
                role, policy, "the fused dinkster-kitchen Sol backend does not support this machine"
            )
        return AttentionSelection(
            kernel=_SOL,
            status=AttentionStatus(
                requested_policy=typed_policy,
                role=typed_role,
                primary="sol",
                fallback="sdpa",
                reason=(
                    "explicit training-free Sol sparse attention policy; built-in SDPA "
                    "serves masked, causal, grouped-query, non-BF16, unequal-geometry, "
                    "and non-128-head-dim invocations"
                ),
                authenticated=False,
                provider_versions=(),
                adapter_contract=ATTENTION_ADAPTER_CONTRACT,
                device_kind="unknown",
                device_sm=None,
                sdpa_torch_runtime="unknown",
            ),
        )
    if policy == "sage":
        if _SAGE_ATTENTION is None or sage2_distribution_version() is None:
            return _portable_attention_selection(
                role, policy, "a supported SageAttention distribution is not installed"
            )
        if not sage2_attention_available():
            return _portable_attention_selection(
                role, policy, "SageAttention 2 does not support this machine"
            )
        return AttentionSelection(
            kernel=_SAGE2,
            status=AttentionStatus(
                requested_policy=typed_policy,
                role=typed_role,
                primary="sage",
                fallback="sdpa",
                reason=(
                    "explicit SageAttention 2 policy; built-in SDPA serves "
                    "masked, non-fp16/bf16, wider-than-128-head-dim, and "
                    "causal cross-length invocations the quantized kernels "
                    "cannot execute"
                ),
                authenticated=False,
                provider_versions=(),
                adapter_contract=ATTENTION_ADAPTER_CONTRACT,
                device_kind="unknown",
                device_sm=None,
                sdpa_torch_runtime="unknown",
            ),
        )
    if policy == "dinkster_kitchen_int8":
        missing_apis = _missing_kitchen_int8_apis()
        if missing_apis:
            return _portable_attention_selection(
                role,
                policy,
                f"installed dinkster-kitchen is missing required APIs: {', '.join(missing_apis)}",
            )
        if not dinkster_kitchen_int8_available():
            return _portable_attention_selection(
                role, policy, "dinkster-kitchen INT8 attention does not support this machine"
            )
        return AttentionSelection(
            kernel=_COMFY_KITCHEN_INT8,
            status=AttentionStatus(
                requested_policy=typed_policy,
                role=typed_role,
                primary="dinkster_kitchen_int8",
                fallback="sdpa",
                reason=(
                    "explicit dinkster-kitchen INT8 policy; built-in SDPA serves "
                    "causal, grouped-query, and wider-than-256-head-dim "
                    "invocations the INT8 kernel cannot execute"
                ),
                authenticated=False,
                provider_versions=(),
                adapter_contract=ATTENTION_ADAPTER_CONTRACT,
                device_kind="unknown",
                device_sm=None,
                sdpa_torch_runtime="unknown",
            ),
        )
    return AttentionSelection(
        kernel=_SDPA,
        status=AttentionStatus(
            requested_policy=typed_policy,
            role=typed_role,
            primary="sdpa",
            fallback=None,
            reason="explicit built-in SDPA policy",
            authenticated=False,
            provider_versions=(),
            adapter_contract=ATTENTION_ADAPTER_CONTRACT,
            device_kind="unknown",
            device_sm=None,
            sdpa_torch_runtime="unknown",
        ),
    )


def unauthenticated_auto_attention(role: AttentionRole) -> AttentionSelection:
    selection = select_attention(role, "sdpa")
    return replace(
        selection,
        status=replace(
            selection.status,
            requested_policy="auto",
            reason="unauthenticated auto route retains built-in SDPA compatibility",
        ),
    )


def verify_attention_route_evidence(
    resolved: AttentionRuntimeStatus, token: AttentionRouteToken | None
) -> None:
    """Raise unless authenticated evidence describes this process.

    An authenticated token binds provider versions and device facts into the
    runtime identity, so it must match locally rediscovered evidence: a token
    minted on a different install or device would sign one identity while
    executing different numerics.
    """
    if not resolved.authenticated:
        return
    if resolved.adapter_contract_revision != ATTENTION_ADAPTER_CONTRACT:
        raise AttentionSelectionError(
            "authenticated attention adapter contract does not match "
            f"the local selector contract {ATTENTION_ADAPTER_CONTRACT!r}"
        )
    try:
        rediscovered = derive_attention_route_token(
            discover_attention_capabilities(),
            AttentionPolicyConfig(
                resolved.requested_policy,
                resolved.requested_role_policies,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise AttentionSelectionError(
            "authenticated attention route token cannot be derived from this process's "
            "runtime capabilities"
        ) from exc

    assert token is not None
    if canonical_attention_route_token_bytes(token) != canonical_attention_route_token_bytes(
        rediscovered
    ):
        raise AttentionSelectionError(
            "authenticated attention route token does not match this process's "
            "rediscovered runtime evidence"
        )


def resolve_role_attention(
    role: AttentionRole,
    policy: AttentionPolicy,
    token: AttentionRouteToken | None,
) -> AttentionSelection:
    """Select one role's kernel and cross-check it against authenticated evidence.

    A named policy requires an authenticated route token; the local selection
    must match the token's route for the role, and the returned status carries
    the token's authenticated evidence. A token's per-role policy override
    replaces the requested policy for its role. An override for a role a
    family never instantiates is inert; every instantiated role resolves
    through this helper, so an override on an instantiated role is honored
    or selection refuses.
    """
    resolved = resolve_attention_runtime_status(policy, token)
    verify_attention_route_evidence(resolved, token)
    effective_policy = resolve_role_policy(
        resolved.requested_policy, resolved.requested_role_policies, role
    )
    selection = (
        unauthenticated_auto_attention(role)
        if token is None and effective_policy == "auto"
        else select_attention(role, effective_policy)
    )
    status = selection.status
    routes = {route.role: route for route in resolved.routes}
    route = routes[role]
    if (status.primary, status.fallback) != (route.primary, route.fallback):
        raise AttentionSelectionError(
            f"attention selection for role {role!r} does not match the authenticated runtime route"
        )
    resolved_status = AttentionStatus(
        requested_policy=status.requested_policy,
        role=status.role,
        primary=status.primary,
        fallback=status.fallback,
        reason=status.reason,
        authenticated=resolved.authenticated,
        provider_versions=resolved.provider_versions,
        adapter_contract=(
            resolved.adapter_contract_revision
            if resolved.authenticated
            else status.adapter_contract
        ),
        device_kind=resolved.device_kind,
        device_sm=resolved.device_sm,
        sdpa_torch_runtime=resolved.sdpa_torch_runtime,
    )
    return AttentionSelection(
        kernel=(
            schedule_aware_attention_kernel(resolved_status.primary, selection.kernel)
            if role in ("unet", "flux")
            else selection.kernel
        ),
        status=resolved_status,
    )


def attention_provider_identity(status: AttentionStatus) -> tuple[str, str | None]:
    """Map one selection's primary route to its provider ID and version.

    This mapping lives with the selector so model families consume whatever
    provider the generic selection authenticated without maintaining their
    own provider lists.
    """
    if type(cast("object", status)) is not AttentionStatus:
        raise AttentionSelectionError("provider identity requires an exact AttentionStatus")
    if status.primary == "sdpa":
        return BUILTIN_SDPA_PROVIDER, None
    if status.primary == "bounded":
        return BOUNDED_ATTENTION_PROVIDER, None
    if status.primary == "sage":
        version = sage2_distribution_version()
        if version is None:
            raise AttentionSelectionError("the SageAttention distribution version is unavailable")
        return SAGE2_PROVIDER, version
    if status.primary == "sol":
        return SOL_ATTENTION_PROVIDER, _distribution_version("dinkster-kitchen")
    if status.primary == "dinkster_kitchen_int8":
        return COMFY_KITCHEN_INT8_PROVIDER, _distribution_version("dinkster-kitchen")
    raise AttentionSelectionError(
        f"attention primary {status.primary!r} has no provider identity mapping"
    )


def _backend_available(namespace: object) -> bool:
    return (
        callable(getattr(namespace, "is_available", None)) and cast("Any", namespace).is_available()
    )


def discover_attention_capabilities(
    *,
    torch_module: object = torch,
    device_kind: str | None = None,
    device_sm: int | None = None,
) -> AttentionCapabilityEvidence:
    """Discover this worker's immutable attention capability evidence.

    Detection is capability-first (cuda, xpu, mps, cpu) and splits the shared
    CUDA device type on the HIP runtime, so ROCm and CUDA workers can never
    share one route identity. On ROCm builds ``device_sm`` carries the HIP
    device capability pair (gfx1100 reports 110), not an NVIDIA SM.

    SDPA is always available on a supported torch worker, so every role can
    truthfully advertise it without a device-specific probe.
    """
    version = getattr(torch_module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise AttentionSelectionError("torch runtime has no version")
    hip_version = _hip_runtime_version(torch_module)
    if device_kind is None:
        cuda = cast("Any", getattr(torch_module, "cuda", None))
        if _backend_available(cuda):
            device_kind = "cuda" if hip_version is None else "rocm"
            if device_sm is None:
                capability = cuda.get_device_capability()
                device_sm = int(capability[0]) * 10 + int(capability[1])
        elif _backend_available(getattr(torch_module, "xpu", None)):
            device_kind = "xpu"
        else:
            mps = getattr(getattr(torch_module, "backends", None), "mps", None)
            device_kind = "mps" if _backend_available(mps) else "cpu"
    providers: list[tuple[str, str]] = [("torch", version)]
    if device_kind == "cuda" and hip_version is not None:
        raise AttentionSelectionError(
            "a cuda route token cannot be issued by a ROCm torch build; use device_kind 'rocm'"
        )
    if device_kind == "rocm":
        if hip_version is None:
            raise AttentionSelectionError(
                "a rocm route token requires a ROCm torch build with torch.version.hip"
            )
        providers.append(("hip", hip_version))
    elif device_kind == "xpu":
        if device_sm is not None:
            raise AttentionSelectionError("an xpu route token has no SM capability")
        xpu_version = getattr(getattr(torch_module, "version", None), "xpu", None)
        if isinstance(xpu_version, str) and xpu_version:
            providers.append(("xpu", xpu_version))
    available_policies: list[AttentionPolicy] = ["sdpa"]
    if sage2_attention_available():
        sage_version = sage2_distribution_version()
        assert sage_version is not None
        available_policies.append("sage")
        providers.append(("sageattention", sage_version))
    kitchen_available = False
    if (
        device_kind == "cuda"
        and device_sm is not None
        and device_sm >= 80
        and _sol_backend_available()
    ):
        available_policies.append("sol")
        kitchen_available = True
    if dinkster_kitchen_int8_available():
        available_policies.append("dinkster_kitchen_int8")
        kitchen_available = True
    if kitchen_available:
        providers.append(("dinkster-kitchen", _distribution_version("dinkster-kitchen")))
    runtime_line = version.split("+")[0]
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=tuple(available_policies),
        provider_versions=tuple(sorted(providers)),
        adapter_contract_revision=ATTENTION_ADAPTER_CONTRACT,
        device_kind=device_kind,
        device_sm=device_sm,
        sdpa_torch_runtime=runtime_line,
    )


def discover_attention_route_token(
    policy: AttentionPolicy = "auto",
    *,
    requested_role_policies: Sequence[tuple[str, AttentionPolicy]] = (),
    torch_module: object = torch,
    device_kind: str | None = None,
    device_sm: int | None = None,
) -> AttentionRouteToken:
    """Derive one route token from fresh local capabilities and policy."""
    try:
        config = AttentionPolicyConfig(policy, tuple(requested_role_policies))
        return derive_attention_route_token(
            discover_attention_capabilities(
                torch_module=torch_module,
                device_kind=device_kind,
                device_sm=device_sm,
            ),
            config,
        )
    except (TypeError, ValueError) as exc:
        raise AttentionSelectionError(str(exc)) from exc


__all__ = [
    "ATTENTION_ADAPTER_CONTRACT",
    "BUILTIN_SDPA_PROVIDER",
    "COMFY_KITCHEN_INT8_PROVIDER",
    "SAGE2_PROVIDER",
    "SOL_ATTENTION_PROVIDER",
    "AttentionBlockKernel",
    "AttentionBlockResult",
    "AttentionError",
    "AttentionKernel",
    "PackedAttentionFacts",
    "AttentionPolicy",
    "AttentionRole",
    "AttentionSelection",
    "AttentionSelectionError",
    "AttentionStatus",
    "AttentionTensorLease",
    "AttentionValidationError",
    "QkvConsumingAttentionKernel",
    "attention_provider_identity",
    "bind_packed_attention_kernel",
    "builtin_sdpa_kernel",
    "dinkster_kitchen_int8_available",
    "discover_attention_capabilities",
    "discover_attention_route_token",
    "exclude_packed_attention_modifiers",
    "resolve_role_attention",
    "sage2_attention_available",
    "sage2_distribution_version",
    "schedule_aware_attention_kernel",
    "select_attention",
    "sol_attention_available",
]
