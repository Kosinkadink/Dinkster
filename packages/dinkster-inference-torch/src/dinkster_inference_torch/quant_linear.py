"""Module-held fp8 weight storage: the quantized Linear layer.

The reference holds scaled-fp8 weights as a QuantizedTensor wrapper
subclass inside mixed_precision_ops.Linear (comfy/ops.py @ b78cec87)
and needs custom ``_apply``/``_load_from_state_dict``/``state_dict``
overrides to keep the wrapper alive through device moves and
(de)serialization. Dinkster registers the same three tensors as ORDINARY
module state instead - ``weight`` is an fp8 Parameter (the qdata),
``weight_scale``/``input_scale`` are float32 scalar buffers - so
``state_dict()``, ``.to()``, ``_apply()``, torch.compile, and
assign-loading all work through stock nn.Module mechanics with no
overrides at all. Storage stays visible at every use site, matching
quant.py's no-wrapper stance.

Two forward paths, chosen by module state that is fixed before the
first forward (compile discipline: dynamo guards on the plain bool
attribute, never on per-call Python policy):

- DEQUANT (default, always correct, every device): dequantize the
  weight to the bind-time compute dtype (``qdata.to(dtype) * scale``,
  the exact quant.py/cast_weight product) and run a plain F.linear.
  This is the reference's cast_bias_weight route for quantized
  weights, its full_precision_matrix_mult mode, and the CPU fallback.
  Autograd flows to the input through it (the fp8 weight itself never
  requires grad).
- FP8 MATMUL (opt in via :meth:`bind_fp8_matmul`, e4m3fn only):
  quantize the input per-tensor against ``input_scale`` (checkpoint
  value, or the neutral 1.0 - the reference drops stored 1.0 scales
  at load and uses ones at runtime; clamp-to-fp8-range + cast is
  bit-identical either way) through Kitchen's CUDA route, then eager torch.
  Run tensor-wise scaled-mm with both scales, accumulating at compute dtype.
  The matmul backend is Kitchen ``scaled_mm_v2`` when that capability is
  importable, otherwise eager ``torch._scaled_mm``. Both call the same underlying
  operation on CUDA (``torch._scaled_mm`` below torch 2.10,
  ``torch.nn.functional.scaled_mm`` from 2.10 onward; kitchen 0.2.31
  operation on CUDA, and on HIP Kitchen serves its WMMA kernel. Native proofs
  pin all backends bit-for-bit on the installed torch. Inputs of rank
  other than 2/3 fall back to dequant, like the reference.

When module residency is enrolled, resident units keep these exact
paths. Offloaded unpatched fp8 weights move as qdata + scales and keep
the hardware path bit-for-bit; an offloaded patched fp8 weight uses
the dequant path because ``DeferredPatch`` necessarily dequantizes it.

``input_scale`` is ALWAYS registered (default 1.0). A checkpoint
without one loads the neutral value; saving then emits it, which the
comfy_quant format accepts. This keeps the forward branch-free over
scale presence.

Weight mutation goes through :meth:`set_weight` (the reference
set_weight: requantize_from_float with a recalculated scale and
seeded stochastic rounding, orig dtype preserved) or the
:meth:`stored`/:meth:`load_stored` pair, which bridges to the
Fp8ScaledWeight value type so the stage-4c patch flow
(dequantize -> patch -> seeded requantize, apply.py) applies to
module-held storage unchanged.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import torch
from dinkster_inference import INT8_BACKWARD_TEMP_LIMIT_BYTES

from ._nvfp4_diagnostics import Nvfp4DiagnosticsRecorder
from .dtype_policy import amd_fp8_matmul_supported, torch_version_numeric
from .ops import cast_weight
from .quant import (
    FP8_DTYPES,
    Fp8ScaledWeight,
    Int8PackedWeight,
    Nvfp4PackedWeight,
    requantize_fp8_scaled,
)

if TYPE_CHECKING:
    from .module_residency import LayerLease, ResidencyBinding

__all__ = [
    "Fp8Linear",
    "Int8Embedding",
    "Int8Linear",
    "default_fp8_matmul",
    "linear_input_act",
    "prepare_fp8_matmul_runtime",
    "supports_fp8_matmul",
]


Fp8MatmulBackend = Literal["kitchen", "torch"]
ScaledMm = Callable[..., torch.Tensor]
QuantizeFp8 = Callable[[torch.Tensor, torch.Tensor, torch.dtype], torch.Tensor]
_kitchen_probed = False
_kitchen_scaled_mm_v2: ScaledMm | None = None
_kitchen_quantize_probed = False
_kitchen_quantize_per_tensor_fp8: QuantizeFp8 | None = None


class Int8ExecutionError(RuntimeError):
    """The required Comfy Kitchen INT8 operation is unavailable or failed."""


def _int8_native_matmul_supported(device: torch.device) -> bool:
    """dinkster-kitchen int8_linear bottoms out in torch._int_mm, which has
    no MPS kernel; there INT8 layers execute through the dequant route
    instead."""
    return device.type != "mps"


_INT8_TRAINING_BACKWARD_TEMP_BYTES = INT8_BACKWARD_TEMP_LIMIT_BYTES
_INT8_DEQUANT_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
    input_act: Literal["gelu_tanh", "swiglu", "rms_norm"] | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        kitchen = cast(Any, importlib.import_module("dinkster_kitchen"))
        operation = kitchen.int8_linear
    except (ImportError, AttributeError) as error:
        raise Int8ExecutionError("INT8 execution requires dinkster-kitchen int8_linear") from error
    try:
        return cast(
            torch.Tensor,
            operation(
                input,
                weight,
                weight_scale,
                bias,
                out_dtype=out_dtype,
                convrot=convrot,
                convrot_groupsize=convrot_groupsize,
                input_act=input_act,
                input_act_weight=input_act_weight,
                input_act_eps=input_act_eps,
                residual=residual,
                residual_scale=residual_scale,
            ),
        )
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        raise Int8ExecutionError(f"dinkster-kitchen int8_linear failed: {error}") from error


def _dequantize_int8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    if not convrot:
        return weight.to(dtype=dtype) * weight_scale.to(dtype=dtype)
    try:
        importlib.import_module("dinkster_kitchen")
        dequantized = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight(
            weight, weight_scale, convrot_groupsize
        )
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        raise Int8ExecutionError(
            f"dinkster-kitchen ConvRot INT8 dequantization failed: {error}"
        ) from error
    return dequantized.to(dtype=dtype)


def _int8_embedding(
    indices: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    try:
        importlib.import_module("dinkster_kitchen")
        dtype_code = _INT8_DEQUANT_DTYPE_CODES[out_dtype]
        result = torch.ops.dinkster_kitchen.dequantize_int8_embedding(
            weight,
            weight_scale,
            indices.to(weight.device),
            convrot_groupsize if convrot else 0,
            dtype_code,
        )
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        raise Int8ExecutionError(
            f"dinkster-kitchen INT8 embedding dequantization failed: {error}"
        ) from error
    return result.to(dtype=out_dtype)


def _dequantize_int8_training_chunk(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    if not convrot:
        return weight.to(dtype=dtype) * weight_scale.to(dtype=dtype)
    try:
        importlib.import_module("dinkster_kitchen")
        dtype_code = _INT8_DEQUANT_DTYPE_CODES[dtype]
        return cast(
            torch.Tensor,
            torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight_dtype(
                weight,
                weight_scale,
                convrot_groupsize,
                dtype_code,
            ),
        )
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        raise Int8ExecutionError(
            f"dinkster-kitchen ConvRot INT8 training dequantization failed: {error}"
        ) from error


def _int8_training_chunk_features(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> int:
    output_bytes = torch.empty((), dtype=dtype).element_size()
    if convrot and convrot_groupsize != 256:
        # Kitchen's eager 16/64-wide ConvRot path holds two float32
        # intermediates while producing the requested output dtype.
        temporary_bytes_per_element = 9
    elif convrot:
        temporary_bytes_per_element = 1 + output_bytes
    else:
        temporary_bytes_per_element = 1 + 2 * output_bytes
    alignment = convrot_groupsize if convrot else 16
    elements = _INT8_TRAINING_BACKWARD_TEMP_BYTES // (weight.shape[0] * temporary_bytes_per_element)
    chunk = max(alignment, (elements // alignment) * alignment)
    return min(weight.shape[1], chunk)


class _Int8FusedTraining(torch.autograd.Function):
    """Fused W8A8 forward with a bounded frozen-weight input gradient."""

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        convrot: bool,
        convrot_groupsize: int,
    ) -> torch.Tensor:
        device_type = input.device.type
        if torch.is_autocast_enabled(device_type):
            matmul_dtype = torch.get_autocast_dtype(device_type)
        else:
            matmul_dtype = compute_dtype
        fused_input = input if input.dtype == matmul_dtype else input.to(dtype=matmul_dtype)
        ctx.save_for_backward(weight, weight_scale)
        ctx.dinkster_device_type = device_type
        ctx.dinkster_matmul_dtype = matmul_dtype
        ctx.dinkster_input_dtype = input.dtype
        ctx.dinkster_input_shape = input.shape
        ctx.dinkster_convrot = convrot
        ctx.dinkster_convrot_groupsize = convrot_groupsize
        return _int8_linear(
            fused_input,
            weight,
            weight_scale,
            bias,
            out_dtype=compute_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
        grad_input: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            weight, weight_scale = ctx.saved_tensors
            chunk_features = _int8_training_chunk_features(
                weight,
                dtype=ctx.dinkster_matmul_dtype,
                convrot=ctx.dinkster_convrot,
                convrot_groupsize=ctx.dinkster_convrot_groupsize,
            )
            grad_output_2d = grad_output.reshape(-1, weight.shape[0]).to(
                dtype=ctx.dinkster_matmul_dtype
            )
            grad_input_2d = torch.empty(
                (grad_output_2d.shape[0], weight.shape[1]),
                dtype=ctx.dinkster_matmul_dtype,
                device=grad_output.device,
            )
            with torch.autocast(ctx.dinkster_device_type, enabled=False):
                for start in range(0, weight.shape[1], chunk_features):
                    stop = min(start + chunk_features, weight.shape[1])
                    dequantized = _dequantize_int8_training_chunk(
                        weight[:, start:stop],
                        weight_scale,
                        dtype=ctx.dinkster_matmul_dtype,
                        convrot=ctx.dinkster_convrot,
                        convrot_groupsize=ctx.dinkster_convrot_groupsize,
                    )
                    grad_input_2d[:, start:stop] = grad_output_2d.matmul(dequantized)
            grad_input = grad_input_2d.reshape(ctx.dinkster_input_shape)
            if grad_input.dtype != ctx.dinkster_input_dtype:
                grad_input = grad_input.to(dtype=ctx.dinkster_input_dtype)
        return grad_input, None, None, None, None, None, None


class Int8Linear(torch.nn.Module):
    """Linear backed by Comfy tensorwise INT8 storage and ConvRot metadata."""

    weight_scale: torch.Tensor
    _residency: ResidencyBinding | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        compute_dtype: torch.dtype,
        per_channel: bool | None = None,
        convrot: bool,
        convrot_groupsize: int,
        full_precision_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.per_channel = convrot if per_channel is None else per_channel
        if convrot and not self.per_channel:
            raise ValueError("ConvRot INT8 storage requires per-channel scales")
        self.full_precision_matmul = full_precision_matmul
        self.fused_training = False
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.int8),
            requires_grad=False,
        )
        scale_shape = (out_features, 1) if self.per_channel else ()
        self.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32))
        if bias:
            self.bias: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def bind_fused_training(self, enabled: bool) -> None:
        """Select the CUDA W8A8 forward for frozen-base training."""
        if enabled:
            if self.full_precision_matmul:
                raise ValueError(
                    "layer is pinned to full-precision matmul by its checkpoint config"
                )
            layout_error = self._fused_training_layout_error(self.in_features)
            if layout_error is not None:
                raise ValueError(layout_error)
            if self.bias is not None and self.bias.requires_grad:
                raise ValueError("fused INT8 training does not support trainable bias")
        self.fused_training = enabled

    def _fused_training_layout_error(self, in_features: int) -> str | None:
        if in_features % 16 != 0:
            return f"fused INT8 training requires input features divisible by 16, got {in_features}"
        if self.convrot:
            group = self.convrot_groupsize
            if group < 4 or group & (group - 1) or group.bit_length() % 2 == 0:
                return (
                    "fused INT8 training requires ConvRot group size to be a power of 4 >= 4,"
                    f" got {group}"
                )
            if in_features % group != 0:
                return (
                    "fused INT8 training requires input features divisible by ConvRot group size"
                    f" {group}, got {in_features}"
                )
        return None

    def bind_residency(self, binding: ResidencyBinding) -> None:
        self._residency = binding

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._residency
        if binding is None or binding.mechanism.is_loaded(binding.unit):
            return None
        weight_key = binding.key("weight")
        requests: list[tuple[str, torch.dtype | None]] = [
            (
                weight_key,
                self.compute_dtype if binding.mechanism.weight_functions(weight_key) else None,
            ),
        ]
        if self.bias is not None:
            requests.append((binding.key("bias"), self.compute_dtype))
        return binding.mechanism, tuple(requests)

    def _execute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        input_act: Literal["gelu_tanh", "swiglu", "rms_norm"] | None = None,
        input_act_weight: torch.Tensor | None = None,
        input_act_eps: float = 0.0,
        residual: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fused_training:
            if self.full_precision_matmul:
                raise Int8ExecutionError(
                    "fused INT8 training cannot use a layer pinned to full-precision matmul"
                )
            layout_error = self._fused_training_layout_error(weight.shape[1])
            if layout_error is not None:
                raise Int8ExecutionError(layout_error)
            if bias is not None and bias.requires_grad:
                raise Int8ExecutionError("fused INT8 training does not support trainable bias")
            if input.device.type != "cuda":
                raise Int8ExecutionError("fused INT8 training requires CUDA input")
            if torch.is_grad_enabled() and input.requires_grad:
                return cast(
                    torch.Tensor,
                    _Int8FusedTraining.apply(
                        input,
                        weight,
                        weight_scale,
                        bias,
                        self.compute_dtype,
                        self.convrot,
                        self.convrot_groupsize,
                    ),
                )
            return _int8_linear(
                input,
                weight,
                weight_scale,
                bias,
                out_dtype=self.compute_dtype,
                convrot=self.convrot,
                convrot_groupsize=self.convrot_groupsize,
                input_act=input_act,
                input_act_weight=input_act_weight,
                input_act_eps=input_act_eps,
                residual=residual,
                residual_scale=residual_scale,
            )
        if self.full_precision_matmul:
            use_dequant_route = True
        else:
            if torch.is_grad_enabled() and input.requires_grad:
                raise Int8ExecutionError("INT8 matrix multiplication is inference-only")
            use_dequant_route = not _int8_native_matmul_supported(input.device)
        if use_dequant_route:
            if input_act is not None:
                input = _input_activation(input, input_act, input_act_weight, input_act_eps)
            dequantized = _dequantize_int8(
                weight,
                weight_scale,
                dtype=self.compute_dtype,
                convrot=self.convrot,
                convrot_groupsize=self.convrot_groupsize,
            )
            cast_bias = None if bias is None else bias.to(dtype=self.compute_dtype)
            output = torch.nn.functional.linear(input, dequantized, cast_bias)
            return _residual_epilogue(output, residual, residual_scale)
        return _int8_linear(
            input,
            weight,
            weight_scale,
            bias,
            out_dtype=self.compute_dtype,
            convrot=self.convrot,
            convrot_groupsize=self.convrot_groupsize,
            input_act=input_act,
            input_act_weight=input_act_weight,
            input_act_eps=input_act_eps,
            residual=residual,
            residual_scale=residual_scale,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._forward(input)

    def _forward(
        self,
        input: torch.Tensor,
        input_act: Literal["gelu_tanh", "swiglu", "rms_norm"] | None = None,
        input_act_weight: torch.Tensor | None = None,
        input_act_eps: float = 0.0,
        residual: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        binding = self._residency
        if binding is not None and not binding.mechanism.is_loaded(binding.unit):
            with binding.lease() as lease:
                weight_key = binding.key("weight")
                if binding.mechanism.weight_functions(weight_key):
                    if input_act is not None:
                        input = _input_activation(input, input_act, input_act_weight, input_act_eps)
                    output = torch.nn.functional.linear(
                        input,
                        lease.get("weight", dtype=self.compute_dtype),
                        None if self.bias is None else lease.get("bias", dtype=self.compute_dtype),
                    )
                    return _residual_epilogue(output, residual, residual_scale)
                stored = lease.get_stored("weight")
                if not isinstance(stored, Int8PackedWeight):
                    raise TypeError("Int8Linear residency weight is not folded INT8 storage")
                return self._execute(
                    input,
                    stored.qdata,
                    stored.scale,
                    None if self.bias is None else lease.get("bias", dtype=self.compute_dtype),
                    input_act,
                    input_act_weight,
                    input_act_eps,
                    residual,
                    residual_scale,
                )
        return self._execute(
            input,
            self.weight,
            self.weight_scale,
            self.bias,
            input_act,
            input_act_weight,
            input_act_eps,
            residual,
            residual_scale,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" bias={self.bias is not None}, compute_dtype={self.compute_dtype},"
            f" convrot={self.convrot}, convrot_groupsize={self.convrot_groupsize}"
        )


class Int8Embedding(torch.nn.Module):
    """Embedding backed by Comfy tensorwise INT8 storage and ConvRot metadata."""

    weight_scale: torch.Tensor
    _residency: ResidencyBinding | None = None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        compute_dtype: torch.dtype,
        per_channel: bool | None = None,
        convrot: bool,
        convrot_groupsize: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compute_dtype = compute_dtype
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.per_channel = convrot if per_channel is None else per_channel
        if convrot and not self.per_channel:
            raise ValueError("ConvRot INT8 storage requires per-channel scales")
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
            requires_grad=False,
        )
        scale_shape = (num_embeddings, 1) if self.per_channel else ()
        self.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32))

    def bind_residency(self, binding: ResidencyBinding) -> None:
        self._residency = binding

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._residency
        if binding is None or binding.mechanism.is_loaded(binding.unit):
            return None
        weight_key = binding.key("weight")
        return (
            binding.mechanism,
            (
                (
                    weight_key,
                    self.compute_dtype if binding.mechanism.weight_functions(weight_key) else None,
                ),
            ),
        )

    def _execute(
        self,
        indices: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return _int8_embedding(
            indices,
            weight,
            weight_scale,
            out_dtype=self.compute_dtype,
            convrot=self.convrot,
            convrot_groupsize=self.convrot_groupsize,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        binding = self._residency
        if binding is not None and not binding.mechanism.is_loaded(binding.unit):
            with binding.lease() as lease:
                weight_key = binding.key("weight")
                if binding.mechanism.weight_functions(weight_key):
                    weight = lease.get("weight", dtype=self.compute_dtype)
                    return torch.nn.functional.embedding(
                        indices.to(weight.device),
                        weight,
                    )
                stored = lease.get_stored("weight")
                if not isinstance(stored, Int8PackedWeight):
                    raise TypeError("Int8Embedding residency weight is not folded INT8 storage")
                return self._execute(indices, stored.qdata, stored.scale)
        return self._execute(indices, self.weight, self.weight_scale)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim},"
            f" compute_dtype={self.compute_dtype}, convrot={self.convrot},"
            f" convrot_groupsize={self.convrot_groupsize}"
        )


def _swiglu(input: torch.Tensor) -> torch.Tensor:
    gate, up = input.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate)
    # The in-place multiply overwrites values autograd's mul backward
    # needs; keep it as the inference fast path only.
    return activated * up if torch.is_grad_enabled() else activated.mul_(up)


def _input_activation(
    input: torch.Tensor,
    input_act: Literal["gelu_tanh", "swiglu", "rms_norm"],
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
) -> torch.Tensor:
    if input_act == "swiglu":
        return _swiglu(input)
    if input_act == "rms_norm":
        return torch.nn.functional.rms_norm(
            input, (input.shape[-1],), input_act_weight, input_act_eps
        )
    return torch.nn.functional.gelu(input, approximate="tanh")


def _residual_epilogue(
    output: torch.Tensor,
    residual: torch.Tensor | None,
    residual_scale: torch.Tensor | None,
) -> torch.Tensor:
    if residual is None:
        return output
    if residual_scale is None:
        return output.add_(residual)
    return residual.addcmul(output, residual_scale)


def linear_input_act(
    linear: torch.nn.Module,
    input: torch.Tensor,
    input_act: Literal["gelu_tanh", "swiglu", "rms_norm"] | None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    *,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(linear, Int8Linear) and not (torch.is_grad_enabled() and input.requires_grad):
        if (
            input_act_weight is None
            and input_act_eps == 0.0
            and residual is None
            and residual_scale is None
        ):
            return linear._forward(input, input_act)  # pyright: ignore[reportPrivateUsage]
        return linear._forward(  # pyright: ignore[reportPrivateUsage]
            input,
            input_act,
            input_act_weight,
            input_act_eps,
            residual,
            residual_scale,
        )
    activated = (
        input
        if input_act is None
        else _input_activation(input, input_act, input_act_weight, input_act_eps)
    )
    return _residual_epilogue(cast(torch.Tensor, linear(activated)), residual, residual_scale)


class Nvfp4ExecutionError(RuntimeError):
    """The requested NVFP4 Kitchen execution path is unavailable or failed."""


class _Nvfp4Kitchen(NamedTuple):
    quantize: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    dequantize: Callable[..., torch.Tensor]
    scaled_mm: Callable[..., torch.Tensor]
    registry: Any


_nvfp4_kitchen_probed = False
_nvfp4_kitchen: _Nvfp4Kitchen | None = None


def _probe_nvfp4_kitchen() -> _Nvfp4Kitchen | None:
    """Lazily resolve the complete NVFP4 Kitchen capability set."""
    global _nvfp4_kitchen_probed, _nvfp4_kitchen
    if _nvfp4_kitchen_probed:
        return _nvfp4_kitchen
    _nvfp4_kitchen_probed = True
    try:
        module = cast(Any, importlib.import_module("dinkster_kitchen"))
        quantize = module.quantize_nvfp4
        dequantize = module.dequantize_nvfp4
        scaled_mm = module.scaled_mm_nvfp4
        registry = module.registry
        if all(callable(function) for function in (quantize, dequantize, scaled_mm)) and callable(
            getattr(registry, "get_capable_backend", None)
        ):
            _nvfp4_kitchen = _Nvfp4Kitchen(
                cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], quantize),
                cast(Callable[..., torch.Tensor], dequantize),
                cast(Callable[..., torch.Tensor], scaled_mm),
                registry,
            )
    except Exception:  # noqa: BLE001 - optional Kitchen capability probe
        _nvfp4_kitchen = None
    return _nvfp4_kitchen


def _require_nvfp4_kitchen() -> _Nvfp4Kitchen:
    kitchen = _probe_nvfp4_kitchen()
    if kitchen is None:
        raise Nvfp4ExecutionError(
            "NVFP4 execution requires dinkster-kitchen quantize_nvfp4,"
            " dequantize_nvfp4, scaled_mm_nvfp4, and backend capability probing"
        )
    return kitchen


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _nvfp4_route(
    full_precision_matmul: bool,
    device: torch.device,
    ndim: int,
) -> tuple[bool, str]:
    """Whether NVFP4 native execution may be attempted, plus the
    diagnostics route token.

    Native NVFP4 is NVIDIA-only: on ROCm builds an AMD device also
    reports device type ``cuda`` and its capability major is a gfx
    generation (gfx1200 reports 12), not an NVIDIA SM, so HIP is
    refused before the SM 10 check ever reads device properties.
    """
    if full_precision_matmul:
        return False, "route_full_precision"
    if device.type != "cuda":
        return False, "route_non_cuda"
    if torch.version.hip is not None:
        return False, "route_hip"
    if ndim < 2:
        return False, "route_rank"
    properties = torch.cuda.get_device_properties(device)
    if properties.major >= 10:
        return True, "route_native"
    return False, "route_pre_sm10"


def supports_fp8_matmul(
    device: torch.device | None = None,
    *,
    support_override: bool = False,
) -> bool:
    """Mirror ComfyUI ``supports_fp8_compute`` @ b78cec87 exactly.

    ``support_override`` is the library-facing analog of upstream's
    ``SUPPORT_FP8_OPS`` hook: true bypasses every vendor, device, and
    version gate. On ROCm builds support follows the narrow AMD
    architecture gate (:func:`~.dtype_policy.amd_fp8_matmul_supported`).
    Otherwise support is NVIDIA-only. SM 9+ is accepted; SM below 8 and
    SM 8.0-8.8 are refused; SM 8.9 needs torch 2.3+, or torch 2.4+ on
    Windows.
    """
    if support_override:
        return True
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:
        return amd_fp8_matmul_supported(device if device is not None else torch.device("cuda"))
    if torch.version.cuda is None:
        return False
    properties = torch.cuda.get_device_properties(device)
    if properties.major >= 9:
        return True
    if properties.major < 8 or properties.minor < 9:
        return False
    minimum = (2, 4) if sys.platform == "win32" else (2, 3)
    return torch_version_numeric() >= minimum


def default_fp8_matmul(
    device: torch.device | None = None,
    *,
    requested: bool = False,
    support_override: bool = False,
) -> bool:
    """Upstream-parity policy input for a future backend settings seat.

    The default is deliberately OFF: ComfyUI selects fp8 operations
    only when ``--fast fp8_matrix_mult`` (or its internal optimization
    request) opts in. ``requested=True`` applies the exact capability
    matrix from :func:`supports_fp8_matmul`. This function does not wire
    CLI or server policy.
    """
    return requested and supports_fp8_matmul(
        device,
        support_override=support_override,
    )


def _probe_kitchen_scaled_mm_v2() -> ScaledMm | None:
    """Lazily resolve kitchen's torch-2.10+ scaled-mm surface."""
    global _kitchen_probed, _kitchen_scaled_mm_v2
    if _kitchen_probed:
        return _kitchen_scaled_mm_v2
    _kitchen_probed = True
    try:
        module = cast(Any, importlib.import_module("dinkster_kitchen.scaled_mm_v2"))
        has_scaled_mm_v2 = module.has_scaled_mm_v2
        scaled_mm_v2 = module.scaled_mm_v2
        if bool(has_scaled_mm_v2()) and callable(scaled_mm_v2):
            _kitchen_scaled_mm_v2 = cast(ScaledMm, scaled_mm_v2)
    except Exception:  # noqa: BLE001 - an optional accelerator probe is best-effort
        _kitchen_scaled_mm_v2 = None
    return _kitchen_scaled_mm_v2


def _probe_kitchen_quantize_per_tensor_fp8() -> QuantizeFp8 | None:
    """Lazily resolve Kitchen's fused tensorwise FP8 quantizer."""
    global _kitchen_quantize_probed, _kitchen_quantize_per_tensor_fp8
    if _kitchen_quantize_probed:
        return _kitchen_quantize_per_tensor_fp8
    _kitchen_quantize_probed = True
    try:
        module = cast(Any, importlib.import_module("dinkster_kitchen"))
        quantize = module.quantize_per_tensor_fp8
        if callable(quantize):
            _kitchen_quantize_per_tensor_fp8 = cast(QuantizeFp8, quantize)
    except Exception:  # noqa: BLE001 - an optional accelerator probe is best-effort
        _kitchen_quantize_per_tensor_fp8 = None
    return _kitchen_quantize_per_tensor_fp8


def select_fp8_matmul_backend() -> Fp8MatmulBackend:
    """Kitchen when its v2 API exists, otherwise eager torch."""
    return "kitchen" if _probe_kitchen_scaled_mm_v2() is not None else "torch"


def prepare_fp8_matmul_runtime() -> Fp8MatmulBackend:
    """Resolve FP8 accelerator operations before accepting inference work."""
    backend = select_fp8_matmul_backend()
    _probe_kitchen_quantize_per_tensor_fp8()
    return backend


def _torch_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch._scaled_mm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )
    # torch 2.4 returned (output, output_amax); 2.5+ returns the
    # output tensor alone (kitchen normalizes the same way).
    return output[0] if isinstance(output, tuple) else output


def _quantize_per_tensor_fp8_eager(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    fp8_max = torch.finfo(dtype).max
    scaled = input.float() / scale.float()
    return torch.clamp(scaled, -fp8_max, fp8_max).to(dtype)


def fp8_matmul_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    backend: Fp8MatmulBackend,
) -> torch.Tensor:
    """Shared scaled/plain fp8 route. ``weight`` is untransposed qdata."""
    if torch.is_grad_enabled() and (
        input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "the fp8 matmul route is inference-only (torch._scaled_mm"
            " has no backward); run under no_grad, or leave"
            " bind_fp8_matmul off for gradient work"
        )
    shape = input.shape
    x = input.reshape(-1, shape[-1]) if input.ndim == 3 else input
    kitchen_quantize = _probe_kitchen_quantize_per_tensor_fp8()
    if backend == "kitchen":
        scaled = _kitchen_scaled_mm_v2
    else:
        scaled = None
    # dinkster-kitchen's CUDA backend exports operands via __dlpack__ with
    # no device guard, and torch refuses cross-device DLPack export, so
    # pin the CUDA context to the input's device for the kitchen calls
    # (kitchen also reads torch.cuda.current_device() capability state).
    device_guard = torch.cuda.device(input.device) if input.device.type == "cuda" else nullcontext()
    with device_guard:
        if kitchen_quantize is not None:
            qdata = kitchen_quantize(x, input_scale, torch.float8_e4m3fn)
        else:
            qdata = _quantize_per_tensor_fp8_eager(x, input_scale, torch.float8_e4m3fn)
        cast_bias = None if bias is None else bias.to(dtype=out_dtype)
        # cuBLASLt fuses the bias epilogue only for fp16/bf16 outputs;
        # float32 adds it after the matmul in both backends.
        fused_bias = None if out_dtype == torch.float32 else cast_bias
        if scaled is None:
            output = _torch_scaled_mm(
                qdata.contiguous(),
                weight.t(),
                scale_a=input_scale,
                scale_b=weight_scale,
                bias=fused_bias,
                out_dtype=out_dtype,
            )
        else:
            output = scaled(
                qdata.contiguous(),
                weight.t(),
                scale_a=input_scale,
                scale_b=weight_scale,
                bias=fused_bias,
                out_dtype=out_dtype,
            )
    if cast_bias is not None and fused_bias is None:
        output = output + cast_bias
    if input.ndim == 3:
        output = output.reshape(shape[0], shape[1], weight.shape[0])
    return output


class Fp8Linear(torch.nn.Module):
    """A Linear whose weight is stored quantized: fp8 qdata plus a
    float32 per-tensor weight scale (and an input scale for the fp8
    matmul path). State-dict layout is the comfy_quant per-layer
    contract: ``weight`` (fp8), ``weight_scale``, ``input_scale``,
    optional ``bias``. Construction leaves parameters empty
    (checkpoint-owned; load with ``assign=True`` to preserve exact
    storage, the house pattern)."""

    weight_scale: torch.Tensor
    input_scale: torch.Tensor
    _residency: ResidencyBinding | None = None
    _residency_route: (
        tuple[ResidencyBinding, bool, tuple[tuple[str, torch.dtype | None], ...]] | None
    ) = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        compute_dtype: torch.dtype,
        full_precision_matmul: bool = False,
    ) -> None:
        super().__init__()
        if fp8_dtype not in FP8_DTYPES:
            raise ValueError(f"not an fp8 dtype: {fp8_dtype}")
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        #: The reference's full_precision_matrix_mult flag: always
        #: dequantize and matmul at compute dtype.
        self.full_precision_matmul = full_precision_matmul
        #: Enabled by :meth:`bind_fp8_matmul` after a capability check;
        #: never flipped inside forward (compile discipline).
        self.fp8_matmul = False
        self._fp8_matmul_backend: Fp8MatmulBackend = "torch"
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=fp8_dtype),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones((), dtype=torch.float32))
        if bias:
            self.bias: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def bind_fp8_matmul(self, enabled: bool) -> None:
        """Choose the matmul route BEFORE the first (compiled) forward.
        Enabling is refused for e5m2 storage (``torch._scaled_mm``
        requires an e4m3fn operand pair here, matching the reference
        fp8_linear's dtype gate) and when ``full_precision_matmul`` is
        set. The fp8 matmul route is inference-only: a forward on a
        gradient-requiring input raises (``torch._scaled_mm`` has no
        backward); gradient work stays on the dequant route."""
        if enabled and self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"fp8 matmul requires float8_e4m3fn storage, got {self.weight.dtype}")
        if enabled and self.full_precision_matmul:
            raise ValueError("layer is pinned to full-precision matmul by its checkpoint config")
        if enabled:
            self._fp8_matmul_backend = prepare_fp8_matmul_runtime()
        self.fp8_matmul = enabled
        self._residency_route = None

    def _residency_uses_raw_storage(self, name: str) -> bool:
        return name == "weight"

    def bind_residency(self, binding: ResidencyBinding) -> None:
        """Bind post-assembly routed access to module state."""
        self._residency = binding
        self._residency_route = None

    def _offloaded_residency_route(
        self, binding: ResidencyBinding
    ) -> tuple[bool, tuple[tuple[str, torch.dtype | None], ...]]:
        route = self._residency_route
        if route is not None and route[0] is binding:
            return route[1], route[2]
        weight_key = binding.key("weight")
        raw_weight = not binding.mechanism.weight_functions(weight_key)
        requests: list[tuple[str, torch.dtype | None]] = [
            (weight_key, None if raw_weight else self.compute_dtype)
        ]
        if raw_weight and self.fp8_matmul:
            requests.append((binding.key("input_scale"), self.input_scale.dtype))
        if self.bias is not None:
            requests.append((binding.key("bias"), self.compute_dtype))
        frozen_requests = tuple(requests)
        self._residency_route = (binding, raw_weight, frozen_requests)
        return raw_weight, frozen_requests

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._residency
        if binding is None or binding.mechanism.is_loaded(binding.unit):
            return None
        _raw_weight, requests = self._offloaded_residency_route(binding)
        return binding.mechanism, requests

    # --- storage bridge (patch/requantize flow) ---------------------

    def stored(self) -> Fp8ScaledWeight:
        """The weight as the stage-4c value type (shares storage; do
        not mutate the result's tensors directly - use
        :meth:`load_stored`)."""
        return Fp8ScaledWeight(self.weight.data, self.weight_scale, self.compute_dtype)

    def load_stored(self, stored: Fp8ScaledWeight) -> None:
        """Write a (patched/requantized) Fp8ScaledWeight back into the
        registered storage."""
        if stored.qdata.dtype != self.weight.dtype:
            raise ValueError(
                f"stored qdata is {stored.qdata.dtype}, layer stores {self.weight.dtype}"
            )
        if tuple(stored.qdata.shape) != tuple(self.weight.shape):
            raise ValueError(
                f"stored shape {tuple(stored.qdata.shape)} does not"
                f" match layer shape {tuple(self.weight.shape)}"
            )
        with torch.no_grad():
            self.weight.copy_(stored.qdata)
            self.weight_scale.copy_(stored.scale)

    def set_weight(self, weight: torch.Tensor, *, seed: int = 0) -> None:
        """Requantize a full-precision weight into this layer's storage
        form (the reference set_weight: recalculated scale, seeded
        stochastic rounding when ``seed > 0``, fp8 dtype preserved).
        ``weight`` is consumed as an owned buffer (mutated in place by
        the stochastic path, like quantize_fp8_scaled)."""
        self.load_stored(requantize_fp8_scaled(self.stored(), weight, seed=seed))

    # --- forward -----------------------------------------------------

    def _dequant_forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = self.compute_dtype
        weight = self.weight.to(dtype=dtype) * self.weight_scale.to(dtype=dtype)
        bias = None if self.bias is None else self.bias.to(dtype=dtype)
        return torch.nn.functional.linear(input, weight, bias)

    def _fp8_matmul_forward(self, input: torch.Tensor) -> torch.Tensor:
        # comfy/ops.py fp8_linear + kitchen _handle_fp8_linear
        # @ b78cec87 / 0.2.31: reshape 3D to 2D, quantize the input
        # per-tensor (float32 division, clamp to the fp8 range, cast),
        # scaled_mm against
        # the transposed qdata with both scales.
        return fp8_matmul_forward(
            input,
            self.weight,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=self.compute_dtype,
            backend=self._fp8_matmul_backend,
        )

    def _routed_dequant_forward(
        self, input: torch.Tensor, binding: ResidencyBinding, *, raw_weight: bool
    ) -> torch.Tensor:
        with binding.lease() as lease:
            if raw_weight:
                stored = lease.get_stored("weight")
                if not isinstance(stored, Fp8ScaledWeight):
                    raise TypeError("Fp8Linear residency weight is not folded fp8 storage")
                weight = cast_weight(stored, dtype=self.compute_dtype)
            else:
                weight = lease.get("weight", dtype=self.compute_dtype)
            bias = None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)
            return torch.nn.functional.linear(input, weight, bias)

    def _routed_fp8_matmul_forward(
        self, input: torch.Tensor, binding: ResidencyBinding
    ) -> torch.Tensor:
        with binding.lease() as lease:
            stored = lease.get_stored("weight")
            if not isinstance(stored, Fp8ScaledWeight):
                raise TypeError("Fp8Linear residency weight is not folded fp8 storage")
            input_scale = lease.get("input_scale", dtype=self.input_scale.dtype)
            bias = None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)

            return fp8_matmul_forward(
                input,
                stored.qdata,
                input_scale=input_scale,
                weight_scale=stored.scale,
                bias=bias,
                out_dtype=self.compute_dtype,
                backend=self._fp8_matmul_backend,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._residency
        if binding is not None and not binding.mechanism.is_loaded(binding.unit):
            raw_weight, _requests = self._offloaded_residency_route(binding)
            if self.fp8_matmul and input.ndim in (2, 3) and raw_weight:
                return self._routed_fp8_matmul_forward(input, binding)
            return self._routed_dequant_forward(input, binding, raw_weight=raw_weight)
        if self.fp8_matmul and input.ndim in (2, 3):
            return self._fp8_matmul_forward(input)
        return self._dequant_forward(input)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features},"
            f" out_features={self.out_features},"
            f" bias={self.bias is not None},"
            f" fp8_dtype={self.weight.dtype},"
            f" compute_dtype={self.compute_dtype}"
        )


class Nvfp4Linear(torch.nn.Module):
    """Package-internal Linear backed by ordinary NVFP4 module state."""

    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor
    input_scale: torch.Tensor | None
    _residency: ResidencyBinding | None = None
    _diagnostics: Nvfp4DiagnosticsRecorder | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        compute_dtype: torch.dtype,
        pre_quant_scale: bool = False,
        input_scale: bool = True,
        full_precision_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.full_precision_matmul = full_precision_matmul
        padded_in = _round_up(in_features, 16)
        padded_out = _round_up(out_features, 16)
        self.weight = torch.nn.Parameter(
            torch.empty((padded_out, padded_in // 2), dtype=torch.uint8),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                (_round_up(padded_out, 128), _round_up(padded_in // 16, 4)),
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))
        self.register_buffer(
            "input_scale", torch.empty((), dtype=torch.float32) if input_scale else None
        )
        if pre_quant_scale:
            self.pre_quant_scale: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(in_features, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("pre_quant_scale", None)
        if bias:
            self.bias: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def bind_residency(self, binding: ResidencyBinding) -> None:
        self._residency = binding

    def _bind_diagnostics(self, recorder: Nvfp4DiagnosticsRecorder) -> None:
        if self._diagnostics is not None and self._diagnostics is not recorder:
            raise RuntimeError("Nvfp4Linear is already bound to another runtime")
        self._diagnostics = recorder

    @contextmanager
    def _diagnosed_lease(self, binding: ResidencyBinding) -> Generator[LayerLease]:
        try:
            with binding.lease() as lease:
                if self._diagnostics is not None:
                    self._diagnostics.observe_state(self, "loaded")
                yield lease
        finally:
            if self._diagnostics is not None:
                loaded = binding.mechanism.is_loaded(binding.unit)
                self._diagnostics.observe_state(self, "loaded" if loaded else "offloaded")

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._residency
        if binding is None or binding.mechanism.is_loaded(binding.unit):
            return None
        weight_key = binding.key("weight")
        requests = [
            (
                weight_key,
                self.compute_dtype if binding.mechanism.weight_functions(weight_key) else None,
            ),
        ]
        if self.input_scale is not None:
            requests.append((binding.key("input_scale"), self.input_scale.dtype))
        if self.pre_quant_scale is not None:
            requests.append((binding.key("pre_quant_scale"), self.pre_quant_scale.dtype))
        if self.bias is not None:
            requests.append((binding.key("bias"), self.bias.dtype))
        return binding.mechanism, tuple(requests)

    def _validate_input(self, input: torch.Tensor) -> None:
        if input.ndim < 1:
            raise Nvfp4ExecutionError(
                f"NVFP4 Linear requires rank >= 1 input, got rank {input.ndim}"
            )
        if input.shape[-1] != self.in_features:
            raise Nvfp4ExecutionError(
                f"NVFP4 Linear expected input width {self.in_features}, got {input.shape[-1]}"
            )

    def _dequant_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        kitchen = _require_nvfp4_kitchen()
        try:
            dequantized = kitchen.dequantize(
                weight,
                weight_scale_2,
                weight_scale,
                output_type=self.compute_dtype,
            )
        except torch.OutOfMemoryError:
            if self._diagnostics is not None:
                self._diagnostics.record("dequantize_error")
            raise
        except Exception as error:
            if self._diagnostics is not None:
                self._diagnostics.record("dequantize_error")
            raise Nvfp4ExecutionError(
                f"dinkster-kitchen dequantize_nvfp4 failed: {error}"
            ) from error
        if self._diagnostics is not None:
            self._diagnostics.record("dequantize_success")
        dequantized = dequantized[: self.out_features, : self.in_features]
        cast_bias = None if bias is None else bias.to(dtype=self.compute_dtype)
        return torch.nn.functional.linear(input, dequantized, cast_bias)

    def _cuda_backend(
        self,
        kitchen: _Nvfp4Kitchen,
        operation: str,
        kwargs: dict[str, object],
    ) -> bool:
        try:
            backend = kitchen.registry.get_capable_backend(operation, kwargs)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            if type(error).__name__ == "NoCapableBackendError":
                return False
            raise Nvfp4ExecutionError(
                f"dinkster-kitchen {operation} capability probe failed: {error}"
            ) from error
        return backend == "cuda"

    def _fast_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if torch.is_grad_enabled() and input.requires_grad:
            raise Nvfp4ExecutionError(
                "NVFP4 native matrix multiplication is inference-only;"
                " gradient-requiring input is not supported"
            )
        kitchen = _require_nvfp4_kitchen()
        original_shape = input.shape
        x = input.reshape(-1, original_shape[-1]) if input.ndim >= 3 else input
        if input_scale is None:
            input_scale = torch.amax(x.abs()).to(dtype=torch.float32) / (448.0 * 6.0)
        pad_input = x.shape[0] % 16 != 0 or x.shape[1] % 16 != 0
        quantize_kwargs: dict[str, object] = {
            "x": x,
            "per_tensor_scale": input_scale,
            "pad_16x": pad_input,
        }
        if not self._cuda_backend(kitchen, "quantize_nvfp4", quantize_kwargs):
            if self._diagnostics is not None:
                self._diagnostics.record("route_no_quantize_backend")
            return None
        try:
            q_input, input_block_scale = kitchen.quantize(
                x.contiguous(), input_scale, pad_16x=pad_input
            )
        except torch.OutOfMemoryError:
            if self._diagnostics is not None:
                self._diagnostics.record("quantize_error")
            raise
        except Exception as error:
            if self._diagnostics is not None:
                self._diagnostics.record("quantize_error")
            raise Nvfp4ExecutionError(f"dinkster-kitchen quantize_nvfp4 failed: {error}") from error
        if self._diagnostics is not None:
            self._diagnostics.record("quantize_success")
        cast_bias = None if bias is None else bias.to(dtype=self.compute_dtype)
        mm_kwargs: dict[str, object] = {
            "a": q_input,
            "b": weight,
            "tensor_scale_a": input_scale,
            "tensor_scale_b": weight_scale_2,
            "block_scale_a": input_block_scale,
            "block_scale_b": weight_scale,
            "bias": cast_bias,
            "out_dtype": self.compute_dtype,
        }
        if not self._cuda_backend(kitchen, "scaled_mm_nvfp4", mm_kwargs):
            if self._diagnostics is not None:
                self._diagnostics.record("route_no_scaled_mm_backend")
            return None
        try:
            output = kitchen.scaled_mm(**mm_kwargs)
        except torch.OutOfMemoryError:
            if self._diagnostics is not None:
                self._diagnostics.record("scaled_mm_error")
            raise
        except Exception as error:
            if self._diagnostics is not None:
                self._diagnostics.record("scaled_mm_error")
            raise Nvfp4ExecutionError(
                f"dinkster-kitchen scaled_mm_nvfp4 failed: {error}"
            ) from error
        if self._diagnostics is not None:
            self._diagnostics.record("scaled_mm_success")
        output = output[: x.shape[0], : self.out_features]
        if input.ndim >= 3:
            output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def _execute(
        self,
        input: torch.Tensor,
        *,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor | None,
        pre_quant_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_input(input)
        recorder = self._diagnostics
        if pre_quant_scale is not None:
            input = input * pre_quant_scale.to(dtype=input.dtype)
        native, route = _nvfp4_route(self.full_precision_matmul, input.device, input.ndim)
        if native:
            output = self._fast_forward(
                input,
                weight,
                weight_scale,
                weight_scale_2,
                input_scale,
                bias,
            )
            if output is not None:
                if recorder is not None:
                    recorder.record("route_native")
                return output
            if recorder is not None:
                recorder.record("route_backend_fallback")
        if recorder is not None and not native:
            recorder.record(route)
        return self._dequant_forward(
            input,
            weight,
            weight_scale,
            weight_scale_2,
            bias,
        )

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._residency
        offloaded = binding is not None and not binding.mechanism.is_loaded(binding.unit)
        if self._diagnostics is not None:
            self._diagnostics.observe_state(self, "offloaded" if offloaded else "loaded")
        if binding is not None and offloaded:
            with self._diagnosed_lease(binding) as lease:
                weight_key = binding.key("weight")
                if binding.mechanism.weight_functions(weight_key):
                    if self._diagnostics is not None:
                        self._diagnostics.record("route_deferred_patch")
                    self._validate_input(input)
                    pre_quant_scale = (
                        None
                        if self.pre_quant_scale is None
                        else lease.get("pre_quant_scale", dtype=self.pre_quant_scale.dtype)
                    )
                    if pre_quant_scale is not None:
                        input = input * pre_quant_scale.to(dtype=input.dtype)
                    bias = (
                        None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)
                    )
                    return torch.nn.functional.linear(
                        input,
                        lease.get("weight", dtype=self.compute_dtype),
                        bias,
                    )
                stored = lease.get_stored("weight")
                if not isinstance(stored, Nvfp4PackedWeight):
                    raise TypeError("Nvfp4Linear residency weight is not folded NVFP4 storage")
                return self._execute(
                    input,
                    weight=stored.qdata,
                    weight_scale=stored.block_scale,
                    weight_scale_2=stored.tensor_scale,
                    input_scale=(
                        None
                        if self.input_scale is None
                        else lease.get("input_scale", dtype=self.input_scale.dtype)
                    ),
                    pre_quant_scale=(
                        None
                        if self.pre_quant_scale is None
                        else lease.get("pre_quant_scale", dtype=self.pre_quant_scale.dtype)
                    ),
                    bias=(None if self.bias is None else lease.get("bias", dtype=self.bias.dtype)),
                )
        return self._execute(
            input,
            weight=self.weight,
            weight_scale=self.weight_scale,
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            pre_quant_scale=self.pre_quant_scale,
            bias=self.bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._diagnostics is None:
            return self._forward(input)
        with self._diagnostics.invocation():
            return self._forward(input)
