"""Native packed quantized-weight storage.

The reference stores fp8 weights as a comfy-kitchen QuantizedTensor
(fp8 qdata + one float32 per-tensor scale + the compute dtype it
dequantizes to), with quantize/dequantize/requantize semantics defined
by comfy/quant_ops.py _TensorCoreFP8LayoutBase and comfy_kitchen's
TensorCoreFP8Layout @ b78cec87 / kitchen 0.2.31. Dinkster keeps the same
three tensors-worth of state in a plain frozen value type instead of a
torch wrapper subclass: nothing here is dispatched through torch
overrides, so the storage form stays visible at every use site (the
cast-at-use layer dequantizes explicitly).

NVFP4 uses the same explicit package-internal value approach for its
packed qdata, block scale, tensor scale, logical shape, and compute
dtype. Its seeded writeback is the current ComfyUI stochastic
dequantize-patch-requantize algorithm.

INT8 likewise keeps qdata, scale, compute dtype, and ConvRot layout
facts together so patched weights use Kitchen's seeded
dequantize-patch-requantize path without changing their storage form.
Dinkster's fused dequantization is preferred for supported CUDA weights;
Kitchen remains the fallback.

Math, transcribed exactly:

- quantize (scale="recalculate", inplace_ops=True):
  ``scale = amax(|t|).float() / finfo(fp8).max``, with the reference's
  too-small-scale correction when the source dtype is neither float32
  nor bfloat16; then ``t *= (1/scale).to(t.dtype)`` and a seeded
  stochastic rounding to fp8 (seed > 0), or kitchen's
  quantize_per_tensor_fp8 nearest path (seed == 0:
  ``t * (1/scale)`` - reciprocal-multiply, bit-exact with kitchen's
  eager kernel - clamp to the fp8 range, cast).
- dequantize: ``qdata.to(dtype) * scale.to(dtype)``
  (kitchen dequantize_per_tensor_fp8).
- requantize: quantize with a recalculated scale, preserving the
  stored ``orig_dtype`` - the net effect of the reference set_weight's
  ``requantize_from_float(...).to(self.weight.dtype)``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch

from ._nvfp4_diagnostics import Nvfp4DiagnosticsRecorder
from .rounding import stochastic_rounding

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@dataclass(frozen=True)
class Int8PackedWeight:
    """Tensorwise INT8 weight with the layout facts needed for exact writeback."""

    qdata: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    convrot: bool
    convrot_groupsize: int

    def __post_init__(self) -> None:
        if self.qdata.dtype != torch.int8 or self.qdata.ndim != 2:
            raise ValueError("INT8 qdata must be rank-2 int8")
        if self.scale.dtype != torch.float32:
            raise ValueError("INT8 scale must be float32")
        if self.scale.shape not in (torch.Size([]), torch.Size((self.qdata.shape[0], 1))):
            raise ValueError("INT8 scale must be scalar or one value per output row")
        if self.convrot and self.scale.ndim == 0:
            raise ValueError("ConvRot INT8 storage requires one scale per output row")
        if type(self.convrot_groupsize) is not int or (self.convrot and self.convrot_groupsize < 1):
            raise ValueError("INT8 ConvRot group size must be a positive integer when enabled")

    @property
    def shape(self) -> tuple[int, int]:
        return cast(tuple[int, int], tuple(self.qdata.shape))

    @property
    def device(self) -> torch.device:
        return self.qdata.device

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        target = self.orig_dtype if dtype is None else dtype
        if not self.convrot:
            return self.qdata.to(dtype=target) * self.scale.to(dtype=target)
        try:
            kernels = cast(Any, importlib.import_module("dinkster_kernels"))
            if bool(kernels.dequantize_int8_convrot_weight_available()) and bool(
                kernels.dequantize_int8_convrot_weight_supported(
                    self.qdata,
                    self.scale,
                    self.convrot_groupsize,
                )
            ):
                value = kernels.dequantize_int8_convrot_weight(
                    self.qdata,
                    self.scale,
                    self.convrot_groupsize,
                )
                return value.to(dtype=target)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - an optional accelerator is best-effort
            pass
        try:
            importlib.import_module("comfy_kitchen")
            value = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(
                self.qdata,
                self.scale,
                self.convrot_groupsize,
            )
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"comfy-kitchen ConvRot INT8 dequantization failed: {error}"
            ) from error
        return value.to(dtype=target)


@dataclass(frozen=True)
class Fp8ScaledWeight:
    """One weight stored in fp8: quantized data, a 0-dim float32
    per-tensor scale, and the dtype it dequantizes to by default
    (the model's compute dtype - QuantizedTensor.params.orig_dtype in
    the reference)."""

    qdata: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.qdata.dtype not in FP8_DTYPES:
            raise ValueError(f"qdata must be an fp8 tensor, got {self.qdata.dtype}")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 1:
            raise ValueError(
                "scale must be a single float32 value, got"
                f" {self.scale.dtype} with {self.scale.numel()} elements"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """The weight's logical shape (SizedTensor protocol)."""
        return tuple(self.qdata.shape)

    @property
    def device(self) -> torch.device:
        return self.qdata.device

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """``qdata.to(dtype) * scale.to(dtype)`` (kitchen
        dequantize_per_tensor_fp8); ``dtype`` defaults to
        ``orig_dtype``. This IS the reference quantized patch-flow
        prologue: casting the wrapper to the intermediate dtype and
        convert_func-dequantizing collapses to exactly this product."""
        target = self.orig_dtype if dtype is None else dtype
        return self.qdata.to(dtype=target) * self.scale.to(dtype=target)


@dataclass(frozen=True)
class Nvfp4PackedWeight:
    """Package-internal packed NVFP4 weight and its complete scale state."""

    qdata: torch.Tensor
    block_scale: torch.Tensor
    tensor_scale: torch.Tensor
    logical_shape: tuple[int, int]
    orig_dtype: torch.dtype
    recorder: Nvfp4DiagnosticsRecorder | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows, columns = self.logical_shape
        if self.qdata.dtype != torch.uint8 or self.qdata.ndim != 2:
            raise ValueError("NVFP4 qdata must be rank-2 uint8")
        if tuple(self.qdata.shape) != (_round_up(rows, 16), _round_up(columns, 16) // 2):
            raise ValueError("NVFP4 qdata shape does not match logical shape")
        expected = (_round_up(rows, 128), _round_up(columns // 16, 4))
        if (
            self.block_scale.dtype != torch.float8_e4m3fn
            or tuple(self.block_scale.shape) != expected
        ):
            raise ValueError("NVFP4 block-scale state has invalid dtype or shape")
        if self.tensor_scale.dtype != torch.float32 or self.tensor_scale.shape != torch.Size([]):
            raise ValueError("NVFP4 tensor scale must be scalar float32")

    @property
    def shape(self) -> tuple[int, int]:
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.qdata.device

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        kitchen = cast(Any, importlib.import_module("comfy_kitchen"))
        target = self.orig_dtype if dtype is None else dtype
        try:
            value = kitchen.dequantize_nvfp4(
                self.qdata, self.tensor_scale, self.block_scale, output_type=target
            )
        except BaseException:
            if self.recorder is not None:
                self.recorder.record("dequantize_error")
            raise
        if self.recorder is not None:
            self.recorder.record("dequantize_success")
        return value[: self.logical_shape[0], : self.logical_shape[1]]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def requantize_int8(
    stored: Int8PackedWeight, tensor: torch.Tensor, *, seed: int = 0
) -> Int8PackedWeight:
    """Recalculate scales and requantize while preserving the INT8 layout."""
    kitchen_tensor = cast(Any, importlib.import_module("comfy_kitchen.tensor"))
    qdata, params = kitchen_tensor.TensorWiseINT8Layout.quantize(
        tensor,
        scale="recalculate",
        stochastic_rounding=seed,
        inplace_ops=True,
        is_weight=True,
        per_channel=stored.scale.ndim > 0,
        convrot=stored.convrot,
        convrot_groupsize=stored.convrot_groupsize,
    )
    return replace(stored, qdata=qdata, scale=params.scale)


def requantize_nvfp4(
    stored: Nvfp4PackedWeight, tensor: torch.Tensor, *, seed: int = 0
) -> Nvfp4PackedWeight:
    """Recalculate scales and requantize while preserving logical identity."""
    scale = (torch.amax(tensor.abs()) / (448.0 * 6.0)).to(torch.float32)
    try:
        if seed == 0:
            kitchen = cast(Any, importlib.import_module("comfy_kitchen"))
            qdata, block = kitchen.quantize_nvfp4(tensor, scale, pad_16x=True)
        else:
            qdata, block = _stochastic_quantize_nvfp4(tensor, scale, seed)
    except BaseException:
        if stored.recorder is not None:
            stored.recorder.record("requantize_error")
        raise
    if stored.recorder is not None:
        stored.recorder.record("requantize_success")
    return replace(stored, qdata=qdata, block_scale=block, tensor_scale=scale)


def _stochastic_float_to_fp4_e2m1(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shape = tensor.shape
    sign = torch.signbit(tensor).to(torch.uint8)
    exponent = torch.floor(torch.log2(tensor.abs()) + 1.0).clamp(0, 3)
    value = (
        tensor
        + (
            torch.rand(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
                generator=generator,
            )
            - 0.5
        )
        * (2 ** (exponent - 2.0))
        * 1.25
    )
    value = value.abs()
    exponent = torch.floor(torch.log2(value) + 1.1925).clamp(0, 3)
    mantissa = (
        torch.where(
            exponent > 0,
            (value / (2.0 ** (exponent - 1)) - 1.0) * 2.0,
            value * 2.0,
            out=value,
        )
        .round()
        .to(torch.uint8)
    )
    fp4 = (sign << 3) | (exponent.to(torch.uint8) << 1) | mantissa
    flat = fp4.view(-1)
    return ((flat[0::2] << 4) | flat[1::2]).reshape(list(shape)[:-1] + [-1])


def _to_nvfp4_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = input_matrix.shape
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows, padded_columns = row_blocks * 128, column_blocks * 4
    if (rows, columns) != (padded_rows, padded_columns):
        padded = torch.zeros(
            (padded_rows, padded_columns),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :columns] = input_matrix
        input_matrix = padded
    blocks = input_matrix.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(padded_rows, padded_columns)


def _stochastic_quantize_nvfp4(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    seed: int,
    *,
    block_size: int = 4096 * 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = tensor.shape
    padded_rows, padded_columns = _round_up(rows, 16), _round_up(columns, 16)
    if (rows, columns) != (padded_rows, padded_columns):
        tensor = torch.nn.functional.pad(
            tensor, (0, padded_columns - columns, 0, padded_rows - rows)
        )
    original_shape = tensor.shape
    qdata = torch.empty(
        (*original_shape[:-1], original_shape[-1] // 2),
        dtype=torch.uint8,
        device=tensor.device,
    )
    block_scale = torch.empty(
        (*original_shape[:-1], original_shape[-1] // 16),
        dtype=torch.float8_e4m3fn,
        device=tensor.device,
    )
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    num_slices = max(1, tensor.numel() / block_size)
    slice_size = max(1, round(tensor.shape[0] / num_slices))
    for start in range(0, tensor.shape[0], slice_size):
        chunk = tensor[start : start + slice_size]
        blocked = chunk.reshape(chunk.shape[0], -1, 16)
        chunk_scale = torch.clamp(
            (torch.amax(torch.abs(blocked), dim=-1) / 6.0) / scale.to(chunk.dtype),
            max=448.0,
        ).to(torch.float8_e4m3fn)
        blocked = blocked / (scale.to(chunk.dtype) * chunk_scale.to(chunk.dtype)).unsqueeze(-1)
        qdata[start : start + slice_size].copy_(
            _stochastic_float_to_fp4_e2m1(blocked.view(chunk.shape).nan_to_num(), generator)
        )
        block_scale[start : start + slice_size].copy_(chunk_scale)
    return qdata, _to_nvfp4_blocked(block_scale)


def quantize_fp8_scaled(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
    *,
    seed: int = 0,
) -> Fp8ScaledWeight:
    """Quantize ``tensor`` to fp8 with a recalculated per-tensor scale
    (_TensorCoreFP8LayoutBase.quantize @ b78cec87, scale="recalculate",
    inplace_ops=True). ``seed > 0`` rounds stochastically (the
    load-bearing writeback path); ``seed == 0`` is the nearest path.

    Callers pass an owned buffer: like the reference with
    inplace_ops=True, the stochastic path scales ``tensor`` in place.
    """
    if dtype not in FP8_DTYPES:
        raise ValueError(f"not an fp8 dtype: {dtype}")

    orig_dtype = tensor.dtype
    fp8_max = torch.finfo(dtype).max
    scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / fp8_max
    if tensor.dtype not in (torch.float32, torch.bfloat16):
        # Prevent the scale from underflowing the source dtype
        # (reference comment: "Prevent scale from being too small").
        info = torch.finfo(tensor.dtype)
        scale = 1.0 / torch.clamp(1.0 / scale, min=info.min, max=info.max)

    if seed > 0:
        tensor *= (1.0 / scale).to(tensor.dtype)
        qdata = stochastic_rounding(tensor, dtype, seed=seed)
    else:
        # kitchen quantize_per_tensor_fp8 (eager): reciprocal-multiply,
        # clamp to the fp8 range, cast.
        temp = tensor * (1.0 / scale).to(tensor.dtype)
        temp = torch.clamp(temp, -fp8_max, fp8_max, out=temp)
        qdata = temp.to(dtype)

    return Fp8ScaledWeight(qdata, scale.float(), orig_dtype)


def requantize_fp8_scaled(
    stored: Fp8ScaledWeight,
    tensor: torch.Tensor,
    *,
    seed: int = 0,
) -> Fp8ScaledWeight:
    """Quantize ``tensor`` back into ``stored``'s storage form: fresh
    recalculated scale, same fp8 dtype, ``orig_dtype`` preserved from
    ``stored`` (the reference set_weight requantize_from_float +
    ``.to(self.weight.dtype)``). Mutates ``tensor`` like
    quantize_fp8_scaled."""
    fresh = quantize_fp8_scaled(tensor, stored.qdata.dtype, seed=seed)
    return replace(fresh, orig_dtype=stored.orig_dtype)


__all__ = [
    "FP8_DTYPES",
    "Fp8ScaledWeight",
    "quantize_fp8_scaled",
    "requantize_fp8_scaled",
]
