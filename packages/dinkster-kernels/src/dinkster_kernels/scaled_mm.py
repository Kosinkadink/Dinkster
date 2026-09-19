"""Tensor-wise fp8 scaled matmul as a Dinkster-owned route.

``scaled_mm`` is a plain function, not a custom op: the operations it
dispatches to (``torch.nn.functional.scaled_mm`` from torch 2.10,
``torch._scaled_mm`` before it) are already compile-visible torch ops,
and wrapping them in a custom op would hide the native op from
inductor. Dispatch follows the installed torch: the public functional
surface with tensor-wise scale recipes and ``use_fast_accum=False``
where it exists, otherwise the legacy private op with its torch-2.4
tuple return normalized away. Both arms run the same underlying
cuBLASLt path, so for a fixed torch build the dispatch choice does not
move bits. The functional surface resolves once at import (a
function-local import would risk a graph break inside fullgraph
compiled forwards).

:func:`scaled_mm_available` gates the route to CUDA hosts and declines
HIP runtimes, where dinkster-kitchen serves the fp8 matmul through its
own WMMA kernel; consumers keep kitchen and eager-torch fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import torch


class _FunctionalScaledMm(NamedTuple):
    """torch 2.10+'s public scaled-mm surface with its recipe enums."""

    op: Callable[..., torch.Tensor]
    tensor_wise: object
    no_swizzle: object


def _resolve_functional_scaled_mm() -> _FunctionalScaledMm | None:
    functional = cast(Any, torch.nn.functional)
    op = getattr(functional, "scaled_mm", None)
    if not callable(op):
        return None
    return _FunctionalScaledMm(
        op=cast(Callable[..., torch.Tensor], op),
        tensor_wise=functional.ScalingType.TensorWise,
        no_swizzle=functional.SwizzleType.NO_SWIZZLE,
    )


_FUNCTIONAL_SCALED_MM = _resolve_functional_scaled_mm()


def scaled_mm_available() -> bool:
    """Whether the owned fp8 scaled-matmul route can serve this host.

    False on HIP runtimes (dinkster-kitchen's WMMA kernel owns that
    route) and on hosts with no CUDA device. Per-operand eligibility
    (fp8 dtypes, compute capability) stays with the consumer's
    capability checks.
    """
    if getattr(torch.version, "hip", None) is not None:
        return False
    return torch.cuda.is_available()


def scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Tensor-wise scaled matmul of fp8 operands.

    ``input`` is (M, K) fp8 row-major, ``weight`` is (K, N) fp8 (the
    caller passes its untransposed (N, K) qdata as ``weight.t()``),
    and the scales are float32 scalars. ``bias`` fuses into the
    epilogue only for half-precision ``out_dtype`` (cuBLASLt's
    contract); callers add a float32 bias after the matmul.
    """
    functional = _FUNCTIONAL_SCALED_MM
    if functional is not None:
        return functional.op(
            input,
            weight,
            scale_a=scale_a,
            scale_recipe_a=functional.tensor_wise,
            scale_b=scale_b,
            scale_recipe_b=functional.tensor_wise,
            swizzle_a=functional.no_swizzle,
            swizzle_b=functional.no_swizzle,
            bias=bias,
            output_dtype=out_dtype,
            use_fast_accum=False,
        )
    output = torch._scaled_mm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        input,
        weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )
    # torch 2.4 returned (output, output_amax); 2.5+ returns the
    # output tensor alone.
    return output[0] if isinstance(output, tuple) else output
