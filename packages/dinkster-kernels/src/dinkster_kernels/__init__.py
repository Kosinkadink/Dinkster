"""Dinkster-owned device kernels behind torch custom operators.

Every triton kernel is exposed as a torch custom op with a fake
implementation for compile/export, plus a cached availability probe
that treats a failed triton import or kernel compile as ineligibility,
never an error. Routes that dispatch to native torch ops
(:mod:`scaled_mm`) stay plain functions so inductor sees the
underlying op. Consumers select routes through the probes and keep a
reference fallback, so this package is safe to import (and its op
schemas register) on hosts with no CUDA device or triton at all.
"""

from .fp8_quantize import (
    quantize_per_tensor_fp8,
    quantize_per_tensor_fp8_available,
    quantize_per_tensor_fp8_supported,
)
from .gguf_q4_0 import (
    Q4_0_BLOCK_BYTES,
    Q4_0_BLOCK_ELEMENTS,
    gguf_q4_0_decode,
    gguf_q4_0_linear,
    gguf_q4_0_linear_available,
)
from .gguf_q4_k import (
    Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMENTS,
    gguf_q4_k_decode,
    gguf_q4_k_linear,
    gguf_q4_k_linear_available,
)
from .gguf_q5_k import (
    Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS,
    gguf_q5_k_decode,
    gguf_q5_k_linear,
    gguf_q5_k_linear_available,
)
from .gguf_q6_k import (
    Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS,
    gguf_q6_k_decode,
    gguf_q6_k_linear,
    gguf_q6_k_linear_available,
)
from .gguf_q8_0 import (
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
    gguf_q8_0_decode,
    gguf_q8_0_linear,
    gguf_q8_0_linear_available,
)
from .int8_convrot import (
    CONVROT_GROUP_SIZE,
    dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_available,
    dequantize_int8_convrot_weight_supported,
)
from .rope import (
    apply_rope,
    apply_rope_available,
    apply_rope_supported,
)
from .scaled_mm import (
    scaled_mm,
    scaled_mm_available,
)

__all__ = [
    "CONVROT_GROUP_SIZE",
    "Q4_0_BLOCK_BYTES",
    "Q4_0_BLOCK_ELEMENTS",
    "Q4_K_BLOCK_BYTES",
    "Q4_K_BLOCK_ELEMENTS",
    "Q5_K_BLOCK_BYTES",
    "Q5_K_BLOCK_ELEMENTS",
    "Q6_K_BLOCK_BYTES",
    "Q6_K_BLOCK_ELEMENTS",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_ELEMENTS",
    "apply_rope",
    "apply_rope_available",
    "apply_rope_supported",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_available",
    "dequantize_int8_convrot_weight_supported",
    "gguf_q4_0_decode",
    "gguf_q4_0_linear",
    "gguf_q4_0_linear_available",
    "gguf_q4_k_decode",
    "gguf_q4_k_linear",
    "gguf_q4_k_linear_available",
    "gguf_q5_k_decode",
    "gguf_q5_k_linear",
    "gguf_q5_k_linear_available",
    "gguf_q6_k_decode",
    "gguf_q6_k_linear",
    "gguf_q6_k_linear_available",
    "gguf_q8_0_decode",
    "gguf_q8_0_linear",
    "gguf_q8_0_linear_available",
    "quantize_per_tensor_fp8",
    "quantize_per_tensor_fp8_available",
    "quantize_per_tensor_fp8_supported",
    "scaled_mm",
    "scaled_mm_available",
]
