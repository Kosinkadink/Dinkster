# dinkster-kernels

Dinkster-owned device kernels behind torch custom operators. Routes:
packed-domain GGUF matmul over the encoded-resident layouts Q4_0,
Q4_K, Q5_K, Q6_K, and Q8_0; the fused RoPE rotation (`apply_rope`);
per-tensor FP8 input quantization (`quantize_per_tensor_fp8`); the
tensor-wise FP8 scaled matmul (`scaled_mm`); and fused INT8 ConvRot weight
dequantization (`dequantize_int8_convrot_weight`) (issue #709).

## Contract

Every triton kernel is a torch custom op (`dinkster_kernels::...`) with a fake
implementation for compile/export and opcheck coverage. Op schemas
register on import with no triton or CUDA requirement; the actual
device implementations import triton lazily. Consumers select routes
through the cached availability probes (`gguf_q4_0_linear_available()`
and its per-layout siblings), which share one host-capability check
and treat a failed triton import or kernel compile - triton's launcher
build needs a host C compiler - as route ineligibility, never an
error. Every consumer keeps a reference fallback; the ops themselves
fail closed on ineligible shapes, dtypes, and devices.

`scaled_mm` is deliberately NOT a custom op: it dispatches to torch's
own scaled-matmul operation (`torch.nn.functional.scaled_mm` from
torch 2.10, `torch._scaled_mm` before it), which is already
compile-visible, and wrapping it would hide the native op from
inductor. Its probe (`scaled_mm_available()`) is a host check only:
it declines HIP runtimes, where dinkster-kitchen's WMMA kernel owns the
route, and hosts with no CUDA device.

`quantize_per_tensor_fp8` accepts float32, float16, and bfloat16 CUDA
inputs with one float32 scale and returns contiguous e4m3fn or e5m2
storage. Its operation-specific availability probe compiles and executes
the kernel before the route is selected. The consumer eligibility
predicate requires at least 256 input elements; smaller calls keep the
Kitchen or eager fallback rather than claiming an unmeasured speedup.

`dequantize_int8_convrot_weight` supports the 256-wide regular Hadamard
layout used by MiniMax H3. It fuses rowwise INT8 dequantization with the
four radix-4 inverse-rotation stages and returns a contiguous float32
weight. Other group sizes remain eligible for consumer fallbacks.

## Numerical contract

FP8 input quantization is bit-identical to dinkster-kitchen's CUDA route,
including saturation, signed zero, and nonfinite values. For finite
inputs it is also bit-identical to the eager float32-divide, clamp, and
cast reference. Eligible input tensors therefore do not move the scaled
matmul result or runtime identity.

ConvRot dequantization is bit-identical to dinkster-kitchen's fused CUDA
route. Each radix-4 stage uses explicit IEEE float32 operations in the
same order, including the exact multiply by 0.5, so the owned route does
not change runtime identity.

The decode ops (`gguf_q4_0_decode` and its per-layout siblings) are
bit-identical to the vectorized torch decoders in
`dinkster_inference_torch.gguf_linear`: each layout's single float32
rounding happens at the same point on both sides (Q8_0's
int8-quant-times-float16-scale product is exact in float32; Q4_0's
nibble-minus-8 offset is exact and its scale multiply is the one
rounding; Q4_K's and Q5_K's quant-times-group-scale and min products
are both exact and their subtraction is the one rounding; Q6_K's
super-scale-times-int8-scale product and quant-minus-32 offset are
exact and the final multiply is the one rounding). The linear ops
decode the same exact values inside the matmul tiles and accumulate in
float32; only tile accumulation order differs from
decode-then-`F.linear`, so their outputs are value-close, not
bit-equal, to that reference. The tile configuration follows the
input row count, so exact float values can also differ between row
counts; for a fixed input shape the kernels are deterministic. Route
selection is therefore recorded by consumers as an identity-visible
fact wherever the fused route can alter default outputs.

## Environments

The workspace root venv is torch-free; torch is an optional extra
(`dinkster-kernels[torch]`). Tests execute meaningfully only in a CUDA
environment (`.venv-gpu`); elsewhere they skip. Type-check on GPU
machines with `.venv/bin/pyright -p packages/dinkster-kernels`, which
resolves against `.venv-gpu`.
