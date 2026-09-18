"""Linear execution over encoded GGUF quantized blocks.

The memory and balanced residency modes keep quantized linear weights
as their encoded GGUF blocks (uint8 rows, one per block) and decode
them on use. The vectorized decoders here are bit-identical to the
pure reference decoders in dinkster_inference.gguf: the reference rounds
every intermediate to float32, and each of its roundings corresponds
to exactly one torch float32 operation whose other intermediates are
exactly representable (a float16 scale times a small integer carries
at most about 22 significant bits), so both sides round the same
exact values at the same points. The eager loader shares these
decoders, so every residency mode produces bit-identical weights.
The balanced mode additionally holds decoded weights in a budgeted
sticky cache so repeated forwards skip the decode.

A fused route (:meth:`GgufEncodedLinear.bind_fused_matmul`)
executes forwards over fused-supported layouts (Q4_0, Q4_K, Q5_K,
Q6_K, and Q8_0) through dinkster-kernels' packed-domain matmul, which decodes blocks
inside the kernel tiles instead of materializing the weight. Its
outputs are value-close, not bit-equal, to the decode route (only
tile accumulation order differs). A bound layer executes fused only
on CUDA blocks and only at or below the layout's
:data:`FUSED_MATMUL_MAX_TOKENS` threshold; everything else falls back
to the decode route unchanged. Assembly binds the route by default
for memory-residency layers that pass the capability check
(:meth:`GgufEncodedLinear.bind_default_fused_matmul`); other layers
stay on the decode route unless bound explicitly.
"""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import torch
from dinkster_inference import builtin_gguf_storage_registry

from .residency_timing import COMPUTE, DEQUANT, timed_phase

log = logging.getLogger("dinkster.inference_torch.gguf_linear")

if TYPE_CHECKING:
    from .module_residency import ResidencyBinding

__all__ = [
    "FUSED_MATMUL_MAX_TOKENS",
    "GGUF_BLOCK_DECODERS",
    "GGUF_BLOCK_SHAPES",
    "GgufDecodedCache",
    "GgufDecodedCacheResidency",
    "GgufEncodedLinear",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_ELEMENTS",
    "decode_q4_0_blocks",
    "decode_q4_k_blocks",
    "decode_q5_k_blocks",
    "decode_q6_k_blocks",
    "decode_q8_0_blocks",
    "synthetic_gguf_blocks",
]

#: (block_elements, block_bytes) for every encoded-resident GGML
#: layout, keyed by GGML type name and derived from the storage
#: registry so the executor cannot drift from the layouts admission
#: accepts.
GGUF_BLOCK_SHAPES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        layout.ggml_type.name: (layout.ggml_type.block_elements, layout.ggml_type.block_bytes)
        for layout in builtin_gguf_storage_registry()
    }
)

Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES = GGUF_BLOCK_SHAPES["Q8_0"]

_FUSED_COMPUTE_DTYPES = (torch.float16, torch.bfloat16)


class _FusedGgufLinear(Protocol):
    def __call__(
        self,
        input: torch.Tensor,
        blocks: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor: ...


#: GGML type -> (availability probe, linear op) attribute names on
#: dinkster_kernels for the layouts with a fused matmul route.
_FUSED_OP_NAMES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "Q4_0": ("gguf_q4_0_linear_available", "gguf_q4_0_linear"),
        "Q4_K": ("gguf_q4_k_linear_available", "gguf_q4_k_linear"),
        "Q5_K": ("gguf_q5_k_linear_available", "gguf_q5_k_linear"),
        "Q6_K": ("gguf_q6_k_linear_available", "gguf_q6_k_linear"),
        "Q8_0": ("gguf_q8_0_linear_available", "gguf_q8_0_linear"),
    }
)

#: Largest flattened token count the fused route executes per layout;
#: larger inputs take the decode route, whose cuBLAS matmul wins once
#: compute dominates the decode cost. Each threshold is the largest
#: swept token count with no measured shape regression beyond noise
#: (RTX 5090, attn 4096x4096 / ffn 10240x4096 and 4096x10240 /
#: diffusion 3072x3072; fused wins grow toward small token counts and
#: losses grow beyond the threshold).
FUSED_MATMUL_MAX_TOKENS: Mapping[str, int] = MappingProxyType(
    {
        "Q4_0": 1024,
        "Q4_K": 128,
        "Q5_K": 16,
        "Q6_K": 512,
        "Q8_0": 1024,
    }
)

_fused_linear_ops: dict[str, _FusedGgufLinear | None] = {}


def _probe_fused_gguf_linear(ggml_type: str) -> _FusedGgufLinear | None:
    """Lazily resolve dinkster-kernels' fused linear op for ``ggml_type``."""
    if ggml_type in _fused_linear_ops:
        return _fused_linear_ops[ggml_type]
    op: _FusedGgufLinear | None = None
    try:
        available_name, op_name = _FUSED_OP_NAMES[ggml_type]
        module = cast(Any, importlib.import_module("dinkster_kernels"))
        if bool(getattr(module, available_name)()):
            op = cast(_FusedGgufLinear, getattr(module, op_name))
    except Exception:  # noqa: BLE001 - an optional accelerator probe is best-effort
        op = None
    _fused_linear_ops[ggml_type] = op
    return op


def _fused_gguf_linear(
    op: _FusedGgufLinear | None,
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    """Dispatch to the fused packed-domain matmul kernel."""
    if torch.is_grad_enabled() and input.requires_grad:
        raise RuntimeError(
            "the fused GGUF matmul route is inference-only (the kernel has"
            " no backward); run under no_grad, or leave bind_fused_matmul"
            " off for gradient work"
        )
    if op is None:
        raise RuntimeError("fused GGUF matmul route used without a successful bind_fused_matmul")
    return op(input, blocks, bias, out_features)


def _fp16_column(blocks: torch.Tensor, offset: int) -> torch.Tensor:
    """Read the float16 at byte ``offset`` of every block as float32, shape (n, 1)."""

    return blocks[:, offset : offset + 2].contiguous().view(torch.float16).to(torch.float32)


def decode_q8_0_blocks(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode Q8_0 blocks (uint8, shape (n, 34)) to a float32 tensor."""

    scales = _fp16_column(blocks, 0)
    quants = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    return (quants * scales).reshape(logical_shape)


def decode_q4_0_blocks(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode Q4_0 blocks (uint8, shape (n, 18)) to a float32 tensor.

    Each block is a float16 scale and 16 packed bytes; the 16 low
    nibbles decode first, then the 16 high nibbles, each as
    scale * (nibble - 8).
    """

    scales = _fp16_column(blocks, 0)
    packed = blocks[:, 2:]
    quants = torch.cat((packed & 0x0F, packed >> 4), dim=1).to(torch.float32) - 8.0
    return (quants * scales).reshape(logical_shape)


def _kquant_scale_mins(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a K-quant super-block's eight 6-bit (scale, min) pairs.

    Returns the effective per-group scales ``d * sc`` and offsets
    ``dmin * mn`` as float32 tensors of shape (n, 8). Both products
    are exact in float32 (float16 times a 6-bit integer).
    """

    d = _fp16_column(blocks, 0)
    dmin = _fp16_column(blocks, 2)
    packed = blocks[:, 4:16]
    low, mid, high = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    sc = torch.cat((low & 0x3F, (high & 0x0F) | ((low >> 6) << 4)), dim=1)
    mn = torch.cat((mid & 0x3F, (high >> 4) | ((mid >> 6) << 4)), dim=1)
    return d * sc.to(torch.float32), dmin * mn.to(torch.float32)


def decode_q4_k_blocks(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode Q4_K super-blocks (uint8, shape (n, 144)) to a float32 tensor.

    Each of the four 32-byte chunks yields its 32 low nibbles under
    one (scale, min) pair, then its 32 high nibbles under the next.
    The quant-times-scale product is exact in float32, so the single
    rounding happens at the subtraction, exactly where the reference
    decoder rounds.
    """

    group_scales, group_mins = _kquant_scale_mins(blocks)
    quants = blocks[:, 16:].reshape(-1, 4, 1, 32)
    q = torch.cat((quants & 0x0F, quants >> 4), dim=2).to(torch.float32)
    scales = group_scales.reshape(-1, 4, 2, 1)
    mins = group_mins.reshape(-1, 4, 2, 1)
    return (q * scales - mins).reshape(logical_shape)


def decode_q5_k_blocks(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode Q5_K super-blocks (uint8, shape (n, 176)) to a float32 tensor.

    Like Q4_K, plus a fifth quant bit per element taken from the
    32-byte high-bit plane: chunk c's low nibbles use bit 2c and its
    high nibbles bit 2c + 1.
    """

    group_scales, group_mins = _kquant_scale_mins(blocks)
    high = blocks[:, 16:48].reshape(-1, 1, 1, 32)
    low = blocks[:, 48:].reshape(-1, 4, 1, 32)
    shifts = torch.arange(8, dtype=torch.uint8, device=blocks.device).reshape(1, 4, 2, 1)
    bits = (high >> shifts) & 1
    q = (torch.cat((low & 0x0F, low >> 4), dim=2) | (bits << 4)).to(torch.float32)
    scales = group_scales.reshape(-1, 4, 2, 1)
    mins = group_mins.reshape(-1, 4, 2, 1)
    return (q * scales - mins).reshape(logical_shape)


def decode_q6_k_blocks(blocks: torch.Tensor, logical_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode Q6_K super-blocks (uint8, shape (n, 210)) to a float32 tensor.

    Each half of a super-block interleaves 128 six-bit quants from a
    64-byte nibble plane and a 32-byte two-bit plane, scaled by the
    float16 super-scale times one of sixteen int8 group scales. The
    super-scale product is exact in float32; the final multiply is
    the one reference rounding.
    """

    low = blocks[:, :128].reshape(-1, 2, 2, 32)
    high = blocks[:, 128:192].reshape(-1, 2, 32)
    scales = blocks[:, 192:208].contiguous().view(torch.int8)
    d = _fp16_column(blocks, 208)
    low_a, low_b = low[:, :, 0], low[:, :, 1]
    q = (
        torch.stack(
            (
                (low_a & 0x0F) | ((high & 3) << 4),
                (low_b & 0x0F) | (((high >> 2) & 3) << 4),
                (low_a >> 4) | (((high >> 4) & 3) << 4),
                (low_b >> 4) | (((high >> 6) & 3) << 4),
            ),
            dim=2,
        ).to(torch.float32)
        - 32.0
    )
    group_scales = scales.reshape(-1, 2, 4, 2).to(torch.float32).repeat_interleave(16, dim=3)
    return ((d.reshape(-1, 1, 1, 1) * group_scales) * q).reshape(logical_shape)


#: Vectorized decoders for every GGML layout admitted as encoded storage,
#: keyed by GGML type name. Each takes (blocks, logical_shape) where
#: ``blocks`` is uint8 of shape (block_count, block_bytes).
GGUF_BLOCK_DECODERS: Mapping[str, Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]] = (
    MappingProxyType(
        {
            "Q4_0": decode_q4_0_blocks,
            "Q4_K": decode_q4_k_blocks,
            "Q5_K": decode_q5_k_blocks,
            "Q6_K": decode_q6_k_blocks,
            "Q8_0": decode_q8_0_blocks,
        }
    )
)

#: Byte offsets of every float16 scale field per encoded layout. Only
#: these bytes can push a decoded value to inf or NaN; every other byte
#: holds integer quant or packing state.
_FP16_SCALE_OFFSETS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "Q4_0": (0,),
        "Q8_0": (0,),
        "Q4_K": (0, 2),
        "Q5_K": (0, 2),
        "Q6_K": (208,),
    }
)

#: Finite float16 scale values cycled through the synthetic blocks:
#: zero, the smallest subnormal, the smallest normal, ordinary
#: magnitudes of both signs, and the float16 maximum.
_FINITE_FP16_SCALES = (
    0.0,
    5.9604644775390625e-08,
    6.103515625e-05,
    0.5,
    1.0,
    -1.5,
    3.140625,
    -448.0,
    65504.0,
)


def synthetic_gguf_blocks(type_name: str, count: int, *, seed: int) -> torch.Tensor:
    """Deterministic encoded CPU blocks whose decoded values are finite.

    A backend parity fixture: quant, scale-pack, and plane bytes take
    arbitrary values while every float16 scale field cycles through
    finite edge scales, so the same blocks can be decoded on the CPU
    and on an accelerator and compared exactly (NaN never compares
    equal). Returns uint8 of shape (count, block_bytes).
    """
    if type_name not in GGUF_BLOCK_SHAPES:
        raise ValueError(f"no encoded GGUF layout named {type_name!r}")
    _, block_bytes = GGUF_BLOCK_SHAPES[type_name]
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randint(0, 256, (count, block_bytes), dtype=torch.uint8, generator=generator)
    offsets = _FP16_SCALE_OFFSETS[type_name]
    for row in range(count):
        for position, offset in enumerate(offsets):
            value = _FINITE_FP16_SCALES[(row * len(offsets) + position) % len(_FINITE_FP16_SCALES)]
            encoded = torch.tensor([value], dtype=torch.float16).view(torch.uint8)
            blocks[row, offset : offset + 2] = encoded
    return blocks


class GgufDecodedCache:
    """Budgeted sticky cache of decoded GGUF linear weights.

    Entries fill in first-use order and stay resident until evicted
    through :meth:`free_bytes`, :meth:`clear`, or allocation recovery;
    an entry that does not fit is skipped without displacing an existing
    one. Encoded-resident linears all run on every forward pass, so the decode
    cost saved per cached byte is uniform across layers and
    replacement churn (LRU on a cyclic access pattern evicts every
    entry right before its reuse) can only lose. A refused key is
    offered again on its next forward, so the cache fills later when
    memory frees up.

    A budget of None admits by live measurement: an offer is kept
    only while the offered weight's device retains the default
    inference working reserve in free memory. The decoded weight
    already exists when it is offered, so the measurement includes
    it, and every auto cache on a device - across components,
    assemblies, and models - admits against the same live figure
    without shared state or overcommit.
    """

    def __init__(self, budget: int | None) -> None:
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("decoded cache budget must be a non-negative byte count or None")
        self._budget = budget
        self._used = 0
        self._entries: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    @property
    def budget_bytes(self) -> int | None:
        """The configured byte budget, or None for live auto admission."""
        return self._budget

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    def _admits(self, device: torch.device, nbytes: int) -> bool:
        if self._budget is not None:
            return self._used + nbytes <= self._budget
        from .memory import MemoryPolicy, get_free_memory

        free = get_free_memory(device).free_total
        return int(free) >= MemoryPolicy().minimum_inference_memory()

    def get(self, key: str) -> torch.Tensor | None:
        with self._lock:
            return self._entries.get(key)

    def offer(self, key: str, weight: torch.Tensor) -> torch.Tensor:
        """Insert ``weight`` under ``key`` if the budget admits it.

        Returns the cached tensor for ``key`` - the already-resident
        entry when one exists, otherwise ``weight`` itself whether or
        not it was admitted.
        """
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing
            nbytes = weight.numel() * weight.element_size()
            if self._admits(weight.device, nbytes):
                self._entries[key] = weight
                self._used += nbytes
            return weight

    def free_bytes(self, size: int) -> int:
        """Evict entries in reverse insertion order until at least
        ``size`` bytes are freed; returns the bytes actually freed.
        In-flight forwards keep their decoded tensor alive through
        their own reference, so eviction never corrupts a consumer."""
        freed = 0
        with self._lock:
            for key in reversed(list(self._entries)):
                if freed >= size:
                    break
                weight = self._entries.pop(key)
                nbytes = weight.numel() * weight.element_size()
                self._used -= nbytes
                freed += nbytes
        return freed

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._entries.clear()
            self._used = 0


class GgufEncodedLinear(torch.nn.Module):
    """Linear whose weight stays resident as encoded GGUF quantized blocks."""

    weight_blocks: torch.Tensor
    _residency: ResidencyBinding | None = None
    _fused_op: _FusedGgufLinear | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        ggml_type: str = "Q8_0",
        bias: bool,
        compute_dtype: torch.dtype,
        decoded_cache: GgufDecodedCache | None = None,
        cache_key: str = "",
    ) -> None:
        super().__init__()
        shape = GGUF_BLOCK_SHAPES.get(ggml_type)
        if shape is None:
            supported = ", ".join(GGUF_BLOCK_SHAPES)
            raise ValueError(
                f"GGML type {ggml_type!r} has no encoded-resident layout (supported: {supported})"
            )
        block_elements, block_bytes = shape
        elements = out_features * in_features
        if elements <= 0 or elements % block_elements:
            raise ValueError(
                f"{ggml_type} linear weight of {out_features}x{in_features} does not"
                f" split into {block_elements}-element blocks"
            )
        if decoded_cache is not None and not cache_key:
            raise ValueError("a decoded cache requires a non-empty cache key")
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type = ggml_type
        self._decode = GGUF_BLOCK_DECODERS[ggml_type]
        self.compute_dtype = compute_dtype
        self.decoded_cache = decoded_cache
        self.cache_key = cache_key
        #: Enabled by :meth:`bind_fused_matmul` after a capability check;
        #: never flipped inside forward (compile discipline).
        self.fused_matmul = False
        self._fused_max_tokens = FUSED_MATMUL_MAX_TOKENS.get(ggml_type, 0)
        self.register_buffer(
            "weight_blocks",
            torch.empty((elements // block_elements, block_bytes), dtype=torch.uint8),
        )
        if bias:
            self.bias: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        # A device or dtype move invalidates cached decoded weights;
        # drop the shared cache so forwards re-decode from the moved
        # blocks instead of consuming a stale-device tensor.
        if self.decoded_cache is not None:
            self.decoded_cache.clear()
        return super()._apply(fn, recurse)

    def bind_fused_matmul(self, enabled: bool) -> None:
        """Choose the matmul route BEFORE the first forward. Enabling
        requires a fused-supported layout whose rows split into whole
        blocks (``in_features`` divisible by the layout's block
        elements), a float16 or bfloat16 compute dtype, and a working
        dinkster-kernels probe (CUDA device plus a triton kernel
        compile). The fused route is inference-only: a forward on a
        gradient-requiring input raises; gradient work stays on the
        decode route. Forwards over CPU blocks, and forwards whose
        flattened token count exceeds the layout's
        :data:`FUSED_MATMUL_MAX_TOKENS` threshold (where the decode
        route's cuBLAS matmul measures faster), fall back to the
        decode route unchanged."""
        if enabled:
            if self.ggml_type not in _FUSED_OP_NAMES:
                supported = " or ".join(_FUSED_OP_NAMES)
                log.warning(
                    "fused GGUF matmul supports %s, got %s; falling back to decode matmul",
                    supported,
                    self.ggml_type,
                )
                enabled = False
            block_elements, _ = GGUF_BLOCK_SHAPES[self.ggml_type]
            if enabled and self.in_features % block_elements:
                log.warning(
                    "fused %s matmul needs in_features divisible by %s, got %s; "
                    "falling back to decode matmul",
                    self.ggml_type,
                    block_elements,
                    self.in_features,
                )
                enabled = False
            if enabled and self.compute_dtype not in _FUSED_COMPUTE_DTYPES:
                log.warning(
                    "fused %s matmul computes at float16 or bfloat16, got %s; "
                    "falling back to decode matmul",
                    self.ggml_type,
                    self.compute_dtype,
                )
                enabled = False
            # Seeding the lazy probe here also keeps its importlib call
            # out of a cold compiled forward (graph-break discipline).
            op = _probe_fused_gguf_linear(self.ggml_type) if enabled else None
            if enabled and op is None:
                log.warning(
                    "the fused GGUF matmul route is unavailable on this host "
                    "(dinkster-kernels probe failed: no CUDA device, no triton, "
                    "or no host C compiler); falling back to decode matmul"
                )
                enabled = False
            if enabled:
                assert op is not None
                self._fused_op = op
        if not enabled:
            self._fused_op = None
        self.fused_matmul = enabled

    def bind_default_fused_matmul(self) -> bool:
        """Bind the fused route when this layer and host support it and
        report whether it bound; an unsupported layout, unsplittable
        rows, non-half compute dtype, or failed kernel probe keeps the
        decode route instead of raising. This is the default-enablement
        entry: assembly calls it for memory-residency layers, where the
        decode route pays a full weight decode on every forward."""
        if self.ggml_type not in _FUSED_OP_NAMES:
            return False
        block_elements, _ = GGUF_BLOCK_SHAPES[self.ggml_type]
        if self.in_features % block_elements:
            return False
        if self.compute_dtype not in _FUSED_COMPUTE_DTYPES:
            return False
        if _probe_fused_gguf_linear(self.ggml_type) is None:
            return False
        self.bind_fused_matmul(True)
        return True

    def _fused_route_admits(self, input: torch.Tensor) -> bool:
        # Flattened token count at or below the layout threshold;
        # written multiplication-form so compile guards stay integral.
        return input.numel() <= self._fused_max_tokens * self.in_features

    def bind_residency(self, binding: ResidencyBinding) -> None:
        self._residency = binding

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._residency
        if binding is None or binding.mechanism.is_loaded(binding.unit):
            return None
        requests: list[tuple[str, torch.dtype | None]] = [(binding.key("weight_blocks"), None)]
        if self.bias is not None:
            requests.append((binding.key("bias"), self.compute_dtype))
        return binding.mechanism, tuple(requests)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._residency
        if binding is not None and not binding.mechanism.is_loaded(binding.unit):
            # An offloaded unit decodes from leased blocks and skips
            # the decoded cache both ways: repopulating a
            # device-resident cache would defeat the offload the
            # manager just performed.
            with binding.lease() as lease:
                stored = lease.get_stored("weight_blocks")
                if not isinstance(stored, torch.Tensor):
                    raise TypeError("encoded GGUF residency storage is not a block tensor")
                collector = lease.timing_collector()
                if self.fused_matmul and stored.is_cuda and self._fused_route_admits(input):
                    # The fused kernel decodes inside the matmul tiles,
                    # so there is no DEQUANT phase to bracket.
                    bias = (
                        None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)
                    )
                    if collector is None:
                        return _fused_gguf_linear(
                            self._fused_op, input, stored, bias, self.out_features
                        )
                    collector.count_forward()
                    with timed_phase(collector, COMPUTE, stored.device):
                        return _fused_gguf_linear(
                            self._fused_op, input, stored, bias, self.out_features
                        )
                if collector is None:
                    weight = self._decode(stored, (self.out_features, self.in_features))
                    bias = (
                        None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)
                    )
                    return torch.nn.functional.linear(input, weight.to(self.compute_dtype), bias)
                collector.count_forward()
                with timed_phase(collector, DEQUANT, stored.device):
                    weight = self._decode(stored, (self.out_features, self.in_features))
                bias = None if self.bias is None else lease.get("bias", dtype=self.compute_dtype)
                with timed_phase(collector, DEQUANT, stored.device):
                    weight = weight.to(self.compute_dtype)
                with timed_phase(collector, COMPUTE, stored.device):
                    return torch.nn.functional.linear(input, weight, bias)
        if self.fused_matmul and self.weight_blocks.is_cuda and self._fused_route_admits(input):
            # The fused route never materializes the decoded weight, so
            # the decoded cache does not apply.
            return _fused_gguf_linear(
                self._fused_op, input, self.weight_blocks, self.bias, self.out_features
            )
        cache = self.decoded_cache
        if cache is not None:
            weight = cache.get(self.cache_key)
            if weight is None:
                try:
                    decoded = self._decode(
                        self.weight_blocks, (self.out_features, self.in_features)
                    )
                except torch.OutOfMemoryError:
                    if cache.used_bytes == 0:
                        raise
                    cache.clear()
                    from .memory import soft_empty_cache

                    soft_empty_cache(self.weight_blocks.device)
                    decoded = self._decode(
                        self.weight_blocks, (self.out_features, self.in_features)
                    )
                weight = cache.offer(self.cache_key, decoded.to(self.compute_dtype))
            return torch.nn.functional.linear(input, weight, self.bias)
        weight = self._decode(self.weight_blocks, (self.out_features, self.in_features))
        return torch.nn.functional.linear(input, weight.to(self.compute_dtype), self.bias)


class GgufDecodedCacheResidency:
    """A module's decoded GGUF caches as one residency mechanism.

    Makes cache bytes visible to a :class:`~.residency.ResidencyManager`
    so memory pressure reclaims them: pass this to ``load()`` and
    ``free()`` like any other mechanism. The caches hold derived data
    only (decoded copies of encoded blocks that stay resident), so
    eviction is pure - nothing moves to an offload device, and a later
    forward re-decodes and re-offers per the cache contract.

    ``offloaded_bytes`` is always zero: there is nothing to load, so a
    manager ``load()`` only registers the mechanism. ``demand_paged``
    is False even though the caches refill at use time, because a
    demand-paged victim's bytes are counted as reusable without
    eviction, and cache entries stay allocated until they are actually
    evicted.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        caches: list[GgufDecodedCache] = []
        anchor: GgufEncodedLinear | None = None
        for child in module.modules():
            if not isinstance(child, GgufEncodedLinear) or child.decoded_cache is None:
                continue
            if anchor is None:
                anchor = child
            if not any(cache is child.decoded_cache for cache in caches):
                caches.append(child.decoded_cache)
        if anchor is None:
            raise ValueError("module has no cache-backed encoded GGUF linears")
        self._caches = tuple(caches)
        self._anchor = anchor

    @property
    def load_device(self) -> torch.device:
        """Where the encoded blocks live, hence where decoded entries
        allocate. Anchored to a linear, not its buffer, so device
        moves (which swap buffer tensors and clear the caches) keep
        this current."""
        return self._anchor.weight_blocks.device

    @property
    def demand_paged(self) -> bool:
        return False

    def total_bytes(self) -> int:
        return sum(cache.used_bytes for cache in self._caches)

    def loaded_bytes(self) -> int:
        return self.total_bytes()

    def automatically_reclaimable_bytes(self) -> int:
        return 0

    def offloaded_bytes(self) -> int:
        return 0

    def partially_load(self, extra_memory: int | None) -> int:
        """Caches refill on forward, not on load; only the reference's
        negative-allowance shrink applies. Returns the change in
        loaded bytes."""
        if extra_memory is not None and extra_memory < 0:
            return -self.partially_unload(-extra_memory)
        return 0

    def partially_unload(self, memory_to_free: int) -> int:
        freed = 0
        for cache in self._caches:
            if freed >= memory_to_free:
                break
            freed += cache.free_bytes(memory_to_free - freed)
        return freed

    def unload(self) -> None:
        for cache in self._caches:
            cache.clear()

    def release_working_buffers(self) -> bool:
        return False

    def working_set_reservation_bytes(self) -> int:
        return 0

    def reserve_working_set(self) -> AbstractContextManager[None]:
        return nullcontext()
