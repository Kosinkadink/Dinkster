"""Fused Q4_K op tests.

Self-contained: the reference decode below restates the Q4_K layout
(float16 super scale d at bytes 0:2 and min scale dmin at 2:4, twelve
packed 6-bit (scale, min) bytes at 4:16, 128 quant bytes at 16:144 as
four 32-byte chunks whose low nibbles decode under group 2c and high
nibbles under group 2c + 1, each element as
q * (d * sc) - (dmin * mn)) rather than importing
dinkster-inference-torch, so this package tests its own numerical
contract in isolation. Cross-package bit-identity against the
inference decoders lives in dinkster-inference-torch's GPU suite.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from dinkster_kernels import (
    Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMENTS,
    gguf_q4_k_decode,
    gguf_q4_k_linear,
    gguf_q4_k_linear_available,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and find_spec("triton") is not None),
    reason="fused Q4_K kernels need a CUDA device and triton",
)


def _group_scale_mins(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = blocks[:, 0:2].contiguous().view(torch.float16).to(torch.float32)
    dmin = blocks[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
    packed = blocks[:, 4:16]
    low, mid, high = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    sc = torch.cat((low & 0x3F, (high & 0x0F) | ((low >> 6) << 4)), dim=1)
    mn = torch.cat((mid & 0x3F, (high >> 4) | ((mid >> 6) << 4)), dim=1)
    return d * sc.to(torch.float32), dmin * mn.to(torch.float32)


def _reference_decode(blocks: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    scales, mins = _group_scale_mins(blocks)
    quants = blocks[:, 16:].reshape(-1, 4, 1, 32)
    q = torch.cat((quants & 0x0F, quants >> 4), dim=2).to(torch.float32)
    values = q * scales.reshape(-1, 4, 2, 1) - mins.reshape(-1, 4, 2, 1)
    return values.reshape(out_features, in_features)


def _random_blocks(
    out_features: int, in_features: int, *, seed: int, device: str = "cpu"
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = out_features * in_features // Q4_K_BLOCK_ELEMENTS
    blocks = torch.randint(
        0, 256, (count, Q4_K_BLOCK_BYTES), dtype=torch.uint8, generator=generator
    )
    # Finite, non-degenerate super scales: random bytes make inf/nan fp16.
    for offset in (0, 2):
        scales = torch.empty(count, dtype=torch.float32).uniform_(-2.0, 2.0, generator=generator)
        blocks[:, offset : offset + 2] = scales.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    return blocks.to(device)


def _pin_negative_zero_block(blocks: torch.Tensor) -> None:
    """Make block 0 decode to all -0.0: zero quants and mins under a
    negative group scale, so q * (d * sc) is -0.0 and the min
    subtraction preserves the sign. Only a bit-pattern compare can
    tell the result from +0.0."""
    blocks[0, 4:] = 0
    blocks[0, 4] = 1  # group 0 scale sc = 1
    blocks[0, 0:2] = torch.tensor([-1.5], dtype=torch.float16).view(torch.uint8)
    blocks[0, 2:4] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)


def _skip_unless_available() -> None:
    if not gguf_q4_k_linear_available():
        pytest.skip("fused Q4_K probe reports unavailable (no host C compiler for triton?)")


def _assert_as_accurate_as_linear(
    fused: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """The fused kernel consumes the same half-precision operands as
    decode-then-``F.linear`` and accumulates in float32; only the tile
    summation order differs. Fixed rtol/atol cannot express that
    contract (accumulation noise is absolute in the intermediate
    magnitude, so it explodes relatively where terms cancel to near
    zero), so compare both routes to the float64 sum of the identical
    operands and require the fused error to stay within a small factor
    of the reference route's."""
    reference = torch.nn.functional.linear(x, weight, bias)
    exact = torch.nn.functional.linear(
        x.double(), weight.double(), None if bias is None else bias.double()
    )
    fused_error = (fused.double() - exact).abs().max().item()
    reference_error = (reference.double() - exact).abs().max().item()
    assert fused_error <= 4.0 * reference_error + 1e-6, (
        f"fused error {fused_error} vs reference error {reference_error}"
    )


def test_import_and_probe_never_raise() -> None:
    # The probe is the availability contract: callable anywhere,
    # boolean, stable across calls (cached).
    first = gguf_q4_k_linear_available()
    assert isinstance(first, bool)
    assert gguf_q4_k_linear_available() == first


def test_block_constants() -> None:
    assert Q4_K_BLOCK_ELEMENTS == 256
    assert Q4_K_BLOCK_BYTES == 144


@requires_cuda
@pytest.mark.parametrize(
    ("out_features", "in_features"),
    [
        (1, 256),  # single block, one masked launch program
        (3, 256),  # 768 elements: under the 1024-element program
        (4, 256),  # 1024 elements: exactly one full program
        (5, 256),  # 1280 elements: tail spills into a second program
        (4, 1280),  # 5120 elements: multi-program interior
    ],
)
def test_decode_bit_exact_vs_reference(out_features: int, in_features: int) -> None:
    _skip_unless_available()
    blocks = _random_blocks(out_features, in_features, seed=0)
    _pin_negative_zero_block(blocks)
    expected = _reference_decode(blocks, out_features, in_features)
    assert expected.view(torch.int32)[0, 0] != 0  # really -0.0
    decoded = gguf_q4_k_decode(blocks.cuda(), out_features, in_features)
    assert decoded.dtype == torch.float32
    assert torch.equal(decoded.cpu().view(torch.int32), expected.view(torch.int32))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 64, 256),
        (17, 96, 512),
        (33, 40, 256),
        (128, 192, 512),
    ],
)
def test_linear_matches_decode_reference(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    _skip_unless_available()
    blocks = _random_blocks(n, k, seed=m * 1000 + n).cuda()
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((m, k), dtype=dtype, device="cuda", generator=generator)
    bias = torch.randn(n, dtype=dtype, device="cuda", generator=generator)
    weight = gguf_q4_k_decode(blocks, n, k).to(dtype)
    for b in (None, bias):
        fused = gguf_q4_k_linear(x, blocks, b, n)
        assert fused.dtype == dtype
        _assert_as_accurate_as_linear(fused, x, weight, b)


@requires_cuda
def test_linear_batched_input_shape() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 256, seed=3).cuda()
    x = torch.randn((2, 5, 256), dtype=torch.float16, device="cuda")
    out = gguf_q4_k_linear(x, blocks, None, 64)
    assert out.shape == (2, 5, 64)
    flat = gguf_q4_k_linear(x.reshape(-1, 256), blocks, None, 64)
    assert torch.equal(out.reshape(-1, 64), flat)


@requires_cuda
def test_linear_deterministic() -> None:
    _skip_unless_available()
    blocks = _random_blocks(96, 512, seed=11).cuda()
    x = torch.randn((17, 512), dtype=torch.bfloat16, device="cuda")
    first = gguf_q4_k_linear(x, blocks, None, 96)
    second = gguf_q4_k_linear(x, blocks, None, 96)
    assert torch.equal(first, second)


@requires_cuda
def test_linear_empty_batch() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 256, seed=5).cuda()
    x = torch.empty((0, 256), dtype=torch.float16, device="cuda")
    out = gguf_q4_k_linear(x, blocks, None, 64)
    assert out.shape == (0, 64)


@requires_cuda
def test_validation_errors() -> None:
    _skip_unless_available()
    good = _random_blocks(64, 256, seed=9).cuda()
    x = torch.randn((4, 256), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="uint8"):
        gguf_q4_k_decode(good.to(torch.int8), 64, 256)
    with pytest.raises(ValueError, match="whole 256-element blocks"):
        gguf_q4_k_linear(torch.randn((4, 384), dtype=torch.float16, device="cuda"), good, None, 64)
    with pytest.raises(ValueError, match="do not hold"):
        gguf_q4_k_decode(good, 64, 512)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        gguf_q4_k_linear(x.to(torch.float32), good, None, 64)
    with pytest.raises(ValueError, match="bias"):
        gguf_q4_k_linear(x, good, torch.randn(64, dtype=torch.float32, device="cuda"), 64)
    with pytest.raises(ValueError, match="blocks on the input device"):
        gguf_q4_k_linear(x, good.cpu(), None, 64)
    with pytest.raises(ValueError, match="bias on the input device"):
        gguf_q4_k_linear(x, good, torch.randn(64, dtype=torch.float16), 64)


@requires_cuda
def test_linear_noncontiguous_bias() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 256, seed=13).cuda()
    x = torch.randn((4, 256), dtype=torch.float16, device="cuda")
    strided = torch.randn(128, dtype=torch.float16, device="cuda")[::2]
    assert not strided.is_contiguous()
    out = gguf_q4_k_linear(x, blocks, strided, 64)
    assert torch.equal(out, gguf_q4_k_linear(x, blocks, strided.contiguous(), 64))


@requires_cuda
def test_opcheck_both_ops() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 256, seed=21).cuda()
    x = torch.randn((3, 256), dtype=torch.float16, device="cuda")
    bias = torch.randn(64, dtype=torch.float16, device="cuda")
    torch.library.opcheck(gguf_q4_k_decode, (blocks, 64, 256))
    torch.library.opcheck(gguf_q4_k_linear, (x, blocks, bias, 64))
    torch.library.opcheck(gguf_q4_k_linear, (x, blocks, None, 64))
