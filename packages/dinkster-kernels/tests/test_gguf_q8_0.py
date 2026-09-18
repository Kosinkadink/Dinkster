"""Fused Q8_0 op tests.

Self-contained: the reference decode below restates the Q8_0 layout
(float16 scale at bytes 0:2, 32 int8 quants at bytes 2:34, decode =
quant * scale) rather than importing dinkster-inference-torch, so this
package tests its own numerical contract in isolation. Cross-package
bit-identity against the inference decoders lives in
dinkster-inference-torch's GPU suite.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from dinkster_kernels import (
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
    gguf_q8_0_decode,
    gguf_q8_0_linear,
    gguf_q8_0_linear_available,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and find_spec("triton") is not None),
    reason="fused Q8_0 kernels need a CUDA device and triton",
)


def _reference_decode(blocks: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    scales = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    quants = blocks[:, 2:].view(torch.int8).to(torch.float32)
    return (quants * scales).reshape(out_features, in_features)


def _random_blocks(
    out_features: int, in_features: int, *, seed: int, device: str = "cpu"
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = out_features * in_features // Q8_0_BLOCK_ELEMENTS
    blocks = torch.randint(
        0, 256, (count, Q8_0_BLOCK_BYTES), dtype=torch.uint8, generator=generator
    )
    # Finite, non-degenerate scales: random bytes make inf/nan fp16.
    scales = torch.empty(count, dtype=torch.float32).uniform_(-2.0, 2.0, generator=generator)
    blocks[:, :2] = scales.to(torch.float16).view(torch.uint8).reshape(count, 2)
    return blocks.to(device)


def _skip_unless_available() -> None:
    if not gguf_q8_0_linear_available():
        pytest.skip("fused Q8_0 probe reports unavailable (no host C compiler for triton?)")


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
    first = gguf_q8_0_linear_available()
    assert isinstance(first, bool)
    assert gguf_q8_0_linear_available() == first


def test_block_constants() -> None:
    assert Q8_0_BLOCK_ELEMENTS == 32
    assert Q8_0_BLOCK_BYTES == 34


@requires_cuda
@pytest.mark.parametrize(
    ("out_features", "in_features"),
    [
        (1, 32),  # single block, one masked launch program
        (31, 32),  # 992 elements: just under the 1024-element program
        (32, 32),  # 1024 elements: exactly one full program
        (33, 32),  # 1056 elements: tail spills into a second program
        (48, 96),  # multi-program interior
    ],
)
def test_decode_bit_exact_vs_reference(out_features: int, in_features: int) -> None:
    _skip_unless_available()
    blocks = _random_blocks(out_features, in_features, seed=0)
    # A negative scale over zero quants decodes to -0.0: only a
    # bit-pattern compare can tell it from +0.0, so pin one such block.
    blocks[0, 2:] = 0
    blocks[0, :2] = torch.tensor([-1.5], dtype=torch.float16).view(torch.uint8)
    expected = _reference_decode(blocks, out_features, in_features)
    assert expected.view(torch.int32)[0, 0] != 0  # really -0.0
    decoded = gguf_q8_0_decode(blocks.cuda(), out_features, in_features)
    assert decoded.dtype == torch.float32
    assert torch.equal(decoded.cpu().view(torch.int32), expected.view(torch.int32))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 64, 32),
        (17, 96, 64),
        (33, 40, 96),
        (128, 256, 128),
    ],
)
def test_linear_matches_decode_reference(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    _skip_unless_available()
    blocks = _random_blocks(n, k, seed=m * 1000 + n).cuda()
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((m, k), dtype=dtype, device="cuda", generator=generator)
    bias = torch.randn(n, dtype=dtype, device="cuda", generator=generator)
    weight = gguf_q8_0_decode(blocks, n, k).to(dtype)
    for b in (None, bias):
        fused = gguf_q8_0_linear(x, blocks, b, n)
        assert fused.dtype == dtype
        _assert_as_accurate_as_linear(fused, x, weight, b)


@requires_cuda
def test_linear_batched_input_shape() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 32, seed=3).cuda()
    x = torch.randn((2, 5, 32), dtype=torch.float16, device="cuda")
    out = gguf_q8_0_linear(x, blocks, None, 64)
    assert out.shape == (2, 5, 64)
    flat = gguf_q8_0_linear(x.reshape(-1, 32), blocks, None, 64)
    assert torch.equal(out.reshape(-1, 64), flat)


@requires_cuda
def test_linear_deterministic() -> None:
    _skip_unless_available()
    blocks = _random_blocks(96, 64, seed=11).cuda()
    x = torch.randn((17, 64), dtype=torch.bfloat16, device="cuda")
    first = gguf_q8_0_linear(x, blocks, None, 96)
    second = gguf_q8_0_linear(x, blocks, None, 96)
    assert torch.equal(first, second)


@requires_cuda
def test_linear_empty_batch() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 32, seed=5).cuda()
    x = torch.empty((0, 32), dtype=torch.float16, device="cuda")
    out = gguf_q8_0_linear(x, blocks, None, 64)
    assert out.shape == (0, 64)


@requires_cuda
def test_validation_errors() -> None:
    _skip_unless_available()
    good = _random_blocks(64, 32, seed=9).cuda()
    x = torch.randn((4, 32), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="uint8"):
        gguf_q8_0_decode(good.to(torch.int8), 64, 32)
    with pytest.raises(ValueError, match="whole 32-element blocks"):
        gguf_q8_0_linear(torch.randn((4, 48), dtype=torch.float16, device="cuda"), good, None, 64)
    with pytest.raises(ValueError, match="do not hold"):
        gguf_q8_0_decode(good, 64, 64)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        gguf_q8_0_linear(x.to(torch.float32), good, None, 64)
    with pytest.raises(ValueError, match="bias"):
        gguf_q8_0_linear(x, good, torch.randn(64, dtype=torch.float32, device="cuda"), 64)
    with pytest.raises(ValueError, match="blocks on the input device"):
        gguf_q8_0_linear(x, good.cpu(), None, 64)
    with pytest.raises(ValueError, match="bias on the input device"):
        gguf_q8_0_linear(x, good, torch.randn(64, dtype=torch.float16), 64)


@requires_cuda
def test_linear_noncontiguous_bias() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 32, seed=13).cuda()
    x = torch.randn((4, 32), dtype=torch.float16, device="cuda")
    strided = torch.randn(128, dtype=torch.float16, device="cuda")[::2]
    assert not strided.is_contiguous()
    out = gguf_q8_0_linear(x, blocks, strided, 64)
    assert torch.equal(out, gguf_q8_0_linear(x, blocks, strided.contiguous(), 64))


@requires_cuda
def test_opcheck_both_ops() -> None:
    _skip_unless_available()
    blocks = _random_blocks(64, 32, seed=21).cuda()
    x = torch.randn((3, 32), dtype=torch.float16, device="cuda")
    bias = torch.randn(64, dtype=torch.float16, device="cuda")
    torch.library.opcheck(gguf_q8_0_decode, (blocks, 64, 32))
    torch.library.opcheck(gguf_q8_0_linear, (x, blocks, bias, 64))
    torch.library.opcheck(gguf_q8_0_linear, (x, blocks, None, 64))
