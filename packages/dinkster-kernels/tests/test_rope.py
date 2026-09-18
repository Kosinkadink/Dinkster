"""Fused RoPE op tests.

Self-contained: the reference below restates the rotation
(comfy/ldm/flux/math.py _apply_rope1 @ b78cec87: float32 pair math via
one multiply and one ``torch.addcmul``, cast back to the input dtype)
rather than importing dinkster-inference-torch, so this package tests its
own numerical contract in isolation. The claim is BIT-identity to the
reference on the same device, so comparisons view the bits as integers
(``torch.equal`` calls -0.0 and +0.0 equal).
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from dinkster_kernels import (
    apply_rope,
    apply_rope_available,
    apply_rope_supported,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and find_spec("triton") is not None),
    reason="the fused RoPE kernel needs a CUDA device and triton",
)

_INT_VIEWS = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
}


def _reference(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        out = torch.addcmul(freqs_cis[..., 0] * x_[..., 0], freqs_cis[..., 1], x_[..., 1])
        return out.reshape(x.shape).type_as(x)

    return rotate(xq), rotate(xk)


def _frequencies(
    batch: int, seq_len: int, head_dim: int, *, seed: int, device: str = "cuda"
) -> torch.Tensor:
    """Random rotation-shaped float32 frequencies (values need not be
    trigonometric for a rounding contract)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (batch, 1, seq_len, head_dim // 2, 2, 2),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def _assert_bit_identical(got: torch.Tensor, expected: torch.Tensor) -> None:
    assert got.dtype == expected.dtype
    view = _INT_VIEWS[got.dtype]
    assert torch.equal(got.view(view), expected.view(view))


def _skip_unless_available() -> None:
    if not apply_rope_available():
        pytest.skip("fused RoPE probe reports unavailable (no host C compiler for triton?)")


def test_import_and_probe_never_raise() -> None:
    first = apply_rope_available()
    assert isinstance(first, bool)
    assert apply_rope_available() == first


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("batch", "heads", "seq_len", "head_dim", "freq_batch"),
    [
        (1, 1, 1, 2, 1),  # minimal: one pair, one masked program
        (1, 24, 4608, 128, 1),  # Flux 1024x1024 attention shape
        (2, 12, 333, 64, 1),  # odd seq: program tail masking
        (2, 12, 333, 64, 2),  # per-batch frequencies
        (3, 40, 75, 128, 3),
    ],
)
def test_bit_identical_vs_reference(
    dtype: torch.dtype,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    freq_batch: int,
) -> None:
    _skip_unless_available()
    generator = torch.Generator(device="cuda").manual_seed(7)
    xq = torch.randn(
        (batch, heads, seq_len, head_dim), dtype=dtype, device="cuda", generator=generator
    )
    xk = torch.randn(
        (batch, heads, seq_len, head_dim), dtype=dtype, device="cuda", generator=generator
    )
    freqs = _frequencies(freq_batch, seq_len, head_dim, seed=batch)
    expected_q, expected_k = _reference(xq, xk, freqs)
    got_q, got_k = apply_rope(xq, xk, freqs)
    _assert_bit_identical(got_q, expected_q)
    _assert_bit_identical(got_k, expected_k)


@requires_cuda
def test_negative_zero_preserved() -> None:
    """A -0.0 rotation output survives bitwise: zero inputs against a
    negative frequency row produce fma(f1, 0, f0*0) = -0.0 patterns
    that torch.equal alone cannot distinguish."""
    _skip_unless_available()
    xq = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device="cuda")
    xk = torch.zeros_like(xq)
    freqs = -torch.ones((1, 1, 1, 1, 2, 2), dtype=torch.float32, device="cuda")
    expected_q, _ = _reference(xq, xk, freqs)
    got_q, _ = apply_rope(xq, xk, freqs)
    assert expected_q.view(torch.int32).any()  # really -0.0
    _assert_bit_identical(got_q, expected_q)


@requires_cuda
def test_noncontiguous_inputs() -> None:
    _skip_unless_available()
    wide = torch.randn((2, 4, 16, 128), dtype=torch.float16, device="cuda")
    xq = wide[..., ::2]  # (2, 4, 16, 64), non-contiguous
    xk = wide[..., 1::2]
    assert not xq.is_contiguous()
    freqs = _frequencies(1, 16, 64, seed=5)
    expected_q, expected_k = _reference(xq, xk, freqs)
    got_q, got_k = apply_rope(xq, xk, freqs)
    _assert_bit_identical(got_q, expected_q)
    _assert_bit_identical(got_k, expected_k)


@requires_cuda
def test_deterministic() -> None:
    _skip_unless_available()
    xq = torch.randn((2, 8, 100, 64), dtype=torch.bfloat16, device="cuda")
    xk = torch.randn_like(xq)
    freqs = _frequencies(1, 100, 64, seed=11)
    first = apply_rope(xq, xk, freqs)
    second = apply_rope(xq, xk, freqs)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


@requires_cuda
def test_empty_batch() -> None:
    _skip_unless_available()
    xq = torch.empty((0, 8, 16, 64), dtype=torch.float16, device="cuda")
    xk = torch.empty_like(xq)
    freqs = _frequencies(1, 16, 64, seed=3)
    got_q, got_k = apply_rope(xq, xk, freqs)
    assert got_q.shape == xq.shape and got_k.shape == xk.shape


@requires_cuda
def test_validation_errors() -> None:
    _skip_unless_available()
    xq = torch.randn((1, 2, 3, 8), dtype=torch.float16, device="cuda")
    xk = torch.randn_like(xq)
    freqs = _frequencies(1, 3, 8, seed=1)
    with pytest.raises(ValueError, match="xq and xk must match"):
        apply_rope(xq, xk.to(torch.bfloat16), freqs)
    with pytest.raises(ValueError, match="even head_dim"):
        apply_rope(xq[..., :7], xk[..., :7], freqs)
    with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
        apply_rope(xq.double(), xk.double(), freqs)
    with pytest.raises(ValueError, match="freqs_cis must be float32"):
        apply_rope(xq, xk, freqs.half())
    with pytest.raises(ValueError, match="freqs_cis must be"):
        apply_rope(xq, xk, freqs[:, :, :2])


def test_supported_predicate_matches_contract() -> None:
    xq = torch.randn((1, 2, 3, 8), dtype=torch.float16)
    xk = torch.randn_like(xq)
    freqs = torch.randn((1, 1, 3, 4, 2, 2), dtype=torch.float32)
    assert not apply_rope_supported(xq, xk, freqs)  # cpu tensors
    if torch.cuda.is_available():
        assert apply_rope_supported(xq.cuda(), xk.cuda(), freqs.cuda())
        assert not apply_rope_supported(xq.cuda(), xk.cuda(), freqs.half().cuda())
        assert not apply_rope_supported(xq.cuda(), xk.cuda(), freqs)  # device mix


@requires_cuda
def test_opcheck() -> None:
    _skip_unless_available()
    xq = torch.randn((1, 2, 3, 8), dtype=torch.float16, device="cuda")
    xk = torch.randn_like(xq)
    freqs = _frequencies(1, 3, 8, seed=9)
    torch.library.opcheck(apply_rope, (xq, xk, freqs))


@requires_cuda
def test_opcheck_dense_noncontiguous() -> None:
    """Fake and eager metadata must agree for dense transposed inputs,
    which the eager body copies to contiguous before allocating."""
    _skip_unless_available()
    xq = torch.randn((1, 3, 8, 3), dtype=torch.float16, device="cuda").transpose(-1, -2)
    xk = torch.randn((1, 3, 8, 3), dtype=torch.float16, device="cuda").transpose(-1, -2)
    assert not xq.is_contiguous()
    freqs = _frequencies(1, 3, 8, seed=13)
    torch.library.opcheck(apply_rope, (xq, xk, freqs))


@requires_cuda
def test_opcheck_singleton_noncanonical_strides() -> None:
    """Fake and eager metadata must agree for inputs that are
    contiguous with non-canonical singleton-dimension strides, where
    preserve-format allocation would keep the input strides but the
    op allocates canonical contiguous outputs."""
    _skip_unless_available()
    xq = torch.randn((2, 3, 1, 8), dtype=torch.float16, device="cuda").transpose(1, 2)
    xk = torch.randn((2, 3, 1, 8), dtype=torch.float16, device="cuda").transpose(1, 2)
    assert xq.is_contiguous() and xq.stride() != torch.empty(xq.shape).stride()
    freqs = _frequencies(1, 3, 8, seed=17)
    torch.library.opcheck(apply_rope, (xq, xk, freqs))


@requires_cuda
def test_opcheck_zero_batch_noncanonical_strides() -> None:
    """Fake and eager metadata must agree for zero-size inputs with
    non-canonical strides, where real and fake preserve-format
    allocations diverge; the op sidesteps that by allocating canonical
    contiguous outputs."""
    _skip_unless_available()
    xq = torch.empty((3, 0, 4, 8), dtype=torch.float16, device="cuda").transpose(0, 1)
    xk = torch.empty((3, 0, 4, 8), dtype=torch.float16, device="cuda").transpose(0, 1)
    assert xq.stride() != torch.empty(xq.shape).stride()
    freqs = _frequencies(1, 4, 8, seed=19)
    torch.library.opcheck(apply_rope, (xq, xk, freqs))


def test_fake_output_strides_match_eager_allocation() -> None:
    """The fake reports exactly the strides the eager body allocates:
    canonical contiguous for every accepted input, including dense
    non-contiguous, singleton-dimension, and zero-size non-canonical
    layouts."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    for base_shape, transpose_dims in (
        ((1, 3, 8, 3), (-1, -2)),  # dense non-contiguous
        ((2, 3, 1, 8), (1, 2)),  # contiguous, singleton-dim strides
        ((3, 0, 4, 8), (0, 1)),  # zero-size, non-canonical strides
        ((0, 4, 3, 8), (1, 2)),  # zero-size, transposed inner dims
    ):
        real = torch.empty(base_shape, dtype=torch.float16).transpose(*transpose_dims)
        real_k = torch.empty(base_shape, dtype=torch.float16).transpose(*transpose_dims)
        expected_stride = torch.empty_like(
            real.contiguous(), memory_format=torch.contiguous_format
        ).stride()
        _, _, seq_len, head_dim = real.shape
        xq = fake_mode.from_tensor(real)
        xk = fake_mode.from_tensor(real_k)
        with fake_mode:
            freqs = torch.empty((1, 1, seq_len, head_dim // 2, 2, 2), dtype=torch.float32)
            out_q, out_k = apply_rope(xq, xk, freqs)
        assert out_q.shape == real.shape and out_k.shape == real.shape
        assert out_q.stride() == expected_stride
        assert out_k.stride() == expected_stride
