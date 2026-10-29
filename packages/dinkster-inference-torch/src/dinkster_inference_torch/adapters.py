"""Adapter tensor math: the torch half of the stage-4a decode specs.

Each class is a near-transcription of one
comfy/weight_adapter/*.calculate_weight @ b78cec87, holding the same
weights the reference tuple held (constructor keywords instead of
positional slots) and satisfying the stage-1 ``WeightAdapter``
protocol (``target_shape`` + ``calculate``). ``calculate`` mutates and
returns ``weight`` exactly like the reference - the apply layer owns
copying (comfy's patch_weight_to_device passes a fresh
intermediate-dtype copy, and so does Dinkster's caller).

Deliberate deviations from the reference, all LOUD:

- Upstream wraps the math in try/except, logs
  "ERROR <name> <key> <exc>" and returns the weight UNPATCHED; here
  the exception propagates as AdapterMathError. A LoRA that cannot be
  applied is a failed job, not a silently unmodified model.
- GLoRA: when neither orientation probe resolves a rank but alpha is
  set, upstream hits an unbound local inside its try (logged, weight
  unpatched); here it is an explicit AdapterMathError.
- Two upstream-broken paths are FIXED here rather than inherited
  (docs/comfyui-issues/ has the analyses; no oracle golden can exist
  until upstream fixes them, so their tests verify independently):
  LoKr Tucker+conv makes the einsum-produced w2 contiguous before
  torch.kron; BOFT interpolates partial strength with an identity in
  bi's dtype (as oft.py does) instead of the intermediate-dtype eye.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from typing import Any, TypeVar, cast

import torch

from .tensor_ops import (
    cast_to_device,
    identity,
    pad_tensor_to_shape,
    weight_decompose,
)

A = TypeVar("A")


def _payload_tensors(adapter: object, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
    return tuple(
        value for name in names if isinstance((value := getattr(adapter, name)), torch.Tensor)
    )


def _rebuild_payloads(adapter: A, names: Sequence[str], replacements: Sequence[torch.Tensor]) -> A:
    payload_names = [name for name in names if isinstance(getattr(adapter, name), torch.Tensor)]
    if len(payload_names) != len(replacements):
        raise ValueError(
            f"{type(adapter).__name__} needs {len(payload_names)} payload replacements,"
            f" got {len(replacements)}"
        )
    rebuilt = copy(adapter)
    for name, replacement in zip(payload_names, replacements, strict=True):
        setattr(cast(Any, rebuilt), name, replacement)
    return rebuilt


def _delta_hook(
    function: Callable[[torch.Tensor], torch.Tensor] | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The per-entry delta hook, None-as-identity (the reference's
    ``if function is None: function = lambda a: a``)."""
    return identity if function is None else function


class AdapterMathError(Exception):
    """Adapter tensor math failed (shape mismatch, malformed weights).

    Replaces the reference's log-and-return-unpatched error path."""


class LoRAAdapter:
    """comfy/weight_adapter/lora.py calculate_weight @ b78cec87.

    Reference weights tuple: (up, down, alpha, mid, dora_scale,
    reshape)."""

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        alpha: float | None = None,
        mid: torch.Tensor | None = None,
        dora_scale: torch.Tensor | None = None,
        reshape: tuple[int, ...] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.up = up
        self.down = down
        self.alpha = alpha
        self.mid = mid
        self.dora_scale = dora_scale
        self.reshape = reshape
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return self.reshape if self.reshape is not None else base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(self, ("up", "down", "mid", "dora_scale"))

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> LoRAAdapter:
        return _rebuild_payloads(self, ("up", "down", "mid", "dora_scale"), replacements)

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mat1 = cast_to_device(self.up, weight.device, self.intermediate_dtype)
        mat2 = cast_to_device(self.down, weight.device, self.intermediate_dtype)

        if self.reshape is not None:
            weight = pad_tensor_to_shape(weight, self.reshape)

        if self.alpha is not None:
            alpha = self.alpha / mat2.shape[0]
        else:
            alpha = 1.0

        if self.mid is not None:
            mat3 = cast_to_device(self.mid, weight.device, self.intermediate_dtype)
            final_shape = [
                mat2.shape[1],
                mat2.shape[0],
                mat3.shape[2],
                mat3.shape[3],
            ]
            mat2 = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mat3.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        try:
            lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(
                weight.shape
            )
            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)(((strength * alpha) * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"lora: {exc}") from exc
        return weight


class LoHaAdapter:
    """comfy/weight_adapter/loha.py calculate_weight @ b78cec87.

    Reference weights tuple: (w1_a, w1_b, alpha, w2_a, w2_b, t1, t2,
    dora_scale)."""

    def __init__(
        self,
        w1_a: torch.Tensor,
        w1_b: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        *,
        alpha: float | None = None,
        t1: torch.Tensor | None = None,
        t2: torch.Tensor | None = None,
        dora_scale: torch.Tensor | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.w1_a = w1_a
        self.w1_b = w1_b
        self.w2_a = w2_a
        self.w2_b = w2_b
        self.alpha = alpha
        self.t1 = t1
        self.t2 = t2
        self.dora_scale = dora_scale
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(self, ("w1_a", "w1_b", "w2_a", "w2_b", "t1", "t2", "dora_scale"))

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> LoHaAdapter:
        return _rebuild_payloads(
            self,
            ("w1_a", "w1_b", "w2_a", "w2_b", "t1", "t2", "dora_scale"),
            replacements,
        )

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.alpha is not None:
            alpha = self.alpha / self.w1_b.shape[0]
        else:
            alpha = 1.0

        if self.t1 is not None and self.t2 is not None:
            m1 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                cast_to_device(self.t1, weight.device, self.intermediate_dtype),
                cast_to_device(self.w1_b, weight.device, self.intermediate_dtype),
                cast_to_device(self.w1_a, weight.device, self.intermediate_dtype),
            )
            m2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                cast_to_device(self.t2, weight.device, self.intermediate_dtype),
                cast_to_device(self.w2_b, weight.device, self.intermediate_dtype),
                cast_to_device(self.w2_a, weight.device, self.intermediate_dtype),
            )
        else:
            m1 = torch.mm(
                cast_to_device(self.w1_a, weight.device, self.intermediate_dtype),
                cast_to_device(self.w1_b, weight.device, self.intermediate_dtype),
            )
            m2 = torch.mm(
                cast_to_device(self.w2_a, weight.device, self.intermediate_dtype),
                cast_to_device(self.w2_b, weight.device, self.intermediate_dtype),
            )

        try:
            lora_diff = (m1 * m2).reshape(weight.shape)
            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)(((strength * alpha) * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"loha: {exc}") from exc
        return weight


class LoKrAdapter:
    """comfy/weight_adapter/lokr.py calculate_weight @ b78cec87.

    Reference weights tuple: (w1, w2, alpha, w1_a, w1_b, w2_a, w2_b,
    t2, dora_scale)."""

    def __init__(
        self,
        *,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
        w1_a: torch.Tensor | None = None,
        w1_b: torch.Tensor | None = None,
        w2_a: torch.Tensor | None = None,
        w2_b: torch.Tensor | None = None,
        t2: torch.Tensor | None = None,
        alpha: float | None = None,
        dora_scale: torch.Tensor | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w1_a = w1_a
        self.w1_b = w1_b
        self.w2_a = w2_a
        self.w2_b = w2_b
        self.t2 = t2
        self.alpha = alpha
        self.dora_scale = dora_scale
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(
            self,
            ("w1", "w2", "w1_a", "w1_b", "w2_a", "w2_b", "t2", "dora_scale"),
        )

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> LoKrAdapter:
        return _rebuild_payloads(
            self,
            ("w1", "w2", "w1_a", "w1_b", "w2_a", "w2_b", "t2", "dora_scale"),
            replacements,
        )

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dim: int | None = None

        if self.w1 is None:
            if self.w1_a is None or self.w1_b is None:
                raise AdapterMathError("lokr: w1 absent and w1_a/w1_b decomposition incomplete")
            dim = self.w1_b.shape[0]
            w1 = torch.mm(
                cast_to_device(self.w1_a, weight.device, self.intermediate_dtype),
                cast_to_device(self.w1_b, weight.device, self.intermediate_dtype),
            )
        else:
            w1 = cast_to_device(self.w1, weight.device, self.intermediate_dtype)

        if self.w2 is None:
            if self.w2_a is None or self.w2_b is None:
                raise AdapterMathError("lokr: w2 absent and w2_a/w2_b decomposition incomplete")
            dim = self.w2_b.shape[0]
            if self.t2 is None:
                w2 = torch.mm(
                    cast_to_device(self.w2_a, weight.device, self.intermediate_dtype),
                    cast_to_device(self.w2_b, weight.device, self.intermediate_dtype),
                )
            else:
                # .contiguous() diverges from the reference ON PURPOSE:
                # upstream feeds the non-contiguous einsum output to
                # torch.kron, which always raises, so this path never
                # applies there (docs/comfyui-issues/
                # lokr-tucker-kron-noncontiguous.md). Numerically a
                # no-op; verified independently in test_adapters.py.
                w2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    cast_to_device(self.t2, weight.device, self.intermediate_dtype),
                    cast_to_device(self.w2_b, weight.device, self.intermediate_dtype),
                    cast_to_device(self.w2_a, weight.device, self.intermediate_dtype),
                ).contiguous()
        else:
            w2 = cast_to_device(self.w2, weight.device, self.intermediate_dtype)

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        if self.alpha is not None and dim is not None:
            alpha = self.alpha / dim
        else:
            alpha = 1.0

        try:
            lora_diff = torch.kron(w1, w2).reshape(weight.shape)
            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)(((strength * alpha) * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"lokr: {exc}") from exc
        return weight


class GLoRAAdapter:
    """comfy/weight_adapter/glora.py calculate_weight @ b78cec87.

    Reference weights tuple: (a1, a2, b1, b2, alpha, dora_scale).
    Old-vs-new orientation is decided from tensor shapes exactly as
    upstream does."""

    def __init__(
        self,
        a1: torch.Tensor,
        a2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        *,
        alpha: float | None = None,
        dora_scale: torch.Tensor | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.dora_scale = dora_scale
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(self, ("a1", "a2", "b1", "b2", "dora_scale"))

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> GLoRAAdapter:
        return _rebuild_payloads(self, ("a1", "a2", "b1", "b2", "dora_scale"), replacements)

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        rank: int | None = None
        old_glora = False
        if self.b2.shape[1] == self.b1.shape[0] == self.a1.shape[0] == self.a2.shape[1]:
            rank = self.a1.shape[0]
            old_glora = True

        if self.b2.shape[0] == self.b1.shape[1] == self.a1.shape[1] == self.a2.shape[0]:
            if (
                old_glora
                and self.a2.shape[0] == weight.shape[0]
                and weight.shape[0] == weight.shape[1]
            ):
                pass
            else:
                old_glora = False
                rank = self.a2.shape[0]

        a1 = cast_to_device(self.a1.flatten(start_dim=1), weight.device, self.intermediate_dtype)
        a2 = cast_to_device(self.a2.flatten(start_dim=1), weight.device, self.intermediate_dtype)
        b1 = cast_to_device(self.b1.flatten(start_dim=1), weight.device, self.intermediate_dtype)
        b2 = cast_to_device(self.b2.flatten(start_dim=1), weight.device, self.intermediate_dtype)

        if self.alpha is not None:
            if rank is None:
                raise AdapterMathError("glora: alpha set but neither orientation matched")
            alpha = self.alpha / rank
        else:
            alpha = 1.0

        try:
            if old_glora:
                lora_diff = (
                    torch.mm(b2, b1)
                    + torch.mm(
                        torch.mm(
                            weight.flatten(start_dim=1).to(dtype=self.intermediate_dtype),
                            a2,
                        ),
                        a1,
                    )
                ).reshape(weight.shape)
            else:
                if weight.dim() > 2:
                    lora_diff = torch.einsum(
                        "o i ..., i j -> o j ...",
                        torch.einsum(
                            "o i ..., i j -> o j ...",
                            weight.to(dtype=self.intermediate_dtype),
                            a1,
                        ),
                        a2,
                    ).reshape(weight.shape)
                else:
                    lora_diff = torch.mm(
                        torch.mm(weight.to(dtype=self.intermediate_dtype), a1),
                        a2,
                    ).reshape(weight.shape)
                lora_diff += torch.mm(b1, b2).reshape(weight.shape)

            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)(((strength * alpha) * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"glora: {exc}") from exc
        return weight


class OFTAdapter:
    """comfy/weight_adapter/oft.py calculate_weight @ b78cec87 with the trainer alpha contract.

    Reference weights tuple: (blocks, rescale, alpha, dora_scale);
    alpha is the raw OFT constraint, not a scale. The Cayley norm bound
    is ``alpha * out_dim``, where ``out_dim = block_num * block_size``;
    raw alpha still feeds DoRA decomposition. ``rescale`` is accepted and
    IGNORED: the reference casts it to device but never uses it in
    calculate_weight at this pin (dead slot kept for tuple parity)."""

    def __init__(
        self,
        blocks: torch.Tensor,
        *,
        rescale: torch.Tensor | None = None,
        alpha: float | None = None,
        dora_scale: torch.Tensor | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.blocks = blocks
        self.rescale = rescale
        self.alpha = alpha
        self.dora_scale = dora_scale
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(self, ("blocks", "rescale", "dora_scale"))

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> OFTAdapter:
        return _rebuild_payloads(self, ("blocks", "rescale", "dora_scale"), replacements)

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        alpha = self.alpha if self.alpha is not None else 0.0
        blocks = cast_to_device(self.blocks, weight.device, self.intermediate_dtype)

        block_num, block_size = blocks.shape[0], blocks.shape[1]
        constraint = alpha * block_num * block_size

        try:
            eye = torch.eye(block_size, device=blocks.device, dtype=blocks.dtype)
            q = blocks - blocks.transpose(1, 2)
            normed_q = q
            if constraint > 0:
                q_norm = torch.norm(q) + 1e-8
                if q_norm > constraint:
                    normed_q = q * constraint / q_norm
            # float() to prevent unsupported dtypes in .inverse()
            r = (eye + normed_q) @ (eye - normed_q).float().inverse()
            r = r.to(weight)
            eye_w = torch.eye(block_size, device=weight.device, dtype=weight.dtype)
            shape = list(weight.shape[1:])
            lora_diff = torch.einsum(
                "k n m, k n ... -> k m ...",
                (r * strength) - strength * eye_w,
                weight.view(block_num, block_size, *shape),
            ).view(-1, *shape)
            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)((strength * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"oft: {exc}") from exc
        return weight


class BOFTAdapter:
    """comfy/weight_adapter/boft.py calculate_weight @ b78cec87 with the trainer alpha contract.

    Reference weights tuple: (blocks, rescale, alpha, dora_scale);
    blocks are rank 4 (butterfly stages). Alpha is the raw BOFT
    constraint; the Cayley norm bound is ``alpha * out_dim``, where
    ``out_dim = block_num * boft_b``. Raw alpha still feeds DoRA
    decomposition."""

    def __init__(
        self,
        blocks: torch.Tensor,
        *,
        rescale: torch.Tensor | None = None,
        alpha: float | None = None,
        dora_scale: torch.Tensor | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.blocks = blocks
        self.rescale = rescale
        self.alpha = alpha
        self.dora_scale = dora_scale
        self.intermediate_dtype = intermediate_dtype

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return _payload_tensors(self, ("blocks", "rescale", "dora_scale"))

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> BOFTAdapter:
        return _rebuild_payloads(self, ("blocks", "rescale", "dora_scale"), replacements)

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Upstream reads v[2] and compares alpha > 0 directly; a None
        # alpha would TypeError inside its try (logged, unpatched).
        alpha = self.alpha if self.alpha is not None else 0.0
        blocks = cast_to_device(self.blocks, weight.device, self.intermediate_dtype)
        rescale = None
        if self.rescale is not None:
            rescale = cast_to_device(self.rescale, weight.device, self.intermediate_dtype)

        boft_m, block_num, boft_b = (
            blocks.shape[0],
            blocks.shape[1],
            blocks.shape[2],
        )
        constraint = alpha * block_num * boft_b

        try:
            eye = torch.eye(boft_b, device=blocks.device, dtype=blocks.dtype)
            q = blocks - blocks.transpose(-1, -2)
            normed_q = q
            if constraint > 0:
                q_norm = torch.norm(q) + 1e-8
                if q_norm > constraint:
                    normed_q = q * constraint / q_norm
            # float() to prevent unsupported dtypes in .inverse()
            r = (eye + normed_q) @ (eye - normed_q).float().inverse()
            r = r.to(weight)
            inp = org = weight

            r_b = boft_b // 2
            for i in range(boft_m):
                bi = r[i]
                g = 2
                k = 2**i * r_b
                if strength != 1:
                    # Diverges from the reference ON PURPOSE: upstream
                    # interpolates with the intermediate-dtype eye, which
                    # promotes bi to float32 on half-precision weights and
                    # makes the einsum below always raise, so partial
                    # strength never applies there (docs/comfyui-issues/
                    # boft-fp16-partial-strength-dtype-mismatch.md).
                    # Identity in bi's dtype, exactly like oft.py's eye_w.
                    # Numerically identical when weight is float32;
                    # verified against an fp32 run in test_adapters.py.
                    bi = bi * strength + (1 - strength) * eye.to(bi)
                inp = (
                    inp.unflatten(0, (-1, g, k))
                    .transpose(1, 2)
                    .flatten(0, 2)
                    .unflatten(0, (-1, boft_b))
                )
                inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
                inp = inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)

            if rescale is not None:
                inp = inp * rescale

            lora_diff = inp - org
            lora_diff = cast_to_device(lora_diff, weight.device, self.intermediate_dtype)
            if self.dora_scale is not None:
                weight = weight_decompose(
                    self.dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    self.intermediate_dtype,
                    function,
                )
            else:
                weight += _delta_hook(function)((strength * lora_diff).type(weight.dtype))
        except AdapterMathError:
            raise
        except Exception as exc:
            raise AdapterMathError(f"boft: {exc}") from exc
        return weight


__all__ = [
    "AdapterMathError",
    "BOFTAdapter",
    "GLoRAAdapter",
    "LoHaAdapter",
    "LoKrAdapter",
    "LoRAAdapter",
    "OFTAdapter",
]
