from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from dinkster_inference_wire.sparse import (
    SparseLatent,
    SparseSupport,
)


@dataclass(frozen=True)
class Tensor:
    values: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.values), len(self.values[0]) if self.values else 0)

    def _apply(self, other: Tensor | float, operation: str) -> Tensor:
        if type(other) is Tensor:
            peer = other.values
        else:
            scalar = cast(float, other)
            peer = tuple((scalar,) * self.shape[1] for _ in self.values)

        def apply(a: float, b: float) -> float:
            if operation == "add":
                return a + b
            if operation == "sub":
                return a - b
            if operation == "mul":
                return a * b
            return a / b

        return Tensor(
            tuple(
                tuple(apply(a, b) for a, b in zip(left, right, strict=True))
                for left, right in zip(self.values, peer, strict=True)
            )
        )

    def __add__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "add")

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "sub")

    def __mul__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "mul")

    def __truediv__(self, other: Tensor | float) -> Tensor:
        return self._apply(other, "div")


def support(digit: str = "1") -> SparseSupport[Tensor]:
    return SparseSupport(
        Tensor(((0, 1, 2, 3), (0, 4, 5, 6))),
        (2,),
        64,
        (-0.5, -0.5, -0.5),
        (1.0 / 64, 1.0 / 64, 1.0 / 64),
        "sha256:" + digit * 64,
    )


def test_sparse_latent_arithmetic_preserves_and_authenticates_support() -> None:
    first = SparseLatent(support(), Tensor(((1.0, 2.0), (3.0, 4.0))))
    second = SparseLatent(first.support, Tensor(((5.0, 6.0), (7.0, 8.0))))
    assert (first + second).features.values == ((6.0, 8.0), (10.0, 12.0))
    rehydrated = SparseLatent(support(), second.features)
    assert (first + rehydrated).features.values == ((6.0, 8.0), (10.0, 12.0))
    assert (first * 2.0).support is first.support
    assert (first / 2.0).features.values == ((0.5, 1.0), (1.5, 2.0))
    with pytest.raises(ValueError, match="identical support"):
        _ = first + SparseLatent(support("2"), second.features)
