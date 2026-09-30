from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.flux_window_multigpu_receipt import (
    ReceiptError,
    _job_metrics,
    _tensor_difference,
    _workload_windows,
    run_mint,
)


def _args(tmp_path: Path, gpu_count: int) -> argparse.Namespace:
    return argparse.Namespace(
        dinkster_root=tmp_path,
        fork_root=tmp_path,
        checkpoint=tmp_path / "missing.safetensors",
        gpu=[f"GPU-{index}" for index in range(gpu_count)],
    )


def test_flux_receipt_admits_three_gpu_fallback(tmp_path: Path) -> None:
    with pytest.raises(ReceiptError, match="checkpoint does not exist"):
        run_mint(_args(tmp_path, 3))


def test_flux_receipt_rejects_unsupported_gpu_count(tmp_path: Path) -> None:
    with pytest.raises(ReceiptError, match="two, three, or four GPU UUIDs"):
        run_mint(_args(tmp_path, 1))


def test_flux_receipt_scales_window_geometry_with_resolution() -> None:
    assert _workload_windows("two-windows", 1024) == (
        tuple(range(96)),
        tuple(range(32, 128)),
    )
    assert _workload_windows("three-windows", 1024) == (
        tuple(range(56)),
        tuple(range(36, 92)),
        tuple(range(72, 128)),
    )


def test_flux_receipt_quantifies_exact_tensor_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        shape = (3,)
        dtype = "float32"

        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values

        def detach(self) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def contiguous(self) -> Tensor:
            return self

        def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, other: object
        ) -> object:
            assert isinstance(other, Tensor)
            return tuple(
                left != right
                for left, right in zip(self.values, other.values, strict=True)
            )

        def __sub__(self, other: object) -> Tensor:
            assert isinstance(other, Tensor)
            return Tensor(
                tuple(left - right for left, right in zip(self.values, other.values, strict=True))
            )

        def float(self) -> Tensor:
            return self

        def abs(self) -> Tensor:
            return Tensor(tuple(abs(value) for value in self.values))

        def max(self) -> SimpleNamespace:
            return SimpleNamespace(item=lambda: max(self.values))

        def numel(self) -> int:
            return len(self.values)

    torch = SimpleNamespace(
        count_nonzero=lambda values: SimpleNamespace(item=lambda: sum(values)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _tensor_difference(Tensor((1.0, 2.0, 3.0)), Tensor((1.0, 2.25, 2.5))) == {
        "bit_identical": False,
        "differing_values": 2,
        "max_abs_difference": 0.5,
    }


def test_flux_receipt_excludes_tensors_and_hashes_from_job_metrics() -> None:
    assert _job_metrics(
        {
            "latent": object(),
            "image": object(),
            "latent_sha256": "latent",
            "image_sha256": "image",
            "whole_job_seconds": 1.25,
        }
    ) == {"whole_job_seconds": 1.25}
