from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.flux_window_multigpu_receipt import ReceiptError, run_mint


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
