from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from tools import minimax_h3_multigpu_receipt as receipt  # noqa: E402
from tools.minimax_h3_multigpu_receipt import _sparse_quality_oracle  # noqa: E402


def test_pre_reset_baselines_do_not_relabel_bf16_guidance_as_int8() -> None:
    assert receipt.BASELINES[("RipperPC", "guidance", "sdpa", "production")] == 1.802
    assert ("RipperPC", "guidance", "dinkster_kitchen_int8", "production") not in (
        receipt.BASELINES
    )


def test_sparse_quality_oracle_reports_dense_relative_error_by_role(tmp_path) -> None:
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "audio": torch.tensor([3.0, 4.0]),
            "video": torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        },
        reference_path,
    )
    torch.save(
        {
            "audio": torch.tensor([0.0, 4.0]),
            "video": torch.tensor([[2.0, -2.0], [3.0, -6.0]]),
        },
        candidate_path,
    )

    oracle = _sparse_quality_oracle(reference_path, candidate_path)

    assert oracle["baseline"].startswith("same-session dense SDPA")
    audio = oracle["metrics_by_role"]["audio"]
    assert audio["shape"] == [2]
    assert audio["max_abs"] == 3.0
    assert audio["rmse"] == pytest.approx(3.0 / 2.0**0.5)
    assert audio["relative_rmse"] == pytest.approx(0.6)
    video = oracle["metrics_by_role"]["video"]
    assert video["shape"] == [2, 2]
    assert video["max_abs"] == 2.0
    assert video["rmse"] == pytest.approx(5.0**0.5 / 2.0)
    assert video["cosine_similarity"] == pytest.approx(39.0 / (30.0 * 53.0) ** 0.5)


def test_sparse_mint_reads_quality_tensors_before_scratch_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    dinkster = tmp_path / "dinkster"
    fork = tmp_path / "fork"
    dinkster.mkdir()
    fork.mkdir()
    observed: list[tuple[bool, bool]] = []

    def run_serial(
        _args: argparse.Namespace, scratch: Path, *, sparse_enabled: bool = False
    ) -> dict[str, object]:
        tensor = scratch / ("sparse.pt" if sparse_enabled else "serial.pt")
        torch.save({"video": torch.tensor([float(sparse_enabled)])}, tensor)
        return {"_tensor_output": str(tensor), "output_hashes": {"video": "hash"}}

    def quality(reference: Path, candidate: Path) -> dict[str, object]:
        observed.append((reference.is_file(), candidate.is_file()))
        return {"quality": "loaded"}

    monkeypatch.setattr(
        receipt, "_git", lambda _root, *arguments: "" if arguments[0] == "status" else "head"
    )
    monkeypatch.setattr(receipt, "_run_serial", run_serial)
    monkeypatch.setattr(receipt, "_sparse_quality_oracle", quality)
    monkeypatch.setattr(receipt, "_performance", lambda *_args: {"speedup": 1.0})
    monkeypatch.setattr(receipt, "_classify_host", lambda *_args: "test-host")
    monkeypatch.setattr(receipt, "_nvidia_smi", lambda *_args: "topology")
    output = tmp_path / "receipt.json"
    args = argparse.Namespace(
        dinkster_root=dinkster,
        fork_root=fork,
        model=model,
        mode="sparse",
        policy="sdpa",
        width=64,
        height=64,
        frames=5,
        steps=1,
        seed=1,
        warmups=0,
        repeats=1,
        gpu=["GPU-test"],
        output=output,
        candidate_first=False,
    )

    assert receipt.run_mint(args) == 0
    assert observed == [(True, True)]
    assert output.is_file()
