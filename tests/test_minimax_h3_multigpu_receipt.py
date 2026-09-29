from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from tools import minimax_h3_multigpu_receipt as receipt  # noqa: E402
from tools.minimax_h3_multigpu_receipt import (  # noqa: E402
    _CollectiveTimer,
    _sparse_quality_oracle,
)


def test_collective_timer_records_and_resets_cuda_event_durations() -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Event:
        def record(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 125.0

    class Distributed:
        @staticmethod
        def all_gather(*args: object, **kwargs: object) -> str:
            calls.append(("all_gather", args, kwargs))
            return "gathered"

        @staticmethod
        def all_reduce(*args: object, **kwargs: object) -> str:
            calls.append(("all_reduce", args, kwargs))
            return "reduced"

        @staticmethod
        def all_to_all(*args: object, **kwargs: object) -> str:
            calls.append(("all_to_all", args, kwargs))
            return "exchanged"

    fake_torch = cast(
        "Any",
        type(
            "FakeTorch",
            (),
            {
                "cuda": type("Cuda", (), {"Event": staticmethod(lambda **_kwargs: Event())})(),
                "distributed": Distributed(),
            },
        )(),
    )
    timer = _CollectiveTimer(fake_torch)
    timer.install()

    assert fake_torch.distributed.all_gather("peer") == "gathered"
    assert fake_torch.distributed.all_reduce("failure", op="max") == "reduced"
    assert fake_torch.distributed.all_to_all("out", "in") == "exchanged"
    assert [call[0] for call in calls] == ["all_gather", "all_reduce", "all_to_all"]
    assert timer.elapsed_seconds() == pytest.approx(0.375)

    timer.reset()
    assert timer.elapsed_seconds() == 0.0


def test_performance_reports_compute_and_collective_shares_from_slowest_rank() -> None:
    reference = {"timing": {"median_sample_seconds": 12.0}}
    candidate = [
        {
            "timing": {
                "median_sample_seconds": 7.0,
                "median_compute_seconds": 5.0,
                "median_communication_sync_seconds": 2.0,
                "median_compute_share": 5.0 / 7.0,
                "median_communication_sync_share": 2.0 / 7.0,
            }
        },
        {
            "timing": {
                "median_sample_seconds": 8.0,
                "median_compute_seconds": 5.5,
                "median_communication_sync_seconds": 2.5,
                "median_compute_share": 5.5 / 8.0,
                "median_communication_sync_share": 2.5 / 8.0,
            }
        },
    ]

    performance = receipt._performance(reference, candidate)  # pyright: ignore[reportPrivateUsage]

    assert performance["candidate_median_seconds"] == 8.0
    assert performance["candidate_compute_seconds"] == 5.5
    assert performance["candidate_communication_sync_seconds"] == 2.5
    assert performance["candidate_compute_share"] == 5.5 / 8.0
    assert performance["candidate_communication_sync_share"] == 2.5 / 8.0


def test_pre_reset_baselines_do_not_relabel_bf16_guidance_as_int8() -> None:
    assert receipt.BASELINES[("RipperPC", "guidance", "sdpa", "production")] == 1.802
    assert ("RipperPC", "guidance", "dinkster_kitchen_int8", "production") not in (
        receipt.BASELINES
    )
    assert ("X570", "sequence", "dinkster_kitchen_int8", "production") not in (receipt.BASELINES)


def test_mint_arguments_initialize_worker_only_tensor_output() -> None:
    args = receipt._parser().parse_args(  # pyright: ignore[reportPrivateUsage]
        [
            "mint",
            "--model",
            "model.safetensors",
            "--mode",
            "sequence",
            "--policy",
            "sdpa",
            "--dinkster-root",
            "Dinkster",
            "--fork-root",
            "dinkster-inference",
            "--gpu",
            "GPU-test",
            "--output",
            "receipt.json",
            "--candidate-first",
        ]
    )

    assert args.tensor_output is None


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
