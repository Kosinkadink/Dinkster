from __future__ import annotations

import argparse
import json
import struct
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


def test_attention_timer_records_selected_backend_calls() -> None:
    class Event:
        def record(self) -> None:
            pass

        def elapsed_time(self, _other: object) -> float:
            return 25.0

    fake_torch = cast(
        "Any",
        type(
            "FakeTorch",
            (),
            {"cuda": type("Cuda", (), {"Event": staticmethod(lambda **_kwargs: Event())})()},
        )(),
    )
    timer = receipt._AttentionTimer(  # pyright: ignore[reportPrivateUsage]
        fake_torch, lambda value: value + 1
    )

    assert timer(4) == 5
    assert timer.evidence() == {"calls": 1, "seconds": 0.025}
    assert timer.total_calls == 1
    timer.reset()
    assert timer.evidence() == {"calls": 0, "seconds": 0.0}
    assert timer.total_calls == 1


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
    assert receipt.INT8_WEIGHT_BASELINES[("X570", "sequence", "sdpa", "production")] == 1.327


def test_model_artifact_identifies_int8_convrot_and_excludes_quant_metadata(
    tmp_path: Path,
) -> None:
    quantization = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64},
        separators=(",", ":"),
    ).encode()
    header = {
        "block.weight": {"dtype": "I8", "shape": [3, 5], "data_offsets": [0, 15]},
        "block.comfy_quant": {
            "dtype": "U8",
            "shape": [len(quantization)],
            "data_offsets": [15, 15 + len(quantization)],
        },
        "norm.weight": {
            "dtype": "BF16",
            "shape": [5],
            "data_offsets": [15 + len(quantization), 25 + len(quantization)],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    model = tmp_path / "model.safetensors"
    model.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(15) + quantization + bytes(10)
    )

    artifact = receipt._model_artifact(model)  # pyright: ignore[reportPrivateUsage]

    assert artifact["provider"] == "int8-convrot"
    assert artifact["logical_weight_bytes"] == 25
    assert artifact["quantized_linear_weights"] == 1
    assert artifact["tensor_dtype_counts"] == {"BF16": 1, "I8": 1, "U8": 1}
    assert artifact["quantization_format"] == "int8_tensorwise"
    assert artifact["convrot_group_sizes"] == [64]
    assert artifact["sha256"] == receipt.hashlib.sha256(model.read_bytes()).hexdigest()

    non_convrot = tmp_path / "non-convrot.safetensors"
    non_convrot.write_bytes(model.read_bytes().replace(b'"convrot":true', b'"convrot":null'))
    assert (
        receipt._model_artifact(non_convrot)[  # pyright: ignore[reportPrivateUsage]
            "provider"
        ]
        == "mixed-linear"
    )


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
            "--model-source-url",
            "https://models.example/model.safetensors",
            "--model-source-revision",
            "revision-test",
            "--gpu",
            "GPU-test",
            "--output",
            "receipt.json",
            "--candidate-first",
        ]
    )

    assert args.tensor_output is None


def test_dense_receipt_requires_fa4_policy_and_both_decoders(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    dinkster = tmp_path / "dinkster"
    fork = tmp_path / "fork"
    dinkster.mkdir()
    fork.mkdir()
    base = [
        "mint",
        "--model",
        str(model),
        "--mode",
        "dense",
        "--dinkster-root",
        str(dinkster),
        "--fork-root",
        str(fork),
        "--model-source-url",
        "https://models.example/model.safetensors",
        "--model-source-revision",
        "revision-test",
        "--gpu",
        "GPU-test",
        "--output",
        str(tmp_path / "receipt.json"),
    ]

    wrong_policy = receipt._parser().parse_args(  # pyright: ignore[reportPrivateUsage]
        [*base, "--policy", "sdpa"]
    )
    with pytest.raises(receipt.ReceiptError, match="requires flash4_sm120_dense"):
        receipt.run_mint(wrong_policy)

    missing_decoders = receipt._parser().parse_args(  # pyright: ignore[reportPrivateUsage]
        [*base, "--policy", "flash4_sm120_dense"]
    )
    with pytest.raises(receipt.ReceiptError, match="requires video and audio VAEs"):
        receipt.run_mint(missing_decoders)


def test_dense_mint_routes_sdpa_and_fa4_and_rejects_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.safetensors"
    video_vae = tmp_path / "video-vae.safetensors"
    audio_vae = tmp_path / "audio-vae.safetensors"
    for path in (model, video_vae, audio_vae):
        path.write_bytes(b"artifact")
    dinkster = tmp_path / "dinkster"
    fork = tmp_path / "fork"
    dinkster.mkdir()
    fork.mkdir()
    policies: list[str] = []

    def run_serial(
        args: argparse.Namespace,
        _scratch: Path,
        *,
        sparse_enabled: bool = False,
        policy: str | None = None,
        name: str | None = None,
    ) -> dict[str, object]:
        del sparse_enabled, name
        selected = policy or args.policy
        policies.append(selected)
        output_hashes = {"latents": {"video": "same"}, "decoded": {"video": "same"}}
        if selected == "flash4_sm120_dense":
            output_hashes["decoded"]["video"] = "different"
        return {"output_hashes": output_hashes}

    monkeypatch.setattr(
        receipt, "_git", lambda _root, *arguments: "" if arguments[0] == "status" else "head"
    )
    monkeypatch.setattr(receipt, "_run_serial", run_serial)
    monkeypatch.setattr(
        receipt,
        "_model_artifact",
        lambda *_args: {
            "sha256": "model-sha256",
            "provider": "bf16-linear",
            "logical_weight_bytes": 8,
            "tensor_dtype_counts": {"BF16": 1},
            "quantized_linear_weights": 0,
            "quantization_format": None,
            "convrot_group_sizes": [],
        },
    )
    args = argparse.Namespace(
        dinkster_root=dinkster,
        fork_root=fork,
        model=model,
        video_vae=video_vae,
        audio_vae=audio_vae,
        mode="dense",
        policy="flash4_sm120_dense",
        width=64,
        height=64,
        frames=5,
        steps=1,
        seed=1,
        warmups=1,
        repeats=3,
        gpu=["GPU-test"],
        output=tmp_path / "receipt.json",
        candidate_first=False,
        model_source_url="https://models.example/model.safetensors",
        model_source_revision="revision-test",
    )

    with pytest.raises(receipt.ReceiptError, match="output hashes differ"):
        receipt.run_mint(args)

    assert policies == ["sdpa", "flash4_sm120_dense"]


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
    monkeypatch.setattr(
        receipt,
        "_model_artifact",
        lambda *_args: {
            "sha256": "model-sha256",
            "provider": "bf16-linear",
            "logical_weight_bytes": 5,
            "tensor_dtype_counts": {"BF16": 1},
            "quantized_linear_weights": 0,
            "quantization_format": None,
            "convrot_group_sizes": [],
        },
    )
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
        model_source_url="https://models.example/model.safetensors",
        model_source_revision="revision-test",
    )

    assert receipt.run_mint(args) == 0
    assert observed == [(True, True)]
    assert output.is_file()
