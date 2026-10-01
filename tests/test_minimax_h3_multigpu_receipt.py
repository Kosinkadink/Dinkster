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
        def get_world_size() -> int:
            return 4

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

    class Tensor:
        @staticmethod
        def numel() -> int:
            return 8

        @staticmethod
        def element_size() -> int:
            return 2

    tensor = Tensor()

    assert fake_torch.distributed.all_gather("peer", tensor) == "gathered"
    assert fake_torch.distributed.all_reduce(tensor, op="max") == "reduced"
    assert fake_torch.distributed.all_to_all("out", [tensor] * 4) == "exchanged"
    assert [call[0] for call in calls] == ["all_gather", "all_reduce", "all_to_all"]
    assert timer.elapsed_seconds() == pytest.approx(0.375)
    assert timer.evidence() == {
        "all_gather": {
            "calls": 1,
            "seconds": 0.125,
            "input_payload_bytes": 16,
            "logical_peer_bytes": 48,
        },
        "all_reduce": {
            "calls": 1,
            "seconds": 0.125,
            "input_payload_bytes": 16,
            "logical_peer_bytes": 24,
        },
        "all_to_all": {
            "calls": 1,
            "seconds": 0.125,
            "input_payload_bytes": 64,
            "logical_peer_bytes": 48,
        },
    }

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
    assert args.repeats == 5


def _matrix_result(world_size: int, rank: int = 0) -> dict[str, object]:
    hashes = {
        "latents": {"video": "latent-video", "audio": "latent-audio"},
        "decoded": {"video": "decoded-video", "audio": "decoded-audio"},
    }
    return {
        "rank": rank,
        "world_size": world_size,
        "output_hashes": {
            "cold": hashes,
            "warmups": [hashes],
            "measured": [hashes, hashes, hashes],
        },
        "timing": {"whole_job_seconds": [10.0 + rank, 11.0 + rank, 12.0 + rank]},
        "peak_allocated_bytes": 100 + rank,
        "peak_reserved_bytes": 200 + rank,
    }


def test_matrix_hash_validation_rejects_a_later_repeat() -> None:
    first = _matrix_result(1)
    second = _matrix_result(2, 1)
    cast("dict[str, Any]", second["output_hashes"])["measured"][-1] = {
        "latents": {"video": "changed", "audio": "latent-audio"},
        "decoded": {"video": "decoded-video", "audio": "decoded-audio"},
    }
    segments = [
        {"arm": "serial", "ranks": [first]},
        {"arm": "u2", "ranks": [_matrix_result(2), second]},
    ]

    with pytest.raises(receipt.ReceiptError, match="u2 rank 1 measured-2"):
        receipt._require_identical_matrix_hashes(segments)  # pyright: ignore[reportPrivateUsage]


def test_sequence_matrix_uses_mirrored_order_and_selected_u2_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str], int, int]] = []

    def run_serial(
        _args: argparse.Namespace,
        _scratch: Path,
        *,
        name: str,
        gpu: str,
        repeats: int,
        warmups: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((name, [gpu], repeats, warmups))
        result = _matrix_result(1)
        cast("dict[str, Any]", result["output_hashes"])["warmups"] = [
            cast("dict[str, Any]", result["output_hashes"])["cold"]
        ] * warmups
        cast("dict[str, Any]", result["output_hashes"])["measured"] = [
            cast("dict[str, Any]", result["output_hashes"])["cold"]
        ] * repeats
        cast("dict[str, Any]", result["timing"])["whole_job_seconds"] = [10.0] * repeats
        return result

    def run_distributed(
        _args: argparse.Namespace,
        _scratch: Path,
        *,
        name: str,
        gpus: list[str],
        repeats: int,
        warmups: int,
    ) -> list[dict[str, object]]:
        calls.append((name, list(gpus), repeats, warmups))
        return [
            run_serial(
                _args,
                _scratch,
                name=f"{name}-rank{rank}",
                gpu=gpu,
                repeats=repeats,
                warmups=warmups,
            )
            for rank, gpu in enumerate(gpus)
        ]

    monkeypatch.setattr(receipt, "_run_serial", run_serial)
    monkeypatch.setattr(receipt, "_run_distributed", run_distributed)
    args = argparse.Namespace(
        repeats=5,
        warmups=1,
        gpu=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
        u2_gpu=["GPU-1", "GPU-3"],
    )

    segments, summaries = receipt._run_sequence_matrix(  # pyright: ignore[reportPrivateUsage]
        args, tmp_path
    )

    assert [segment["arm"] for segment in segments] == [
        "serial",
        "u2",
        "u4",
        "u4",
        "u2",
        "serial",
    ]
    assert [(call[2], call[3]) for call in calls if "rank" not in call[0]] == [
        (3, 1),
        (3, 1),
        (3, 1),
        (2, 0),
        (2, 0),
        (2, 0),
    ]
    assert segments[1]["gpu_uuids"] == ["GPU-1", "GPU-3"]
    assert summaries["serial"]["world_size"] == 1
    assert summaries["u2"]["world_size"] == 2
    assert summaries["u4"]["world_size"] == 4


def test_matrix_receipt_records_provider_boundaries_and_u2_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.safetensors"
    video_vae = tmp_path / "video.safetensors"
    audio_vae = tmp_path / "audio.safetensors"
    for path in (model, video_vae, audio_vae):
        path.write_bytes(b"artifact")
    summaries = {
        "serial": {"median_whole_job_seconds": 20.0},
        "u2": {"median_whole_job_seconds": 12.0},
        "u4": {"median_whole_job_seconds": 13.0},
    }
    monkeypatch.setattr(receipt, "_run_sequence_matrix", lambda *_args: ([], summaries))
    monkeypatch.setattr(receipt, "_git", lambda *_args: "head")
    monkeypatch.setattr(receipt, "_nvidia_smi", lambda *_args: "topology")
    output = tmp_path / "receipt.json"
    args = argparse.Namespace(
        dinkster_root=tmp_path,
        fork_root=tmp_path,
        model=model,
        video_vae=video_vae,
        audio_vae=audio_vae,
        model_digest="blake3:model",
        video_vae_digest="blake3:video",
        audio_vae_digest="blake3:audio",
        model_source_url="https://models.example/model.safetensors",
        model_source_revision="revision-test",
        width=1344,
        height=768,
        frames=124,
        steps=20,
        seed=1,
        warmups=1,
        repeats=5,
        policy="sdpa",
        gpu=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
        u2_gpu=["GPU-1", "GPU-3"],
        output=output,
    )

    assert (
        receipt._mint_sequence_matrix(  # pyright: ignore[reportPrivateUsage]
            args, {"provider": "bf16-linear"}
        )
        == 0
    )

    minted = json.loads(output.read_text())
    execution = minted["execution"]
    assert execution["production_route"] == "u2-fallback"
    assert execution["u4_beats_u2"] is False
    assert execution["provider_cells"]["ring4"]["status"] == "provider-unavailable"
    assert execution["provider_cells"]["ulysses2_ring2"]["status"] == "provider-unavailable"


def test_matrix_requires_four_unique_gpus_and_both_decoders(tmp_path: Path) -> None:
    base = {
        "dinkster_root": tmp_path,
        "fork_root": tmp_path,
        "model": tmp_path / "model.safetensors",
        "mode": "sequence",
        "policy": "sdpa",
        "width": 64,
        "height": 64,
        "frames": 5,
        "steps": 1,
        "seed": 1,
        "warmups": 1,
        "repeats": 5,
        "output": tmp_path / "receipt.json",
        "matrix": True,
        "candidate_first": False,
        "model_source_url": "https://models.example/model.safetensors",
        "model_source_revision": "revision-test",
        "tensor_output": None,
    }
    with pytest.raises(receipt.ReceiptError, match="four unique GPU"):
        receipt.run_mint(
            argparse.Namespace(
                **base,
                gpu=["GPU-0", "GPU-1", "GPU-2"],
                u2_gpu=["GPU-0", "GPU-1"],
                video_vae=tmp_path / "video.safetensors",
                audio_vae=tmp_path / "audio.safetensors",
            )
        )
    with pytest.raises(receipt.ReceiptError, match="video and audio VAEs"):
        receipt.run_mint(
            argparse.Namespace(
                **base,
                gpu=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
                u2_gpu=["GPU-0", "GPU-1"],
                video_vae=None,
                audio_vae=None,
            )
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
