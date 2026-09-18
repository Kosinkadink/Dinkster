"""GGUF cross-implementation parity contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.inference_parity import harness
from tools.inference_parity.gguf_comfyui_adapter import _regular_decode
from tools.inference_parity.gguf_reference import compare_reference


def _receipt(root: Path, engine: str) -> dict[str, Any]:
    image = root / f"{engine}-image.npy"
    latent = root / f"{engine}-latent.npy"
    np.save(image, np.zeros((1, 16, 16, 3), dtype=np.float32), allow_pickle=False)
    np.save(latent, np.zeros((1, 4, 2, 2), dtype=np.float32), allow_pickle=False)
    return {
        "attention_backend": "sdpa",
        "engine": engine,
        "gguf": "0.19.0",
        "image_path": str(image),
        "latent_path": str(latent),
        "python": "3.12.3",
        "q8_tensors": [{"key": "weight", "sha256": "a" * 64, "shape": [2, 2]}],
        "runtime_facts": ["artifact=digest"],
        "tensor_map": [{"key": "weight", "shape": [2, 2], "type": "Q8_0"}],
        "torch": "2.13.0+cu130",
    }


def _policy() -> dict[str, Any]:
    return {
        "workloads": {
            "test": {
                "reference": {
                    "image": {"max_abs": "0", "mean_abs": "0", "ssim_minimum": "1"},
                    "latent": {"max_abs": "0", "mean_abs": "0"},
                }
            }
        }
    }


def _workload() -> dict[str, Any]:
    return {
        "engines": {
            "comfyui": {"commit": "c" * 40},
            "dinkster": {"commit": "d" * 40},
        },
        "id": "test",
    }


def test_reference_requires_exact_mapping_and_every_q8_digest(tmp_path: Path) -> None:
    comfyui = _receipt(tmp_path, "comfyui")
    dinkster = _receipt(tmp_path, "dinkster")
    result = compare_reference(comfyui, dinkster, _policy(), _workload())
    assert result["overall_pass"] is True
    assert result["q8_dequantization"] == {
        "bit_exact": True,
        "complete": True,
        "mismatched_keys": [],
        "tensor_count": 1,
    }
    forged = deepcopy(dinkster)
    forged["q8_tensors"][0]["sha256"] = "b" * 64
    result = compare_reference(comfyui, forged, _policy(), _workload())
    assert result["overall_pass"] is False
    assert result["q8_dequantization"]["mismatched_keys"] == ["weight"]
    forged = deepcopy(dinkster)
    forged["tensor_map"][0]["shape"] = [4, 1]
    result = compare_reference(comfyui, forged, _policy(), _workload())
    assert result["overall_pass"] is False
    assert result["tensor_mapping"]["exact"] is False
    missing = deepcopy(dinkster)
    missing["q8_tensors"] = []
    result = compare_reference(comfyui, missing, _policy(), _workload())
    assert result["overall_pass"] is False
    assert result["q8_dequantization"]["complete"] is False
    duplicate = deepcopy(dinkster)
    duplicate["tensor_map"].append(deepcopy(duplicate["tensor_map"][0]))
    result = compare_reference(comfyui, duplicate, _policy(), _workload())
    assert result["overall_pass"] is False
    assert result["tensor_mapping"]["keys_unique"] is False


def test_gguf_manifest_freezes_repeated_and_settings_change_requests() -> None:
    manifest = harness.load_json(Path("tools/inference_parity/workloads_gguf.json"))
    acceptance = harness.load_json(Path("tools/inference_parity/acceptance_gguf.json"))
    harness.validate_manifest(manifest)
    harness.validate_acceptance(acceptance)
    workload = manifest["workloads"][0]
    requests = harness.workload_timing_requests(workload)
    assert tuple(request["phase"] for request in requests) == (
        "cold",
        "discard-1",
        "discard-2",
        "discard-3",
        "repeat-1",
        "repeat-2",
        "repeat-3",
        "repeat-4",
        "repeat-5",
        "repeat-6",
        "repeat-7",
        "settings-change-1",
        "settings-change-2",
    )
    assert all(request.get("record") is False for request in requests[1:4])
    assert all(request["cfg"] == 6.0 for request in requests[-2:])
    assert workload["engines"]["comfyui"]["commit"] == ("947c2749dd04c51ef0e21b069544d8b0b4f9b411")
    assert workload["engines"]["comfyui"]["extensions"][0]["commit"] == (
        "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
    )
    assert workload["engines"]["comfyui"]["dependencies"] == {
        "gguf": "0.19.0",
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
    }
    malformed = deepcopy(manifest)
    malformed["workloads"][0]["execution"]["timing"]["settings_change"]["cfg"] = float("nan")
    with pytest.raises(harness.HarnessError, match="settings_change is invalid"):
        harness.validate_manifest(malformed)
    malformed = deepcopy(manifest)
    del malformed["workloads"][0]["engines"]["comfyui"]["adapter"]
    with pytest.raises(harness.HarnessError, match="adapter must be str"):
        harness.validate_manifest(malformed)


@pytest.mark.parametrize(
    ("head", "status", "match"),
    (
        ("b" * 40, "", "extension checkout changed"),
        ("a" * 40, " M extension.py", "extension checkout became dirty"),
    ),
)
def test_reference_extension_recheck_refuses_nested_repo_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    status: str,
    match: str,
) -> None:
    extension = tmp_path / "custom_nodes" / "ComfyUI-GGUF"
    pins = {
        "extensions": [
            {
                "commit": "a" * 40,
                "resolved_path": str(extension),
            }
        ]
    }
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: status if args[0] == "status" else head,
    )
    with pytest.raises(harness.HarnessError, match=match):
        harness.validate_extensions_unchanged(pins, "comfyui", activity="reference execution")


def test_single_gpu_timing_history_is_valid() -> None:
    history = harness.load_json(Path("tools/inference_parity/timings_gguf.json"))
    harness.validate_timing_history(history)
    forged = deepcopy(history)
    forged["hardware"]["gpus"] = []
    with pytest.raises(harness.HarnessError, match="nonempty exact GPU inventory"):
        harness.validate_timing_history(forged)


def test_gguf_runtime_receipt_requires_pinned_dependency_versions() -> None:
    workload = harness.load_json(Path("tools/inference_parity/workloads_gguf.json"))["workloads"][0]
    runtime_reply = {
        "actual_dtypes": {
            "codec": "float32",
            "dequant": "float16",
            "diffusion": "float16",
            "text": "float32",
        },
        "attention_backend": "pytorch-sdpa",
        "gguf": "0.19.0",
        "memory_flags": {"cpu": False, "highvram": False, "normalvram": True},
        "memory_policy": "VRAMState.NORMAL_VRAM",
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
    }
    runtime = harness._adapter_runtime(runtime_reply, workload, "comfyui")
    assert runtime["attention_backend_normalized"] == "sdpa"
    forged = deepcopy(runtime_reply)
    forged["actual_dtypes"]["diffusion"] = "float64"
    with pytest.raises(harness.HarnessError, match="runtime actual_dtypes mismatch"):
        harness._adapter_runtime(forged, workload, "comfyui")
    forged = {**runtime_reply, "attention_backend": "arbitrary"}
    with pytest.raises(harness.HarnessError, match="runtime attention_backend mismatch"):
        harness._adapter_runtime(forged, workload, "comfyui")
    with pytest.raises(harness.HarnessError, match="python mismatch"):
        harness._adapter_runtime({**runtime_reply, "python": "3.13.0"}, workload, "comfyui")
    with pytest.raises(harness.HarnessError, match="gguf is required"):
        harness._adapter_runtime(
            {key: value for key, value in runtime_reply.items() if key != "gguf"},
            workload,
            "comfyui",
        )
    with pytest.raises(harness.HarnessError, match="torch mismatch"):
        harness._adapter_runtime({**runtime_reply, "torch": "2.12.0+cu130"}, workload, "comfyui")


def test_timing_correctness_groups_outputs_by_request_settings(tmp_path: Path) -> None:
    workload = harness.load_json(Path("tools/inference_parity/workloads_gguf.json"))["workloads"][0]
    acceptance = harness.load_json(Path("tools/inference_parity/acceptance_gguf.json"))
    requests = harness.workload_timing_requests(workload)
    processes = []
    for process_index, engine in enumerate(harness.TIMING_ENGINE_ORDER, 1):
        observations = {}
        for request in requests:
            cfg = request.get("cfg", workload["execution"]["cfg"])
            value = 1.0 if cfg == 6.0 else 0.0
            path = tmp_path / f"{process_index}-{request['phase']}.npy"
            np.save(path, np.full((1, 2, 2, 3), value, dtype=np.float32), allow_pickle=False)
            observations[request["phase"]] = {
                "outputs": {"image": {"digest": harness.digest_file(path), "path": str(path)}},
                "runtime": {"torch": "2.13.0+cu130"},
            }
        processes.append({"engine": {"id": engine}, "observations": observations})
    assert harness._timing_correctness(processes, acceptance, workload)["pass"] is True
    divergent = tmp_path / "1-repeat-2.npy"
    np.save(divergent, np.ones((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes[0]["observations"]["repeat-2"]["outputs"]["image"] = {
        "digest": harness.digest_file(divergent),
        "path": str(divergent),
    }
    with pytest.raises(harness.HarnessError, match="output correctness failed"):
        harness._timing_correctness(processes, acceptance, workload)


def test_regular_decode_refuses_hidden_tiled_fallback() -> None:
    class VAE:
        def decode_tiled_(self) -> None:
            return None

        def decode(self, _latent: object) -> str:
            self.decode_tiled_()
            return "image"

    with pytest.raises(RuntimeError, match="fell back to tiled"):
        _regular_decode(VAE(), object())


def test_repeated_timing_summary_gates_warm_median_and_records_spread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = harness.load_json(Path("tools/inference_parity/workloads_gguf.json"))
    workload = manifest["workloads"][0]
    requests = harness.workload_timing_requests(workload)
    phases = tuple(request["phase"] for request in requests)
    commit = "a" * 40
    processes = []
    for process_index, engine in enumerate(harness.TIMING_ENGINE_ORDER, 1):
        directory = tmp_path / f"{process_index:02d}-{engine}"
        directory.mkdir()
        for kind in ("stderr", "stdout"):
            (directory / f"{engine}.{kind}.txt").write_text("")
        base = 100 if engine == "comfyui" else 80
        observations = {}
        for index, request in enumerate(requests):
            recorded = request.get("record", True)
            metrics = {}
            if recorded:
                elapsed = base + (index % 3)
                metrics = {
                    "cold_load_ns": 20,
                    "decode_ns": 10,
                    "end_to_end_ns": elapsed,
                    "max_memory_allocated_bytes": 1000,
                    "max_memory_reserved_bytes": 1100,
                    "peak_ram_bytes": 2000,
                    "peak_vram_bytes": 3000,
                    "sampling_ns": elapsed - 10,
                }
            observations[request["phase"]] = {
                "metrics": metrics,
                "runtime": {
                    "actual_dtypes": {"diffusion": "float16"},
                    "attention_backend": "sdpa",
                    "memory_policy": "resident",
                    "torch": "2.13.0+cu130",
                },
            }
        inventory = {
            "cpu": "cpu",
            "gpus": [
                {
                    "driver": "1",
                    "index": 0,
                    "memory_total_mib": 1,
                    "name": "gpu",
                    "uuid": "GPU-test",
                }
            ],
            "hostname": "host",
            "platform": "linux",
            "ram_bytes": 1,
        }
        processes.append(
            {
                "engine": {"commit": engine[0] * 40, "id": engine},
                "hardware": {
                    "device_ordinal": 0,
                    "device_uuid": "GPU-test",
                    "inventory": inventory,
                },
                "harness": {"commit": commit},
                "observations": observations,
                "stderr_digest": harness.digest_bytes(b""),
                "stderr_path": f"{engine}.stderr.txt",
                "stdout_digest": harness.digest_bytes(b""),
                "stdout_path": f"{engine}.stdout.txt",
            }
        )
    result = harness.summarize_timing(
        processes,
        {
            "acceptance_manifest_digest": workload["acceptance_manifest"]["digest"],
            "checks": [{} for _ in range(len(phases) * 4)],
            "pass": True,
            "torch": "2.13.0+cu130",
        },
        workload,
        harness.load_json(Path("tools/inference_parity/timings_gguf.json")),
        {"dinkster": Path.cwd()},
        tmp_path,
        "2026-08-18T00:00:00+00:00",
    )
    assert result["status"] == "PASS"
    assert result["escalation"] == {"reasons": [], "triggered": False}
    assert result["engines"]["comfyui"]["warm_end_to_end"]["count"] == 14
    assert result["ratios"]["dinkster_over_comfyui_warm_end_to_end"] == "0.801980198"

    processes[-1]["engine"]["commit"] = "f" * 40
    with pytest.raises(harness.HarnessError, match="comfyui commit changed"):
        harness.summarize_timing(
            processes,
            {
                "acceptance_manifest_digest": workload["acceptance_manifest"]["digest"],
                "checks": [{} for _ in range(len(phases) * 4)],
                "pass": True,
                "torch": "2.13.0+cu130",
            },
            workload,
            harness.load_json(Path("tools/inference_parity/timings_gguf.json")),
            {"dinkster": Path.cwd()},
            tmp_path,
            "2026-08-18T00:00:00+00:00",
        )
