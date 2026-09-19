from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tools.inference_parity.chroma_e2e as chroma_e2e
from tools.inference_parity.chroma_e2e import (
    COMFYUI_COMMIT,
    MEASURED_RUNS,
    PROCESS_ORDER,
    TEMPLATES,
    GateError,
    _artifact_receipts,
    _asset,
    _clear_cublas_workspaces,
    _comfyui_residency_receipt,
    _gpu_target,
    _process_vram_bytes,
    _residency_receipt,
    _trim_process_heap,
    _verify_physical_gpu,
    build_verdict,
    extract_workload,
    run_compare,
)


def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    payloads = {
        "chroma": b"chroma weights",
        "t5xxl": b"t5 weights",
        "vae": b"vae weights",
    }
    for name, payload in payloads.items():
        relative = f"models/{name}.safetensors"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        monkeypatch.setitem(
            chroma_e2e.ARTIFACTS,
            name,
            {
                "path": relative,
                "revision": "test",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "url": "https://example.invalid/test",
            },
        )
    return payloads


@pytest.mark.skipif(os.name != "posix", reason="requires descriptor-bound POSIX records")
def test_artifact_preflight_record_skips_measured_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_assets.integrity as integrity
    from dinkster_assets import AssetRef

    payloads = _artifact_root(tmp_path, monkeypatch)
    receipts = _artifact_receipts(tmp_path, "chroma")
    assert all("verification" in receipt for receipt in receipts.values())

    def unexpected_hash(_handle: object) -> str:
        raise AssertionError("preflight-verified artifact was rehashed")

    monkeypatch.setattr(integrity, "_hash_handle", unexpected_hash)
    assert _asset(AssetRef, receipts["chroma"]).read_bytes() == payloads["chroma"]


@pytest.mark.skipif(os.name != "posix", reason="requires descriptor-bound POSIX records")
def test_artifact_preflight_record_fails_closed_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import AssetIntegrityError, AssetRef

    _artifact_root(tmp_path, monkeypatch)
    receipt = _artifact_receipts(tmp_path, "chroma")["chroma"]
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"replacement weights")
    replacement.replace(receipt["path"])

    with pytest.raises(AssetIntegrityError, match="stale_ingest_record"):
        _asset(AssetRef, receipt).read_bytes()


def _workflow(variant: str) -> dict[str, Any]:
    radiance = variant == "radiance"
    nodes = [
        {
            "id": 1,
            "type": "CFGGuider",
            "inputs": [
                {"name": "positive", "link": 101},
                {"name": "negative", "link": 102},
            ],
            "widgets_values": [3.5],
        },
        {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["positive"]},
        {"id": 3, "type": "CLIPTextEncode", "widgets_values": ["negative"]},
        {
            "id": 4,
            "type": "EmptyChromaRadianceLatentImage" if radiance else "EmptySD3LatentImage",
            "widgets_values": [1024, 1024, 1],
        },
        {
            "id": 5,
            "type": "BetaSamplingScheduler" if radiance else "BasicScheduler",
            "widgets_values": [30, 0.4, 0.4] if radiance else ["beta", 26, 1.0],
        },
        {"id": 6, "type": "UNETLoader", "widgets_values": [f"{variant}.safetensors"]},
        {"id": 7, "type": "CLIPLoader", "widgets_values": ["t5.safetensors"]},
        {"id": 8, "type": "KSamplerSelect", "widgets_values": ["euler"]},
        {"id": 9, "type": "ModelSamplingAuraFlow", "widgets_values": [1.0]},
        {"id": 10, "type": "RandomNoise", "widgets_values": [12345]},
        {
            "id": 11,
            "type": "VAELoader",
            "widgets_values": ["pixel_space" if radiance else "ae.safetensors"],
        },
    ]
    if radiance:
        nodes.append(
            {
                "id": 12,
                "type": "ChromaRadianceOptions",
                "widgets_values": [True, 1.0, 0.0, -1, False],
            }
        )
    graph = {
        "links": [[101, 2, 0, 1, 0, "CONDITIONING"], [102, 3, 0, 1, 1, "CONDITIONING"]],
        "nodes": nodes,
    }
    return {"definitions": {"subgraphs": [graph]}} if radiance else graph


@pytest.mark.parametrize("variant", ["chroma", "radiance"])
def test_extract_workload_preserves_official_variant_semantics(
    tmp_path: Path, variant: str
) -> None:
    path = tmp_path / TEMPLATES[variant]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_workflow(variant)))

    workload = extract_workload(tmp_path, variant)

    assert workload["variant"] == variant
    assert (workload["width"], workload["height"]) == (1024, 1024)
    assert workload["sampler"] == "euler"
    assert workload["scheduler"]["steps"] == (30 if variant == "radiance" else 26)
    assert workload["vae_name"] == ("pixel_space" if variant == "radiance" else "ae.safetensors")
    if variant == "radiance":
        assert workload["model_options"] == {
            "end_sigma": 0.0,
            "force_sequential_txt_ids": False,
            "nerf_tile_size": -1,
            "preserve_wrapper": True,
            "start_sigma": 1.0,
        }
    else:
        assert workload["model_options"] is None


def test_comfyui_residency_receipt_records_vbar_and_pin_state() -> None:
    subsets = {
        name: (SimpleNamespace(size=index), [object()] * index, [index - 1], [index * 2], [], {})
        for index, name in enumerate(
            ("weights", "weights-loaded", "patches", "patches-loaded"), start=1
        )
    }
    pin_state = {**subsets, "active": True}
    model = SimpleNamespace(dynamic_pins={"cuda:0": pin_state}, model_loaded_weight_memory=12)
    patcher = SimpleNamespace(
        load_device="cuda:0",
        loaded_size=lambda: 34,
        model=model,
        model_size=lambda: 56,
        pinned_memory_size=lambda: 78,
        _vbar_get=lambda: SimpleNamespace(loaded_size=lambda: 22),
    )

    receipt = _comfyui_residency_receipt(patcher)

    assert receipt == {
        "active": True,
        "loaded_bytes": 34,
        "model_loaded_weight_bytes": 12,
        "model_size_bytes": 56,
        "pinned_host_bytes": 78,
        "subsets": {
            "weights": {
                "host_buffer_bytes": 1,
                "pinned_host_bytes": 2,
                "stack_entries": 1,
                "stack_split": 0,
            },
            "weights-loaded": {
                "host_buffer_bytes": 2,
                "pinned_host_bytes": 4,
                "stack_entries": 2,
                "stack_split": 1,
            },
            "patches": {
                "host_buffer_bytes": 3,
                "pinned_host_bytes": 6,
                "stack_entries": 3,
                "stack_split": 2,
            },
            "patches-loaded": {
                "host_buffer_bytes": 4,
                "pinned_host_bytes": 8,
                "stack_entries": 4,
                "stack_split": 3,
            },
        },
        "vbar_loaded_bytes": 22,
    }


def test_stateless_residency_receipt_does_not_inspect_the_codec() -> None:
    def refuse_recording(_value: object) -> object:
        raise AssertionError("stateless codec was inspected for weight residency")

    assert _residency_receipt(object(), refuse_recording, stateless=True) == {"stateless": True}


def test_clear_cublas_workspaces_is_capability_gated() -> None:
    calls: list[str] = []
    _clear_cublas_workspaces(
        SimpleNamespace(
            _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("clear"))
        )
    )
    _clear_cublas_workspaces(SimpleNamespace(_C=SimpleNamespace()))

    assert calls == ["clear"]


def test_process_heap_trim_is_capability_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(chroma_e2e.psutil, "heap_trim", lambda: calls.append("trim"))
    _trim_process_heap()
    monkeypatch.setattr(chroma_e2e.psutil, "heap_trim", None)
    _trim_process_heap()

    assert calls == ["trim"]


def test_gpu_target_requires_a_consistent_physical_identity() -> None:
    target = {
        "expected_cuda_visible_devices": "2",
        "gpu_index": 2,
        "gpu_uuid": "GPU-healthy",
    }
    assert _gpu_target(target) == (2, "GPU-healthy")

    for update, message in (
        ({"gpu_index": -1}, "non-negative integer"),
        ({"gpu_uuid": "not-a-gpu"}, "NVIDIA GPU UUID"),
        ({"expected_cuda_visible_devices": "3"}, "visibility"),
    ):
        with pytest.raises(GateError, match=message):
            _gpu_target({**target, **update})


def test_physical_gpu_identity_is_verified_before_use(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="GPU-healthy, 595.84\n")

    monkeypatch.setattr(chroma_e2e.subprocess, "run", run)

    assert _verify_physical_gpu(2, "GPU-healthy") == "595.84"
    assert calls[0][1:3] == ["-i", "2"]
    with pytest.raises(GateError, match="physical GPU 2 changed"):
        _verify_physical_gpu(2, "GPU-other")


def test_process_vram_measurement_fails_closed_when_pid_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    result = SimpleNamespace(stdout="123, 456\n")
    monkeypatch.setattr(chroma_e2e.subprocess, "run", lambda *_args, **_kwargs: result)

    assert _process_vram_bytes(123) == 456 * 1024 * 1024
    with pytest.raises(GateError, match="process 789 is missing from physical GPU 2"):
        _process_vram_bytes(789)


def _receipt(tmp_path: Path, name: str, value: float) -> dict[str, Any]:
    array = np.full((1, 2), value, dtype=np.float32)
    path = tmp_path / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    return {
        "dtype": "float32",
        "path": str(path),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "shape": [1, 2],
    }


def _metrics(value: float, seconds: float) -> dict[str, Any]:
    return {
        "peak_process_rss_bytes": value,
        "peak_process_vram_bytes": value,
        "seconds": seconds,
        "torch_peak_allocated_bytes": value,
        "torch_peak_reserved_bytes": value,
    }


def _record(
    tmp_path: Path, engine: str, index: int, *, value: float = 1.0, slower: bool = False
) -> dict[str, Any]:
    from dinkster_protocol import ATTENTION_ROLES

    stem = f"{index}-{engine}"
    cold_image = _receipt(tmp_path, f"{stem}-cold-image", value)
    cold_latent = _receipt(tmp_path, f"{stem}-cold-latent", value)
    resource = 110.0 if engine == "comfyui" else 100.0
    seconds = 1.0 if engine == "comfyui" else (1.1 if slower else 0.9)
    measured = []
    for run in range(MEASURED_RUNS):
        measured.append(
            {
                **_metrics(resource, seconds),
                "image": _receipt(tmp_path, f"{stem}-warm-{run}-image", value),
                "index": run,
                "latent": _receipt(tmp_path, f"{stem}-warm-{run}-latent", value),
            }
        )
    if engine == "comfyui":
        fallback = {
            "attention_backend": "sdpa",
            "lowvram_patch_count": 0,
            "mechanism": "aimdo",
            "model_loaded_bytes": 1,
            "model_patcher": "ModelPatcherDynamic",
            "oom_count": 0,
        }
    else:
        fallback = {
            "attention_backends": ["sdpa"],
            "logical_loaded_bytes": 1,
            "logical_offloaded_bytes": 0,
            "oom_count": 0,
            "route": {
                "dynamic_components": ["diffusion"],
                "fallback_components": [],
                "fallback_reason": None,
                "mechanism": "aimdo",
                "requested": "auto",
                "resident_components": [],
            },
        }
    runtime_settings = {
        "aimdo_device_extra_vram_headroom_bytes": 0,
        "aimdo_enabled": True,
        "aimdo_nvml_pressure": False,
        "aimdo_simple_vram_headroom_bytes": 256 * 1024 * 1024,
        "attention_backend": "sdpa",
        "diffusion_compute_dtype": "bfloat16",
        "text_compute_dtype": "float32",
        "vae_compute_dtype": "bfloat16",
        "weight_dtype": "default",
    }
    workload = {
        "scheduler": {"steps": 1},
        "seed": 12345,
        "template_commit": chroma_e2e.TEMPLATE_COMMIT,
        "template_path": TEMPLATES["chroma"],
        "template_sha256": "0" * 64,
        "variant": "chroma",
    }
    source_commit = COMFYUI_COMMIT if engine == "comfyui" else "candidate"
    record = {
        "arrays": {
            "cold_image": cold_image,
            "cold_latent": cold_latent,
            "conditioning": {
                "negative": _receipt(tmp_path, f"{stem}-negative", value),
                "positive": _receipt(tmp_path, f"{stem}-positive", value),
            },
            "initial_noise": _receipt(tmp_path, f"{stem}-noise", value),
            "intermediates": {
                "0": {
                    "current": _receipt(tmp_path, f"{stem}-current", value),
                    "denoised": _receipt(tmp_path, f"{stem}-denoised", value),
                }
            },
            "sigmas": _receipt(tmp_path, f"{stem}-sigmas", value),
        },
        "cold": _metrics(resource, seconds * 2),
        "device": {
            "dinkster_aimdo": "0.4.15",
            "dinkster_kitchen": "0.2.31",
            "cuda": "13.0",
            "driver": "595.84",
            "gpu_index": 3,
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "gpu_total_memory": 1,
            "gpu_uuid": "GPU-test",
            "python": "3.12.11",
            "torch": "2.13.0+cu130",
        },
        "engine": engine,
        "fallback": fallback,
        "measured": measured,
        "post_cleanup": {
            "rss_bytes": resource,
            "torch_allocated_bytes": resource,
            "torch_reserved_bytes": resource,
            "vram_bytes": resource,
        },
        "runtime_settings": runtime_settings,
        "schema": 1,
        "source_commit": source_commit,
        "status": "success",
        "variant": "chroma",
        "workload": workload,
    }
    identities = {
        "chroma": "dinkster.chroma.diffusion",
        "pixel_vae": "dinkster.chroma.pixel_vae",
        "t5xxl": "dinkster.chroma.t5xxl",
        "vae": "dinkster.chroma.vae",
    }
    if engine == "dinkster":
        record["attention_route_token"] = {
            "adapterContractRevision": "dinkster.attention-kernel.v1",
            "deviceKind": "cuda",
            "deviceSm": 120,
            "providerVersions": [["torch", "2.13.0+cu130"]],
            "requestedPolicy": "sdpa",
            "routes": [
                {"fallback": None, "primary": "sdpa", "role": role} for role in ATTENTION_ROLES
            ],
            "sdpaTorchRuntime": "2.13.0",
            "version": 1,
        }
        record["dinkster_identities"] = {
            name: f"{identity}.sdpa" for name, identity in identities.items()
        }
    artifacts = {
        name: {
            **chroma_e2e.ARTIFACTS[name],
            "blake3": f"blake3:{'0' * 64}",
            "path": str(tmp_path / str(chroma_e2e.ARTIFACTS[name]["path"])),
            "verification": {"name": name},
        }
        for name in ("chroma", "t5xxl", "vae")
    }
    preflight = {
        "artifacts": artifacts,
        "dinkster_commit": "candidate",
        "dinkster_unrouted_identities": identities,
        "engine": engine,
        "expected_cuda_visible_devices": "3",
        "gpu_index": 3,
        "gpu_uuid": "GPU-test",
        "harness_sha256": hashlib.sha256(Path(chroma_e2e.__file__).read_bytes()).hexdigest(),
        "runtime_settings": runtime_settings,
        "schema": 1,
        "source_commit": source_commit,
        "thread_id": chroma_e2e.THREAD_ID,
        "variant": "chroma",
        "workload": workload,
    }
    preflight_path = tmp_path / f"{stem}-preflight.json"
    preflight_path.write_text(json.dumps(preflight, sort_keys=True))
    record["preflight"] = str(preflight_path)
    record["preflight_sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    return record


def _records(
    tmp_path: Path, *, dinkster_value: float = 1.0, slower: bool = False
) -> list[dict[str, Any]]:
    return [
        _record(
            tmp_path,
            engine,
            index,
            value=dinkster_value if engine == "dinkster" else 1.0,
            slower=slower,
        )
        for index, engine in enumerate(PROCESS_ORDER)
    ]


def test_verdict_covers_correctness_cold_warm_memory_and_fallback(tmp_path: Path) -> None:
    verdict = build_verdict(_records(tmp_path), "candidate")

    assert verdict["overall_pass"] is True
    assert verdict["correctness"]["pass"] is True
    assert verdict["performance"]["pass"] is True
    assert set(verdict["performance"]["metrics"]) == {"cold", "warm"}
    assert verdict["performance"]["metrics"]["warm"]["dinkster"]["p95"] == 0.9
    assert verdict["performance"]["metrics"]["warm"]["dinkster"]["maximum"] == 0.9
    assert verdict["memory"]["pass"] is True
    assert set(verdict["memory"]["metrics"]) == {"cold", "warm", "post_cleanup"}
    assert verdict["fallback"]["pass"] is True


def test_verdict_reports_cross_engine_drift_without_tolerance(tmp_path: Path) -> None:
    verdict = build_verdict(_records(tmp_path, dinkster_value=2.0), "candidate")

    assert verdict["correctness"]["pass"] is False
    assert verdict["overall_pass"] is False
    assert verdict["correctness"]["comparisons"]["cold_image"]["max_abs"] == 1.0


def test_verdict_rejects_nondeterminism_and_sidecar_tampering(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records[1]["measured"][0]["image"] = _receipt(tmp_path, "changed-warm", 2.0)
    with pytest.raises(GateError, match="changed between cold and warm"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    Path(records[0]["arrays"]["cold_image"]["path"]).write_bytes(b"tampered")
    with pytest.raises((GateError, ValueError)):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    records[0]["arrays"]["intermediates"] = {}
    with pytest.raises(GateError, match="sampling states"):
        build_verdict(records, "candidate")


def test_verdict_blocks_speed_memory_and_residency_regressions(tmp_path: Path) -> None:
    slower = build_verdict(_records(tmp_path, slower=True), "candidate")
    assert slower["performance"]["pass"] is False
    assert slower["overall_pass"] is False

    records = _records(tmp_path)
    for record in records:
        if record["engine"] == "dinkster":
            record["cold"]["peak_process_vram_bytes"] = 120
            for run in record["measured"]:
                run["peak_process_vram_bytes"] = 120
    memory = build_verdict(records, "candidate")
    assert memory["memory"]["pass"] is False
    assert memory["overall_pass"] is False

    records = _records(tmp_path)
    for record in records:
        if record["engine"] == "dinkster":
            record["post_cleanup"]["vram_bytes"] = 120
    memory = build_verdict(records, "candidate")
    assert memory["memory"]["pass"] is False
    assert memory["overall_pass"] is False

    records = _records(tmp_path)
    records[1]["fallback"]["route"]["mechanism"] = "eager"
    fallback = build_verdict(records, "candidate")
    assert fallback["fallback"]["pass"] is False
    assert fallback["overall_pass"] is False


def test_verdict_reports_allocator_peaks_without_comparing_residency_backends(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path)
    for record in records:
        if record["engine"] != "dinkster":
            continue
        record["cold"]["torch_peak_allocated_bytes"] = 1_000
        record["cold"]["torch_peak_reserved_bytes"] = 1_000
        for run in record["measured"]:
            run["torch_peak_allocated_bytes"] = 1_000
            run["torch_peak_reserved_bytes"] = 1_000

    verdict = build_verdict(records, "candidate")

    assert verdict["memory"]["pass"] is True
    assert (
        verdict["memory"]["metrics"]["warm"]["torch_peak_allocated_bytes"]["dinkster"]["maximum"]
        == 1_000
    )


def test_verdict_rejects_process_source_and_device_drift(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records[0], records[1] = records[1], records[0]
    with pytest.raises(GateError, match="record order"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    records[2]["source_commit"] = "wrong"
    with pytest.raises(GateError, match="source_commit"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    records[3]["device"]["dinkster_aimdo"] = "different"
    with pytest.raises(GateError, match="runtime or device"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    records[1]["runtime_settings"]["aimdo_simple_vram_headroom_bytes"] = 0
    with pytest.raises(GateError, match="runtime_settings"):
        build_verdict(records, "candidate")


def test_verdict_binds_preflight_artifacts_and_identities(tmp_path: Path) -> None:
    records = _records(tmp_path)
    Path(records[0]["preflight"]).write_text("{}")
    with pytest.raises(GateError, match="preflight digest differs"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    preflight_path = Path(records[0]["preflight"])
    preflight = json.loads(preflight_path.read_text())
    preflight["artifacts"]["chroma"]["sha256"] = "forged"
    preflight_path.write_text(json.dumps(preflight, sort_keys=True))
    records[0]["preflight_sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    with pytest.raises(GateError, match="pinned artifact"):
        build_verdict(records, "candidate")

    records = _records(tmp_path)
    records[2]["dinkster_identities"]["chroma"] = "different"
    with pytest.raises(GateError, match="dinkster_identities changed"):
        build_verdict(records, "candidate")

    record = _record(tmp_path, "comfyui", 4)
    preflight_path = Path(record["preflight"])
    preflight = json.loads(preflight_path.read_text())
    workload = {**preflight["workload"], "variant": "radiance"}
    workload["template_path"] = TEMPLATES["radiance"]
    radiance = {
        **chroma_e2e.ARTIFACTS["radiance"],
        "blake3": f"blake3:{'0' * 64}",
        "path": str(tmp_path / str(chroma_e2e.ARTIFACTS["radiance"]["path"])),
        "verification": {"name": "radiance"},
    }
    preflight.update(
        artifacts={"radiance": radiance, "t5xxl": preflight["artifacts"]["t5xxl"]},
        dinkster_unrouted_identities={
            "pixel_vae": "dinkster.radiance.pixel_vae",
            "radiance": "dinkster.radiance.diffusion",
            "t5xxl": "dinkster.radiance.t5xxl",
        },
        variant="radiance",
        workload=workload,
    )
    preflight_path.write_text(json.dumps(preflight, sort_keys=True))
    record.update(
        preflight_sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        variant="radiance",
        workload=workload,
    )
    assert chroma_e2e._validated_preflight(record, "candidate")["variant"] == "radiance"


def test_verdict_rejects_incomplete_production_residency(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records[0]["fallback"]["model_loaded_bytes"] = -1
    assert build_verdict(records, "candidate")["fallback"]["pass"] is False

    records = _records(tmp_path)
    records[1]["fallback"]["logical_loaded_bytes"] = 0
    records[1]["fallback"]["logical_offloaded_bytes"] = 0
    assert build_verdict(records, "candidate")["fallback"]["pass"] is False

    records = _records(tmp_path)
    records[1]["fallback"]["route"]["fallback_components"] = ["diffusion"]
    assert build_verdict(records, "candidate")["fallback"]["pass"] is False


def test_compare_persists_a_failed_gate_verdict(tmp_path: Path) -> None:
    inputs = []
    for index, record in enumerate(_records(tmp_path, slower=True)):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        inputs.append(path)
    output = tmp_path / "verdict.json"

    result = run_compare(
        argparse.Namespace(inputs=inputs, output=output, dinkster_commit="candidate")
    )

    assert result == 1
    assert json.loads(output.read_text())["overall_pass"] is False
