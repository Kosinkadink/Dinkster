"""Matched official-artifact Chroma and Chroma Radiance E2E comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from dinkster_inference import ChromaComponentRole
    from dinkster_protocol import AttentionPolicy, AttentionRouteToken

COMFYUI_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
TEMPLATE_COMMIT = "d3b4a9e89573162b005961865164c18c8ae2206b"
THREAD_ID = "T-01a05f1e-0cc0-72e6-b18b-83eaa33a38ab"
PROCESS_ORDER = ("comfyui", "dinkster", "dinkster", "comfyui")
MEASURED_RUNS = 3
POLL_INTERVAL_SECONDS = 0.1

ARTIFACTS = {
    "chroma": {
        "path": "diffusion_models/Chroma1-HD-fp8mixed.safetensors",
        "revision": "47f45ad2f72b2bccaa808418aeedca8c49d67974",
        "sha256": "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575",
        "size_bytes": 9_193_379_316,
        "url": "https://huggingface.co/Comfy-Org/Chroma1-HD_repackaged/resolve/47f45ad2f72b2bccaa808418aeedca8c49d67974/split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors",
    },
    "radiance": {
        "path": "diffusion_models/chroma-radiance-x0.safetensors",
        "revision": "c030c66a6aa7ff42dfe5f7c1a1e9cdc2652701d1",
        "sha256": "086e11d033ccd7470e67fa80e00a29902df2868cc84e16df0b48853be3a8672a",
        "size_bytes": 19_012_346_326,
        "url": "https://huggingface.co/Comfy-Org/Chroma1-Radiance_Repackaged/resolve/c030c66a6aa7ff42dfe5f7c1a1e9cdc2652701d1/split_files/diffusion_models/chroma-radiance-x0.safetensors",
    },
    "t5xxl": {
        "path": "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "revision": "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5",
        "sha256": "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a",
        "size_bytes": 5_157_348_688,
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn_scaled.safetensors",
    },
    "vae": {
        "path": "vae/ae.safetensors",
        "revision": "5b072540ef86570fecb8249c505f23d5bdeb88cd",
        "sha256": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "size_bytes": 335_304_388,
        "url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/5b072540ef86570fecb8249c505f23d5bdeb88cd/split_files/vae/ae.safetensors",
    },
}

TEMPLATES = {
    "chroma": "templates/image_chroma_text_to_image.json",
    "radiance": "templates/image_chroma1_radiance_text_to_image.json",
}


class GateError(RuntimeError):
    """The evidence or execution does not satisfy the comparison contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_source(root: Path, expected_commit: str, name: str) -> None:
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise GateError(f"{name} must be at {expected_commit}, got {actual}")
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise GateError(f"{name} source must be clean: {root}\n{dirty}")


def _one(nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any]:
    matches = [node for node in nodes if node.get("type") == node_type]
    if len(matches) != 1:
        raise GateError(f"workflow requires exactly one {node_type}, got {len(matches)}")
    return matches[0]


def _link_origin(graph: dict[str, Any], link_id: int) -> int:
    for link in graph["links"]:
        if isinstance(link, list):
            if link[0] == link_id:
                return int(link[1])
        elif link["id"] == link_id:
            return int(link["origin_id"])
    raise GateError(f"workflow link {link_id} is missing")


def _linked_node(
    graph: dict[str, Any], nodes_by_id: dict[int, dict[str, Any]], node: dict[str, Any], name: str
) -> dict[str, Any]:
    entry = next((value for value in node["inputs"] if value["name"] == name), None)
    if entry is None or entry.get("link") is None:
        raise GateError(f"workflow input {node['type']}.{name} is not linked")
    return nodes_by_id[_link_origin(graph, int(entry["link"]))]


def extract_workload(template_root: Path, variant: str) -> dict[str, Any]:
    """Derive the production workload from the pinned official workflow graph."""
    template_path = template_root / TEMPLATES[variant]
    template_bytes = template_path.read_bytes()
    document = json.loads(template_bytes)
    if variant == "radiance":
        subgraphs = document.get("definitions", {}).get("subgraphs", [])
        if len(subgraphs) != 1:
            raise GateError("Radiance workflow requires exactly one subgraph")
        graph = subgraphs[0]
    else:
        graph = document
    nodes = graph["nodes"]
    nodes_by_id = {int(node["id"]): node for node in nodes}
    guider = _one(nodes, "CFGGuider")
    positive = _linked_node(graph, nodes_by_id, guider, "positive")
    negative = _linked_node(graph, nodes_by_id, guider, "negative")
    if positive["type"] != "CLIPTextEncode" or negative["type"] != "CLIPTextEncode":
        raise GateError("CFG conditioning must come directly from CLIPTextEncode")
    latent_type = (
        "EmptyChromaRadianceLatentImage" if variant == "radiance" else "EmptySD3LatentImage"
    )
    latent = _one(nodes, latent_type)
    scheduler_type = "BetaSamplingScheduler" if variant == "radiance" else "BasicScheduler"
    scheduler = _one(nodes, scheduler_type)
    options = _one(nodes, "ChromaRadianceOptions") if variant == "radiance" else None
    vae_name = _one(nodes, "VAELoader")["widgets_values"][0]
    expected_vae = "pixel_space" if variant == "radiance" else "ae.safetensors"
    if vae_name != expected_vae:
        raise GateError(f"{variant} workflow VAE changed from {expected_vae!r} to {vae_name!r}")
    scheduler_values = scheduler["widgets_values"]
    scheduler_receipt: dict[str, Any]
    if variant == "radiance":
        scheduler_receipt = {
            "alpha": float(scheduler_values[1]),
            "beta": float(scheduler_values[2]),
            "kind": "beta",
            "steps": int(scheduler_values[0]),
        }
    else:
        scheduler_receipt = {
            "denoise": float(scheduler_values[2]),
            "kind": str(scheduler_values[0]),
            "steps": int(scheduler_values[1]),
        }
    option_values = None
    if options is not None:
        values = options["widgets_values"]
        option_values = {
            "end_sigma": float(values[2]),
            "force_sequential_txt_ids": bool(values[4]) if len(values) > 4 else False,
            "nerf_tile_size": int(values[3]),
            "preserve_wrapper": bool(values[0]),
            "start_sigma": float(values[1]),
        }
    return {
        "cfg": float(guider["widgets_values"][0]),
        "clip_name": str(_one(nodes, "CLIPLoader")["widgets_values"][0]),
        "height": int(latent["widgets_values"][1]),
        "model_name": str(_one(nodes, "UNETLoader")["widgets_values"][0]),
        "model_options": option_values,
        "negative_prompt": str(negative["widgets_values"][0]),
        "positive_prompt": str(positive["widgets_values"][0]),
        "sampler": str(_one(nodes, "KSamplerSelect")["widgets_values"][0]),
        "sampling_shift": float(_one(nodes, "ModelSamplingAuraFlow")["widgets_values"][0]),
        "scheduler": scheduler_receipt,
        "seed": int(_one(nodes, "RandomNoise")["widgets_values"][0]),
        "template_commit": TEMPLATE_COMMIT,
        "template_path": TEMPLATES[variant],
        "template_sha256": _sha256_bytes(template_bytes),
        "tokenizer": {"min_length": 0, "min_padding": 0},
        "vae_name": vae_name,
        "variant": variant,
        "warm_measured_runs": MEASURED_RUNS,
        "warmup_runs_discarded": 1,
        "width": int(latent["widgets_values"][0]),
    }


def _artifact_receipts(artifact_root: Path, variant: str) -> dict[str, dict[str, Any]]:
    from dinkster_assets import new_hasher
    from dinkster_assets.integrity import verification_record

    names = [variant, "t5xxl"] + ([] if variant == "radiance" else ["vae"])
    receipts: dict[str, dict[str, Any]] = {}
    for name in names:
        expected = ARTIFACTS[name]
        path = artifact_root / str(expected["path"])
        sha256_hasher = hashlib.sha256()
        blake3_hasher = new_hasher()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha256_hasher.update(chunk)
                blake3_hasher.update(chunk)
            after = os.fstat(handle.fileno())
        before_fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_fingerprint != after_fingerprint:
            raise GateError(f"{name} artifact changed while its digests were computed")
        size = after.st_size
        sha256 = sha256_hasher.hexdigest()
        if size != expected["size_bytes"] or sha256 != expected["sha256"]:
            raise GateError(f"{name} artifact differs: got {size} bytes and sha256:{sha256}")
        blake3 = f"blake3:{blake3_hasher.hexdigest()}"
        verification = verification_record(blake3, after)
        if verification is None:
            raise GateError(f"cannot bind {name} artifact verification to its hashed file")
        receipts[name] = {
            **expected,
            "blake3": blake3,
            "path": str(path.resolve()),
            "verification": verification.to_json(),
        }
    return receipts


def _verify_artifact_receipts(receipts: dict[str, dict[str, Any]]) -> None:
    for name, receipt in receipts.items():
        path = Path(receipt["path"])
        size = path.stat().st_size
        sha256 = _sha256_file(path)
        if size != receipt["size_bytes"] or sha256 != receipt["sha256"]:
            raise GateError(
                f"{name} artifact changed after preflight: got {size} bytes and sha256:{sha256}"
            )


def _dinkster_identities(
    dinkster_root: Path,
    artifacts: dict[str, dict[str, Any]],
    variant: str,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    require_torch_free: bool = True,
) -> dict[str, str]:
    for source in sorted((dinkster_root / "packages").glob("*/src")):
        sys.path.insert(0, str(source))
    from dinkster_inference import (
        build_runtime_identity_from_facts,
        chroma_component_family_id,
        chroma_component_runtime_identity,
        default_diffusion_dtype,
        default_text_dtype,
        default_vae_dtype,
        load_safetensors_header,
        plan_chroma_split_component,
    )

    roles: dict[str, ChromaComponentRole] = {
        variant: "diffusion",
        "t5xxl": "t5xxl",
        "vae": "vae",
    }
    identities = {}
    for name, receipt in artifacts.items():
        path = Path(receipt["path"])
        blake3 = str(receipt["blake3"])
        source = load_safetensors_header(
            path,
            asset_digest=blake3,
            asset_size=int(receipt["size_bytes"]),
        )
        role = roles[name]
        planned = plan_chroma_split_component(source, role=role, path=path)
        family_id = chroma_component_family_id(planned)
        compute_dtype = {
            "diffusion": default_diffusion_dtype(family_id),
            "t5xxl": default_text_dtype(family_id),
            "vae": default_vae_dtype(family_id),
        }[role]
        identities[name] = chroma_component_runtime_identity(
            planned,
            role,
            compute_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
    identities["pixel_vae"] = build_runtime_identity_from_facts(
        "dinkster.chroma_radiance",
        ("family=dinkster.chroma_radiance", "component=pixel_space"),
        diffusion_dtype="unloaded",
        text_dtype="unloaded",
        vae_dtype=default_vae_dtype("dinkster.chroma_radiance").name,
        fp8_matmul=False,
    )
    if require_torch_free and "torch" in sys.modules:
        raise GateError("torch was imported during offline identity planning")
    return identities


def _require_callables(module: Any, owners: dict[str, tuple[str, ...]]) -> None:
    for owner_name, method_names in owners.items():
        owner = getattr(module, owner_name, None)
        if owner is None:
            raise GateError(f"launch boundary requires {module.__name__}.{owner_name}")
        for method_name in method_names:
            if not callable(getattr(owner, method_name, None)):
                raise GateError(
                    f"launch boundary requires {module.__name__}.{owner_name}.{method_name}"
                )


def _validate_launch_boundary(source_root: Path, engine: str, variant: str) -> None:
    if engine == "comfyui":
        sys.path.insert(0, str(source_root))
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy_extras.nodes_chroma_radiance as radiance_nodes
        import comfy_extras.nodes_cond as cond_nodes
        import comfy_extras.nodes_custom_sampler as sampling_nodes
        import comfy_extras.nodes_model_advanced as model_nodes
        import comfy_extras.nodes_sd3 as sd3_nodes
        import nodes

        _require_callables(
            nodes,
            {
                "CLIPLoader": ("load_clip",),
                "CLIPTextEncode": ("encode",),
                "UNETLoader": ("load_unet",),
                "VAEDecode": ("decode",),
                "VAELoader": ("load_vae",),
            },
        )
        _require_callables(cond_nodes, {"T5TokenizerOptions": ("execute",)})
        _require_callables(
            sampling_nodes,
            {
                "BasicScheduler": ("execute",),
                "BetaSamplingScheduler": ("execute",),
                "CFGGuider": ("execute",),
                "KSamplerSelect": ("execute",),
                "RandomNoise": ("execute",),
                "SamplerCustomAdvanced": ("execute",),
            },
        )
        _require_callables(model_nodes, {"ModelSamplingAuraFlow": ("patch_aura",)})
        if variant == "radiance":
            _require_callables(
                radiance_nodes,
                {
                    "ChromaRadianceOptions": ("execute",),
                    "EmptyChromaRadianceLatentImage": ("execute",),
                },
            )
        else:
            _require_callables(sd3_nodes, {"EmptySD3LatentImage": ("execute",)})
    else:
        for source in sorted((source_root / "packages").glob("*/src")):
            sys.path.insert(0, str(source))
        sys.path.insert(0, str(source_root / "src"))
        import dinkster_compat_comfy.native_arm as arm
        from dinkster_workers.execution import use_execution_context

        if not callable(use_execution_context):
            raise GateError(
                "launch boundary requires dinkster_workers.execution.use_execution_context"
            )

        owners = {
            "GenerationBasicScheduler": ("execute",),
            "GenerationBetaSamplingScheduler": ("execute",),
            "GenerationCFGGuider": ("execute",),
            "GenerationClipTextEncode": ("execute",),
            "GenerationKSamplerSelect": ("execute",),
            "GenerationRandomNoise": ("execute",),
            "GenerationSamplerCustomAdvanced": ("execute",),
            "GenerationT5TokenizerOptions": ("execute",),
            "GenerationVAEDecode": ("execute",),
            "NativeLoadClip": ("execute",),
            "NativeLoadDiffusionModel": ("execute",),
            "NativeLoadVae": ("execute",),
        }
        owners[
            "GenerationEmptyChromaRadianceLatentImage"
            if variant == "radiance"
            else "GenerationEmptySD3LatentImage"
        ] = ("execute",)
        owners[
            "GenerationChromaRadianceOptions"
            if variant == "radiance"
            else "GenerationChromaModelSampling"
        ] = ("execute",)
        if variant == "radiance":
            owners["GenerationChromaModelSampling"] = ("execute",)
        _require_callables(arm, owners)
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise GateError("CUDA initialized during offline launch-boundary validation")


def run_preflight(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise GateError("offline preflight requires CUDA_VISIBLE_DEVICES empty or -1")
    gpu_index, gpu_uuid = _gpu_target(
        {
            "expected_cuda_visible_devices": str(args.gpu_index),
            "gpu_index": args.gpu_index,
            "gpu_uuid": args.gpu_uuid,
        }
    )
    source_root = args.source_root.resolve()
    dinkster_root = args.dinkster_root.resolve()
    template_root = args.template_root.resolve()
    expected_source = COMFYUI_COMMIT if args.engine == "comfyui" else args.dinkster_commit
    _require_source(source_root, expected_source, args.engine)
    _require_source(dinkster_root, args.dinkster_commit, "Dinkster")
    _require_source(template_root, TEMPLATE_COMMIT, "workflow_templates")
    for source in sorted((dinkster_root / "packages").glob("*/src")):
        sys.path.insert(0, str(source))
    workload = extract_workload(template_root, args.variant)
    artifacts = _artifact_receipts(args.artifact_root.resolve(), args.variant)
    identities = _dinkster_identities(dinkster_root, artifacts, args.variant)
    from dinkster_inference import (
        default_diffusion_dtype,
        default_text_dtype,
        default_vae_dtype,
    )
    from dinkster_memory import DEFAULT_ACCELERATOR_HEADROOM_BYTES

    family_id = "dinkster.chroma_radiance" if args.variant == "radiance" else "dinkster.chroma"
    runtime_settings = {
        "aimdo_device_extra_vram_headroom_bytes": 0,
        "aimdo_enabled": True,
        "aimdo_nvml_pressure": False,
        "aimdo_simple_vram_headroom_bytes": DEFAULT_ACCELERATOR_HEADROOM_BYTES,
        "attention_backend": "sdpa",
        "diffusion_compute_dtype": default_diffusion_dtype(family_id).name,
        "text_compute_dtype": default_text_dtype(family_id).name,
        "vae_compute_dtype": default_vae_dtype(family_id).name,
        "weight_dtype": "default",
    }
    _validate_launch_boundary(source_root, args.engine, args.variant)
    harness_path = Path(__file__).resolve()
    command = [
        str(args.run_python.absolute()),
        str(harness_path),
        "run",
        "--engine",
        args.engine,
        "--variant",
        args.variant,
        "--source-root",
        str(source_root),
        "--preflight",
        str(args.output.resolve()),
        "--output",
        str(args.run_output.resolve()),
    ]
    record = {
        "artifacts": artifacts,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dinkster_commit": args.dinkster_commit,
        "dinkster_unrouted_identities": identities,
        "engine": args.engine,
        "expected_cuda_visible_devices": str(gpu_index),
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "harness_sha256": _sha256_file(harness_path),
        "launch_command": command,
        "runtime_settings": runtime_settings,
        "schema": 1,
        "source_commit": expected_source,
        "source_root": str(source_root),
        "thread_id": THREAD_ID,
        "variant": args.variant,
        "workload": workload,
    }
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise GateError("CUDA initialized during offline preflight")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


def _gpu_target(preflight: dict[str, Any]) -> tuple[int, str]:
    index = preflight.get("gpu_index")
    uuid = preflight.get("gpu_uuid")
    if type(index) is not int or index < 0:
        raise GateError("preflight GPU index must be a non-negative integer")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise GateError("preflight GPU UUID must be an NVIDIA GPU UUID")
    if preflight.get("expected_cuda_visible_devices") != str(index):
        raise GateError("preflight CUDA visibility does not match its GPU index")
    return index, uuid


def _verify_physical_gpu(gpu_index: int, expected_uuid: str) -> str:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [value.strip() for value in query.split(",")]
    if len(values) != 2:
        raise GateError(f"physical GPU {gpu_index} identity is malformed: {query!r}")
    uuid, driver = values
    if uuid != expected_uuid:
        raise GateError(f"physical GPU {gpu_index} changed: expected {expected_uuid}, got {uuid}")
    return driver


def _acquire_gpu_claim(gpu_index: int, purpose: str) -> Any:
    fcntl = import_module("fcntl")
    path = Path.home() / f"gpu-claims/gpu{gpu_index}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.seek(0)
    handle.truncate()
    handle.write(
        f"thread={THREAD_ID}\npurpose={purpose}\n"
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _process_vram_bytes(pid: int) -> int:
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_index is None or not gpu_index.isdecimal():
        raise GateError("CUDA_VISIBLE_DEVICES must contain one physical GPU index")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_index,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        pid_text, _, memory_text = line.partition(",")
        if pid_text.strip() == str(pid):
            return int(memory_text.strip()) * 1024 * 1024
    raise GateError(f"process {pid} is missing from physical GPU {gpu_index}")


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise GateError("process RSS is unavailable")


def _trim_process_heap() -> None:
    heap_trim = getattr(psutil, "heap_trim", None)
    if heap_trim is not None:
        heap_trim()


def _active_cuda_allocations(torch: Any) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int, str], int] = {}
    for segment in torch.cuda.memory_snapshot():
        for block in segment["blocks"]:
            state = str(block["state"])
            if not state.startswith("active_"):
                continue
            key = (int(block["size"]), int(block.get("requested_size", 0)), state)
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "count": count,
            "requested_bytes": requested,
            "size_bytes": size,
            "state": state,
        }
        for (size, requested, state), count in sorted(counts.items(), reverse=True)
    ]


def _dinkster_residency_receipt(handle: Any) -> list[dict[str, Any]]:
    receipts = []
    for mechanism in handle.mechanisms:
        vbar = mechanism._vbar
        stream_state = mechanism._stream_state
        receipts.append(
            {
                "arena_bytes": (
                    0
                    if stream_state is None
                    else sum(
                        mechanism._backend.cast_arena_size(arena)
                        for arena in stream_state.arenas.values()
                    )
                ),
                "eager_loaded_bytes": mechanism._eager_loaded_bytes(),
                "loaded_bytes": mechanism.loaded_bytes(),
                "offloaded_bytes": mechanism.offloaded_bytes(),
                "pinned_host_bytes": sum(pin.tensor.nbytes for pin in mechanism._pins.values()),
                "promoted_units": sorted(mechanism._promoted_units),
                "reservation_bytes": mechanism._reservation_bytes,
                "total_bytes": mechanism.total_bytes(),
                "vbar_loaded_bytes": (0 if vbar is None else mechanism._backend.loaded_size(vbar)),
                "working_set_reservation_bytes": mechanism.working_set_reservation_bytes(),
            }
        )
    return receipts


def _comfyui_residency_receipt(patcher: Any) -> dict[str, Any]:
    pin_state = patcher.model.dynamic_pins[patcher.load_device]
    vbar = patcher._vbar_get()
    subsets = {}
    for name in ("weights", "weights-loaded", "patches", "patches-loaded"):
        host_buffer, stack, stack_split, pinned_size, *_ = pin_state[name]
        subsets[name] = {
            "host_buffer_bytes": int(host_buffer.size),
            "pinned_host_bytes": int(pinned_size[0]),
            "stack_entries": len(stack),
            "stack_split": int(stack_split[0]),
        }
    return {
        "active": bool(pin_state["active"]),
        "loaded_bytes": int(patcher.loaded_size()),
        "model_loaded_weight_bytes": int(patcher.model.model_loaded_weight_memory),
        "model_size_bytes": int(patcher.model_size()),
        "pinned_host_bytes": int(patcher.pinned_memory_size()),
        "subsets": subsets,
        "vbar_loaded_bytes": 0 if vbar is None else int(vbar.loaded_size()),
    }


def _residency_receipt(value: Any, recorder: Any, *, stateless: bool) -> Any:
    if stateless:
        return {"stateless": True}
    return recorder(value)


class _MemoryMonitor:
    def __init__(self) -> None:
        self.peak_rss = 0
        self.peak_vram = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="chroma-e2e-memory", daemon=True)

    def _sample(self) -> None:
        self.peak_rss = max(self.peak_rss, _rss_bytes())
        self.peak_vram = max(self.peak_vram, _process_vram_bytes(os.getpid()))

    def _run(self) -> None:
        while not self._stop.wait(POLL_INTERVAL_SECONDS):
            self._sample()

    def __enter__(self) -> _MemoryMonitor:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_error: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def _device_receipt(torch: Any, gpu_index: int, gpu_uuid: str) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise GateError(f"CUDA_VISIBLE_DEVICES must be {gpu_index}")
    if torch.cuda.device_count() != 1:
        raise GateError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    driver = _verify_physical_gpu(gpu_index, gpu_uuid)
    properties = torch.cuda.get_device_properties(0)
    return {
        "comfy_aimdo": version("comfy-aimdo"),
        "comfy_kitchen": version("comfy-kitchen"),
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu_index": gpu_index,
        "gpu_name": properties.name,
        "gpu_total_memory": int(properties.total_memory),
        "gpu_uuid": gpu_uuid,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _array(np: Any, tensor: Any) -> Any:
    return tensor.detach().float().cpu().contiguous().numpy().astype(np.float32, copy=False)


def _save_array(np: Any, output_dir: Path, name: str, value: Any) -> dict[str, Any]:
    path = output_dir / f"{name}.npy"
    array = np.asarray(value)
    np.save(path, array, allow_pickle=False)
    return {
        "dtype": str(array.dtype),
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
        "shape": list(array.shape),
    }


def _run_metrics(torch: Any, execute: Any) -> tuple[Any, Any, dict[str, Any]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = {
        "rss_bytes": _rss_bytes(),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "vram_bytes": _process_vram_bytes(os.getpid()),
    }
    with _MemoryMonitor() as monitor:
        started = time.perf_counter_ns()
        latent, image = execute()
        torch.cuda.synchronize()
        seconds = (time.perf_counter_ns() - started) / 1e9
    after = {
        "rss_bytes": _rss_bytes(),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "vram_bytes": _process_vram_bytes(os.getpid()),
    }
    return (
        latent,
        image,
        {
            "after": after,
            "before": before,
            "peak_process_rss_bytes": monitor.peak_rss,
            "peak_process_vram_bytes": monitor.peak_vram,
            "seconds": seconds,
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    )


def _clear_cublas_workspaces(torch: Any) -> None:
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()


def _profile_pipeline(torch: Any, sample: Any, finish: Any) -> tuple[Any, Any, dict[str, float]]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    sampled = sample()
    torch.cuda.synchronize()
    sampled_at = time.perf_counter_ns()
    latent, image = finish(sampled)
    torch.cuda.synchronize()
    finished = time.perf_counter_ns()
    return (
        latent,
        image,
        {
            "decode_seconds": (finished - sampled_at) / 1e9,
            "sampling_seconds": (sampled_at - started) / 1e9,
        },
    )


def _selected_steps(total: int) -> set[int]:
    return {0, total // 2, total - 1}


def _expected_step_names(workload: dict[str, Any]) -> set[str]:
    return {str(step) for step in _selected_steps(int(workload["scheduler"]["steps"]))}


def _bootstrap_comfyui_aimdo(simple_vram_headroom: int) -> Any:
    import comfy_aimdo.control as control

    try:
        initialized = control.init(
            simple_vram_headroom=simple_vram_headroom,
            nvml_pressure=False,
        )
    except TypeError:
        try:
            initialized = control.init(simple_vram_headroom=simple_vram_headroom)
        except TypeError:
            initialized = control.init()
    if initialized is not True:
        raise GateError(f"ComfyUI DynamicVRAM bootstrap returned {initialized!r}")
    return control


def _run_comfyui(
    source_root: Path, preflight: dict[str, Any], output_dir: Path, checkpoint: Any
) -> dict[str, Any]:
    settings = preflight["runtime_settings"]
    simple_vram_headroom = int(settings["aimdo_simple_vram_headroom_bytes"])
    extra_vram_headroom = int(settings["aimdo_device_extra_vram_headroom_bytes"])
    expected_dtypes = {
        "diffusion_compute_dtype": "bfloat16",
        "text_compute_dtype": "float32",
        "vae_compute_dtype": "bfloat16",
    }
    if any(settings[name] != value for name, value in expected_dtypes.items()):
        raise GateError("recorded compute dtypes cannot be configured identically in ComfyUI")
    sys.argv = [
        sys.argv[0],
        "--use-pytorch-cross-attention",
        "--bf16-unet",
        "--bf16-text-enc",
        "--reserve-vram",
        str(simple_vram_headroom / 1024**3),
        "--disable-nvml-pressure",
        "--bf16-vae",
    ]
    sys.path.insert(0, str(source_root))
    import comfy.options

    comfy.options.enable_args_parsing()
    control = _bootstrap_comfyui_aimdo(simple_vram_headroom)
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sample
    import comfy_extras.nodes_chroma_radiance as radiance_nodes
    import comfy_extras.nodes_cond as cond_nodes
    import comfy_extras.nodes_custom_sampler as sampling_nodes
    import comfy_extras.nodes_model_advanced as model_nodes
    import comfy_extras.nodes_sd3 as sd3_nodes
    import folder_paths
    import latent_preview
    import nodes
    import numpy as np
    import torch

    try:
        aimdo_ready = control.init_devices(
            (device.index, extra_vram_headroom)
            for device in comfy.model_management.get_all_torch_devices()
        )
    except TypeError:
        aimdo_ready = control.init_devices(
            device.index for device in comfy.model_management.get_all_torch_devices()
        )
    if aimdo_ready is not True:
        raise GateError(f"ComfyUI DynamicVRAM device activation returned {aimdo_ready!r}")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    if not comfy.model_management.pytorch_attention_enabled():
        raise GateError("ComfyUI did not activate the requested PyTorch SDPA backend")

    def execute_node(call: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        finally:
            comfy.model_management.reset_cast_buffers()

    artifact_paths = {name: Path(value["path"]) for name, value in preflight["artifacts"].items()}
    folder_paths.add_model_folder_path(
        "diffusion_models", str(artifact_paths[preflight["variant"]].parent)
    )
    folder_paths.add_model_folder_path("text_encoders", str(artifact_paths["t5xxl"].parent))
    if "vae" in artifact_paths:
        folder_paths.add_model_folder_path("vae", str(artifact_paths["vae"].parent))
    workload = preflight["workload"]
    states: dict[int, tuple[Any, Any]] = {}
    original_callback = latent_preview.prepare_callback

    def diagnostic_callback(_model: Any, steps: int, x0_output: dict[str, Any]) -> Any:
        selected = _selected_steps(steps)

        def callback(step: int, denoised: Any, current: Any, _total: int) -> None:
            x0_output["x0"] = denoised
            if step in selected:
                states[step] = (
                    current.detach().float().cpu().contiguous(),
                    denoised.detach().float().cpu().contiguous(),
                )

        return callback

    pipeline: dict[str, Any] = {}

    def build() -> None:
        model = execute_node(nodes.UNETLoader().load_unet, workload["model_name"], "default")[0]
        if workload["model_options"] is not None:
            model = execute_node(
                radiance_nodes.ChromaRadianceOptions.execute,
                model=model,
                **workload["model_options"],
            )[0]
        model = execute_node(
            model_nodes.ModelSamplingAuraFlow().patch_aura, model, workload["sampling_shift"]
        )[0]
        clip = execute_node(
            nodes.CLIPLoader().load_clip, workload["clip_name"], "chroma", "default"
        )[0]
        clip = execute_node(cond_nodes.T5TokenizerOptions.execute, clip, **workload["tokenizer"])[0]
        positive = execute_node(nodes.CLIPTextEncode().encode, clip, workload["positive_prompt"])[0]
        negative = execute_node(nodes.CLIPTextEncode().encode, clip, workload["negative_prompt"])[0]
        if workload["variant"] == "radiance":
            latent = execute_node(
                radiance_nodes.EmptyChromaRadianceLatentImage.execute,
                width=workload["width"],
                height=workload["height"],
                batch_size=1,
            )[0]
            vae = execute_node(nodes.VAELoader().load_vae, "pixel_space")[0]
        else:
            latent = execute_node(
                sd3_nodes.EmptySD3LatentImage.execute,
                workload["width"],
                workload["height"],
                1,
            )[0]
            vae = execute_node(nodes.VAELoader().load_vae, workload["vae_name"])[0]
        if workload["scheduler"]["kind"] == "beta" and workload["variant"] == "radiance":
            sigmas = execute_node(
                sampling_nodes.BetaSamplingScheduler.execute,
                model,
                workload["scheduler"]["steps"],
                workload["scheduler"]["alpha"],
                workload["scheduler"]["beta"],
            )[0]
        else:
            sigmas = execute_node(
                sampling_nodes.BasicScheduler.execute,
                model,
                workload["scheduler"]["kind"],
                workload["scheduler"]["steps"],
                workload["scheduler"]["denoise"],
            )[0]
        if model.model_dtype() is not torch.bfloat16:
            raise GateError(
                f"ComfyUI Chroma diffusion dtype is {model.model_dtype()}, not bfloat16"
            )
        text_compute_dtype = clip.patcher.get_model_object("manual_cast_dtype")
        if text_compute_dtype is not torch.float32:
            raise GateError(
                f"ComfyUI Chroma text compute dtype is {text_compute_dtype}, not float32"
            )
        if workload["variant"] == "chroma" and vae.vae_dtype is not torch.bfloat16:
            raise GateError(f"ComfyUI Chroma VAE dtype is {vae.vae_dtype}, not bfloat16")
        pipeline.update(
            {
                "clip": clip,
                "guider": execute_node(
                    sampling_nodes.CFGGuider.execute,
                    model,
                    positive,
                    negative,
                    workload["cfg"],
                )[0],
                "latent": latent,
                "model": model,
                "negative": negative,
                "noise": execute_node(sampling_nodes.RandomNoise.execute, workload["seed"])[0],
                "positive": positive,
                "sampler": execute_node(sampling_nodes.KSamplerSelect.execute, workload["sampler"])[
                    0
                ],
                "sigmas": sigmas,
                "vae": vae,
            }
        )

    def sample() -> Any:
        return execute_node(
            sampling_nodes.SamplerCustomAdvanced.execute,
            pipeline["noise"],
            pipeline["guider"],
            pipeline["sampler"],
            pipeline["sigmas"],
            pipeline["latent"],
        )

    def finish(result: Any) -> tuple[Any, Any]:
        image = execute_node(nodes.VAEDecode().decode, pipeline["vae"], result[0])[0]
        return result[0]["samples"], image

    def execute() -> tuple[Any, Any]:
        return finish(sample())

    latent_preview.prepare_callback = diagnostic_callback

    def cold() -> tuple[Any, Any]:
        build()
        return execute()

    cold_latent, cold_image, cold_metrics = _run_metrics(torch, cold)
    latent_preview.prepare_callback = original_callback
    conditioning = {
        "negative": _save_array(
            np,
            output_dir,
            "conditioning-negative",
            _array(np, pipeline["negative"][0][0]),
        ),
        "positive": _save_array(
            np,
            output_dir,
            "conditioning-positive",
            _array(np, pipeline["positive"][0][0]),
        ),
    }
    noise = pipeline["noise"].generate_noise(pipeline["latent"])
    intermediates = {}
    for step, (current, denoised) in states.items():
        intermediates[str(step)] = {
            "current": _save_array(np, output_dir, f"step-{step}-current", _array(np, current)),
            "denoised": _save_array(np, output_dir, f"step-{step}-denoised", _array(np, denoised)),
        }
    arrays = {
        "cold_image": _save_array(np, output_dir, "cold-image", _array(np, cold_image)),
        "cold_latent": _save_array(np, output_dir, "cold-latent", _array(np, cold_latent)),
        "conditioning": conditioning,
        "initial_noise": _save_array(np, output_dir, "initial-noise", _array(np, noise)),
        "intermediates": intermediates,
        "sigmas": _save_array(np, output_dir, "sigmas", _array(np, pipeline["sigmas"])),
    }
    checkpoint({"arrays": arrays, "cold": cold_metrics, "measured": []})
    warmup_profile: dict[str, float] = {}

    def profile_warmup() -> tuple[Any, Any]:
        latent, image, phases = _profile_pipeline(torch, sample, finish)
        warmup_profile.update(phases)
        return latent, image

    _run_metrics(torch, profile_warmup)
    checkpoint({"diagnostics": {"discarded_warmup": warmup_profile}})
    measured = []
    for index in range(MEASURED_RUNS):
        latent, image, metrics = _run_metrics(torch, execute)
        metrics["index"] = index
        metrics["image"] = _save_array(np, output_dir, f"warm-{index}-image", _array(np, image))
        metrics["latent"] = _save_array(np, output_dir, f"warm-{index}-latent", _array(np, latent))
        measured.append(metrics)
        checkpoint({"measured": measured})
    model_patcher = pipeline["model"]
    fallback = {
        "attention_backend": "sdpa",
        "mechanism": "aimdo",
        "lowvram_patch_count": int(model_patcher.lowvram_patch_counter()),
        "model_loaded_bytes": int(model_patcher.loaded_size()),
        "model_patcher": type(model_patcher).__name__,
        "oom_count": 0,
    }
    residency = {
        "clip": _comfyui_residency_receipt(pipeline["clip"].patcher),
        "model": _comfyui_residency_receipt(model_patcher),
        "vae": _residency_receipt(
            pipeline["vae"].patcher,
            _comfyui_residency_receipt,
            stateless=workload["variant"] == "radiance",
        ),
    }
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)
    gc.collect()
    _clear_cublas_workspaces(torch)
    torch.cuda.empty_cache()
    _trim_process_heap()
    result = {
        "arrays": arrays,
        "cold": cold_metrics,
        "diagnostics": {
            "active_cuda_allocations_after_cleanup": _active_cuda_allocations(torch),
            "discarded_warmup": warmup_profile,
            "residency_before_cleanup": residency,
        },
        "fallback": fallback,
        "measured": measured,
        "post_cleanup": {
            "rss_bytes": _rss_bytes(),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            "vram_bytes": _process_vram_bytes(os.getpid()),
        },
    }
    checkpoint(result)
    return result


def _asset(asset_ref: Any, receipt: dict[str, Any]) -> Any:
    from dinkster_assets.integrity import AssetVerificationRecord
    from dinkster_assets.model import AssetResolution

    path = Path(receipt["path"])
    digest = receipt["blake3"]
    verification = AssetVerificationRecord.from_json(digest, receipt.get("verification"))
    if verification is None:
        raise GateError(f"artifact verification is missing or malformed: {path}")

    class Resolver:
        def resolve(self, requested: str) -> Path | None:
            return path if requested == digest else None

        def resolve_asset(self, requested: str) -> AssetResolution | None:
            return AssetResolution(path, verification) if requested == digest else None

    return asset_ref(digest, path.name, int(receipt["size_bytes"]), resolver=Resolver())


def _execution_context(
    workers: Any,
    identity: str,
    role: str,
    attention_route_token: Any,
    compute_dtype: str,
) -> Any:
    values = {"diffusion": "unloaded", "text": "unloaded", "vae": "unloaded"}
    if role == "vae-pixel":
        values["vae"] = compute_dtype
    else:
        values[role] = compute_dtype
    return workers.ExecutionContext(
        "native",
        identity,
        diffusion_dtype=values["diffusion"],
        text_dtype=values["text"],
        vae_dtype=values["vae"],
        attention_policy="sdpa",
        attention_route_token=attention_route_token,
        preview_mode="off",
    )


def _run_dinkster(
    source_root: Path, preflight: dict[str, Any], output_dir: Path, checkpoint: Any
) -> dict[str, Any]:
    for source in sorted((source_root / "packages").glob("*/src")):
        sys.path.insert(0, str(source))
    sys.path.insert(0, str(source_root / "src"))
    from dinkster_inference import (
        default_diffusion_dtype,
        default_text_dtype,
        default_vae_dtype,
    )
    from dinkster_memory import DEFAULT_ACCELERATOR_HEADROOM_BYTES
    from dinkster_workers.host import _bootstrap_aimdo, _prepare_accelerator_runtime

    settings = preflight["runtime_settings"]
    simple_vram_headroom = int(settings["aimdo_simple_vram_headroom_bytes"])
    if simple_vram_headroom != DEFAULT_ACCELERATOR_HEADROOM_BYTES:
        raise GateError("recorded Aimdo headroom differs from the Dinkster production default")
    if int(settings["aimdo_device_extra_vram_headroom_bytes"]) != 0:
        raise GateError("Dinkster production default has zero extra per-device Aimdo headroom")
    if settings["aimdo_nvml_pressure"] is not False:
        raise GateError("recorded Aimdo NVML pressure differs from the Dinkster production default")
    family_id = (
        "dinkster.chroma_radiance" if preflight["variant"] == "radiance" else "dinkster.chroma"
    )
    expected_dtypes = {
        "diffusion_compute_dtype": default_diffusion_dtype(family_id).name,
        "text_compute_dtype": default_text_dtype(family_id).name,
        "vae_compute_dtype": default_vae_dtype(family_id).name,
    }
    if any(settings[name] != value for name, value in expected_dtypes.items()):
        raise GateError("recorded compute dtypes differ from Dinkster production defaults")
    if not _bootstrap_aimdo(True, simple_vram_headroom=simple_vram_headroom):
        raise GateError("Dinkster production-default Aimdo bootstrap failed")
    os.environ["DINKSTER_AIMDO_ARM"] = "auto"
    import dinkster_compat_comfy.native_arm as arm
    import dinkster_workers as workers
    import numpy as np
    import torch

    if not _prepare_accelerator_runtime(True):
        raise GateError("Dinkster production accelerator runtime preparation failed")
    from dinkster_assets import AssetRef
    from dinkster_inference import split_component_conditioning
    from dinkster_inference_torch import (
        collect_partial_residency_timing,
        discover_attention_route_token,
        materialize_basic_conditioning,
        prepare_noise,
    )
    from dinkster_protocol import canonical_attention_route_token_bytes
    from dinkster_workers.execution import use_execution_context

    workload = preflight["workload"]
    receipts = preflight["artifacts"]
    attention_route_token = discover_attention_route_token("sdpa")
    identities = _dinkster_identities(
        source_root,
        receipts,
        workload["variant"],
        attention_policy="sdpa",
        attention_route_token=attention_route_token,
        require_torch_free=False,
    )
    checkpoint(
        {
            "attention_route_token": json.loads(
                canonical_attention_route_token_bytes(attention_route_token)
            ),
            "dinkster_identities": identities,
        }
    )
    states: dict[int, tuple[Any, Any]] = {}

    class DiagnosticEmitter:
        def stage(self) -> Any:
            return nullcontext()

        def on_state(self, event: Any) -> None:
            if event.step in _selected_steps(event.total):
                if event.denoised is None:
                    raise GateError(
                        f"Dinkster diagnostic step {event.step} omitted the denoised state"
                    )
                states[event.step] = (
                    event.current.detach().float().cpu().contiguous(),
                    event.denoised.detach().float().cpu().contiguous(),
                )

    original_emitter = arm.sampling_preview_emitter
    pipeline: dict[str, Any] = {}

    def build() -> None:
        model_asset = _asset(AssetRef, receipts[workload["variant"]])
        text_asset = _asset(AssetRef, receipts["t5xxl"])
        with use_execution_context(
            _execution_context(
                workers,
                identities[workload["variant"]],
                "diffusion",
                attention_route_token,
                settings["diffusion_compute_dtype"],
            )
        ):
            model_handle = arm.NativeLoadDiffusionModel.execute(
                diffusion_model=model_asset, weight_dtype="default"
            )["model"]
        model = model_handle
        if workload["model_options"] is not None:
            model = arm.GenerationChromaRadianceOptions.execute(
                model=model, **workload["model_options"]
            )["model"]
        model = arm.GenerationChromaModelSampling.execute(
            model=model, shift=workload["sampling_shift"]
        )["model"]
        with use_execution_context(
            _execution_context(
                workers,
                identities["t5xxl"],
                "text",
                attention_route_token,
                settings["text_compute_dtype"],
            )
        ):
            clip_handle = arm.NativeLoadClip.execute(
                text_encoder=text_asset, type="chroma", device="default"
            )["clip"]
        clip = arm.GenerationT5TokenizerOptions.execute(clip=clip_handle, **workload["tokenizer"])[
            "clip"
        ]
        positive = arm.GenerationClipTextEncode.execute(
            text=workload["positive_prompt"], clip=clip
        )["conditioning"]
        negative = arm.GenerationClipTextEncode.execute(
            text=workload["negative_prompt"], clip=clip
        )["conditioning"]
        if workload["variant"] == "radiance":
            latent = arm.GenerationEmptyChromaRadianceLatentImage.execute(
                width=workload["width"], height=workload["height"], batch_size=1
            )["latent"]
            with use_execution_context(
                _execution_context(
                    workers,
                    identities["pixel_vae"],
                    "vae-pixel",
                    attention_route_token,
                    settings["vae_compute_dtype"],
                )
            ):
                vae_handle = arm.NativeLoadVae.execute(pixel_space=True)["vae"]
        else:
            latent = arm.GenerationEmptySD3LatentImage.execute(
                width=workload["width"], height=workload["height"], batch_size=1
            )["latent"]
            with use_execution_context(
                _execution_context(
                    workers,
                    identities["vae"],
                    "vae",
                    attention_route_token,
                    settings["vae_compute_dtype"],
                )
            ):
                vae_handle = arm.NativeLoadVae.execute(vae=_asset(AssetRef, receipts["vae"]))["vae"]
        if workload["variant"] == "radiance":
            sigmas = arm.GenerationBetaSamplingScheduler.execute(
                model=model,
                steps=workload["scheduler"]["steps"],
                alpha=workload["scheduler"]["alpha"],
                beta=workload["scheduler"]["beta"],
            )["sigmas"]
        else:
            sigmas = arm.GenerationBasicScheduler.execute(
                model=model,
                scheduler=workload["scheduler"]["kind"],
                steps=workload["scheduler"]["steps"],
                denoise=workload["scheduler"]["denoise"],
            )["sigmas"]
        pipeline.update(
            {
                "clip_handle": clip_handle,
                "guider": arm.GenerationCFGGuider.execute(
                    model=model,
                    positive=positive,
                    negative=negative,
                    cfg=workload["cfg"],
                )["guider"],
                "latent": latent,
                "model_handle": model_handle,
                "noise": arm.GenerationRandomNoise.execute(noise_seed=workload["seed"])["noise"],
                "positive": positive,
                "negative": negative,
                "sampler": arm.GenerationKSamplerSelect.execute(sampler_name=workload["sampler"])[
                    "sampler"
                ],
                "sigmas": sigmas,
                "vae_handle": vae_handle,
            }
        )

    def sample() -> Any:
        return arm.GenerationSamplerCustomAdvanced.execute(
            noise=pipeline["noise"],
            guider=pipeline["guider"],
            sampler=pipeline["sampler"],
            sigmas=pipeline["sigmas"],
            latent_image=pipeline["latent"],
        )["output"]

    def finish(output: Any) -> tuple[Any, Any]:
        image = arm.GenerationVAEDecode.execute(samples=output, vae=pipeline["vae_handle"])["image"]
        return output["samples"], image

    def execute() -> tuple[Any, Any]:
        return finish(sample())

    arm.sampling_preview_emitter = lambda _handle: DiagnosticEmitter()

    def cold() -> tuple[Any, Any]:
        build()
        return execute()

    cold_latent, cold_image, cold_metrics = _run_metrics(torch, cold)
    arm.sampling_preview_emitter = original_emitter
    arrays = {
        "cold_image": _save_array(np, output_dir, "cold-image", _array(np, cold_image)),
        "cold_latent": _save_array(np, output_dir, "cold-latent", _array(np, cold_latent)),
    }
    conditioning = {}
    for lane in ("positive", "negative"):
        stripped, _binding = split_component_conditioning(pipeline[lane])
        value = materialize_basic_conditioning(stripped, device="cpu").embeddings
        conditioning[lane] = _save_array(np, output_dir, f"conditioning-{lane}", _array(np, value))
    arrays["conditioning"] = conditioning
    latent = pipeline["latent"]["samples"]
    arrays["initial_noise"] = _save_array(
        np,
        output_dir,
        "initial-noise",
        _array(np, prepare_noise(latent, workload["seed"])),
    )
    arrays["sigmas"] = _save_array(
        np, output_dir, "sigmas", np.asarray(pipeline["sigmas"].values, dtype=np.float32)
    )
    arrays["intermediates"] = {}
    for step, (current, denoised) in states.items():
        arrays["intermediates"][str(step)] = {
            "current": _save_array(np, output_dir, f"step-{step}-current", _array(np, current)),
            "denoised": _save_array(np, output_dir, f"step-{step}-denoised", _array(np, denoised)),
        }
    checkpoint({"arrays": arrays, "cold": cold_metrics, "measured": []})
    warmup_profile: dict[str, Any] = {}

    def profile_warmup() -> tuple[Any, Any]:
        latent_value, image, phases = _profile_pipeline(torch, sample, finish)
        warmup_profile.update(phases)
        return latent_value, image

    with collect_partial_residency_timing() as timing:
        _run_metrics(torch, profile_warmup)
    warmup_profile["partial_residency"] = asdict(timing.report())
    checkpoint({"diagnostics": {"discarded_warmup": warmup_profile}})
    measured = []
    for index in range(MEASURED_RUNS):
        latent_value, image, metrics = _run_metrics(torch, execute)
        metrics["index"] = index
        metrics["image"] = _save_array(np, output_dir, f"warm-{index}-image", _array(np, image))
        metrics["latent"] = _save_array(
            np, output_dir, f"warm-{index}-latent", _array(np, latent_value)
        )
        measured.append(metrics)
        checkpoint({"measured": measured})
    handle = pipeline["model_handle"]
    route = handle.residency_route
    attention_backends = sorted(
        {status.primary for status in handle.runtime.attention_status.values()}
    )
    fallback = {
        "attention_backends": attention_backends,
        "logical_loaded_bytes": sum(value.loaded_bytes() for value in handle.mechanisms),
        "logical_offloaded_bytes": sum(value.offloaded_bytes() for value in handle.mechanisms),
        "oom_count": 0,
        "route": None if route is None else asdict(route),
    }
    residency = {
        "model_handle": _dinkster_residency_receipt(pipeline["model_handle"]),
        "clip_handle": _dinkster_residency_receipt(pipeline["clip_handle"]),
        "vae_handle": _residency_receipt(
            pipeline["vae_handle"],
            _dinkster_residency_receipt,
            stateless=workload["variant"] == "radiance",
        ),
    }
    for name in ("model_handle", "clip_handle", "vae_handle"):
        value = pipeline[name]
        if hasattr(value, "terminal_release") and not value.released:
            value.terminal_release()
    gc.collect()
    _clear_cublas_workspaces(torch)
    torch.cuda.empty_cache()
    _trim_process_heap()
    result = {
        "arrays": arrays,
        "cold": cold_metrics,
        "diagnostics": {
            "active_cuda_allocations_after_cleanup": _active_cuda_allocations(torch),
            "discarded_warmup": warmup_profile,
            "residency_before_cleanup": residency,
        },
        "fallback": fallback,
        "measured": measured,
        "post_cleanup": {
            "rss_bytes": _rss_bytes(),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            "vram_bytes": _process_vram_bytes(os.getpid()),
        },
    }
    checkpoint(result)
    return result


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_gpu(args: argparse.Namespace) -> int:
    preflight = json.loads(args.preflight.read_text())
    gpu_index, gpu_uuid = _gpu_target(preflight)
    claim = _acquire_gpu_claim(gpu_index, f"chroma-e2e-{args.engine}-{args.variant}")
    partial: dict[str, Any] = {}
    device: dict[str, Any] | None = None

    def write_record(status: str, failure: dict[str, Any] | None = None) -> None:
        nonlocal device
        torch = sys.modules.get("torch")
        if torch is not None and device is None:
            device = _device_receipt(torch, gpu_index, gpu_uuid)
        record = {
            **partial,
            "device": device,
            "engine": args.engine,
            "preflight": str(args.preflight.resolve()),
            "preflight_sha256": (
                _sha256_file(args.preflight) if args.preflight.is_file() else None
            ),
            "runtime_settings": preflight.get("runtime_settings"),
            "schema": 1,
            "source_commit": preflight.get("source_commit"),
            "status": status,
            "variant": args.variant,
            "workload": preflight.get("workload"),
        }
        if failure is not None:
            record["failure"] = failure
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.output, record)

    def checkpoint(update: dict[str, Any]) -> None:
        partial.update(update)
        write_record("running")

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
            raise GateError(f"CUDA_VISIBLE_DEVICES must be {gpu_index}")
        _verify_physical_gpu(gpu_index, gpu_uuid)
        if preflight["harness_sha256"] != _sha256_file(Path(__file__).resolve()):
            raise GateError("harness changed after offline preflight")
        if preflight["engine"] != args.engine or preflight["variant"] != args.variant:
            raise GateError("preflight engine or variant differs from launch")
        if preflight["source_root"] != str(args.source_root.resolve()):
            raise GateError("source root differs from offline preflight")
        _require_source(args.source_root.resolve(), preflight["source_commit"], args.engine)
        _verify_artifact_receipts(preflight["artifacts"])
        output_dir = args.output.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.engine == "comfyui":
            result = _run_comfyui(args.source_root.resolve(), preflight, output_dir, checkpoint)
        else:
            result = _run_dinkster(args.source_root.resolve(), preflight, output_dir, checkpoint)
        partial.update(result)
        write_record("success")
        return 0
    except BaseException as exc:
        oom = "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()
        fallback = dict(partial.get("fallback", {}))
        fallback["oom_count"] = int(fallback.get("oom_count", 0)) + int(oom)
        fallback["outcome"] = "oom" if oom else "error"
        partial["fallback"] = fallback
        write_record(
            "failed",
            {
                "message": str(exc),
                "oom": oom,
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            },
        )
        raise
    finally:
        fcntl = import_module("fcntl")
        fcntl.flock(claim.fileno(), fcntl.LOCK_UN)
        claim.close()


def _load_array(np: Any, receipt: dict[str, Any]) -> Any:
    array = np.load(receipt["path"], allow_pickle=False)
    if list(array.shape) != receipt["shape"] or str(array.dtype) != receipt["dtype"]:
        raise GateError(f"array metadata differs: {receipt['path']}")
    if _sha256_bytes(array.tobytes(order="C")) != receipt["sha256"]:
        raise GateError(f"array digest differs: {receipt['path']}")
    return array


def _compare_array(np: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    reference = _load_array(np, left)
    candidate = _load_array(np, right)
    if reference.shape != candidate.shape:
        raise GateError(f"array shapes differ: {reference.shape} != {candidate.shape}")
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "p99_abs": float(np.quantile(difference, 0.99)) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
    }


def _array_receipts(value: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    if {"dtype", "path", "sha256", "shape"} <= value.keys():
        return {prefix: value}
    receipts: dict[str, dict[str, Any]] = {}
    for name, nested in value.items():
        if isinstance(nested, dict):
            child = f"{prefix}/{name}" if prefix else name
            receipts.update(_array_receipts(nested, child))
    return receipts


def _metrics(records: list[dict[str, Any]], key: str, phase: str) -> dict[str, Any]:
    if phase == "warm":
        values = [float(run[key]) for record in records for run in record["measured"]]
    else:
        values = [float(record[phase][key]) for record in records]
    median = statistics.median(values)
    return {
        "mad": statistics.median(abs(value - median) for value in values),
        "maximum": max(values),
        "median": median,
        "p95": _percentile(values, 0.95),
        "range": [min(values), max(values)],
        "values": values,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _fallback_ok(record: dict[str, Any]) -> bool:
    fallback = record["fallback"]
    if fallback.get("oom_count") != 0:
        return False
    if record["engine"] == "comfyui":
        return (
            fallback.get("attention_backend") == "sdpa"
            and fallback.get("lowvram_patch_count") == 0
            and fallback.get("mechanism") == "aimdo"
            and type(fallback.get("model_loaded_bytes")) is int
            and fallback["model_loaded_bytes"] >= 0
            and fallback.get("model_patcher") == "ModelPatcherDynamic"
        )
    route = fallback.get("route")
    return (
        fallback.get("attention_backends") == ["sdpa"]
        and type(fallback.get("logical_loaded_bytes")) is int
        and fallback["logical_loaded_bytes"] >= 0
        and type(fallback.get("logical_offloaded_bytes")) is int
        and fallback["logical_offloaded_bytes"] >= 0
        and fallback["logical_loaded_bytes"] + fallback["logical_offloaded_bytes"] > 0
        and isinstance(route, dict)
        and route.get("dynamic_components") == ["diffusion"]
        and route.get("fallback_components") == []
        and route.get("requested") == "auto"
        and route.get("mechanism") == "aimdo"
        and route.get("fallback_reason") is None
        and route.get("resident_components") == []
    )


def _validated_preflight(record: dict[str, Any], dinkster_commit: str) -> dict[str, Any]:
    path_value = record.get("preflight")
    expected_digest = record.get("preflight_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise GateError("record preflight path is missing")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise GateError("record preflight digest is missing")
    path = Path(path_value)
    try:
        actual_digest = _sha256_file(path)
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"record preflight is unavailable or malformed: {path}") from exc
    if actual_digest != expected_digest:
        raise GateError(f"record preflight digest differs: {path}")
    if not isinstance(raw, dict):
        raise GateError(f"record preflight must be an object: {path}")
    preflight: dict[str, Any] = raw
    if record.get("schema") != 1 or preflight.get("schema") != 1:
        raise GateError("record and preflight schemas must both be 1")
    if preflight.get("thread_id") != THREAD_ID:
        raise GateError("preflight thread differs from the acceptance owner")
    for key in ("engine", "runtime_settings", "source_commit", "variant", "workload"):
        if record.get(key) != preflight.get(key):
            raise GateError(f"record {key} differs from its preflight")
    if preflight.get("dinkster_commit") != dinkster_commit:
        raise GateError("preflight Dinkster source differs")
    if preflight.get("harness_sha256") != _sha256_file(Path(__file__).resolve()):
        raise GateError("preflight harness differs from the comparison harness")

    gpu_index, gpu_uuid = _gpu_target(preflight)
    device = record.get("device")
    if not isinstance(device, dict) or (device.get("gpu_index"), device.get("gpu_uuid")) != (
        gpu_index,
        gpu_uuid,
    ):
        raise GateError("record physical GPU differs from its preflight")

    variant = preflight.get("variant")
    if variant not in TEMPLATES:
        raise GateError(f"preflight variant is unsupported: {variant!r}")
    workload = preflight["workload"]
    if (
        not isinstance(workload, dict)
        or workload.get("template_commit") != TEMPLATE_COMMIT
        or workload.get("template_path") != TEMPLATES[variant]
        or not isinstance(workload.get("template_sha256"), str)
    ):
        raise GateError("preflight workflow differs from the pinned template")
    expected_names = {variant, "t5xxl"}
    if variant == "chroma":
        expected_names.add("vae")
    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise GateError("preflight artifact coverage differs from the pinned workload")
    for name in expected_names:
        receipt = artifacts[name]
        expected = ARTIFACTS[name]
        if not isinstance(receipt, dict) or any(
            receipt.get(key) != expected[key] for key in ("revision", "sha256", "size_bytes", "url")
        ):
            raise GateError(f"preflight {name} artifact differs from the pinned artifact")
        blake3 = receipt.get("blake3")
        if (
            not isinstance(blake3, str)
            or not blake3.startswith("blake3:")
            or not isinstance(receipt.get("verification"), dict)
        ):
            raise GateError(f"preflight {name} artifact verification is missing")

    identities = preflight.get("dinkster_unrouted_identities")
    identity_names = expected_names | {"pixel_vae"}
    if (
        not isinstance(identities, dict)
        or set(identities) != identity_names
        or not all(isinstance(value, str) and value for value in identities.values())
    ):
        raise GateError("preflight Dinkster identity coverage is missing or malformed")
    return preflight


def build_verdict(records: list[dict[str, Any]], dinkster_commit: str) -> dict[str, Any]:
    import numpy as np
    from dinkster_protocol import attention_route_token_from_wire

    if [record["engine"] for record in records] != list(PROCESS_ORDER):
        raise GateError("record order must be comfyui, dinkster, dinkster, comfyui")
    preflights = [_validated_preflight(record, dinkster_commit) for record in records]
    first_preflight = preflights[0]
    for preflight in preflights[1:]:
        for key in ("artifacts", "dinkster_unrouted_identities", "gpu_index", "gpu_uuid"):
            if preflight[key] != first_preflight[key]:
                raise GateError(f"process preflight {key} differs")
    first = records[0]
    for record in records:
        if record["status"] != "success":
            raise GateError("every engine process must succeed")
        if record["variant"] != first["variant"] or record["workload"] != first["workload"]:
            raise GateError("records must carry one identical workload")
        if record["runtime_settings"] != first["runtime_settings"]:
            raise GateError("records must carry identical runtime settings")
        if len(record["measured"]) != MEASURED_RUNS:
            raise GateError(f"each process must have {MEASURED_RUNS} measured warm runs")
        if record["device"] != first["device"]:
            raise GateError("engine runtime or device receipts differ")
        if set(record["arrays"]["intermediates"]) != _expected_step_names(record["workload"]):
            raise GateError("first, middle, and final sampling states were not all captured")
    if any(record["source_commit"] != COMFYUI_COMMIT for record in (records[0], records[3])):
        raise GateError("ComfyUI source receipt differs")
    if any(record["source_commit"] != dinkster_commit for record in (records[1], records[2])):
        raise GateError("Dinkster source receipt differs")
    for record in records[1:3]:
        try:
            token = attention_route_token_from_wire(record.get("attention_route_token"))
        except (TypeError, ValueError) as exc:
            raise GateError("Dinkster attention route token is missing or malformed") from exc
        if (
            token.requested_policy != "sdpa"
            or token.device_kind != "cuda"
            or any((route.primary, route.fallback) != ("sdpa", None) for route in token.routes)
        ):
            raise GateError("Dinkster attention route token differs from the SDPA GPU workload")
        identities = record.get("dinkster_identities")
        if (
            not isinstance(identities, dict)
            or set(identities) != set(first_preflight["dinkster_unrouted_identities"])
            or not all(isinstance(value, str) and value for value in identities.values())
        ):
            raise GateError("Dinkster routed identity coverage is missing or malformed")
    for key in ("attention_route_token", "dinkster_identities"):
        if records[1][key] != records[2][key]:
            raise GateError(f"Dinkster {key} changed between processes")
    comfyui = [records[0], records[3]]
    dinkster = [records[1], records[2]]
    flattened = [_array_receipts(record["arrays"]) for record in records]
    for index, record in enumerate(records):
        for receipt in flattened[index].values():
            _load_array(np, receipt)
        for run in record["measured"]:
            for output, cold_name in (("image", "cold_image"), ("latent", "cold_latent")):
                comparison = _compare_array(np, flattened[index][cold_name], run[output])
                if not comparison["exact"]:
                    raise GateError(
                        f"{record['engine']} {output} changed between cold and warm runs"
                    )
    for left, right in ((0, 3), (1, 2)):
        if flattened[left].keys() != flattened[right].keys():
            raise GateError(f"{records[left]['engine']} process array coverage differs")
        if any(
            not _compare_array(np, flattened[left][name], flattened[right][name])["exact"]
            for name in flattened[left]
        ):
            raise GateError(f"{records[left]['engine']} output changed between processes")
    if flattened[0].keys() != flattened[1].keys():
        raise GateError("ComfyUI and Dinkster captured array coverage differs")
    comparisons = {
        name: _compare_array(np, flattened[0][name], flattened[1][name]) for name in flattened[0]
    }
    performance = {
        phase: {
            engine: _metrics(group, "seconds", phase)
            for engine, group in (("comfyui", comfyui), ("dinkster", dinkster))
        }
        for phase in ("cold", "warm")
    }
    memory_keys = (
        "peak_process_rss_bytes",
        "peak_process_vram_bytes",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
    )
    residual_keys = (
        "rss_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "vram_bytes",
    )
    memory = {
        phase: {
            key: {
                engine: _metrics(group, key, phase)
                for engine, group in (("comfyui", comfyui), ("dinkster", dinkster))
            }
            for key in memory_keys
        }
        for phase in ("cold", "warm")
    }
    memory["post_cleanup"] = {
        key: {
            engine: _metrics(group, key, "post_cleanup")
            for engine, group in (("comfyui", comfyui), ("dinkster", dinkster))
        }
        for key in residual_keys
    }
    compared_statistics = ("median", "p95", "maximum")
    performance_pass = all(
        values["dinkster"][statistic] <= values["comfyui"][statistic]
        for values in performance.values()
        for statistic in compared_statistics
    )
    peak_memory_pass = all(
        values["dinkster"][statistic] <= values["comfyui"][statistic]
        for phase in ("cold", "warm")
        for key in ("peak_process_rss_bytes", "peak_process_vram_bytes")
        for values in (memory[phase][key],)
        for statistic in compared_statistics
    )
    residual_memory_pass = all(
        values["dinkster"][statistic] <= values["comfyui"][statistic]
        for values in memory["post_cleanup"].values()
        for statistic in compared_statistics
    )
    memory_pass = peak_memory_pass and residual_memory_pass
    fallback_pass = all(_fallback_ok(record) for record in records)
    correctness_pass = all(value["exact"] for value in comparisons.values())
    return {
        "correctness": {"comparisons": comparisons, "pass": correctness_pass},
        "evidence": {
            "artifacts": first_preflight["artifacts"],
            "attention_route_token": records[1]["attention_route_token"],
            "dinkster_identities": records[1]["dinkster_identities"],
            "dinkster_unrouted_identities": first_preflight["dinkster_unrouted_identities"],
            "harness_sha256": first_preflight["harness_sha256"],
            "preflight_sha256": [record["preflight_sha256"] for record in records],
        },
        "fallback": {
            "pass": fallback_pass,
            "records": [record["fallback"] for record in records],
        },
        "memory": {"metrics": memory, "pass": memory_pass},
        "overall_pass": correctness_pass and performance_pass and memory_pass and fallback_pass,
        "performance": {"metrics": performance, "pass": performance_pass},
        "policy": {
            "correctness": "all retained intermediates and outputs must be bit-exact",
            "fallback": "both engines must use production Aimdo without fallback or OOM",
            "memory": (
                "Dinkster process cold and warm peaks and all residual median, p95, and maximum"
                " must not exceed ComfyUI; allocator peak metrics are diagnostic because VBAR"
                " allocations are outside PyTorch"
            ),
            "performance": (
                "Dinkster cold and warm median, p95, and maximum must not exceed ComfyUI"
            ),
            "process_order": list(PROCESS_ORDER),
        },
        "variant": first["variant"],
        "workload": first["workload"],
    }


def run_compare(args: argparse.Namespace) -> int:
    records = [json.loads(path.read_text()) for path in args.inputs]
    verdict = build_verdict(records, args.dinkster_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return 0 if verdict["overall_pass"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--artifact-root", type=Path, required=True)
    preflight.add_argument("--dinkster-commit", required=True)
    preflight.add_argument("--dinkster-root", type=Path, required=True)
    preflight.add_argument("--engine", choices=PROCESS_ORDER, required=True)
    preflight.add_argument("--gpu-index", type=int, required=True)
    preflight.add_argument("--gpu-uuid", required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--run-output", type=Path, required=True)
    preflight.add_argument("--run-python", type=Path, required=True)
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--template-root", type=Path, required=True)
    preflight.add_argument("--variant", choices=TEMPLATES, required=True)
    preflight.set_defaults(func=run_preflight)
    run = commands.add_parser("run")
    run.add_argument("--engine", choices=PROCESS_ORDER, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--variant", choices=TEMPLATES, required=True)
    run.set_defaults(func=run_gpu)
    compare = commands.add_parser("compare")
    compare.add_argument("--dinkster-commit", required=True)
    compare.add_argument("--inputs", type=Path, nargs=4, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=run_compare)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
