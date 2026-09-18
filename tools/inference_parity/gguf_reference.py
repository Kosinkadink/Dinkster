"""Run the pinned CPU GGUF dequantization and image comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.inference_parity.harness import (
    HarnessError,
    _git_output,
    canonical_bytes,
    digest_bytes,
    digest_file,
    image_ssim,
    load_json,
    validate_acceptance,
    validate_extensions_unchanged,
    validate_manifest,
    validate_pins,
    write_json,
)


def _run_adapter(
    workload: dict[str, Any],
    engine: str,
    roots: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    engine_pin = workload["engines"][engine]
    python = Path(engine_pin["python"].format(root=str(roots[engine])))
    adapter = Path(__file__).with_name(engine_pin["adapter"])
    receipt_path = output_dir / f"{engine}-reference.json"
    result = subprocess.run(
        [
            str(python),
            str(adapter),
            "--repo",
            str(roots[engine]),
            "--artifact-root",
            str(roots["artifact"]),
            "--output-dir",
            str(output_dir),
            "--workload-json",
            json.dumps(workload["execution"], sort_keys=True),
            "--reference-output",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / f"{engine}-reference.stdout.txt").write_text(result.stdout)
    (output_dir / f"{engine}-reference.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise HarnessError(f"{engine} reference adapter failed: {result.stderr.strip()}")
    return load_json(receipt_path)


def _metrics(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> dict[str, str]:
    if left.shape != right.shape:
        raise HarnessError(f"reference output shape mismatch: {left.shape} != {right.shape}")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "max_abs": format(float(difference.max()), ".12g"),
        "mean_abs": format(float(difference.mean()), ".12g"),
        "rmse": format(float(np.sqrt(np.mean(difference * difference))), ".12g"),
    }


def compare_reference(
    comfyui: dict[str, Any],
    dinkster: dict[str, Any],
    acceptance: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    rules = acceptance["workloads"][workload["id"]]["reference"]
    comfy_map = [
        {key: tensor[key] for key in ("key", "shape", "type")} for tensor in comfyui["tensor_map"]
    ]
    dinkster_map = [
        {key: tensor[key] for key in ("key", "shape", "type")} for tensor in dinkster["tensor_map"]
    ]
    comfy_map_keys = [tensor["key"] for tensor in comfy_map]
    dinkster_map_keys = [tensor["key"] for tensor in dinkster_map]
    map_keys_unique = len(set(comfy_map_keys)) == len(comfy_map_keys) and len(
        set(dinkster_map_keys)
    ) == len(dinkster_map_keys)
    mapping_pass = map_keys_unique and comfy_map == dinkster_map
    comfy_q8_keys = {tensor["key"] for tensor in comfy_map if tensor["type"] == "Q8_0"}
    dinkster_q8_keys = {tensor["key"] for tensor in dinkster_map if tensor["type"] == "Q8_0"}
    comfy_q8_receipt_keys = [tensor["key"] for tensor in comfyui["q8_tensors"]]
    dinkster_q8_receipt_keys = [tensor["key"] for tensor in dinkster["q8_tensors"]]
    q8_receipt_keys_unique = len(set(comfy_q8_receipt_keys)) == len(comfy_q8_receipt_keys) and len(
        set(dinkster_q8_receipt_keys)
    ) == len(dinkster_q8_receipt_keys)
    comfy_q8 = {tensor["key"]: tensor for tensor in comfyui["q8_tensors"]}
    dinkster_q8 = {tensor["key"]: tensor for tensor in dinkster["q8_tensors"]}
    q8_keys_pass = (
        q8_receipt_keys_unique
        and set(comfy_q8) == comfy_q8_keys
        and set(dinkster_q8) == dinkster_q8_keys
        and comfy_q8_keys == dinkster_q8_keys
    )
    q8_mismatches = [
        key
        for key in sorted(comfy_q8.keys() & dinkster_q8.keys())
        if comfy_q8[key] != dinkster_q8[key]
    ]
    q8_pass = q8_keys_pass and not q8_mismatches
    comfy_image = np.load(comfyui["image_path"], allow_pickle=False)
    dinkster_image = np.load(dinkster["image_path"], allow_pickle=False)
    comfy_latent = np.load(comfyui["latent_path"], allow_pickle=False)
    dinkster_latent = np.load(dinkster["latent_path"], allow_pickle=False)
    image = _metrics(comfy_image, dinkster_image)
    image["ssim"] = format(image_ssim(comfy_image, dinkster_image), ".12g")
    latent = _metrics(comfy_latent, dinkster_latent)
    image_pass = (
        float(image["max_abs"]) <= float(rules["image"]["max_abs"])
        and float(image["mean_abs"]) <= float(rules["image"]["mean_abs"])
        and float(image["ssim"]) >= float(rules["image"]["ssim_minimum"])
    )
    latent_pass = float(latent["max_abs"]) <= float(rules["latent"]["max_abs"]) and float(
        latent["mean_abs"]
    ) <= float(rules["latent"]["mean_abs"])
    return {
        "artifact_identity": dinkster["runtime_facts"],
        "engines": {
            "comfyui": {
                "attention_backend": comfyui["attention_backend"],
                "commit": workload["engines"]["comfyui"]["commit"],
                "gguf": comfyui["gguf"],
                "python": comfyui["python"],
                "torch": comfyui["torch"],
            },
            "dinkster": {
                "attention_backend": dinkster["attention_backend"],
                "commit": workload["engines"]["dinkster"]["commit"],
                "python": dinkster["python"],
                "torch": dinkster["torch"],
            },
        },
        "image": {**image, "pass": image_pass},
        "latent": {**latent, "pass": latent_pass},
        "overall_pass": mapping_pass and q8_pass and image_pass and latent_pass,
        "q8_dequantization": {
            "bit_exact": q8_pass,
            "complete": q8_keys_pass,
            "mismatched_keys": q8_mismatches,
            "tensor_count": len(comfy_q8),
        },
        "schema": 1,
        "tensor_mapping": {
            "exact": mapping_pass,
            "keys_unique": map_keys_unique,
            "tensor_count": len(comfy_map),
        },
        "understood_differences": [
            {
                "cause": (
                    "Dinkster and ComfyUI use independently ported SDXL modules whose CPU "
                    "float32 operation ordering produces bounded accumulation drift"
                ),
                "effect": {"image": image, "sampled_latent": latent},
                "id": "sdxl-module-float32-accumulation",
                "seams_proven_exact": ["Q8_0 dequantization", "tensor mapping"],
            },
            {
                "cause": (
                    "Dinkster predecodes GGUF weights once while ComfyUI-GGUF dequantizes "
                    "weights at each operation"
                ),
                "effect": "quantified by the paired warm performance receipt",
                "id": "predecode-versus-per-operation-dequantization",
                "seams_proven_exact": ["all Q8_0 realized float32 weights"],
            },
        ],
        "workload_id": workload["id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--dinkster-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_json(args.manifest)
        acceptance = load_json(args.acceptance)
        validate_manifest(manifest)
        validate_acceptance(acceptance)
        workload = next(item for item in manifest["workloads"] if item["id"] == args.workload)
        roots = {
            "artifact": args.artifact_root.resolve(),
            "comfyui": args.comfyui_root.resolve(),
            "dinkster": args.dinkster_root.resolve(),
            "template": args.dinkster_root.resolve(),
        }
        acceptance_digest = digest_bytes(canonical_bytes(acceptance))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        pins = {
            engine: validate_pins(workload, engine, roots, acceptance_digest)
            for engine in ("comfyui", "dinkster")
        }
        receipts = {}
        for engine in ("comfyui", "dinkster"):
            receipt = _run_adapter(workload, engine, roots, args.output_dir)
            receipts[engine] = receipt
            for dependency, expected in workload["engines"][engine]["dependencies"].items():
                if receipt.get(dependency) != expected:
                    raise HarnessError(
                        f"{engine} {dependency} mismatch: expected {expected}, "
                        f"got {receipt.get(dependency)}"
                    )
            if _git_output(roots[engine], "rev-parse", "HEAD") != pins[engine]["engine_commit"]:
                raise HarnessError(f"{engine} checkout changed during reference execution")
            if _git_output(roots[engine], "status", "--porcelain"):
                raise HarnessError(f"{engine} checkout became dirty during reference execution")
            validate_extensions_unchanged(pins[engine], engine, activity="reference execution")
        result = compare_reference(receipts["comfyui"], receipts["dinkster"], acceptance, workload)
        result["acceptance_manifest_digest"] = acceptance_digest
        result["pins"] = pins
        for engine in ("comfyui", "dinkster"):
            result["engines"][engine]["commit"] = pins[engine]["engine_commit"]
        result["logs"] = {
            engine: {
                kind: {
                    "digest": digest_file(args.output_dir / f"{engine}-reference.{kind}.txt"),
                    "path": f"{engine}-reference.{kind}.txt",
                }
                for kind in ("stderr", "stdout")
            }
            for engine in ("comfyui", "dinkster")
        }
        write_json(args.output_dir / "result.json", result)
        return 0 if result["overall_pass"] else 1
    except (HarnessError, StopIteration) as exc:
        print(f"GGUF reference comparison refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
