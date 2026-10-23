"""Snapshot ComfyUI core import schemas without retaining upstream executors.

Run in a CPU environment with the pinned checkout's requirements installed:

    python tools/gen_comfy_core_schemas.py --comfyui PATH

The checkout must be clean and at REFERENCE. Custom and partner API nodes
are not loaded. Run twice and compare the printed SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REFERENCE = "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", required=True, type=Path)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    root = args.comfyui.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REFERENCE:
        raise RuntimeError(f"ComfyUI must be at {REFERENCE}, not {head}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise RuntimeError("ComfyUI checkout must be clean")
    for source in sorted((ROOT / "packages").glob("*/src")):
        sys.path.insert(0, str(source))
    os.environ["DINKSTER_COMFYUI_ROOT"] = str(root)
    os.environ.pop("DINKSTER_COMFY_NODES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.argv = [sys.argv[0], "--cpu"]
    from dinkster_compat_comfy.bootstrap import load_comfyui_nodes
    from dinkster_schema import SCHEMA_WIRE_VERSION, schema_to_wire

    translation = load_comfyui_nodes()
    schemas = {
        node.schema().node_type: schema_to_wire(node.schema()) for node in translation.node_classes
    }
    for name in ("KSampler", "CLIPTextEncode", "CheckpointLoaderSimple", "VAEDecode"):
        if f"comfy.{name}" not in schemas:
            raise RuntimeError(f"core schema missing: {name}")
    payload = {
        "sourceRepository": "https://github.com/Comfy-Org/ComfyUI",
        "sourceCommit": head,
        "schemaWireVersion": SCHEMA_WIRE_VERSION,
        "schemas": schemas,
        "opaqueTypes": sorted(translation.opaque_types),
        "skipped": translation.skipped,
    }
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    args.output.write_bytes(data)
    print(f"{len(schemas)} schemas; sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
