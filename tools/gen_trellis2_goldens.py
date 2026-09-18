#!/usr/bin/env python3
"""Generate TRELLIS.2 flow goldens by executing the pinned Microsoft source."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, cast

import torch

REFERENCE_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
GENERATOR_TORCH = "2.6.0+cu124"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "TRELLIS.2"
DEFAULT_OUTPUT = (
    ROOT
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "trellis2_microsoft_source.json"
)

SOURCE_CONFIG: dict[str, object] = {
    "resolution": 16,
    "in_channels": 32,
    "out_channels": 32,
    "model_channels": 12,
    "cond_channels": 8,
    "num_blocks": 1,
    "num_heads": 2,
    "mlp_ratio": 4,
    "pe_mode": "rope",
    "share_mod": True,
    "initialization": "scaled",
    "qk_rms_norm": True,
    "qk_rms_norm_cross": True,
    "dtype": "float32",
}


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes()


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(state[key]))
    return digest.hexdigest()


def _source_revision(source: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _generate(source: Path) -> dict[str, object]:
    revision = _source_revision(source)
    if revision != REFERENCE_COMMIT:
        raise RuntimeError(f"TRELLIS.2 source must be at {REFERENCE_COMMIT}, found {revision}")
    if torch.__version__ != GENERATOR_TORCH:
        raise RuntimeError(f"generator requires torch {GENERATOR_TORCH}, found {torch.__version__}")

    os.environ["ATTN_BACKEND"] = "naive"
    sys.path.insert(0, str(ROOT / "packages" / "dinkster-inference-torch" / "tests"))
    sys.path.insert(0, str(source))
    fill = importlib.import_module("unet_fill")
    source_module = importlib.import_module("trellis2.models.sparse_structure_flow")
    source_type = cast(Any, source_module.SparseStructureFlowModel)

    model = source_type(**SOURCE_CONFIG)
    entries = [(key, list(value.shape)) for key, value in model.named_parameters()]
    state = cast(dict[str, torch.Tensor], fill.fill_state_dict(entries))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["rope_phases"] or unexpected:
        raise RuntimeError(
            f"unexpected source state contract: missing={missing}, unexpected={unexpected}"
        )

    latent = cast(
        torch.Tensor,
        fill.hashed_input("trellis2:dense:latent", (1, 32, 16, 16, 16)),
    )
    timestep = torch.tensor([731.25], dtype=torch.float32)
    context = cast(
        torch.Tensor,
        fill.hashed_input("trellis2:dense:context", (1, 7, 8)),
    )
    with torch.no_grad():
        output = cast(torch.Tensor, model(latent, timestep, context))

    output_bytes = _tensor_bytes(output)
    compressed = zlib.compress(output_bytes, level=9)
    return {
        "format": 1,
        "source": {
            "repository": "https://github.com/microsoft/TRELLIS.2",
            "commit": revision,
            "module": "trellis2.models.sparse_structure_flow.SparseStructureFlowModel",
        },
        "generator": {"torch": torch.__version__, "attention_backend": "naive"},
        "config": SOURCE_CONFIG,
        "inputs": {
            "latent_key": "trellis2:dense:latent",
            "latent_shape": list(latent.shape),
            "latent_sha256": _tensor_digest(latent),
            "timestep": 731.25,
            "context_key": "trellis2:dense:context",
            "context_shape": list(context.shape),
            "context_sha256": _tensor_digest(context),
        },
        "state": {
            "parameter_count": len(entries),
            "sha256": _state_digest(state),
        },
        "output": {
            "shape": list(output.shape),
            "dtype": "float32",
            "raw_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "encoding": "base64+zlib",
            "data": base64.b64encode(compressed).decode("ascii"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = _generate(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
