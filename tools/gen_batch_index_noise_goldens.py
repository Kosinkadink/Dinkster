"""Generate per-batch noise-index goldens from pinned ComfyUI.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/python tools/gen_batch_index_noise_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the
fixture.

Cases execute the reference KSampler noise idiom: LatentFromBatch.frombatch
produces the latent's ``batch_index`` list and common_ksampler passes it to
comfy.sample.prepare_noise as batch_inds. frombatch only yields contiguous
index ranges, so the repeated- and gap-index cases call prepare_noise
directly with the index lists a latent dict can carry (any node may attach
``batch_index``; the reference draws shared noise for repeated indices and
draws-and-discards rows before skipped ones). The draws are plain
generator-fed randn with no microarchitecture-dispatched kernels, so the
fixture needs no CPU pin.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "batch_index_noise_b78cec87.json"
)


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_record(tensor: Any) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "values": [float(value) for value in cpu.reshape(-1).tolist()],
    }


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    # Golden math executes on CPU tensors; forcing CPU keeps the import
    # working on hosts whose torch build has no CUDA support.
    _comfy_args.cpu = True

    import torch  # pyright: ignore[reportMissingImports]
    from comfy import sample as comfy_sample  # pyright: ignore[reportMissingImports]
    from nodes import LatentFromBatch  # pyright: ignore[reportMissingImports]

    def frombatch_case(
        *, source_batch: int, batch_index: int, length: int, seed: int
    ) -> dict[str, object]:
        latent = {"samples": torch.zeros((source_batch, 4, 6, 8), dtype=torch.float32)}
        sliced = LatentFromBatch().frombatch(latent, batch_index, length)[0]
        noise_inds = sliced["batch_index"]
        noise = comfy_sample.prepare_noise(sliced["samples"], seed, noise_inds)
        return {
            "inputs": {
                "source_batch": source_batch,
                "batch_index": batch_index,
                "length": length,
                "seed": seed,
            },
            "noise_inds": list(noise_inds),
            "result": _tensor_record(noise),
        }

    def direct_case(*, noise_inds: list[int] | None, batch: int, seed: int) -> dict[str, object]:
        latent = torch.zeros((batch, 4, 6, 8), dtype=torch.float32)
        noise = comfy_sample.prepare_noise(latent, seed, noise_inds)
        return {
            "inputs": {"batch": batch, "seed": seed},
            "noise_inds": noise_inds,
            "result": _tensor_record(noise),
        }

    cases = {
        "frombatch-gap": frombatch_case(source_batch=6, batch_index=2, length=2, seed=7),
        "frombatch-single": frombatch_case(source_batch=4, batch_index=3, length=1, seed=7),
        "frombatch-full": frombatch_case(source_batch=3, batch_index=0, length=3, seed=99),
        "direct-shared": direct_case(noise_inds=[1, 1, 3], batch=3, seed=7),
        "direct-multi": direct_case(noise_inds=[0, 2, 2, 5], batch=4, seed=1234),
        "direct-none": direct_case(noise_inds=None, batch=6, seed=7),
    }

    document: dict[str, object] = {"comfy_baseline": BASELINE, "cases": cases}
    provenance = tuple_provenance(str(torch.__version__))
    if provenance:
        document["_meta"] = provenance
    return document


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    document = build_goldens(comfy_root.resolve())
    import torch  # pyright: ignore[reportMissingImports]

    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(document, indent=2) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
