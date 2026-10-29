"""Generate variation-noise goldens from pinned ComfyUI-Inspire-Pack.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies plus opencv (Inspire's utils module
imports cv2 at module scope; the executed math is torch-only):

    COMFYUI_ROOT=/path/to/ComfyUI INSPIRE_ROOT=/path/to/ComfyUI-Inspire-Pack \
      /path/to/python tools/gen_inspire_variation_goldens.py

Both checkouts must be clean and pinned to the commits below. Run the
generator twice and compare the printed sha256 before committing the
fixture.

Every case executes the reference prepare_noise / mix_noise / slerp
from inspire/libs/utils.py on CPU float32 tensors. Norm/acos/sin
kernels dispatch on CPU microarchitecture, so the fixture records the
mint host's CPU and the test loader skips exact-equality replay on
other CPUs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

INSPIRE_BASELINE = "d23db9aa544de9a6d4c609cb7005fa9e0d42031d"
COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "inspire_variation_d23db9aa.json"
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
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


def _require_pin(root: Path, baseline: str, name: str) -> None:
    if _git(root, "rev-parse", "HEAD") != baseline:
        raise RuntimeError(f"{name} must be pinned to {baseline}")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError(f"{name} checkout must be clean")


def build_goldens(comfy_root: Path, inspire_root: Path) -> dict[str, object]:
    _require_pin(comfy_root, COMFY_BASELINE, "ComfyUI")
    _require_pin(inspire_root, INSPIRE_BASELINE, "ComfyUI-Inspire-Pack")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    # Golden math executes on CPU tensors; forcing CPU keeps the import
    # working on hosts whose torch build has no CUDA support.
    _comfy_args.cpu = True

    # Inspire's utils module uses flat imports (folder_paths resolves to
    # the ComfyUI root inserted above), so it loads by file path under a
    # private module name.
    spec = importlib.util.spec_from_file_location(
        "inspire_pinned_utils", inspire_root / "inspire" / "libs" / "utils.py"
    )
    assert spec is not None and spec.loader is not None
    inspire_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspire_utils)

    import torch  # pyright: ignore[reportMissingImports]

    def linspace(first: float, last: float, shape: tuple[int, ...]) -> Any:
        count = 1
        for size in shape:
            count *= size
        return torch.linspace(first, last, count, dtype=torch.float32).reshape(shape)

    def prepare_case(
        *,
        batch: int = 3,
        seed: int = 12345,
        noise_inds: list[int] | None = None,
        batch_seed_mode: str = "comfy",
        variation_seed: int = 777,
        variation_strength: float = 0.0,
        variation_method: str = "linear",
    ) -> dict[str, object]:
        latent = torch.zeros((batch, 4, 6, 8), dtype=torch.float32)
        result = inspire_utils.prepare_noise(
            latent,
            seed,
            noise_inds=noise_inds,
            noise_device="cpu",
            incremental_seed_mode=batch_seed_mode,
            variation_seed=variation_seed,
            variation_strength=variation_strength,
            variation_method=variation_method,
        )
        return {
            "inputs": {
                "batch": batch,
                "seed": seed,
                "noise_inds": noise_inds,
                "batch_seed_mode": batch_seed_mode,
                "variation_seed": variation_seed,
                "variation_strength": variation_strength,
                "variation_method": variation_method,
            },
            "result": _tensor_record(result),
        }

    prepare_cases = {
        "comfy-baseline": prepare_case(),
        "comfy-linear": prepare_case(variation_strength=0.35),
        "comfy-slerp": prepare_case(variation_strength=0.35, variation_method="slerp"),
        "comfy-inds-skip": prepare_case(batch=2, noise_inds=[2, 3]),
        "comfy-inds-shared": prepare_case(batch=3, noise_inds=[1, 1, 3]),
        "comfy-inds-variation-ignored": prepare_case(
            batch=2, noise_inds=[2, 3], variation_strength=0.6
        ),
        "incremental": prepare_case(batch_seed_mode="incremental"),
        "incremental-linear": prepare_case(batch_seed_mode="incremental", variation_strength=0.25),
        "varinc-005-linear": prepare_case(
            batch_seed_mode="variation str inc:0.05", variation_strength=0.1
        ),
        "varinc-001-slerp": prepare_case(
            batch_seed_mode="variation str inc:0.01",
            variation_strength=0.0,
            variation_method="slerp",
        ),
    }

    # Direct mix cases pin the slerp NaN guard (a zero-norm row is
    # unreachable through gaussian prepare_noise draws) and the linear
    # scale division on crafted operands.
    low_zero_row = linspace(-1.15, 1.3, (2, 4, 6, 8))
    low_zero_row[0] = 0.0
    high_zero_row = linspace(0.9, -1.05, (2, 4, 6, 8))
    high_zero_row[1] = 0.0
    mix_sources = {
        "low_zero_row": low_zero_row,
        "high_plain": linspace(1.2, -0.8, (2, 4, 6, 8)),
        "low_plain": linspace(-0.7, 1.05, (2, 4, 6, 8)),
        "high_zero_row": high_zero_row,
    }

    def mix_case(low: str, high: str, strength: float, variation_method: str) -> dict[str, object]:
        result = inspire_utils.mix_noise(
            mix_sources[low].clone(),
            mix_sources[high].clone(),
            strength,
            variation_method=variation_method,
        )
        return {
            "inputs": {
                "low": low,
                "high": high,
                "strength": strength,
                "variation_method": variation_method,
            },
            "result": _tensor_record(result),
        }

    mix_cases = {
        "slerp-low-guard": mix_case("low_zero_row", "high_plain", 0.4, "slerp"),
        "slerp-high-guard": mix_case("low_plain", "high_zero_row", 0.4, "slerp"),
        "linear-scaled": mix_case("low_plain", "high_plain", 0.3, "linear"),
    }

    document: dict[str, object] = {
        "inspire_baseline": INSPIRE_BASELINE,
        "comfy_baseline": COMFY_BASELINE,
        "mix_sources": {name: _tensor_record(tensor) for name, tensor in mix_sources.items()},
        "prepare_cases": prepare_cases,
        "mix_cases": mix_cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        document["_meta"] = provenance
    return document


def main() -> None:
    comfy_configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(comfy_configured) if comfy_configured else REPO.parent / "ComfyUI"
    inspire_configured = os.environ.get("INSPIRE_ROOT")
    inspire_root = (
        Path(inspire_configured) if inspire_configured else REPO.parent / "Inspire-pinned"
    )
    document = build_goldens(comfy_root.resolve(), inspire_root.resolve())
    import torch  # pyright: ignore[reportMissingImports]

    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(document, indent=2) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
