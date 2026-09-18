"""Generate noise-node goldens from pinned ComfyUI-KJNodes 3f200542.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI KJNODES_ROOT=/path/to/ComfyUI-KJNodes \
      /path/to/python tools/gen_kj_noise_goldens.py

Both checkouts must be clean and pinned to the commits below (ComfyUI is
needed only to satisfy the KJNodes module imports; the executed math is
torch-only). Run the generator twice and compare the printed sha256 before
committing the fixture.

The executed kernels (randn draws are seed-deterministic, but std
reductions and bilinear interpolation dispatch on CPU microarchitecture)
drift by ULPs across CPU classes, so the fixture records the mint host's
CPU and the test loader skips exact-equality replay on other CPUs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

KJ_BASELINE = "3f20054214fec9f9234fd3841ae6f1e4287948f6"
COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "kj_noise_3f200542.json"
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


def build_goldens(comfy_root: Path, kj_root: Path) -> dict[str, object]:
    _require_pin(comfy_root, COMFY_BASELINE, "ComfyUI")
    _require_pin(kj_root, KJ_BASELINE, "ComfyUI-KJNodes")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    # Golden math executes on CPU tensors; forcing CPU keeps the import
    # working on hosts whose torch build has no CUDA support.
    _comfy_args.cpu = True

    # KJNodes' nodes/nodes.py is loaded by file path under a private module
    # name: its "from nodes import MAX_RESOLUTION" must resolve to ComfyUI's
    # root nodes.py, which the plain module name would shadow.
    spec = importlib.util.spec_from_file_location(
        "kjnodes_pinned_nodes", kj_root / "nodes" / "nodes.py"
    )
    assert spec is not None and spec.loader is not None
    kj_nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kj_nodes)

    import torch  # pyright: ignore[reportMissingImports]

    def linspace(first: float, last: float, shape: tuple[int, ...]) -> Any:
        count = 1
        for size in shape:
            count *= size
        return torch.linspace(first, last, count, dtype=torch.float32).reshape(shape)

    sources = {
        "latent_a": linspace(-1.2, 1.35, (2, 4, 6, 8)),
        "noise_a": linspace(0.8, -1.05, (2, 4, 6, 8)),
        "latent_batch3": linspace(-1.1, 1.2, (3, 4, 6, 8)),
        "noise_batch3": linspace(0.55, -0.85, (3, 4, 6, 8)),
        "mask_small": linspace(0.0, 1.0, (1, 5, 7)),
        "mask_batch2": linspace(0.1, 0.9, (2, 4, 6)),
    }

    generate = kj_nodes.GenerateNoise()
    inject = kj_nodes.InjectNoiseToLatent()

    generate_defaults: dict[str, object] = {
        "width": 64,
        "height": 48,
        "batch_size": 2,
        "seed": 123,
        "multiplier": 1.0,
        "constant_batch_noise": False,
        "normalize": False,
        "latent_channels": "4",
        "shape": "BCHW",
    }

    sigma_values = [12.25, 5.5, 0.75]
    scale_factor = 0.13025

    def generate_case(
        *, sigmas: list[float] | None = None, **overrides: object
    ) -> dict[str, object]:
        inputs = {**generate_defaults, **overrides}
        kwargs: dict[str, Any] = dict(inputs)
        if sigmas is not None:
            kwargs["sigmas"] = torch.tensor(sigmas, dtype=torch.float32)
            kwargs["model"] = SimpleNamespace(
                model=SimpleNamespace(latent_format=SimpleNamespace(scale_factor=scale_factor))
            )
        result = generate.generatenoise(**kwargs)[0]["samples"]
        case: dict[str, object] = {"inputs": inputs, "result": _tensor_record(result)}
        if sigmas is not None:
            case["sigmas"] = sigmas
            case["scale_factor"] = scale_factor
        return case

    generate_cases = {
        "default": generate_case(),
        "seed777": generate_case(seed=777),
        "multiplier": generate_case(multiplier=2.5),
        "normalize": generate_case(normalize=True),
        "constant-batch": generate_case(constant_batch_noise=True, batch_size=3),
        "channels16": generate_case(latent_channels="16"),
        "bcthw": generate_case(shape="BCTHW", batch_size=3),
        "btchw": generate_case(shape="BTCHW", batch_size=3),
        "sigma": generate_case(sigmas=sigma_values),
        "sigma-combined": generate_case(sigmas=sigma_values, multiplier=0.5, normalize=True),
    }

    inject_defaults: dict[str, object] = {
        "strength": 0.1,
        "normalize": False,
        "average": False,
        "mix_randn_amount": 0.0,
        "seed": 123,
    }

    def inject_case(
        *,
        latents: str = "latent_a",
        noise: str = "noise_a",
        mask: str | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        inputs = {**inject_defaults, **overrides}
        result = inject.injectnoise(
            latents={"samples": sources[latents].clone()},
            noise={"samples": sources[noise].clone()},
            mask=None if mask is None else sources[mask].clone(),
            **inputs,
        )[0]["samples"]
        return {
            "inputs": inputs,
            "latents": latents,
            "noise": noise,
            "mask": mask,
            "result": _tensor_record(result),
        }

    inject_cases = {
        "default": inject_case(),
        "strength": inject_case(strength=1.5),
        "average": inject_case(average=True),
        "normalize": inject_case(strength=0.75, normalize=True),
        "mask": inject_case(mask="mask_small"),
        "mask-batch-repeat": inject_case(
            latents="latent_batch3", noise="noise_batch3", mask="mask_batch2"
        ),
        "mix-randn": inject_case(mix_randn_amount=0.35, seed=456),
        "combined": inject_case(
            strength=1.2, normalize=True, mask="mask_small", mix_randn_amount=0.2, seed=789
        ),
    }

    document: dict[str, object] = {
        "kj_baseline": KJ_BASELINE,
        "comfy_baseline": COMFY_BASELINE,
        "sources": {name: _tensor_record(tensor) for name, tensor in sources.items()},
        "generate_cases": generate_cases,
        "inject_cases": inject_cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        document["_meta"] = provenance
    return document


def main() -> None:
    comfy_configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(comfy_configured) if comfy_configured else REPO.parent / "ComfyUI"
    kj_configured = os.environ.get("KJNODES_ROOT")
    kj_root = Path(kj_configured) if kj_configured else REPO.parent / "KJNodes-pinned"
    document = build_goldens(comfy_root.resolve(), kj_root.resolve())
    import torch  # pyright: ignore[reportMissingImports]

    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(document, indent=2) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
