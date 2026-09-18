"""Generate a tiny LTX diffusion video VAE decoder golden from ComfyUI.

Usage:
    .venv-torch/bin/python tools/gen_ltx_diffusion_vae_goldens.py \
        --comfy-root /path/to/ComfyUI

Darwin fixtures use Python 3.12.11 and torch 2.13.0 in a platform-tuple file.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from golden_platform import platform_golden_path, tuple_provenance

REFERENCE_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
GENERATOR_TORCH = "2.13.0" if sys.platform == "darwin" else "2.13.0+cpu"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/ltx_diffusion_vae_goldens.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.comfy_root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"reference is at {commit}; required {REFERENCE_COMMIT}")
    if dirty:
        raise SystemExit(f"reference checkout must be clean:\n{dirty}")

    sys.path.insert(0, str(REPO / "packages/dinkster-inference-torch/tests"))
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()

    import torch
    from comfy.ldm.lightricks.vae.na_diffusion_decoder import NADiffusionDecoder
    from unet_fill import fill_state_dict, hashed_input

    module_path = Path(sys.modules[NADiffusionDecoder.__module__].__file__ or "").resolve()
    if not module_path.is_relative_to(root):
        raise SystemExit(f"reference module imported from {module_path}")
    if torch.__version__ != GENERATOR_TORCH:
        raise SystemExit(
            f"goldens require torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
        )
    model = NADiffusionDecoder(
        in_channels=2,
        out_channels=1,
        patch_size=1,
        head_dim=4,
        stage_channels=(4, 4, 4, 4, 4),
        stage_depths=(1, 0, 0, 0, 1),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        stage5_kernel=(1, 1, 1),
        t_emb_dim=4,
    )
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    latent = hashed_input("ltx-diffusion-vae:latent", (1, 2, 2, 2, 2))
    with torch.no_grad():
        context = model.forward_pre_diffusion(latent)
        output = model(latent, generator=torch.Generator().manual_seed(13))

    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "generator": "tools/gen_ltx_diffusion_vae_goldens.py",
            "platform": sys.platform,
            "python": platform.python_version(),
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "state_dict": entries,
        "context": context.tolist(),
        "output": output.tolist(),
    }
    out = platform_golden_path(OUT, torch.__version__)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
