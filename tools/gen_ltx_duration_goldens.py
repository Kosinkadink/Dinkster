"""Generate LTX duration-head goldens from pinned ComfyUI.

Usage:
    .venv-torch/bin/python tools/gen_ltx_duration_goldens.py \
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
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/ltx_duration_goldens.json"


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
    import torch
    from comfy.ldm.lightricks.duration_head import DurationHead, seconds_to_num_frames
    from unet_fill import fill_state_dict, hashed_input

    module_path = Path(sys.modules[DurationHead.__module__].__file__ or "").resolve()
    if not module_path.is_relative_to(root):
        raise SystemExit(f"reference module imported from {module_path}")
    if torch.__version__ != GENERATOR_TORCH:
        raise SystemExit(
            f"goldens require torch {GENERATOR_TORCH}; this interpreter has {torch.__version__}"
        )
    module = DurationHead()
    entries = [(key, list(value.shape)) for key, value in module.state_dict().items()]
    module.load_state_dict(fill_state_dict(entries), strict=True)

    cases: dict[str, object] = {}
    for name, use_video, use_audio in (
        ("audio_video", True, True),
        ("video_only", True, False),
        ("audio_only", False, True),
    ):
        video = hashed_input(f"{name}:video", (2, 3, 4096)) if use_video else None
        audio = hashed_input(f"{name}:audio", (2, 4, 2048)) if use_audio else None
        with torch.no_grad():
            output = module(video, audio)
        cases[name] = {"output": output.tolist()}

    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "generator": "tools/gen_ltx_duration_goldens.py",
            "platform": sys.platform,
            "python": platform.python_version(),
            "torch": torch.__version__,
            **tuple_provenance(torch.__version__),
        },
        "state_dict": entries,
        "cases": cases,
        "frames": {
            "below_minimum": seconds_to_num_frames(0.1, 24.0, 1.0, 20.0),
            "inside": seconds_to_num_frames(4.2, 24.0, 1.0, 20.0),
            "above_maximum": seconds_to_num_frames(40.0, 24.0, 1.0, 20.0),
        },
    }
    out = platform_golden_path(OUT, torch.__version__)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
