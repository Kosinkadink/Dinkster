r"""Generate Wan ATI trajectory goldens by executing pinned ComfyUI.

Run with the ComfyUI interpreter so its declared dependencies are available::

    F:\workspaces\station1\stations\station14\installs\ComfyUI\.venv\Scripts\python.exe \
        tools\gen_wan_ati_goldens.py

The generator exports ComfyUI commit b78cec87 to a temporary directory and
executes its Wan track preprocessing and latent motion projection on CPU.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from golden_platform import platform_golden_path, tuple_provenance

COMFY_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
WIDTH = 32
HEIGHT = 16
LENGTH = 17
TEMPERATURE = 7.5
TOPK = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comfy_root() -> Path:
    configured = os.environ.get("DINKSTER_COMFYUI_ROOT")
    candidates = (
        *((Path(configured).resolve(),) if configured else ()),
        _repo_root().parent / "ComfyUI",
        _repo_root().parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("set DINKSTER_COMFYUI_ROOT to a ComfyUI checkout")


def _export(repo: Path, destination: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", COMFY_COMMIT],
        text=True,
    ).strip()
    if revision != COMFY_COMMIT:
        raise RuntimeError(f"resolved ComfyUI revision {revision}, expected {COMFY_COMMIT}")
    archive = destination / "comfy.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(archive), revision],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()


def _tensor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.flatten().tolist(),
    }


def _track_json() -> str:
    tracks = [
        [{"x": 3.0 + index * 0.15, "y": 2.0 + index * 0.07} for index in range(121)],
        [{"x": 28.0 - index * 0.11, "y": 13.0 - index * 0.05} for index in range(121)],
    ]
    return json.dumps(tracks, separators=(",", ":"))


def _build_payload(source: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy_extras.nodes_wan import (
        pad_pts,
        parse_json_tracks,
        patch_motion,
        process_tracks,
    )

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    track_json = _track_json()
    track_data = parse_json_tracks(track_json)
    if isinstance(track_data[0][0], dict):
        track_data = [track_data]
    processed = []
    for batch in track_data:
        arrays = [pad_pts(track) for track in batch]
        processed.append(
            process_tracks(
                np.stack(arrays, axis=0),
                (WIDTH, HEIGHT),
                LENGTH - 1,
            ).unsqueeze(0)
        )
    video = torch.linspace(
        -1.25,
        1.75,
        16 * ((LENGTH - 1) // 4 + 1) * (HEIGHT // 8) * (WIDTH // 8),
        dtype=torch.float32,
    ).reshape(1, 16, (LENGTH - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
    mask, feature = patch_motion(
        processed,
        video,
        temperature=TEMPERATURE,
        topk=TOPK,
        vae_divide=(4, 16),
    )
    return {
        "reference": {
            "repository": "https://github.com/Comfy-Org/ComfyUI",
            "commit": COMFY_COMMIT,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": sys.platform,
            **tuple_provenance(str(torch.__version__)),
        },
        "case": {
            "tracks": track_json,
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "batch_size": 1,
            "temperature": TEMPERATURE,
            "topk": TOPK,
            "processed_tracks": [_tensor(value) for value in processed],
            "video": _tensor(video),
            "mask": _tensor(mask),
            "feature": _tensor(feature),
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dinkster-wan-ati-golden-") as directory:
        source = Path(directory)
        _export(_comfy_root(), source)
        payload = _build_payload(source)
    output = platform_golden_path(
        _repo_root()
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "wan_ati_goldens.json",
        str(torch.__version__),
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
