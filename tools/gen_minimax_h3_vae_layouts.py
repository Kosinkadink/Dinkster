"""Generate the torch-free MiniMax H3 codec layout tables.

Writes the exact key -> shape mapping of the production-default video
and audio VAE modules so split-checkpoint planning can validate codec
sources without constructing torch modules:

- packages/dinkster-inference/src/dinkster_inference/data/minimax_h3_video_vae_layout.json.gz
- packages/dinkster-inference/src/dinkster_inference/data/minimax_h3_audio_vae_layout.json.gz

Run under the torch environment:

    .venv-torch/bin/python tools/gen_minimax_h3_vae_layouts.py

Prints the sha256 of each UNCOMPRESSED payload; those pins live next to
the loaders in dinkster_inference/minimax_h3_codecs.py.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import torch
from dinkster_inference_torch.minimax_h3_audio import MiniMaxH3AudioVAE
from dinkster_inference_torch.minimax_h3_video_vae import MiniMaxH3VideoVAE


def _layout(module: torch.nn.Module) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in module.state_dict().items()}


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    data = repo / "packages/dinkster-inference/src/dinkster_inference/data"
    with torch.device("meta"):
        layouts = {
            "minimax_h3_video_vae_layout.json.gz": _layout(MiniMaxH3VideoVAE()),
            "minimax_h3_audio_vae_layout.json.gz": _layout(MiniMaxH3AudioVAE()),
        }
    for name, layout in layouts.items():
        raw = json.dumps(layout, separators=(",", ":")).encode()
        out = data / name
        out.write_bytes(gzip.compress(raw, mtime=0))
        print(f"{out}: {len(layout)} keys, sha256 {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
