"""Generate Wav2Vec2-large goldens by executing pinned ComfyUI b78cec87.

The source tree is materialized from the exact git object with ``git archive``.
Set ``DINKSTER_WAN22_S2V_AUDIO_ENCODER`` to the official
``wav2vec2_large_english_fp16.safetensors`` artifact and run from the Dinkster
root with Python 3.12 and torch available:

    python tools/gen_wav2vec2_goldens.py

Set ``COMFYUI_ROOT`` when ComfyUI is not beside Dinkster (or beside its parent,
as in station delegate clones).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
ARTIFACT_SIZE = 630_997_322
ARTIFACT_SHA256 = "f0017a43ea57ef6b3d4866be607844bbd8cada6d30966f7d70044ed0d63d3f9e"
OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "wav2vec2_goldens.json"


def _comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "ComfyUI",
        REPO.parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set COMFYUI_ROOT to the ComfyUI git checkout")


def _artifact() -> Path:
    configured = os.environ.get("DINKSTER_WAN22_S2V_AUDIO_ENCODER")
    if configured is None:
        raise SystemExit("set DINKSTER_WAN22_S2V_AUDIO_ENCODER to the official artifact")
    path = Path(configured).resolve()
    if not path.is_file() or path.stat().st_size != ARTIFACT_SIZE:
        raise SystemExit("Wav2Vec2 artifact is missing or has the wrong byte size")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != ARTIFACT_SHA256:
        raise SystemExit("Wav2Vec2 artifact SHA256 differs from the official artifact")
    return path


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _enc(value: Any) -> dict[str, Any]:
    import torch

    assert isinstance(value, torch.Tensor)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.float().flatten().tolist(),
    }


def main() -> None:
    comfy_root = _comfy_root()
    artifact = _artifact()
    resolved = _git(comfy_root, "rev-parse", REFERENCE_COMMIT).decode().strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {REFERENCE_COMMIT}")

    archive = _git(comfy_root, "archive", "--format=tar", REFERENCE_COMMIT)
    with tempfile.TemporaryDirectory(prefix="dinkster-wav2vec2-reference-") as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
            payload.extractall(source, filter="data")
        sys.path.insert(0, str(source))
        sys.argv.extend(("--cpu", "--use-pytorch-cross-attention"))

        import comfy.options

        comfy.options.enable_args_parsing()

        import comfy.ops
        import comfy.utils
        import torch
        from comfy.audio_encoders.wav2vec2 import Wav2Vec2Model

        state = comfy.utils.load_torch_file(str(artifact), safe_load=True)
        state = comfy.utils.state_dict_prefix_replace(state, {"wav2vec2.": ""})
        encoder = Wav2Vec2Model(
            embed_dim=1024,
            num_heads=16,
            num_layers=24,
            conv_norm=True,
            conv_bias=True,
            do_normalize=True,
            do_stable_layer_norm=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
            operations=comfy.ops.manual_cast,
        ).eval()
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        assert not missing
        assert set(unexpected) == {"lm_head.bias", "lm_head.weight"}
        waveform = torch.linspace(-1.0, 1.0, 6_400, dtype=torch.float32).reshape(1, 1, -1)
        with torch.inference_mode():
            encoded_audio, layers = encoder(waveform)
        assert len(layers) == 25
        sample_indices = (
            torch.linspace(
                0,
                layers[0].numel() - 1,
                64,
                dtype=torch.float64,
            )
            .round()
            .to(torch.int64)
        )
        payload = {
            "_meta": {
                "reference": "ComfyUI Wav2Vec2Model",
                "commit": REFERENCE_COMMIT,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "source": "git archive of exact commit object",
                "attention": "--use-pytorch-cross-attention",
                "artifact_size": ARTIFACT_SIZE,
                "artifact_sha256": ARTIFACT_SHA256,
            },
            "input_shape": list(waveform.shape),
            "audio_samples": waveform.shape[2],
            "layer_shape": list(layers[0].shape),
            "sample_indices": sample_indices.tolist(),
            "layer_samples": [
                layer.flatten().index_select(0, sample_indices).float().tolist() for layer in layers
            ],
            "output": _enc(encoded_audio),
        }
        OUT.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            newline="\n",
        )


if __name__ == "__main__":
    main()
