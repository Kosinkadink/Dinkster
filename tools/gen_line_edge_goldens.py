"""Generate learned line/edge goldens from pinned comfyui_controlnet_aux nodes.

Usage from the Dinkster repository root with CUDA hidden::

    CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python tools/gen_line_edge_goldens.py \
        .e2e/comfyui_controlnet_aux .e2e/ComfyUI .e2e/ckpts

Run the command twice and compare its printed SHA-256 before committing a
refreshed fixture.
"""

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
from typing import Any

import cv2
import numpy as np
import skimage
import torch

CONTROLNET_AUX_COMMIT = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
COMFYUI_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
SOURCE_SHA256 = "28e2ffe0c96d7c7d44c45ff10c6754ef9741c638caa06c619774f82a0d4e12c5"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "line_edge_controlnet_aux_59b1fc4.json"

ARTIFACTS = {
    "lllyasviel/Annotators/ControlNetHED.pth": (
        29_444_406,
        "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa",
    ),
    "lllyasviel/Annotators/sk_model.pth": (
        17_173_511,
        "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    ),
    "lllyasviel/Annotators/sk_model2.pth": (
        17_173_511,
        "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
    ),
    "lllyasviel/Annotators/netG.pth": (
        217_631_959,
        "ccabdcc3f5cf3c07cf65d58776acb21df7dfda825cdc70c9766a93fd62bfc488",
    ),
    "lllyasviel/Annotators/erika.pth": (
        172_789_563,
        "badbd6baf013cefbd98993307b02cc14a26c770d067416e4fdecc8720b88feeb",
    ),
    "lllyasviel/Annotators/mlsd_large_512_fp32.pth": (
        6_341_481,
        "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
    ),
    "bdsqlsz/qinglong_controlnet-lllite/Annotators/7_model.pth": (
        247_232,
        "b9037964149c55156c6adbffdfbd7e8ca7d2ef2a4d90573520efa7f3a1aacf06",
    ),
    "TheMistoAI/MistoLine/Anyline/MTEED.pth": (
        248_298,
        "a3c2d8a8ce9422555c787160bd46362d761325a565333c0e3f6a53e0bae2abdb",
    ),
}

CASES: dict[str, tuple[str, dict[str, object]]] = {
    "lineart-realistic": ("LineArtPreprocessor", {"coarse": "disable"}),
    "lineart-realistic-coarse": ("LineArtPreprocessor", {"coarse": "enable"}),
    "lineart-anime": ("AnimeLineArtPreprocessor", {}),
    "lineart-manga": ("Manga2Anime_LineArt_Preprocessor", {}),
    "hed-soft": ("HEDPreprocessor", {"safe": "disable"}),
    "hed-safe": ("HEDPreprocessor", {"safe": "enable"}),
    "hed-scribble": ("FakeScribblePreprocessor", {"safe": "enable"}),
    "hed-scribble-unsafe": ("FakeScribblePreprocessor", {"safe": "disable"}),
    "teed": ("TEEDPreprocessor", {"safe_steps": 2}),
    "teed-unquantized": ("TEEDPreprocessor", {"safe_steps": 0}),
    "mlsd": ("M-LSDPreprocessor", {"score_threshold": 0.1, "dist_threshold": 0.1}),
    "mlsd-empty": ("M-LSDPreprocessor", {"score_threshold": 2.0, "dist_threshold": 20.0}),
    "anyline-standard": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_standard"},
    ),
    "anyline-realistic": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_realisitic"},
    ),
    "anyline-anime": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_anime"},
    ),
    "anyline-manga": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "manga_line"},
    ),
}


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _check_checkout(path: Path, expected: str, name: str) -> None:
    actual = _git(path, "rev-parse", "HEAD")
    if actual != expected:
        raise SystemExit(f"{name} must be at {expected}, got {actual}")
    if _git(path, "status", "--porcelain"):
        raise SystemExit(f"{name} checkout must be clean")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_artifacts(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for relative, (size, expected) in ARTIFACTS.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or _sha256(path) != expected:
            raise SystemExit(f"artifact does not match pinned bytes: {path}")
        records[relative] = {"sizeBytes": size, "sha256": expected}
    return records


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes()
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "zlibBase64": base64.b64encode(zlib.compress(payload, level=9)).decode("ascii"),
    }


def _execute(module: Any, image: torch.Tensor) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for name, (node_type, inputs) in CASES.items():
        node_class = module.NODE_CLASS_MAPPINGS[node_type]
        node = node_class()
        execute = getattr(node, node_class.FUNCTION)
        output = execute(image=image, resolution=256, **inputs)[0]
        outputs[name] = np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    reference = args.reference.resolve()
    comfyui = args.comfyui.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    _check_checkout(reference, CONTROLNET_AUX_COMMIT, "comfyui_controlnet_aux")
    _check_checkout(comfyui, COMFYUI_COMMIT, "ComfyUI")
    artifacts = _check_artifacts(checkpoint_root)
    source_path = reference / "examples" / "example_anyline.png"
    if _sha256(source_path) != SOURCE_SHA256:
        raise SystemExit("official example input does not match the pinned source revision")
    if torch.cuda.is_available():
        raise SystemExit("goldens must be generated with CUDA unavailable")

    os.environ["AUX_ANNOTATOR_CKPTS_PATH"] = str(checkpoint_root)
    os.environ["AUX_USE_SYMLINKS"] = "False"
    sys.path.insert(0, str(comfyui))
    sys.path.insert(0, str(reference.parent))
    sys.path.insert(0, str(reference / "src"))
    sys.argv = [sys.argv[0], "--cpu"]
    comfy_options = importlib.import_module("comfy.options")
    comfy_options.enable_args_parsing()
    module = importlib.import_module("comfyui_controlnet_aux")

    from PIL import Image

    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
    image = torch.from_numpy(source.copy()[None].astype(np.float32) / 255.0)
    outputs = _execute(module, image)
    document = {
        "controlnetAuxCommit": CONTROLNET_AUX_COMMIT,
        "comfyuiCommit": COMFYUI_COMMIT,
        "device": "cpu",
        "resolution": 256,
        "source": {"path": "examples/example_anyline.png", **_record(source)},
        "artifacts": artifacts,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikitImage": skimage.__version__,
            "torch": torch.__version__,
        },
        "cases": {name: _record(output) for name, output in outputs.items()},
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"{args.output}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
