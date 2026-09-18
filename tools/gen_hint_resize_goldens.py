"""Generate hint-resize parity vectors with pinned comfyui_controlnet_aux."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASELINE = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
REPO = Path(__file__).resolve().parent.parent
BASE_OUT = REPO / "tests" / "goldens" / "hint_resize_controlnet_aux_59b1fc4.json"


def _out_path() -> Path:
    """Linux mints the base fixture; other platforms mint a variant keyed by
    the libraries that execute the resize kernels (cv2) and the rounding/cast
    (numpy), since those drift by one LSB across platform builds."""
    if sys.platform.startswith("linux"):
        return BASE_OUT
    key = f"{sys.platform}-numpy{np.__version__}-opencv{cv2.__version__}"
    return BASE_OUT.with_name(f"{BASE_OUT.stem}.{key}{BASE_OUT.suffix}")


def _continuous_source() -> np.ndarray:
    rows, columns, channels = np.indices((65, 83, 3))
    return ((columns * 17 + rows * 29 + channels * 67 + (columns * rows) % 251) % 256).astype(
        np.uint8
    )


def _categorical_source(source: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(source // 64 * 64, dtype=np.uint8)


def _binary_source() -> np.ndarray:
    source = np.zeros((65, 83, 3), dtype=np.uint8)
    cv2.line(source, (3, 61), (79, 4), (255, 255, 255), 1)
    cv2.circle(source, (41, 32), 18, (255, 255, 255), 1)
    return source


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"controlnet_aux must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("controlnet_aux checkout must be clean")


def _load_reference(reference: Path) -> Any:
    package_name = "_dinkster_controlnet_aux_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(reference)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    log_module = types.ModuleType(f"{package_name}.log")
    log_module.log = object()  # type: ignore[attr-defined]
    sys.modules[log_module.__name__] = log_module

    class ResizeMode(Enum):
        RESIZE = "Just Resize"
        INNER_FIT = "Crop and Resize"
        OUTER_FIT = "Resize and Fill"

    utils_module = types.ModuleType(f"{package_name}.utils")
    utils_module.ResizeMode = ResizeMode  # type: ignore[attr-defined]
    utils_module.safe_numpy = lambda value: np.ascontiguousarray(value.copy())  # type: ignore[attr-defined]

    def get_unique_axis0(data: np.ndarray) -> np.ndarray:
        array = np.asanyarray(data)
        indexes = np.lexsort(array.T)
        array = array[indexes]
        unique = np.empty(len(array), dtype=np.bool_)
        unique[:1] = True
        unique[1:] = np.any(array[:-1] != array[1:], axis=-1)
        return array[unique]

    utils_module.get_unique_axis0 = get_unique_axis0  # type: ignore[attr-defined]
    sys.modules[utils_module.__name__] = utils_module
    sys.modules.setdefault("torch", types.ModuleType("torch"))

    module_name = f"{package_name}.hint_image_enchance"
    spec = importlib.util.spec_from_file_location(module_name, reference / "hint_image_enchance.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned hint resize implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.HintImageEnchance()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    _check_reference(reference)
    resize = _load_reference(reference)

    continuous = _continuous_source()
    categorical = _categorical_source(continuous)
    binary = _binary_source()
    cases = {
        "continuous_stretch": resize.execute_resize(continuous, 96, 80),
        "continuous_fill": resize.execute_inner_fit(continuous, 96, 80),
        "continuous_fit": resize.execute_outer_fit(continuous, 96, 80),
        "categorical_stretch": resize.execute_resize(categorical, 96, 80),
        "binary_stretch": resize.execute_resize(binary, 166, 130),
    }
    document = {
        "baseline": BASELINE,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "sources": {
            "continuous": _record(continuous),
            "categorical": _record(categorical),
            "binary": _record(binary),
        },
        "cases": {name: _record(value) for name, value in cases.items()},
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    out = _out_path()
    out.write_bytes(data)
    print(f"{out}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
