"""Generate pose-rendering goldens from comfyui_controlnet_aux 59b1fc4.

Usage:
    .venv/bin/python tools/gen_pose_goldens.py /path/to/comfyui_controlnet_aux
"""

from __future__ import annotations

import argparse
import ast
import base64
import colorsys
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import cv2
import numpy as np

BASELINE = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "pose_controlnet_aux_59b1fc4.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _load_reference(reference: Path) -> tuple[Any, Any, Any, Any]:
    if _git(reference, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"controlnet_aux must be checked out at {BASELINE}")
    if _git(reference, "status", "--short"):
        raise RuntimeError("controlnet_aux checkout must be clean")

    package = ModuleType("custom_controlnet_aux")
    dwpose = ModuleType("custom_controlnet_aux.dwpose")
    body = ModuleType("custom_controlnet_aux.dwpose.body")
    keypoint_type = namedtuple("Keypoint", "x y score id", defaults=(1.0, -1))
    body_result_type = namedtuple(
        "BodyResult", "keypoints total_score total_parts", defaults=(0.0, 0)
    )
    pose_result_type = namedtuple("PoseResult", "body left_hand right_hand face")
    body.Keypoint = keypoint_type  # type: ignore[attr-defined]
    body.BodyResult = body_result_type  # type: ignore[attr-defined]
    package.__path__ = []  # type: ignore[attr-defined]
    dwpose.__path__ = []  # type: ignore[attr-defined]
    sys.modules[package.__name__] = package
    sys.modules[dwpose.__name__] = dwpose
    sys.modules[body.__name__] = body

    matplotlib = ModuleType("matplotlib")
    matplotlib_colors = ModuleType("matplotlib.colors")
    matplotlib_colors.hsv_to_rgb = lambda value: np.asarray(  # type: ignore[attr-defined]
        colorsys.hsv_to_rgb(*value)
    )
    matplotlib.colors = matplotlib_colors  # type: ignore[attr-defined]
    sys.modules[matplotlib.__name__] = matplotlib
    sys.modules[matplotlib_colors.__name__] = matplotlib_colors

    util_path = reference / "src" / "custom_controlnet_aux" / "dwpose" / "util.py"
    spec = importlib.util.spec_from_file_location("custom_controlnet_aux.dwpose.util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {util_path}")
    util = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = util
    spec.loader.exec_module(util)

    init_path = reference / "src" / "custom_controlnet_aux" / "dwpose" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    names = {"draw_animalposes", "draw_animalpose", "draw_poses"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise RuntimeError("pinned pose rendering functions were not found")
    namespace = {
        "List": list,
        "Keypoint": keypoint_type,
        "PoseResult": pose_result_type,
        "cv2": cv2,
        "np": np,
        "util": util,
    }
    statements: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(init_path), "exec"), namespace)
    return (
        namespace["draw_poses"],
        namespace["draw_animalposes"],
        keypoint_type,
        pose_result_type,
    )


def _flat(points: Sequence[tuple[float, float] | None]) -> list[float]:
    return [
        value
        for point in points
        for value in ((point[0], point[1], 1.0) if point is not None else (0.0, 0.0, 0.0))
    ]


def _source() -> dict[str, dict[str, object]]:
    body = [
        (0.50, 0.10),
        (0.50, 0.25),
        (0.35, 0.27),
        (0.27, 0.42),
        (0.20, 0.58),
        (0.65, 0.27),
        (0.73, 0.42),
        (0.80, 0.58),
        (0.42, 0.52),
        (0.40, 0.70),
        (0.38, 0.90),
        (0.58, 0.52),
        (0.60, 0.70),
        (0.62, 0.90),
        (0.46, 0.08),
        (0.54, 0.08),
        (0.42, 0.10),
        None,
    ]
    hand = [(0.20, 0.58)] + [
        (0.16 + finger * 0.025, 0.55 - joint * (0.035 + finger * 0.002))
        for finger in range(5)
        for joint in range(1, 5)
    ]
    face = [
        (0.44, 0.08),
        (0.46, 0.06),
        (0.48, 0.05),
        (0.50, 0.05),
        (0.52, 0.05),
        (0.54, 0.06),
        (0.56, 0.08),
    ]
    animal = [
        (18, 10),
        (25, 8),
        (31, 12),
        (35, 23),
        (32, 34),
        (43, 20),
        (51, 26),
        (57, 34),
        (44, 30),
        (50, 42),
        (55, 53),
        (28, 32),
        (25, 44),
        (22, 56),
        (38, 34),
        (37, 46),
        (36, 58),
    ]
    return {
        "human": {
            "canvas_width": 80,
            "canvas_height": 64,
            "people": [
                {
                    "pose_keypoints_2d": _flat(body),
                    "face_keypoints_2d": _flat(face),
                    "hand_left_keypoints_2d": _flat(hand),
                    "hand_right_keypoints_2d": None,
                }
            ],
        },
        "animal": {
            "version": "ap10k",
            "canvas_width": 72,
            "canvas_height": 64,
            "animals": [[list(point) + [1.0] for point in animal]],
        },
    }


def _points(raw: object, keypoint_type: Any) -> list[object | None] | None:
    if raw is None:
        return None
    values = cast("list[float]", raw)
    return [
        keypoint_type(values[index], values[index + 1]) if values[index + 2] >= 1.0 else None
        for index in range(0, len(values), 3)
    ]


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def build_goldens(reference: Path) -> dict[str, object]:
    draw_poses, draw_animals, keypoint_type, pose_result_type = _load_reference(reference)
    sources = _source()
    human = cast("dict[str, object]", sources["human"])
    person = cast("dict[str, object]", cast("list[object]", human["people"])[0])
    body_result_type = namedtuple(
        "BodyResult", "keypoints total_score total_parts", defaults=(0.0, 0)
    )
    pose = pose_result_type(
        body_result_type(_points(person["pose_keypoints_2d"], keypoint_type)),
        _points(person["hand_left_keypoints_2d"], keypoint_type),
        _points(person["hand_right_keypoints_2d"], keypoint_type),
        _points(person["face_keypoints_2d"], keypoint_type),
    )
    human_output = draw_poses(
        [pose], cast("int", human["canvas_height"]), cast("int", human["canvas_width"])
    )

    animal = cast("dict[str, object]", sources["animal"])
    animal_rows = cast("list[list[list[float]]]", animal["animals"])
    animals = [[keypoint_type(*point[:2]) for point in row] for row in animal_rows]
    animal_output = draw_animals(
        animals, cast("int", animal["canvas_height"]), cast("int", animal["canvas_width"])
    )
    return {
        "baseline": BASELINE,
        "opencv": cv2.__version__,
        "cases": {
            "human": {"source": human, "output": _record(human_output)},
            "animal": {"source": animal, "output": _record(animal_output)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_goldens(args.reference.resolve()), indent=2, sort_keys=True) + "\n"
    OUT.write_text(payload, encoding="utf-8")
    print(hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
