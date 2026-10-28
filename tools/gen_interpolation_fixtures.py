"""Generate CPU interpolation fixtures from the immutable ComfyUI reference.

Run with the CPU test interpreter: tools/gen_interpolation_fixtures.py REFERENCE_ROOT.
Only composite definitions execute; torch.nn supplies operations and every
parameter is replaced before execution. No pretrained weights are needed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REFERENCE_COMMIT = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"


def main() -> None:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        == REFERENCE_COMMIT
    )
    output = (
        Path(__file__).resolve().parents[1]
        / "packages/dinkster-nodes-comfy/tests/fixtures/interpolation"
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    records = {}
    arrays = {}
    images = torch.linspace(-0.125, 1.125, 3 * 64 * 64 * 3).reshape(3, 64, 64, 3)
    arrays["images"] = images.numpy()
    for kind, filename, class_name in (
        ("film", "film_net.py", "FILMNet"),
        ("rife", "ifnet.py", "IFNet"),
    ):
        path = root / "comfy_extras/frame_interpolation_models" / filename
        tree = ast.parse(path.read_text())
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Import)
                and any(alias.name == "comfy.ops" for alias in node.names)
                or isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ops" for t in node.targets)
            )
        ]
        scope = {"ops": torch.nn}
        exec(compile(tree, str(path), "exec"), scope)
        model = (
            scope[class_name]()
            if kind == "film"
            else scope[class_name](head_ch=6, channels=(8, 10, 12, 14, 16))
        )
        shapes = {key: list(value.shape) for key, value in model.state_dict().items()}
        generator = torch.Generator().manual_seed(82371)
        state = {
            key: torch.randn(shape, generator=generator) * 0.003 for key, shape in shapes.items()
        }
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            first, second = images[:1].movedim(-1, 1), images[1:2].movedim(-1, 1)
            arrays[kind] = model(first, second, timestep=0.375).numpy()
        records[kind] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shapes": shapes,
        }
    metadata = {
        "reference_commit": REFERENCE_COMMIT,
        "seed": 82371,
        "weight_scale": 0.003,
        "timestep": 0.375,
        "models": records,
    }
    (output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    np.savez_compressed(output / "outputs.npz", **arrays)
    for path in sorted(output.iterdir()):
        print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
