"""CPU contracts and host transport for native WanDancer video conditioning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import register_inference_types as register_inference_types


def test_schema_and_cold_source_tree_discovery() -> None:

    root = Path(__file__).resolve().parents[3]
    python_path = os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )
    script = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_wan.wandancer_video as module",
            "module.WanDancerVideo.define_schema()",
            "forbidden = sorted(name for name in sys.modules if (",
            "    name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            "))",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    discovery = json.loads(result.stdout)
    assert Path(discovery["path"]).resolve() == (
        root / "packages/dinkster-nodes-wan/src/dinkster_wan/wandancer_video.py"
    )
    assert discovery["forbidden"] == []
