"""CPU contracts for loading RAFT optical-flow assets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_schema_is_canonical_and_cold_imports_no_tensor_runtime(tmp_path: Path) -> None:

    root = Path(__file__).resolve().parents[3]
    python_path = os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )
    cold_discovery = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_video.optical_flow as module",
            "module.OpticalFlowLoader.schema()",
            "forbidden = sorted(",
            "    name for name in sys.modules",
            "    if name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            ")",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            cold_discovery,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": python_path,
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    discovery = json.loads(result.stdout)
    assert (
        Path(discovery["path"]).resolve()
        == root / "packages/dinkster-nodes-video/src/dinkster_video/optical_flow.py"
    )
    assert discovery["forbidden"] == []
