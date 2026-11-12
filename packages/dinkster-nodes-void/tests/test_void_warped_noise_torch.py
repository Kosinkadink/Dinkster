"""CPU contracts for VOID optical-flow-warped latents."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dinkster_void import register_types as register_types


def _python_path(root: Path) -> str:
    return os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )


def test_schema_is_canonical_and_cold_import_avoids_tensor_runtime(tmp_path: Path) -> None:

    root = Path(__file__).resolve().parents[3]
    discovery = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_void.void_warped_noise as module",
            "module.VOIDWarpedNoise.schema()",
            "forbidden = sorted(name for name in sys.modules if (",
            "    name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            "))",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", discovery],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": _python_path(root)},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["path"]).resolve() == (
        root / "packages/dinkster-nodes-void/src/dinkster_void/void_warped_noise.py"
    )
    assert payload["forbidden"] == []
