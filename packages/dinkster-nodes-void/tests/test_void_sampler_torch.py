"""CPU behavior for the native VOID DDIM sampler."""

from __future__ import annotations

import os
import subprocess
import sys

from dinkster_api.v1 import register_inference_types as register_inference_types


def test_void_sampler_schema_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_void.void_sampler import VOIDSampler; "
            "VOIDSampler.schema(); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr
