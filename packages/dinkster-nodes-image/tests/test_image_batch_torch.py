"""Public CPU image dimension and rebatch behavior."""

from __future__ import annotations

import os
import subprocess
import sys


def test_empty_image_initializes_cpu_policy_in_a_cold_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_image.ported.image_core import EmptyImage\n"
            "assert 'dinkster_inference.model_management' not in sys.modules\n"
            "image = EmptyImage.execute(\n"
            "    width=73, height=41, batch_size=2, color=0x32B76D)['IMAGE']\n"
            "assert image.device.type == 'cpu'\n"
            "assert tuple(image.shape) == (2, 41, 73, 3)\n"
            "import torch\n"
            "expected = torch.tensor([50 / 255, 183 / 255, 109 / 255], dtype=image.dtype)\n"
            "torch.testing.assert_close(image, expected.expand_as(image), rtol=0, atol=0)\n",
        ],
        env={**os.environ, "DINKSTER_ACCELERATOR": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
