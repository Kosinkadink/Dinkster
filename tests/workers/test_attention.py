from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))


def test_attention_runtime_resolves_upstream_named_attention() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from dinkster_workers import attention

runtime = attention.create_attention_runtime()
assert "sdpa" in runtime.capabilities.available_policies
selected = runtime.resolve(runtime.route_token)
q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10
k = q + 0.2
v = q - 0.1
expected = (
    torch.nn.functional.scaled_dot_product_attention(
        q.reshape(1, 3, 2, 4).transpose(1, 2),
        k.reshape(1, 3, 2, 4).transpose(1, 2),
        v.reshape(1, 3, 2, 4).transpose(1, 2),
    )
    .transpose(1, 2)
    .reshape(1, 3, 8)
)
torch.testing.assert_close(selected(q, k, v, 2), expected)
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
