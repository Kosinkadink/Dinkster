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


def test_native_sampling_choices_match_the_installed_upstream_engine() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
from dinkster_inference_wire.process import initialize_inference_process
from dinkster_sampling.standard_sampling import KSampler
from dinkster_schema import ComboOption, ComboWidget

initialize_inference_process()
samplers = importlib.import_module("dinkster_inference.samplers")
schema = KSampler.schema()
for name, expected in (
    ("sampler_name", samplers.KSampler.SAMPLERS),
    ("scheduler", samplers.KSampler.SCHEDULERS),
):
    spec = schema.input(name)
    assert spec is not None and isinstance(spec.widget, ComboWidget)
    assert all(isinstance(option, ComboOption) for option in spec.widget.options)
    assert [option.value for option in spec.widget.options] == list(expected)
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
