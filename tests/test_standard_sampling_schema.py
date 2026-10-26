"""Torch-free schema contracts for the standard sampler execution nodes."""

from __future__ import annotations

import sys

from dinkster_native.standard_sampling import STANDARD_SAMPLING_NODES


def test_standard_sampler_nodes_preserve_the_generation_schema_contract() -> None:
    assert "torch" not in sys.modules
    standard, advanced = STANDARD_SAMPLING_NODES

    assert standard.schema().node_type == "dinkster.ksampler"
    assert [input.id for input in standard.schema().inputs] == [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "conditioning_batching",
        "max_fused_lanes",
    ]
    assert advanced.schema().node_type == "dinkster.ksampler_advanced"
    assert [input.id for input in advanced.schema().inputs] == [
        "model",
        "add_noise",
        "noise_seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise",
        "conditioning_batching",
        "max_fused_lanes",
    ]
