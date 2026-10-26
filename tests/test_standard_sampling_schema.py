"""Torch-free schema contracts for native standard sampler nodes."""

import subprocess
import sys

from dinkster_api.v1 import NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy.standard_sampling import (
    STANDARD_SAMPLING_NODES,
    KSampler,
    KSamplerAdvanced,
)


def test_standard_sampling_schemas_match_comfyui() -> None:
    standard, advanced = STANDARD_SAMPLING_NODES
    assert standard is KSampler
    assert advanced is KSamplerAdvanced
    assert standard.schema().node_type == "KSampler"
    assert advanced.schema().node_type == "KSamplerAdvanced"
    assert advanced.schema().display_name == "KSampler (Advanced)"
    assert standard.schema().category == advanced.schema().category == "model/sampling"
    assert tuple(input.id for input in standard.schema().inputs) == (
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
    )
    assert tuple(input.id for input in advanced.schema().inputs) == (
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
    )
    seed = standard.schema().inputs[1].widget
    noise_seed = advanced.schema().inputs[2].widget
    cfg = standard.schema().inputs[3].widget
    denoise = standard.schema().inputs[-1].widget
    assert isinstance(seed, NumberWidget) and seed.min == 0 and seed.max == 0xFFFFFFFFFFFFFFFF
    assert isinstance(noise_seed, NumberWidget)
    assert seed.step is None and noise_seed.step is None
    assert seed.control_after_generate == noise_seed.control_after_generate == "fixed"
    for index in (3, 10, 11):
        widget = advanced.schema().inputs[index].widget
        assert isinstance(widget, NumberWidget) and widget.step is None
    assert isinstance(cfg, NumberWidget) and (cfg.min, cfg.max, cfg.step, cfg.round) == (
        0.0,
        100.0,
        0.1,
        0.01,
    )
    assert isinstance(denoise, NumberWidget) and (denoise.min, denoise.max, denoise.step) == (
        0.0,
        1.0,
        0.01,
    )
    assert standard.schema().outputs[0].id == advanced.schema().outputs[0].id == "LATENT"
    assert standard.schema().outputs[0].doc == "The denoised latent."
    for node in STANDARD_SAMPLING_NODES:
        assert schema_from_wire(schema_to_wire(node.schema())) == node.schema()


def test_standard_sampling_cold_discovery_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from dinkster_nodes_comfy.standard_sampling import STANDARD_SAMPLING_NODES; "
            "[node.schema() for node in STANDARD_SAMPLING_NODES]; "
            "assert 'torch' not in sys.modules; "
            "assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
