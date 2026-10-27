"""CPU process initialization and ordinary latent-node hosting."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import materialize_latent
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_comfy.mochi import EmptyMochiLatentVideo

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("accelerator", ("cpu", "auto"))
def test_fresh_cpu_process_initializes_once(accelerator: str) -> None:
    script = """
import sys
from concurrent.futures import ThreadPoolExecutor
from dinkster_api.v1 import initialize_inference_process
assert 'torch' not in sys.modules
assert 'dinkster_inference.model_management' not in sys.modules
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda _: initialize_inference_process(), range(8)))
import torch
from dinkster_inference import model_management as memory
from dinkster_inference.cli_args import args
assert memory.get_torch_device().type == 'cpu'
assert memory.intermediate_device().type == 'cpu'
assert torch.zeros((2, 3), device=memory.intermediate_device()).sum() == 0
args.cpu = False
initialize_inference_process()
assert args.cpu is False, 'initialized process policy was rewritten'
assert memory.get_torch_device().type == 'cpu'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": accelerator},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mochi_temporal_boundary_and_asymmetric_dimensions() -> None:
    for length, frames in ((12, 2), (13, 3)):
        result = EmptyMochiLatentVideo.execute(width=40, height=24, length=length, batch_size=2)
        samples = result["LATENT"]["samples"]
        assert samples.shape == (2, 12, frames, 3, 5)
        assert samples.device.type == "cpu"
        assert samples.dtype == torch.float32
        assert not samples.count_nonzero()


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_creates_image_and_mochi_latents_twice(in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"}
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "image": GraphNode(
                        "EmptyLatentImage", {"width": 40, "height": 24, "batch_size": 2}
                    ),
                    "mochi": GraphNode(
                        "EmptyMochiLatentVideo",
                        {"width": 40, "height": 24, "length": 13, "batch_size": 2},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["image", "mochi"])
                for node, shape in (("image", (2, 4, 3, 5)), ("mochi", (2, 12, 3, 3, 5))):
                    latent = materialize_latent(result.outputs[node]["LATENT"].resolve())
                    samples = latent["samples"]
                    assert samples.shape == shape
                    assert samples.dtype == torch.float32
                    assert samples.device.type == "cpu"
                    assert not samples.count_nonzero()
                assert latent.keys() == {"samples"}
        finally:
            await composer.close()

    asyncio.run(scenario())
