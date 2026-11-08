from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dinkster.compose import compose_serving, default_pack_spec
from tests.async_scenarios import run_scenario

torch = pytest.importorskip("torch")


def test_first_party_generation_packs_compose_with_cpu_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.version.cuda is not None:
        cpu_runtime = tmp_path / "cpu-runtime"
        requirements = [
            f"{name}=={importlib.metadata.version(name).split('+')[0]}+cpu"
            for name in ("torch", "torchvision")
        ]
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(cpu_runtime),
                "--no-deps",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                *requirements,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_first_party_generation_packs_compose_with_cpu_torch",
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(cpu_runtime), os.environ.get("PYTHONPATH", ""))),
                "DINKSTER_ACCELERATOR": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
            },
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()

    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")

    async def scenario() -> None:
        composition = await compose_serving(
            (
                default_pack_spec("dinkster-models"),
                default_pack_spec("dinkster-conditioning"),
                default_pack_spec("dinkster-latent"),
                default_pack_spec("dinkster-sampling"),
            ),
            include_default_packs=False,
        )
        try:
            assert {
                "dinkster.models.checkpoint_loader_simple",
                "dinkster.conditioning.clip_text_encode",
                "dinkster.latent.vae_decode",
                "dinkster.sampling.k_sampler",
            } <= composition.schemas.keys()
        finally:
            await composition.close()

    run_scenario(scenario())
