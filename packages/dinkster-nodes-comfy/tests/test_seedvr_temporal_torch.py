"""SeedVR2 temporal chunking and merging."""

import dinkster_nodes_comfy.seedvr_temporal as seedvr_temporal
import torch
from dinkster_nodes_comfy.seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge


def _latent(frames: int = 7) -> dict[str, object]:
    return {
        "samples": torch.arange(16 * frames * 2 * 2, dtype=torch.float64).reshape(
            1, 16, frames, 2, 2
        ),
        "metadata": {"source": "seedvr"},
        "noise_mask": torch.zeros(1),
    }


def test_manual_chunks_and_merges_with_runt_final_chunk(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    source = _latent()
    chunks = SeedVR2TemporalChunk.execute(
        latent=source,
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    assert [chunk["samples"].shape[2] for chunk in chunks["latents"]] == [4, 4, 3]
    assert chunks["temporal_overlap"] == 2
    merged = SeedVR2TemporalMerge.execute(
        latents=chunks["latents"], temporal_overlap=chunks["temporal_overlap"]
    )["latent"]
    assert merged["samples"].shape == source["samples"].shape
    assert merged["samples"].dtype == source["samples"].dtype
    assert merged["metadata"] == source["metadata"]
    assert "noise_mask" not in merged


def test_hann_crossfade_weights_overlap(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    first = {"samples": torch.zeros((1, 16, 3, 1, 1), dtype=torch.float32)}
    second = {"samples": torch.ones((1, 16, 3, 1, 1), dtype=torch.float32)}
    merged = SeedVR2TemporalMerge.execute(latents=[first, second], temporal_overlap=2)["latent"][
        "samples"
    ]
    assert torch.equal(merged[:, :, 0], first["samples"][:, :, 0])
    assert torch.equal(merged[:, :, -1], second["samples"][:, :, -1])
    assert torch.allclose(merged[:, :, 1], torch.zeros((1, 16, 1, 1)))
    assert torch.allclose(merged[:, :, 2], torch.ones((1, 16, 1, 1)))


def test_auto_chunking_uses_free_memory_boundary(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    source = _latent(8)

    class Management:
        @staticmethod
        def get_torch_device() -> str:
            return "cpu"

        @staticmethod
        def get_free_memory(device: str) -> int:
            assert device == "cpu"
            return int((8.5 + 4 * 0.55 + 0.55 * (2 * 2 * 64 / 1e6) * 2.1) * 1024**3)

    original = seedvr_temporal.importlib.import_module

    def import_module(name: str):
        return Management if name == "dinkster_inference.model_management" else original(name)

    monkeypatch.setattr(seedvr_temporal.importlib, "import_module", import_module)
    chunks = SeedVR2TemporalChunk.execute(
        latent=source, temporal_overlap=0, chunking_mode={"chunking_mode": "auto"}
    )
    assert [chunk["samples"].shape[2] for chunk in chunks["latents"]] == [2, 2, 2, 2]
