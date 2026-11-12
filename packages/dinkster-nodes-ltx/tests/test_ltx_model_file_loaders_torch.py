"""CPU checks for bound LTX audio and text-encoder assets."""

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_assets import MountDef, MountTable
from safetensors.torch import load_file, save_file


@pytest.mark.parametrize("device", (None, "default", "cpu", "other"))
def test_ltx_text_encoder_loader_preserves_file_order_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str | None
) -> None:
    from dinkster_ltx import model_file_loaders as loaders

    initialize_inference_process()
    from dinkster_inference import sd

    encoder, checkpoint = tmp_path / "encoder.safetensors", tmp_path / "checkpoint.safetensors"
    save_file({"marker": torch.tensor([3.0])}, encoder)
    save_file({"marker": torch.tensor([-7.0])}, checkpoint)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="checkpoint", path=tmp_path), kind="model/checkpoint")
    table.add(MountDef(id="encoder", path=tmp_path), kind="model/text-encoder")

    def load(*, ckpt_paths: list[str], clip_type: object, **options: Any) -> tuple[float, ...]:
        assert clip_type == sd.CLIPType.LTXV
        expected = (
            {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
            if device == "cpu"
            else {}
        )
        assert options["model_options"] == expected
        return tuple(float(load_file(path)["marker"][0]) for path in ckpt_paths)

    monkeypatch.setattr(sd, "load_clip", load)
    result = loaders.LTXAVTextEncoderLoader.execute(
        text_encoder=table.bind_name(encoder.name, "model/text-encoder"),
        ckpt_name=table.bind_name(checkpoint.name, "model/checkpoint"),
        **({"device": device} if device is not None else {}),
    )
    assert result == {"CLIP": (3.0, -7.0)}
