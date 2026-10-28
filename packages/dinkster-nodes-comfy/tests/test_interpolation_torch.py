"""Public CPU interpolation order, padding, progress and OOM recovery."""

from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.interpolation import FrameInterpolate
from dinkster_schema import use_reporter


class LinearInterpolation:
    pad_align = 4

    def __init__(self, mode: str) -> None:
        self.mode = mode
        if mode.startswith("multi"):
            self.forward_multi_timestep = self.multi

    def memory_used_forward(self, shape, dtype):
        return shape[1] * shape[2] * dtype.itemsize

    def extract_features(self, image):
        return image.mean()

    def __call__(self, first, second, *, timestep, cache):
        assert first.shape[2:] == (4, 8)
        assert torch.equal(cache["img0"], first[0].mean())
        assert torch.equal(cache["img1"], second[0].mean())
        if self.mode == "fatal" or ("oom" in self.mode and first.shape[0] > 1):
            raise torch.OutOfMemoryError("controlled interpolation allocation")
        return torch.lerp(first, second, timestep)

    def multi(self, first, second, times, *, cache):
        if self.mode == "multi-oom":
            raise torch.OutOfMemoryError("controlled multi-timestep allocation")
        return torch.cat([torch.lerp(first, second, t) for t in times])


@pytest.mark.parametrize("mode", ("single", "multi", "single-oom", "multi-oom"))
def test_frame_order_clamping_padding_progress_and_recovery(monkeypatch, mode: str) -> None:
    initialize_inference_process()
    from dinkster_inference import model_management

    monkeypatch.setattr(model_management, "load_models_gpu", lambda *args, **kwargs: None)
    model = LinearInterpolation(mode)
    patcher = SimpleNamespace(
        model=model, load_device=torch.device("cpu"), model_dtype=lambda: torch.float32
    )
    images = torch.linspace(-0.25, 1.25, 3 * 3 * 5 * 3).reshape(3, 3, 5, 3)
    before = images.clone()
    reports = []
    with use_reporter(lambda name, data, blob: reports.append((name, dict(data)))):
        output = FrameInterpolate.execute(interp_model=patcher, images=images, multiplier=4)[
            "IMAGE"
        ]
    expected = [images[0]]
    for index in range(2):
        expected.extend(torch.lerp(images[index], images[index + 1], t / 4) for t in (1, 2, 3))
        expected.append(images[index + 1])
    torch.testing.assert_close(output, torch.stack(expected).clamp(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(images, before, rtol=0, atol=0)
    progress = [data for name, data in reports if name == "progress"]
    assert progress[0] == {"step": 0, "total": 6, "text": "Frame interpolation"}
    assert progress[-1] == {"step": 6, "total": 6, "text": "Frame interpolation"}
    assert [p["step"] for p in progress] == sorted({p["step"] for p in progress})
    assert all(p["total"] == 6 for p in progress)


def test_single_frame_passes_through_and_minimum_batch_oom_propagates(monkeypatch) -> None:
    initialize_inference_process()
    from dinkster_inference import model_management

    images = torch.ones((1, 3, 5, 3)) * 1.25
    assert (
        FrameInterpolate.execute(interp_model=None, images=images, multiplier=2)["IMAGE"] is images
    )
    monkeypatch.setattr(model_management, "load_models_gpu", lambda *args, **kwargs: None)
    patcher = SimpleNamespace(
        model=LinearInterpolation("fatal"),
        load_device=torch.device("cpu"),
        model_dtype=lambda: torch.float32,
    )
    with pytest.raises(torch.OutOfMemoryError, match="controlled"):
        FrameInterpolate.execute(
            interp_model=patcher, images=images.repeat(2, 1, 1, 1), multiplier=2
        )
