"""Surface extraction, variable-length batches and the frozen wire schema."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import initialize_inference_process

SplatToMesh = importlib.import_module("dinkster_3d.splat_mesh").SplatToMesh


@pytest.mark.parametrize(
    ("accelerator_budget", "color_sharpen"),
    [(None, 1.0), (None, 2.0), (0, 1.0), (0, 2.0), (2_900_000, 2.0)],
)
def test_splat_mesh_counts_filter_padding_and_preserve_linear_color(
    monkeypatch: pytest.MonkeyPatch, accelerator_budget: int | None, color_sharpen: float
) -> None:
    initialize_inference_process()
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    # CPU:0 simulates an accelerator whose allocator enforces the reported budget.
    # The 65^3 grid fits 2.9 MB without, but not with, the sharpened-color normalizer.
    device = torch.device("cpu" if accelerator_budget is None else "cpu:0")
    monkeypatch.setattr(management, "get_torch_device", lambda: device)
    if accelerator_budget is not None:

        def is_cpu(dev: Any) -> bool:
            return dev == torch.device("cpu")

        def free_memory(dev: Any) -> int:
            return accelerator_budget

        monkeypatch.setattr(management, "is_device_cpu", is_cpu)
        monkeypatch.setattr(management, "get_free_memory", free_memory)
        zeros = torch.zeros
        allocated = 0

        def budgeted_zeros(*size: Any, **kwargs: Any) -> Any:
            nonlocal allocated
            value = zeros(*size, **kwargs)
            if kwargs.get("device") == device:
                allocated += value.numel() * value.element_size()
                if allocated > accelerator_budget:
                    raise torch.OutOfMemoryError("simulated accelerator budget exceeded")
            return value

        monkeypatch.setattr(torch, "zeros", budgeted_zeros)
    positions = torch.tensor([[[0.3, 0.5, -0.7], [20, 20, 20]]] * 2)
    splat = {
        "positions": positions,
        "scales": torch.full_like(positions, 0.2),
        "rotations": torch.tensor([[[1.0, 0, 0, 0]]] * 2).expand(2, 2, 4),
        "opacities": torch.tensor([[[1.0], [1.0]], [[0.001], [1.0]]]),
        "sh": torch.tensor(
            [[[[(0.8 - 0.5) / 0.28209479177387814, 0.0, -0.5 / 0.28209479177387814]]]] * 2
        ).expand(2, 2, 1, 3),
        "counts": torch.tensor([1, 1]),
    }
    mesh: Any = SplatToMesh.execute(
        splat=splat,
        resolution=64,
        kernel=8,
        min_component=0,
        min_opacity=0.02,
        color_sharpen=color_sharpen,
        smooth=1,
    )["mesh"]
    assert mesh.unlit is True
    assert int(mesh.vertex_counts[0]) > 0 and int(mesh.face_counts[0]) > 0
    assert int(mesh.vertex_counts[1]) == 0 and int(mesh.face_counts[1]) == 0
    verts = mesh.vertices[0, : mesh.vertex_counts[0]].numpy()
    # The output rotates 180 degrees about X, and padding must not affect the extent.
    np.testing.assert_allclose(
        (verts.min(0) + verts.max(0)) / 2, [0.3, -0.5, 0.7], atol=0.03, rtol=0
    )
    assert np.max(np.abs(verts)) < 2
    faces = mesh.faces[0, : mesh.face_counts[0]]
    assert int(faces.min()) >= 0 and int(faces.max()) < int(mesh.vertex_counts[0])
    color = mesh.vertex_colors[0, : mesh.vertex_counts[0]].numpy()
    expected = [((0.8 + 0.055) / 1.055) ** 2.4, ((0.5 + 0.055) / 1.055) ** 2.4, 0]
    np.testing.assert_allclose(color, np.broadcast_to(expected, color.shape), atol=0.001, rtol=0)
    removed: Any = SplatToMesh.execute(splat=splat, resolution=64, kernel=8, min_component=100000)[
        "mesh"
    ]
    assert removed.vertex_counts.tolist() == [0, 0]
    assert removed.face_counts.tolist() == [0, 0]
