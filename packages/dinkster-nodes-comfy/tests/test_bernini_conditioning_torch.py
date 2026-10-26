"""CPU behavior for Bernini conditioning."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_nodes_comfy.bernini_conditioning import BerniniConditioning


class ControlledVAE:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(image.clone())
        return image.mean(dim=(0, 1, 2), keepdim=True)


def test_bernini_encodes_ordered_context_and_preserves_sources(monkeypatch: Any) -> None:
    import dinkster_nodes_comfy.bernini_conditioning as module

    monkeypatch.setattr(module, "initialize_inference_process", lambda: None)
    management = type("Management", (), {"intermediate_device": staticmethod(lambda: "cpu")})
    actual_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name, package=None: (
            management
            if name == "dinkster_inference.model_management"
            else actual_import(name, package)
        ),
    )
    vae = ControlledVAE()
    positive = [[torch.tensor([1.0]), {"source": "positive"}]]
    negative = [[torch.tensor([2.0]), {"source": "negative"}]]
    source = torch.arange(3 * 9 * 13 * 4, dtype=torch.float32).reshape(3, 9, 13, 4)
    reference_video = torch.arange(5 * 10 * 20 * 4, dtype=torch.float32).reshape(5, 10, 20, 4)
    references = {
        "reference_image_10": torch.full((1, 12, 6, 4), 10.0),
        "reference_image_2": torch.full((2, 7, 19, 4), 2.0),
        "reference_image_0": None,
    }
    result = BerniniConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=32,
        length=2,
        batch_size=2,
        source_video=source.numpy(),
        reference_video=reference_video.numpy(),
        reference_images=references,
        ref_max_size=16,
    )
    context = result["positive"][0][1]["context_latents"]
    assert positive[0][1] == {"source": "positive"}
    assert negative[0][1] == {"source": "negative"}
    assert result["positive"][0][0] is positive[0][0]
    assert result["negative"][0][0] is negative[0][0]
    assert len(context) == 5
    assert [tuple(image.shape) for image in vae.inputs] == [
        (2, 32, 16, 3),
        (2, 16, 16, 3),
        (1, 16, 16, 3),
        (1, 16, 16, 3),
        (1, 16, 16, 3),
    ]
    assert result["latent"]["samples"].shape == (2, 16, 1, 4, 2)
    no_context = BerniniConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=16,
        height=16,
        length=5,
        batch_size=1,
    )
    assert no_context["positive"] is positive
    assert no_context["negative"] is negative
    assert no_context["latent"]["samples"].shape == (1, 16, 2, 2, 2)
