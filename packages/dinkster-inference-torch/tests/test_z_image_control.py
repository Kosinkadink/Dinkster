"""Standalone Z-Image control assembly contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference.z_image import z_image_control_layout
from dinkster_inference_torch import (
    ZImage,
    ZImageControl,
    assemble_z_image_control,
    select_attention,
)
from dinkster_inference_torch import assemble as assemble_module
from dinkster_inference_torch import z_image_control as control_module
from dinkster_inference_torch.module_residency import enroll_component


def test_z_image_control_assembly_uses_flux_attention_and_bf16(
    monkeypatch: Any,
) -> None:
    seen: dict[str, object] = {}

    def load(component: object, build: Any, *, compute_dtype: torch.dtype, **kwargs: object) -> Any:
        seen["component"] = component
        seen["dtype"] = compute_dtype
        seen["fp8_matmul"] = kwargs["fp8_matmul"]
        with torch.device("meta"):
            model = build(object(), operations=assemble_module.INITLESS)
        assert set(model.state_dict()) == set(z_image_control_layout())
        return model

    monkeypatch.setattr(assemble_module, "_load_component", load)
    component = object()
    assembled = assemble_z_image_control(
        cast(Any, SimpleNamespace(control=component, asset_digest="blake3:" + "0" * 64))
    )
    assert seen == {"component": component, "dtype": torch.bfloat16, "fp8_matmul": False}
    assert assembled.compute_dtype == torch.bfloat16
    assert assembled.attention_status.role == "flux"


def test_z_image_control_resource_accepts_residency_load_and_unload() -> None:
    model = ZImageControl.__new__(ZImageControl)
    torch.nn.Module.__init__(model)
    test_model = cast(Any, model)
    test_model.layer = assemble_module.INITLESS.linear(2, 2, bias=False)
    model.load_state_dict(
        {"layer.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        strict=True,
        assign=True,
    )
    digest = control_module.z_image_control_resource_digest("blake3:" + "0" * 64, torch.bfloat16)
    control_module._bind_z_image_control_resource(  # pyright: ignore[reportPrivateUsage]
        model, digest
    )
    original = test_model.layer.weight
    mechanism = enroll_component(
        model,
        load_device="cpu",
        offload_device="cpu",
        patch_set=PatchSet(
            {"layer.weight": (PatchEntry(DiffPatch(torch.ones_like(test_model.layer.weight))),)}
        ),
    )

    mechanism.partially_load(None)
    assert test_model.layer.weight is not original
    control_module.validate_z_image_control_resource(model, digest)

    mechanism.unload()
    assert test_model.layer.weight is original
    control_module.validate_z_image_control_resource(model, digest)


def test_z_image_control_matches_comfy_block_order_and_pads_odd_latents() -> None:
    events: list[str] = []
    seen: dict[str, object] = {}

    class Embed(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.width = width

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros(*value.shape[:-1], self.width, device=value.device)

    class Timestep(torch.nn.Module):
        def forward(self, value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(value.shape[0], 256, device=value.device, dtype=dtype)

    class Rope(torch.nn.Module):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(ids.shape[0], 1, ids.shape[1], 1, 2, device=ids.device)

    class Block(torch.nn.Module):
        def __init__(self, name: str, increment: float = 0.0) -> None:
            super().__init__()
            self.name = name
            self.increment = increment

        def forward(self, value: torch.Tensor, *_args: object) -> torch.Tensor:
            events.append(self.name)
            return value + self.increment

    class Final(torch.nn.Module):
        def forward(self, value: torch.Tensor, _modulation: torch.Tensor) -> torch.Tensor:
            return torch.zeros(value.shape[0], value.shape[1], 64, device=value.device)

    class Control:
        injection_blocks = (0, 5, 10, 15, 20, 25)

        def embed(self, latent: torch.Tensor) -> torch.Tensor:
            seen["control_shape"] = tuple(latent.shape)
            return torch.zeros(latent.shape[0], 32, 3840, device=latent.device)

        def refine(
            self,
            control: torch.Tensor,
            _rope: torch.Tensor,
            _modulation: torch.Tensor,
        ) -> torch.Tensor:
            events.extend(("refine0", "refine1"))
            return control

        def inject(
            self,
            index: int,
            control: torch.Tensor,
            base: torch.Tensor,
            _rope: torch.Tensor,
            _modulation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            events.append(f"inject{index}")
            if index == 0:
                seen["first_base"] = base.clone()
            return torch.zeros_like(control), control

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        del k, v, mask, causal, scale, enable_gqa
        return q

    with torch.device("meta"):
        model = ZImage(operations=assemble_module.INITLESS, attention_kernel=attention)
    test_model = cast(Any, model)
    test_model.cap_embedder = Embed(3840)
    test_model.x_embedder = Embed(3840)
    test_model.t_embedder = Timestep()
    test_model.rope_embedder = Rope()
    test_model.context_refiner = torch.nn.ModuleList([Block("context0"), Block("context1")])
    test_model.noise_refiner = torch.nn.ModuleList([Block("noise0", 1.0), Block("noise1", 1.0)])
    test_model.layers = torch.nn.ModuleList([Block(f"main{i}", 10.0) for i in range(30)])
    test_model.final_layer = Final()

    output = test_model(
        torch.zeros(1, 16, 7, 15),
        torch.zeros(1),
        torch.zeros(1, 32, 2560),
        control=Control(),
        control_latent=torch.zeros(1, 16, 7, 15),
        control_gains=(1.0,) * 6,
    )

    assert output.shape == (1, 16, 7, 15)
    assert seen["control_shape"] == (1, 16, 7, 15)
    torch.testing.assert_close(seen["first_base"], torch.full((1, 32, 3840), 2.0))
    assert events.count("refine0") == events.count("refine1") == 1
    assert events.index("noise0") < events.index("refine0")
    assert events.index("refine0") < events.index("refine1")
    assert events.index("refine1") < events.index("noise1")
    assert events.index("main0") < events.index("inject0")
    assert events.index("main5") < events.index("inject1")


def test_z_image_control_runs_noise_refiners_consecutively() -> None:
    events: list[str] = []

    class Block(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, value: torch.Tensor, *_args: object) -> torch.Tensor:
            events.append(self.name)
            return value + 1

    with torch.device("meta"):
        control = ZImageControl(
            operations=assemble_module.INITLESS,
            attention_kernel=select_attention("flux").kernel,
        )
    test_control = cast(Any, control)
    test_control.control_noise_refiner = torch.nn.ModuleList((Block("refine0"), Block("refine1")))

    output = test_control.refine(
        torch.zeros(1, 4, 8),
        torch.zeros(1, 1, 4, 1, 2),
        torch.zeros(1, 8),
    )

    assert events == ["refine0", "refine1"]
    torch.testing.assert_close(output, torch.full((1, 4, 8), 2.0))


def test_z_image_control_embed_circularly_pads_odd_latents() -> None:
    with torch.device("meta"):
        control = ZImageControl(
            operations=assemble_module.INITLESS,
            attention_kernel=select_attention("flux").kernel,
        )
    test_control = cast(Any, control)
    test_control.control_all_x_embedder = torch.nn.ModuleDict({"2-1": torch.nn.Linear(64, 2)})

    embedded = test_control.embed(torch.arange(16 * 7 * 15).view(1, 16, 7, 15).float())

    assert embedded.shape == (1, 32, 2)
