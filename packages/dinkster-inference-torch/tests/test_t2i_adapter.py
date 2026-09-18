"""SD1.5 full T2I Adapter strict loading and residual mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dinkster_inference import (
    FLOAT32,
    FLOAT64,
    ComponentPlan,
    ControlApplication,
    PayloadReference,
    PercentRange,
    SD15T2IAdapterConfig,
    sd15_t2i_adapter_layout,
)
from dinkster_inference.assembly import T2IAdapterAssemblyPlan
from dinkster_inference_torch import SDControlConditioning, assemble_sd15_t2i_adapter
from dinkster_inference_torch.controlnet import sd_control_hint_digest
from dinkster_inference_torch.operations import bound_compute_dtype
from dinkster_inference_torch.t2i_adapter import SD15T2IAdapter

CHECKPOINT = Path(
    r"C:\Users\kosin\comfy-vibe-station\pr-tracker\stations\station2\ComfyUI"
    r"\models\controlnet\t2iadapter_canny_sd15v2.pth"
)
ASSET_DIGEST = "blake3:0c483d0094b18c9f542ab7586869e52c6d410c48941513bcd0b156d63c5ae5ac"


def _plan(path: Path) -> T2IAdapterAssemblyPlan:
    state = torch.load(path, map_location="cpu", weights_only=True)
    keys = tuple(state)
    component = ComponentPlan(
        "t2i_adapter",
        path,
        SD15T2IAdapterConfig(),
        {key: key for key in keys},
        {key: FLOAT32 for key in keys},
        {},
    )
    return T2IAdapterAssemblyPlan(component, ASSET_DIGEST, tuple(sorted(keys)))


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="official local fixture is unavailable")
def test_official_checkpoint_strict_loads_and_is_sealed() -> None:
    assembled = assemble_sd15_t2i_adapter(_plan(CHECKPOINT), adapter_dtype=torch.float32)
    assert type(assembled.adapter) is SD15T2IAdapter
    assert assembled.resource_digest == assembled.adapter.resource_digest
    assert len(assembled.adapter.state_dict()) == 38
    hint = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 1, 64, 64)
    with torch.no_grad():
        residuals = assembled.adapter(hint)
    expected = {
        2: (684.36865234375, -0.024283111095428467),
        5: (-2714.114013671875, -10.007241249084473),
        8: (-7971.71875, -10.905508041381836),
        11: (-1944.7042236328125, -29.281890869140625),
    }
    for index, (total, first) in expected.items():
        assert float(residuals.down[index].sum()) == pytest.approx(total, abs=1e-5)
        assert float(residuals.down[index].flatten()[0]) == pytest.approx(first, abs=1e-6)
    assert all(
        bool(torch.count_nonzero(residuals.down[index]) == 0)
        for index in set(range(12)) - set(expected)
    )
    assert bool(torch.count_nonzero(residuals.middle) == 0)
    hint_digest = sd_control_hint_digest(hint)
    conditioning = SDControlConditioning(
        ControlApplication(
            "canny-adapter",
            PayloadReference(hint_digest),
            1.0,
            PercentRange(0.0, 1.0),
        ),
        assembled.adapter,
        hint,
        assembled.resource_digest,
        hint_digest,
    )
    assert conditioning.model is assembled.adapter
    with torch.no_grad():
        next(assembled.adapter.parameters()).add_(1)
    from dinkster_inference_torch.t2i_adapter import validate_sd15_t2i_adapter_resource

    with pytest.raises(ValueError, match="provenance"):
        validate_sd15_t2i_adapter_resource(assembled.adapter, assembled.resource_digest)


def test_float64_storage_uses_the_resolved_adapter_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = sd15_t2i_adapter_layout()
    state = {
        key: torch.empty(shape, dtype=torch.float64, device="meta") for key, shape in layout.items()
    }
    component = ComponentPlan(
        "t2i_adapter",
        Path("float64-adapter.pth"),
        SD15T2IAdapterConfig(),
        {key: key for key in layout},
        {key: FLOAT64 for key in layout},
        {},
    )
    plan = T2IAdapterAssemblyPlan(component, ASSET_DIGEST, tuple(sorted(layout)))

    def load_state(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return state

    monkeypatch.setattr(torch, "load", load_state)

    with torch.device("meta"):
        assembled = assemble_sd15_t2i_adapter(plan, adapter_dtype=torch.float32)

    assert {parameter.dtype for parameter in assembled.adapter.parameters()} == {torch.float64}
    assert bound_compute_dtype(assembled.adapter.conv_in) is torch.float32


def test_adapter_maps_features_to_canonical_down_sites() -> None:
    with torch.device("meta"):
        model = SD15T2IAdapter()
        residuals = model(torch.empty(1, 1, 64, 64, device="meta"))
    assert tuple(residuals.down[index].shape[1] for index in (2, 5, 8, 11)) == (
        320,
        640,
        1280,
        1280,
    )
    assert tuple(residuals.middle.shape) == (1, 1280, 1, 1)
