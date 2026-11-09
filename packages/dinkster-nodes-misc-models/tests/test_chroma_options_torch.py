from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_misc_models.chroma_options import ChromaRadianceOptions


class Patcher:
    def __init__(self, wrapper: Any = None) -> None:
        self.model_options: dict[str, Any] = {}
        if wrapper is not None:
            self.model_options["model_function_wrapper"] = wrapper
        self.wrapper: Any = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.model_options = self.model_options.copy()
        return result

    def set_model_unet_function_wrapper(self, wrapper: Any) -> None:
        self.wrapper = wrapper


def _apply_model(input_value: Any, timestep: Any, **c: Any) -> dict[str, Any]:
    return {"input": input_value, "timestep": timestep, "c": c}


def _args(timestep: list[float], c: dict[str, Any]) -> dict[str, Any]:
    return {"input": "input", "timestep": torch.tensor(timestep), "c": c}


def test_no_chroma_options_returns_the_original_model() -> None:
    source = Patcher()
    result = ChromaRadianceOptions.execute(
        model=source,
        preserve_wrapper=True,
        start_sigma=1.0,
        end_sigma=0.0,
        nerf_tile_size=-1,
        force_sequential_txt_ids=False,
    )["MODEL"]
    assert result is source


@pytest.mark.parametrize(
    ("timestep", "active"),
    (
        ([0.1], False),
        ([0.25], True),
        ([0.1, 0.5], True),
        ([0.75], True),
        ([0.9], False),
    ),
)
def test_chroma_options_apply_at_inclusive_max_batch_sigma_boundaries(
    timestep: list[float], active: bool
) -> None:
    source = Patcher()
    patched = ChromaRadianceOptions.execute(
        model=source,
        preserve_wrapper=False,
        start_sigma=0.75,
        end_sigma=0.25,
        nerf_tile_size=64,
        force_sequential_txt_ids=True,
    )["MODEL"]
    original_options = {"existing": "keep"}
    original_c = {"other": "value", "transformer_options": original_options}
    result = patched.wrapper(_apply_model, _args(timestep, original_c))
    assert source.wrapper is None
    assert original_c == {"other": "value", "transformer_options": {"existing": "keep"}}
    assert result["c"]["other"] == "value"
    if active:
        assert result["c"]["transformer_options"] == {
            "existing": "keep",
            "chroma_radiance_options": {"nerf_tile_size": 64, "use_sequential_txt_ids": True},
        }
    else:
        assert result["c"] == original_c


@pytest.mark.parametrize("preserve_wrapper", (True, False))
def test_chroma_options_preserves_or_bypasses_an_existing_wrapper(preserve_wrapper: bool) -> None:
    calls: list[dict[str, Any]] = []

    def old_wrapper(apply_model: Any, args: dict[str, Any]) -> Any:
        calls.append(args)
        return {"old": True, "c": args["c"]}

    patched = ChromaRadianceOptions.execute(
        model=Patcher(old_wrapper),
        preserve_wrapper=preserve_wrapper,
        start_sigma=0.8,
        end_sigma=0.2,
        nerf_tile_size=0,
        force_sequential_txt_ids=False,
    )["MODEL"]
    result = patched.wrapper(_apply_model, _args([0.1], {"other": "value"}))
    if preserve_wrapper:
        assert result == {"old": True, "c": {"other": "value"}}
        assert len(calls) == 1
    else:
        assert result == {
            "input": "input",
            "timestep": torch.tensor([0.1]),
            "c": {"other": "value"},
        }
        assert calls == []
