"""CPU behavior coverage for native custom-sampling guider nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.cfg_guiders import (
    CFG_GUIDER_NODES,
    BasicGuider,
    CFGGuider,
    CFGOverride,
    DualCFGGuider,
    DualModelGuider,
)


class _Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 10.0 - percent * 10.0


class _Model:
    device = torch.device("cpu")
    model_loaded_weight_memory = 0

    def __init__(self) -> None:
        self.model_sampling = _Sampling()

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def parameters(self) -> list[torch.Tensor]:
        return []

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}


def _patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_Model(), torch.device("cpu"), torch.device("cpu"))


def _conditioning(value: float, tag: str) -> list[list[object]]:
    return [[torch.tensor([value]), {"tag": tag}]]


def test_cold_schemas_and_registry_export_exact_five_nodes_without_torch_import() -> None:
    assert tuple(node.__name__ for node in CFG_GUIDER_NODES) == (
        "CFGGuider",
        "BasicGuider",
        "DualCFGGuider",
        "DualModelGuider",
        "CFGOverride",
    )
    assert [node.schema().node_type for node in CFG_GUIDER_NODES] == [
        "CFGGuider",
        "BasicGuider",
        "DualCFGGuider",
        "DualModelGuider",
        "CFGOverride",
    ]
    assert tuple(port.id for port in DualModelGuider.schema().inputs) == (
        "model",
        "model_negative",
        "positive",
        "cfg",
        "negative",
    )


def test_basic_cfg_and_dual_model_build_pinned_guiders_and_default_negative() -> None:
    model = _patcher()
    basic = BasicGuider.execute(model=model, conditioning=_conditioning(1.0, "basic"))["GUIDER"]
    cfg = CFGGuider.execute(
        model=model,
        positive=_conditioning(2.0, "positive"),
        negative=_conditioning(-3.0, "negative"),
        cfg=3.5,
    )["GUIDER"]
    dual = DualModelGuider.execute(
        model=model,
        positive=_conditioning(4.0, "positive"),
        cfg=2.0,
    )["GUIDER"]

    assert basic.model_patcher is model
    assert set(basic.original_conds) == {"positive"}
    assert cfg.model_patcher is model
    assert cfg.cfg == 3.5
    assert set(cfg.original_conds) == {"positive", "negative"}
    assert "cross_attn" not in dual.original_conds["negative"][0]


@pytest.mark.parametrize(
    ("style", "expected"),
    (("regular", 11.0), ("nested", 19.0)),
)
def test_dual_cfg_uses_pinned_condition_batch_and_cfg_formula(
    monkeypatch: pytest.MonkeyPatch, style: str, expected: float
) -> None:
    import dinkster_inference.samplers as samplers

    model = _patcher()
    guider = DualCFGGuider.execute(
        model=model,
        cond1=_conditioning(1.0, "positive"),
        cond2=_conditioning(2.0, "middle"),
        negative=_conditioning(3.0, "negative"),
        cfg_conds=2.0,
        cfg_cond2_negative=3.0,
        style=style,
    )["GUIDER"]
    guider.inner_model = SimpleNamespace()
    guider.conds = {"negative": "negative", "middle": "middle", "positive": "positive"}

    def calc(_model: Any, conditions: list[Any], *_args: Any) -> list[torch.Tensor]:
        values = {"negative": 1.0, "middle": 3.0, "positive": 5.0, None: 0.0}
        return [torch.tensor([values[condition]]) for condition in conditions]

    monkeypatch.setattr(samplers, "calc_cond_batch", calc)
    result = guider.predict_noise(torch.tensor([0.0]), torch.tensor([1.0]))
    torch.testing.assert_close(result, torch.tensor([expected]))
    assert guider.original_conds["middle"][0]["prompt_type"] == "negative"


def test_dual_cfg_preserves_cfg_one_optimization_and_disable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.samplers as samplers

    guider = DualCFGGuider.execute(
        model=_patcher(),
        cond1=_conditioning(1.0, "positive"),
        cond2=_conditioning(2.0, "middle"),
        negative=_conditioning(3.0, "negative"),
        cfg_conds=1.0,
        cfg_cond2_negative=1.0,
        style="regular",
    )["GUIDER"]
    guider.inner_model = SimpleNamespace()
    guider.conds = {"negative": "negative", "middle": "middle", "positive": "positive"}
    calls: list[list[Any]] = []

    def calc(_model: Any, conditions: list[Any], *_args: Any) -> list[torch.Tensor]:
        calls.append(conditions)
        return [torch.zeros(1) for _ in conditions]

    monkeypatch.setattr(samplers, "calc_cond_batch", calc)
    guider.predict_noise(torch.zeros(1), torch.ones(1))
    assert calls == [[None, None, "positive"]]
    guider.predict_noise(torch.zeros(1), torch.ones(1), {"disable_cfg1_optimization": True})
    assert calls[-1] == ["negative", "middle", "positive"]


def test_dual_model_guider_routes_unconditional_prediction_to_the_negative_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.samplers as samplers

    positive_model = _patcher()
    negative_model = _patcher()
    guider = DualModelGuider.execute(
        model=positive_model,
        model_negative=negative_model,
        positive=_conditioning(1.0, "positive"),
        negative=_conditioning(-1.0, "negative"),
        cfg=2.0,
    )["GUIDER"]
    guider.inner_model = "positive-model"
    guider.uncond_inner = "negative-model"
    guider.conds = {"positive": "positive"}
    guider._uncond_conds = "negative"
    calls: list[tuple[object, list[object], dict[str, object]]] = []

    def calc(
        inner_model: object,
        conditions: list[object],
        _x: object,
        _timestep: object,
        options: dict[str, object],
    ) -> list[torch.Tensor]:
        calls.append((inner_model, conditions, options))
        return [torch.tensor([10.0 if inner_model == "positive-model" else 2.0])]

    monkeypatch.setattr(samplers, "calc_cond_batch", calc)

    def cfg_function(
        _model: object,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        cfg: float,
        *_args: object,
        **_kwargs: object,
    ) -> torch.Tensor:
        return uncond_output + cfg * (cond_output - uncond_output)

    monkeypatch.setattr(samplers, "cfg_function", cfg_function)

    clone = object()
    result = guider.predict_noise(
        torch.zeros(1), torch.ones(1), {"multigpu_clones": [clone], "keep": "value"}
    )

    torch.testing.assert_close(result, torch.tensor([18.0]))
    assert calls == [
        ("positive-model", ["positive"], {"multigpu_clones": [clone], "keep": "value"}),
        ("negative-model", ["negative"], {"keep": "value"}),
    ]
    guider.cfg = 1.0
    assert guider.predict_noise(torch.zeros(1), torch.ones(1)).item() == 10.0


def test_dual_model_guider_cleans_up_negative_model_when_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.sampler_helpers as sampler_helpers
    import dinkster_inference.samplers as samplers

    positive_model = _patcher()
    negative_model = _patcher()
    events: list[str] = []
    prepared_negative = [{"marker": "negative"}]

    def prepare(
        model: object, _shape: object, conditions: dict[str, list[dict[str, str]]], _options: object
    ) -> tuple[object, dict[str, list[dict[str, str]]], list[str]]:
        assert model is negative_model
        assert conditions["negative"] == prepared_negative
        assert conditions["negative"] is not prepared_negative
        return SimpleNamespace(process_latent_in=lambda latent: latent), conditions, ["loaded"]

    def failed_outer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(sampler_helpers, "prepare_sampling", prepare)
    monkeypatch.setattr(sampler_helpers, "cleanup_models", lambda *_args: events.append("cleanup"))
    monkeypatch.setattr(negative_model, "pre_run", lambda: events.append("pre_run"))
    monkeypatch.setattr(negative_model, "cleanup", lambda: events.append("model_cleanup"))
    monkeypatch.setattr(samplers.CFGGuider, "outer_sample", failed_outer)

    guider = DualModelGuider.execute(
        model=positive_model,
        model_negative=negative_model,
        positive=_conditioning(1.0, "positive"),
        negative=_conditioning(-1.0, "negative"),
        cfg=2.0,
    )["GUIDER"]
    guider.conds = {"negative": prepared_negative}

    with pytest.raises(RuntimeError, match="sampling failed"):
        guider.outer_sample(torch.zeros(1), torch.zeros(1), None, torch.zeros(1))

    assert events == ["pre_run", "model_cleanup", "cleanup"]
    assert guider.uncond_inner is None


def test_cfg_override_uses_inclusive_bounds_and_restores_nested_cfg_on_error() -> None:
    model = _patcher()
    first = CFGOverride.execute(model=model, cfg=2.0, start_percent=0.2, end_percent=0.8)["MODEL"]
    patched = CFGOverride.execute(model=first, cfg=7.0, start_percent=0.4, end_percent=0.6)["MODEL"]
    wrappers = patched.get_all_wrappers("predict_noise")

    guider = SimpleNamespace(cfg=1.0)

    class Executor:
        class_obj = guider

        def __init__(self, callback: Any) -> None:
            self.callback = callback

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.callback(*args, **kwargs)

    def invoke(index: int, sigma: float, expected_cfg: float, fail: bool = False) -> float:
        if index == len(wrappers):
            assert guider.cfg == expected_cfg
            if fail:
                raise RuntimeError("expected")
            return guider.cfg
        return wrappers[index](
            Executor(lambda *args, **kwargs: invoke(index + 1, sigma, expected_cfg, fail)),
            torch.zeros(1),
            torch.tensor([sigma]),
        )

    assert invoke(0, 5.0, 7.0) == 7.0
    assert guider.cfg == 1.0
    assert invoke(0, 2.0, 2.0) == 2.0
    assert guider.cfg == 1.0
    with pytest.raises(RuntimeError, match="expected"):
        invoke(0, 5.0, 7.0, fail=True)
    assert guider.cfg == 1.0
