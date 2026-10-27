"""CPU contracts for MiniMax control preparation and model callbacks."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.minimax_h3_control import MiniMaxH3FunControlNetApply


class Vae:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []
        self.bad_shape = False

    def spacial_compression_encode(self) -> int:
        return 1

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self.inputs.append(frames.clone())
        if self.bad_shape:
            return torch.zeros(1)
        count = ((frames.shape[0] - 5) // 17) * 5 + 2
        indices = torch.linspace(0, frames.shape[0] - 1, count).long()
        return frames[indices].permute(3, 0, 1, 2).unsqueeze(0).double()


class Control:
    injection_layers = [2, 5]

    def __init__(self, post_norm: bool = False) -> None:
        self.inpaint_post_norm = post_norm
        self.initial: torch.Tensor | None = None

    def init_stream(self, image: torch.Tensor, *_: Any) -> torch.Tensor:
        self.initial = image.clone()
        return image.clone()

    def step(self, index: int, stream: torch.Tensor, *_: Any, **__: Any) -> Any:
        return stream + 1, torch.full_like(stream, index + 2)


class Model:
    def __init__(self) -> None:
        self.model_options: dict[str, Any] = {}
        self.wrappers: list[Any] = []

    def clone(self) -> Model:
        cloned = Model()
        cloned.model_options = copy.deepcopy(self.model_options)
        return cloned

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return SimpleNamespace(percent_to_sigma=lambda percent: 8 * (1 - percent))

    def add_wrapper(self, _: Any, wrapper: Any) -> None:
        self.wrappers.append(wrapper)

    def set_model_patch_replace(self, patch: Any, family: str, kind: str, index: int) -> None:
        self.model_options.setdefault("transformer_options", {}).setdefault(
            "patches_replace", {}
        ).setdefault(family, {})[kind, index] = patch


@pytest.fixture
def restored(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    initialize_inference_process()
    from dinkster_inference import model_management

    loaded = [object()]
    restored: list[Any] = []
    monkeypatch.setattr(model_management, "loaded_models", lambda **_: loaded)
    monkeypatch.setattr(model_management, "load_models_gpu", lambda models: restored.append(models))
    return restored


def apply(**kwargs: Any) -> Any:
    return MiniMaxH3FunControlNetApply.execute(
        model=kwargs.pop("model", Model()),
        model_patch=kwargs.pop("model_patch", SimpleNamespace(model=Control())),
        vae=kwargs.pop("vae", Vae()),
        strength=kwargs.pop("strength", 0.25),
        start_percent=0.25,
        end_percent=0.75,
        **kwargs,
    )["MODEL"]


@pytest.mark.parametrize("strength,control,mask", [(0.0, True, True), (1.0, False, False)])
def test_bypass_preserves_identity(strength: float, control: bool, mask: bool) -> None:
    source = Model()
    assert (
        apply(
            model=source,
            strength=strength,
            control_video=object() if control else None,
            mask=object() if mask else None,
            source_video=object(),
        )
        is source
    )
    assert not source.wrappers and source.model_options == {}


@pytest.mark.parametrize(
    "post_norm,with_hint,with_source",
    [
        (False, False, False),
        (False, True, True),
        (True, True, True),
        (True, False, False),
    ],
)
def test_mask_preparation_and_source_preservation(
    restored: list[Any],
    post_norm: bool,
    with_hint: bool,
    with_source: bool,
) -> None:
    from dinkster_inference.ldm.minimax.vae import IMAGENET_MEAN

    source, vae = Model(), Vae()
    video = torch.arange(2 * 2 * 3 * 4.0).reshape(2, 2, 3, 4) / 50
    mask = torch.tensor([[[0.0, 0.5, 0.5001], [1.0, 0.1, 0.9]]])
    before_video, before_mask = video.clone(), mask.clone()
    result = apply(
        model=source,
        vae=vae,
        model_patch=SimpleNamespace(model=Control(post_norm)),
        control_video=video if with_hint else None,
        mask=mask,
        source_video=video if with_source else None,
    )
    patch = result.wrappers[0].__self__
    patch.prepare_control_latent((1, 3, 2, 2, 3))
    assert patch.control_latent.shape == (1, 7, 2, 2, 3)
    assert patch.control_latent.dtype == torch.float32
    visibility = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    expected = (video[0, ..., :3] if with_source else torch.zeros(2, 3, 3)) * visibility[..., None]
    if post_norm:
        expected += (1 - visibility[..., None]) * torch.tensor(IMAGENET_MEAN)
    assert torch.equal(vae.inputs[-1][0], expected)
    assert torch.equal(
        vae.inputs[-1][-1],
        (
            video[-1, ..., :3] * visibility[..., None]
            + ((1 - visibility[..., None]) * torch.tensor(IMAGENET_MEAN) if post_norm else 0)
        )
        if with_source
        else expected,
    )
    assert torch.equal(patch.control_latent[0, 3, 0], visibility)
    if not with_hint:
        assert torch.count_nonzero(patch.control_latent[:, :3]) == 0
    assert torch.equal(video, before_video) and torch.equal(mask, before_mask)
    assert source.model_options == {} and not source.wrappers
    assert len(restored) == 1


def test_cache_shape_and_restore_after_encode_failure(restored: list[Any]) -> None:
    vae = Vae()
    patch = (
        apply(vae=vae, control_video=torch.ones(1, 2, 3, 3), source_video=object())
        .wrappers[0]
        .__self__
    )
    patch.prepare_control_latent((1, 3, 2, 2, 3))
    patch.prepare_control_latent((1, 3, 2, 2, 3))
    assert len(vae.inputs) == len(restored) == 1
    patch.prepare_control_latent((1, 3, 7, 2, 3))
    assert vae.inputs[-1].shape == (22, 2, 3, 3)
    assert len(restored) == 2
    vae.bad_shape = True
    with pytest.raises(ValueError, match="does not match the target"):
        patch.prepare_control_latent((1, 3, 12, 2, 3))
    assert len(restored) == 3
    assert patch.control_latent.shape == (1, 3, 7, 2, 3)
    patch.cleanup()
    assert patch.control_latent is None and patch.control_latent_shape is None


@pytest.mark.parametrize(
    "sigma,fallback,active",
    [
        (2.0, False, True),
        (6.0, False, True),
        (1.999, False, False),
        (6.001, False, False),
        (4.0, True, True),
    ],
)
def test_wrapper_closed_window_and_failure_cleanup(
    restored: list[Any],
    sigma: float,
    fallback: bool,
    active: bool,
) -> None:
    result = apply(control_video=torch.ones(1, 2, 3, 3))
    patch = result.wrappers[0].__self__

    def executor(*_: Any, **__: Any) -> Any:
        assert patch.active is active
        patch.control_stream, patch.pristine_stream = object(), object()
        raise RuntimeError("executor failed")

    with pytest.raises(RuntimeError, match="executor failed"):
        result.wrappers[0](
            executor,
            [torch.zeros(1, 3, 2, 2, 3)],
            torch.tensor([sigma * 1000], dtype=torch.float64),
            None,
            {} if fallback else {"sigmas": torch.tensor([sigma], dtype=torch.float64)},
        )
    assert patch.control_stream is patch.pristine_stream is None
    assert len(restored) == int(active)


def test_existing_block_hook_pristine_input_and_audio_exclusion(restored: list[Any]) -> None:
    source, control = Model(), Control()

    def previous(args: Any, _: Any) -> Any:
        args["img"].add_(10)
        return {"img": args["img"].clone()}

    source.model_options = {
        "transformer_options": {"patches_replace": {"dit": {("double_block", 2): previous}}}
    }
    result = apply(
        model=source,
        model_patch=SimpleNamespace(model=control),
        control_video=torch.ones(1, 2, 3, 3),
    )
    patch = result.wrappers[0].__self__
    patch.active = True
    patch.prepare_control_latent((1, 3, 2, 2, 3))
    blocks = result.model_options["transformer_options"]["patches_replace"]["dit"]
    image = torch.arange(15.0).reshape(5, 3)
    original = image.clone()
    args = dict(
        img=image,
        layout=SimpleNamespace(audio_pos=torch.tensor([1, 4])),
        t_emb=None,
        mod_segments=None,
        rope_freqs=None,
        transformer_options={},
    )
    first = blocks["double_block", 2](args, {})
    assert torch.equal(control.initial, original)
    expected = original + 10.5
    expected[[1, 4]] -= 0.5
    assert torch.equal(first["img"], expected)
    args["img"] = first["img"]
    second = blocks["double_block", 5](
        args, {"original_block": lambda values: {"img": values["img"].clone()}}
    )
    expected[[0, 2, 3]] += 0.75
    assert torch.equal(second["img"], expected)
    assert blocks["double_block", 2].models() == [patch.model_patch]
    blocks["double_block", 2].to(torch.float16)
    assert patch.control_latent.dtype == torch.float32
    blocks["double_block", 2].to(torch.device("cpu"))
    assert patch.control_stream is None
    blocks["double_block", 5].cleanup()
    assert not patch.active and patch.control_latent is None
    assert (
        source.model_options["transformer_options"]["patches_replace"]["dit"]["double_block", 2]
        is previous
    )
