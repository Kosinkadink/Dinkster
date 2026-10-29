from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from dinkster_inference import (
    QWEN_IMAGE,
    QWEN_IMAGE_LAYERED_CONFIG,
    WAN21_CODEC,
    ComponentApplication,
    ConditioningCarrier,
    InferenceCodecHandle,
    InferenceRuntimeHandle,
    SizedTensor,
    split_component_conditioning,
)
from dinkster_inference_torch import QwenImageConditioning, QwenImageRuntime
from dinkster_inference_torch import qwen_image_control as control_module
from dinkster_inference_torch.qwen_image import QwenImage
from dinkster_inference_torch.qwen_image_assembly import AssembledQwenImage
from dinkster_inference_torch.qwen_image_control import (
    QwenImageControlConditioning,
    QwenImageDiffSynthExecution,
    QwenImageDiffSynthPatch,
    QwenImageInstantXControlNet,
    qwen_image_control_resource_digest,
    qwen_image_diffsynth_resource_digest,
)
from dinkster_inference_torch.qwen_image_text import QwenImageTextModel
from dinkster_inference_torch.wan21_vae import WanVAE
from dinkster_model_qwen_image import provider

_REPO = Path(__file__).resolve().parents[3]
_DIRECT_OPERATION_GOLDEN = json.loads(
    (_REPO / "tests" / "goldens" / "comfy_direct_operations_b78cec87.json").read_text()
)


class _ComponentHandle:
    load_device = torch.device("cpu")

    def __init__(self, role: str, component: object, events: list[str]) -> None:
        digest = "1" if role == "qwen" else "2"
        self.resource_identity = "native:dinkster.qwen_image:" + digest * 64
        self.module = component
        self._events = events
        self._staged = False

    @property
    def component(self) -> object:
        if not self._staged:
            raise RuntimeError("component read outside its lease")
        return self.module

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self):
        self._events.append("stage")
        self._staged = True
        try:
            yield
        finally:
            self._staged = False

    @contextmanager
    def stage_with(self, _runtime_handle: object, _role: str):
        yield


def test_control_loader_plans_assembles_and_publishes_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Asset:
        digest = "blake3:" + "1" * 64
        size = 123

        def local_path(self) -> Path:
            return Path("control.safetensors")

    source = object()
    component_plan = object()
    plan = SimpleNamespace(control=component_plan)
    control = object()
    published = object()
    calls: list[tuple[object, ...]] = []

    def load(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return source

    def plan_control(value: object, *, asset_digest: str) -> object:
        calls.append(("plan", value, asset_digest))
        return plan

    def assemble(value: object, *, compute_dtype: torch.dtype) -> object:
        calls.append(("assemble", value, compute_dtype))
        return SimpleNamespace(control=control)

    class Publisher:
        def publish(self, module: object, *, resource_identity: str) -> object:
            calls.append(("publish", module, resource_identity))
            return published

    def resource_identity(_value: object) -> str:
        return "identity"

    monkeypatch.setattr(provider, "load_safetensors_header", load)
    monkeypatch.setattr(provider, "plan_qwen_image_control", plan_control)
    monkeypatch.setattr(provider, "assemble_qwen_image_control", assemble)
    monkeypatch.setattr(provider, "_control_resource_identity", resource_identity)
    monkeypatch.setattr(provider, "component_publisher", lambda: Publisher())

    assert provider.execute_load_qwen_image_control(control_net=Asset()) == {"control": published}
    assert calls == [
        ("header", Path("control.safetensors"), Asset.digest, 123),
        ("plan", source, Asset.digest),
        ("assemble", plan, torch.bfloat16),
        ("publish", control, "identity"),
    ]


def test_control_apply_builds_identity_bound_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.device("meta"):
        control = QwenImageInstantXControlNet()
    digest = qwen_image_control_resource_digest("blake3:" + "3" * 64, "instantx", torch.bfloat16)
    control_module._bind_qwen_image_control_resource(  # pyright: ignore[reportPrivateUsage]
        control, digest
    )
    handle = _ComponentHandle("qwen", control, [])
    captured: list[ComponentApplication] = []

    def capture(model: object, application: ComponentApplication) -> object:
        assert model == "base-model"
        captured.append(application)
        return "applied-model"

    monkeypatch.setattr(provider, "_append_application", capture)
    hint = torch.zeros((1, 16, 1, 2, 2))
    result = provider.execute_apply_qwen_image_control(
        model="base-model",
        control=handle,
        hint={"samples": hint},
        strength=1.5,
        start_percent=0.25,
        end_percent=0.75,
    )

    assert result == {"model": "applied-model"}
    application = captured[0]
    assert application.family_id == "dinkster.qwen_image"
    assert application.role == "diffusion"
    kwargs = application.materialize_application_kwargs(object(), control, hint)
    prepared = kwargs["control"]
    assert type(prepared) is QwenImageControlConditioning
    assert prepared.model is control
    assert prepared.application.strength == 1.5
    assert prepared.application.window.start_percent == 0.25
    assert prepared.application.window.end_percent == 0.75


def test_diffsynth_apply_prepares_execution_inside_component_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.device("meta"):
        patch = QwenImageDiffSynthPatch()
    digest = qwen_image_diffsynth_resource_digest("blake3:" + "4" * 64, "diffsynth", torch.bfloat16)
    control_module._bind_qwen_image_diffsynth_resource(  # pyright: ignore[reportPrivateUsage]
        patch, digest
    )
    handle = _ComponentHandle("qwen", patch, [])
    captured: list[ComponentApplication] = []

    def capture(_model: object, application: ComponentApplication) -> object:
        captured.append(application)
        return "applied-model"

    def prepare(_self: QwenImageDiffSynthPatch, latent: torch.Tensor) -> torch.Tensor:
        assert latent.shape == (1, 16, 1, 2, 2)
        return torch.zeros((1, 1, 3072), dtype=torch.bfloat16)

    monkeypatch.setattr(provider, "_append_application", capture)
    monkeypatch.setattr(QwenImageDiffSynthPatch, "prepare_condition", prepare)
    hint = torch.zeros((1, 16, 1, 2, 2))
    result = provider.execute_apply_qwen_image_diffsynth(
        model=object(),
        patch=handle,
        hint={"samples": hint},
        strength=-0.5,
    )

    assert result == {"model": "applied-model"}
    application = captured[0]
    kwargs = application.materialize_application_kwargs(
        object(), patch, torch.zeros((1, 16, 1, 2, 2))
    )
    prepared = kwargs["diffsynth"]
    assert isinstance(prepared, tuple) and len(prepared) == 1
    assert type(prepared[0]) is QwenImageDiffSynthExecution
    assert prepared[0].model is patch
    assert prepared[0].strength == -0.5


def test_layered_latent_matches_qwen_layer_geometry() -> None:
    assert _DIRECT_OPERATION_GOLDEN["format"] == "dinkster-comfy-direct-operation-golden/1"
    assert _DIRECT_OPERATION_GOLDEN["referenceCommit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    operation = next(
        item
        for item in _DIRECT_OPERATION_GOLDEN["operations"]
        if item["sourceNode"] == "EmptyQwenImageLayeredLatentImage"
    )
    assert operation["scope"] == "operation-only-not-family-parity"
    for case in operation["cases"]:
        result = provider.execute_empty_qwen_image_layered_latent(**case["inputs"])
        samples = cast("dict[str, torch.Tensor]", result["latent"])["samples"]
        contiguous = samples.detach().cpu().contiguous()
        assert {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype).removeprefix("torch."),
            "nonzero": torch.count_nonzero(contiguous).item(),
            "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        } == case["output"]

    with pytest.raises(ValueError, match="multiples of 16"):
        provider.execute_empty_qwen_image_layered_latent(
            width=641, height=480, layers=3, batch_size=1
        )


def test_component_text_encoder_emits_bound_base_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    handle = _ComponentHandle("qwen", object(), events)

    class TextRuntime:
        def __init__(self, component: object) -> None:
            assert component is handle.module

        def encode_text(self, text: str) -> QwenImageConditioning:
            assert text == "a lighthouse"
            events.append("encode")
            return QwenImageConditioning(torch.ones((1, 2, 8)))

    monkeypatch.setattr(provider, "QwenImageTextRuntime", TextRuntime)
    result = provider.execute_qwen_image_edit_encode(
        clip=handle,
        vae=None,
        prompt="a lighthouse",
        image=None,
    )

    bound = result["conditioning"]
    assert type(bound) is ConditioningCarrier
    _, binding = split_component_conditioning(bound)
    assert binding is not None
    assert binding.role == "qwen2_5_vl_7b"
    assert binding.identity == handle.resource_identity
    assert events == ["stage", "encode"]


def test_component_edit_encoder_leases_text_and_vae_before_reading_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    text_handle = _ComponentHandle("qwen", object(), events)
    vae_handle = _ComponentHandle("vae", object(), events)

    class TextRuntime:
        def __init__(self, component: object) -> None:
            assert component is text_handle.module

        def encode_edit_text(
            self,
            _text: str,
            _contents: tuple[torch.Tensor, ...],
            **_kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            events.append("text")
            return torch.ones((1, 2, 8)), torch.ones((1, 2), dtype=torch.int64)

    class CodecRuntime:
        def __init__(self, component: object) -> None:
            assert component is vae_handle.module

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            events.append("vae")
            return torch.zeros((content.shape[0], 16, 1, 2, 2))

    monkeypatch.setattr(provider, "QwenImageTextRuntime", TextRuntime)
    monkeypatch.setattr(provider, "WanVAECodecRuntime", CodecRuntime)

    def identity_resize(image: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return image

    monkeypatch.setattr(provider, "resize_qwen_image_content", identity_resize)
    image = np.zeros((16, 16, 3), dtype=np.float32)

    result = provider.execute_qwen_image_edit_encode(
        clip=text_handle,
        vae=vae_handle,
        prompt="edit",
        image=image,
    )

    edit_bound = result["conditioning"]
    assert type(edit_bound) is ConditioningCarrier
    _, edit_binding = split_component_conditioning(edit_bound)
    assert edit_binding is not None
    assert events == ["stage", "text", "stage", "vae"]


def test_layered_node_output_runs_through_the_layered_runtime() -> None:
    class Diffusion(torch.nn.Module):
        config = QWEN_IMAGE_LAYERED_CONFIG

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            latent: torch.Tensor,
            _timesteps: torch.Tensor,
            _context: torch.Tensor,
            _attention_mask: torch.Tensor | None = None,
            _ref_latents: tuple[torch.Tensor, ...] = (),
            _additional_t_cond: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.zeros_like(latent) + self.weight

    class VAE(torch.nn.Module):
        def process_out(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent[:, :3].repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)

    native = QwenImageRuntime(
        AssembledQwenImage(
            diffusion=cast("QwenImage", Diffusion()),
            text=cast("QwenImageTextModel", torch.nn.Identity()),
            vae=cast("WanVAE", VAE()),
            family=QWEN_IMAGE,
        ),
        runtime_identity="test.qwen-image-layered-node",
    )
    generated = provider.execute_empty_qwen_image_layered_latent(
        width=16, height=16, layers=3, batch_size=1
    )
    latent = cast("dict[str, torch.Tensor]", generated["latent"])["samples"]

    sampled = native.sample(
        latent,
        cond=QwenImageConditioning(torch.zeros((1, 2, 3584))),
        sampler_id="euler",
        scheduler_id="simple",
        steps=1,
    )

    assert sampled.shape == (1, 16, 4, 2, 2)
    assert native.decode_latent(sampled).shape == (1, 3, 4, 16, 16)


def test_edit_provider_stages_text_and_codec_and_returns_rich_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Runtime:
        def encode_edit_text(
            self,
            prompt: str,
            images: tuple[torch.Tensor, ...],
            *,
            edit_plus: bool,
            image_slots: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert prompt == "turn it blue"
            assert len(images) == 2
            assert edit_plus
            assert image_slots == (2, 3)
            assert all(image.shape == (1, 3, 32, 48) for image in images)
            events.append("text")
            return torch.ones((1, 4, 8)), torch.ones((1, 4), dtype=torch.int64)

    class Handle:
        load_device = torch.device("cpu")

        @contextmanager
        def stage(self, role: str):
            events.append(f"stage-{role}")
            yield

    class Codec:
        descriptor = WAN21_CODEC
        load_device = torch.device("cpu")

        @contextmanager
        def stage(self):
            events.append("stage-vae")
            yield

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            events.append("encode")
            return torch.zeros((content.shape[0], 16, 1, 2, 2))

    handle = Handle()
    codec = Codec()

    def require_runtime(_value: object, _name: str) -> InferenceRuntimeHandle:
        return cast("InferenceRuntimeHandle", handle)

    def require_codec(_value: object, _name: str) -> InferenceCodecHandle[SizedTensor]:
        return cast("InferenceCodecHandle[SizedTensor]", codec)

    def require_qwen(_handle: InferenceRuntimeHandle) -> Runtime:
        return Runtime()

    def identity_resize(image: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return image

    monkeypatch.setattr(provider, "require_inference_runtime_handle", require_runtime)
    monkeypatch.setattr(provider, "InferenceRuntimeHandle", Handle)
    monkeypatch.setattr(
        provider,
        "require_inference_codec_handle",
        require_codec,
    )
    monkeypatch.setattr(provider, "_qwen_runtime", require_qwen)
    monkeypatch.setattr(
        provider,
        "resize_qwen_image_content",
        identity_resize,
    )

    image = np.zeros((32, 48, 3), dtype=np.float32)
    result = provider.execute_qwen_image_edit_plus_encode(
        clip=handle,
        vae=SimpleNamespace(),
        prompt="turn it blue",
        images=(None, image, image),
    )
    assert type(result["conditioning"]) is ConditioningCarrier
    assert events == ["stage-text", "text", "stage-vae", "encode", "encode"]
