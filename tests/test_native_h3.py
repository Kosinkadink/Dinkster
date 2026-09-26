"""MiniMax H3 tests for the ordinary Comfy graph contract."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dinkster_assets import AssetRef
from dinkster_compat_comfy import native_arm
from dinkster_compat_comfy import translate as comfy_translate
from dinkster_compat_comfy.native import (
    AUDIO,
    DINKSTER_CLIP,
    DINKSTER_CONDITIONING,
    DINKSTER_LATENT,
    DINKSTER_VAE,
    FLOAT,
    IMAGE,
    INT,
    LATENT,
    MASK,
    MODEL_PATCH,
    NATIVE_NODES,
    STRING,
    VAE,
    MiniMaxH3AudioReferenceValue,
    MiniMaxH3ImageReferenceValue,
    MiniMaxH3VideoReferenceValue,
)
from dinkster_compat_comfy.native_residency import NativeRuntimeHandle
from dinkster_graph import Graph, GraphNode, validate
from dinkster_inference import (
    CONDITIONING_TYPE_ID,
    MINIMAX_H3_VIDEO_MASK_MAPPING,
    MultiStreamLatent,
    register_conditioning_type,
)
from dinkster_schema import ComboWidget, NumberWidget, build_schemas, schema_signature
from dinkster_values import ResidencyTable, TypeRegistry


class FakeTensor:
    layout = "strided"

    def __init__(
        self,
        shape: tuple[int, ...],
        device: object = "cpu",
        content: bytes | None = None,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = "torch.float32"
        self.device = device
        self.content = bytes(math.prod(shape)) if content is None else content
        self.contiguous_calls = 0

    def permute(self, *axes: int) -> FakeTensor:
        return FakeTensor(tuple(self.shape[index] for index in axes), self.device, self.content)

    def unsqueeze(self, axis: int) -> FakeTensor:
        shape = list(self.shape)
        shape.insert(axis, 1)
        return FakeTensor(tuple(shape), self.device, self.content)

    def to(self, device: object) -> FakeTensor:
        selected = device.device if type(device) is FakeTensor else device
        return FakeTensor(self.shape, selected, self.content)

    def clone(self) -> FakeTensor:
        return FakeTensor(self.shape, self.device, self.content)

    def contiguous(self) -> FakeTensor:
        self.contiguous_calls += 1
        return self

    def detach(self) -> FakeTensor:
        return self

    def view(self, _dtype: object) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return SimpleNamespace(tobytes=lambda: self.content)

    def is_floating_point(self) -> bool:
        return True

    def __getitem__(self, index: object) -> FakeTensor:
        if type(index) is tuple and len(index) == 2 and index[0] is Ellipsis:
            selected = cast("tuple[object, object]", index)[1]
            if isinstance(selected, slice):
                start, stop, step = selected.indices(self.shape[-1])
                return FakeTensor(
                    (*self.shape[:-1], len(range(start, stop, step))),
                    self.device,
                    self.content,
                )
        if isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            return FakeTensor(
                (len(range(start, stop, step)), *self.shape[1:]), self.device, self.content
            )
        if type(index) is int:
            return FakeTensor(self.shape[1:], self.device, self.content)
        raise TypeError("only leading indexing is supported")

    def split(self, size: int) -> tuple[FakeTensor, ...]:
        assert size == 1
        return tuple(
            FakeTensor((1, *self.shape[1:]), self.device, self.content)
            for _ in range(self.shape[0])
        )


class IntegerImageTensor:
    layout = "strided"

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        *,
        divisor: float | None = None,
        contiguous: bool = False,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.divisor = divisor
        self.was_made_contiguous = contiguous

    def is_floating_point(self) -> bool:
        return self.dtype == "float32"

    def to(self, *, dtype: object) -> IntegerImageTensor:
        return IntegerImageTensor(self.shape, dtype)

    def __truediv__(self, divisor: float) -> IntegerImageTensor:
        return IntegerImageTensor(self.shape, self.dtype, divisor=divisor)

    def contiguous(self) -> IntegerImageTensor:
        return IntegerImageTensor(
            self.shape,
            self.dtype,
            divisor=self.divisor,
            contiguous=True,
        )


def _streams(*, reverse: bool = False) -> MultiStreamLatent[FakeTensor]:
    pairs = (
        ("video", FakeTensor((1, 24, 12, 32, 48))),
        ("audio", FakeTensor((1, 32, 2, 20))),
    )
    return MultiStreamLatent.from_pairs(reversed(pairs) if reverse else pairs)


def _latent(*, reverse: bool = False) -> dict[str, object]:
    return {"samples": _streams(reverse=reverse)}


def _conditioning_rows(value: object) -> list[list[object]]:
    inference = __import__("dinkster_inference")
    return cast("list[list[object]]", _resident_payload(value, inference).conditioning)


def _resident_payload(value: object, inference: Any) -> Any:
    assert type(value) is inference.ConditioningCarrier
    bindings = cast("Any", value).bindings
    assert len(bindings) == 1
    assert bindings[0].kind == "resident"
    return bindings[0].payload


def _install_sampling_runtime(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> None:
    runtime.runtime_identity = "native:dinkster.minimax_h3:" + "1" * 64
    runtime.conditioning_identity = runtime.runtime_identity
    monkeypatch.setattr(
        native_arm,
        "resolve_minimax_h3_component_execution",
        lambda _handle, positive, negative, _inference, **_kwargs: (runtime, positive, negative),
    )


def _install_upscale(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, str]]:
    calls: list[tuple[int, int, str]] = []

    def upscale(value: FakeTensor, width: int, height: int, method: str, crop: str) -> FakeTensor:
        assert method == "lanczos"
        calls.append((width, height, crop))
        return FakeTensor((value.shape[0], value.shape[1], height, width))

    monkeypatch.setitem(
        sys.modules, "dinkster_inference_torch.resize", SimpleNamespace(common_upscale=upscale)
    )
    return calls


def _fake_torch() -> object:
    return SimpleNamespace(
        Tensor=FakeTensor,
        strided="strided",
        uint8="uint8",
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        inference_mode=nullcontext,
        ones_like=lambda value: FakeTensor(value.shape),
    )


@pytest.mark.parametrize(("dtype", "maximum"), (("uint8", 255), ("uint16", 65535)))
def test_h3_image_batch_normalizes_integer_asset_storage(dtype: str, maximum: int) -> None:
    torch = SimpleNamespace(
        Tensor=IntegerImageTensor,
        strided="strided",
        uint8="uint8",
        uint16="uint16",
        float32="float32",
        iinfo=lambda selected: SimpleNamespace(max={"uint8": 255, "uint16": 65535}[selected]),
    )

    result = native_arm._minimax_h3_image_batch(
        IntegerImageTensor((2, 3, 5, 3), dtype),
        torch,
        "reference",
    )

    assert result.shape == (2, 3, 5, 3)
    assert result.dtype == "float32"
    assert result.divisor == maximum
    assert result.was_made_contiguous is True


def test_h3_schemas_use_declared_resident_graph_types() -> None:
    nodes = {node.schema().node_type: node for node in NATIVE_NODES}
    arms = {node.schema().node_type: node for node in native_arm.NATIVE_ARM_NODES}
    expected = {
        "dinkster.apply_minimax_h3_fun_control_patch",
        "dinkster.empty_minimax_h3_av",
        "dinkster.set_latent_mask_from_frames",
        "dinkster.set_latent_mask_from_time_ranges",
        "dinkster.inspect_latent_mask",
        "dinkster.minimax_h3_t2va_conditioning",
        "dinkster.minimax_h3_fl2va_conditioning",
        "dinkster.minimax_h3_ref2va_conditioning",
        "dinkster.minimax_h3_add_guide",
        "dinkster.minimax_h3_motion_context",
        "dinkster.minimax_h3_av_encode",
        "dinkster.minimax_h3_av_decode",
        "dinkster.concat_av_latent",
        "dinkster.separate_av_latent",
        "dinkster.preview_latent_visual",
        "dinkster.preview_latent_audio",
    }
    assert expected <= nodes.keys()
    assert expected <= arms.keys()
    assert "dinkster.frame_range_mask" in nodes
    control = nodes["dinkster.apply_minimax_h3_fun_control_patch"].schema()
    assert tuple(item.id for item in control.inputs) == (
        "model",
        "model_patch",
        "vae",
        "strength",
        "start_percent",
        "end_percent",
        "control_video",
        "mask",
        "source_video",
    )
    assert control.inputs[1].type == MODEL_PATCH
    assert control.aliases == ("MiniMaxH3FunControlNetApply",)
    empty_schema = nodes["dinkster.empty_minimax_h3_av"].schema()
    assert tuple(item.id for item in empty_schema.inputs) == ("width", "height", "frame_count")
    assert empty_schema.outputs[0].type.types == ("dinkster.latent",)
    encode_schema = nodes["dinkster.minimax_h3_av_encode"].schema()
    assert tuple(item.id for item in encode_schema.inputs) == (
        "video_vae",
        "audio_vae",
        "frames",
        "audio",
    )
    assert encode_schema.inputs[0].type == encode_schema.inputs[1].type == VAE
    assert encode_schema.outputs[0].type == LATENT
    decode_schema = nodes["dinkster.minimax_h3_av_decode"].schema()
    assert tuple(item.id for item in decode_schema.inputs) == (
        "video_vae",
        "audio_vae",
        "latent",
    )
    assert decode_schema.inputs[0].type == decode_schema.inputs[1].type == VAE
    assert decode_schema.inputs[-1].type == LATENT
    conditioning = nodes["dinkster.minimax_h3_t2va_conditioning"].schema()
    assert tuple(item.id for item in conditioning.inputs) == (
        "clip",
        "target",
        "prompt",
    )
    assert tuple(item.type for item in conditioning.inputs[:2]) == (
        DINKSTER_CLIP,
        DINKSTER_LATENT,
    )
    assert tuple(item.id for item in conditioning.outputs) == ("conditioning",)
    assert all(item.type == DINKSTER_CONDITIONING for item in conditioning.outputs)
    fl2va = nodes["dinkster.minimax_h3_fl2va_conditioning"].schema()
    assert tuple(item.id for item in fl2va.inputs[:2]) == ("clip", "video_vae")
    assert (fl2va.inputs[0].type, fl2va.inputs[1].type) == (DINKSTER_CLIP, DINKSTER_VAE)
    ref2va = nodes["dinkster.minimax_h3_ref2va_conditioning"].schema()
    assert tuple(item.id for item in ref2va.inputs[:3]) == (
        "clip",
        "video_vae",
        "audio_vae",
    )
    assert tuple(item.type for item in ref2va.inputs[:3]) == (
        DINKSTER_CLIP,
        DINKSTER_VAE,
        DINKSTER_VAE,
    )
    guide = nodes["dinkster.minimax_h3_add_guide"].schema()
    assert tuple(item.id for item in guide.inputs) == (
        "positive",
        "vae",
        "audio_vae",
        "latent",
        "image",
        "audio",
        "frame_idx",
    )
    assert tuple(item.type for item in guide.inputs) == (
        DINKSTER_CONDITIONING,
        DINKSTER_VAE,
        DINKSTER_VAE,
        DINKSTER_LATENT,
        IMAGE,
        AUDIO,
        INT,
    )
    assert guide.aliases == ("MiniMaxH3AddGuide",)
    assert guide.outputs[0].type == DINKSTER_CONDITIONING
    motion = nodes["dinkster.minimax_h3_motion_context"].schema()
    assert tuple(item.id for item in motion.inputs) == (
        "positive",
        "latent",
        "previous_latent",
        "context_length",
    )
    assert tuple(item.type for item in motion.inputs) == (
        DINKSTER_CONDITIONING,
        DINKSTER_LATENT,
        DINKSTER_LATENT,
        INT,
    )
    assert motion.inputs[-1].default == 22
    assert motion.inputs[-1].widget == NumberWidget(min=5, max=3600, step=17)
    assert tuple(item.type for item in motion.outputs) == (DINKSTER_CONDITIONING, FLOAT)
    assert motion.aliases == ("MiniMaxH3MotionContext",)
    for node_type in expected:
        assert schema_signature(nodes[node_type].schema()) == schema_signature(
            arms[node_type].schema()
        )

    empty = {item.id: item for item in empty_schema.inputs}
    assert empty["width"].widget == NumberWidget(min=32, max=16384, step=32)
    assert empty["height"].widget == NumberWidget(min=32, max=16384, step=32)
    assert empty["frame_count"].widget == NumberWidget(min=5, max=3600, step=17)
    ref_size = nodes["dinkster.minimax_h3_ref2va_conditioning"].schema().inputs[-1]
    assert ref_size.default == "match"
    assert isinstance(ref_size.widget, ComboWidget)
    assert ref_size.widget.options == ("match", "max")
    frame_mask = nodes["dinkster.frame_range_mask"].schema()
    assert tuple(item.id for item in frame_mask.inputs) == (
        "width",
        "height",
        "frames",
        "ranges",
    )
    assert frame_mask.outputs[0].type == MASK
    set_frames = nodes["dinkster.set_latent_mask_from_frames"].schema()
    assert tuple(item.id for item in set_frames.inputs) == (
        "latent",
        "vae",
        "mask",
        "spatial_reduction",
        "temporal_reduction",
        "operation",
    )
    assert tuple(item.type for item in set_frames.inputs[:3]) == (LATENT, VAE, MASK)
    assert set_frames.outputs[0].type == LATENT
    set_times = nodes["dinkster.set_latent_mask_from_time_ranges"].schema()
    assert tuple(item.id for item in set_times.inputs) == (
        "latent",
        "vae",
        "ranges",
        "selected",
        "unselected",
        "operation",
    )
    inspect = nodes["dinkster.inspect_latent_mask"].schema()
    assert tuple(item.type for item in inspect.inputs) == (LATENT, VAE)
    assert tuple(item.type for item in inspect.outputs) == (MASK, STRING)


def test_deleted_h3_bundle_loader_is_an_unknown_node_type() -> None:
    graph = Graph(nodes={"loader": GraphNode("dinkster.load_minimax_h3", {})})

    diagnostics = validate(graph, build_schemas(NATIVE_NODES), ["loader"])

    assert [(item.code, item.message) for item in diagnostics] == [
        ("unknown-node-type", "unknown node type: dinkster.load_minimax_h3")
    ]


def test_official_h3_fun_workflow_resolves_control_nodes_and_widgets() -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    fixture = fixture_root / "video_minimax_h3_fun_controlnet_union.json"
    source = json.loads(
        (fixture_root / "video_minimax_h3_fun_controlnet_union.SOURCE.json").read_text()
    )
    payload = fixture.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source["sha256"]
    assert source == {
        "comfyui_revision": "b5cc8830279eae909a59de030af1e50761c36751",
        "path": "templates/video_minimax_h3_fun_controlnet_union.json",
        "sha256": "ec82867872aaf4c28901fafaf4e3fed3081e0c74c1968c34e4194878c5c0c0af",
        "workflow_templates_revision": "7c25a3c586484601f94b7e8f8b14c23b2c95a096",
    }
    workflow = json.loads(payload)
    nodes = {node["type"]: node for node in workflow["nodes"]}
    schemas = {node.schema().node_type: node.schema() for node in NATIVE_NODES}
    aliases = {alias: schema for schema in schemas.values() for alias in schema.aliases}

    subgraph = workflow["definitions"]["subgraphs"][0]
    assert subgraph["id"] == "622cceeb-7ba3-4dd9-b0e8-00066c5222a4"
    assert {node["id"]: node["type"] for node in subgraph["nodes"]} == {
        671: "SDPoseKeypointExtractor",
        672: "SDPoseDrawKeypoints",
        673: "CheckpointLoaderSimple",
        674: "ResizeImageMaskNode",
        677: "UNETLoader",
        678: "RTDETR_detect",
        692: "GetVideoComponents",
    }
    assert {
        (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"])
        for link in subgraph["links"]
        if link["origin_id"] > 0 and link["target_id"] > 0
    } == {
        (671, 0, 672, 0),
        (673, 0, 671, 0),
        (673, 2, 671, 1),
        (674, 0, 671, 2),
        (674, 0, 678, 1),
        (677, 0, 678, 0),
        (678, 0, 671, 3),
        (692, 0, 674, 0),
    }
    promoted = next(node for node in workflow["nodes"] if node["id"] == 700)
    assert promoted["type"] == subgraph["id"]
    assert promoted["widgets_values_named"] == {
        "resize_type.longer_size": 1024,
        "scale_method": "lanczos",
        "draw_body": True,
        "draw_hands": True,
        "draw_face": True,
        "draw_feet": True,
        "stick_width": 4,
        "face_point_size": 2,
        "score_threshold": 0.51,
        "threshold": 0.5,
        "class_name": "person",
        "max_detections": 2,
        "ckpt_name": "sdpose_wholebody_fp16.safetensors",
        "unet_name": "rt_detr_v4-x-hgnet_fp16.safetensors",
    }

    loader = aliases["ModelPatchLoader"]
    apply = aliases["MiniMaxH3FunControlNetApply"]
    assert loader.node_type == "dinkster.load_z_image_control_patch"
    assert loader.dispatch_affinity == "native"
    assert apply.node_type == "dinkster.apply_minimax_h3_fun_control_patch"
    assert nodes["ModelPatchLoader"]["widgets_values"] == [
        "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
    ]
    assert nodes["MiniMaxH3FunControlNetApply"]["widgets_values"] == [1, 0, 1]
    assert tuple(item.id for item in apply.inputs[3:6]) == (
        "strength",
        "start_percent",
        "end_percent",
    )


def test_h3_fun_control_noop_preserves_model_identity() -> None:
    model = object()
    assert (
        native_arm.NativeApplyMiniMaxH3FunControlPatch.execute(
            model=model,
            model_patch=object(),
            vae=object(),
            strength=0.0,
            start_percent=0.0,
            end_percent=1.0,
            control_video=object(),
        )["model"]
        is model
    )


def test_h3_fun_control_registers_family_neutral_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = importlib.import_module("dinkster_native.families.minimax_h3")
    family_id = "dinkster.minimax_h3"

    class Control:
        pass

    class ComponentHandle:
        resource_identity = f"native:{family_id}:" + "1" * 64
        component = Control()

        @staticmethod
        def require_active() -> None:
            pass

    class ComponentApplication:
        def __init__(
            self,
            family_id: str,
            role: str,
            handle: object,
            identity: str,
            materialize: object,
            **kwargs: object,
        ) -> None:
            self.family_id = family_id
            self.role = role
            self.handle = handle
            self.application_identity = identity
            self.materialize_application_kwargs = materialize
            self.resident_dependencies = kwargs["resident_dependencies"]

    class ApplicationChain:
        def __init__(self, model: object, applications: tuple[object, ...]) -> None:
            self.model = model
            self.applications = applications

        def append(self, application: object) -> ApplicationChain:
            return ApplicationChain(self.model, (*self.applications, application))

    inference = SimpleNamespace(
        ApplicationChain=ApplicationChain,
        ComponentApplication=ComponentApplication,
        MINIMAX_H3_CONFIG=SimpleNamespace(family_id=family_id),
        extend_runtime_identity=lambda _identity, _parts: f"native:{family_id}:" + "2" * 64,
    )
    inference_torch = SimpleNamespace(
        MiniMaxH3FunControl=Control,
        minimax_h3_fun_control_hint_digest=lambda _value: "3" * 64,
    )
    real_import = family.importlib.import_module
    monkeypatch.setattr(
        family.importlib,
        "import_module",
        lambda name: (
            inference
            if name == "dinkster_inference"
            else inference_torch
            if name == "dinkster_inference_torch"
            else real_import(name)
        ),
    )
    monkeypatch.setattr(family, "NativeComponentHandle", ComponentHandle)
    monkeypatch.setattr(
        family,
        "_native_handle",
        lambda _model, _name: SimpleNamespace(recipe=SimpleNamespace(family_id=family_id)),
    )
    vae = SimpleNamespace(resource_identity=f"native:{family_id}:" + "4" * 64)
    monkeypatch.setattr(family, "_minimax_h3_video_vae_runtime", lambda *_args: (vae, object()))
    monkeypatch.setattr(family, "_torch", _fake_torch)
    base = object()

    result = family.NativeApplyMiniMaxH3FunControlPatch.execute(
        model=base,
        model_patch=ComponentHandle(),
        vae=object(),
        strength=0.75,
        start_percent=0.1,
        end_percent=0.9,
        control_video=FakeTensor((2, 8, 8, 3)),
    )["model"]

    assert type(result) is ApplicationChain
    assert result.model is base
    assert len(result.applications) == 1
    application = cast("Any", result.applications[0])
    assert application.family_id == family_id
    assert application.role == "diffusion"
    assert application.resident_dependencies == (vae,)
    assert not hasattr(result, "minimax_h3_control")


def test_generic_av_concat_and_separate_preserve_roles_masks_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)

    def normalize_mask(
        mask: FakeTensor | MultiStreamLatent[FakeTensor],
        latent: MultiStreamLatent[FakeTensor],
    ) -> MultiStreamLatent[FakeTensor]:
        if type(mask) is FakeTensor:
            masks = {latent.roles[0]: mask}
        else:
            masks = {
                stream.role: stream.payload
                for stream in cast("MultiStreamLatent[FakeTensor]", mask).streams
            }
        return MultiStreamLatent.from_pairs(
            (
                stream.role,
                masks.get(stream.role, FakeTensor(stream.payload.shape)),
            )
            for stream in latent.streams
        )

    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch",
        SimpleNamespace(normalize_latent_mask=normalize_mask),
    )
    video_samples = FakeTensor((1, 24, 12, 32, 48))
    audio_samples = FakeTensor((1, 32, 2, 20))
    audio_mask = FakeTensor((1, 1, 2, 20))
    combined = native_arm.NativeConcatAVLatent.execute(
        video_latent={"samples": video_samples, "video_key": "video"},
        audio_latent={
            "samples": audio_samples,
            "noise_mask": audio_mask,
            "shared_key": "audio-wins",
        },
    )["latent"]
    assert isinstance(combined, dict)
    samples = combined["samples"]
    assert type(samples) is MultiStreamLatent
    assert samples.roles == ("video", "audio")
    assert samples.by_role("video") is video_samples
    assert samples.by_role("audio") is audio_samples
    masks = combined["noise_mask"]
    assert type(masks) is MultiStreamLatent
    assert masks.roles == ("video", "audio")
    assert masks.by_role("video").shape == video_samples.shape
    assert masks.by_role("audio") is audio_mask
    assert combined["shared_key"] == "audio-wins"

    separated = cast(
        "dict[str, dict[str, Any]]",
        native_arm.NativeSeparateAVLatent.execute(latent=combined),
    )
    assert separated["video_latent"]["samples"] is video_samples
    assert separated["audio_latent"]["samples"] is audio_samples
    assert separated["video_latent"]["noise_mask"].shape == video_samples.shape
    assert separated["audio_latent"]["noise_mask"] is audio_mask


def test_set_frame_mask_preserves_metadata_and_exact_role_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _fake_torch()
    monkeypatch.setattr(native_arm, "_torch", lambda: torch)
    monkeypatch.setattr(
        native_arm,
        "_latent_mask_codec_runtime",
        lambda *_args: SimpleNamespace(latent_mask_mapping=MINIMAX_H3_VIDEO_MASK_MAPPING),
    )
    video = FakeTensor((1, 24, 7, 2, 3))
    audio = FakeTensor((1, 32, 2, 20))
    source = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    content_mask = FakeTensor((22, 32, 48))
    converted = FakeTensor((7, 2, 3))
    expanded = FakeTensor(video.shape)
    calls: list[dict[str, object]] = []

    def normalize(
        mask: FakeTensor | MultiStreamLatent[FakeTensor],
        latent: MultiStreamLatent[FakeTensor],
    ) -> MultiStreamLatent[FakeTensor]:
        if type(mask) is FakeTensor:
            masks = {latent.roles[0]: mask}
        else:
            masks = {
                stream.role: stream.payload
                for stream in cast("MultiStreamLatent[FakeTensor]", mask).streams
            }
        return MultiStreamLatent.from_pairs(
            (
                stream.role,
                (
                    expanded
                    if masks.get(stream.role) is converted
                    else masks.get(stream.role, FakeTensor(stream.payload.shape))
                ),
            )
            for stream in latent.streams
        )

    fake_inference_torch = SimpleNamespace(
        content_mask_to_latent_mask=lambda mask, target, mapping, **kwargs: (
            calls.append({"mask": mask, "target": target, "mapping": mapping, **kwargs})
            or converted
        ),
        normalize_latent_mask=normalize,
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", fake_inference_torch)

    result = native_arm.NativeSetLatentMaskFromFrames.execute(
        latent={"samples": source, "metadata": "kept"},
        vae=object(),
        mask=content_mask,
        spatial_reduction="mean",
        temporal_reduction="last",
        operation="replace",
    )["latent"]

    assert isinstance(result, dict)
    assert result["samples"] is source
    assert result["metadata"] == "kept"
    masks = result["noise_mask"]
    assert type(masks) is MultiStreamLatent
    assert masks.roles == ("video", "audio")
    assert masks.by_role("video") is expanded
    assert masks.by_role("audio").shape == audio.shape
    assert calls == [
        {
            "mask": content_mask,
            "target": video,
            "mapping": MINIMAX_H3_VIDEO_MASK_MAPPING,
            "spatial_reduction": "mean",
            "temporal_reduction": "last",
        }
    ]


def test_compat_multistream_role_sidecar_round_trips_samples_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NestedTensor:
        def __init__(self, tensors: tuple[object, ...]) -> None:
            self.tensors = tensors

        def unbind(self) -> tuple[object, ...]:
            return self.tensors

    monkeypatch.setitem(
        sys.modules, "comfy.nested_tensor", SimpleNamespace(NestedTensor=NestedTensor)
    )
    samples = _streams()
    masks = MultiStreamLatent.from_pairs(
        (("video", FakeTensor((1, 1, 12, 32, 48))), ("audio", FakeTensor((1, 1, 2, 20))))
    )
    lowered = cast(
        "dict[str, Any]",
        comfy_translate.to_comfy_multistream(
            {"samples": samples, "noise_mask": masks, "metadata": "kept"}
        ),
    )
    assert type(lowered["samples"]) is NestedTensor
    assert type(lowered["noise_mask"]) is NestedTensor
    assert lowered["dinkster.multi_stream_roles@1"] == {
        "version": 1,
        "roles": ("video", "audio"),
    }

    restored = cast("dict[str, Any]", comfy_translate.from_comfy_multistream(lowered))
    assert type(restored["samples"]) is MultiStreamLatent
    assert restored["samples"].roles == ("video", "audio")
    assert type(restored["noise_mask"]) is MultiStreamLatent
    assert restored["noise_mask"].roles == ("video", "audio")
    assert "dinkster.multi_stream_roles@1" not in restored
    assert restored["metadata"] == "kept"

    legacy_concat = comfy_translate._declare_fixed_av_output(
        "LTXVConcatAVLatent",
        {"samples": NestedTensor((samples.by_role("video"), samples.by_role("audio")))},
    )
    restored_concat = cast("dict[str, Any]", comfy_translate.from_comfy_multistream(legacy_concat))
    assert restored_concat["samples"].roles == ("video", "audio")
    legacy_split = cast(
        "dict[str, Any]",
        comfy_translate._declare_fixed_av_output(
            "LTXVSeparateAVLatent",
            {"samples": samples.by_role("video"), "dinkster.multi_stream_roles@1": {}},
        ),
    )
    assert "dinkster.multi_stream_roles@1" not in legacy_split


def test_h3_candidate_path_does_not_hash_or_open_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.safetensors"

    class Resolver:
        def resolve(self, digest: str) -> Path:
            del digest
            return path

    resolver = Resolver()
    asset = AssetRef("blake3:" + "1" * 64, path.name, 1, resolver=resolver)
    monkeypatch.setattr(
        AssetRef,
        "local_path",
        lambda _asset: (_ for _ in ()).throw(AssertionError("candidate lookup must not hash")),
    )
    monkeypatch.setattr(
        AssetRef,
        "open",
        lambda _asset: (_ for _ in ()).throw(AssertionError("candidate lookup must not open")),
    )

    assert native_arm._component_candidate_path(asset) == path


def test_residency_accepts_exact_h3_units_and_rejects_bad_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged: list[tuple[object, ...]] = []
    unloaded: list[tuple[object, ...]] = []

    class Device:
        type = "cpu"

    class Coordinator:
        def enroll(self, _handle: object) -> None:
            pass

        def locked(self) -> Any:
            return nullcontext()

        def stage(self, _handle: object, mechanisms: tuple[object, ...], **_kwargs: object) -> Any:
            staged.append(tuple(mechanisms))
            return nullcontext()

        def unload_stage(
            self, _handle: object, mechanisms: tuple[object, ...], **_kwargs: object
        ) -> None:
            unloaded.append(tuple(mechanisms))

    class Mechanism:
        demand_paged = False

    names = ("fl2va_dit", "ref2va_dit", "conditioner", "video_vae", "audio_vae")
    mechanisms: dict[str, object] = {name: Mechanism() for name in names}

    class Policy:
        enrollment_components = names
        enrollment_orders = (
            ("fl2va_dit", "conditioner", "video_vae", "audio_vae"),
            ("ref2va_dit", "conditioner", "video_vae", "audio_vae"),
            names,
        )
        component_roles = {
            "fl2va_dit": "fl2va_dit",
            "ref2va_dit": "ref2va_dit",
            "conditioner": "text",
            "video_vae": "vae",
            "audio_vae": "vae",
        }
        diffusion_roles = frozenset({"fl2va_dit", "ref2va_dit"})
        unload_after_stage = frozenset({"vae"})
        resident_components = frozenset({"video_vae", "audio_vae"})

    policy = Policy()
    inference_torch = SimpleNamespace(NativeResidencyPolicy=Policy)
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: inference_torch if name == "dinkster_inference_torch" else real_import(name),
    )

    def construct(
        enrolled: dict[str, object],
        component_names: tuple[str, ...] = names,
    ) -> NativeRuntimeHandle:
        assembled = SimpleNamespace(
            **{name: mechanisms[name] if name in component_names else None for name in names}
        )
        return NativeRuntimeHandle(
            SimpleNamespace(
                runtime_identity="identity",
                assembled=assembled,
                residency_policy=policy,
            ),
            "cpu",
            recipe=cast("Any", SimpleNamespace(runtime_identity="identity", attachments=())),
            coordinator=cast("Any", Coordinator()),
            _torch_module=SimpleNamespace(device=lambda _: Device()),
            _enroll_assembled=cast("Any", lambda *_args, **_kwargs: enrolled),
        )

    handle = construct(mechanisms)
    assert handle.mechanisms == tuple(mechanisms.values())
    assert handle.__dict__["_unload_after_stage"] == frozenset({"vae"})
    assert handle.__dict__["_by_role"] == {
        "fl2va_dit": (mechanisms["fl2va_dit"],),
        "ref2va_dit": (mechanisms["ref2va_dit"],),
        "text": (mechanisms["conditioner"],),
        "vae": (mechanisms["video_vae"], mechanisms["audio_vae"]),
    }
    with handle.stage("fl2va_dit"):
        pass
    with handle.stage("ref2va_dit"):
        pass
    with handle.stage("text"):
        pass
    with handle.stage("vae", observer_stage="condition"):
        pass
    assert staged == [
        (mechanisms["fl2va_dit"],),
        (mechanisms["ref2va_dit"],),
        (mechanisms["conditioner"],),
        (mechanisms["video_vae"], mechanisms["audio_vae"]),
    ]
    assert unloaded == [(mechanisms["video_vae"], mechanisms["audio_vae"])]
    for component_names in (
        ("fl2va_dit", "conditioner", "video_vae", "audio_vae"),
        ("ref2va_dit", "conditioner", "video_vae", "audio_vae"),
    ):
        selected = {name: mechanisms[name] for name in component_names}
        single_role = construct(selected, component_names)
        assert single_role.mechanisms == tuple(selected.values())
        assert single_role.__dict__["_unload_after_stage"] == frozenset({"vae"})
        with single_role.stage("vae", observer_stage="condition"):
            pass
    common_names = ("conditioner", "video_vae", "audio_vae")
    with pytest.raises(
        ValueError,
        match="native runtime enrollment must return exactly .* in declaration order",
    ):
        construct({name: mechanisms[name] for name in common_names}, common_names)
    with pytest.raises(ValueError, match="must return exactly"):
        construct(dict(tuple(mechanisms.items())[:-1]))
    with pytest.raises(ValueError, match="must return exactly"):
        construct({**mechanisms, "unknown": object()})
    wrong_order = {
        name: mechanisms[name]
        for name in ("conditioner", "fl2va_dit", "ref2va_dit", "video_vae", "audio_vae")
    }
    with pytest.raises(ValueError, match="must return exactly"):
        construct(wrong_order)


def test_empty_returns_exact_ordered_multistream_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    streams = _streams()
    fake_module = SimpleNamespace(
        empty_minimax_h3_av=lambda **kwargs: calls.append(kwargs) or streams
    )
    real_import = importlib.import_module
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "dinkster_inference_torch" else real_import(name),
    )
    result = native_arm.NativeEmptyMiniMaxH3AV.execute(width=640, height=480, frame_count=22)
    assert result == {"latent": {"samples": streams}}
    assert cast("Any", result["latent"])["samples"].roles == ("video", "audio")
    assert calls == [
        {"width": 640, "height": 480, "frame_count": 22, "device": "cpu", "dtype": "bf16"}
    ]


def test_conditioning_returns_exactly_one_prepared_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    calls: list[dict[str, object]] = []
    stage_calls: list[str] = []

    class Runtime:
        def condition(self, request: object, **kwargs: object) -> str:
            calls.append({"request": request, **kwargs})
            return "prepared"

    conditioner = SimpleNamespace(
        load_device="cuda:0",
        resource_identity="native:dinkster.minimax_h3:" + "1" * 64,
        stage=lambda **_kwargs: stage_calls.append("conditioner") or nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda clip: (conditioner, (), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    result = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=object(), target=_latent(), prompt="prompt"
    )
    assert tuple(result) == ("conditioning",)
    conditioning = _conditioning_rows(result["conditioning"])
    assert len(conditioning) == 1 and conditioning[0][1] == {}
    carrier = cast("Any", conditioning[0][0])
    assert type(carrier) is inference.PreparedMultiStreamConditioning
    assert carrier.runtime_identity == conditioner.resource_identity
    assert carrier.payload == "prepared"
    resident = cast("Any", result["conditioning"])
    assert resident._dinkster_resident_owner is conditioner
    assert resident._dinkster_resident_refs == ()
    assert resident.bindings[0].fingerprint.startswith("minimax-h3-conditioning:")
    assert cast("Any", calls[0]["target"]).roles == ("video", "audio")
    assert all(
        stream.payload.device == "cuda:0" for stream in cast("Any", calls[0]["target"]).streams
    )
    assert calls[0]["frame_count"] == 39
    assert stage_calls == ["conditioner"]


def test_conditioning_adapts_an_ordinary_empty_latent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapted = MultiStreamLatent.from_pairs(
        (
            ("video", FakeTensor((1, 24, 1, 32, 48))),
            ("audio", FakeTensor((1, 32, 2, 2))),
        )
    )
    adaptations: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []

    class Runtime:
        def adapt_multistream_latent(self, latent: object, **kwargs: object) -> object:
            adaptations.append({"latent": latent, **kwargs})
            return adapted

        def condition(self, request: object, **kwargs: object) -> str:
            conditions.append({"request": request, **kwargs})
            return "prepared"

    conditioner = SimpleNamespace(
        load_device="cuda:0",
        resource_identity="native:dinkster.minimax_h3:" + "1" * 64,
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda clip: (conditioner, (), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    ordinary = FakeTensor((1, 4, 64, 96))

    result = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=object(),
        target={"samples": ordinary, "downscale_ratio_spacial": 8},
        prompt="prompt",
    )

    assert tuple(result) == ("conditioning",)
    assert adaptations == [
        {
            "latent": ordinary,
            "source_spatial_downscale": 8,
            "source_temporal_downscale": None,
        }
    ]
    assert cast("Any", conditions[0]["target"]).roles == ("video", "audio")
    assert conditions[0]["frame_count"] == 1


@pytest.mark.parametrize("task", ("fl2va", "ref2va"))
def test_task_specific_conditioning_adapts_target_before_geometry(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    adapted = MultiStreamLatent.from_pairs(
        (
            ("video", FakeTensor((1, 24, 1, 32, 48))),
            ("audio", FakeTensor((1, 32, 2, 2))),
        )
    )
    adaptations: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []

    class Runtime:
        def adapt_multistream_latent(self, latent: object, **kwargs: object) -> object:
            adaptations.append({"latent": latent, **kwargs})
            return adapted

        def condition(self, request: object, **kwargs: object) -> str:
            conditions.append({"request": request, **kwargs})
            return "prepared"

    conditioner = SimpleNamespace(
        load_device="cuda:0",
        resource_identity="native:dinkster.minimax_h3:" + "1" * 64,
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda *_args: (conditioner, (), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    _install_upscale(monkeypatch)
    ordinary = FakeTensor((1, 4, 64, 96))
    target = {"samples": ordinary, "downscale_ratio_spacial": 8}

    if task == "fl2va":
        result = native_arm.NativeMiniMaxH3FL2VAConditioning.execute(
            clip=object(),
            video_vae=object(),
            target=target,
            prompt="prompt",
            first_image=FakeTensor((1, 512, 768, 3)),
        )
    else:
        result = native_arm.NativeMiniMaxH3REF2VAConditioning.execute(
            clip=object(),
            video_vae=object(),
            audio_vae=object(),
            target=target,
            prompt="prompt",
            references=(MiniMaxH3ImageReferenceValue(FakeTensor((1, 512, 768, 3))),),
            ref_image_size="match",
        )

    assert tuple(result) == ("conditioning",)
    assert adaptations == [
        {
            "latent": ordinary,
            "source_spatial_downscale": 8,
            "source_temporal_downscale": None,
        }
    ]
    assert cast("Any", conditions[0]["target"]).roles == ("video", "audio")
    assert conditions[0]["frame_count"] == 1


def test_fl2va_conditioning_fingerprint_changes_with_one_keyframe_pixel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = "native:dinkster.minimax_h3:" + "1" * 64

    class Runtime:
        @staticmethod
        def condition(request: object, **_kwargs: object) -> object:
            return SimpleNamespace(task=cast("Any", request).task)

    conditioner = SimpleNamespace(
        load_device="cpu",
        resource_identity=identity,
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda *_args: (conditioner, (), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(native_arm, "_minimax_h3_resize", lambda value, *_args: value)
    first = bytes(32 * 32 * 3)
    second = bytes((1,)) + first[1:]

    fingerprints = tuple(
        cast("Any", result["conditioning"])._dinkster_resident_fingerprint
        for result in (
            native_arm.NativeMiniMaxH3FL2VAConditioning.execute(
                clip=object(),
                video_vae=object(),
                target=_latent(),
                prompt="same prompt",
                first_image=FakeTensor((1, 32, 32, 3), content=content),
            )
            for content in (first, second)
        )
    )

    assert fingerprints[0] != fingerprints[1]


def test_two_conditioning_nodes_prepare_two_independent_prompt_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    calls: list[dict[str, object]] = []
    identity = "native:dinkster.minimax_h3:" + "1" * 64

    class Runtime:
        def condition(self, request: object, **kwargs: object) -> object:
            calls.append({"request": request, **kwargs})
            return SimpleNamespace(
                task=cast("Any", request).task,
                prompt=cast("Any", request).prompt,
            )

    conditioner = SimpleNamespace(
        load_device="cuda:0",
        resource_identity=identity,
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda clip: (conditioner, (), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)

    clip = object()
    target = _latent()
    positive_result = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=clip,
        target=target,
        prompt="a singer",
    )
    negative_result = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=clip,
        target=target,
        prompt="off-key vocals",
    )

    assert [cast("Any", call["request"]).prompt for call in calls] == [
        "a singer",
        "off-key vocals",
    ]
    assert all(type(call["request"]) is inference.MiniMaxH3T2VARequest for call in calls)
    positive = cast("Any", _conditioning_rows(positive_result["conditioning"])[0][0])
    negative = cast("Any", _conditioning_rows(negative_result["conditioning"])[0][0])
    assert positive.runtime_identity == negative.runtime_identity == identity
    assert positive.payload is not negative.payload
    assert positive.payload.prompt == "a singer"
    assert negative.payload.prompt == "off-key vocals"
    assert (
        cast("Any", positive_result["conditioning"])._dinkster_resident_fingerprint
        != cast("Any", negative_result["conditioning"])._dinkster_resident_fingerprint
    )


@pytest.mark.parametrize("task", ("t2v", "i2v", "r2v"))
def test_h3_task_seam_reaches_sampler_through_resident_wire(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    identity = "native:dinkster.minimax_h3:" + "1" * 64
    sampled = _streams()
    sampler_calls: list[dict[str, object]] = []

    class Handle:
        load_device = "cpu"
        resource_identity = identity

        @staticmethod
        def stage(**_kwargs: object) -> nullcontext[None]:
            return nullcontext()

    class ConditionerRuntime:
        @staticmethod
        def condition(request: object, **_kwargs: object) -> object:
            return SimpleNamespace(task=cast("Any", request).task)

    class SamplingRuntime:
        runtime_identity = identity
        conditioning_identity = identity
        model_role = "ref2va_dit" if task == "r2v" else "fl2va_dit"

        @staticmethod
        def run_ksampler_as_custom(latent: object, **kwargs: object) -> object:
            sampler_calls.append({"latent": latent, **kwargs})
            return sampled

        sample_multistream = run_ksampler_as_custom

    conditioner = Handle()
    codecs = () if task == "t2v" else (Handle(),) if task == "i2v" else (Handle(), Handle())
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda *_args: (conditioner, codecs, ConditionerRuntime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    _install_upscale(monkeypatch)
    image = FakeTensor((1, 32, 32, 3))
    if task == "t2v":
        conditioned = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
            clip=object(), target=_latent(), prompt="prompt"
        )
    elif task == "i2v":
        conditioned = native_arm.NativeMiniMaxH3FL2VAConditioning.execute(
            clip=object(),
            video_vae=object(),
            target=_latent(),
            prompt="prompt",
            first_image=image,
        )
    else:
        conditioned = native_arm.NativeMiniMaxH3REF2VAConditioning.execute(
            clip=object(),
            video_vae=object(),
            audio_vae=object(),
            target=_latent(),
            prompt="prompt",
            references=(MiniMaxH3ImageReferenceValue(image),),
            ref_image_size="match",
        )

    resident = cast("Any", conditioned["conditioning"])
    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=ResidencyTable())
    wrapped = registry.wrap(CONDITIONING_TYPE_ID, resident)
    encoded = spec.encode(wrapped.resolve())
    restored = spec.decode(encoded)
    assert restored is resident
    assert wrapped.fingerprint == resident._dinkster_resident_fingerprint
    assert resident._dinkster_resident_owner is conditioner
    assert resident._dinkster_resident_refs == codecs

    model_handle = SimpleNamespace(
        load_device="cpu",
        runtime=SamplingRuntime(),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda _role, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (model_handle, (), {}, None, None, (), None, ()),
    )
    _install_sampling_runtime(monkeypatch, SamplingRuntime())
    monkeypatch.setattr(native_arm, "_catalog_id", lambda _registry, value, _kind: value)
    result = native_arm.NativeKSampler.execute(
        model=object(),
        seed=7,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=restored,
        negative=[],
        latent_image=_latent(),
        denoise=1.0,
    )
    assert cast("Any", result["latent"])["samples"] is sampled
    assert len(sampler_calls) == 1
    expected_task = {"t2v": "t2va", "i2v": "fl2va", "r2v": "ref2va"}[task]
    assert cast("Any", sampler_calls[0]["conditioning"]).task.value == expected_task


def test_conditioning_stages_each_participating_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    stage_calls: list[str] = []

    class Runtime:
        def condition(self, request: object, **_kwargs: object) -> object:
            assert type(request) is inference.MiniMaxH3FL2VARequest
            return SimpleNamespace(task=inference.MiniMaxH3Task.FL2VA)

    conditioner = SimpleNamespace(
        load_device="cuda:0",
        resource_identity="native:dinkster.minimax_h3:" + "1" * 64,
        stage=lambda **_kwargs: stage_calls.append("conditioner") or nullcontext(),
    )
    video_vae = SimpleNamespace(
        stage=lambda **_kwargs: stage_calls.append("video_vae") or nullcontext()
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda clip, video: (conditioner, (video_vae,), Runtime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    _install_upscale(monkeypatch)
    native_arm.NativeMiniMaxH3FL2VAConditioning.execute(
        clip=object(),
        video_vae=object(),
        target=_latent(),
        prompt="prompt",
        first_image=FakeTensor((1, 512, 768, 3)),
    )
    assert stage_calls == ["conditioner", "video_vae"]


def test_add_guide_trims_resamples_crops_and_chains_positive_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    family = importlib.import_module("dinkster_native.families.minimax_h3")
    identity = "native:dinkster.minimax_h3:" + "1" * 64
    target = MultiStreamLatent.from_pairs(
        (
            ("video", FakeTensor((1, 24, 7, 2, 3))),
            ("audio", FakeTensor((1, 32, 2, 37))),
        )
    )
    initial = SimpleNamespace(guides=())
    rows = [[inference.PreparedMultiStreamConditioning(identity, initial), {}]]
    positive: object = inference.ResidentConditioningCarrier(
        family._MiniMaxH3ResidentConditioning(rows, object(), (), "test:h3:conditioning")
    )
    stage_calls: list[str] = []
    video_inputs: list[FakeTensor] = []
    video_outputs: list[FakeTensor] = []
    audio_inputs: list[object] = []
    adapter_calls: list[tuple[object, object]] = []
    resamples: list[tuple[tuple[int, ...], int]] = []

    class VideoRuntime:
        def encode_video(self, value: FakeTensor) -> FakeTensor:
            video_inputs.append(value)
            temporal = 1 if value.shape[2] == 1 else 2
            output = FakeTensor((1, 24, temporal, 2, 3), value.device)
            video_outputs.append(output)
            return output

    class AudioRuntime:
        def encode_audio(self, value: object) -> FakeTensor:
            audio_inputs.append(value)
            return FakeTensor((1, 32, 2, 50), cast("Any", value).waveform.device)

    video_handle = SimpleNamespace(
        load_device="cuda:0",
        stage=lambda **_kwargs: stage_calls.append("video") or nullcontext(),
    )
    audio_handle = SimpleNamespace(
        load_device="cuda:1",
        stage=lambda **_kwargs: stage_calls.append("audio") or nullcontext(),
    )

    def add_guide(prepared: object, selected_target: object, guide: object) -> object:
        adapter_calls.append((selected_target, guide))
        return SimpleNamespace(guides=(*cast("Any", prepared).guides, guide))

    def resample(value: FakeTensor, sample_rate: int) -> FakeTensor:
        resamples.append((value.shape, sample_rate))
        return FakeTensor((1, 2, 32_000), value.device)

    fake_torch_module = SimpleNamespace(add_minimax_h3_timeline_guide=add_guide)
    real_import = importlib.import_module
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_video_vae_runtime",
        lambda *_args: (video_handle, VideoRuntime()),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_audio_vae_runtime",
        lambda *_args: (audio_handle, AudioRuntime()),
    )
    monkeypatch.setattr(native_arm, "_minimax_h3_resample_audio", resample)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_torch_module if name == "dinkster_inference_torch" else real_import(name),
    )
    resize_calls = _install_upscale(monkeypatch)

    first = native_arm.NativeMiniMaxH3AddGuide.execute(
        positive=positive,
        vae=object(),
        audio_vae=object(),
        latent={"samples": target},
        image=FakeTensor((10, 64, 96, 3)),
        audio={"waveform": FakeTensor((2, 2, 44_100)), "sample_rate": 44_100},
        frame_idx=6,
    )
    second = native_arm.NativeMiniMaxH3AddGuide.execute(
        positive=first["positive"],
        vae=object(),
        latent={"samples": target},
        image=FakeTensor((3, 64, 96, 3)),
        frame_idx=-1,
    )

    first_guide = cast("Any", adapter_calls[0][1])
    second_guide = cast("Any", adapter_calls[1][1])
    assert adapter_calls[0][0] is adapter_calls[1][0] is target
    assert (first_guide.frame_index, first_guide.frame_count) == (6, 5)
    assert first_guide.latent.roles == ("video", "audio")
    assert first_guide.latent.by_role("video").shape == (1, 24, 2, 2, 3)
    assert first_guide.latent.by_role("video").device == target.by_role("video").device
    assert first_guide.latent.by_role("video").dtype == video_outputs[0].dtype
    assert first_guide.latent.by_role("audio").shape == (1, 32, 2, 27)
    assert (second_guide.frame_index, second_guide.frame_count) == (21, 1)
    assert second_guide.latent.roles == ("video",)
    assert second_guide.latent.by_role("video").device == target.by_role("video").device
    assert second_guide.latent.by_role("video").dtype == video_outputs[1].dtype
    assert [value.shape for value in video_inputs] == [
        (1, 3, 5, 32, 48),
        (1, 3, 1, 32, 48),
    ]
    assert cast("Any", audio_inputs[0]).waveform.shape == (1, 2, 32_000)
    assert cast("Any", audio_inputs[0]).sample_rate == 32_000
    assert resamples == [((1, 2, 44_100), 44_100)]
    assert resize_calls == [(48, 32, "center"), (48, 32, "center")]
    assert stage_calls == ["video", "audio", "video"]
    first_carrier = cast("Any", first["positive"])
    second_carrier = cast("Any", second["positive"])
    assert type(first_carrier) is inference.ConditioningCarrier
    assert type(second_carrier) is inference.ConditioningCarrier
    assert first_carrier._dinkster_resident_fingerprint != "test:h3:conditioning"
    assert (
        second_carrier._dinkster_resident_fingerprint
        != first_carrier._dinkster_resident_fingerprint
    )
    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=ResidencyTable())
    wrapped = registry.wrap(CONDITIONING_TYPE_ID, second_carrier)
    assert spec.decode(spec.encode(wrapped.resolve())) is second_carrier
    output = _resident_payload(second_carrier, inference).conditioning[0][0]
    assert output.runtime_identity == identity
    assert len(output.payload.guides) == 2


def test_motion_context_delegates_av_tail_selection_and_preserves_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    family = importlib.import_module("dinkster_native.families.minimax_h3")
    identity = "native:dinkster.minimax_h3:" + "1" * 64
    initial = object()
    rows = [[inference.PreparedMultiStreamConditioning(identity, initial), {}]]
    positive: object = inference.ResidentConditioningCarrier(
        family._MiniMaxH3ResidentConditioning(rows, object(), (), "test:h3:motion")
    )
    target = _streams()
    previous = _streams()
    calls: list[tuple[object, object, object, int]] = []

    def add_motion_context(
        prepared: object,
        selected_target: object,
        selected_previous: object,
        context_length: int,
    ) -> tuple[str, float]:
        calls.append((prepared, selected_target, selected_previous, context_length))
        return "continued", 22 / 24

    fake_torch_module = SimpleNamespace(add_minimax_h3_motion_context=add_motion_context)
    real_import = importlib.import_module
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_torch_module if name == "dinkster_inference_torch" else real_import(name),
    )

    result = native_arm.NativeMiniMaxH3MotionContext.execute(
        positive=positive,
        latent={"samples": target},
        previous_latent={"samples": previous},
        context_length=22,
    )

    assert calls == [(initial, target, previous, 22)]
    assert result["trim_time"] == 22 / 24
    carrier = cast("Any", result["positive"])
    assert type(carrier) is inference.ConditioningCarrier
    assert carrier._dinkster_resident_fingerprint != "test:h3:motion"
    output = _resident_payload(carrier, inference).conditioning[0][0]
    assert output.runtime_identity == identity
    assert output.payload == "continued"


def test_every_h3_conditioning_output_is_valid_on_the_production_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    identity = "native:dinkster.minimax_h3:" + "1" * 64

    class ConditionerRuntime:
        @staticmethod
        def condition(request: object, **_kwargs: object) -> object:
            return SimpleNamespace(task=cast("Any", request).task)

    class VideoRuntime:
        @staticmethod
        def encode_video(_value: object) -> FakeTensor:
            return FakeTensor((1, 24, 1, 32, 48))

    owner = SimpleNamespace(
        load_device="cpu",
        resource_identity=identity,
        stage=lambda **_kwargs: nullcontext(),
    )
    codecs = (owner,)
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda *_args: (owner, codecs, ConditionerRuntime()),
    )
    monkeypatch.setattr(native_arm, "_empty_minimax_h3_target", lambda *_args: _latent())
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_video_vae_runtime",
        lambda *_args: (owner, VideoRuntime()),
    )
    _install_upscale(monkeypatch)
    inference_torch = SimpleNamespace(
        add_minimax_h3_timeline_guide=lambda _prepared, _target, _guide: object(),
        add_minimax_h3_motion_context=lambda _prepared, _target, _previous, length: (
            object(),
            length / 24,
        ),
    )
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: inference_torch if name == "dinkster_inference_torch" else real_import(name),
    )

    image = FakeTensor((1, 32, 32, 3))
    latent = _latent()
    results = {
        "dinkster.minimax_h3_t2va_conditioning": (
            native_arm.NativeMiniMaxH3T2VAConditioning.execute(
                clip=owner, target=latent, prompt="prompt"
            )["conditioning"]
        ),
        "dinkster.minimax_h3_fl2va_conditioning": (
            native_arm.NativeMiniMaxH3FL2VAConditioning.execute(
                clip=owner,
                video_vae=owner,
                target=latent,
                prompt="prompt",
                first_image=image,
            )["conditioning"]
        ),
        "dinkster.minimax_h3_ref2va_conditioning": (
            native_arm.NativeMiniMaxH3REF2VAConditioning.execute(
                clip=owner,
                video_vae=owner,
                audio_vae=owner,
                target=latent,
                prompt="prompt",
                references=(MiniMaxH3ImageReferenceValue(image),),
                ref_image_size="match",
            )["conditioning"]
        ),
        "dinkster.minimax_h3_image_to_video": native_arm.NativeMiniMaxH3ImageToVideo.execute(
            clip=owner,
            vae=owner,
            prompt="prompt",
            width=32,
            height=32,
            length=5,
        )["positive"],
        "dinkster.minimax_h3_reference_to_video": (
            native_arm.NativeMiniMaxH3ReferenceToVideo.execute(
                clip=owner,
                vae=owner,
                audio_vae=owner,
                prompt="prompt",
                width=32,
                height=32,
                length=5,
                ref_image_size="match",
                ref_images={"ref_image_1": image},
                ref_videos={},
                ref_video_audios={},
                ref_audios={},
            )["positive"]
        ),
    }
    results["dinkster.minimax_h3_add_guide"] = native_arm.NativeMiniMaxH3AddGuide.execute(
        positive=results["dinkster.minimax_h3_reference_to_video"],
        latent=latent,
        vae=owner,
        image=image,
        frame_idx=0,
    )["positive"]
    results["dinkster.minimax_h3_motion_context"] = native_arm.NativeMiniMaxH3MotionContext.execute(
        positive=results["dinkster.minimax_h3_add_guide"],
        latent=latent,
        previous_latent=latent,
        context_length=5,
    )["positive"]

    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=ResidencyTable())
    assert spec.coerce is not None
    for value in results.values():
        coerced = spec.coerce(value)
        wrapped = registry.wrap(CONDITIONING_TYPE_ID, coerced)
        assert spec.decode(spec.encode(wrapped.resolve())) is value

    with pytest.raises(TypeError, match="requires a resident carrier"):
        native_arm._minimax_h3_rewrap_conditioning([], [], inference, "test")
    unstable = SimpleNamespace(
        streams=(
            SimpleNamespace(
                role="video",
                payload=SimpleNamespace(shape=(1,), dtype="torch.float32"),
            ),
        )
    )
    with pytest.raises(TypeError, match="stable tensor bytes"):
        native_arm._minimax_h3_latent_fingerprint(unstable, _fake_torch())

    assert set(results) == {
        "dinkster.minimax_h3_t2va_conditioning",
        "dinkster.minimax_h3_fl2va_conditioning",
        "dinkster.minimax_h3_ref2va_conditioning",
        "dinkster.minimax_h3_image_to_video",
        "dinkster.minimax_h3_reference_to_video",
        "dinkster.minimax_h3_add_guide",
        "dinkster.minimax_h3_motion_context",
    }


def test_add_guide_requires_content_and_the_matching_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    identity = "native:dinkster.minimax_h3:" + "1" * 64
    positive = [[inference.PreparedMultiStreamConditioning(identity, object()), {}]]
    latent = {"samples": _streams()}
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch", SimpleNamespace())

    with pytest.raises(ValueError, match="require an image or audio"):
        native_arm.NativeMiniMaxH3AddGuide.execute(
            positive=positive,
            latent=latent,
            frame_idx=0,
        )
    with pytest.raises(ValueError, match="requires the vae input"):
        native_arm.NativeMiniMaxH3AddGuide.execute(
            positive=positive,
            latent=latent,
            image=FakeTensor((1, 32, 32, 3)),
            frame_idx=0,
        )
    with pytest.raises(ValueError, match="requires the audio_vae input"):
        native_arm.NativeMiniMaxH3AddGuide.execute(
            positive=positive,
            latent=latent,
            audio={"waveform": FakeTensor((1, 2, 800)), "sample_rate": 32_000},
            frame_idx=0,
        )


def test_component_handle_validation_refuses_wrong_role_family_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = "native:dinkster.minimax_h3:" + "1" * 64

    class Handle:
        def __init__(self, *, family: str, role: str, resource_identity: str = identity) -> None:
            self.resource_identity = resource_identity
            self.recipe = SimpleNamespace(
                family_id=family,
                runtime_identity=resource_identity,
                sources=(SimpleNamespace(role=role),),
            )

        def require_active(self) -> None:
            pass

    monkeypatch.setattr(native_arm, "NativeComponentHandle", Handle)
    valid = Handle(family="dinkster.minimax_h3", role="video-vae")
    family_id = "dinkster.minimax_h3"
    assert (
        native_arm.load_registered_component(valid, "video_vae", "video-vae", family_id=family_id)
        is valid
    )
    for value in (
        object(),
        Handle(family="dinkster.other", role="video-vae"),
        Handle(family="dinkster.minimax_h3", role="audio-vae"),
        Handle(
            family="dinkster.minimax_h3",
            role="video-vae",
            resource_identity="native:dinkster.minimax_h3:not-a-digest",
        ),
    ):
        with pytest.raises(
            TypeError,
            match="video_vae must be a native MiniMax H3 video VAE component",
        ):
            native_arm.load_registered_component(
                value, "video_vae", "video-vae", family_id=family_id
            )


def test_h3_model_admission_requires_the_exact_model_and_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    identity = "native:dinkster.minimax_h3:" + "1" * 64

    class MiniMaxH3DiTRuntime:
        runtime_identity = identity
        model_role = "fl2va-dit"

    class DuckModel:
        runtime_identity = identity
        model_role = "fl2va-dit"

    class Handle:
        def __init__(self, runtime: object, family_id: str = "dinkster.minimax_h3") -> None:
            self.runtime = runtime
            self.recipe = SimpleNamespace(
                family_id=family_id,
                runtime_identity=identity,
                sources=(SimpleNamespace(role="diffusion"),),
            )

        def require_active(self) -> None:
            pass

    fake_module = SimpleNamespace(MiniMaxH3DiTRuntime=MiniMaxH3DiTRuntime)
    real_import = importlib.import_module
    monkeypatch.setattr(native_arm, "NativeRuntimeHandle", Handle)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "dinkster_inference_torch" else real_import(name),
    )

    exact = Handle(MiniMaxH3DiTRuntime())
    assert native_arm._minimax_h3_model_handle(exact, inference) is exact
    assert native_arm._minimax_h3_model_handle(Handle(DuckModel()), inference) is None
    assert (
        native_arm._minimax_h3_model_handle(
            Handle(MiniMaxH3DiTRuntime(), family_id="dinkster.other"), inference
        )
        is None
    )


@pytest.mark.parametrize("task_name", ("T2VA", "REF2VA"))
def test_ordinary_ksampler_stages_standalone_h3_dit_and_forwards_sampling(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    inference = __import__("dinkster_inference")
    sampled = _streams()
    calls: list[tuple[object, dict[str, object]]] = []
    stage_calls: list[str] = []
    prepared = SimpleNamespace(task=getattr(inference.MiniMaxH3Task, task_name))
    conditioner_identity = "native:dinkster.minimax_h3:" + "1" * 64

    class Runtime:
        def run_ksampler_as_custom(self, latent: object, **kwargs: object) -> object:
            calls.append((latent, kwargs))
            return sampled

        sample_multistream = run_ksampler_as_custom

    handle = SimpleNamespace(
        load_device="cuda:0",
        runtime=SimpleNamespace(model_role="fl2va-dit"),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda role, **_kwargs: stage_calls.append(role) or nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (handle, (), {}, None, None, (), None, ()),
    )
    _install_sampling_runtime(monkeypatch, Runtime())
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(native_arm, "_catalog_id", lambda _registry, value, _kind: value)
    conditioned = [[inference.PreparedMultiStreamConditioning(conditioner_identity, prepared), {}]]
    latent = {
        "samples": _streams(),
        "noise_mask": FakeTensor((1, 1, 12, 32, 48)),
        "metadata": "preserved",
    }
    result = native_arm.NativeKSampler.execute(
        model=object(),
        seed=7,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=conditioned,
        negative=[],
        latent_image=latent,
        denoise=0.75,
    )
    assert cast("Any", result["latent"])["samples"] is sampled
    assert cast("Any", result["latent"])["metadata"] == "preserved"
    assert all(stream.payload.device == "cuda:0" for stream in cast("Any", calls[0][0]).streams)
    assert cast("Any", calls[0][1]["denoise_mask"]).device == "cuda:0"
    assert calls[0][1]["conditioning"] is prepared
    assert calls[0][1]["sampler_id"] == "euler"
    assert calls[0][1]["scheduler_id"] == "simple"
    assert calls[0][1]["steps"] == 4
    assert calls[0][1]["seed"] == 7
    assert stage_calls == ["diffusion"]


def test_ordinary_ksampler_adapts_empty_latent_for_h3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    adapted = MultiStreamLatent.from_pairs(
        (
            ("video", FakeTensor((1, 24, 1, 32, 48))),
            ("audio", FakeTensor((1, 32, 2, 2))),
        )
    )
    sampled = _streams()
    adaptations: list[dict[str, object]] = []
    sampling: list[object] = []

    class Runtime:
        def adapt_multistream_latent(self, latent: object, **kwargs: object) -> object:
            adaptations.append({"latent": latent, **kwargs})
            return adapted

        def run_ksampler_as_custom(self, latent: object, **_kwargs: object) -> object:
            sampling.append(latent)
            return sampled

        sample_multistream = run_ksampler_as_custom

    handle = SimpleNamespace(
        load_device="cuda:0",
        runtime=SimpleNamespace(model_role="fl2va-dit"),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda _role, **_kwargs: nullcontext(),
    )
    runtime = Runtime()
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (handle, (), {}, None, None, (), None, ()),
    )
    _install_sampling_runtime(monkeypatch, runtime)
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    monkeypatch.setattr(native_arm, "_catalog_id", lambda _registry, value, _kind: value)
    identity = "native:dinkster.minimax_h3:" + "1" * 64
    conditioned = [
        [
            inference.PreparedMultiStreamConditioning(
                identity,
                SimpleNamespace(task=inference.MiniMaxH3Task.T2VA),
            ),
            {},
        ]
    ]
    ordinary = FakeTensor((1, 4, 64, 96))

    result = native_arm.NativeKSampler.execute(
        model=object(),
        seed=7,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=conditioned,
        negative=[],
        latent_image={
            "samples": ordinary,
            "downscale_ratio_spacial": 8,
            "metadata": "preserved",
        },
        denoise=0.75,
    )

    assert adaptations == [
        {
            "latent": ordinary,
            "source_spatial_downscale": 8,
            "source_temporal_downscale": None,
        }
    ]
    assert cast("Any", sampling[0]).roles == ("video", "audio")
    assert all(stream.payload.device == "cuda:0" for stream in cast("Any", sampling[0]).streams)
    output = cast("dict[str, object]", result["latent"])
    assert output["samples"] is sampled
    assert output["metadata"] == "preserved"
    assert "downscale_ratio_spacial" not in output


def test_provider_ksampler_samples_h3_with_prepared_guidance_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioner_identity = "native:dinkster.minimax_h3:" + "1" * 64

    class ConditionerRuntime:
        def condition(self, request: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                task=cast("Any", request).task,
                prompt=cast("Any", request).prompt,
            )

    conditioner = SimpleNamespace(
        load_device="cpu",
        resource_identity=conditioner_identity,
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_conditioner_runtime",
        lambda clip: (conditioner, (), ConditionerRuntime()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    positive = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=object(),
        target=_latent(),
        prompt="a singer",
    )["conditioning"]
    negative = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
        clip=object(),
        target=_latent(),
        prompt="off-key vocals",
    )["conditioning"]

    calls: list[dict[str, object]] = []

    class SamplingRuntime:
        def run_ksampler_as_custom(self, latent: object, **kwargs: object) -> object:
            assert cast("Any", latent).roles == ("video", "audio")
            calls.append(kwargs)
            return latent

        sample_multistream = run_ksampler_as_custom

    model_handle = SimpleNamespace(
        load_device="cpu",
        runtime=SimpleNamespace(family=None),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda _role, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (model_handle, (), {}, None, None, (), None, ()),
    )
    monkeypatch.setattr(native_arm, "_minimax_h3_model_handle", lambda *_args: model_handle)
    _install_sampling_runtime(monkeypatch, SamplingRuntime())
    monkeypatch.setattr(native_arm, "_catalog_id", lambda _registry, value, _kind: value)

    result = native_arm.GenerationKSampler.execute(
        model=model_handle,
        seed=7,
        steps=2,
        cfg=2.5,
        sampler_name="euler",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=_latent(),
        denoise=1.0,
    )

    assert cast("Any", result["latent"])["samples"].roles == ("video", "audio")
    assert cast("Any", calls[0]["conditioning"]).prompt == "a singer"
    guidance = cast("Any", calls[0]["cfg"])
    assert guidance.uncond.prompt == "off-key vocals"
    assert guidance.scale == 2.5


def test_ksampler_forwards_cfg_and_validates_conditioner_identity_and_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    sampled = _streams()
    calls: list[dict[str, object]] = []
    conditioner_identity = "native:dinkster.minimax_h3:" + "1" * 64
    other_conditioner_identity = "native:dinkster.minimax_h3:" + "2" * 64
    prepared = SimpleNamespace(task=inference.MiniMaxH3Task.T2VA)
    uncond = SimpleNamespace(task=inference.MiniMaxH3Task.T2VA)

    class Runtime:
        def run_ksampler_as_custom(self, latent: object, **kwargs: object) -> object:
            if cast("Any", latent).roles != ("video", "audio"):
                raise TypeError("H3 sampling requires ordered video/audio streams")
            calls.append(kwargs)
            return sampled

        sample_multistream = run_ksampler_as_custom

    handle = SimpleNamespace(
        load_device="cpu",
        runtime=SimpleNamespace(model_role="fl2va-dit"),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda _role, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (handle, (), {}, None, None, (), None, ()),
    )
    _install_sampling_runtime(monkeypatch, Runtime())
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    conditioned = [[inference.PreparedMultiStreamConditioning(conditioner_identity, prepared), {}]]
    negative_conditioned = [
        [inference.PreparedMultiStreamConditioning(conditioner_identity, uncond), {}]
    ]

    def sample(**changes: object) -> None:
        arguments = {
            "model": object(),
            "seed": 0,
            "steps": 1,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": conditioned,
            "negative": [],
            "latent_image": _latent(),
            "denoise": 1.0,
        }
        arguments.update(changes)
        native_arm.NativeKSampler.execute(**cast("Any", arguments))

    sample(cfg=2.5, negative=negative_conditioned)
    assert cast("Any", calls[-1]["cfg"]).scale == 2.5
    assert cast("Any", calls[-1]["cfg"]).uncond is uncond
    sample()
    assert cast("Any", calls[-1]["cfg"]).scale == 1.0
    assert cast("Any", calls[-1]["cfg"]).uncond is None
    with pytest.raises(TypeError, match="prepared multi-stream"):
        sample(negative=[[object(), {}]])
    wrong_family = [
        [
            inference.PreparedMultiStreamConditioning(
                "native:dinkster.other:" + "3" * 64,
                prepared,
            ),
            {},
        ]
    ]
    with pytest.raises(ValueError, match="different runtime"):
        sample(positive=wrong_family)
    malformed = [[inference.PreparedMultiStreamConditioning("identity", prepared), {}]]
    with pytest.raises(ValueError, match="different runtime"):
        sample(positive=malformed)
    different_negative = [
        [
            inference.PreparedMultiStreamConditioning(other_conditioner_identity, uncond),
            {},
        ]
    ]
    with pytest.raises(ValueError, match="different runtime"):
        sample(negative=different_negative)
    with pytest.raises(TypeError, match="ordered video/audio"):
        sample(latent_image=_latent(reverse=True))
    assert len(calls) == 2


def test_h3_dit_runtime_composes_execution_identity_and_preserves_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    calls: list[str] = []
    dit_identity = "native:dinkster.minimax_h3:" + "1" * 64
    conditioner_identity = "native:dinkster.minimax_h3:" + "2" * 64

    class Runtime:
        def __init__(self) -> None:
            self.assembled = SimpleNamespace(diffusion="diffusion")
            self.model_role = "fl2va-dit"
            self.runtime_identity = dit_identity
            self.receipt_identity = "receipt"

        def with_conditioner(self, identity: str) -> object:
            calls.append(identity)
            return SimpleNamespace(base=self, conditioning_identity=identity)

    fake_module = SimpleNamespace(MiniMaxH3DiTRuntime=Runtime)
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "dinkster_inference_torch" else real_import(name),
    )
    handle = SimpleNamespace(
        runtime=Runtime(),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            runtime_identity=dit_identity,
            sources=(SimpleNamespace(role="diffusion"),),
            knobs=SimpleNamespace(diffusion_dtype="bfloat16"),
        ),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_model_handle",
        lambda value, _inference: value if type(value.runtime) is Runtime else None,
    )
    runtime = native_arm._minimax_h3_dit_runtime(
        cast("Any", handle),
        conditioner_identity,
        inference,
    )
    assert runtime.base is handle.runtime
    assert runtime.conditioning_identity == conditioner_identity
    assert calls == [conditioner_identity]
    handle.runtime = object()
    with pytest.raises(TypeError, match="standalone DiT component model"):
        native_arm._minimax_h3_dit_runtime(
            cast("Any", handle),
            conditioner_identity,
            inference,
        )


def test_wrong_role_h3_dit_error_surfaces_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = __import__("dinkster_inference")
    conditioner_identity = "native:dinkster.minimax_h3:" + "1" * 64
    prepared = SimpleNamespace(task=inference.MiniMaxH3Task.REF2VA)

    class Runtime:
        def run_ksampler_as_custom(self, _latent: object, **_kwargs: object) -> object:
            raise ValueError("MiniMax H3 fl2va_dit component cannot sample task REF2VA")

        sample_multistream = run_ksampler_as_custom

    handle = SimpleNamespace(
        load_device="cpu",
        runtime=SimpleNamespace(model_role="fl2va-dit"),
        recipe=SimpleNamespace(
            family_id="dinkster.minimax_h3",
            sources=(SimpleNamespace(role="diffusion"), SimpleNamespace(role="conditioner")),
        ),
        stage=lambda _role, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_native_model",
        lambda *_args: (handle, (), {}, None, None, (), None, ()),
    )
    _install_sampling_runtime(monkeypatch, Runtime())
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    conditioning = [[inference.PreparedMultiStreamConditioning(conditioner_identity, prepared), {}]]
    with pytest.raises(
        ValueError,
        match="MiniMax H3 fl2va_dit component cannot sample task REF2VA",
    ):
        native_arm.NativeKSampler.execute(
            model=object(),
            seed=0,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=conditioning,
            negative=[],
            latent_image=_latent(),
            denoise=1.0,
        )


def test_encode_decode_preserve_ordered_streams_and_audio_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = __import__("dinkster_inference")
    stage_calls: list[str] = []

    class Runtime:
        def encode_video(self, value: FakeTensor) -> FakeTensor:
            assert value.shape == (1, 3, 5, 64, 96)
            return FakeTensor((1, 24, 2, 4, 6))

        def encode_audio(self, value: object) -> FakeTensor:
            assert cast("Any", value).sample_rate == 32_000
            return FakeTensor((1, 32, 2, 2))

        def decode_video(self, value: FakeTensor) -> FakeTensor:
            assert value.shape == (1, 24, 2, 4, 6)
            return FakeTensor((1, 3, 5, 64, 96))

        def decode_audio(self, value: FakeTensor) -> object:
            assert value.shape == (1, 32, 2, 2)
            return inference.MiniMaxH3AudioContent(FakeTensor((1, 2, 1600)), 32_000)

    video_handle = SimpleNamespace(
        load_device="cpu",
        stage=lambda **_kwargs: stage_calls.append("video") or nullcontext(),
    )
    audio_handle = SimpleNamespace(
        load_device="cpu",
        stage=lambda **_kwargs: stage_calls.append("audio") or nullcontext(),
    )
    runtime = Runtime()
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_video_vae_runtime",
        lambda _value: (video_handle, runtime),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_audio_vae_runtime",
        lambda _value: (audio_handle, runtime),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    encoded = native_arm.NativeMiniMaxH3AVEncode.execute(
        video_vae=object(),
        audio_vae=object(),
        frames=FakeTensor((5, 64, 96, 3)),
        audio={"waveform": FakeTensor((1, 2, 1600)), "sample_rate": 32_000},
    )["latent"]
    streams = cast("Any", encoded)["samples"]
    assert type(streams) is MultiStreamLatent
    assert streams.roles == ("video", "audio")
    decoded = native_arm.NativeMiniMaxH3AVDecode.execute(
        video_vae=object(),
        audio_vae=object(),
        latent=encoded,
    )
    assert cast("Any", decoded["frames"]).shape == (5, 64, 96, 3)
    assert cast("Any", decoded["frames"]).contiguous_calls == 1
    assert cast("Any", decoded["audio"])["waveform"].shape == (1, 2, 1600)
    assert cast("Any", decoded["audio"])["sample_rate"] == 32_000
    assert stage_calls == ["video", "audio", "video", "audio"]


def test_codec_validation_rejects_bad_audio_and_latent_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = SimpleNamespace(
        load_device="cpu",
        stage=lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_video_vae_runtime",
        lambda _value: (handle, object()),
    )
    monkeypatch.setattr(
        native_arm,
        "_minimax_h3_audio_vae_runtime",
        lambda _value: (handle, object()),
    )
    monkeypatch.setattr(native_arm, "_torch", _fake_torch)
    with pytest.raises(ValueError, match="batch size one"):
        native_arm.NativeMiniMaxH3AVEncode.execute(
            video_vae=object(),
            audio_vae=object(),
            frames=FakeTensor((5, 64, 96, 3)),
            audio={"waveform": FakeTensor((2, 2, 1600)), "sample_rate": 32_000},
        )
    with pytest.raises(TypeError, match="ordered video/audio"):
        native_arm.NativeMiniMaxH3AVDecode.execute(
            video_vae=object(),
            audio_vae=object(),
            latent=_latent(reverse=True),
        )


def test_reference_order_limits_and_video_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    image = MiniMaxH3ImageReferenceValue(object())
    video = MiniMaxH3VideoReferenceValue(object(), None)
    audio = MiniMaxH3AudioReferenceValue(object(), 44_100)
    assert native_arm._canonical_minimax_h3_references((audio, video, image)) == (
        image,
        video,
        audio,
    )
    with pytest.raises(TypeError, match="exact MiniMax H3"):
        native_arm._canonical_minimax_h3_references((object(),))
    with pytest.raises(ValueError, match="at most 9 images"):
        native_arm._canonical_minimax_h3_references((image,) * 10)

    calls = _install_upscale(monkeypatch)
    frames, indices = native_arm._minimax_h3_video_frames(FakeTensor((50, 1080, 1920, 3)), 39)
    assert frames.shape == (39, 768, 1344, 3)
    assert indices == (0, 12, 24, 36)
    assert calls == [(1344, 768, "disabled")]
