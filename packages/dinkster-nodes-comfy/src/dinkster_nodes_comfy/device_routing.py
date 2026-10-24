"""Native device-routing nodes for resident inference values."""

from __future__ import annotations

import copy
import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_COMBO,
    CORE_INT,
    MODEL_TYPE,
    VAE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_INT = TypeExpr.concrete(CORE_INT)


def _model_management() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.model_management")


def _multigpu() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.multigpu")


def _device_input(options: tuple[str, ...]) -> InputSpec:
    return InputSpec(
        "device",
        _COMBO,
        required=False,
        default="default",
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _force_supported_compute_dtype(patcher: Any, device: object) -> None:
    management = _model_management()
    weight_dtype = patcher.model_dtype()
    cast_dtype = management.unet_manual_cast(weight_dtype, device)
    if cast_dtype is None:
        return
    logging.info(
        "Select Model Device: using %s compute dtype on %s (model weight dtype was %s).",
        cast_dtype,
        device,
        weight_dtype,
    )
    patcher.set_model_compute_dtype(cast_dtype)


def _remember_base_devices(patcher: Any) -> None:
    if not hasattr(patcher.model, "_select_base_load_device"):
        patcher.model._select_base_load_device = patcher.load_device
        patcher.model._select_base_offload_device = patcher.offload_device


def _propagate_base_devices(source: Any, target: Any) -> None:
    if hasattr(source, "_select_base_load_device") and not hasattr(
        target, "_select_base_load_device"
    ):
        target._select_base_load_device = source._select_base_load_device
        target._select_base_offload_device = source._select_base_offload_device


def _retarget_patcher(patcher: Any, load_device: object, offload_device: object) -> Any:
    if patcher.load_device == load_device:
        patcher.offload_device = offload_device
        return patcher
    source = patcher.model
    routed = patcher.deepclone_multigpu(new_load_device=load_device)
    routed.offload_device = offload_device
    _propagate_base_devices(source, routed.model)
    if hasattr(routed, "register_load_device"):
        routed.register_load_device(routed.load_device)
    return routed


def _apply_patcher_device(
    patcher: Any, resolved: object | None, base_offload: object | None = None
) -> Any:
    _remember_base_devices(patcher)
    original_load = patcher.model._select_base_load_device
    original_offload = (
        patcher.model._select_base_offload_device if base_offload is None else base_offload
    )
    if resolved is None:
        return _retarget_patcher(patcher, original_load, original_offload)
    if getattr(resolved, "type", None) == "cpu":
        if patcher.is_dynamic():
            patcher = patcher.clone(disable_dynamic=True)
        patcher.load_device = resolved
        patcher.offload_device = resolved
        return patcher
    return _retarget_patcher(patcher, resolved, original_offload)


def _prune_multigpu_collision(model: Any, primary_device: object) -> None:
    clones = model.get_additional_models_with_key("multigpu")
    filtered = [clone for clone in clones if clone.load_device != primary_device]
    if len(filtered) == len(clones):
        return
    logging.info(
        "Select Model Device: pruning MultiGPU clone on %s that collides with the primary model.",
        primary_device,
    )
    model.set_additional_models("multigpu", filtered)
    if hasattr(model, "match_multigpu_clones"):
        model.match_multigpu_clones()


class MultiGPUWorkUnits(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MultiGPU_WorkUnits",
            display_name="MultiGPU CFG Split",
            category="advanced/multigpu",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "max_gpus",
                    _INT,
                    required=False,
                    default=2,
                    widget=NumberWidget(min=1, step=1),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
            occupies=("gpu",),
        )

    @classmethod
    def execute(cls, *, model: object, max_gpus: int = 2) -> Mapping[str, object]:
        routed = _multigpu().create_multigpu_deepclones(model, max_gpus, reuse_loaded=True)
        return cls.outputs(MODEL=routed)


class SelectModelDevice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SelectModelDevice",
            display_name="Select Model Device",
            category="advanced/multigpu",
            inputs=(InputSpec("model", _MODEL), _device_input(("default", "cpu"))),
            outputs=(OutputSpec("MODEL", _MODEL),),
            occupies=("gpu",),
        )

    @classmethod
    def execute(cls, *, model: Any, device: str = "default") -> Mapping[str, object]:
        routed = model.clone()
        resolved = _model_management().resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(
                "Select Model Device: requested device %r is not available; "
                "passing through unchanged.",
                device,
            )
            return cls.outputs(MODEL=routed)
        try:
            routed = _apply_patcher_device(routed, resolved)
        except RuntimeError as error:
            logging.warning(
                "Select Model Device: cannot retarget model, passing through unchanged. (%s)", error
            )
            return cls.outputs(MODEL=routed)
        if resolved is not None:
            _force_supported_compute_dtype(routed, resolved)
            _prune_multigpu_collision(routed, routed.load_device)
        return cls.outputs(MODEL=routed)


class SelectCLIPDevice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SelectCLIPDevice",
            display_name="Select CLIP Device",
            category="advanced/multigpu",
            inputs=(InputSpec("clip", _CLIP), _device_input(("default", "cpu"))),
            outputs=(OutputSpec("CLIP", _CLIP),),
            occupies=("gpu",),
        )

    @classmethod
    def execute(cls, *, clip: Any, device: str = "default") -> Mapping[str, object]:
        routed = clip.clone()
        resolved = _model_management().resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(
                "Select CLIP Device: requested device %r is not available; "
                "passing through unchanged.",
                device,
            )
            return cls.outputs(CLIP=routed)
        try:
            routed.patcher = _apply_patcher_device(routed.patcher, resolved)
        except RuntimeError as error:
            logging.warning(
                "Select CLIP Device: cannot retarget CLIP, passing through unchanged. (%s)", error
            )
        return cls.outputs(CLIP=routed)


class SelectVAEDevice(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SelectVAEDevice",
            display_name="Select VAE Device",
            category="advanced/multigpu",
            inputs=(InputSpec("vae", _VAE), _device_input(("default",))),
            outputs=(OutputSpec("VAE", _VAE),),
            occupies=("gpu",),
        )

    @classmethod
    def execute(cls, *, vae: Any, device: str = "default") -> Mapping[str, object]:
        routed = copy.copy(vae)
        routed.patcher = routed.patcher.clone()
        management = _model_management()
        resolved = management.resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(
                "Select VAE Device: requested device %r is not available; "
                "passing through unchanged.",
                device,
            )
            return cls.outputs(VAE=routed)
        if resolved is not None and getattr(resolved, "type", None) == "cpu":
            logging.info(
                "Select VAE Device: CPU is not a supported choice, passing through unchanged."
            )
            return cls.outputs(VAE=routed)
        if not hasattr(routed, "_select_base_device"):
            routed._select_base_device = routed.device
        try:
            routed.patcher = _apply_patcher_device(
                routed.patcher,
                resolved,
                base_offload=management.vae_offload_device(),
            )
        except RuntimeError as error:
            logging.warning(
                "Select VAE Device: cannot retarget VAE, passing through unchanged. (%s)", error
            )
            return cls.outputs(VAE=routed)
        routed.first_stage_model = routed.patcher.model
        routed.device = routed._select_base_device if resolved is None else resolved
        return cls.outputs(VAE=routed)


DEVICE_ROUTING_NODES: tuple[type[Node], ...] = (
    MultiGPUWorkUnits,
    SelectModelDevice,
    SelectCLIPDevice,
    SelectVAEDevice,
)


__all__ = ["DEVICE_ROUTING_NODES"]
