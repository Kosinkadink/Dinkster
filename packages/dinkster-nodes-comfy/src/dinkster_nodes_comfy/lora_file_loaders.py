"""LoRA file loaders backed by governed assets."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any, Protocol, cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    MODEL_TYPE,
    AssetRef,
    AssetWidget,
    InputFamilySpec,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    obtain_resident,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)

_MODEL_TOOLTIP = "The diffusion model the LoRA will be applied to."
_CLIP_TOOLTIP = "The CLIP model the LoRA will be applied to."
_LORA_TOOLTIP = "The name of the LoRA."
_STRENGTH_TOOLTIP = "How strongly to modify the {}. This value can be negative."
_STACK_STRENGTH_TOOLTIP = "How strongly to apply this LoRA. 0 = off, negative inverts the effect."


class _Storage(Protocol):
    def data_ptr(self) -> int: ...

    def nbytes(self) -> int: ...


class _Device(Protocol):
    type: str


class _Tensor(Protocol):
    device: _Device

    def untyped_storage(self) -> _Storage: ...


def _strength(name: str, *, doc: str = "") -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=1.0,
        widget=NumberWidget(min=-100.0, max=100.0, step=0.01),
        doc=doc,
    )


def _asset_input(*, doc: str = _LORA_TOOLTIP) -> InputSpec:
    return InputSpec("lora_name", _ASSET, widget=AssetWidget(kind="model/lora"), doc=doc)


def _ram_bytes(value: Any) -> int:
    candidate: object = value
    if isinstance(value, tuple):
        candidate = cast("object", value[0])
    if not isinstance(candidate, Mapping):
        return 0
    weights = cast("Mapping[str, _Tensor]", candidate)
    storages: dict[int, int] = {}
    for tensor in weights.values():
        if tensor.device.type == "cpu":
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def _load_weights(lora_name: object, *, bypass: bool) -> Any:
    if not isinstance(lora_name, (AssetRef, LocalAssetRef)):
        raise TypeError("lora_name must be an AssetRef or LocalAssetRef")
    # Asset authority must be checked even when the resident entry already exists.
    path = str(lora_name.local_path())
    initialize_inference_process()
    identity = (
        ("verified", lora_name.digest)
        if isinstance(lora_name, AssetRef)
        else ("local", lora_name.binding)
    )
    key = json.dumps(
        {"purpose": "bypass-lora-weights" if bypass else "lora-weights", "asset": identity},
        separators=(",", ":"),
    )
    utils = importlib.import_module("dinkster_inference.utils")
    return obtain_resident(
        key,
        lambda: utils.load_torch_file(path, safe_load=True, return_metadata=not bypass),
        ram_bytes=_ram_bytes,
    )


def _apply_lora(
    model: Any,
    clip: Any,
    lora_name: object,
    strength_model: float,
    strength_clip: float,
    *,
    bypass: bool,
) -> tuple[Any, Any]:
    if strength_model == 0 and strength_clip == 0:
        return model, clip
    loaded = _load_weights(lora_name, bypass=bypass)
    sd = importlib.import_module("dinkster_inference.sd")
    if bypass:
        return sd.load_bypass_lora_for_models(model, clip, loaded, strength_model, strength_clip)
    weights, metadata = loaded
    return sd.load_lora_for_models(
        model, clip, weights, strength_model, strength_clip, lora_metadata=metadata
    )


def _loader_schema(node_type: str, *, bypass: bool, model_only: bool) -> NodeSchema:
    if model_only:
        return NodeSchema(
            node_type=node_type,
            display_name=(
                "Load LoRA (Bypass, Model Only) (for debugging)" if bypass else "Load LoRA"
            ),
            category="model/loaders",
            description=(
                "Apply LoRA in bypass mode. Unlike regular LoRA, this doesn't modify "
                "model weights - instead it injects the LoRA computation during forward "
                "pass. Useful for training scenarios."
                if bypass
                else "This LoRAs loader is used to modify the diffusion model, altering the way in "
                "which latents are denoised such as applying styles. Multiple LoRA nodes can be "
                "linked together."
            ),
            search_terms=()
            if bypass
            else ("lora", "load lora", "apply lora", "lora loader", "lora model"),
            inputs=(InputSpec("model", _MODEL), _asset_input(doc=""), _strength("strength_model")),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
    return NodeSchema(
        node_type=node_type,
        display_name="Load LoRA (Bypass) (For debugging)"
        if bypass
        else "Load LoRA (Model and CLIP)",
        category="model/loaders",
        description=(
            "Apply LoRA in bypass mode. Unlike regular LoRA, this doesn't modify model "
            "weights - instead it injects the LoRA computation during forward pass. Useful "
            "for training scenarios."
            if bypass
            else "This LoRA loader is used to modify both diffusion and CLIP models, "
            "altering the way in which latents are denoised such as applying styles. "
            "Multiple LoRA nodes can be linked together."
        ),
        search_terms=()
        if bypass
        else ("lora", "load lora", "apply lora", "lora loader", "lora model"),
        inputs=(
            InputSpec("model", _MODEL, doc=_MODEL_TOOLTIP),
            InputSpec("clip", _CLIP, doc=_CLIP_TOOLTIP),
            _asset_input(),
            _strength("strength_model", doc=_STRENGTH_TOOLTIP.format("diffusion model")),
            _strength("strength_clip", doc=_STRENGTH_TOOLTIP.format("CLIP model")),
        ),
        outputs=(
            OutputSpec("MODEL", _MODEL, doc="The modified diffusion model."),
            OutputSpec("CLIP", _CLIP, doc="The modified CLIP model."),
        ),
    )


class LoraLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _loader_schema("LoraLoader", bypass=False, model_only=False)

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        clip: Any,
        lora_name: object,
        strength_model: float,
        strength_clip: float,
    ) -> Mapping[str, object]:
        model, clip = _apply_lora(
            model, clip, lora_name, strength_model, strength_clip, bypass=False
        )
        return cls.outputs(MODEL=model, CLIP=clip)


class LoraLoaderModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _loader_schema("LoraLoaderModelOnly", bypass=False, model_only=True)

    @classmethod
    def execute(
        cls, *, model: Any, lora_name: object, strength_model: float
    ) -> Mapping[str, object]:
        model, _ = _apply_lora(model, None, lora_name, strength_model, 0.0, bypass=False)
        return cls.outputs(MODEL=model)


class LoraLoaderBypass(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _loader_schema("LoraLoaderBypass", bypass=True, model_only=False)

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        clip: Any,
        lora_name: object,
        strength_model: float,
        strength_clip: float,
    ) -> Mapping[str, object]:
        model, clip = _apply_lora(
            model, clip, lora_name, strength_model, strength_clip, bypass=True
        )
        return cls.outputs(MODEL=model, CLIP=clip)


class LoraLoaderBypassModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _loader_schema("LoraLoaderBypassModelOnly", bypass=True, model_only=True)

    @classmethod
    def execute(
        cls, *, model: Any, lora_name: object, strength_model: float
    ) -> Mapping[str, object]:
        model, _ = _apply_lora(model, None, lora_name, strength_model, 0.0, bypass=True)
        return cls.outputs(MODEL=model)


def _stack_family(doc: str) -> InputFamilySpec:
    return InputFamilySpec(
        "loras",
        (
            InputSpec(
                "lora_name",
                _ASSET,
                widget=AssetWidget(kind="model/lora"),
                doc="The name of the LoRA file to apply.",
            ),
            _strength("strength", doc=_STACK_STRENGTH_TOOLTIP),
            InputSpec(
                "enabled",
                TypeExpr.concrete(CORE_BOOLEAN),
                default=True,
                doc="Turn off to skip this LoRA without changing its file or strength.",
            ),
        ),
        min_members=1,
        max_members=20,
        doc=doc,
        display_name="LoRA",
    )


def _rows(loras: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows: dict[str, dict[str, object]] = {}
    for input_id, value in loras.items():
        row, separator, field = input_id.partition(".")
        if not separator:
            continue
        rows.setdefault(row, {})[field] = value
    return tuple(rows.values())


def _row_strength(row: Mapping[str, object]) -> float:
    strength = row.get("strength", 1.0)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("LoRA strength must be a number")
    return float(strength)


class LoadLoraModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadLoraModel",
            display_name="Load LoRA (Model)",
            category="model/loaders",
            description=(
                "Apply a stack of LoRAs to a diffusion model. Add one row per LoRA; each row picks "
                "a LoRA file and its strength."
            ),
            search_terms=("lora", "load lora", "apply lora", "lora model", "lora stack"),
            inputs=(
                InputSpec("model", _MODEL, doc="The diffusion model the LoRAs will be applied to."),
            ),
            input_families=(_stack_family("Each row applies one LoRA to the model."),),
            outputs=(OutputSpec("MODEL", _MODEL, doc="The modified diffusion model."),),
        )

    @classmethod
    def execute(cls, *, model: Any, loras: Mapping[str, object]) -> Mapping[str, object]:
        for row in _rows(loras):
            lora_name = row.get("lora_name")
            strength = _row_strength(row)
            if not lora_name or not row.get("enabled", True) or strength == 0:
                continue
            model, _ = _apply_lora(model, None, lora_name, strength, 0.0, bypass=False)
        return cls.outputs(MODEL=model)


class LoadLoraTextEncoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadLoraTextEncoder",
            display_name="Load LoRA (Text Encoder)",
            category="model/loaders",
            description=(
                "Apply a stack of LoRAs to a CLIP text encoder. Add one row per LoRA; "
                "each row picks "
                "a LoRA file and its strength."
            ),
            search_terms=("lora", "load lora", "apply lora", "clip lora", "lora stack"),
            inputs=(
                InputSpec("clip", _CLIP, doc="The CLIP text encoder the LoRAs will be applied to."),
            ),
            input_families=(_stack_family("Each row applies one LoRA to the text encoder."),),
            outputs=(OutputSpec("CLIP", _CLIP, doc="The modified CLIP text encoder."),),
        )

    @classmethod
    def execute(cls, *, clip: Any, loras: Mapping[str, object]) -> Mapping[str, object]:
        for row in _rows(loras):
            lora_name = row.get("lora_name")
            strength = _row_strength(row)
            if not lora_name or not row.get("enabled", True) or strength == 0:
                continue
            _, clip = _apply_lora(None, clip, lora_name, 0.0, strength, bypass=False)
        return cls.outputs(CLIP=clip)


LORA_FILE_LOADER_NODES = (
    LoraLoader,
    LoraLoaderModelOnly,
    LoadLoraModel,
    LoadLoraTextEncoder,
    LoraLoaderBypass,
    LoraLoaderBypassModelOnly,
)
