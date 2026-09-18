"""Generate the image pack's maintained ComfyUI alias registry.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      PYTHONPATH=/path/to/comfy-dependencies:<all local package src paths> \
      /path/to/python tools/gen_image_comfy_aliases.py

The ComfyUI checkout must be clean and pinned to the commit below. Ecosystem
schemas are transcribed as inert interface data from their pinned sources;
the generator never imports ecosystem packages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from dinkster_compat_comfy import CompatTranslation, translate_node, translate_v3_schema
from dinkster_nodes_image import IMAGE_NODES
from dinkster_schema import (
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilyMapping,
    InputFamilyMember,
    InputFamilySpec,
    InputSpec,
    MappingSource,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    TypeExpr,
    ValueTransform,
    schema_from_wire,
)
from dinkster_schema.model import DynamicEntry, NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire
from layer_comfy_aliases import layer_alias_data

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
MTB_BASELINE = "b35b5d8a17c0d59e80a8b3627b679c2c1003d04f"
KJ_BASELINE = "827fe6ee0ed7348d8daa988ed852bedf1272380c"
ESSENTIALS_BASELINE = "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9"
CONTROLNET_AUX_BASELINE = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-nodes-image" / "comfy-aliases.json"
DEPTH_ANYTHING_V2_OUT = (
    REPO / "packages" / "dinkster-vision-depth-anything-v2" / "comfy-aliases.inactive.json"
)
DEPTH_ANYTHING_V2_SOURCE_TYPES = frozenset(
    {
        "comfy.comfyui_controlnet_aux.AIO_Preprocessor",
        "comfy.comfyui_controlnet_aux.DepthAnythingV2Preprocessor",
    }
)
CONTROLNET_AUX_AIO_OPTIONS = (
    "none",
    "TilePreprocessor",
    "TTPlanet_TileGF_Preprocessor",
    "TTPlanet_TileSimple_Preprocessor",
    "ImageLuminanceDetector",
    "ImageIntensityDetector",
    "DepthAnythingPreprocessor",
    "Zoe_DepthAnythingPreprocessor",
    "HEDPreprocessor",
    "FakeScribblePreprocessor",
    "OneFormer-COCO-SemSegPreprocessor",
    "OneFormer-ADE20K-SemSegPreprocessor",
    "PyraCannyPreprocessor",
    "LineartStandardPreprocessor",
    "M-LSDPreprocessor",
    "MediaPipe-FaceMeshPreprocessor",
    "SAMPreprocessor",
    "DensePosePreprocessor",
    "UniFormer-SemSegPreprocessor",
    "SemSegPreprocessor",
    "BAE-NormalMapPreprocessor",
    "MeshGraphormer-DepthMapPreprocessor",
    "PiDiNetPreprocessor",
    "ShufflePreprocessor",
    "Zoe-DepthMapPreprocessor",
    "DSINE-NormalMapPreprocessor",
    "AnyLineArtPreprocessor_aux",
    "AnimeFace_SemSegPreprocessor",
    "LineArtPreprocessor",
    "ColorPreprocessor",
    "OpenposePreprocessor",
    "LeReS-DepthMapPreprocessor",
    "BinaryPreprocessor",
    "TEEDPreprocessor",
    "DepthAnythingV2Preprocessor",
    "MiDaS-NormalMapPreprocessor",
    "MiDaS-DepthMapPreprocessor",
    "CannyEdgePreprocessor",
    "ScribblePreprocessor",
    "Scribble_XDoG_Preprocessor",
    "Scribble_PiDiNet_Preprocessor",
    "DWPreprocessor",
    "AnimalPosePreprocessor",
    "AnimeLineArtPreprocessor",
    "Metric3D-DepthMapPreprocessor",
    "Metric3D-NormalMapPreprocessor",
    "Manga2Anime_LineArt_Preprocessor",
)


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _MTBBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "x": ("INT", {"default": 0, "max": 10_000_000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10_000_000, "min": 0, "step": 1}),
                "width": (
                    "INT",
                    {"default": 256, "max": 10_000_000, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 10_000_000, "min": 0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "do_crop"
    CATEGORY = "mtb/crop"

    def do_crop(self, x: int, y: int, width: int, height: int) -> tuple[tuple[int, int, int, int]]:
        return ((x, y, width, height),)


class _MTBUncrop:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "border_blending": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_uncrop"
    CATEGORY = "mtb/crop"

    def do_uncrop(
        self,
        image: object,
        crop_image: object,
        bbox: object,
        border_blending: float = 0.25,
    ) -> tuple[object]:
        del crop_image, bbox, border_blending
        return (image,)


class _KJCreateShapeMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "shape": (["circle", "square", "triangle"], {"default": "circle"}),
                "frames": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "location_x": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "location_y": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512}),
                "frame_width": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "frame_height": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "shape_width": ("INT", {"default": 128, "min": 8, "max": 4096}),
                "shape_height": ("INT", {"default": 128, "min": 8, "max": 4096}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"

    def createshapemask(self) -> None:
        raise RuntimeError("static schema shim")


class _KJCreateTextMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "invert": ("BOOLEAN", {"default": False}),
                "frames": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "text_x": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "text_y": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "font_size": ("INT", {"default": 32, "min": 8, "max": 4096}),
                "font_color": ("STRING", {"default": "white"}),
                "text": ("STRING", {"default": "HELLO!", "multiline": True}),
                "font": (("Arial.ttf",),),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "start_rotation": ("INT", {"default": 0, "min": 0, "max": 359}),
                "end_rotation": ("INT", {"default": 0, "min": -359, "max": 359}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes/text"

    def createtextmask(self) -> None:
        raise RuntimeError("static schema shim")


class _KJGrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "incremental_expandrate": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "lerp_alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "decay_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {"fill_holes": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "expand_mask"
    CATEGORY = "KJNodes/masking"

    def expand_mask(self) -> None:
        raise RuntimeError("static schema shim")


class _KJColorToMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": False}),
                "red": ("INT", {"default": 0, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
                "threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "clip"
    CATEGORY = "KJNodes/masking"

    def clip(self) -> None:
        raise RuntimeError("static schema shim")


class _KJGetMaskSizeAndCount:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "width", "height", "count")
    FUNCTION = "getsize"
    CATEGORY = "KJNodes/masking"

    def getsize(self) -> None:
        raise RuntimeError("static schema shim")


class _KJDrawMaskOnImage:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "0, 0, 0"}),
            },
            "optional": {"device": (["cpu", "gpu"], {"default": "cpu"})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "KJNodes/masking"

    def apply(self) -> None:
        raise RuntimeError("static schema shim")


class _KJBboxVisualize:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxes": ("BBOX,BOUNDING_BOX",),
                "line_width": ("INT", {"default": 1, "min": 1, "max": 10}),
                "bbox_format": (["xywh", "xyxy"], {"default": "xywh"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "visualizebbox"
    CATEGORY = "KJNodes/masking"

    def visualizebbox(self) -> None:
        raise RuntimeError("static schema shim")


class _EssentialsMaskFromColor:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {"default": 255, "min": 0, "max": 255}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255}),
                "threshold": ("INT", {"default": 0, "min": 0, "max": 127}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self) -> None:
        raise RuntimeError("static schema shim")


class _EssentialsTransitionMask:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "frames": ("INT", {"default": 16, "min": 1, "max": 9999}),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "end_frame": ("INT", {"default": 9999, "min": 0}),
                "transition_type": (
                    [
                        "horizontal slide",
                        "vertical slide",
                        "horizontal bar",
                        "vertical bar",
                        "center box",
                        "horizontal door",
                        "vertical door",
                        "circle",
                        "fade",
                    ],
                ),
                "timing_function": (["linear", "in", "out", "in-out"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self) -> None:
        raise RuntimeError("static schema shim")


class _EssentialsDrawText:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "text": ("STRING", {"default": "Hello, World!", "multiline": True}),
                "font": (("Arial.ttf",),),
                "size": ("INT", {"default": 56, "min": 1, "max": 9999}),
                "color": ("STRING", {"default": "#FFFFFF"}),
                "background_color": ("STRING", {"default": "#00000000"}),
                "shadow_distance": ("INT", {"default": 0, "min": 0, "max": 100}),
                "shadow_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "horizontal_align": (["left", "center", "right"],),
                "vertical_align": (["top", "center", "bottom"],),
                "offset_x": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "offset_y": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "direction": (["ltr", "rtl"],),
            },
            "optional": {"img_composite": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "essentials/text"

    def execute(self) -> None:
        raise RuntimeError("static schema shim")


def _static_schema(
    node_class: str,
    *,
    namespace: str,
    category: str | None = None,
    required: dict[str, object],
    optional: dict[str, object] | None = None,
    returns: tuple[str, ...] = ("IMAGE",),
    return_names: tuple[str, ...] | None = None,
    input_is_list: bool = False,
    output_is_list: tuple[bool, ...] | None = None,
) -> NodeSchema:
    def input_types(_cls: type[Any]) -> dict[str, object]:
        value: dict[str, object] = {"required": required}
        if optional:
            value["optional"] = optional
        return value

    def execute(_self: object) -> None:
        raise RuntimeError("static schema shim")

    attributes: dict[str, object] = {
        "INPUT_TYPES": classmethod(input_types),
        "RETURN_TYPES": returns,
        "FUNCTION": "execute",
        "CATEGORY": category or namespace,
        "execute": execute,
    }
    if return_names is not None:
        attributes["RETURN_NAMES"] = return_names
    if input_is_list:
        attributes["INPUT_IS_LIST"] = True
    if output_is_list is not None:
        attributes["OUTPUT_IS_LIST"] = output_is_list
    shim = type(f"_{node_class.replace('+', 'Plus')}", (), attributes)
    return translate_node(
        node_class,
        shim,
        CompatTranslation(),
        namespace=namespace,
    ).schema()


def _dynamic_image_schema(schema: NodeSchema) -> NodeSchema:
    return dataclasses.replace(
        schema,
        input_families=(
            InputFamilySpec(
                "images",
                TypeExpr.concrete("comfy.IMAGE"),
                min_members=1,
                max_members=1000,
                member_prefix="image_",
            ),
        ),
    )


def _family_members(*source_inputs: str) -> InputFamilyMapping:
    return InputFamilyMapping.from_members(
        *(
            InputFamilyMember.build(
                str(index),
                inputs={"value": MappingSource.copy(source_input)},
            )
            for index, source_input in enumerate(source_inputs, start=1)
        )
    )


def _with_predicate(
    existing: ReplacementPredicate | None,
    predicate: ReplacementPredicate,
) -> ReplacementPredicate:
    if existing is None:
        return predicate
    return ReplacementPredicate.all_of(existing, predicate)


def _fanout_target_choice(
    case: ReplacementCase,
    *,
    target_input: str,
    source_input: str,
    choices: Mapping[str, str],
    fallback: str,
) -> tuple[ReplacementCase, ...]:
    """Resolve a source-dependent target choice into guarded literal cases."""
    if fallback not in choices:
        raise ValueError(f"fallback {fallback!r} is not a declared source choice")
    inputs = dict(case.inputs)
    selector = inputs.pop(target_input, None)
    if selector is None or selector.input != source_input:
        raise ValueError(f"target choice {target_input!r} must read {source_input!r}")
    ordered = [item for item in choices.items() if item[0] != fallback]
    ordered.append((fallback, choices[fallback]))
    return tuple(
        dataclasses.replace(
            case,
            when=(
                case.when
                if source == fallback
                else _with_predicate(
                    case.when,
                    ReplacementPredicate.value_equals(source_input, source),
                )
            ),
            inputs=tuple(
                {
                    target_input: MappingSource.constant(target),
                    **inputs,
                }.items()
            ),
        )
        for source, target in ordered
    )


def _dynamic_input_ids(entries: Sequence[DynamicEntry]) -> frozenset[str]:
    found: set[str] = set()
    for entry in entries:
        if isinstance(entry, InputSpec):
            found.add(entry.id)
        elif isinstance(entry, InputFamilySpec):
            found.update(_dynamic_input_ids(entry.template))
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                found.update(_dynamic_input_ids(option.inputs))
        elif isinstance(entry, DynamicSlotSpec):
            found.add(entry.id)
            found.update(_dynamic_input_ids(entry.inputs))
            for variant in entry.variants or ():
                found.update(_dynamic_input_ids(variant.inputs))
    return frozenset(found)


def _project_dynamic_owner(
    schema: NodeSchema,
    *,
    mapped_inputs: Mapping[str, MappingSource],
    values: Mapping[str, object],
    linked_inputs: Sequence[str],
    existing_choices: Mapping[str, str],
) -> tuple[dict[str, str], frozenset[str], dict[str, str]]:
    """Project logical native input names onto one selected dynamic interface."""
    logical_inputs = frozenset((*mapped_inputs, *values, *linked_inputs))
    selector_ids: set[str] = set()
    paths: dict[str, str] = {input_spec.id: input_spec.id for input_spec in schema.inputs}
    choices = dict(existing_choices)

    def add_path(input_id: str, path: str) -> None:
        previous = paths.get(input_id)
        if previous is not None and previous != path:
            raise ValueError(
                f"{schema.node_type}: selected dynamic inputs reuse logical id {input_id!r}"
            )
        paths[input_id] = path

    def walk(entries: Sequence[DynamicEntry], parent: str = "") -> None:
        for entry in entries:
            path = f"{parent}.{entry.id}" if parent else entry.id
            if isinstance(entry, InputSpec):
                add_path(entry.id, path)
                continue
            if isinstance(entry, InputFamilySpec):
                continue
            if isinstance(entry, DynamicComboSpec):
                selector_ids.add(entry.id)
                selected = choices.get(path)
                mapped = mapped_inputs.get(entry.id)
                stored = values.get(entry.id)
                if selected is None and mapped is not None:
                    if mapped.kind != "constant" or not isinstance(mapped.value, str):
                        raise ValueError(
                            f"{schema.node_type}: dynamic choice {path!r} must be literal"
                        )
                    selected = mapped.value
                if selected is None and stored is not None:
                    if not isinstance(stored, str):
                        raise ValueError(
                            f"{schema.node_type}: dynamic choice {path!r} must be a string"
                        )
                    selected = stored
                if selected is None:
                    selected = entry.default
                if selected is None:
                    raise ValueError(f"{schema.node_type}: dynamic choice {path!r} is required")
                option = entry.option(selected)
                if option is None:
                    raise ValueError(
                        f"{schema.node_type}: dynamic choice {path!r} has unknown option "
                        f"{selected!r}"
                    )
                choices[path] = selected
                walk(option.inputs, path)
                continue
            if not isinstance(entry, DynamicSlotSpec):
                raise TypeError(f"unsupported dynamic entry: {entry!r}")
            selected = choices.get(path)
            if selected is None and entry.id in logical_inputs:
                variants = entry.variants or ()
                if len(variants) != 1:
                    raise ValueError(
                        f"{schema.node_type}: dynamic slot {path!r} needs an explicit variant"
                    )
                selected = variants[0].key
            if selected is None:
                if entry.required:
                    raise ValueError(f"{schema.node_type}: dynamic slot {path!r} is required")
                continue
            variant = entry.variant(selected)
            if variant is None:
                raise ValueError(
                    f"{schema.node_type}: dynamic slot {path!r} has unknown variant {selected!r}"
                )
            choices[path] = selected
            add_path(entry.id, path)
            walk((*entry.inputs, *variant.inputs), path)

    walk((*schema.combos, *schema.slots))
    dynamic_ids = _dynamic_input_ids((*schema.combos, *schema.slots))
    inactive = sorted((logical_inputs & dynamic_ids) - paths.keys() - selector_ids)
    if inactive:
        raise ValueError(
            f"{schema.node_type}: inputs are inactive for selected dynamic choices: "
            + ", ".join(inactive)
        )
    return paths, frozenset(selector_ids), choices


_DYNAMIC_TARGET_SCHEMAS = {
    schema.node_type: schema
    for node in IMAGE_NODES
    if (schema := node.schema()).combos or schema.slots
}


def _materialize_dynamic_case(case: ReplacementCase) -> ReplacementCase:
    helper_types = {local_id: node.type for local_id, node in case.nodes or ()}

    def split(address: str) -> tuple[str, str]:
        local_id, separator, ref = address.partition(":")
        if separator and local_id in helper_types:
            return local_id, ref
        return "", address

    inputs_by_owner: dict[str, dict[str, MappingSource]] = {}
    for address, source in case.inputs:
        local_id, ref = split(address)
        inputs_by_owner.setdefault(local_id, {})[ref] = source
    links_by_owner: dict[str, list[str]] = {}
    for link in case.links:
        local_id, ref = split(link.to)
        links_by_owner.setdefault(local_id, []).append(ref)
    values_by_owner = {local_id: dict(node.values) for local_id, node in case.nodes or ()}
    choices_by_owner: dict[str, dict[str, str]] = {}
    for address, choice in case.slot_variants:
        local_id, path = split(address)
        choices_by_owner.setdefault(local_id, {})[path] = choice

    projections: dict[str, tuple[dict[str, str], frozenset[str], dict[str, str]]] = {}
    owner_types = {"": case.to, **helper_types}
    for local_id, node_type in owner_types.items():
        schema = _DYNAMIC_TARGET_SCHEMAS.get(node_type)
        if schema is None:
            continue
        projections[local_id] = _project_dynamic_owner(
            schema,
            mapped_inputs=inputs_by_owner.get(local_id, {}),
            values=values_by_owner.get(local_id, {}),
            linked_inputs=links_by_owner.get(local_id, ()),
            existing_choices=choices_by_owner.get(local_id, {}),
        )

    if not projections:
        return case

    def project(address: str, *, allow_selector: bool = False) -> str | None:
        local_id, ref = split(address)
        projection = projections.get(local_id)
        if projection is None:
            return address
        paths, selectors, _choices = projection
        if ref in selectors:
            if allow_selector:
                return None
            raise ValueError(
                f"{owner_types[local_id]}: dynamic selector {ref!r} cannot receive a link"
            )
        projected = paths.get(ref, ref)
        return f"{local_id}:{projected}" if local_id else projected

    projected_inputs = tuple(
        (projected, source)
        for address, source in case.inputs
        if (projected := project(address, allow_selector=True)) is not None
    )

    def project_helper_values(
        local_id: str,
        values: Sequence[tuple[str, object]],
    ) -> tuple[tuple[str, object], ...]:
        projected_values: list[tuple[str, object]] = []
        for ref, value in values:
            projected = project(f"{local_id}:{ref}", allow_selector=True)
            if projected is not None:
                projected_values.append((projected.split(":", 1)[-1], value))
        return tuple(projected_values)

    projected_nodes = None
    if case.nodes is not None:
        projected_nodes = tuple(
            (
                local_id,
                dataclasses.replace(
                    node,
                    values=project_helper_values(local_id, node.values),
                ),
            )
            for local_id, node in case.nodes
        )
    slot_variants = dict(case.slot_variants)
    for local_id, (_paths, _selectors, choices) in projections.items():
        prefix = f"{local_id}:" if local_id else ""
        slot_variants.update((prefix + path, choice) for path, choice in choices.items())
    return dataclasses.replace(
        case,
        nodes=projected_nodes,
        inputs=projected_inputs,
        links=tuple(
            dataclasses.replace(link, to=projected)
            for link in case.links
            if (projected := project(link.to)) is not None
        ),
        slot_variants=tuple(slot_variants.items()),
    )


def _materialize_dynamic_rule(rule: ReplacementRule) -> ReplacementRule:
    return dataclasses.replace(
        rule,
        cases=tuple(_materialize_dynamic_case(case) for case in rule.cases),
    )


def _record(
    *,
    source_pack: str,
    node_class: str,
    revision: str,
    carrier: str,
    rule: ReplacementRule,
    tier: str,
    evidence: list[str],
    tolerances: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    confidence: dict[str, object] = {"tier": tier, "evidence": evidence}
    if tolerances is not None:
        confidence["tolerances"] = tolerances
    return {
        "id": f"comfy_alias:{source_pack}/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": source_pack,
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": revision,
        },
        "replacement": rule_to_wire(_materialize_dynamic_rule(rule)),
        "confidence": confidence,
    }


def _core_v3_schema(node_class: type[Any]) -> NodeSchema:
    return translate_v3_schema(node_class.GET_SCHEMA(), CompatTranslation())


def _preprocessor_alias_data() -> tuple[list[NodeSchema], list[dict[str, object]]]:
    namespace = "comfyui_controlnet_aux"
    line_category = "ControlNet Preprocessors/Line Extractors"
    adapter_category = "ControlNet Preprocessors/T2IAdapter-only"
    recolor_category = "ControlNet Preprocessors/Recolor"
    tile_category = "ControlNet Preprocessors/tile"
    resolution = ("INT", {"default": 512, "min": 64, "max": 16384, "step": 64})
    image: dict[str, object] = {"image": ("IMAGE",)}
    source_schemas = [
        _static_schema(
            "CannyEdgePreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": (
                    "INT",
                    {"default": 200, "min": 0, "max": 255, "step": 1},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "PyraCannyPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "low_threshold": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1}),
                "high_threshold": (
                    "INT",
                    {"default": 128, "min": 0, "max": 255, "step": 1},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "LineartStandardPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "guassian_sigma": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "intensity_threshold": (
                    "INT",
                    {"default": 8, "min": 0, "max": 16, "step": 1},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "ScribblePreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={"resolution": resolution},
        ),
        _static_schema(
            "Scribble_XDoG_Preprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "threshold": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1}),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "LineArtPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "coarse": (["disable", "enable"], {"default": "disable"}),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "AnimeLineArtPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={"resolution": resolution},
        ),
        _static_schema(
            "Manga2Anime_LineArt_Preprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={"resolution": resolution},
        ),
        _static_schema(
            "AnyLineArtPreprocessor_aux",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "merge_with_lineart": (
                    [
                        "lineart_standard",
                        "lineart_realisitic",
                        "lineart_anime",
                        "manga_line",
                    ],
                    {"default": "lineart_standard"},
                ),
                "resolution": (
                    "INT",
                    {"default": 1280, "min": 64, "max": 16384, "step": 8},
                ),
                "lineart_lower_bound": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "lineart_upper_bound": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "object_min_size": (
                    "INT",
                    {"default": 36, "min": 1, "max": 16384, "step": 1},
                ),
                "object_connectivity": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16384, "step": 1},
                ),
            },
        ),
        *(
            _static_schema(
                node_class,
                namespace=namespace,
                category=line_category,
                required=image,
                optional={
                    "safe": (["enable", "disable"], {"default": "enable"}),
                    "resolution": resolution,
                },
            )
            for node_class in (
                "HEDPreprocessor",
                "FakeScribblePreprocessor",
                "PiDiNetPreprocessor",
                "Scribble_PiDiNet_Preprocessor",
            )
        ),
        _static_schema(
            "TEEDPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "safe_steps": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "M-LSDPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "score_threshold": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01},
                ),
                "dist_threshold": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 20.0, "step": 0.01},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "DiffusionEdge_Preprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "environment": (
                    ["indoor", "urban", "natrual"],
                    {"default": "indoor"},
                ),
                "patch_batch_size": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16, "step": 1},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "BinaryPreprocessor",
            namespace=namespace,
            category=line_category,
            required=image,
            optional={
                "bin_threshold": (
                    "INT",
                    {"default": 100, "min": 0, "max": 255, "step": 1},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "ColorPreprocessor",
            namespace=namespace,
            category=adapter_category,
            required=image,
            optional={"resolution": resolution},
        ),
        _static_schema(
            "ImageLuminanceDetector",
            namespace=namespace,
            category=recolor_category,
            required=image,
            optional={
                "gamma_correction": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "ImageIntensityDetector",
            namespace=namespace,
            category=recolor_category,
            required=image,
            optional={
                "gamma_correction": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "ShufflePreprocessor",
            namespace=namespace,
            category=adapter_category,
            required=image,
            optional={
                "resolution": resolution,
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1},
                ),
            },
        ),
        _static_schema(
            "TilePreprocessor",
            namespace=namespace,
            category=tile_category,
            required=image,
            optional={
                "pyrUp_iters": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "TTPlanet_TileGF_Preprocessor",
            namespace=namespace,
            category=tile_category,
            required=image,
            optional={
                "scale_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.01},
                ),
                "blur_strength": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.01},
                ),
                "radius": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "eps": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "TTPlanet_TileSimple_Preprocessor",
            namespace=namespace,
            category=tile_category,
            required=image,
            optional={
                "scale_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.01},
                ),
                "blur_strength": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.01},
                ),
            },
        ),
        _static_schema(
            "InpaintPreprocessor",
            namespace=namespace,
            category="ControlNet Preprocessors/others",
            required={"image": ("IMAGE",), "mask": ("MASK",)},
            optional={"black_pixel_for_xinsir_cn": ("BOOLEAN", {"default": False})},
        ),
        _static_schema(
            "HintImageEnchance",
            namespace=namespace,
            category="ControlNet Preprocessors",
            required={
                "hint_image": ("IMAGE",),
                "image_gen_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "image_gen_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "resize_mode": (
                    ["Just Resize", "Crop and Resize", "Resize and Fill"],
                    {"default": "Just Resize"},
                ),
            },
        ),
        _static_schema(
            "PixelPerfectResolution",
            namespace=namespace,
            category="ControlNet Preprocessors",
            required={
                "original_image": ("IMAGE",),
                "image_gen_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "image_gen_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "resize_mode": (
                    ["Just Resize", "Crop and Resize", "Resize and Fill"],
                    {"default": "Just Resize"},
                ),
            },
            returns=("INT",),
            return_names=("RESOLUTION (INT)",),
        ),
        _static_schema(
            "DepthAnythingV2Preprocessor",
            namespace=namespace,
            category="ControlNet Preprocessors/Normal and Depth Estimators",
            required=image,
            optional={
                "ckpt_name": (
                    [
                        "depth_anything_v2_vitg.pth",
                        "depth_anything_v2_vitl.pth",
                        "depth_anything_v2_vitb.pth",
                        "depth_anything_v2_vits.pth",
                    ],
                    {"default": "depth_anything_v2_vitl.pth"},
                ),
                "resolution": resolution,
            },
        ),
        _static_schema(
            "AIO_Preprocessor",
            namespace=namespace,
            category="ControlNet Preprocessors",
            required=image,
            optional={
                "preprocessor": (
                    CONTROLNET_AUX_AIO_OPTIONS,
                    {"default": "none"},
                ),
                "resolution": resolution,
            },
        ),
    ]

    def record(
        node_class: str,
        carrier: str,
        inputs: dict[str, MappingSource],
        *,
        source_output: str = "image",
        target_output: str = "image",
        note: str | None = None,
        tier: str = "equivalent",
        tolerance: float = 1.0 / 255.0 + 1e-7,
        evidence: list[str] | None = None,
    ) -> dict[str, object]:
        return _record(
            source_pack=namespace,
            node_class=node_class,
            revision=CONTROLNET_AUX_BASELINE,
            carrier=carrier,
            rule=ReplacementRule(
                from_type=f"comfy.{namespace}.{node_class}",
                note=note or "",
                cases=(
                    ReplacementCase.build(
                        carrier,
                        inputs=inputs,
                        outputs={target_output: source_output},
                    ),
                ),
            ),
            tier=tier,
            evidence=evidence
            or [
                "tests/test_image_preprocessors.py::test_checkpoint_free_preprocessors_match_controlnet_aux_goldens",
                "tests/test_image_preprocessors.py::test_controlnet_aux_aliases_preserve_preprocessor_parameters",
            ],
            tolerances=(
                [{"metric": "max_abs", "operator": "<=", "value": tolerance}]
                if tier == "equivalent"
                else None
            ),
        )

    common = {"image": MappingSource.copy("image"), "resolution": MappingSource.copy("resolution")}
    records = [
        record(
            "CannyEdgePreprocessor",
            "dinkster.preprocess.edges",
            {
                **common,
                "method": MappingSource.constant("canny"),
                "low_threshold": MappingSource.copy("low_threshold"),
                "high_threshold": MappingSource.copy("high_threshold"),
            },
        ),
        record(
            "PyraCannyPreprocessor",
            "dinkster.preprocess.edges",
            {
                **common,
                "method": MappingSource.constant("pyramid_canny"),
                "low_threshold": MappingSource.copy("low_threshold"),
                "high_threshold": MappingSource.copy("high_threshold"),
            },
        ),
        record(
            "LineartStandardPreprocessor",
            "dinkster.preprocess.lineart",
            {
                **common,
                "gaussian_sigma": MappingSource.copy("guassian_sigma"),
                "intensity_threshold": MappingSource.copy("intensity_threshold"),
            },
        ),
        record(
            "ScribblePreprocessor",
            "dinkster.preprocess.scribble",
            {**common, "method": MappingSource.constant("threshold")},
        ),
        record(
            "Scribble_XDoG_Preprocessor",
            "dinkster.preprocess.scribble",
            {
                **common,
                "method": MappingSource.constant("xdog"),
                "threshold": MappingSource.copy("threshold"),
            },
        ),
        _record(
            source_pack=namespace,
            node_class="LineArtPreprocessor",
            revision=CONTROLNET_AUX_BASELINE,
            carrier="dinkster.preprocess.lineart_realistic",
            rule=ReplacementRule(
                from_type=f"comfy.{namespace}.LineArtPreprocessor",
                cases=(
                    ReplacementCase.build(
                        "dinkster.preprocess.lineart_realistic",
                        when=ReplacementPredicate.value_equals("coarse", "enable"),
                        inputs={
                            **common,
                            "provider": MappingSource.constant("dinkster-vision-hed"),
                            "coarse": MappingSource.constant(True),
                        },
                        outputs={"image": "image"},
                    ),
                    ReplacementCase.build(
                        "dinkster.preprocess.lineart_realistic",
                        inputs={
                            **common,
                            "provider": MappingSource.constant("dinkster-vision-hed"),
                            "coarse": MappingSource.constant(False),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors",
                "tests/test_vision_line_edge_schema.py::test_line_edge_aliases_preserve_parameters_and_refuse_excluded_models",
            ],
        ),
        record(
            "AnimeLineArtPreprocessor",
            "dinkster.preprocess.lineart_anime",
            {**common, "provider": MappingSource.constant("dinkster-vision-hed")},
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors"
            ],
        ),
        record(
            "Manga2Anime_LineArt_Preprocessor",
            "dinkster.preprocess.lineart_manga",
            {**common, "provider": MappingSource.constant("dinkster-vision-hed")},
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors"
            ],
        ),
        record(
            "AnyLineArtPreprocessor_aux",
            "dinkster.preprocess.anyline",
            {
                **common,
                "provider": MappingSource.constant("dinkster-vision-hed"),
                "merge_with_lineart": MappingSource.copy("merge_with_lineart"),
                "lineart_lower_bound": MappingSource.copy("lineart_lower_bound"),
                "lineart_upper_bound": MappingSource.copy("lineart_upper_bound"),
                "object_min_size": MappingSource.copy("object_min_size"),
                "object_connectivity": MappingSource.copy("object_connectivity"),
            },
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_anyline_merge_arms_match_pinned_controlnet_aux_vectors"
            ],
        ),
        *(
            _record(
                source_pack=namespace,
                node_class=node_class,
                revision=CONTROLNET_AUX_BASELINE,
                carrier="dinkster.preprocess.model_edges",
                rule=ReplacementRule(
                    from_type=f"comfy.{namespace}.{node_class}",
                    cases=(
                        ReplacementCase.build(
                            "dinkster.preprocess.model_edges",
                            when=ReplacementPredicate.value_equals("safe", "disable"),
                            inputs={
                                **common,
                                "provider": MappingSource.constant("dinkster-vision-hed"),
                                "safe": MappingSource.constant(False),
                                "scribble": MappingSource.constant(scribble),
                            },
                            outputs={"image": "image"},
                        ),
                        ReplacementCase.build(
                            "dinkster.preprocess.model_edges",
                            inputs={
                                **common,
                                "provider": MappingSource.constant("dinkster-vision-hed"),
                                "safe": MappingSource.constant(True),
                                "scribble": MappingSource.constant(scribble),
                            },
                            outputs={"image": "image"},
                        ),
                    ),
                ),
                tier="exact",
                evidence=[
                    "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors",
                    "tests/test_vision_line_edge_schema.py::test_line_edge_aliases_preserve_parameters_and_refuse_excluded_models",
                ],
            )
            for node_class, scribble in (
                ("HEDPreprocessor", False),
                ("FakeScribblePreprocessor", True),
            )
        ),
        record(
            "TEEDPreprocessor",
            "dinkster.preprocess.teed",
            {
                **common,
                "provider": MappingSource.constant("dinkster-vision-hed"),
                "safe_steps": MappingSource.copy("safe_steps"),
            },
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors"
            ],
        ),
        record(
            "M-LSDPreprocessor",
            "dinkster.preprocess.mlsd",
            {
                **common,
                "provider": MappingSource.constant("dinkster-vision-hed"),
                "score_threshold": MappingSource.copy("score_threshold"),
                "distance_threshold": MappingSource.copy("dist_threshold"),
            },
            tier="exact",
            evidence=[
                "packages/dinkster-vision-hed/tests/test_provider.py::test_learned_preprocessors_match_pinned_controlnet_aux_vectors"
            ],
        ),
        *(
            _record(
                source_pack=namespace,
                node_class=node_class,
                revision=CONTROLNET_AUX_BASELINE,
                carrier="dinkster.preprocess.model_edges",
                rule=ReplacementRule(
                    from_type=f"comfy.{namespace}.{node_class}",
                    note=note,
                    cases=(
                        ReplacementCase.build(
                            "dinkster.preprocess.model_edges",
                            inputs={
                                "image": MappingSource.copy("image"),
                                "provider": MappingSource.from_value(
                                    refusal_input, ValueTransform.enum_rename({})
                                ),
                                "safe": MappingSource.constant(True),
                                "scribble": MappingSource.constant(False),
                                "resolution": MappingSource.copy("resolution"),
                            },
                            outputs={"image": "image"},
                        ),
                    ),
                ),
                tier="exact",
                evidence=[
                    "tests/test_vision_line_edge_schema.py::test_line_edge_aliases_preserve_parameters_and_refuse_excluded_models"
                ],
            )
            for node_class, refusal_input, note in (
                (
                    "PiDiNetPreprocessor",
                    "safe",
                    "PiDiNet is not translated because its bundled license requires "
                    "separate permission for commercial use.",
                ),
                (
                    "Scribble_PiDiNet_Preprocessor",
                    "safe",
                    "Scribble PiDiNet is not translated because its bundled license "
                    "requires separate permission for commercial use.",
                ),
                (
                    "DiffusionEdge_Preprocessor",
                    "environment",
                    "Diffusion Edge is not translated because Dinkster does not provide "
                    "its three large checkpoints and runtime-installed dependency path.",
                ),
            )
        ),
        record(
            "BinaryPreprocessor",
            "dinkster.preprocess.binary",
            {
                **common,
                "threshold": MappingSource.copy("bin_threshold"),
            },
        ),
        record(
            "ColorPreprocessor",
            "dinkster.preprocess.color_hint",
            {
                **common,
                "method": MappingSource.constant("palette"),
                "gamma": MappingSource.constant(1.0),
            },
        ),
        record(
            "ImageLuminanceDetector",
            "dinkster.preprocess.color_hint",
            {
                **common,
                "method": MappingSource.constant("luminance"),
                "gamma": MappingSource.copy("gamma_correction"),
            },
        ),
        record(
            "ImageIntensityDetector",
            "dinkster.preprocess.color_hint",
            {
                **common,
                "method": MappingSource.constant("intensity"),
                "gamma": MappingSource.copy("gamma_correction"),
            },
        ),
        record(
            "ShufflePreprocessor",
            "dinkster.preprocess.content_shuffle",
            {**common, "seed": MappingSource.copy("seed")},
            note=(
                "The native target makes seed zero deterministic instead of using process-global "
                "random state; nonzero JSON-safe seeds preserve the source flow. The source's "
                "uint64 seeds above the JSON-safe integer maximum require review."
            ),
            tier="parametric",
        ),
        record(
            "TilePreprocessor",
            "dinkster.preprocess.tile_hint",
            {
                "image": MappingSource.copy("image"),
                "method": MappingSource.constant("pyramid"),
                "iterations": MappingSource.copy("pyrUp_iters"),
            },
            note="The source resolution input is accepted but ignored by its detector.",
        ),
        record(
            "TTPlanet_TileGF_Preprocessor",
            "dinkster.preprocess.tile_hint",
            {
                "image": MappingSource.copy("image"),
                "method": MappingSource.constant("guided"),
                "scale_factor": MappingSource.copy("scale_factor"),
                "blur_strength": MappingSource.copy("blur_strength"),
                "radius": MappingSource.copy("radius"),
                "epsilon": MappingSource.copy("eps"),
            },
            note="The source wrapper declares resolution but does not pass it to the detector.",
        ),
        record(
            "TTPlanet_TileSimple_Preprocessor",
            "dinkster.preprocess.tile_hint",
            {
                "image": MappingSource.copy("image"),
                "method": MappingSource.constant("simple"),
                "scale_factor": MappingSource.copy("scale_factor"),
                "blur_strength": MappingSource.copy("blur_strength"),
            },
        ),
        _record(
            source_pack=namespace,
            node_class="InpaintPreprocessor",
            revision=CONTROLNET_AUX_BASELINE,
            carrier="dinkster.preprocess.inpaint_hint",
            rule=ReplacementRule(
                from_type=f"comfy.{namespace}.InpaintPreprocessor",
                cases=(
                    ReplacementCase.build(
                        "dinkster.preprocess.inpaint_hint",
                        when=ReplacementPredicate.value_equals("black_pixel_for_xinsir_cn", True),
                        inputs={
                            "image": MappingSource.copy("image"),
                            "mask": MappingSource.copy("mask"),
                            "masked_value": MappingSource.constant("black"),
                        },
                        outputs={"image": "image"},
                    ),
                    ReplacementCase.build(
                        "dinkster.preprocess.inpaint_hint",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "mask": MappingSource.copy("mask"),
                            "masked_value": MappingSource.constant("negative_one"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_preprocessors.py::test_inpaint_hint_resizes_and_broadcasts_masks",
                "tests/test_image_preprocessors.py::test_controlnet_aux_aliases_preserve_preprocessor_parameters",
            ],
            tolerances=[
                {
                    "metric": "max_abs",
                    "operator": "<=",
                    "value": 1.0 / 255.0 + 1e-7,
                }
            ],
        ),
        record(
            "PixelPerfectResolution",
            "dinkster.preprocess.hint_resolution",
            {
                "image": MappingSource.copy("original_image"),
                "target_width": MappingSource.copy("image_gen_width"),
                "target_height": MappingSource.copy("image_gen_height"),
                "resize_mode": MappingSource.from_value(
                    "resize_mode",
                    ValueTransform.enum_rename(
                        {
                            "Just Resize": "stretch",
                            "Crop and Resize": "fill",
                            "Resize and Fill": "fit",
                        }
                    ),
                ),
            },
            source_output="RESOLUTION (INT)",
            target_output="resolution",
            tolerance=0.0,
        ),
        record(
            "HintImageEnchance",
            "dinkster.preprocess.hint_resize",
            {
                "image": MappingSource.copy("hint_image"),
                "target_width": MappingSource.copy("image_gen_width"),
                "target_height": MappingSource.copy("image_gen_height"),
                "resize_mode": MappingSource.from_value(
                    "resize_mode",
                    ValueTransform.enum_rename(
                        {
                            "Just Resize": "stretch",
                            "Crop and Resize": "fill",
                            "Resize and Fill": "fit",
                        }
                    ),
                ),
            },
            evidence=[
                "tests/test_image_preprocessors.py::test_hint_image_resize_matches_controlnet_aux_goldens",
                "tests/test_image_preprocessors.py::test_controlnet_aux_aliases_preserve_preprocessor_parameters",
            ],
        ),
        record(
            "DepthAnythingV2Preprocessor",
            "dinkster.preprocess.model_depth",
            {
                **common,
                "provider": MappingSource.from_value(
                    "ckpt_name",
                    ValueTransform.enum_rename(
                        {"depth_anything_v2_vitl.pth": "dinkster-vision-depth-anything-v2"}
                    ),
                ),
            },
            tier="exact",
            evidence=[
                "packages/dinkster-vision-depth-anything-v2/tests/test_provider.py::test_depth_output_matches_pinned_controlnet_aux_vector",
                "tests/test_vision_depth_anything_v2_schema.py::test_depth_aliases_are_provider_local_and_inactive",
            ],
        ),
        record(
            "AIO_Preprocessor",
            "dinkster.preprocess.model_depth",
            {
                **common,
                "provider": MappingSource.from_value(
                    "preprocessor",
                    ValueTransform.enum_rename(
                        {"DepthAnythingV2Preprocessor": "dinkster-vision-depth-anything-v2"}
                    ),
                ),
            },
            note=(
                "Only the DepthAnythingV2Preprocessor selector is supported; every other "
                "selector fails closed during enum translation."
            ),
            tier="exact",
            evidence=[
                "packages/dinkster-vision-depth-anything-v2/tests/test_provider.py::test_depth_output_matches_pinned_controlnet_aux_vector",
                "tests/test_vision_depth_anything_v2_schema.py::test_depth_aliases_are_provider_local_and_inactive",
            ],
        ),
    ]
    return source_schemas, records


def build_depth_anything_v2_registry() -> dict[str, object]:
    source_schemas, records = _preprocessor_alias_data()
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [
            schema_to_wire(schema)
            for schema in source_schemas
            if schema.node_type in DEPTH_ANYTHING_V2_SOURCE_TYPES
        ],
        "records": [
            record
            for record in records
            if cast("dict[str, object]", record["source"])["nodeType"]
            in DEPTH_ANYTHING_V2_SOURCE_TYPES
        ],
    }


def build_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != COMFY_BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.cpu = True
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_compositing,
        nodes_dataset,
        nodes_image_compare,
        nodes_images,
        nodes_mask,
        nodes_morphology,
        nodes_post_processing,
        nodes_rebatch,
    )
    from nodes import (  # pyright: ignore[reportMissingImports]
        EmptyImage,
        ImageBatch,
        ImageInvert,
        ImagePadForOutpaint,
        ImageScale,
        ImageScaleBy,
    )

    transitions = [
        "horizontal slide",
        "vertical slide",
        "box",
        "circle",
        "horizontal door",
        "vertical door",
        "fade",
    ]
    easings = [
        "linear",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "bounce",
        "elastic",
        "glitchy",
        "exponential_ease_out",
    ]
    kj_transition_controls: dict[str, object] = {
        "interpolation": (easings,),
        "transition_type": (transitions,),
        "transitioning_frames": ("INT", {"default": 2, "min": 0, "max": 4096}),
        "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
        "reverse": ("BOOLEAN", {"default": False}),
        "device": (["CPU", "GPU"], {"default": "CPU"}),
    }
    batch_layout_source_schemas = [
        _dynamic_image_schema(
            _static_schema(
                "ImageBatchMulti",
                namespace="comfyui-kjnodes",
                required={"inputcount": ("INT", {"default": 2, "min": 2, "max": 1000})},
                return_names=("images",),
            )
        ),
        _static_schema(
            "ImageBatchRepeatInterleaving",
            namespace="comfyui-kjnodes",
            required={
                "images": ("IMAGE",),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            optional={"mask": ("MASK",)},
            returns=("IMAGE", "MASK"),
        ),
        _static_schema(
            "InsertImagesToBatchIndexed",
            namespace="comfyui-kjnodes",
            required={
                "original_images": ("IMAGE",),
                "images_to_insert": ("IMAGE",),
                "indexes": ("STRING", {"default": "0, 1, 2"}),
            },
            optional={"mode": (["replace", "insert"], {"default": "replace"})},
        ),
        _static_schema(
            "ReverseImageBatch",
            namespace="comfyui-kjnodes",
            required={"images": ("IMAGE",)},
        ),
        _static_schema(
            "ReplaceImagesInBatch",
            namespace="comfyui-kjnodes",
            required={"start_index": ("INT", {"default": 1, "min": 0, "max": 4096})},
            optional={
                "original_images": ("IMAGE",),
                "replacement_images": ("IMAGE",),
                "original_masks": ("MASK",),
                "replacement_masks": ("MASK",),
            },
            returns=("IMAGE", "MASK"),
        ),
        _static_schema(
            "ShuffleImageBatch",
            namespace="comfyui-kjnodes",
            required={
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 123, "min": 0}),
            },
        ),
        _static_schema(
            "ImageGridComposite2x2",
            namespace="comfyui-kjnodes",
            required={f"image{index}": ("IMAGE",) for index in range(1, 5)},
        ),
        _static_schema(
            "ImageGridComposite3x3",
            namespace="comfyui-kjnodes",
            required={f"image{index}": ("IMAGE",) for index in range(1, 10)},
        ),
        _static_schema(
            "ImageGridtoBatch",
            namespace="comfyui-kjnodes",
            required={
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 3, "min": 1, "max": 8}),
                "rows": ("INT", {"default": 0, "min": 0, "max": 8}),
            },
        ),
        _dynamic_image_schema(
            _static_schema(
                "TransitionImagesMulti",
                namespace="comfyui-kjnodes",
                required={
                    "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000}),
                    **kj_transition_controls,
                    "transitioning_frames": (
                        "INT",
                        {"default": 2, "min": 2, "max": 4096},
                    ),
                },
            )
        ),
        _static_schema(
            "TransitionImagesInBatch",
            namespace="comfyui-kjnodes",
            required={
                "images": ("IMAGE",),
                **kj_transition_controls,
                "transitioning_frames": ("INT", {"default": 1, "min": 0, "max": 4096}),
            },
        ),
        _static_schema(
            "ImageBatchJoinWithTransition",
            namespace="comfyui-kjnodes",
            required={
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "start_index": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                **kj_transition_controls,
                "transitioning_frames": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
        ),
        _static_schema(
            "ImageBatchMultiple+",
            namespace="comfyui_essentials",
            required={
                "image_1": ("IMAGE",),
                "method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "lanczos"},
                ),
            },
            optional={f"image_{index}": ("IMAGE",) for index in range(2, 6)},
        ),
        _static_schema(
            "ImageExpandBatch+",
            namespace="comfyui_essentials",
            required={
                "image": ("IMAGE",),
                "size": ("INT", {"default": 16, "min": 1}),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],),
            },
        ),
        _static_schema(
            "ImageFromBatch+",
            namespace="comfyui_essentials",
            required={
                "image": ("IMAGE",),
                "start": ("INT", {"default": 0, "min": 0}),
                "length": ("INT", {"default": -1, "min": -1}),
            },
        ),
        _static_schema(
            "ImageListToBatch+",
            namespace="comfyui_essentials",
            required={"image": ("IMAGE",)},
            input_is_list=True,
        ),
        _static_schema(
            "ImageBatchToList+",
            namespace="comfyui_essentials",
            required={"image": ("IMAGE",)},
            output_is_list=(True,),
        ),
        _static_schema(
            "ImageTile+",
            namespace="comfyui_essentials",
            required={
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 256}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 256}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5}),
                "overlap_x": ("INT", {"default": 0, "min": 0}),
                "overlap_y": ("INT", {"default": 0, "min": 0}),
            },
            returns=("IMAGE", "INT", "INT", "INT", "INT"),
            return_names=("IMAGE", "tile_width", "tile_height", "overlap_x", "overlap_y"),
        ),
        _static_schema(
            "ImageUntile+",
            namespace="comfyui_essentials",
            required={
                "tiles": ("IMAGE",),
                "overlap_x": ("INT", {"default": 0, "min": 0}),
                "overlap_y": ("INT", {"default": 0, "min": 0}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 256}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 256}),
            },
        ),
        _static_schema(
            "MaskBatch+",
            namespace="comfyui_essentials",
            required={"mask1": ("MASK",), "mask2": ("MASK",)},
            returns=("MASK",),
        ),
        _static_schema(
            "MaskExpandBatch+",
            namespace="comfyui_essentials",
            required={
                "mask": ("MASK",),
                "size": ("INT", {"default": 16, "min": 1}),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],),
            },
            returns=("MASK",),
        ),
        _static_schema(
            "MaskFromBatch+",
            namespace="comfyui_essentials",
            required={
                "mask": ("MASK",),
                "start": ("INT", {"default": 0, "min": 0}),
                "length": ("INT", {"default": 1, "min": 1}),
            },
            returns=("MASK",),
        ),
    ]
    core_snapshot = json.loads(
        (
            REPO
            / "packages"
            / "dinkster-compat-comfy"
            / "src"
            / "dinkster_compat_comfy"
            / "core_schemas.json"
        ).read_text(encoding="utf-8")
    )
    resize_source_wire = cast(
        "dict[str, Any]",
        cast("dict[str, object]", core_snapshot["schemas"])["comfy.ResizeImageMaskNode"],
    )
    geometry_source_schemas = [
        schema_from_wire(resize_source_wire),
        _static_schema(
            "ImageResizeKJ",
            namespace="comfyui-kjnodes",
            required={
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
                "keep_proportion": ("BOOLEAN", {"default": False}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512}),
            },
            optional={
                "get_image_size": ("IMAGE",),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            returns=("IMAGE", "INT", "INT"),
            return_names=("IMAGE", "width", "height"),
        ),
        _static_schema(
            "ImageResizeKJv2",
            namespace="comfyui-kjnodes",
            required={
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "upscale_method": (
                    [
                        "nearest-exact",
                        "bilinear",
                        "area",
                        "bicubic",
                        "lanczos",
                        "nvidia_rtx_vsr",
                    ],
                ),
                "keep_proportion": (
                    [
                        "stretch",
                        "resize",
                        "pad",
                        "pad_edge",
                        "pad_edge_pixel",
                        "crop",
                        "pillarbox_blur",
                        "total_pixels",
                    ],
                    {"default": False},
                ),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center"},
                ),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512}),
            },
            optional={"mask": ("MASK",), "device": (["cpu", "gpu"],)},
            returns=("IMAGE", "INT", "INT", "MASK"),
            return_names=("IMAGE", "width", "height", "mask"),
        ),
        _static_schema(
            "GetImageSizeAndCount",
            namespace="comfyui-kjnodes",
            required={"image": ("IMAGE",)},
            returns=("IMAGE", "INT", "INT", "INT"),
            return_names=("image", "width", "height", "count"),
        ),
        _static_schema(
            "ImageCropByMask",
            namespace="comfyui-kjnodes",
            required={"image": ("IMAGE",), "mask": ("MASK",)},
        ),
        _static_schema(
            "ImageResize+",
            namespace="comfyui_essentials",
            required={
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384}),
                "interpolation": (
                    ["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],
                ),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (
                    [
                        "always",
                        "downscale if bigger",
                        "upscale if smaller",
                        "if bigger area",
                        "if smaller area",
                    ],
                ),
                "multiple_of": ("INT", {"default": 0, "min": 0, "max": 512}),
            },
            returns=("IMAGE", "INT", "INT"),
            return_names=("IMAGE", "width", "height"),
        ),
        _static_schema(
            "ImageCrop+",
            namespace="comfyui_essentials",
            required={
                "image": ("IMAGE",),
                "width": ("INT", {"default": 256, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 256, "min": 0, "max": 16384}),
                "position": (
                    [
                        "top-left",
                        "top-center",
                        "top-right",
                        "right-center",
                        "bottom-right",
                        "bottom-center",
                        "bottom-left",
                        "left-center",
                        "center",
                    ],
                ),
                "x_offset": ("INT", {"default": 0, "min": -99999}),
                "y_offset": ("INT", {"default": 0, "min": -99999}),
            },
            returns=("IMAGE", "INT", "INT"),
            return_names=("IMAGE", "x", "y"),
        ),
        _static_schema(
            "GetImageSize+",
            namespace="comfyui_essentials",
            required={"image": ("IMAGE",)},
            returns=("INT", "INT", "INT"),
            return_names=("width", "height", "count"),
        ),
        _static_schema(
            "ImageRemoveAlpha+",
            namespace="comfyui_essentials",
            required={"image": ("IMAGE",)},
        ),
        _static_schema(
            "MaskBoundingBox+",
            namespace="comfyui_essentials",
            required={
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 256}),
            },
            optional={"image_optional": ("IMAGE",)},
            returns=("MASK", "IMAGE", "INT", "INT", "INT", "INT"),
            return_names=("MASK", "IMAGE", "x", "y", "width", "height"),
        ),
    ]
    preprocessor_source_schemas, preprocessor_records = _preprocessor_alias_data()
    preprocessor_source_schemas = [
        schema
        for schema in preprocessor_source_schemas
        if schema.node_type not in DEPTH_ANYTHING_V2_SOURCE_TYPES
    ]
    preprocessor_records = [
        record
        for record in preprocessor_records
        if cast("dict[str, object]", record["source"])["nodeType"]
        not in DEPTH_ANYTHING_V2_SOURCE_TYPES
    ]

    source_schemas = [
        translate_node("ImageBatch", ImageBatch, CompatTranslation()).schema(),
        _core_v3_schema(nodes_post_processing.BatchImagesNode),
        _core_v3_schema(nodes_images.ResizeAndPadImage),
        _core_v3_schema(nodes_image_compare.ImageCompare),
        _core_v3_schema(nodes_images.RepeatImageBatch),
        _core_v3_schema(nodes_images.ImageFromBatch),
        _core_v3_schema(nodes_rebatch.ImageRebatch),
        _core_v3_schema(nodes_images.ImageStitch),
        *batch_layout_source_schemas,
        *geometry_source_schemas,
        _core_v3_schema(nodes_images.BoundingBox),
        translate_node("ImageScale", ImageScale, CompatTranslation()).schema(),
        translate_node("ImageScaleBy", ImageScaleBy, CompatTranslation()).schema(),
        _core_v3_schema(nodes_post_processing.ImageScaleToTotalPixels),
        _core_v3_schema(nodes_images.ImageScaleToMaxDimension),
        _core_v3_schema(nodes_images.ImageFlip),
        _core_v3_schema(nodes_images.ImageRotate),
        translate_node(
            "ImagePadForOutpaint",
            ImagePadForOutpaint,
            CompatTranslation(),
        ).schema(),
        _core_v3_schema(nodes_images.ImageCrop),
        _core_v3_schema(nodes_images.ImageCropV2),
        translate_node("BBox (mtb)", _MTBBox, CompatTranslation(), namespace="comfy-mtb").schema(),
        translate_node(
            "Uncrop (mtb)",
            _MTBUncrop,
            CompatTranslation(),
            namespace="comfy-mtb",
        ).schema(),
        _core_v3_schema(nodes_images.GetImageSize),
        _core_v3_schema(nodes_mask.ImageCompositeMasked),
        _core_v3_schema(nodes_post_processing.Blend),
        _core_v3_schema(nodes_compositing.PorterDuffImageComposite),
        translate_node("ImageInvert", ImageInvert, CompatTranslation()).schema(),
        _core_v3_schema(nodes_dataset.NormalizeImagesNode),
        _core_v3_schema(nodes_dataset.AdjustBrightnessNode),
        _core_v3_schema(nodes_dataset.AdjustContrastNode),
        _core_v3_schema(nodes_post_processing.Blur),
        _core_v3_schema(nodes_post_processing.Sharpen),
        _core_v3_schema(nodes_post_processing.Quantize),
        _core_v3_schema(nodes_morphology.Morphology),
        _core_v3_schema(nodes_compositing.SplitImageWithAlpha),
        _core_v3_schema(nodes_compositing.JoinImageWithAlpha),
        _core_v3_schema(nodes_morphology.ImageRGBToYUV),
        _core_v3_schema(nodes_morphology.ImageYUVToRGB),
        _core_v3_schema(nodes_mask.SolidMask),
        _core_v3_schema(nodes_mask.InvertMask),
        _core_v3_schema(nodes_mask.CropMask),
        _core_v3_schema(nodes_mask.FeatherMask),
        _core_v3_schema(nodes_mask.GrowMask),
        _core_v3_schema(nodes_mask.ThresholdMask),
        _core_v3_schema(nodes_mask.MaskComposite),
        _core_v3_schema(nodes_mask.ImageToMask),
        _core_v3_schema(nodes_mask.ImageColorToMask),
        _core_v3_schema(nodes_mask.MaskToImage),
        translate_node("EmptyImage", EmptyImage, CompatTranslation()).schema(),
        translate_node(
            "CreateShapeMask",
            _KJCreateShapeMask,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "CreateTextMask",
            _KJCreateTextMask,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "GrowMaskWithBlur",
            _KJGrowMaskWithBlur,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "ColorToMask",
            _KJColorToMask,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "GetMaskSizeAndCount",
            _KJGetMaskSizeAndCount,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "DrawMaskOnImage",
            _KJDrawMaskOnImage,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "BboxVisualize",
            _KJBboxVisualize,
            CompatTranslation(),
            namespace="comfyui-kjnodes",
        ).schema(),
        translate_node(
            "MaskFromColor+",
            _EssentialsMaskFromColor,
            CompatTranslation(),
            namespace="comfyui_essentials",
        ).schema(),
        translate_node(
            "TransitionMask+",
            _EssentialsTransitionMask,
            CompatTranslation(),
            namespace="comfyui_essentials",
        ).schema(),
        translate_node(
            "DrawText+",
            _EssentialsDrawText,
            CompatTranslation(),
            namespace="comfyui_essentials",
        ).schema(),
        *preprocessor_source_schemas,
    ]

    batch_evidence = [
        "tests/test_image_batch_operations.py::test_core_batch_operations_match_pinned_comfy_goldens"
    ]
    ecosystem_batch_evidence = [
        "tests/test_image_batch_operations.py::test_batch_edit_consolidates_core_and_ecosystem_order_operations"
    ]
    grid_evidence = [
        "tests/test_image_layout_operations.py::test_grid_compose_uses_row_major_dynamic_inputs"
    ]
    tile_evidence = ["tests/test_image_tile_operations.py::test_image_tile_split_merge_round_trip"]
    transition_evidence = [
        "tests/test_image_transition_operations.py::test_within_batch_transitions_every_adjacent_pair",
        "tests/test_image_transition_operations.py::test_kj_dynamic_transition_count_preserves_late_and_missing_inputs",
    ]
    transition_rename = ValueTransform.enum_rename(
        {
            "horizontal slide": "horizontal_slide",
            "vertical slide": "vertical_slide",
            "box": "box",
            "circle": "circle",
            "horizontal door": "horizontal_door",
            "vertical door": "vertical_door",
            "fade": "fade",
        }
    )
    replace_image_pair = ReplacementPredicate.all_of(
        ReplacementPredicate.input_connected("original_images"),
        ReplacementPredicate.input_connected("replacement_images"),
    )
    replace_mask_pair = ReplacementPredicate.all_of(
        ReplacementPredicate.input_connected("original_masks"),
        ReplacementPredicate.input_connected("replacement_masks"),
    )
    batch_layout_records = [
        _record(
            source_pack="comfy-core",
            node_class="BatchImagesNode",
            revision="b78cec87",
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.BatchImagesNode",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        inputs={
                            "operation": MappingSource.constant("concat"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("pad_with_one"),
                            "interpolation": MappingSource.constant("bilinear"),
                        },
                        input_families={
                            "images": InputFamilyMapping.copy(
                                "images", inputs={"value": MappingSource.copy("image")}
                            )
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=["tests/test_image_comfy_aliases.py::test_new_image_aliases_execute"],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ResizeAndPadImage",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ResizeAndPadImage",
                cases=tuple(
                    ReplacementCase.build(
                        "dinkster.image.resize",
                        when=ReplacementPredicate.value_equals("padding_color", "black")
                        if pad_value == 0.0
                        else None,
                        inputs={
                            "image": MappingSource.copy("image"),
                            "width": MappingSource.copy("target_width"),
                            "height": MappingSource.copy("target_height"),
                            "interpolation": MappingSource.copy("interpolation"),
                            "target": MappingSource.constant("dimensions"),
                            "mode": MappingSource.constant("pad"),
                            "fit_rounding": MappingSource.constant("floor"),
                            "pad_value": MappingSource.constant(pad_value),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    )
                    for pad_value in (0.0, 1.0)
                ),
            ),
            tier="equivalent",
            evidence=["tests/test_image_comfy_aliases.py::test_new_image_aliases_execute"],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageCompare",
            revision="b78cec87",
            carrier="dinkster.image.compare",
            rule=ReplacementRule(
                from_type="comfy.ImageCompare",
                note="Paired previews retain side identity; compare_view is client state.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.compare",
                        inputs={name: MappingSource.copy(name) for name in ("image_a", "image_b")},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_image_comfy_aliases.py::test_new_image_aliases_execute"],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageBatch",
            revision="b78cec87",
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.ImageBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        inputs={
                            "operation": MappingSource.constant("concat"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("pad_with_one"),
                            "interpolation": MappingSource.constant("bilinear"),
                        },
                        input_families={"images": _family_members("image1", "image2")},
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="exact",
            evidence=batch_evidence,
        ),
        _record(
            source_pack="comfy-core",
            node_class="RepeatImageBatch",
            revision="b78cec87",
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.RepeatImageBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("repeat_all"),
                            "amount": MappingSource.copy("amount"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=batch_evidence,
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageFromBatch",
            revision="b78cec87",
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.ImageFromBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("range"),
                            "start": MappingSource.copy("batch_index"),
                            "count": MappingSource.copy("length"),
                            "range_end": MappingSource.constant("clamp"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=batch_evidence,
        ),
        _record(
            source_pack="comfy-core",
            node_class="RebatchImages",
            revision="b78cec87",
            carrier="dinkster.image.rebatch",
            rule=ReplacementRule(
                from_type="comfy.RebatchImages",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.rebatch",
                        nodes={
                            "batch_size": ReplacementNode.build(
                                "std.list.element", values={"index": 0}
                            )
                        },
                        inputs={
                            "images": MappingSource.copy("images"),
                            "batch_size:list": MappingSource.copy("batch_size"),
                        },
                        links=(ReplacementLink("batch_size:item", "batch_size"),),
                        outputs={"images": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=batch_evidence,
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageStitch",
            revision="b78cec87",
            carrier="dinkster.image.stitch",
            rule=ReplacementRule(
                from_type="comfy.ImageStitch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.stitch",
                        inputs={
                            "first": MappingSource.copy("image1"),
                            "second": MappingSource.copy("image2"),
                            "direction": MappingSource.copy("direction"),
                            "match_image_size": MappingSource.copy("match_image_size"),
                            "spacing_width": MappingSource.copy("spacing_width"),
                            "spacing_color": MappingSource.copy("spacing_color"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_layout_operations.py::test_image_stitch_matches_pinned_comfy_golden"
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageBatchMulti",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageBatchMulti",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        inputs={
                            "operation": MappingSource.constant("concat"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("pad_with_one"),
                            "interpolation": MappingSource.constant("bilinear"),
                            "input_count": MappingSource.copy("inputcount"),
                        },
                        input_families={
                            "images": InputFamilyMapping.copy(
                                "images", inputs={"value": MappingSource.copy("value")}
                            )
                        },
                        outputs={"image": "images"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_kj_dynamic_batch_count_preserves_late_and_missing_inputs",
                "tests/test_image_comfy_aliases.py::test_batch_aliases_preserve_dynamic_list_mask_and_default_contracts",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageBatchRepeatInterleaving",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageBatchRepeatInterleaving",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("repeat_each"),
                            "amount": MappingSource.copy("repeats"),
                            "mask": MappingSource.copy("mask"),
                            "generate_repeat_marker": MappingSource.constant(True),
                        },
                        outputs={"image": "image", "mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_kj_repeat_interleaving_preserves_or_generates_masks",
                "tests/test_image_comfy_aliases.py::test_batch_aliases_preserve_dynamic_list_mask_and_default_contracts",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="InsertImagesToBatchIndexed",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.InsertImagesToBatchIndexed",
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.image.batch.combine",
                            inputs={
                                "operation": MappingSource.from_value(
                                    "mode",
                                    ValueTransform.enum_rename(
                                        {
                                            "replace": "replace_indexed",
                                            "insert": "insert_indexed",
                                        }
                                    ),
                                ),
                                "indexes": MappingSource.copy("indexes"),
                                "shape_policy": MappingSource.constant("strict"),
                                "channel_policy": MappingSource.constant("strict"),
                            },
                            input_families={
                                "images": _family_members("original_images", "images_to_insert")
                            },
                            outputs={"image": "image"},
                        ),
                        target_input="operation",
                        source_input="mode",
                        choices={"replace": "replace_indexed", "insert": "insert_indexed"},
                        fallback="replace",
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_image_batch_combine_supports_dynamic_concat_insert_and_replace"
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ReverseImageBatch",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ReverseImageBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("reverse"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=ecosystem_batch_evidence,
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ReplaceImagesInBatch",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ReplaceImagesInBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        when=ReplacementPredicate.all_of(
                            replace_image_pair,
                            replace_mask_pair,
                        ),
                        inputs={
                            "operation": MappingSource.constant("replace_range"),
                            "index": MappingSource.copy("start_index"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("strict"),
                            "interpolation": MappingSource.constant("lanczos"),
                        },
                        input_families={
                            "images": _family_members("original_images", "replacement_images"),
                            "masks": _family_members("original_masks", "replacement_masks"),
                        },
                        outputs={"image": "image", "mask": "mask"},
                    ),
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        when=replace_image_pair,
                        inputs={
                            "operation": MappingSource.constant("replace_range"),
                            "index": MappingSource.copy("start_index"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("strict"),
                            "interpolation": MappingSource.constant("lanczos"),
                            "mask_fallback": MappingSource.constant("zero_64"),
                        },
                        input_families={
                            "images": _family_members("original_images", "replacement_images")
                        },
                        outputs={"image": "image", "mask": "mask"},
                    ),
                    ReplacementCase.build(
                        "dinkster.mask.batch.combine",
                        when=replace_mask_pair,
                        nodes={
                            "image": ReplacementNode.build(
                                "dinkster.image.generate",
                                values={
                                    "operation": "solid",
                                    "color_source": "hex",
                                    "width": 64,
                                    "height": 64,
                                    "batch_size": 1,
                                    "channels": "rgb",
                                    "color_a": "#000000",
                                },
                            )
                        },
                        inputs={
                            "operation": MappingSource.constant("replace_range"),
                            "index": MappingSource.copy("start_index"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "interpolation": MappingSource.constant("nearest-exact"),
                        },
                        input_families={
                            "masks": _family_members("original_masks", "replacement_masks")
                        },
                        outputs={"image:image": "image", "mask": "mask"},
                    ),
                    ReplacementCase.build(
                        "dinkster.image.generate",
                        nodes={
                            "mask": ReplacementNode.build(
                                "dinkster.mask.make",
                                values={
                                    "operation": "solid",
                                    "width": 64,
                                    "height": 64,
                                    "batch_size": 1,
                                    "foreground": 0.0,
                                },
                            )
                        },
                        inputs={
                            "operation": MappingSource.constant("solid"),
                            "color_source": MappingSource.constant("hex"),
                            "width": MappingSource.constant(64),
                            "height": MappingSource.constant(64),
                            "batch_size": MappingSource.constant(1),
                            "channels": MappingSource.constant("rgb"),
                            "color_a": MappingSource.constant("#000000"),
                        },
                        outputs={"image": "image", "mask:mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_replace_image_and_mask_pairs_preserves_both_output_contracts",
                "tests/test_image_comfy_aliases.py::test_batch_aliases_preserve_dynamic_list_mask_and_default_contracts",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ShuffleImageBatch",
            revision=KJ_BASELINE,
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ShuffleImageBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("shuffle"),
                            "seed": MappingSource.copy("seed"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_shuffle_matches_pytorch_cpu_randperm"
            ],
        ),
        *(
            _record(
                source_pack="comfyui-kjnodes",
                node_class=f"ImageGridComposite{size}x{size}",
                revision=KJ_BASELINE,
                carrier="dinkster.image.grid.compose",
                rule=ReplacementRule(
                    from_type=f"comfy.comfyui-kjnodes.ImageGridComposite{size}x{size}",
                    cases=(
                        ReplacementCase.build(
                            "dinkster.image.grid.compose",
                            inputs={"columns": MappingSource.constant(size)},
                            input_families={
                                "images": _family_members(
                                    *(f"image{index}" for index in range(1, size * size + 1))
                                )
                            },
                            outputs={"image": "image"},
                        ),
                    ),
                ),
                tier="parametric",
                evidence=grid_evidence,
            )
            for size in (2, 3)
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageGridtoBatch",
            revision=KJ_BASELINE,
            carrier="dinkster.image.grid.decompose",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageGridtoBatch",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.grid.decompose",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "columns": MappingSource.copy("columns"),
                            "rows": MappingSource.copy("rows"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_layout_operations.py::test_grid_decompose_matches_kj_crop_and_batch_order"
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="TransitionImagesMulti",
            revision=KJ_BASELINE,
            carrier="dinkster.image.transition",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.TransitionImagesMulti",
                note="device is host-managed in Dinkster.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.transition",
                        inputs={
                            "mode": MappingSource.constant("between_inputs"),
                            "transition": MappingSource.from_value(
                                "transition_type", transition_rename
                            ),
                            "transitioning_frames": MappingSource.copy("transitioning_frames"),
                            "easing": MappingSource.copy("interpolation"),
                            "blur_radius": MappingSource.copy("blur_radius"),
                            "reverse": MappingSource.copy("reverse"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "resize_interpolation": MappingSource.constant("lanczos"),
                            "input_count": MappingSource.copy("inputcount"),
                        },
                        input_families={
                            "images": InputFamilyMapping.copy(
                                "images", inputs={"value": MappingSource.copy("value")}
                            )
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=transition_evidence,
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="TransitionImagesInBatch",
            revision=KJ_BASELINE,
            carrier="dinkster.image.transition",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.TransitionImagesInBatch",
                note="device is host-managed in Dinkster.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.transition",
                        inputs={
                            "mode": MappingSource.constant("within_batch"),
                            "transition": MappingSource.from_value(
                                "transition_type", transition_rename
                            ),
                            "transitioning_frames": MappingSource.copy("transitioning_frames"),
                            "easing": MappingSource.copy("interpolation"),
                            "blur_radius": MappingSource.copy("blur_radius"),
                            "reverse": MappingSource.copy("reverse"),
                            "shape_policy": MappingSource.constant("strict"),
                        },
                        input_families={"images": _family_members("images")},
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=transition_evidence,
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageBatchJoinWithTransition",
            revision=KJ_BASELINE,
            carrier="dinkster.image.transition",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageBatchJoinWithTransition",
                note="device is host-managed in Dinkster.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.transition",
                        inputs={
                            "mode": MappingSource.constant("join_batches"),
                            "transition": MappingSource.from_value(
                                "transition_type", transition_rename
                            ),
                            "transitioning_frames": MappingSource.copy("transitioning_frames"),
                            "easing": MappingSource.copy("interpolation"),
                            "blur_radius": MappingSource.copy("blur_radius"),
                            "reverse": MappingSource.copy("reverse"),
                            "start_index": MappingSource.copy("start_index"),
                            "shape_policy": MappingSource.constant("strict"),
                        },
                        input_families={"images": _family_members("images_1", "images_2")},
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_transition_operations.py::test_join_batches_uses_start_index_and_replaces_the_first_tail"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageBatchMultiple+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageBatchMultiple+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.combine",
                        inputs={
                            "operation": MappingSource.constant("concat"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "channel_policy": MappingSource.constant("strict"),
                            "interpolation": MappingSource.copy("method"),
                        },
                        input_families={
                            "images": _family_members(
                                "image_1", "image_2", "image_3", "image_4", "image_5"
                            )
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_image_batch_combine_has_explicit_spatial_and_channel_policies"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageExpandBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageExpandBatch+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("resize_count"),
                            "size": MappingSource.copy("size"),
                            "expansion": MappingSource.from_value(
                                "method",
                                ValueTransform.enum_rename(
                                    {
                                        "expand": "expand",
                                        "repeat all": "repeat_all",
                                        "repeat first": "repeat_first",
                                        "repeat last": "repeat_last",
                                    }
                                ),
                            ),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_batch_resize_count_matches_essentials_expansion"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageFromBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageFromBatch+",
                note=("A negative length maps to 16384 and clamps to the available batch."),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        when=ReplacementPredicate.value_equals("length", -1),
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("range"),
                            "start": MappingSource.copy("start"),
                            "count": MappingSource.constant(16384),
                            "range_end": MappingSource.constant("clamp"),
                        },
                        outputs={"image": "image"},
                    ),
                    ReplacementCase.build(
                        "dinkster.image.batch.edit",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("range"),
                            "start": MappingSource.copy("start"),
                            "count": MappingSource.copy("length"),
                            "range_end": MappingSource.constant("clamp"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=ecosystem_batch_evidence,
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageListToBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.list.to_batch",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageListToBatch+",
                note="Comfy list-input semantics map to a typed image list.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.list.to_batch",
                        inputs={
                            "images": MappingSource.copy("image"),
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "interpolation": MappingSource.constant("bicubic"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_image_list_conversions_and_rebatch_preserve_frame_order"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageBatchToList+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.batch.to_list",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageBatchToList+",
                note="Comfy list-output semantics map to a typed image list.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.batch.to_list",
                        inputs={"image": MappingSource.copy("image")},
                        outputs={"images": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_image_list_conversions_and_rebatch_preserve_frame_order"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageTile+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.tiles.split",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageTile+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.tiles.split",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "rows": MappingSource.copy("rows"),
                            "columns": MappingSource.copy("cols"),
                            "overlap": MappingSource.copy("overlap"),
                            "overlap_x": MappingSource.copy("overlap_x"),
                            "overlap_y": MappingSource.copy("overlap_y"),
                        },
                        outputs={
                            "tiles": "IMAGE",
                            "tile_width": "tile_width",
                            "tile_height": "tile_height",
                            "overlap_x": "overlap_x",
                            "overlap_y": "overlap_y",
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=tile_evidence,
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageUntile+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.tiles.merge",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageUntile+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.tiles.merge",
                        inputs={
                            "tiles": MappingSource.copy("tiles"),
                            "overlap_x": MappingSource.copy("overlap_x"),
                            "overlap_y": MappingSource.copy("overlap_y"),
                            "rows": MappingSource.copy("rows"),
                            "columns": MappingSource.copy("cols"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=tile_evidence,
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="MaskBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.mask.batch.combine",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.MaskBatch+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.batch.combine",
                        inputs={
                            "shape_policy": MappingSource.constant("resize_to_first"),
                            "interpolation": MappingSource.constant("bicubic"),
                        },
                        input_families={"masks": _family_members("mask1", "mask2")},
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_mask_batch_edit_and_combine_share_batch_order_contracts"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="MaskExpandBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.mask.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.MaskExpandBatch+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.batch.edit",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("resize_count"),
                            "size": MappingSource.copy("size"),
                            "expansion": MappingSource.from_value(
                                "method",
                                ValueTransform.enum_rename(
                                    {
                                        "expand": "expand",
                                        "repeat all": "repeat_all",
                                        "repeat first": "repeat_first",
                                        "repeat last": "repeat_last",
                                    }
                                ),
                            ),
                        },
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_mask_batch_edit_and_combine_share_batch_order_contracts"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="MaskFromBatch+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.mask.batch.edit",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.MaskFromBatch+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.batch.edit",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("range"),
                            "start": MappingSource.copy("start"),
                            "count": MappingSource.copy("length"),
                            "range_end": MappingSource.constant("clamp"),
                        },
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_batch_operations.py::test_mask_batch_edit_and_combine_share_batch_order_contracts"
            ],
        ),
    ]
    resize_info_node = {"info": ReplacementNode.build("dinkster.image.info")}
    resize_info_link = (ReplacementLink("image", "info:image"),)
    resize_outputs = {
        "image": "IMAGE",
        "info:width": "width",
        "info:height": "height",
    }
    resize_mask_outputs = {**resize_outputs, "mask": "mask"}

    def kj_v1_resize_case(
        mode: str,
        *,
        when: ReplacementPredicate | None = None,
        reference: bool = False,
    ) -> ReplacementCase:
        inputs = {
            "image": MappingSource.copy("image"),
            "target": MappingSource.constant("match" if reference else "dimensions"),
            "mode": MappingSource.constant(mode),
            "interpolation": MappingSource.copy("upscale_method"),
            "divisibility": MappingSource.constant("crop"),
            "multiple_of": MappingSource.copy("divisible_by"),
        }
        if reference:
            inputs["reference"] = MappingSource.copy("get_image_size")
        else:
            inputs["width"] = MappingSource.copy("width")
            inputs["height"] = MappingSource.copy("height")
        if mode == "fill":
            inputs["mode_anchor"] = MappingSource.constant("center")
        return ReplacementCase.build(
            "dinkster.image.resize",
            when=when,
            nodes=resize_info_node,
            inputs=inputs,
            links=resize_info_link,
            outputs=resize_outputs,
        )

    kj_v2_mode_choices = {
        "stretch": "stretch",
        "resize": "fit",
        "pad": "pad",
        "pad_edge": "pad",
        "pad_edge_pixel": "pad",
        "crop": "fill",
        "pillarbox_blur": "pad",
    }
    kj_v2_padding_choices = {
        "pad": "constant",
        "pad_edge": "edge_average",
        "pad_edge_pixel": "edge_pixel",
        "pillarbox_blur": "blurred_background",
    }
    kj_anchor_transform = ValueTransform.enum_rename(
        {
            "disabled": "center",
            "center": "center",
            "top": "top",
            "bottom": "bottom",
            "left": "left",
            "right": "right",
        }
    )
    identity_interpolation = ValueTransform.enum_rename(
        {name: name for name in ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")}
    )

    def kj_v2_resize_case(source_mode: str, *, fallback: bool = False) -> ReplacementCase:
        mode = kj_v2_mode_choices[source_mode]
        inputs = {
            "image": MappingSource.copy("image"),
            "target": MappingSource.constant("dimensions"),
            "width": MappingSource.copy("width"),
            "height": MappingSource.copy("height"),
            "mode": MappingSource.constant(mode),
            "interpolation": MappingSource.from_value("upscale_method", identity_interpolation),
            "mask": MappingSource.copy("mask"),
            "divisibility": MappingSource.constant("crop"),
            "multiple_of": MappingSource.copy("divisible_by"),
        }
        if mode in ("fill", "pad"):
            inputs["mode_anchor"] = MappingSource.from_value("crop_position", kj_anchor_transform)
        padding = kj_v2_padding_choices.get(source_mode)
        if padding is not None:
            inputs["mode_padding"] = MappingSource.constant(padding)
        if padding == "constant":
            inputs["pad_color"] = MappingSource.copy("pad_color")
        return ReplacementCase.build(
            "dinkster.image.resize",
            when=(
                None
                if fallback
                else ReplacementPredicate.value_equals("keep_proportion", source_mode)
            ),
            nodes=resize_info_node,
            inputs=inputs,
            links=resize_info_link,
            outputs=resize_mask_outputs,
        )

    def kj_v2_total_pixels_cases() -> tuple[ReplacementCase, ...]:
        cases: list[ReplacementCase] = []
        for width_linked, height_linked in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            predicates = [ReplacementPredicate.value_equals("keep_proportion", "total_pixels")]
            for source, linked in (("width", width_linked), ("height", height_linked)):
                connected = ReplacementPredicate.input_connected(source)
                predicates.append(connected if linked else ReplacementPredicate.not_(connected))
            nodes = {
                **resize_info_node,
                "pixels": ReplacementNode.build(
                    "dinkster.math.expression",
                    values={"expression": "a * b / 1048576"},
                ),
            }
            inputs = {
                "image": MappingSource.copy("image"),
                "target": MappingSource.constant("total_pixels"),
                "mode": MappingSource.constant("stretch"),
                "interpolation": MappingSource.from_value("upscale_method", identity_interpolation),
                "mask": MappingSource.copy("mask"),
                "divisibility": MappingSource.constant("crop"),
                "multiple_of": MappingSource.copy("divisible_by"),
            }
            links = [*resize_info_link, ReplacementLink("pixels:float", "megapixels")]
            for local_id, source, linked, member in (
                ("width", "width", width_linked, "a"),
                ("height", "height", height_linked, "b"),
            ):
                if linked:
                    continue
                nodes[local_id] = ReplacementNode.build("dinkster.int", values={"value": 512})
                inputs[f"{local_id}:value"] = MappingSource.from_value(source)
                links.append(ReplacementLink(f"{local_id}:value", f"pixels:values.{member}"))
            cases.append(
                ReplacementCase.build(
                    "dinkster.image.resize",
                    when=ReplacementPredicate.all_of(*predicates),
                    nodes=nodes,
                    inputs=inputs,
                    input_families={
                        "pixels:values": InputFamilyMapping.from_members(
                            InputFamilyMember.build(
                                "a", inputs={"value": MappingSource.link("width")}
                            ),
                            InputFamilyMember.build(
                                "b", inputs={"value": MappingSource.link("height")}
                            ),
                        )
                    },
                    links=links,
                    outputs=resize_mask_outputs,
                )
            )
        return tuple(cases)

    def resize_predicate(
        selection: str,
        extra: ReplacementPredicate | None = None,
    ) -> ReplacementPredicate:
        selected = ReplacementPredicate.value_equals("resize_type", selection)
        return selected if extra is None else ReplacementPredicate.all_of(selected, extra)

    def resize_cases(
        selection: str,
        *,
        target: str,
        mapped: Mapping[str, MappingSource] | None = None,
        mode: str = "stretch",
        extra: ReplacementPredicate | None = None,
    ) -> tuple[ReplacementCase, ...]:
        predicate = resize_predicate(selection, extra)
        base_inputs = {
            "image": MappingSource.copy("input"),
            "target": MappingSource.constant(target),
            "mode": MappingSource.constant(mode),
            **(mapped or {}),
        }
        output = {"image": "resized"}
        linked = ReplacementPredicate.input_connected("scale_method")
        present = ReplacementPredicate.value_present("scale_method")
        return (
            ReplacementCase.build(
                "dinkster.image.resize",
                when=ReplacementPredicate.all_of(
                    predicate,
                    ReplacementPredicate.any_of(linked, present),
                ),
                inputs={
                    **base_inputs,
                    "interpolation": MappingSource.copy("scale_method"),
                },
                outputs=output,
            ),
            ReplacementCase.build(
                "dinkster.image.resize",
                when=ReplacementPredicate.all_of(
                    predicate,
                    ReplacementPredicate.not_(linked),
                    ReplacementPredicate.not_(present),
                ),
                inputs={
                    **base_inputs,
                    "interpolation": MappingSource.constant("area"),
                },
                outputs=output,
            ),
        )

    crop_unlinked = ReplacementPredicate.not_(
        ReplacementPredicate.input_connected("resize_type.crop")
    )
    crop_missing = ReplacementPredicate.all_of(
        crop_unlinked,
        ReplacementPredicate.not_(ReplacementPredicate.value_present("resize_type.crop")),
    )
    crop_center = ReplacementPredicate.all_of(
        crop_unlinked,
        ReplacementPredicate.value_equals("resize_type.crop", "center"),
    )
    crop_disabled = ReplacementPredicate.all_of(
        crop_unlinked,
        ReplacementPredicate.value_equals("resize_type.crop", "disabled"),
    )

    resize_image_mask_cases = (
        *resize_cases(
            "scale dimensions",
            target="dimensions",
            mapped={
                "width": MappingSource.copy("resize_type.width"),
                "height": MappingSource.copy("resize_type.height"),
            },
            mode="stretch",
            extra=crop_disabled,
        ),
        *resize_cases(
            "scale dimensions",
            target="dimensions",
            mapped={
                "width": MappingSource.copy("resize_type.width"),
                "height": MappingSource.copy("resize_type.height"),
            },
            mode="fill",
            extra=ReplacementPredicate.any_of(crop_center, crop_missing),
        ),
        *resize_cases(
            "scale by multiplier",
            target="factor",
            mapped={"factor": MappingSource.copy("resize_type.multiplier")},
        ),
        *resize_cases(
            "scale longer dimension",
            target="longest",
            mapped={"size": MappingSource.copy("resize_type.longer_size")},
        ),
        *resize_cases(
            "scale shorter dimension",
            target="shortest",
            mapped={"size": MappingSource.copy("resize_type.shorter_size")},
        ),
        *resize_cases(
            "scale width",
            target="width",
            mapped={"width": MappingSource.copy("resize_type.width")},
        ),
        *resize_cases(
            "scale height",
            target="height",
            mapped={"height": MappingSource.copy("resize_type.height")},
        ),
        *resize_cases(
            "scale total pixels",
            target="total_pixels",
            mapped={
                "megapixels": MappingSource.copy("resize_type.megapixels"),
                "resolution_steps": MappingSource.constant(1),
            },
        ),
        *resize_cases(
            "match size",
            target="match",
            mapped={"reference": MappingSource.copy("resize_type.match")},
            mode="stretch",
            extra=crop_disabled,
        ),
        *resize_cases(
            "match size",
            target="match",
            mapped={"reference": MappingSource.copy("resize_type.match")},
            mode="fill",
            extra=ReplacementPredicate.any_of(crop_center, crop_missing),
        ),
        *resize_cases(
            "scale to multiple",
            target="multiple_cover",
            mapped={"multiple_of": MappingSource.copy("resize_type.multiple")},
        ),
        ReplacementCase.build(
            "dinkster.image.resize",
            inputs={
                "image": MappingSource.copy("input"),
                "target": MappingSource.constant("factor"),
                "mode": MappingSource.constant("stretch"),
                "factor": MappingSource.from_value(
                    "resize_type",
                    ValueTransform.enum_rename({}),
                ),
            },
            outputs={"image": "resized"},
        ),
    )

    geometry_records = [
        _record(
            source_pack="comfy-core",
            node_class="ResizeImageMaskNode",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ResizeImageMaskNode",
                note=(
                    "Explicit migration covers all nine dynamic resize selections. Linked crop "
                    "controls and unknown selections fail closed rather than changing behavior."
                ),
                cases=resize_image_mask_cases,
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_comfy_aliases.py::test_resize_image_mask_alias_covers_every_dynamic_selection",
                "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageResizeKJ",
            revision=KJ_BASELINE,
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageResizeKJ",
                note=(
                    "Imported center-fit uses centered fill, and divisibility is finalized by "
                    "cropping after resize. The numeric crop value 0 is accepted as center."
                ),
                cases=(
                    kj_v1_resize_case(
                        "fill",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.input_connected("get_image_size"),
                            ReplacementPredicate.value_equals("crop", "center"),
                        ),
                        reference=True,
                    ),
                    kj_v1_resize_case(
                        "fill",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.input_connected("get_image_size"),
                            ReplacementPredicate.value_equals("crop", 0),
                        ),
                        reference=True,
                    ),
                    kj_v1_resize_case(
                        "stretch",
                        when=ReplacementPredicate.input_connected("get_image_size"),
                        reference=True,
                    ),
                    kj_v1_resize_case(
                        "fill",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("keep_proportion", True),
                            ReplacementPredicate.any_of(
                                ReplacementPredicate.value_equals("crop", "center"),
                                ReplacementPredicate.value_equals("crop", 0),
                            ),
                        ),
                    ),
                    kj_v1_resize_case(
                        "fit",
                        when=ReplacementPredicate.value_equals("keep_proportion", True),
                    ),
                    kj_v1_resize_case(
                        "fill",
                        when=ReplacementPredicate.value_equals("crop", "center"),
                    ),
                    kj_v1_resize_case(
                        "fill",
                        when=ReplacementPredicate.value_equals("crop", 0),
                    ),
                    kj_v1_resize_case("stretch"),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_nodes.py::test_resize_intents_compose_fit_fill_divisibility_and_mask_geometry",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageResizeKJv2",
            revision=KJ_BASELINE,
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageResizeKJv2",
                note=(
                    "Imported divisibility is finalized by cropping after resize. NVIDIA RTX VSR "
                    "fails its closed interpolation transform; device controls source placement "
                    "only. No-mask unpadded output is absent instead of a placeholder."
                ),
                cases=(
                    *(kj_v2_resize_case(mode) for mode in kj_v2_mode_choices if mode != "stretch"),
                    *kj_v2_total_pixels_cases(),
                    kj_v2_resize_case("stretch", fallback=True),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_kj_resize_v2.py::test_kj_resize_v2_modes_map_to_natural_resize_intents",
                "tests/test_image_nodes.py::test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageResize+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageResize+",
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.image.resize",
                            nodes=resize_info_node,
                            inputs={
                                "image": MappingSource.copy("image"),
                                "target": MappingSource.constant("dimensions"),
                                "width": MappingSource.copy("width"),
                                "height": MappingSource.copy("height"),
                                "mode": MappingSource.from_value(
                                    "method",
                                    ValueTransform.enum_rename(
                                        {
                                            "stretch": "stretch",
                                            "keep proportion": "fit",
                                            "fill / crop": "fill",
                                            "pad": "pad",
                                        }
                                    ),
                                ),
                                "interpolation": MappingSource.copy("interpolation"),
                                "apply": MappingSource.from_value(
                                    "condition",
                                    ValueTransform.enum_rename(
                                        {
                                            "always": "always",
                                            "downscale if bigger": "only_if_bigger",
                                            "upscale if smaller": "only_if_smaller",
                                            "if bigger area": "only_if_bigger_area",
                                            "if smaller area": "only_if_smaller_area",
                                        }
                                    ),
                                ),
                                "divisibility": MappingSource.constant("crop"),
                                "multiple_of": MappingSource.copy("multiple_of"),
                            },
                            links=resize_info_link,
                            outputs=resize_outputs,
                        ),
                        target_input="mode",
                        source_input="method",
                        choices={
                            "stretch": "stretch",
                            "keep proportion": "fit",
                            "fill / crop": "fill",
                            "pad": "pad",
                        },
                        fallback="stretch",
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_nodes.py::test_resize_apply_conditions_padding_and_divisibility_are_independent_intents",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="GetImageSizeAndCount",
            revision=KJ_BASELINE,
            carrier="dinkster.image.crop",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.GetImageSizeAndCount",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.crop",
                        nodes=resize_info_node,
                        inputs={"image": MappingSource.copy("image")},
                        links=resize_info_link,
                        outputs={
                            "image": "image",
                            "info:width": "width",
                            "info:height": "height",
                            "info:count": "count",
                        },
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="GetImageSize+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.info",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.GetImageSize+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.info",
                        inputs={"image": MappingSource.copy("image")},
                        outputs={"width": "width", "height": "height", "count": "count"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageCrop+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.region.info",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageCrop+",
                note=(
                    "Disjoint offsets and zero-sized requests fail instead of returning an empty "
                    "image. Other placements preserve the source's clipping and offsets."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.region.info",
                        nodes={"crop": ReplacementNode.build("dinkster.image.crop")},
                        inputs={
                            "crop:image": MappingSource.copy("image"),
                            "crop:source": MappingSource.constant("coordinates"),
                            "crop:width": MappingSource.copy("width"),
                            "crop:height": MappingSource.copy("height"),
                            "crop:placement": MappingSource.from_value(
                                "position",
                                ValueTransform.enum_rename(
                                    {
                                        value: value.replace("-", "_")
                                        for value in (
                                            "top-left",
                                            "top-center",
                                            "top-right",
                                            "right-center",
                                            "bottom-right",
                                            "bottom-center",
                                            "bottom-left",
                                            "left-center",
                                            "center",
                                        )
                                    }
                                ),
                            ),
                            "crop:x": MappingSource.copy("x_offset"),
                            "crop:y": MappingSource.copy("y_offset"),
                            "crop:rounding": MappingSource.constant("floor"),
                            "crop:outside": MappingSource.constant("clip"),
                        },
                        links=(ReplacementLink("crop:region", "region"),),
                        outputs={
                            "crop:image": "IMAGE",
                            "integer_x": "x",
                            "integer_y": "y",
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_nodes.py::test_crop_placement_applies_offsets_and_reports_the_clipped_region",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ImageCropByMask",
            revision=KJ_BASELINE,
            carrier="dinkster.image.crop",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ImageCropByMask",
                note=(
                    "The source derives one region per frame; native derives the union region and "
                    "therefore preserves batching when frame bounds differ."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.crop",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "source": MappingSource.constant("mask"),
                            "mask": MappingSource.copy("mask"),
                            "mask_threshold": MappingSource.constant(0.5),
                            "rounding": MappingSource.constant("floor"),
                            "outside": MappingSource.constant("clip"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_nodes.py::test_crop_derives_union_region_from_mask_and_returns_synchronized_mask"
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="ImageRemoveAlpha+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.channels.split",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.ImageRemoveAlpha+",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.channels.split",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "single_channel_image": MappingSource.constant("preserve"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_channel_split_has_rgb_planes_and_explicit_alpha_polarity",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="MaskBoundingBox+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.crop",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.MaskBoundingBox+",
                note=(
                    "The supplied image must already match the mask dimensions. Its batch must "
                    "equal the mask batch, or the mask must be a singleton."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.crop",
                        when=ReplacementPredicate.input_connected("image_optional"),
                        nodes={"region": ReplacementNode.build("dinkster.region.info")},
                        inputs={
                            "image": MappingSource.copy("image_optional"),
                            "source": MappingSource.constant("mask"),
                            "mask": MappingSource.copy("mask"),
                            "padding": MappingSource.copy("padding"),
                            "mask_blur": MappingSource.copy("blur"),
                            "rounding": MappingSource.constant("floor"),
                            "outside": MappingSource.constant("clip"),
                        },
                        links=(ReplacementLink("region", "region:region"),),
                        outputs={
                            "mask": "MASK",
                            "image": "IMAGE",
                            "region:integer_x": "x",
                            "region:integer_y": "y",
                            "region:integer_width": "width",
                            "region:integer_height": "height",
                        },
                    ),
                    ReplacementCase.build(
                        "dinkster.mask.info",
                        nodes={
                            "image": ReplacementNode.build(
                                "dinkster.mask.to_image", values={"channels": "rgb"}
                            ),
                            "crop": ReplacementNode.build(
                                "dinkster.image.crop",
                                values={
                                    "source": "mask",
                                    "rounding": "floor",
                                    "outside": "clip",
                                },
                            ),
                            "region": ReplacementNode.build("dinkster.region.info"),
                        },
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "crop:padding": MappingSource.copy("padding"),
                            "crop:mask_blur": MappingSource.copy("blur"),
                        },
                        links=(
                            ReplacementLink("mask", "image:mask"),
                            ReplacementLink("mask", "crop:mask"),
                            ReplacementLink("image:image", "crop:image"),
                            ReplacementLink("crop:region", "region:region"),
                        ),
                        outputs={
                            "crop:mask": "MASK",
                            "crop:image": "IMAGE",
                            "region:integer_x": "x",
                            "region:integer_y": "y",
                            "region:integer_width": "width",
                            "region:integer_height": "height",
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_nodes.py::test_crop_derives_union_region_from_mask_and_returns_synchronized_mask",
                "tests/test_image_comfy_aliases.py::test_geometry_aliases_preserve_modes_outputs_and_refusals",
            ],
        ),
    ]
    records = [
        *batch_layout_records,
        *geometry_records,
        _record(
            source_pack="comfy-core",
            node_class="PrimitiveBoundingBox",
            revision="b78cec87",
            carrier="dinkster.region.make",
            rule=ReplacementRule(
                from_type="comfy.PrimitiveBoundingBox",
                cases=(
                    ReplacementCase.build(
                        "dinkster.region.make",
                        inputs={
                            name: MappingSource.copy(name) for name in ("x", "y", "width", "height")
                        },
                        outputs={"region": "_0_BOUNDING_BOX_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-mtb",
            node_class="BBox (mtb)",
            revision=MTB_BASELINE,
            carrier="dinkster.region.make",
            rule=ReplacementRule(
                from_type="comfy.comfy-mtb.BBox (mtb)",
                cases=(
                    ReplacementCase.build(
                        "dinkster.region.make",
                        inputs={
                            name: MappingSource.copy(name) for name in ("x", "y", "width", "height")
                        },
                        outputs={"region": "bbox"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_comfy_aliases.py::test_mtb_bbox_alias_matches_pinned_static_reference"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageScale",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ImageScale",
                note=(
                    "Center crop maps to natural whole-pixel fill; odd crop margins may select "
                    "a neighboring edge pixel."
                ),
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.image.resize",
                            inputs={
                                "image": MappingSource.copy("image"),
                                "target": MappingSource.constant("dimensions"),
                                "width": MappingSource.copy("width"),
                                "height": MappingSource.copy("height"),
                                "mode": MappingSource.from_value(
                                    "crop",
                                    ValueTransform.enum_rename(
                                        {"disabled": "stretch", "center": "fill"}
                                    ),
                                ),
                                "interpolation": MappingSource.copy("upscale_method"),
                            },
                            outputs={"image": "image"},
                        ),
                        target_input="mode",
                        source_input="crop",
                        choices={"disabled": "stretch", "center": "fill"},
                        fallback="disabled",
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
                "tests/test_image_nodes.py::test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageFlip",
            revision="b78cec87",
            carrier="dinkster.image.transform",
            rule=ReplacementRule(
                from_type="comfy.ImageFlip",
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.image.transform",
                            inputs={
                                "image": MappingSource.copy("image"),
                                "operation": MappingSource.from_value(
                                    "flip_method",
                                    ValueTransform.enum_rename(
                                        {
                                            "x-axis: vertically": "flip_vertical",
                                            "y-axis: horizontally": "flip_horizontal",
                                        }
                                    ),
                                ),
                            },
                            outputs={"image": "_0_IMAGE_"},
                        ),
                        target_input="operation",
                        source_input="flip_method",
                        choices={
                            "x-axis: vertically": "flip_vertical",
                            "y-axis: horizontally": "flip_horizontal",
                        },
                        fallback="x-axis: vertically",
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageScaleBy",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ImageScaleBy",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.resize",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "target": MappingSource.constant("factor"),
                            "mode": MappingSource.constant("stretch"),
                            "factor": MappingSource.copy("scale_by"),
                            "interpolation": MappingSource.copy("upscale_method"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
                "tests/test_image_nodes.py::test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageScaleToTotalPixels",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ImageScaleToTotalPixels",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.resize",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "target": MappingSource.constant("total_pixels"),
                            "mode": MappingSource.constant("stretch"),
                            "megapixels": MappingSource.copy("megapixels"),
                            "resolution_steps": MappingSource.copy("resolution_steps"),
                            "interpolation": MappingSource.copy("upscale_method"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
                "tests/test_image_nodes.py::test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageScaleToMaxDimension",
            revision="b78cec87",
            carrier="dinkster.image.resize",
            rule=ReplacementRule(
                from_type="comfy.ImageScaleToMaxDimension",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.resize",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "target": MappingSource.constant("longest"),
                            "mode": MappingSource.constant("stretch"),
                            "size": MappingSource.copy("largest_size"),
                            "interpolation": MappingSource.copy("upscale_method"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_nodes.py::test_core_resize_mappings_match_comfy_goldens",
                "tests/test_image_nodes.py::test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-6}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageRotate",
            revision="b78cec87",
            carrier="dinkster.image.transform",
            rule=ReplacementRule(
                from_type="comfy.ImageRotate",
                cases=tuple(
                    ReplacementCase.build(
                        "dinkster.image.transform",
                        when=(
                            ReplacementPredicate.value_equals("rotation", rotation)
                            if rotation != "270 degrees"
                            else None
                        ),
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("rotate_90"),
                            "steps": MappingSource.constant(steps),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    )
                    for rotation, steps in (
                        ("none", 0),
                        ("90 degrees", 1),
                        ("180 degrees", 2),
                        ("270 degrees", 3),
                    )
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImagePadForOutpaint",
            revision="b78cec87",
            carrier="dinkster.image.transform",
            rule=ReplacementRule(
                from_type="comfy.ImagePadForOutpaint",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.transform",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("pad"),
                            **{
                                name: MappingSource.copy(name)
                                for name in ("left", "top", "right", "bottom", "feathering")
                            },
                        },
                        outputs={"image": "image", "mask": "mask"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageCrop",
            revision="b78cec87",
            carrier="dinkster.image.crop",
            rule=ReplacementRule(
                from_type="comfy.ImageCrop",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.crop",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "source": MappingSource.constant("coordinates"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "x": MappingSource.copy("x"),
                            "y": MappingSource.copy("y"),
                            "rounding": MappingSource.constant("floor"),
                            "outside": MappingSource.constant("comfy"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens",
                "tests/test_image_nodes.py::test_crop_comfy_policy_clamps_the_origin_before_slicing",
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageCropV2",
            revision="b78cec87",
            carrier="dinkster.image.crop",
            rule=ReplacementRule(
                from_type="comfy.ImageCropV2",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.crop",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "source": MappingSource.constant("region"),
                            "region": MappingSource.copy("crop_region"),
                            "rounding": MappingSource.constant("floor"),
                            "outside": MappingSource.constant("comfy"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-mtb",
            node_class="Uncrop (mtb)",
            revision=MTB_BASELINE,
            carrier="dinkster.image.uncrop",
            rule=ReplacementRule(
                from_type="comfy.comfy-mtb.Uncrop (mtb)",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.uncrop",
                        inputs={
                            "base": MappingSource.copy("image"),
                            "crop": MappingSource.copy("crop_image"),
                            "region": MappingSource.copy("bbox"),
                            "border_blending": MappingSource.copy("border_blending"),
                            "rounding": MappingSource.constant("floor"),
                            "interpolation": MappingSource.constant("bicubic"),
                            "outside": MappingSource.constant("ignore"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_nodes.py::test_uncrop_matches_mtb_border_blending_reference"
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 3e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="GetImageSize",
            revision="b78cec87",
            carrier="dinkster.image.info",
            rule=ReplacementRule(
                from_type="comfy.GetImageSize",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.info",
                        inputs={"image": MappingSource.copy("image")},
                        outputs={
                            "width": "_0_INT_",
                            "height": "_1_INT_",
                            "count": "_2_INT_",
                        },
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_nodes.py::test_core_region_crop_transform_pad_and_info_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageCompositeMasked",
            revision="b78cec87",
            carrier="dinkster.image.composite",
            rule=ReplacementRule(
                from_type="comfy.ImageCompositeMasked",
                cases=tuple(
                    ReplacementCase.build(
                        "dinkster.image.composite",
                        when=(
                            ReplacementPredicate.value_equals("resize_source", False)
                            if source_resize == "none"
                            else None
                        ),
                        inputs={
                            "destination": MappingSource.copy("destination"),
                            "source": MappingSource.copy("source"),
                            "x": MappingSource.copy("x"),
                            "y": MappingSource.copy("y"),
                            "mask": MappingSource.copy("mask"),
                            "blend_mode": MappingSource.constant("normal"),
                            "factor": MappingSource.constant(1.0),
                            "source_resize": MappingSource.constant(source_resize),
                            **(
                                {"interpolation": MappingSource.constant("bilinear")}
                                if source_resize != "none"
                                else {}
                            ),
                            "mask_polarity": MappingSource.constant("coverage"),
                            "clamp_output": MappingSource.constant(False),
                            "batch_policy": MappingSource.constant("destination_repeat"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    )
                    for source_resize in ("none", "stretch")
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_composition.py::test_core_composite_mappings_match_comfy_goldens"
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 5e-8}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageBlend",
            revision="b78cec87",
            carrier="dinkster.image.composite",
            rule=ReplacementRule(
                from_type="comfy.ImageBlend",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.composite",
                        inputs={
                            "destination": MappingSource.copy("image1"),
                            "source": MappingSource.copy("image2"),
                            "x": MappingSource.constant(0),
                            "y": MappingSource.constant(0),
                            "blend_mode": MappingSource.from_value(
                                "blend_mode",
                                ValueTransform.enum_rename(
                                    {
                                        "normal": "normal",
                                        "multiply": "multiply",
                                        "screen": "screen",
                                        "overlay": "overlay",
                                        "soft_light": "soft_light",
                                        "difference": "signed_difference",
                                    }
                                ),
                            ),
                            "factor": MappingSource.copy("blend_factor"),
                            "source_resize": MappingSource.constant("fill"),
                            "interpolation": MappingSource.constant("bicubic"),
                            "clamp_output": MappingSource.constant(True),
                            "batch_policy": MappingSource.constant("singleton_broadcast"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_composition.py::test_core_composite_mappings_match_comfy_goldens"
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 2e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="PorterDuffImageComposite",
            revision="b78cec87",
            carrier="dinkster.image.porter_duff",
            rule=ReplacementRule(
                from_type="comfy.PorterDuffImageComposite",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.porter_duff",
                        inputs={
                            "source": MappingSource.copy("source"),
                            "source_alpha_mask": MappingSource.copy("source_alpha"),
                            "destination": MappingSource.copy("destination"),
                            "destination_alpha_mask": MappingSource.copy("destination_alpha"),
                            "mode": MappingSource.copy("mode"),
                            "mask_polarity": MappingSource.constant("transparency"),
                            "batch_policy": MappingSource.constant("truncate_to_shortest"),
                        },
                        outputs={"image": "_0_IMAGE_", "alpha_mask": "_1_MASK_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_composition.py::test_core_porter_duff_mappings_match_comfy_goldens",
                "tests/test_image_composition.py::test_core_porter_duff_center_crops_mismatched_aspects",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 3e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageInvert",
            revision="b78cec87",
            carrier="dinkster.image.adjust",
            rule=ReplacementRule(
                from_type="comfy.ImageInvert",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.adjust",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("invert"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="NormalizeImages",
            revision="b78cec87",
            carrier="dinkster.image.adjust",
            rule=ReplacementRule(
                from_type="comfy.NormalizeImages",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.adjust",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("normalize"),
                            "mean": MappingSource.copy("mean"),
                            "standard_deviation": MappingSource.copy("std"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="AdjustBrightness",
            revision="b78cec87",
            carrier="dinkster.image.adjust",
            rule=ReplacementRule(
                from_type="comfy.AdjustBrightness",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.adjust",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("brightness"),
                            "factor": MappingSource.copy("factor"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="AdjustContrast",
            revision="b78cec87",
            carrier="dinkster.image.adjust",
            rule=ReplacementRule(
                from_type="comfy.AdjustContrast",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.adjust",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "operation": MappingSource.constant("contrast"),
                            "factor": MappingSource.copy("factor"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageBlur",
            revision="b78cec87",
            carrier="dinkster.image.filter",
            rule=ReplacementRule(
                from_type="comfy.ImageBlur",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.filter",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("gaussian_blur"),
                            "radius": MappingSource.copy("blur_radius"),
                            "sigma": MappingSource.copy("sigma"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens",
                "tests/test_image_adjust_filter_channels.py::test_core_filters_match_large_kernel_comfy_goldens",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 8e-5}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageSharpen",
            revision="b78cec87",
            carrier="dinkster.image.filter",
            rule=ReplacementRule(
                from_type="comfy.ImageSharpen",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.filter",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("sharpen"),
                            "radius": MappingSource.copy("sharpen_radius"),
                            "sigma": MappingSource.copy("sigma"),
                            "strength": MappingSource.copy("alpha"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens",
                "tests/test_image_adjust_filter_channels.py::test_core_filters_match_large_kernel_comfy_goldens",
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 0.022}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageQuantize",
            revision="b78cec87",
            carrier="dinkster.image.filter",
            rule=ReplacementRule(
                from_type="comfy.ImageQuantize",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.filter",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "operation": MappingSource.constant("quantize"),
                            "colors": MappingSource.copy("colors"),
                            "dither": MappingSource.copy("dither"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_adjust_filter_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="Morphology",
            revision="b78cec87",
            carrier="dinkster.image.filter",
            rule=ReplacementRule(
                from_type="comfy.Morphology",
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.image.filter",
                            inputs={
                                "image": MappingSource.copy("image"),
                                "operation": MappingSource.copy("operation"),
                                "kernel_size": MappingSource.copy("kernel_size"),
                            },
                            outputs={"image": "_0_IMAGE_"},
                        ),
                        target_input="operation",
                        source_input="operation",
                        choices={
                            operation: operation
                            for operation in (
                                "erode",
                                "dilate",
                                "open",
                                "close",
                                "gradient",
                                "bottom_hat",
                                "top_hat",
                            )
                        },
                        fallback="erode",
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_morphology_mapping_matches_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="SplitImageWithAlpha",
            revision="b78cec87",
            carrier="dinkster.image.channels.split",
            rule=ReplacementRule(
                from_type="comfy.SplitImageWithAlpha",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.channels.split",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "color_space": MappingSource.constant("rgb"),
                            "channel_layout": MappingSource.constant("single"),
                            "mask_polarity": MappingSource.constant("transparency"),
                        },
                        outputs={"image": "_0_IMAGE_", "alpha_mask": "_1_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_channel_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageRGBToYUV",
            revision="b78cec87",
            carrier="dinkster.image.channels.split",
            rule=ReplacementRule(
                from_type="comfy.ImageRGBToYUV",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.channels.split",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "color_space": MappingSource.constant("ycbcr"),
                            "channel_layout": MappingSource.constant("rgb_repeated"),
                            "mask_polarity": MappingSource.constant("transparency"),
                        },
                        outputs={
                            "channel_1": "_0_IMAGE_",
                            "channel_2": "_1_IMAGE_",
                            "channel_3": "_2_IMAGE_",
                        },
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_channel_mappings_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="JoinImageWithAlpha",
            revision="b78cec87",
            carrier="dinkster.image.alpha.join",
            rule=ReplacementRule(
                from_type="comfy.JoinImageWithAlpha",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.alpha.join",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "alpha_mask": MappingSource.copy("alpha"),
                            "mask_polarity": MappingSource.constant("transparency"),
                            "batch_policy": MappingSource.constant("cyclic_repeat"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_channel_mappings_match_comfy_goldens"
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-7}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageYUVToRGB",
            revision="b78cec87",
            carrier="dinkster.image.channels.merge",
            rule=ReplacementRule(
                from_type="comfy.ImageYUVToRGB",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.channels.merge",
                        inputs={
                            "channel_1": MappingSource.copy("Y"),
                            "channel_2": MappingSource.copy("U"),
                            "channel_3": MappingSource.copy("V"),
                            "color_space": MappingSource.constant("ycbcr"),
                            "mask_polarity": MappingSource.constant("transparency"),
                            "batch_policy": MappingSource.constant("strict"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_image_adjust_filter_channels.py::test_core_channel_mappings_match_comfy_goldens"
            ],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-5}],
        ),
        _record(
            source_pack="comfy-core",
            node_class="SolidMask",
            revision="b78cec87",
            carrier="dinkster.mask.make",
            rule=ReplacementRule(
                from_type="comfy.SolidMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.make",
                        inputs={
                            "operation": MappingSource.constant("solid"),
                            "foreground": MappingSource.copy("value"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "batch_size": MappingSource.constant(1),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="InvertMask",
            revision="b78cec87",
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.InvertMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("invert"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="CropMask",
            revision="b78cec87",
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.CropMask",
                note=(
                    "Crops with no source overlap are rejected instead of returning an empty mask."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("crop"),
                            "x": MappingSource.copy("x"),
                            "y": MappingSource.copy("y"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="FeatherMask",
            revision="b78cec87",
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.FeatherMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("feather_edges"),
                            **{
                                name: MappingSource.copy(name)
                                for name in ("left", "top", "right", "bottom")
                            },
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="GrowMask",
            revision="b78cec87",
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.GrowMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("grow_erode"),
                            "radius": MappingSource.copy("expand"),
                            "tapered_corners": MappingSource.copy("tapered_corners"),
                            "edge_policy": MappingSource.constant("reflect"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ThresholdMask",
            revision="b78cec87",
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.ThresholdMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("threshold"),
                            "threshold": MappingSource.copy("value"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="MaskComposite",
            revision="b78cec87",
            carrier="dinkster.mask.combine",
            rule=ReplacementRule(
                from_type="comfy.MaskComposite",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.combine",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("destination", "source", "operation", "x", "y")
                        }
                        | {"batch_policy": MappingSource.constant("source_singleton")},
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens",
                "tests/test_mask_operations.py::test_combine_batch_policies_control_singleton_direction",
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageToMask",
            revision="b78cec87",
            carrier="dinkster.image.to_mask",
            rule=ReplacementRule(
                from_type="comfy.ImageToMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.to_mask",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "policy": MappingSource.constant("channel"),
                            "channel": MappingSource.copy("channel"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="ImageColorToMask",
            revision="b78cec87",
            carrier="dinkster.image.to_mask",
            rule=ReplacementRule(
                from_type="comfy.ImageColorToMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.to_mask",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "policy": MappingSource.constant("exact_color"),
                            "color_source": MappingSource.constant("integer"),
                            "color_value": MappingSource.copy("color"),
                        },
                        outputs={"mask": "_0_MASK_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="MaskToImage",
            revision="b78cec87",
            carrier="dinkster.mask.to_image",
            rule=ReplacementRule(
                from_type="comfy.MaskToImage",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.to_image",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "channels": MappingSource.constant("rgb"),
                        },
                        outputs={"image": "_0_IMAGE_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfy-core",
            node_class="EmptyImage",
            revision="b78cec87",
            carrier="dinkster.image.generate",
            rule=ReplacementRule(
                from_type="comfy.EmptyImage",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.generate",
                        inputs={
                            "operation": MappingSource.constant("solid"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "batch_size": MappingSource.copy("batch_size"),
                            "color_source": MappingSource.constant("integer"),
                            "color_value": MappingSource.copy("color"),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_core_mask_and_image_generation_match_comfy_goldens"
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="CreateShapeMask",
            revision=KJ_BASELINE,
            carrier="dinkster.mask.make",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.CreateShapeMask",
                note=(
                    "Shape size uses pixel-center geometry; even dimensions may differ by one edge "
                    "pixel from Pillow's inclusive bounds."
                ),
                cases=(
                    *_fanout_target_choice(
                        ReplacementCase.build(
                            "dinkster.mask.make",
                            inputs={
                                "operation": MappingSource.from_value(
                                    "shape",
                                    ValueTransform.enum_rename(
                                        {
                                            "circle": "ellipse",
                                            "square": "rectangle",
                                            "triangle": "triangle",
                                        }
                                    ),
                                ),
                                "width": MappingSource.copy("frame_width"),
                                "height": MappingSource.copy("frame_height"),
                                "batch_size": MappingSource.copy("frames"),
                                "x": MappingSource.copy("location_x"),
                                "y": MappingSource.copy("location_y"),
                                "shape_origin": MappingSource.constant("center"),
                                "shape_width": MappingSource.copy("shape_width"),
                                "shape_height": MappingSource.copy("shape_height"),
                                "grow": MappingSource.copy("grow"),
                            },
                            outputs={"mask": "mask", "inverse_mask": "mask_inverted"},
                        ),
                        target_input="operation",
                        source_input="shape",
                        choices={
                            "circle": "ellipse",
                            "square": "rectangle",
                            "triangle": "triangle",
                        },
                        fallback="circle",
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_mask_operations.py::test_make_mask_extended_generators"],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="TransitionMask+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.mask.make",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.TransitionMask+",
                note=(
                    "Single-frame and reversed frame intervals use native defined endpoint "
                    "behavior."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.make",
                        inputs={
                            "operation": MappingSource.constant("transition"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "batch_size": MappingSource.copy("frames"),
                            "start_frame": MappingSource.copy("start_frame"),
                            "end_frame": MappingSource.copy("end_frame"),
                            "transition_type": MappingSource.from_value(
                                "transition_type",
                                ValueTransform.enum_rename(
                                    {
                                        "horizontal slide": "horizontal_slide",
                                        "vertical slide": "vertical_slide",
                                        "horizontal bar": "horizontal_bar",
                                        "vertical bar": "vertical_bar",
                                        "center box": "center_box",
                                        "horizontal door": "horizontal_door",
                                        "vertical door": "vertical_door",
                                        "circle": "circle",
                                        "fade": "fade",
                                    }
                                ),
                            ),
                            "timing_function": MappingSource.from_value(
                                "timing_function",
                                ValueTransform.enum_rename(
                                    {
                                        "linear": "linear",
                                        "in": "in",
                                        "out": "out",
                                        "in-out": "in_out",
                                    }
                                ),
                            ),
                        },
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_mask_operations.py::test_make_mask_extended_generators"],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="CreateTextMask",
            revision=KJ_BASELINE,
            carrier="dinkster.mask.text",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.CreateTextMask",
                note=(
                    "The custom font asset and word wrapping are replaced by Pillow's bundled "
                    "font; text, placement, color, frame count, and rotation range are preserved."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.text",
                        inputs={
                            "text": MappingSource.copy("text"),
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                            "batch_size": MappingSource.copy("frames"),
                            "x": MappingSource.copy("text_x"),
                            "y": MappingSource.copy("text_y"),
                            "font_size": MappingSource.copy("font_size"),
                            "color": MappingSource.copy("font_color"),
                            "mask_value": MappingSource.constant("color_red"),
                            "line_spacing": MappingSource.constant(0),
                            "start_rotation": MappingSource.copy("start_rotation"),
                            "end_rotation": MappingSource.copy("end_rotation"),
                            "invert": MappingSource.copy("invert"),
                        },
                        outputs={"image": "image", "mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_mask_operations.py::test_text_mask_color_rotation_and_outputs"],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="GrowMaskWithBlur",
            revision=KJ_BASELINE,
            carrier="dinkster.mask.morphology",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.GrowMaskWithBlur",
                note=(
                    "Morphology and temporal controls are preserved; Gaussian blur remains float32 "
                    "instead of the source's 8-bit Pillow round trip."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.morphology",
                        inputs={
                            "mask": MappingSource.copy("mask"),
                            "operation": MappingSource.constant("grow_blur"),
                            "radius": MappingSource.copy("expand"),
                            "incremental_expandrate": MappingSource.copy("incremental_expandrate"),
                            "tapered_corners": MappingSource.copy("tapered_corners"),
                            "flip_input": MappingSource.copy("flip_input"),
                            "blur_amount": MappingSource.copy("blur_radius"),
                            "lerp_alpha": MappingSource.copy("lerp_alpha"),
                            "decay_factor": MappingSource.copy("decay_factor"),
                            "fill_holes": MappingSource.copy("fill_holes"),
                            "edge_policy": MappingSource.constant("replicate"),
                        },
                        outputs={"mask": "mask", "inverse_mask": "mask_inverted"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_mask_operations.py::test_grow_blur_temporal_controls"],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="ColorToMask",
            revision=KJ_BASELINE,
            carrier="dinkster.image.to_mask",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.ColorToMask",
                note="per_batch controls source chunking only and is removed.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.to_mask",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "policy": MappingSource.constant("tolerance_color"),
                            "color_source": MappingSource.constant("channels"),
                            "red": MappingSource.copy("red"),
                            "green": MappingSource.copy("green"),
                            "blue": MappingSource.copy("blue"),
                            "tolerance": MappingSource.from_value(
                                "threshold", ValueTransform.scale(1.0 / 255.0)
                            ),
                            "metric": MappingSource.constant("euclidean_rgb_sum"),
                            "invert": MappingSource.copy("invert"),
                        },
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=["tests/test_mask_operations.py::test_image_to_mask_color_sources_and_invert"],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-6}],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="MaskFromColor+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.to_mask",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.MaskFromColor+",
                note=(
                    "The source rounds pixels to 8-bit before applying its per-channel threshold; "
                    "native tolerance compares float pixels directly."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.to_mask",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "policy": MappingSource.constant("tolerance_color"),
                            "color_source": MappingSource.constant("channels"),
                            "red": MappingSource.copy("red"),
                            "green": MappingSource.copy("green"),
                            "blue": MappingSource.copy("blue"),
                            "tolerance": MappingSource.from_value(
                                "threshold", ValueTransform.scale(1.0 / 255.0)
                            ),
                            "metric": MappingSource.constant("max_channel"),
                        },
                        outputs={"mask": "mask"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_mask_operations.py::test_image_to_mask_color_sources_and_invert"],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="GetMaskSizeAndCount",
            revision=KJ_BASELINE,
            carrier="dinkster.mask.info",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.GetMaskSizeAndCount",
                cases=(
                    ReplacementCase.build(
                        "dinkster.mask.info",
                        inputs={"mask": MappingSource.copy("mask")},
                        outputs={
                            "mask": "mask",
                            "width": "width",
                            "height": "height",
                            "count": "count",
                        },
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_mask_operations.py::test_mask_info_reports_stable_union_bounds_and_empty_bounds"
            ],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="DrawMaskOnImage",
            revision=KJ_BASELINE,
            carrier="dinkster.image.draw_mask",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.DrawMaskOnImage",
                note="device controls source placement only and is removed.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.draw_mask",
                        inputs={
                            "image": MappingSource.copy("image"),
                            "mask": MappingSource.copy("mask"),
                            "color": MappingSource.copy("color"),
                            "opacity": MappingSource.constant(1.0),
                            "mask_size": MappingSource.constant("resize_to_image"),
                            "batch_policy": MappingSource.constant("cyclic_repeat"),
                            "alpha_mode": MappingSource.constant("max"),
                        },
                        outputs={"image": "images"},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=["tests/test_image_drawing.py::test_draw_mask_kj_compatibility_controls"],
            tolerances=[{"metric": "max_abs", "operator": "<=", "value": 1e-7}],
        ),
        _record(
            source_pack="comfyui-kjnodes",
            node_class="BboxVisualize",
            revision=KJ_BASELINE,
            carrier="dinkster.image.draw_region",
            rule=ReplacementRule(
                from_type="comfy.comfyui-kjnodes.BboxVisualize",
                note=(
                    "Legacy BBOX tuples and BOUNDING_BOX records are normalized in source order. "
                    "The shorter image or bounding-box batch determines the output batch."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.draw_region",
                        inputs={
                            "image": MappingSource.copy("images"),
                            "region": MappingSource.copy("bboxes"),
                            "shape": MappingSource.constant("rectangle"),
                            "mode": MappingSource.constant("outline"),
                            "line_width": MappingSource.copy("line_width"),
                            "color": MappingSource.constant("#ff0000"),
                            "opacity": MappingSource.constant(1.0),
                            "coordinate_format": MappingSource.copy("bbox_format"),
                            "batch_policy": MappingSource.constant("pairwise_truncate"),
                        },
                        outputs={"image": "images"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=["tests/test_image_drawing.py::test_draw_region_fills_outlines_and_clips"],
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="DrawText+",
            revision=ESSENTIALS_BASELINE,
            carrier="dinkster.image.draw_text",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.DrawText+",
                note=(
                    "Requires img_composite. Font assets, alignment, direction, background, "
                    "shadow, and the separate mask output are not preserved."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.image.draw_text",
                        inputs={
                            "image": MappingSource.copy("img_composite"),
                            "text": MappingSource.copy("text"),
                            "x": MappingSource.copy("offset_x"),
                            "y": MappingSource.copy("offset_y"),
                            "font_size": MappingSource.copy("size"),
                            "color": MappingSource.copy("color"),
                            "opacity": MappingSource.constant(1.0),
                        },
                        outputs={"image": "image"},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_image_drawing.py::test_draw_text_uses_pillow_bundled_default_font_deterministically"
            ],
        ),
        *preprocessor_records,
    ]
    layer_schemas, layer_records = layer_alias_data()
    source_schemas.extend(layer_schemas)
    records.extend(layer_records)
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in source_schemas],
        "records": records,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_registry(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")
    depth_content = (json.dumps(build_depth_anything_v2_registry(), indent=2) + "\n").encode()
    DEPTH_ANYTHING_V2_OUT.write_bytes(depth_content)
    print(f"{DEPTH_ANYTHING_V2_OUT}: sha256={hashlib.sha256(depth_content).hexdigest()}")


if __name__ == "__main__":
    main()
