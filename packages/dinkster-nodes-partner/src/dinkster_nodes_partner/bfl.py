"""Black Forest Labs partner nodes pinned to ComfyUI e651b7be."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    BooleanWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .opspec import (
    Adapter,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    HttpSyncJson,
    InputBinding,
    MaskPrepare,
    MediaConstraints,
    OpSpec,
    SubmitPoll,
)
from .partner_runtime import run_op, worker_runtime_context

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
IMAGE = TypeExpr.concrete("comfy.IMAGE")
MASK = TypeExpr.concrete("comfy.MASK")
FAILED = ("Request Moderated", "Content Moderated", "Error", "Task not found")
MAX_IMAGE_PIXELS = 2048 * 2048
BFL_STATUSES = ("Task not found", "Pending", *FAILED[:3], "Ready")


def _required_text(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


def _aspect_ratio(value: str) -> None:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("aspect_ratio must use width:height format")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("aspect_ratio must use integer width:height values") from exc
    if width <= 0 or height <= 0 or not 0.25 <= width / height <= 4:
        raise ValueError("aspect_ratio must be between 1:4 and 4:1")


@dataclass(frozen=True)
class BFLGenerateResponse:
    id: str
    polling_url: str
    cost: float | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "id")
        _required_text(self.polling_url, "polling_url")


@dataclass(frozen=True)
class BFLStatusResponse:
    id: str
    status: str
    result: Mapping[str, object] | None = None
    progress: float | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "id")
        _required_text(self.status, "status")
        if self.status not in BFL_STATUSES:
            raise ValueError(f"unknown BFL status {self.status!r}")
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("progress must be between zero and one")


@dataclass(frozen=True, kw_only=True)
class FluxUltraRequest:
    prompt: str
    prompt_upsampling: bool | None = None
    seed: int | None = None
    aspect_ratio: str | None = None
    safety_tolerance: int = 6
    output_format: str = "png"
    raw: bool | None = None
    image_prompt: str | None = None
    image_prompt_strength: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt and self.image_prompt is None:
            raise ValueError("prompt must not be empty without image_prompt")
        if self.aspect_ratio is not None:
            _aspect_ratio(self.aspect_ratio)


@dataclass(frozen=True, kw_only=True)
class FluxKontextRequest:
    prompt: str
    guidance: float
    steps: int
    input_image: str | None = None
    seed: int | None = None
    safety_tolerance: int = 2
    output_format: str = "png"
    aspect_ratio: str | None = None
    prompt_upsampling: bool | None = None

    def __post_init__(self) -> None:
        if not self.prompt and self.input_image is None:
            raise ValueError("prompt must not be empty without input_image")
        if self.aspect_ratio is not None:
            _aspect_ratio(self.aspect_ratio)


@dataclass(frozen=True, kw_only=True)
class FluxExpandRequest:
    prompt: str
    top: int
    bottom: int
    left: int
    right: int
    steps: int
    guidance: float
    prompt_upsampling: bool | None = None
    seed: int | None = None
    safety_tolerance: int = 6
    output_format: str = "png"
    image: str | None = None

    def __post_init__(self) -> None:
        if min(self.top, self.bottom, self.left, self.right) < 0:
            raise ValueError("expand dimensions must be non-negative")


@dataclass(frozen=True, kw_only=True)
class FluxFillRequest:
    prompt: str
    steps: int
    guidance: float
    prompt_upsampling: bool | None = None
    seed: int | None = None
    safety_tolerance: int = 6
    output_format: str = "png"
    image: str | None = None
    mask: str | None = None


@dataclass(frozen=True, kw_only=True)
class FluxEraseRequest:
    image: str
    mask: str
    dilate_pixels: int = 10
    seed: int | None = None
    output_format: str = "png"

    def __post_init__(self) -> None:
        _required_text(self.image, "image")
        _required_text(self.mask, "mask")


@dataclass(frozen=True, kw_only=True)
class FluxVTORequest:
    prompt: str
    person: str
    garment: str
    seed: int | None = None
    safety_tolerance: int = 5
    output_format: str = "png"

    def __post_init__(self) -> None:
        _required_text(self.person, "person")
        _required_text(self.garment, "garment")


@dataclass(frozen=True, kw_only=True)
class Flux2Request:
    prompt: str
    width: int = 1024
    height: int = 768
    seed: int | None = None
    prompt_upsampling: bool | None = None
    input_image: str | None = None
    input_image_2: str | None = None
    input_image_3: str | None = None
    input_image_4: str | None = None
    input_image_5: str | None = None
    input_image_6: str | None = None
    input_image_7: str | None = None
    input_image_8: str | None = None
    input_image_9: str | None = None
    safety_tolerance: int = 5
    output_format: str = "png"

    def __post_init__(self) -> None:
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32")


BFL_CONTRACTS: Mapping[str, type[object]] = {
    "BFLFluxExpandImageRequest": FluxExpandRequest,
    "BFLFluxFillImageRequest": FluxFillRequest,
    "BFLFluxEraseRequest": FluxEraseRequest,
    "BFLFluxVTORequest": FluxVTORequest,
    "BFLFluxKontextProGenerateRequest": FluxKontextRequest,
    "BFLFluxProUltraGenerateRequest": FluxUltraRequest,
    "Flux2ProGenerateRequest": Flux2Request,
    "BFLFluxProGenerateResponse": BFLGenerateResponse,
    "BFLFluxStatusResponse": BFLStatusResponse,
}


def _input(
    id: str,
    type_: TypeExpr,
    default: object,
    *,
    required: bool = True,
    widget: BooleanWidget | NumberWidget | StringWidget | None = None,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        id, type_, required=required, default=default, widget=widget, advanced=advanced
    )


def _common_request(
    path: str,
    bindings: tuple[InputBinding, ...],
    fixed: tuple[FixedField, ...],
    *,
    path_input: str | None = None,
    paths: tuple[tuple[str, str], ...] = (),
):
    return (
        HttpSyncJson(
            "submit", path, body=bindings, fixed=fixed, path_input=path_input, paths=paths
        ),
        SubmitPoll(
            "poll",
            "submit",
            completed=("Ready",),
            failed=FAILED,
            queued=(),
            allowed=BFL_STATUSES,
        ),
        DownloadDecode("download", "poll", ("result", "sample"), "image"),
    )


class BFLNode(Node):
    SPEC: OpSpec

    @classmethod
    async def execute(cls, **inputs: object) -> Mapping[str, object]:
        return await run_op(cls.SPEC, inputs, worker_runtime_context())


def _schema(
    upstream_id: str,
    display: str,
    inputs: tuple[InputSpec, ...],
    *,
    combos: tuple[DynamicComboSpec, ...] = (),
    search_visibility: Literal["normal", "deprecated", "hidden"] = "normal",
) -> NodeSchema:
    stem = upstream_id.removesuffix("Node")
    operation = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", stem).lower()
    return NodeSchema(
        node_type=f"partner.bfl.{operation}",
        display_name=display,
        category="partner/image/BFL",
        inputs=inputs,
        outputs=(OutputSpec("image", IMAGE),),
        aliases=(upstream_id,),
        io_bound=True,
        combos=combos,
        search_visibility=search_visibility,
    )


PROMPT = _input("prompt", STRING, "", widget=StringWidget())
# Upstream BFL declares seed max 0xFFFFFFFFFFFFFFFF (2^64-1), but an int
# beyond 2^53-1 cannot round-trip a JSON double, so the wire value would be
# lossy for every client. Clamp to the JSON-double-safe bound; the provider
# accepts any seed in [0, 2^53-1] and randomize stays fully usable.
SEED_WIDGET = NumberWidget(0, 2**53 - 1, control_after_generate="randomize")
SEED = _input("seed", INT, 0, widget=SEED_WIDGET)
UPSAMPLE_FALSE = _input("prompt_upsampling", BOOLEAN, False, required=True)


class FluxProUltraImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "FluxProUltraImageNode",
            "Flux 1.1 [pro] Ultra Image",
            (
                PROMPT,
                UPSAMPLE_FALSE,
                SEED,
                _input("aspect_ratio", STRING, "16:9"),
                _input("raw", BOOLEAN, False),
                _input("image_prompt", IMAGE, None, required=False),
                _input(
                    "image_prompt_strength",
                    FLOAT,
                    0.1,
                    required=False,
                    widget=NumberWidget(0.0, 1.0, 0.01),
                ),
            ),
        )

    SPEC = OpSpec(
        (
            MediaConstraints(
                "validate_aspect", "aspect_ratio", min_aspect_ratio=0.25, max_aspect_ratio=4
            ),
            EncodeMedia(
                "image_prompt_encoded", "image_prompt", max_pixels=MAX_IMAGE_PIXELS, optional=True
            ),
        )
        + _common_request(
            "/proxy/bfl/flux-pro-1.1-ultra/generate",
            (
                InputBinding("prompt", "prompt"),
                InputBinding("prompt_upsampling", "prompt_upsampling"),
                InputBinding("seed", "seed"),
                InputBinding("aspect_ratio", "aspect_ratio"),
                InputBinding("raw", "raw"),
                InputBinding("image_prompt", "image_prompt_encoded", present_if="image_prompt"),
                InputBinding(
                    "image_prompt_strength",
                    "image_prompt_strength",
                    round_digits=2,
                    present_if="image_prompt",
                ),
            ),
            (FixedField("safety_tolerance", 6), FixedField("output_format", "png")),
        )
    )


def _kontext_schema(node: str, display: str) -> NodeSchema:
    return _schema(
        node,
        display,
        (
            PROMPT,
            _input("aspect_ratio", STRING, "16:9"),
            _input("guidance", FLOAT, 3.0, widget=NumberWidget(0.1, 99.0, 0.1)),
            _input("steps", INT, 50, widget=NumberWidget(1, 150)),
            _input("seed", INT, 1234, widget=SEED_WIDGET),
            UPSAMPLE_FALSE,
            _input("input_image", IMAGE, None, required=False),
        ),
    )


def _kontext_spec(path: str) -> OpSpec:
    return OpSpec(
        (
            MediaConstraints(
                "validate_aspect", "aspect_ratio", min_aspect_ratio=0.25, max_aspect_ratio=4
            ),
            EncodeMedia("input_encoded", "input_image", max_pixels=MAX_IMAGE_PIXELS, optional=True),
        )
        + _common_request(
            path,
            (
                InputBinding("prompt", "prompt"),
                InputBinding("aspect_ratio", "aspect_ratio"),
                InputBinding("guidance", "guidance", round_digits=1),
                InputBinding("steps", "steps"),
                InputBinding("seed", "seed"),
                InputBinding("prompt_upsampling", "prompt_upsampling"),
                InputBinding("input_image", "input_encoded", present_if="input_image"),
            ),
            (FixedField("safety_tolerance", 2), FixedField("output_format", "png")),
        )
    )


class FluxKontextProImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _kontext_schema("FluxKontextProImageNode", "Flux.1 Kontext [pro] Image")

    SPEC = _kontext_spec("/proxy/bfl/flux-kontext-pro/generate")


class FluxKontextMaxImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _kontext_schema("FluxKontextMaxImageNode", "Flux.1 Kontext [max] Image")

    SPEC = _kontext_spec("/proxy/bfl/flux-kontext-max/generate")


def _image_pipeline(
    path: str,
    fields: tuple[str, ...],
    fixed: tuple[FixedField, ...],
    *,
    mask: bool = False,
    constraints: bool = False,
    include_image: bool = True,
    rgb: bool = True,
) -> OpSpec:
    adapters: list[Adapter] = []
    if constraints:
        adapters.append(MediaConstraints("validate", "image", min_width=256, min_height=256))
    if include_image:
        adapters.append(EncodeMedia("image_encoded", "image", max_pixels=MAX_IMAGE_PIXELS, rgb=rgb))
    if mask:
        adapters.extend(
            (
                MaskPrepare("mask_prepared", "mask", "image"),
                EncodeMedia("mask_encoded", "mask_prepared", max_pixels=MAX_IMAGE_PIXELS),
            )
        )
    bindings = [InputBinding(name, name) for name in fields]
    if include_image:
        bindings.append(InputBinding("image", "image_encoded"))
    if mask:
        bindings.append(InputBinding("mask", "mask_encoded"))
    return OpSpec(tuple(adapters) + _common_request(path, tuple(bindings), fixed))


class FluxProExpandNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "FluxProExpandNode",
            "Flux.1 Expand Image",
            (
                _input("image", IMAGE, None),
                PROMPT,
                UPSAMPLE_FALSE,
                *(
                    _input(x, INT, 0, widget=NumberWidget(0, 2048))
                    for x in ("top", "bottom", "left", "right")
                ),
                _input("guidance", FLOAT, 60, widget=NumberWidget(1.5, 100)),
                _input("steps", INT, 50, widget=NumberWidget(15, 50)),
                SEED,
            ),
        )

    SPEC = _image_pipeline(
        "/proxy/bfl/flux-pro-1.0-expand/generate",
        (
            "prompt",
            "prompt_upsampling",
            "top",
            "bottom",
            "left",
            "right",
            "guidance",
            "steps",
            "seed",
        ),
        (FixedField("safety_tolerance", 6), FixedField("output_format", "png")),
        rgb=False,
    )


class FluxProFillNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "FluxProFillNode",
            "Flux.1 Fill Image",
            (
                _input("image", IMAGE, None),
                _input("mask", MASK, None),
                PROMPT,
                UPSAMPLE_FALSE,
                _input("guidance", FLOAT, 60, widget=NumberWidget(1.5, 100)),
                _input("steps", INT, 50, widget=NumberWidget(15, 50)),
                SEED,
            ),
        )

    SPEC = _image_pipeline(
        "/proxy/bfl/flux-pro-1.0-fill/generate",
        ("prompt", "prompt_upsampling", "guidance", "steps", "seed"),
        (FixedField("safety_tolerance", 6), FixedField("output_format", "png")),
        mask=True,
    )


class FluxEraseNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "FluxEraseNode",
            "Flux Erase Image",
            (
                _input("image", IMAGE, None),
                _input("mask", MASK, None),
                _input("dilate_pixels", INT, 10, widget=NumberWidget(0, 25)),
                _input(
                    "seed",
                    INT,
                    0,
                    required=False,
                    widget=NumberWidget(0, 2147483647, control_after_generate="randomize"),
                ),
            ),
        )

    SPEC = _image_pipeline(
        "/proxy/bfl/v1/flux-tools/erase-v1",
        ("dilate_pixels", "seed"),
        (FixedField("output_format", "png"),),
        mask=True,
        constraints=True,
    )


class FluxVTONode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "FluxVTONode",
            "Flux Virtual Try-On",
            (_input("person", IMAGE, None), _input("garment", IMAGE, None), PROMPT, SEED),
        )

    SPEC = OpSpec(
        (
            EncodeMedia("person_encoded", "person", max_pixels=MAX_IMAGE_PIXELS, rgb=True),
            EncodeMedia("garment_encoded", "garment", max_pixels=MAX_IMAGE_PIXELS, rgb=True),
        )
        + _common_request(
            "/proxy/bfl/v1/flux-tools/vto-v1",
            (
                InputBinding("person", "person_encoded"),
                InputBinding("garment", "garment_encoded"),
                InputBinding("prompt", "prompt"),
                InputBinding("seed", "seed"),
            ),
            (FixedField("safety_tolerance", 5), FixedField("output_format", "png")),
        )
    )


def _flux2_inputs(include_upsampling: bool = True) -> tuple[InputSpec, ...]:
    result = [
        PROMPT,
        _input("width", INT, 1024, widget=NumberWidget(256, 2048, 32)),
        _input("height", INT, 768, widget=NumberWidget(256, 2048, 32)),
        SEED,
    ]
    if include_upsampling:
        result.append(_input("prompt_upsampling", BOOLEAN, True))
    result.append(_input("images", IMAGE, None, required=False))
    return tuple(result)


def _flux2_spec(path: str, count: int = 9, upsampling: bool = True) -> OpSpec:
    body = [
        InputBinding("prompt", "prompt"),
        InputBinding("width", "width"),
        InputBinding("height", "height"),
        InputBinding("seed", "seed"),
        InputBinding("", "references", expand=True),
    ]
    if upsampling:
        body.insert(4, InputBinding("prompt_upsampling", "prompt_upsampling"))
    return OpSpec(
        (
            EncodeMedia(
                "references",
                "images",
                max_pixels=2048 * 2048,
                optional=True,
                batch_targets=tuple(
                    "input_image" if index == 1 else f"input_image_{index}"
                    for index in range(1, count + 1)
                ),
            ),
        )
        + _common_request(
            path,
            tuple(body),
            (FixedField("safety_tolerance", 5), FixedField("output_format", "png")),
        )
    )


class Flux2ProImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "Flux2ProImageNode",
            "Flux.2 [pro] Image",
            _flux2_inputs(),
            search_visibility="deprecated",
        )

    SPEC = _flux2_spec("/proxy/bfl/flux-2-pro/generate")


class Flux2MaxImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "Flux2MaxImageNode",
            "Flux.2 [max] Image",
            _flux2_inputs(),
            search_visibility="deprecated",
        )

    SPEC = _flux2_spec("/proxy/bfl/flux-2-max/generate")


class Flux2ImageNode(BFLNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        nested = (
            _input("width", INT, 1024, widget=NumberWidget(256, 2048, 32)),
            _input("height", INT, 768, widget=NumberWidget(256, 2048, 32)),
        )
        family = InputFamilySpec(
            "images", IMAGE, min_members=0, member_names=tuple(f"image_{i}" for i in range(1, 9))
        )
        options = tuple(
            DynamicComboOption(key, nested + (family,)) for key in ("Flux.2 [pro]", "Flux.2 [max]")
        )
        return _schema(
            "Flux2ImageNode",
            "Flux.2 Image",
            (PROMPT, SEED),
            combos=(DynamicComboSpec("model", options),),
        )

    SPEC = OpSpec(
        (
            EncodeMedia(
                "references",
                "model.images",
                max_pixels=2048 * 2048,
                optional=True,
                batch_targets=tuple(
                    "input_image" if index == 1 else f"input_image_{index}" for index in range(1, 9)
                ),
            ),
        )
        + _common_request(
            "/proxy/bfl/flux-2-pro/generate",
            (
                InputBinding("prompt", "prompt"),
                InputBinding("width", "model.width"),
                InputBinding("height", "model.height"),
                InputBinding("seed", "seed"),
                InputBinding("", "references", expand=True),
            ),
            (FixedField("safety_tolerance", 5), FixedField("output_format", "png")),
            path_input="model",
            paths=(
                ("Flux.2 [pro]", "/proxy/bfl/flux-2-pro/generate"),
                ("Flux.2 [max]", "/proxy/bfl/flux-2-max/generate"),
            ),
        )
    )


BFL_NODES: list[type[Node]] = [
    FluxProUltraImageNode,
    FluxKontextProImageNode,
    FluxKontextMaxImageNode,
    FluxProExpandNode,
    FluxProFillNode,
    FluxEraseNode,
    FluxVTONode,
    Flux2ProImageNode,
    Flux2MaxImageNode,
    Flux2ImageNode,
]
