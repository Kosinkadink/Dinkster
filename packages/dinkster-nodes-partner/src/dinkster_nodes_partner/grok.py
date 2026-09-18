"""Grok partner nodes pinned to ComfyUI e651b7be."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from dinkster_api.v1 import (
    CORE_INT,
    CORE_STRING,
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
    BatchMapJoin,
    Check,
    CheckInputs,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    HttpSyncJson,
    InputBinding,
    MediaConstraints,
    OpSpec,
    ProxyUpload,
    SubmitPoll,
    ValueConstruct,
)
from .partner_runtime import run_op, worker_runtime_context

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
IMAGE = TypeExpr.concrete("comfy.IMAGE")
VIDEO = TypeExpr.concrete("comfy.VIDEO")
MAX_IMAGE_PIXELS = 2048 * 2048


def _text(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class InputUrlObject:
    url: str

    def __post_init__(self) -> None:
        _text(self.url, "url")


@dataclass(frozen=True, kw_only=True)
class ImageGenerationRequest:
    model: str
    prompt: str
    aspect_ratio: str
    n: int
    seed: int
    response_format: str = "url"
    resolution: str

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.prompt, "prompt")
        _text(self.aspect_ratio, "aspect_ratio")
        if self.n < 1:
            raise ValueError("n must be positive")


@dataclass(frozen=True, kw_only=True)
class ImageEditRequest:
    model: str
    images: list[InputUrlObject]
    prompt: str
    resolution: str
    n: int
    seed: int
    response_format: str = "url"
    aspect_ratio: str | None

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.prompt, "prompt")
        if not self.images:
            raise ValueError("images must not be empty")
        if self.n < 1:
            raise ValueError("n must be positive")


@dataclass(frozen=True, kw_only=True)
class VideoGenerationRequest:
    model: str
    prompt: str
    image: InputUrlObject | None = None
    reference_images: list[InputUrlObject] | None = None
    duration: int
    aspect_ratio: str | None
    resolution: str
    seed: int

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.prompt, "prompt")
        if self.duration < 1:
            raise ValueError("duration must be positive")


@dataclass(frozen=True, kw_only=True)
class VideoExtensionRequest:
    prompt: str
    video: InputUrlObject
    duration: int = 6
    model: str | None = None

    def __post_init__(self) -> None:
        _text(self.prompt, "prompt")
        if self.duration < 1:
            raise ValueError("duration must be positive")


@dataclass(frozen=True, kw_only=True)
class VideoEditRequest:
    model: str
    prompt: str
    video: InputUrlObject
    seed: int

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.prompt, "prompt")


@dataclass(frozen=True, kw_only=True)
class ImageResponseObject:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class UsageObject:
    cost_in_usd_ticks: int | None = None

    def __post_init__(self) -> None:
        if self.cost_in_usd_ticks is not None and self.cost_in_usd_ticks < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True, kw_only=True)
class ImageGenerationResponse:
    data: list[ImageResponseObject]
    usage: UsageObject | None = None

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("data must not be empty")


@dataclass(frozen=True, kw_only=True)
class VideoGenerationResponse:
    request_id: str

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")


@dataclass(frozen=True, kw_only=True)
class VideoResponseObject:
    url: str
    upsampled_prompt: str | None = None
    duration: int

    def __post_init__(self) -> None:
        _text(self.url, "url")
        if self.duration < 0:
            raise ValueError("duration must be non-negative")


@dataclass(frozen=True, kw_only=True)
class VideoStatusResponse:
    status: str | None = None
    video: VideoResponseObject | None = None
    model: str | None = None
    usage: UsageObject | None = None


GROK_CONTRACTS: Mapping[str, type[object]] = {
    name: value
    for name, value in globals().copy().items()
    if name
    in {
        "InputUrlObject",
        "ImageGenerationRequest",
        "ImageEditRequest",
        "VideoGenerationRequest",
        "VideoExtensionRequest",
        "VideoEditRequest",
        "ImageResponseObject",
        "UsageObject",
        "ImageGenerationResponse",
        "VideoGenerationResponse",
        "VideoResponseObject",
        "VideoStatusResponse",
    }
}


def _input(
    id: str, type_: TypeExpr, default: object, *, required: bool = True, widget: object = None
) -> InputSpec:
    return InputSpec(id, type_, required=required, default=default, widget=widget)  # type: ignore[arg-type]


PROMPT = _input("prompt", STRING, "", widget=StringWidget())
SEED = _input(
    "seed", INT, 0, widget=NumberWidget(0, 2147483647, control_after_generate="randomize")
)
PROMPT_CHECK = CheckInputs(
    "validate_prompt",
    (
        Check(
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
            require=(Cond("prompt", "strip_min_len", 1),),
        ),
    ),
)


def _schema(
    upstream: str,
    display: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    *,
    combos: tuple[DynamicComboSpec, ...] = (),
    deprecated: bool = False,
    output: TypeExpr = IMAGE,
) -> NodeSchema:
    operation = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])", "-", upstream.removeprefix("Grok").removesuffix("Node")
    ).lower()
    operation = operation.replace("-node-", "-").removesuffix("-node")
    return NodeSchema(
        node_type=f"partner.grok.{operation}",
        display_name=display,
        category=category,
        inputs=inputs,
        outputs=(OutputSpec("video" if output == VIDEO else "image", output),),
        aliases=(upstream,),
        io_bound=True,
        combos=combos,
        search_visibility="deprecated" if deprecated else "normal",
    )


class GrokNode(Node):
    SPEC: OpSpec

    @classmethod
    async def execute(cls, **inputs: object) -> Mapping[str, object]:
        return await run_op(cls.SPEC, inputs, worker_runtime_context())


def _image_download(source: str = "request") -> DownloadDecode:
    return DownloadDecode(
        "download", source, output="image", items_path=("data",), item_url_path=("url",)
    )


class GrokImageNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "GrokImageNode",
            "Grok Image",
            "partner/image/Grok",
            (
                _input("model", STRING, "grok-imagine-image-quality"),
                PROMPT,
                _input("aspect_ratio", STRING, "1:1"),
                _input("number_of_images", INT, 1, widget=NumberWidget(1, 10)),
                SEED,
                _input("resolution", STRING, "1K", required=False),
            ),
        )

    SPEC = OpSpec(
        (
            PROMPT_CHECK,
            HttpSyncJson(
                "request",
                "/proxy/xai/v1/images/generations",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("aspect_ratio", "aspect_ratio"),
                    InputBinding("n", "number_of_images"),
                    InputBinding("seed", "seed"),
                    InputBinding("resolution", "resolution", string_case="lower"),
                ),
                fixed=(FixedField("response_format", "url"),),
            ),
            _image_download(),
        )
    )


def _edit_spec(dynamic: bool) -> OpSpec:
    prefix = "model." if dynamic else ""
    source = "model.images" if dynamic else "image"
    return OpSpec(
        (
            PROMPT_CHECK,
            CheckInputs(
                "validate_images",
                (
                    Check(
                        "At least one image is required for editing.",
                        require=(Cond(source, "count_ge", 1),),
                    ),
                    Check(
                        "The pro model supports only 1 input image.",
                        when=(Cond("model", "eq", "grok-imagine-image-pro"),),
                        require=(Cond(source, "count_le", 1),),
                    ),
                    Check(
                        "A maximum of 3 input images is supported.",
                        when=(Cond("model", "ne", "grok-imagine-image-pro"),),
                        require=(Cond(source, "count_le", 3),),
                    ),
                    Check(
                        "Custom aspect ratio is only allowed when multiple images are connected "
                        "to the image input.",
                        when=(
                            Cond(prefix + "aspect_ratio", "present"),
                            Cond(prefix + "aspect_ratio", "ne", "auto"),
                        ),
                        require=(Cond(source, "count_ge", 2),),
                    ),
                ),
            ),
            EncodeMedia(
                "encoded",
                source,
                output="data_url",
                max_pixels=MAX_IMAGE_PIXELS,
                optional=False,
                batch_targets=("image_1", "image_2", "image_3"),
            ),
            BatchMapJoin("images", "encoded", "url"),
            HttpSyncJson(
                "request",
                "/proxy/xai/v1/images/edits",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("images", "images"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("resolution", prefix + "resolution", string_case="lower"),
                    InputBinding("n", prefix + "number_of_images"),
                    InputBinding("seed", "seed"),
                    InputBinding(
                        "aspect_ratio",
                        prefix + "aspect_ratio",
                        present_if=prefix + "aspect_ratio",
                        omit_if="auto",
                    ),
                ),
                fixed=(FixedField("response_format", "url"),),
            ),
            _image_download(),
        )
    )


def _edit_nested(max_refs: int, aspect: bool) -> tuple[InputSpec | InputFamilySpec, ...]:
    result: list[InputSpec | InputFamilySpec] = [
        InputFamilySpec(
            "images",
            IMAGE,
            min_members=1,
            member_names=tuple(f"image_{i}" for i in range(1, max_refs + 1)),
        ),
        _input("resolution", STRING, "1K"),
        _input("number_of_images", INT, 1, widget=NumberWidget(1, 10)),
    ]
    if aspect:
        result.append(_input("aspect_ratio", STRING, "auto"))
    return tuple(result)


class GrokImageEditNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "GrokImageEditNode",
            "Grok Image Edit",
            "partner/image/Grok",
            (
                _input("model", STRING, "grok-imagine-image-quality"),
                _input("image", IMAGE, None),
                PROMPT,
                _input("resolution", STRING, "1K"),
                _input("number_of_images", INT, 1, widget=NumberWidget(1, 10)),
                SEED,
                _input("aspect_ratio", STRING, "auto", required=False),
            ),
            deprecated=True,
        )

    SPEC = _edit_spec(False)


class GrokImageEditNodeV2(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        options = (
            DynamicComboOption("grok-imagine-image-quality", _edit_nested(3, True)),
            DynamicComboOption("grok-imagine-image-pro", _edit_nested(1, False)),
            DynamicComboOption("grok-imagine-image", _edit_nested(3, True)),
        )
        return _schema(
            "GrokImageEditNodeV2",
            "Grok Image Edit",
            "partner/image/Grok",
            (PROMPT, SEED),
            combos=(DynamicComboSpec("model", options),),
        )

    SPEC = _edit_spec(True)


def _poll_download(path: str) -> tuple[Adapter, ...]:
    poll = SubmitPoll(
        "poll",
        path,
        status_path=("status",),
        completed=("complete", "none"),
        path_template="/proxy/xai/v1/videos/{value}",
        path_value_path=("request_id",),
    )
    return (
        poll,
        DownloadDecode("download", "poll", ("video", "url"), "video", media_family="video"),
    )


class GrokVideoNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "GrokVideoNode",
            "Grok Video",
            "partner/video/Grok",
            (
                _input("model", STRING, "grok-imagine-video"),
                PROMPT,
                _input("resolution", STRING, "480p"),
                _input("aspect_ratio", STRING, "auto"),
                _input("duration", INT, 6, widget=NumberWidget(1, 15)),
                SEED,
                _input("image", IMAGE, None, required=False),
            ),
            output=VIDEO,
        )

    SPEC = OpSpec(
        (
            PROMPT_CHECK,
            CheckInputs(
                "validate",
                (
                    Check(
                        "The 'grok-imagine-video-1.5' model requires an input image; "
                        "connect one to the 'image' input.",
                        when=(Cond("model", "eq", "grok-imagine-video-1.5"),),
                        require=(Cond("image", "present"),),
                    ),
                    Check(
                        "1080p resolution is only available for grok-imagine-video-1.5, "
                        "not 'grok-imagine-video'.",
                        when=(Cond("resolution", "eq", "1080p"),),
                        require=(Cond("model", "eq", "grok-imagine-video-1.5"),),
                    ),
                    Check(
                        "Only one input image is supported.",
                        require=(Cond("image", "count_le", 1),),
                    ),
                ),
            ),
            EncodeMedia(
                "encoded",
                "image",
                output="data_url",
                max_pixels=MAX_IMAGE_PIXELS,
                optional=True,
            ),
            ValueConstruct(
                "image_object", "video", (InputBinding("url", "encoded", present_if="image"),)
            ),
            HttpSyncJson(
                "submit",
                "/proxy/xai/v1/videos/generations",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("prompt", "prompt"),
                    InputBinding("image", "image_object", present_if="image"),
                    InputBinding("resolution", "resolution"),
                    InputBinding("duration", "duration"),
                    InputBinding("aspect_ratio", "aspect_ratio", omit_if="auto"),
                    InputBinding("seed", "seed"),
                ),
            ),
            *_poll_download("submit"),
        )
    )


def _video_upload_spec(path: str, *, extend: bool) -> OpSpec:
    prefix = "model." if extend else ""
    checks = (
        PROMPT_CHECK,
        MediaConstraints(
            "validate_video",
            "video",
            min_duration=2 if extend else 1,
            max_duration=15 if extend else 8.7,
            max_bytes=50 * 1024 * 1024,
        ),
        ProxyUpload("uploaded", "video", "upload.mp4", "video/mp4"),
        ValueConstruct("video_object", "video", (InputBinding("url", "uploaded"),)),
    )
    if extend:
        body = [
            InputBinding("prompt", "prompt"),
            InputBinding("video", "video_object"),
            InputBinding("duration", prefix + "duration"),
        ]
    else:
        body = [
            InputBinding("model", "model"),
            InputBinding("prompt", "prompt"),
            InputBinding("video", "video_object"),
            InputBinding("seed", "seed"),
        ]
    return OpSpec(
        (*checks, HttpSyncJson("submit", path, body=tuple(body)), *_poll_download("submit"))
    )


class GrokVideoEditNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "GrokVideoEditNode",
            "Grok Video Edit",
            "partner/video/Grok",
            (
                _input("model", STRING, "grok-imagine-video"),
                PROMPT,
                _input("video", VIDEO, None),
                SEED,
            ),
            output=VIDEO,
        )

    SPEC = _video_upload_spec("/proxy/xai/v1/videos/edits", extend=False)


class GrokVideoReferenceNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        nested = (
            InputFamilySpec(
                "reference_images",
                IMAGE,
                min_members=1,
                member_names=tuple(f"reference_{i}" for i in range(1, 8)),
            ),
            _input("resolution", STRING, "480p"),
            _input("aspect_ratio", STRING, "16:9"),
            _input("duration", INT, 6, widget=NumberWidget(2, 10)),
        )
        return _schema(
            "GrokVideoReferenceNode",
            "Grok Reference-to-Video",
            "partner/video/Grok",
            (PROMPT, SEED),
            combos=(
                DynamicComboSpec("model", (DynamicComboOption("grok-imagine-video", nested),)),
            ),
            output=VIDEO,
        )

    SPEC = OpSpec(
        (
            PROMPT_CHECK,
            EncodeMedia(
                "encoded",
                "model.reference_images",
                output="bytes",
                max_pixels=MAX_IMAGE_PIXELS,
                optional=True,
                batch_targets=tuple(f"reference_{i}" for i in range(1, 8)),
            ),
            ProxyUpload("uploaded", "encoded", "reference.png", "image/png", batch=True),
            BatchMapJoin("references", "uploaded", "url"),
            HttpSyncJson(
                "submit",
                "/proxy/xai/v1/videos/generations",
                body=(
                    InputBinding("model", "model"),
                    InputBinding("reference_images", "references", omit_none=False),
                    InputBinding("prompt", "prompt"),
                    InputBinding("resolution", "model.resolution"),
                    InputBinding("duration", "model.duration"),
                    InputBinding("aspect_ratio", "model.aspect_ratio"),
                    InputBinding("seed", "seed"),
                ),
            ),
            *_poll_download("submit"),
        )
    )


class GrokVideoExtendNode(GrokNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "GrokVideoExtendNode",
            "Grok Video Extend",
            "partner/video/Grok",
            (PROMPT, _input("video", VIDEO, None), SEED),
            combos=(
                DynamicComboSpec(
                    "model",
                    (
                        DynamicComboOption(
                            "grok-imagine-video",
                            (_input("duration", INT, 8, widget=NumberWidget(2, 10)),),
                        ),
                    ),
                ),
            ),
            output=VIDEO,
        )

    SPEC = _video_upload_spec("/proxy/xai/v1/videos/extensions", extend=True)


GROK_NODES: list[type[Node]] = [
    GrokImageNode,
    GrokImageEditNode,
    GrokImageEditNodeV2,
    GrokVideoNode,
    GrokVideoReferenceNode,
    GrokVideoEditNode,
    GrokVideoExtendNode,
]
