"""Pure-data partner operation descriptors.

Every descriptor is frozen, validates on construction, and round-trips through
JSON without loss. The closed vocabulary can move the same data across a
repository or network boundary in partner phase 2.
"""
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlparse

ADAPTER_KINDS = (
    "check_inputs",
    "value_construct",
    "http_sync_json",
    "http_sync_binary",
    "submit_poll",
    "multi_stage",
    "proxy_upload",
    "encode_media",
    "media_constraints",
    "mask_prepare",
    "multipart_map",
    "download_decode",
    "batch_map_join",
    "response_select",
    "local_progress",
)


def _required(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must not be empty")


def _validate_body_targets(
    bindings: tuple[InputBinding, ...],
    fixed: tuple[FixedField, ...],
    formatted: tuple[FormatField, ...] = (),
) -> None:
    tracked: list[tuple[tuple[str, ...], str]] = []
    targets = [field.target for field in fixed]
    targets.extend(field.target for field in formatted)
    targets.extend(binding.target for binding in bindings if not binding.expand)
    for target in targets:
        parts = tuple(target.split("."))
        for prior_parts, prior_target in tracked:
            common = min(len(parts), len(prior_parts))
            if parts[:common] == prior_parts[:common]:
                raise ValueError(f"request body conflict between {prior_target!r} and {target!r}")
        tracked.append((parts, target))


@dataclass(frozen=True)
class InputBinding:
    target: str
    input: str
    omit_none: bool = True
    source_path: tuple[str | int, ...] = ()
    round_digits: int | None = None
    present_if: str | None = None
    expand: bool = False
    string_case: Literal["", "lower"] = ""
    omit_if: str | None = None
    value_map: Mapping[str, str | int | float | bool] | None = None

    def __post_init__(self) -> None:
        if not self.expand:
            _required(self.target, "binding target")
        _required(self.input, "binding input")
        if self.round_digits is not None and self.round_digits < 0:
            raise ValueError("binding round_digits must be non-negative")
        if self.string_case not in ("", "lower"):
            raise ValueError("binding string_case must be empty or lower")
        if self.value_map is not None:
            if not self.value_map:
                raise ValueError("binding value_map must not be empty")
            copied: dict[str, str | int | float | bool] = {}
            raw_map = cast("Mapping[object, object]", self.value_map)
            for key, value in raw_map.items():
                if not isinstance(key, str):
                    raise ValueError("binding value_map keys must be strings")
                if not isinstance(value, (str, int, float, bool)) or (
                    isinstance(value, float) and not math.isfinite(value)
                ):
                    raise ValueError("binding value_map values must be finite JSON scalars")
                copied[key] = value
            object.__setattr__(self, "value_map", MappingProxyType(copied))
        if self.expand and (
            self.source_path
            or self.round_digits is not None
            or self.present_if is not None
            or self.string_case
            or self.omit_if is not None
            or self.value_map is not None
        ):
            raise ValueError("expanded bindings cannot use transforms or conditions")


@dataclass(frozen=True)
class FixedField:
    target: str
    value: str | int | float | bool | None

    def __post_init__(self) -> None:
        _required(self.target, "fixed field target")


@dataclass(frozen=True)
class FormatField:
    """Format raw worker inputs, never prior adapter state, into a request field."""

    target: str
    template: str

    def __post_init__(self) -> None:
        _required(self.target, "format field target")
        _required(self.template, "format field template")
        placeholders = re.findall(r"\{([^{}]+)\}", self.template)
        if (
            not self.template.isascii()
            or not placeholders
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", item) for item in placeholders)
            or re.sub(r"\{[^{}]+\}", "", self.template).find("{") >= 0
            or re.sub(r"\{[^{}]+\}", "", self.template).find("}") >= 0
        ):
            raise ValueError(
                "format field template permits literal ASCII and bare {input_id} placeholders"
            )


@dataclass(frozen=True)
class ValueConstruct:
    id: str
    output: str
    bindings: tuple[InputBinding, ...] = ()
    fixed: tuple[FixedField, ...] = ()
    formatted: tuple[FormatField, ...] = ()
    kind: Literal["value_construct"] = "value_construct"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        _required(self.output, "value output")
        _validate_body_targets(self.bindings, self.fixed, self.formatted)


Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class Cond:
    input: str
    op: Literal[
        "eq", "ne", "present", "absent", "count_eq", "count_le", "count_ge", "strip_min_len"
    ]
    value: Scalar = None

    def __post_init__(self) -> None:
        raw_value = cast("object", self.value)
        _required(self.input, "condition input")
        if self.op not in (
            "eq",
            "ne",
            "present",
            "absent",
            "count_eq",
            "count_le",
            "count_ge",
            "strip_min_len",
        ):
            raise ValueError("unknown condition operation")
        if self.op in ("eq", "ne") and self.value is None:
            raise ValueError("eq/ne conditions require a scalar value")
        if raw_value is not None and (
            not isinstance(raw_value, (str, int, float, bool))
            or isinstance(raw_value, float)
            and not math.isfinite(raw_value)
        ):
            raise ValueError("condition value must be a finite JSON scalar or null")
        if self.op in ("present", "absent") and self.value is not None:
            raise ValueError("present/absent conditions require null value")
        if self.op.startswith("count_") and (
            not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 0
        ):
            raise ValueError("count conditions require a non-negative integer")
        if self.op == "strip_min_len" and (
            not isinstance(self.value, int) or isinstance(self.value, bool) or self.value <= 0
        ):
            raise ValueError("strip_min_len requires a positive integer")


@dataclass(frozen=True)
class Check:
    message: str
    when: tuple[Cond, ...] = ()
    require: tuple[Cond, ...] = ()

    def __post_init__(self) -> None:
        _required(self.message, "check message")
        if not self.require:
            raise ValueError("check require must not be empty")
        remainder = self.message.replace("{count}", "")
        if self.message.count("{count}") > 1 or "{" in remainder or "}" in remainder:
            raise ValueError("check message permits only one optional {count} placeholder")
        if "{count}" in self.message:
            count_inputs = {cond.input for cond in self.require if cond.op.startswith("count_")}
            if len(count_inputs) != 1:
                raise ValueError(
                    "check message {count} requires count conditions for exactly one input"
                )


@dataclass(frozen=True)
class CheckInputs:
    id: str
    checks: tuple[Check, ...]
    kind: Literal["check_inputs"] = "check_inputs"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        if not self.checks:
            raise ValueError("check_inputs checks must not be empty")


@dataclass(frozen=True)
class HttpSyncJson:
    id: str
    path: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "POST"
    body: tuple[InputBinding, ...] = ()
    fixed: tuple[FixedField, ...] = ()
    path_input: str | None = None
    paths: tuple[tuple[str, str], ...] = ()
    timeout: float = 3600.0
    max_retries: int = 3
    max_rate_limit_retries: int = 16
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    formatted: tuple[FormatField, ...] = ()
    kind: Literal["http_sync_json"] = "http_sync_json"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        _required(self.path, "HTTP path")
        if self.timeout <= 0 or self.max_retries < 0 or self.max_rate_limit_retries < 0:
            raise ValueError("HTTP timeout must be positive and retry budgets non-negative")
        if self.retry_delay < 0 or self.retry_backoff < 1:
            raise ValueError("retry delay must be non-negative and backoff at least 1")
        _validate_body_targets(self.body, self.fixed, self.formatted)
        if bool(self.path_input) != bool(self.paths):
            raise ValueError("variant request paths require both path_input and paths")
        if len({key for key, _ in self.paths}) != len(self.paths):
            raise ValueError("variant request path keys must be unique")


@dataclass(frozen=True)
class HttpSyncBinary:
    id: str
    kind: Literal["http_sync_binary"] = "http_sync_binary"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")


@dataclass(frozen=True)
class SubmitPoll:
    id: str
    source: str = ""
    url_path: tuple[str | int, ...] = ("polling_url",)
    status_path: tuple[str | int, ...] = ("status",)
    progress_path: tuple[str | int, ...] = ("progress",)
    completed: tuple[str, ...] = (
        "succeeded",
        "succeed",
        "success",
        "completed",
        "finished",
        "done",
        "complete",
    )
    failed: tuple[str, ...] = ("cancelled", "canceled", "canceling", "fail", "failed", "error")
    queued: tuple[str, ...] = (
        "created",
        "queued",
        "queueing",
        "submitted",
        "initializing",
        "wait",
        "in_queue",
    )
    allowed: tuple[str, ...] = ()
    interval: float = 5.0
    max_attempts: int = 480
    timeout: float = 120.0
    max_retries: int = 10
    retry_delay: float = 1.0
    retry_backoff: float = 1.4
    cancel_path: str | None = None
    path_template: str = ""
    path_value_path: tuple[str | int, ...] = ()
    kind: Literal["submit_poll"] = "submit_poll"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        if (
            self.interval < 0
            or self.max_attempts <= 0
            or self.timeout <= 0
            or self.max_retries < 0
            or self.retry_delay < 0
            or self.retry_backoff < 1
        ):
            raise ValueError("poll interval/attempt/retry values are invalid")
        if bool(self.path_template) != bool(self.path_value_path):
            raise ValueError("poll path template requires both path_template and path_value_path")
        if self.path_template and self.path_template.count("{value}") != 1:
            raise ValueError("poll path_template must contain exactly one {value}")
        parsed = urlparse(self.path_template)
        if self.path_template and (
            parsed.scheme or parsed.netloc or self.path_template.startswith(("//", "\\"))
        ):
            raise ValueError("poll path_template must be a relative path")


@dataclass(frozen=True)
class MultiStage:
    id: str
    kind: Literal["multi_stage"] = "multi_stage"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")


@dataclass(frozen=True)
class ProxyUpload:
    """Upload media into operation state.

    An absent optional input is represented by ``None`` in state. Downstream
    ``omit_none`` transforms and ``single_optional`` consumers deliberately
    consume that sentinel rather than relying on a missing state entry.
    """

    id: str
    input: str = ""
    file_name: str = "upload.png"
    content_type: str = "image/png"
    allocation_path: str = "/customers/storage"
    batch: bool = False
    optional: bool = False
    kind: Literal["proxy_upload"] = "proxy_upload"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        _required(self.file_name, "upload file name")
        _required(self.content_type, "upload content type")
        if self.batch and self.optional:
            raise ValueError("proxy_upload optional is incompatible with batch")


@dataclass(frozen=True)
class EncodeMedia:
    """Encode media into operation state; see :class:`ProxyUpload` for optional absence."""

    id: str
    input: str = ""
    media_family: Literal["image", "video", "audio"] = "image"
    format: str = "PNG"
    output: Literal["bytes", "base64", "data_url"] = "base64"
    max_pixels: int | None = None
    rgb: bool = False
    optional: bool = False
    batch_targets: tuple[str, ...] = ()
    source_path: tuple[str | int, ...] = ()
    kind: Literal["encode_media"] = "encode_media"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        if self.media_family not in ("image", "video", "audio"):
            raise ValueError("media_family must be image, video, or audio")
        if self.output not in ("bytes", "base64", "data_url"):
            raise ValueError("media output must be bytes, base64, or data_url")
        allowed = {"image": {"PNG", "JPEG"}, "video": {"MP4"}, "audio": {"MP3"}}
        if self.format not in allowed[self.media_family]:
            raise ValueError("media format is not supported for its family")
        if self.max_pixels is not None and self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        if len(set(self.batch_targets)) != len(self.batch_targets):
            raise ValueError("batch encode targets must be unique")
        if any(not target for target in self.batch_targets):
            raise ValueError("batch encode targets must not be empty")


@dataclass(frozen=True)
class MediaConstraints:
    id: str
    input: str = ""
    min_width: int = 1
    min_height: int = 1
    max_width: int | None = None
    max_height: int | None = None
    max_count: int | None = None
    max_bytes: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    optional: bool = False
    duration_media: Literal["video", "audio"] = "video"
    batch: bool = False
    aspect_strict: bool = False
    kind: Literal["media_constraints"] = "media_constraints"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        if min(self.min_width, self.min_height) <= 0:
            raise ValueError("minimum dimensions must be positive")
        if self.max_width is not None and self.max_width < self.min_width:
            raise ValueError("maximum width must be at least the minimum width")
        if self.max_height is not None and self.max_height < self.min_height:
            raise ValueError("maximum height must be at least the minimum height")
        if self.max_count is not None and self.max_count <= 0:
            raise ValueError("maximum media count must be positive")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("maximum media bytes must be positive")
        if (self.min_aspect_ratio is None) != (self.max_aspect_ratio is None):
            raise ValueError("aspect ratio constraints require both minimum and maximum")
        maximum = self.max_aspect_ratio
        if self.min_aspect_ratio is not None and (
            self.min_aspect_ratio <= 0 or maximum is None or self.min_aspect_ratio > maximum
        ):
            raise ValueError("aspect ratio constraint range is invalid")
        if type(self.aspect_strict) is not bool:
            raise ValueError("aspect_strict must be a boolean")
        if self.aspect_strict and self.min_aspect_ratio is None:
            raise ValueError("aspect_strict requires aspect ratio constraints")
        if self.aspect_strict and self.min_aspect_ratio == maximum:
            raise ValueError("strict aspect ratio constraints require distinct bounds")
        if (self.min_duration is not None and self.min_duration <= 0) or (
            self.max_duration is not None and self.max_duration <= 0
        ):
            raise ValueError("duration constraints must be positive")
        if (
            self.min_duration is not None
            and self.max_duration is not None
            and self.min_duration > self.max_duration
        ):
            raise ValueError("duration constraint range is invalid")
        if self.duration_media not in ("video", "audio"):
            raise ValueError("duration_media must be video or audio")
        duration = self.min_duration is not None or self.max_duration is not None
        if self.duration_media == "audio" and not duration:
            raise ValueError("audio duration_media requires a duration constraint")
        if self.duration_media == "audio" and (
            self.min_width != 1
            or self.min_height != 1
            or self.max_width is not None
            or self.max_height is not None
            or self.max_count is not None
            or self.max_bytes is not None
            or self.min_aspect_ratio is not None
            or self.max_aspect_ratio is not None
        ):
            raise ValueError("audio duration constraints cannot use image/video constraints")
        if self.optional and self.batch:
            raise ValueError("media_constraints optional is incompatible with batch")
        if self.batch and duration and self.duration_media != "video":
            raise ValueError("batch audio duration constraints are unsupported")


@dataclass(frozen=True)
class MaskPrepare:
    id: str
    mask_input: str = ""
    image_input: str = ""
    kind: Literal["mask_prepare"] = "mask_prepare"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")


@dataclass(frozen=True)
class MultipartMap:
    id: str
    kind: Literal["multipart_map"] = "multipart_map"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")


@dataclass(frozen=True)
class DownloadDecode:
    id: str
    source: str = ""
    url_path: tuple[str | int, ...] = ()
    output: str = ""
    media_family: Literal["image", "video", "audio", "application"] = "image"
    byte_cap: int = 1024 * 1024 * 1024
    items_path: tuple[str | int, ...] = ()
    item_url_path: tuple[str | int, ...] = ()
    url_paths: tuple[tuple[str | int, ...], ...] = ()
    kind: Literal["download_decode"] = "download_decode"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        if self.media_family not in ("image", "video", "audio", "application"):
            raise ValueError("unsupported download media family")
        if self.byte_cap <= 0:
            raise ValueError("download byte cap must be positive")
        multi = bool(self.items_path or self.item_url_path)
        if (self.items_path and not self.item_url_path) or (
            self.item_url_path and not (self.items_path or self.url_paths)
        ):
            raise ValueError("download item_url_path requires items_path or candidate url_paths")
        if self.url_path and multi:
            raise ValueError("download url_path is incompatible with item paths")
        if self.url_paths:
            if self.url_path or self.items_path:
                raise ValueError("download url_paths is incompatible with single or items_path")
            if any(not path for path in self.url_paths):
                raise ValueError("download candidate paths must not be empty")
            if len(set(self.url_paths)) != len(self.url_paths):
                raise ValueError("download candidate paths must not contain duplicates")


@dataclass(frozen=True)
class Segment:
    source: str
    fixed: Mapping[str, str] = field(default_factory=dict[str, str])
    wrap_key: str | None = "url"
    mode: Literal[
        "single", "single_optional", "mapping", "mapping_values", "verbatim", "verbatim_optional"
    ] = "single"

    def __post_init__(self) -> None:
        _required(self.source, "segment source")
        if self.mode not in (
            "single",
            "single_optional",
            "mapping",
            "mapping_values",
            "verbatim",
            "verbatim_optional",
        ):
            raise ValueError("unknown segment mode")
        raw_fixed = cast("Mapping[object, object]", self.fixed)
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_fixed.items()
        ):
            raise ValueError("segment fixed fields must map strings to strings")
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))
        if self.mode.startswith("verbatim") or self.mode == "mapping_values":
            if self.fixed or self.wrap_key is not None:
                raise ValueError(
                    "verbatim and mapping_values segments cannot use fixed or wrap_key"
                )
        else:
            if not self.wrap_key:
                raise ValueError("non-verbatim segment wrap_key must not be empty")
            if self.wrap_key in self.fixed:
                raise ValueError("segment fixed fields cannot contain wrap_key")


@dataclass(frozen=True)
class BatchMapJoin:
    id: str
    source: str = ""
    wrap_key: str = ""
    segments: tuple[Segment, ...] = ()
    min_items: int | None = None
    max_items: int | None = None
    min_message: str = "batch_map_join requires at least {count} items"
    max_message: str = "batch_map_join has too many items: {count}"
    kind: Literal["batch_map_join"] = "batch_map_join"

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        _required(self.id, "adapter id")
        legacy = bool(self.source or self.wrap_key)
        if legacy == bool(self.segments):
            raise ValueError(
                "batch_map_join requires exactly one legacy source+wrap_key or segments"
            )
        if legacy:
            _required(self.source, "batch source")
            _required(self.wrap_key, "batch wrap key")
            if (
                self.min_items is not None
                or self.max_items is not None
                or self.min_message != "batch_map_join requires at least {count} items"
                or self.max_message != "batch_map_join has too many items: {count}"
            ):
                raise ValueError("batch item bounds and messages require segments mode")
        if (
            self.min_items is not None
            and (isinstance(self.min_items, bool) or self.min_items < 0)
            or self.max_items is not None
            and (isinstance(self.max_items, bool) or self.max_items < 0)
        ):
            raise ValueError("batch item bounds must be non-negative")
        if (
            self.min_items is not None
            and self.max_items is not None
            and self.min_items > self.max_items
        ):
            raise ValueError("batch item bounds are invalid")
        for message in (self.min_message, self.max_message):
            _required(message, "batch count message")
            if (
                message.count("{count}") > 1
                or "{" in message.replace("{count}", "")
                or "}" in message.replace("{count}", "")
            ):
                raise ValueError(
                    "batch count messages permit only one optional {count} placeholder"
                )


@dataclass(frozen=True)
class ResponseSelect:
    id: str
    source: str
    path: tuple[str | int, ...]
    output: str
    paths: tuple[tuple[str | int, ...], ...] = ()
    kind: Literal["response_select"] = "response_select"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "paths", tuple(tuple(path) for path in self.paths))
        _required(self.id, "adapter id")
        _required(self.source, "response source")
        _required(self.output, "response output")
        if bool(self.path) == bool(self.paths):
            raise ValueError("response selection requires exactly one path or candidate paths")
        if self.paths and (
            any(not path for path in self.paths) or len(set(self.paths)) != len(self.paths)
        ):
            raise ValueError("response candidate paths must be non-empty and unique")
        if any(
            isinstance(part, bool)
            for path in ((self.path,) if self.path else self.paths)
            for part in path
        ):
            raise ValueError("response selection path parts must be strings or integers")


@dataclass(frozen=True)
class LocalProgress:
    id: str
    text: str
    step: int = 0
    total: int = 1
    kind: Literal["local_progress"] = "local_progress"

    def __post_init__(self) -> None:
        _required(self.id, "adapter id")
        _required(self.text, "progress text")
        if self.total <= 0 or self.step < 0 or self.step > self.total:
            raise ValueError("progress requires 0 <= step <= total and total > 0")


Adapter: TypeAlias = (
    CheckInputs
    | ValueConstruct
    | HttpSyncJson
    | HttpSyncBinary
    | SubmitPoll
    | MultiStage
    | ProxyUpload
    | EncodeMedia
    | MediaConstraints
    | MaskPrepare
    | MultipartMap
    | DownloadDecode
    | BatchMapJoin
    | ResponseSelect
    | LocalProgress
)


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("helper fixed values must be finite JSON trees")
        return value
    if isinstance(value, Mapping):
        raw = cast("Mapping[object, object]", value)
        if not all(isinstance(key, str) for key in raw):
            raise ValueError("helper fixed object keys must be strings")
        return MappingProxyType({cast("str", key): _freeze_json(item) for key, item in raw.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in cast("Sequence[object]", value))
    raise ValueError("helper fixed values must be finite JSON trees")


@dataclass(frozen=True)
class HelperBinding:
    name: str
    source: str
    source_kind: Literal["input", "state"] = "input"
    mode: Literal["value", "family", "present"] = "value"
    source_path: tuple[str | int, ...] = ()
    optional: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", tuple(self.source_path))
        _required(self.name, "helper binding name")
        _required(self.source, "helper binding source")
        if any(
            not isinstance(part, (str, int)) or isinstance(part, bool)
            for part in cast("tuple[object, ...]", self.source_path)
        ):
            raise ValueError("helper binding source_path must contain strings or integers")
        if self.source_kind not in ("input", "state"):
            raise ValueError("helper binding source_kind must be input or state")
        if self.mode not in ("value", "family", "present"):
            raise ValueError("helper binding mode must be value, family, or present")
        if self.mode == "family" and self.source_kind != "input":
            raise ValueError("helper family bindings must use input sources")
        if self.mode in ("family", "present") and self.source_path:
            raise ValueError("helper family/present bindings cannot use source_path")


@dataclass(frozen=True)
class HelperCall:
    id: str
    helper_id: str
    stage: str
    placement: Literal["select", "before", "after"]
    anchor: str = ""
    bindings: tuple[HelperBinding, ...] = ()
    fixed: Mapping[str, object] = field(default_factory=dict[str, object])
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        frozen = _freeze_json(self.fixed)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "fixed", frozen)
        _required(self.id, "helper call id")
        _required(self.helper_id, "helper id")
        _required(self.stage, "helper stage")
        if self.placement not in ("select", "before", "after"):
            raise ValueError("helper placement must be select, before, or after")
        if (self.placement == "select") != (not self.anchor):
            raise ValueError("select helpers require no anchor; before/after require an anchor")
        if not self.outputs or any(not output for output in self.outputs):
            raise ValueError("helper outputs must be non-empty strings")
        names = [binding.name for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("helper binding names must be unique")
        if set(names) & set(self.fixed):
            raise ValueError("helper binding and fixed names must be disjoint")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("helper outputs must be unique")


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"adapter field {key!r} must be a string")
    return value


def _integer(data: dict[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"adapter field {key!r} must be an integer")
    return value


def _number(data: dict[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"adapter field {key!r} must be a number")
    return float(value)


def _boolean(data: dict[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"adapter field {key!r} must be a boolean")
    return value


def _optional_integer(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"adapter field {key!r} must be an integer or null")
    return value


def _optional_number(data: dict[str, object], key: str) -> float | None:
    return _number(data, key, 0) if data.get(key) is not None else None


def _binding(data: dict[str, object]) -> InputBinding:
    omit_none = data.get("omit_none", True)
    if not isinstance(omit_none, bool):
        raise ValueError("binding omit_none must be a boolean")
    string_case = data.get("string_case", "")
    if not isinstance(string_case, str):
        raise ValueError("binding string_case must be a string")
    if string_case not in ("", "lower"):
        raise ValueError("binding string_case must be empty or lower")
    raw_map = data.get("value_map")
    if raw_map is not None and not isinstance(raw_map, dict):
        raise ValueError("binding value_map must be an object or null")
    return InputBinding(
        target=_string(data, "target"),
        input=_string(data, "input"),
        omit_none=omit_none,
        source_path=_path(data.get("source_path", []), "source_path"),
        round_digits=_optional_integer(data, "round_digits"),
        present_if=_string(data, "present_if") if data.get("present_if") is not None else None,
        expand=_boolean(data, "expand", False),
        string_case=string_case,
        omit_if=_string(data, "omit_if") if data.get("omit_if") is not None else None,
        value_map=cast("dict[str, str | int | float | bool] | None", raw_map),
    )


def _fixed(data: dict[str, object]) -> FixedField:
    value = data.get("value")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("fixed field values must be scalar or null")
    return FixedField(_string(data, "target"), value)


def _formatted(data: dict[str, object]) -> FormatField:
    return FormatField(_string(data, "target"), _string(data, "template"))


def _formatted_fields(data: dict[str, object]) -> tuple[FormatField, ...]:
    raw = data.get("formatted", [])
    if not isinstance(raw, list):
        raise ValueError("formatted fields must be objects in an array")
    items = cast("list[object]", raw)
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("formatted fields must be objects in an array")
    return tuple(_formatted(cast("dict[str, object]", item)) for item in items)


def _path(value: object, field: str) -> tuple[str | int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(part, (str, int)) and not isinstance(part, bool)
        for part in cast("list[object]", value)
    ):
        raise ValueError(f"adapter field {field!r} must be an array of strings or integers")
    return tuple(cast("list[str | int]", value))


def _paths(value: object, field: str) -> tuple[tuple[str | int, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"adapter field {field!r} must be an array of paths")
    return tuple(_path(item, field) for item in cast("list[object]", value))


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for field in fields(value):
            item = getattr(value, field.name)
            # New additive grammar defaults stay absent so legacy canonical JSON is byte-identical.
            if field.name == "value_map" and item is None:
                continue
            if field.name == "formatted" and not item:
                continue
            if isinstance(value, OpSpec) and field.name == "helper_calls" and not item:
                continue
            if isinstance(value, ProxyUpload) and field.name == "optional" and not item:
                continue
            if isinstance(value, MediaConstraints) and (
                field.name in ("optional", "batch", "aspect_strict")
                and not item
                or field.name == "duration_media"
                and item == "video"
            ):
                continue
            if isinstance(value, ResponseSelect) and field.name == "paths" and not item:
                continue
            if isinstance(value, DownloadDecode) and field.name == "url_paths" and not item:
                continue
            if isinstance(value, BatchMapJoin) and field.name == "segments" and not item:
                continue
            if (
                isinstance(value, BatchMapJoin)
                and field.name in ("min_items", "max_items")
                and item is None
            ):
                continue
            if isinstance(value, BatchMapJoin) and (
                field.name == "min_message"
                and item == "batch_map_join requires at least {count} items"
                or field.name == "max_message"
                and item == "batch_map_join has too many items: {count}"
            ):
                continue
            result[field.name] = _json_value(item)
        return result
    if isinstance(value, Mapping):
        raw_mapping = cast("Mapping[object, object]", value)
        return {str(key): _json_value(item) for key, item in raw_mapping.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in cast("tuple[object, ...]", value)]
    return value


def _simple_adapter(kind: str, adapter_id: str) -> Adapter:
    if kind == "http_sync_binary":
        return HttpSyncBinary(adapter_id)
    if kind == "submit_poll":
        return SubmitPoll(adapter_id)
    if kind == "multi_stage":
        return MultiStage(adapter_id)
    if kind == "proxy_upload":
        return ProxyUpload(adapter_id)
    if kind == "encode_media":
        return EncodeMedia(adapter_id)
    if kind == "media_constraints":
        return MediaConstraints(adapter_id)
    if kind == "mask_prepare":
        return MaskPrepare(adapter_id)
    if kind == "multipart_map":
        return MultipartMap(adapter_id)
    if kind == "download_decode":
        return DownloadDecode(adapter_id)
    raise ValueError(f"unknown adapter kind {kind!r}")


def _adapter_from_json(data: dict[str, object]) -> Adapter:
    kind = _string(data, "kind")
    adapter_id = _string(data, "id")
    if kind in ("value_construct", "check_inputs"):
        if kind == "check_inputs":
            raw_checks = data.get("checks")
            if not isinstance(raw_checks, list):
                raise ValueError("checks must be an array")
            checks: list[Check] = []
            for raw_check in cast("list[object]", raw_checks):
                if not isinstance(raw_check, dict):
                    raise ValueError("checks must be objects")
                check_data = cast("dict[str, object]", raw_check)

                def conditions(
                    name: str, check_data: dict[str, object] = check_data
                ) -> tuple[Cond, ...]:
                    raw = check_data.get(name)
                    if not isinstance(raw, list):
                        raise ValueError(f"check {name} must be an array")
                    result: list[Cond] = []
                    for item in cast("list[object]", raw):
                        if not isinstance(item, dict):
                            raise ValueError("conditions must be objects")
                        value = cast("dict[str, object]", item)
                        if "value" not in value:
                            raise ValueError("condition must contain a value field")
                        raw_value = value["value"]
                        if raw_value is not None and (
                            not isinstance(raw_value, (str, int, float, bool))
                            or isinstance(raw_value, float)
                            and not math.isfinite(raw_value)
                        ):
                            raise ValueError("condition value must be a finite JSON scalar or null")
                        result.append(
                            Cond(
                                _string(value, "input"),
                                cast(
                                    "Literal['eq','ne','present','absent','count_eq','count_le','count_ge','strip_min_len']",
                                    _string(value, "op"),
                                ),
                                raw_value,
                            )
                        )
                    return tuple(result)

                checks.append(
                    Check(_string(check_data, "message"), conditions("when"), conditions("require"))
                )
            return CheckInputs(adapter_id, tuple(checks))
        body = data.get("bindings", [])
        fixed = data.get("fixed", [])
        if not isinstance(body, list) or not isinstance(fixed, list):
            raise ValueError("value fields must be arrays")
        body_items = cast("list[object]", body)
        fixed_items = cast("list[object]", fixed)
        if not all(isinstance(item, dict) for item in body_items):
            raise ValueError("value bindings must be objects")
        if not all(isinstance(item, dict) for item in fixed_items):
            raise ValueError("value fixed fields must be objects")
        return ValueConstruct(
            adapter_id,
            _string(data, "output"),
            tuple(_binding(cast("dict[str, object]", item)) for item in body_items),
            tuple(_fixed(cast("dict[str, object]", item)) for item in fixed_items),
            _formatted_fields(data),
        )
    if kind == "http_sync_json":
        method = _string(data, "method")
        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            raise ValueError(f"unsupported HTTP method {method!r}")
        body_raw = data.get("body", [])
        if not isinstance(body_raw, list):
            raise ValueError("HTTP body bindings must be an array")
        bindings: list[InputBinding] = []
        for raw_binding in cast("list[object]", body_raw):
            if not isinstance(raw_binding, dict):
                raise ValueError("HTTP body bindings must be objects")
            binding = cast("dict[str, object]", raw_binding)
            bindings.append(_binding(binding))
        fixed_raw = data.get("fixed", [])
        if not isinstance(fixed_raw, list):
            raise ValueError("HTTP fixed fields must be an array")
        fixed_fields: list[FixedField] = []
        for item in cast("list[object]", fixed_raw):
            if not isinstance(item, dict):
                raise ValueError("HTTP fixed fields must be objects")
            field = cast("dict[str, object]", item)
            fixed_fields.append(_fixed(field))
        paths_raw = data.get("paths", [])
        if not isinstance(paths_raw, list):
            raise ValueError("HTTP variant paths must be an array")
        paths: list[tuple[str, str]] = []
        for item in cast("list[object]", paths_raw):
            values = cast("list[object]", item) if isinstance(item, list) else []
            if (
                not isinstance(item, list)
                or len(values) != 2
                or not all(isinstance(value, str) for value in values)
            ):
                raise ValueError("HTTP variant paths must be two-string arrays")
            paths.append((cast("str", values[0]), cast("str", values[1])))
        path_input = data.get("path_input")
        if path_input is not None and not isinstance(path_input, str):
            raise ValueError("adapter field 'path_input' must be a string or null")
        return HttpSyncJson(
            id=adapter_id,
            path=_string(data, "path"),
            method=cast("Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']", method),
            body=tuple(bindings),
            fixed=tuple(fixed_fields),
            path_input=path_input,
            paths=tuple(paths),
            timeout=_number(data, "timeout", 3600.0),
            max_retries=_integer(data, "max_retries", 3),
            max_rate_limit_retries=_integer(data, "max_rate_limit_retries", 16),
            retry_delay=_number(data, "retry_delay", 1.0),
            retry_backoff=_number(data, "retry_backoff", 2.0),
            formatted=_formatted_fields(data),
        )
    if kind == "response_select":
        path_raw = data.get("path")
        if not isinstance(path_raw, list) or not all(
            isinstance(part, (str, int)) and not isinstance(part, bool)
            for part in cast("list[object]", path_raw)
        ):
            raise ValueError("response selection path must be an array of strings or integers")
        return ResponseSelect(
            id=adapter_id,
            source=_string(data, "source"),
            path=tuple(cast("list[str | int]", path_raw)),
            output=_string(data, "output"),
            paths=_paths(data.get("paths", []), "paths"),
        )
    if kind == "local_progress":
        return LocalProgress(
            id=adapter_id,
            text=_string(data, "text"),
            step=_integer(data, "step", 0),
            total=_integer(data, "total", 1),
        )
    if kind == "submit_poll":

        def path(name: str, default: tuple[str | int, ...]) -> tuple[str | int, ...]:
            return _path(data.get(name, list(default)), name)

        def strings(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw_values = data.get(name, list(default))
            if not isinstance(raw_values, list):
                raise ValueError(f"adapter field {name!r} must be a string array")
            values = cast("list[object]", raw_values)
            if not all(isinstance(value, str) for value in values):
                raise ValueError(f"adapter field {name!r} must be a string array")
            return tuple(cast("list[str]", raw_values))

        return SubmitPoll(
            adapter_id,
            _string(data, "source"),
            path("url_path", ("polling_url",)),
            path("status_path", ("status",)),
            path("progress_path", ("progress",)),
            strings("completed", SubmitPoll.__dataclass_fields__["completed"].default),
            strings("failed", SubmitPoll.__dataclass_fields__["failed"].default),
            strings("queued", SubmitPoll.__dataclass_fields__["queued"].default),
            strings("allowed", ()),
            _number(data, "interval", 5),
            _integer(data, "max_attempts", 480),
            _number(data, "timeout", 120),
            _integer(data, "max_retries", 10),
            _number(data, "retry_delay", 1),
            _number(data, "retry_backoff", 1.4),
            (_string(data, "cancel_path") if data.get("cancel_path") is not None else None),
            _string(data, "path_template") if "path_template" in data else "",
            path("path_value_path", ()),
        )
    if kind == "proxy_upload":
        return ProxyUpload(
            adapter_id,
            _string(data, "input"),
            _string(data, "file_name"),
            _string(data, "content_type"),
            _string(data, "allocation_path"),
            _boolean(data, "batch", False),
            _boolean(data, "optional", False),
        )
    if kind == "encode_media":
        batch_targets_raw = data.get("batch_targets", [])
        source_path_raw = data.get("source_path", [])
        if not isinstance(batch_targets_raw, list) or not all(
            isinstance(value, str) for value in cast("list[object]", batch_targets_raw)
        ):
            raise ValueError("encode_media batch_targets must be a string array")
        if not isinstance(source_path_raw, list) or not all(
            isinstance(value, (str, int)) and not isinstance(value, bool)
            for value in cast("list[object]", source_path_raw)
        ):
            raise ValueError("encode_media source_path must contain strings or integers")
        return EncodeMedia(
            adapter_id,
            _string(data, "input"),
            cast("Literal['image', 'video', 'audio']", _string(data, "media_family")),
            _string(data, "format"),
            cast("Literal['bytes', 'base64', 'data_url']", _string(data, "output")),
            _optional_integer(data, "max_pixels"),
            _boolean(data, "rgb", False),
            _boolean(data, "optional", False),
            tuple(cast("list[str]", batch_targets_raw)),
            tuple(cast("list[str | int]", source_path_raw)),
        )
    if kind == "media_constraints":
        return MediaConstraints(
            adapter_id,
            _string(data, "input"),
            _integer(data, "min_width", 1),
            _integer(data, "min_height", 1),
            _optional_integer(data, "max_width"),
            _optional_integer(data, "max_height"),
            _optional_integer(data, "max_count"),
            _optional_integer(data, "max_bytes"),
            (
                _number(data, "min_aspect_ratio", 0)
                if data.get("min_aspect_ratio") is not None
                else None
            ),
            (
                _number(data, "max_aspect_ratio", 0)
                if data.get("max_aspect_ratio") is not None
                else None
            ),
            _optional_number(data, "min_duration"),
            _optional_number(data, "max_duration"),
            _boolean(data, "optional", False),
            cast("Literal['video', 'audio']", data.get("duration_media", "video")),
            _boolean(data, "batch", False),
            _boolean(data, "aspect_strict", False),
        )
    if kind == "mask_prepare":
        return MaskPrepare(adapter_id, _string(data, "mask_input"), _string(data, "image_input"))
    if kind == "download_decode":
        return DownloadDecode(
            adapter_id,
            _string(data, "source"),
            _path(data.get("url_path"), "url_path"),
            _string(data, "output"),
            cast(
                "Literal['image', 'video', 'audio', 'application']", _string(data, "media_family")
            ),
            _integer(data, "byte_cap", 1024 * 1024 * 1024),
            _path(data.get("items_path", []), "items_path"),
            _path(data.get("item_url_path", []), "item_url_path"),
            _paths(data.get("url_paths", []), "url_paths"),
        )
    if kind == "batch_map_join":
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ValueError("batch segments must be an array")
        segments: list[Segment] = []
        for raw_segment in cast("list[object]", raw_segments):
            if not isinstance(raw_segment, dict):
                raise ValueError("batch segments must be objects")
            segment = cast("dict[str, object]", raw_segment)
            fixed = segment.get("fixed", {})
            if not isinstance(fixed, dict):
                raise ValueError("segment fixed must be a string mapping")
            raw_fixed = cast("dict[object, object]", fixed)
            if not all(
                isinstance(key, str) and isinstance(value, str) for key, value in raw_fixed.items()
            ):
                raise ValueError("segment fixed must be a string mapping")
            typed_fixed = cast("dict[str, str]", fixed)
            wrap_key = segment.get("wrap_key", "url")
            if wrap_key is not None and not isinstance(wrap_key, str):
                raise ValueError("segment wrap_key must be a string or null")
            segments.append(
                Segment(
                    _string(segment, "source"),
                    typed_fixed,
                    wrap_key,
                    cast("Any", _string(segment, "mode")),
                )
            )
        return BatchMapJoin(
            adapter_id,
            _string(data, "source") if "source" in data else "",
            _string(data, "wrap_key") if "wrap_key" in data else "",
            tuple(segments),
            _optional_integer(data, "min_items"),
            _optional_integer(data, "max_items"),
            _string(data, "min_message")
            if "min_message" in data
            else "batch_map_join requires at least {count} items",
            _string(data, "max_message")
            if "max_message" in data
            else "batch_map_join has too many items: {count}",
        )
    return _simple_adapter(kind, adapter_id)


def _helper_binding_from_json(data: dict[str, object]) -> HelperBinding:
    return HelperBinding(
        name=_string(data, "name"),
        source=_string(data, "source"),
        source_kind=cast("Any", _string(data, "source_kind")),
        mode=cast("Any", _string(data, "mode")),
        source_path=_path(data.get("source_path", []), "source_path"),
        optional=_boolean(data, "optional", False),
    )


def _helper_call_from_json(data: dict[str, object]) -> HelperCall:
    bindings = data.get("bindings", [])
    fixed = data.get("fixed", {})
    outputs = data.get("outputs", [])
    if not isinstance(bindings, list) or not all(isinstance(item, dict) for item in bindings):
        raise ValueError("helper bindings must be objects in an array")
    if not isinstance(fixed, dict):
        raise ValueError("helper fixed must be an object")
    if not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs):
        raise ValueError("helper outputs must be a string array")
    return HelperCall(
        id=_string(data, "id"),
        helper_id=_string(data, "helper_id"),
        stage=_string(data, "stage"),
        placement=cast("Any", _string(data, "placement")),
        anchor=_string(data, "anchor"),
        bindings=tuple(
            _helper_binding_from_json(cast("dict[str, object]", item))
            for item in cast("list[object]", bindings)
        ),
        fixed=cast("dict[str, object]", fixed),
        outputs=tuple(cast("list[str]", outputs)),
    )


@dataclass(frozen=True)
class OpSpec:
    adapters: tuple[Adapter, ...]
    version: int = 1
    helper_calls: tuple[HelperCall, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapters", tuple(self.adapters))
        object.__setattr__(self, "helper_calls", tuple(self.helper_calls))
        if self.version != 1:
            raise ValueError(f"unsupported OpSpec version {self.version}")
        if not self.adapters:
            raise ValueError("OpSpec must contain at least one adapter")
        ids = [adapter.id for adapter in self.adapters]
        if len(ids) != len(set(ids)):
            raise ValueError("OpSpec adapter ids must be unique")
        helper_ids = [call.id for call in self.helper_calls]
        if len(helper_ids) != len(set(helper_ids)):
            raise ValueError("helper call ids must be unique")
        derived = [f"{call.id}.{output}" for call in self.helper_calls for output in call.outputs]
        if len(derived) != len(set(derived)):
            raise ValueError("helper derived output ids must be unique")
        if set(ids) & set(derived):
            raise ValueError("adapter ids must not collide with helper derived output ids")
        known: set[str] = set()
        network_seen = False
        for adapter in self.adapters:
            if isinstance(adapter, CheckInputs) and network_seen:
                raise ValueError("check_inputs adapters must precede network adapters")
            if isinstance(adapter, (HttpSyncJson, SubmitPoll, ProxyUpload, DownloadDecode)):
                network_seen = True
            if (
                isinstance(adapter, ResponseSelect)
                and adapter.source not in known
                and adapter.source not in derived
            ):
                raise ValueError(
                    f"response_select {adapter.id!r} references unknown earlier adapter "
                    f"{adapter.source!r}"
                )
            known.add(adapter.id)
        adapter_positions = {adapter_id: index for index, adapter_id in enumerate(ids)}
        call_positions = {call.id: index for index, call in enumerate(self.helper_calls)}
        call_events = {
            call.id: (
                adapter_positions[call.anchor],
                0 if call.placement == "before" else 2,
            )
            for call in self.helper_calls
            if call.anchor in adapter_positions
        }
        derived_sources = {
            f"{call.id}.{output}": call.id for call in self.helper_calls for output in call.outputs
        }
        for call in self.helper_calls:
            if call.placement == "select":
                raise ValueError("select helpers are only valid as Class3Spec selectors")
            if call.anchor not in adapter_positions:
                raise ValueError(
                    f"helper call {call.id!r} references unknown anchor {call.anchor!r}"
                )
            event = call_events[call.id]
            for binding in call.bindings:
                if binding.source_kind != "state":
                    continue
                if binding.source in adapter_positions:
                    source_event = (adapter_positions[binding.source], 1)
                    available = source_event < event
                elif binding.source in derived_sources:
                    source_call = derived_sources[binding.source]
                    source_event = call_events[source_call]
                    available = source_event < event or (
                        source_event == event
                        and call_positions[source_call] < call_positions[call.id]
                    )
                else:
                    available = False
                if not available:
                    raise ValueError(
                        f"helper call {call.id!r} references unavailable state source "
                        f"{binding.source!r}"
                    )

    def to_json(self) -> str:
        return json.dumps(_json_value(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> OpSpec:
        decoded: object = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("OpSpec JSON must be an object")
        raw = cast("dict[str, object]", decoded)
        if set(raw) not in ({"adapters", "version"}, {"adapters", "helper_calls", "version"}):
            raise ValueError("OpSpec JSON contains unknown or missing fields")
        adapters_raw = raw["adapters"]
        if not isinstance(adapters_raw, list):
            raise ValueError("OpSpec adapters must be a JSON array")
        adapters: list[Adapter] = []
        for item in cast("list[object]", adapters_raw):
            if not isinstance(item, dict):
                raise ValueError("each adapter must be a JSON object")
            adapters.append(_adapter_from_json(cast("dict[str, object]", item)))
        version = raw["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("OpSpec version must be an integer")
        helper_raw = raw.get("helper_calls", [])
        if not isinstance(helper_raw, list) or not all(
            isinstance(item, dict) for item in cast("list[object]", helper_raw)
        ):
            raise ValueError("OpSpec helper_calls must be a JSON array of objects")
        return cls(
            adapters=tuple(adapters),
            version=version,
            helper_calls=tuple(
                _helper_call_from_json(cast("dict[str, object]", item))
                for item in cast("list[object]", helper_raw)
            ),
        )


@dataclass(frozen=True)
class Class3Spec:
    selector: HelperCall
    variants: Mapping[str, OpSpec]
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported Class3Spec version {self.version}")
        if self.selector.placement != "select":
            raise ValueError("Class3Spec selector must use select placement")
        if not self.variants:
            raise ValueError("Class3Spec variants must not be empty")
        if any(not key for key in self.variants):
            raise ValueError("Class3Spec variant names must be non-empty strings")
        copied = dict(sorted(self.variants.items()))
        selector_outputs = {f"{self.selector.id}.{key}" for key in self.selector.outputs}

        def selector_references(value: object) -> set[str]:
            if isinstance(value, str):
                return {value} if value.startswith(f"{self.selector.id}.") else set()
            if isinstance(value, Mapping):
                return {
                    reference for item in value.values() for reference in selector_references(item)
                }
            if isinstance(value, (list, tuple)):
                return {reference for item in value for reference in selector_references(item)}
            return set()

        for name, variant in copied.items():
            adapter_ids = {adapter.id for adapter in variant.adapters}
            helper_outputs = {
                f"{call.id}.{key}" for call in variant.helper_calls for key in call.outputs
            }
            collisions = selector_outputs & (adapter_ids | helper_outputs)
            if collisions:
                raise ValueError(
                    f"Class3Spec variant {name!r} collides with selector outputs: "
                    f"{sorted(collisions)!r}"
                )
            unknown = selector_references(_json_value(variant)) - selector_outputs
            if unknown:
                raise ValueError(
                    f"Class3Spec variant {name!r} references unknown selector outputs: "
                    f"{sorted(unknown)!r}"
                )
        object.__setattr__(self, "variants", MappingProxyType(copied))

    def to_json(self) -> str:
        return json.dumps(_json_value(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Class3Spec:
        raw = json.loads(payload)
        if not isinstance(raw, dict) or set(raw) != {"selector", "variants", "version"}:
            raise ValueError("Class3Spec JSON must contain selector, variants, and version")
        selector = raw["selector"]
        variants = raw["variants"]
        if not isinstance(selector, dict) or not isinstance(variants, dict):
            raise ValueError("Class3Spec selector and variants must be objects")
        return cls(
            selector=_helper_call_from_json(cast("dict[str, object]", selector)),
            variants=MappingProxyType(
                {
                    str(key): OpSpec.from_json(json.dumps(value))
                    for key, value in cast("dict[str, object]", variants).items()
                }
            ),
            version=_integer(cast("dict[str, object]", raw), "version", 1),
        )
