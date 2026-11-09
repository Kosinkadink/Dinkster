"""Conversion between public multi-stream latents and sampler runtime values."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dinkster_api.v1 import LatentStream, MultiStreamLatent


def to_runtime(value: object) -> object:
    if type(value) is not MultiStreamLatent:
        return value
    streams = cast("MultiStreamLatent[object]", value)
    nested_tensor = importlib.import_module("dinkster_inference.nested_tensor")
    return nested_tensor.NestedTensor(stream.payload for stream in streams.streams)


def to_public(value: object, template: object) -> object:
    if type(template) is not MultiStreamLatent:
        return value
    streams = cast("MultiStreamLatent[object]", template)
    payloads = cast(Any, value).unbind()
    return MultiStreamLatent(
        tuple(
            LatentStream(stream.role, payload)
            for stream, payload in zip(streams.streams, payloads, strict=True)
        )
    )
