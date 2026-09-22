"""TripoSplat nodes owned by the isolated native worker."""

from __future__ import annotations

import threading
from collections.abc import Mapping

from dinkster_model_triposplat.nodes import (  # pyright: ignore[reportMissingTypeStubs]
    LoadTripoSplatDecoder,
    LoadTripoSplatVisionEncoder,
    TripoSplatConditioning,
    TripoSplatDecode,
    TripoSplatPreprocessImage,
)

from ..native_residency import NativeComponentPublisher

_component_publisher_instance: NativeComponentPublisher | None = None
_component_publisher_lock = threading.Lock()


def _component_publisher() -> NativeComponentPublisher:
    global _component_publisher_instance  # noqa: PLW0603 - isolated worker lifetime
    if _component_publisher_instance is None:
        with _component_publisher_lock:
            if _component_publisher_instance is None:
                _component_publisher_instance = NativeComponentPublisher()
    return _component_publisher_instance


class NativeLoadTripoSplatVisionEncoder(LoadTripoSplatVisionEncoder):
    @classmethod
    def execute(cls, vision_encoder: object) -> Mapping[str, object]:
        from dinkster_model_triposplat.provider import (  # pyright: ignore[reportMissingTypeStubs]
            execute_load_triposplat_vision_encoder,
        )

        return execute_load_triposplat_vision_encoder(
            vision_encoder=vision_encoder,
            publisher=_component_publisher(),
        )


class NativeLoadTripoSplatDecoder(LoadTripoSplatDecoder):
    @classmethod
    def execute(cls, decoder: object) -> Mapping[str, object]:
        from dinkster_model_triposplat.provider import (  # pyright: ignore[reportMissingTypeStubs]
            execute_load_triposplat_decoder,
        )

        return execute_load_triposplat_decoder(
            decoder=decoder,
            publisher=_component_publisher(),
        )


TRIPOSPLAT_NATIVE_NODES = (
    NativeLoadTripoSplatVisionEncoder,
    NativeLoadTripoSplatDecoder,
    TripoSplatPreprocessImage,
    TripoSplatDecode,
)

__all__ = [
    "NativeLoadTripoSplatDecoder",
    "NativeLoadTripoSplatVisionEncoder",
    "TRIPOSPLAT_NATIVE_NODES",
    "TripoSplatConditioning",
]
