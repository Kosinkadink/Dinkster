"""Public seam for consuming a loaded codec as a distinct resource.

A live codec value is not the model handle: consumers get codec identity
and geometry (the descriptor), an opaque device token, the lease
lifecycle, and encode/decode - nothing else. Tiled execution is exposed by
a separate optional protocol so ordinary consumers cannot reach through
to the runtime that may back the codec.

Callers transfer inputs and execute inside the lease::

    codec = require_inference_codec_handle(value, "vae")
    with codec.stage():
        content = codec.decode_latent(latent)
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from .codecs import CodecDescriptor
from .patches import SizedTensor
from .runtime_handle import InferenceRuntimeHandle, require_inference_runtime_handle

CodecTensorT = TypeVar("CodecTensorT", bound=SizedTensor)


@runtime_checkable
class _CodecRuntime(Protocol[CodecTensorT]):
    def decode_latent(self, latent: CodecTensorT) -> CodecTensorT: ...

    def encode_content(self, content: CodecTensorT) -> CodecTensorT: ...


@runtime_checkable
class InferenceCodecHandle(Protocol[CodecTensorT]):
    """One resident codec: descriptor identity plus leased encode/decode."""

    @property
    def descriptor(self) -> CodecDescriptor:
        """The codec's identity, latent geometry, and tiling declarations."""
        ...

    @property
    def resource_identity(self) -> str:
        """Stable content identity for admission and cache preimages."""
        ...

    @property
    def load_device(self) -> object:
        """Opaque backend device token accepted by tensors used with the
        codec; never None."""
        ...

    def require_active(self) -> None:
        """Raise if the codec's residency was terminally released."""
        ...

    def stage(self) -> AbstractContextManager[None]:
        """Lease the codec for one execution."""
        ...

    def decode_latent(self, latent: CodecTensorT) -> CodecTensorT:
        """Latent -> content; call inside :meth:`stage`."""
        ...

    def encode_content(self, content: CodecTensorT) -> CodecTensorT:
        """Content -> latent; call inside :meth:`stage`."""
        ...


@runtime_checkable
class InferenceTiledCodecHandle(InferenceCodecHandle[CodecTensorT], Protocol[CodecTensorT]):
    """A codec whose implementation accepts explicit tile geometry."""

    def decode_latent_tiled(
        self,
        latent: CodecTensorT,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> CodecTensorT:
        """Tiled latent -> content; geometry uses latent units."""
        ...

    def encode_content_tiled(
        self,
        content: CodecTensorT,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> CodecTensorT:
        """Tiled content -> latent; geometry uses content units."""
        ...


def require_inference_codec_handle(
    value: object, input_id: str
) -> InferenceCodecHandle[SizedTensor]:
    """Fail-closed narrowing of one node input to a codec handle."""

    if not isinstance(value, InferenceCodecHandle):
        raise TypeError(f"{input_id} must be a codec handle, got {type(value).__name__}")
    for name in ("require_active", "stage", "decode_latent", "encode_content"):
        if not callable(getattr(cast("object", value), name)):
            raise TypeError(f"{input_id} {name} must be callable")
    value.require_active()
    descriptor = value.descriptor
    if not isinstance(descriptor, CodecDescriptor):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{input_id} descriptor must be a CodecDescriptor")
    identity = cast("object", value.resource_identity)
    if not isinstance(identity, str) or not identity:
        raise TypeError(f"{input_id} resource identity must be a non-empty string")
    if value.load_device is None:
        raise TypeError(f"{input_id} load device must not be None")
    return cast("InferenceCodecHandle[SizedTensor]", value)


def require_inference_tiled_codec_handle(
    value: object, input_id: str
) -> InferenceTiledCodecHandle[SizedTensor]:
    """Fail-closed narrowing to a codec with explicit tiled execution."""

    codec = require_inference_codec_handle(value, input_id)
    if not codec.descriptor.supports_tiling:
        raise TypeError(f"{input_id} codec does not support tiled execution")
    if not isinstance(value, InferenceTiledCodecHandle):
        raise TypeError(f"{input_id} must support tiled codec execution")
    for name in ("decode_latent_tiled", "encode_content_tiled"):
        if not callable(getattr(cast("object", value), name)):
            raise TypeError(f"{input_id} {name} must be callable")
    return cast("InferenceTiledCodecHandle[SizedTensor]", value)


class RuntimeCodecAdapter(Generic[CodecTensorT]):
    """Publish a runtime handle's codec as a distinct codec resource.

    The adapter shares the handle's residency allocation as an
    implementation detail; consumers hold a separate object with no public
    path back to the model handle, and no identity or equality semantics.
    """

    __slots__ = ("_handle", "_descriptor", "_resource_identity")

    def __init__(
        self,
        handle: InferenceRuntimeHandle,
        *,
        descriptor: CodecDescriptor,
    ) -> None:
        require_inference_runtime_handle(handle, "handle")
        if not isinstance(descriptor, CodecDescriptor):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"descriptor must be a CodecDescriptor, got {type(descriptor).__name__}"
            )
        self._handle = handle
        self._descriptor = descriptor
        self._resource_identity = handle.recipe.runtime_identity

    @property
    def _dinkster_resident_owner(self) -> InferenceRuntimeHandle:
        return self._handle

    @property
    def descriptor(self) -> CodecDescriptor:
        return self._descriptor

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def load_device(self) -> object:
        return self._handle.load_device

    @property
    def accepts_batched_video(self) -> bool:
        return bool(
            getattr(getattr(self._runtime(), "codec", None), "accepts_batched_video", False)
        )

    @property
    def accepts_image_batch_latent(self) -> bool:
        return bool(
            getattr(getattr(self._runtime(), "codec", None), "accepts_image_batch_latent", False)
        )

    @property
    def manages_input_device(self) -> bool:
        return bool(getattr(getattr(self._runtime(), "codec", None), "manages_input_device", False))

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self) -> AbstractContextManager[None]:
        return self._handle.stage("vae")

    def decode_latent(self, latent: CodecTensorT) -> CodecTensorT:
        return self._runtime().decode_latent(latent)

    def decode_latent_tiled(
        self,
        latent: CodecTensorT,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> CodecTensorT:
        runtime = cast("object", self._runtime())
        codec = getattr(runtime, "codec", None)
        decode = getattr(codec, "decode_tiled", None)
        if not callable(decode):
            raise TypeError("runtime does not expose tiled codec execution")
        return cast(
            "CodecTensorT",
            decode(latent, tile=tile, overlap=overlap),
        )

    def encode_content(self, content: CodecTensorT) -> CodecTensorT:
        return self._runtime().encode_content(content)

    def encode_content_tiled(
        self,
        content: CodecTensorT,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> CodecTensorT:
        runtime = cast("object", self._runtime())
        codec = getattr(runtime, "codec", None)
        encode = getattr(codec, "encode_tiled", None)
        if not callable(encode):
            raise TypeError("runtime does not expose tiled codec execution")
        return cast("CodecTensorT", encode(content, tile=tile, overlap=overlap))

    def _runtime(self) -> _CodecRuntime[CodecTensorT]:
        # Re-read per call: the runtime raises once the handle is released
        # or its materialization was dropped, and must never be cached.
        runtime = self._handle.runtime
        if not isinstance(runtime, _CodecRuntime):
            raise TypeError("runtime does not expose codec execution")
        for name in ("decode_latent", "encode_content"):
            if not callable(getattr(cast("object", runtime), name)):
                raise TypeError(f"runtime {name} must be callable")
        return cast("_CodecRuntime[CodecTensorT]", runtime)


__all__ = [
    "InferenceCodecHandle",
    "InferenceTiledCodecHandle",
    "RuntimeCodecAdapter",
    "require_inference_codec_handle",
    "require_inference_tiled_codec_handle",
]
