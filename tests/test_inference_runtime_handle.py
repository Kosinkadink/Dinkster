"""Runtime-handle and codec-handle seam contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from dinkster_inference import (
    FLOAT32,
    CodecDescriptor,
    Conditioning,
    InferenceCodecHandle,
    InferenceRuntimeHandle,
    InferenceTiledCodecHandle,
    LatentDescriptor,
    ReconstructionRecipe,
    RuntimeCodecAdapter,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    require_inference_codec_handle,
    require_inference_runtime_handle,
    require_inference_tiled_codec_handle,
)


def _recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef(
                    digest="blake3:" + "0" * 64,
                    name="weights.safetensors",
                    size=1,
                ),
            ),
        ),
        family_id="dinkster.sd15",
        component_identity=("family=dinkster.sd15", "component=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float16",
            text_dtype="float32",
            vae_dtype="float32",
            fp8_matmul=False,
        ),
    )


class _FakeRuntime:
    def __init__(self, runtime_identity: str) -> None:
        self.runtime_identity = runtime_identity
        self.family = object()
        self.decoded: list[object] = []
        self.encoded: list[object] = []

    def encode_text(self, text: str) -> Conditioning[Any]:
        raise NotImplementedError

    def sample(self, latent: object, **kwargs: object) -> object:
        raise NotImplementedError

    def decode_latent(self, latent: object) -> object:
        self.decoded.append(latent)
        return ("content", latent)

    def encode_content(self, content: object) -> object:
        self.encoded.append(content)
        return ("latent", content)


class _FakeMultiStreamCodecRuntime:
    def __init__(self, runtime_identity: str) -> None:
        self.runtime_identity = runtime_identity
        self.decoded: list[object] = []
        self.encoded: list[object] = []

    def sample_multistream(self, latent: object, **kwargs: object) -> object:
        raise NotImplementedError

    def decode_latent(self, latent: object) -> object:
        self.decoded.append(latent)
        return ("content", latent)

    def encode_content(self, content: object) -> object:
        self.encoded.append(content)
        return ("latent", content)


class _FakeCustomSamplingRuntime:
    def __init__(self, runtime_identity: str) -> None:
        self.runtime_identity = runtime_identity
        self.family = object()

    def custom_sampling_sigmas(
        self, scheduler_id: str, steps: int, denoise: float
    ) -> tuple[float, ...]:
        raise NotImplementedError

    def custom_sampling_beta_sigmas(
        self, steps: int, alpha: float, beta: float
    ) -> tuple[float, ...]:
        raise NotImplementedError

    def custom_sampling_sd_turbo_sigmas(self, steps: int, denoise: float) -> tuple[float, ...]:
        raise NotImplementedError

    def custom_sampling_percent_to_sigma(
        self, percent: float, *, return_actual_sigma: bool
    ) -> float:
        raise NotImplementedError

    def check_custom_sampling(self, request: object, **kwargs: object) -> None:
        raise NotImplementedError

    def sample_custom(self, latent: object, **kwargs: object) -> object:
        raise NotImplementedError


class _ReleasedError(RuntimeError):
    pass


class _FakeHandle:
    def __init__(self, recipe: ReconstructionRecipe, runtime: object) -> None:
        self._recipe = recipe
        self._runtime = runtime
        self.load_device = "cpu"
        self.released = False
        self.staged: list[str] = []

    @property
    def runtime(self) -> Any:
        self.require_active()
        return self._runtime

    @property
    def recipe(self) -> ReconstructionRecipe:
        return self._recipe

    def require_active(self) -> None:
        if self.released:
            raise _ReleasedError("handle was terminally released")

    @contextmanager
    def stage(self, role: str) -> Iterator[None]:
        self.staged.append(role)
        yield


def _handle() -> _FakeHandle:
    recipe = _recipe()
    return _FakeHandle(recipe, _FakeRuntime(recipe.runtime_identity))


def test_runtime_handle_accepts_matching_handle() -> None:
    handle = _handle()
    assert isinstance(handle, InferenceRuntimeHandle)
    assert require_inference_runtime_handle(handle, "model") is handle


def test_runtime_handle_accepts_custom_sampling_only_runtime() -> None:
    recipe = _recipe()
    handle = _FakeHandle(recipe, _FakeCustomSamplingRuntime(recipe.runtime_identity))
    assert require_inference_runtime_handle(handle, "model") is handle


def test_runtime_handle_rejects_foreign_objects() -> None:
    with pytest.raises(TypeError, match="model must be a native runtime handle"):
        require_inference_runtime_handle(object(), "model")


def test_runtime_handle_rejects_non_callable_protocol_member() -> None:
    handle = _handle()
    handle.stage = 1  # type: ignore[assignment] # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match="stage must be callable"):
        require_inference_runtime_handle(handle, "model")


def test_runtime_handle_propagates_released_lifecycle() -> None:
    handle = _handle()
    handle.released = True
    with pytest.raises(_ReleasedError):
        require_inference_runtime_handle(handle, "model")


def test_runtime_handle_rejects_non_recipe() -> None:
    handle = _handle()
    handle._recipe = "not-a-recipe"  # type: ignore[assignment] # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match="recipe must be a ReconstructionRecipe"):
        require_inference_runtime_handle(handle, "model")


def test_runtime_handle_rejects_non_family_runtime() -> None:
    handle = _FakeHandle(_recipe(), object())
    with pytest.raises(TypeError, match="family runtime protocol"):
        require_inference_runtime_handle(handle, "model")


def test_runtime_handle_rejects_empty_identity() -> None:
    handle = _FakeHandle(_recipe(), _FakeRuntime(""))
    with pytest.raises(TypeError, match="non-empty string"):
        require_inference_runtime_handle(handle, "model")


def test_runtime_handle_rejects_identity_mismatch() -> None:
    handle = _FakeHandle(_recipe(), _FakeRuntime("someone-else"))
    with pytest.raises(ValueError, match="does not match its reconstruction recipe"):
        require_inference_runtime_handle(handle, "model")


def _descriptor() -> CodecDescriptor:
    return CodecDescriptor(
        id="dinkster.test_vae",
        display_name="Test VAE",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({FLOAT32}),
    )


def test_codec_adapter_satisfies_codec_protocol() -> None:
    adapter = RuntimeCodecAdapter(_handle(), descriptor=_descriptor())
    assert isinstance(adapter, InferenceCodecHandle)
    assert require_inference_codec_handle(adapter, "vae") is adapter


def test_codec_adapter_satisfies_tiled_codec_protocol_when_declared() -> None:
    descriptor = CodecDescriptor(
        id="dinkster.test_tiled_vae",
        display_name="Test tiled VAE",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({FLOAT32}),
        supports_tiling=True,
    )
    adapter = RuntimeCodecAdapter(_handle(), descriptor=descriptor)
    assert isinstance(adapter, InferenceTiledCodecHandle)
    assert require_inference_tiled_codec_handle(adapter, "vae") is adapter


def test_tiled_codec_handle_requires_declared_and_exposed_support() -> None:
    untiled = CodecDescriptor(
        id="dinkster.test_untiled_vae",
        display_name="Test untiled VAE",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({FLOAT32}),
        supports_tiling=False,
    )
    adapter = RuntimeCodecAdapter(_handle(), descriptor=untiled)
    with pytest.raises(TypeError, match="does not support tiled execution"):
        require_inference_tiled_codec_handle(adapter, "vae")

    class DirectOnly:
        descriptor = _descriptor()
        resource_identity = "native:dinkster.sd15:" + "1" * 64
        load_device = "cpu"

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Iterator[None]:
            yield

        def decode_latent(self, latent: object) -> object:
            return latent

        def encode_content(self, content: object) -> object:
            return content

    with pytest.raises(TypeError, match="must support tiled codec execution"):
        require_inference_tiled_codec_handle(DirectOnly(), "vae")


def test_codec_adapter_leases_and_delegates() -> None:
    handle = _handle()
    descriptor = _descriptor()
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=descriptor)
    assert adapter.descriptor is descriptor
    assert adapter.resource_identity == handle.recipe.runtime_identity
    assert adapter._dinkster_resident_owner is handle  # pyright: ignore[reportPrivateUsage]
    assert adapter.load_device == "cpu"
    with adapter.stage():
        assert handle.staged == ["vae"]
        assert adapter.decode_latent("latent") == ("content", "latent")
        assert adapter.encode_content("content") == ("latent", "content")


def test_codec_adapter_never_caches_the_runtime() -> None:
    handle = _handle()
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    assert adapter.decode_latent("before") == ("content", "before")
    handle.released = True
    with pytest.raises(_ReleasedError):
        adapter.decode_latent("after")
    with pytest.raises(_ReleasedError):
        adapter.require_active()


def test_codec_adapter_reads_layout_and_transfer_capabilities_from_live_codec() -> None:
    handle = _handle()
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    assert not adapter.accepts_batched_video
    assert not adapter.accepts_image_batch_latent
    assert not adapter.manages_input_device
    handle.runtime.codec = SimpleNamespace(
        accepts_batched_video=True,
        accepts_image_batch_latent=True,
        manages_input_device=False,
    )
    assert adapter.accepts_batched_video
    assert adapter.accepts_image_batch_latent
    assert not adapter.manages_input_device
    handle.runtime.codec = SimpleNamespace(
        accepts_batched_video=False,
        accepts_image_batch_latent=False,
        manages_input_device=True,
    )
    assert not adapter.accepts_batched_video
    assert not adapter.accepts_image_batch_latent
    assert adapter.manages_input_device
    handle.released = True
    with pytest.raises(_ReleasedError):
        _ = adapter.accepts_batched_video
    with pytest.raises(_ReleasedError):
        _ = adapter.accepts_image_batch_latent
    with pytest.raises(_ReleasedError):
        _ = adapter.manages_input_device


def test_codec_adapter_captures_resource_identity_at_construction() -> None:
    handle = _handle()
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    identity = adapter.resource_identity

    handle._recipe = "changed"  # type: ignore[assignment] # pyright: ignore[reportAttributeAccessIssue]

    assert adapter.resource_identity == identity


def test_codec_adapter_delegates_to_multistream_runtime_codec() -> None:
    recipe = _recipe()
    runtime = _FakeMultiStreamCodecRuntime(recipe.runtime_identity)
    handle = _FakeHandle(recipe, runtime)
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())

    with adapter.stage():
        assert adapter.decode_latent("latent") == ("content", "latent")
        assert adapter.encode_content("content") == ("latent", "content")

    assert handle.staged == ["vae"]
    assert runtime.decoded == ["latent"]
    assert runtime.encoded == ["content"]


def test_codec_adapter_rejects_runtime_without_codec_execution() -> None:
    recipe = _recipe()

    class _SamplerOnly:
        runtime_identity = recipe.runtime_identity

        def sample_multistream(self, latent: object, **kwargs: object) -> object:
            raise NotImplementedError

    handle = _FakeHandle(recipe, _SamplerOnly())
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    with pytest.raises(TypeError, match="does not expose codec execution"):
        adapter.decode_latent("latent")


def test_codec_adapter_rejects_non_callable_codec_execution() -> None:
    recipe = _recipe()

    class _BadCodec:
        runtime_identity = recipe.runtime_identity
        decode_latent = 1

        def sample_multistream(self, latent: object, **kwargs: object) -> object:
            raise NotImplementedError

        def encode_content(self, content: object) -> object:
            return content

    handle = _FakeHandle(recipe, _BadCodec())
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    with pytest.raises(TypeError, match="decode_latent must be callable"):
        adapter.decode_latent("latent")


def test_codec_adapter_validates_construction() -> None:
    with pytest.raises(TypeError, match="native runtime handle"):
        RuntimeCodecAdapter(object(), descriptor=_descriptor())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CodecDescriptor"):
        RuntimeCodecAdapter(_handle(), descriptor="vae")  # type: ignore[arg-type]


def test_codec_handle_rejects_foreign_objects() -> None:
    with pytest.raises(TypeError, match="vae must be a codec handle"):
        require_inference_codec_handle(object(), "vae")


def test_codec_handle_rejects_non_callable_protocol_member() -> None:
    class _BadDecode:
        descriptor = _descriptor()
        resource_identity = "native:dinkster.sd15:" + "1" * 64
        load_device = "cpu"
        decode_latent = 1

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Iterator[None]:
            yield

        def encode_content(self, content: object) -> object:
            return content

    with pytest.raises(TypeError, match="decode_latent must be callable"):
        require_inference_codec_handle(_BadDecode(), "vae")


def test_codec_handle_rejects_non_descriptor() -> None:
    class _BadDescriptor:
        descriptor = "not-a-descriptor"
        resource_identity = "native:dinkster.sd15:" + "1" * 64
        load_device = "cpu"

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Iterator[None]:
            yield

        def decode_latent(self, latent: object) -> object:
            return latent

        def encode_content(self, content: object) -> object:
            return content

    with pytest.raises(TypeError, match="descriptor must be a CodecDescriptor"):
        require_inference_codec_handle(_BadDescriptor(), "vae")


def test_codec_handle_rejects_empty_resource_identity() -> None:
    class _BadIdentity:
        descriptor = _descriptor()
        resource_identity = ""
        load_device = "cpu"

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Iterator[None]:
            yield

        def decode_latent(self, latent: object) -> object:
            return latent

        def encode_content(self, content: object) -> object:
            return content

    with pytest.raises(TypeError, match="resource identity must be a non-empty string"):
        require_inference_codec_handle(_BadIdentity(), "vae")


def test_codec_handle_rejects_none_device() -> None:
    handle = _handle()
    handle.load_device = None  # type: ignore[assignment]
    adapter: RuntimeCodecAdapter[Any] = RuntimeCodecAdapter(handle, descriptor=_descriptor())
    with pytest.raises(TypeError, match="load device must not be None"):
        require_inference_codec_handle(adapter, "vae")
