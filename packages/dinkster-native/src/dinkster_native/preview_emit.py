"""Throttled sampling-preview emission for the native arm.

One :class:`SamplingPreviewEmitter` serves one sampling call: it receives
solver state events, decodes the current denoised estimate with the
provider resolved for this model's latent space, and ships a small JPEG
through ``report_preview``. Cost stays bounded by construction:

- nothing is built at all when the invocation's preview mode is ``off``
  (the sampler never installs a state callback);
- frames are rate-limited (first step always, then at most one per
  ``min_interval`` seconds);
- decode runs synchronously in the sampling thread (the only place the
  state tensor is safely visible), while JPEG encode and event emission
  run on a single background slot - when that slot is still busy, the
  frame is dropped, never queued.

Previews are droppable side effects: every failure here degrades to "no
preview", never to a failed node.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import io
import threading
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import TracebackType
from typing import Any, cast

from dinkster_assets import declared_asset
from dinkster_inference import (
    BFLOAT16,
    TAEHV_PROVIDER,
    TAESD_PROVIDER,
    TRIPOSPLAT_CONFIG,
    TRIPOSPLAT_SPLAT_PROVIDER,
    EncodedPreviewAnimation,
    LatentDescriptor,
    MultiStreamLatent,
    MultiStreamLatentDescriptor,
    PreviewClip,
    PreviewFrame,
    builtin_families,
    builtin_preview_registry,
    load_safetensors_header,
    plan_taehv_decoder,
    plan_taesd_decoder,
    plan_triposplat_split_component,
    triposplat_component_runtime_identity,
)
from dinkster_protocol import PreviewAnimation, PreviewMode
from dinkster_schema import report_log, report_preview
from dinkster_workers import current_execution_context

from .native_residency import NativeComponentHandle, default_native_residency
from .pool import default_pool
from .resident import resident_resource_id

PREVIEW_MIN_INTERVAL_S = 0.2
PREVIEW_JPEG_QUALITY = 85
PREVIEW_WEBP_QUALITY = 85
PREVIEW_WEBP_FALLBACK_FPS = 8.0
PREVIEW_DECODE_FAILURE_LIMIT = 3


def _encode_jpeg(frame: PreviewFrame) -> bytes:
    pillow = cast("Any", importlib.import_module("PIL.Image"))
    buffer = io.BytesIO()
    pillow.fromarray(cast("Any", frame.rgb), mode="RGB").save(
        buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY
    )
    return buffer.getvalue()


def _encode_webp_animation(clip: PreviewClip) -> EncodedPreviewAnimation:
    """One decoded clip as a self-contained looping animated WebP.

    The container encode (Pillow ``save_all`` with a uniform per-frame
    duration, matching KJNodes' animated-preview transport) runs on the
    emitter's background slot, never the sampling thread."""
    pillow = cast("Any", importlib.import_module("PIL.Image"))
    images = [pillow.fromarray(cast("Any", frame.rgb), mode="RGB") for frame in clip.frames]
    fps = clip.fps if clip.fps is not None else PREVIEW_WEBP_FALLBACK_FPS
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        quality=PREVIEW_WEBP_QUALITY,
        method=4,
    )
    first = clip.frames[0]
    return EncodedPreviewAnimation(
        data=buffer.getvalue(), mime="image/webp", width=first.width, height=first.height
    )


class _BestEffortStage:
    """A residency stage whose own entry and exit failures both degrade
    to unstaged decoding. A sampling-body exception always propagates:
    it is handed to the inner stage's teardown but never swallowed here."""

    def __init__(self, factory: Callable[[], AbstractContextManager[object]]) -> None:
        self._factory = factory
        self._active: AbstractContextManager[object] | None = None

    def __enter__(self) -> _BestEffortStage:
        try:
            candidate = self._factory()
            candidate.__enter__()
        except Exception:
            return self
        self._active = candidate
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        active, self._active = self._active, None
        if active is not None:
            with contextlib.suppress(Exception):
                active.__exit__(exc_type, exc, traceback)
        return False


class SamplingPreviewEmitter:
    """Rate-limited pump from solver state events to preview events."""

    def __init__(
        self,
        decode: Callable[[object], PreviewFrame | PreviewClip | EncodedPreviewAnimation],
        *,
        stream_role: str | None = None,
        min_interval: float = PREVIEW_MIN_INTERVAL_S,
        encode: Callable[[PreviewFrame], bytes] = _encode_jpeg,
        encode_animation: Callable[[PreviewClip], EncodedPreviewAnimation] | None = None,
        emit: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        stage: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        self._decode = decode
        self._stream_role = stream_role
        self._min_interval = min_interval
        self._encode = encode
        # When set, a decoded clip becomes one self-contained animation on
        # the background slot instead of per-frame ring stills.
        self._encode_animation = encode_animation
        self._emit = emit if emit is not None else _report_preview_frame
        self._clock = clock
        self._stage_factory = stage
        # report_preview reads an ambient reporter contextvar that raw
        # threads never see; the background encode runs under a copy of
        # the sampling thread's context to keep the reporter visible.
        self._context = contextvars.copy_context()
        self._slot = threading.Lock()
        self._last_sent: float | None = None
        self._decode_failures = 0

    def stage(self) -> AbstractContextManager[object]:
        """The residency stage keeping this emitter's decoder weights
        placed for the sampling call. Best-effort: a failure entering or
        exiting the stage degrades to unstaged decode (whose failures
        on_state swallows), never to a failed node."""
        if self._stage_factory is None:
            return nullcontext()
        return _BestEffortStage(self._stage_factory)

    def on_state(self, event: Any) -> None:
        try:
            if self._decode_failures >= PREVIEW_DECODE_FAILURE_LIMIT:
                return
            now = self._clock()
            if self._last_sent is not None and now - self._last_sent < self._min_interval:
                return
            if not self._slot.acquire(blocking=False):
                return
            handed_off = False
            try:
                state = event.denoised if event.denoised is not None else event.current
                if type(state) is MultiStreamLatent:
                    if self._stream_role is None:
                        return
                    streams = cast("MultiStreamLatent[Any]", state)
                    if self._stream_role not in streams.roles:
                        return
                    state = streams.by_role(self._stream_role)
                try:
                    frame = self._decode(state)
                except Exception:
                    # A persistently failing decoder (an OOM-thrashing
                    # quality decode) stops being retried; occasional
                    # failures recover on the next state event.
                    self._decode_failures += 1
                    return
                self._decode_failures = 0
                worker = threading.Thread(
                    target=self._encode_and_report,
                    args=(frame,),
                    name="dinkster-preview-encode",
                    daemon=True,
                )
                # Only a started thread owns the slot: a start() failure
                # must release it here, not leak it forever.
                worker.start()
                handed_off = True
                self._last_sent = now
            finally:
                if not handed_off:
                    self._slot.release()
        except Exception:
            # A preview must never fail the sampling it observes.
            return

    def _encode_and_report(
        self, item: PreviewFrame | PreviewClip | EncodedPreviewAnimation
    ) -> None:
        try:
            if self._encode_animation is not None and type(item) is PreviewClip:
                item = self._encode_animation(item)
            if type(item) is EncodedPreviewAnimation:
                # A self-contained animation ships its container bytes
                # as-is; re-encoding would discard the animation.
                self._context.copy().run(
                    self._emit,
                    item.data,
                    item.mime,
                    item.width,
                    item.height,
                    stream=self._stream_role,
                    frame_index=None,
                    frame_count=None,
                    fps=None,
                )
                return
            if type(item) is PreviewClip:
                entries: tuple[tuple[PreviewFrame, int | None], ...] = tuple(
                    zip(item.frames, item.frame_indices, strict=True)
                )
                frame_count: int | None = item.frame_count
                fps: float | None = item.fps
            else:
                entries = ((cast("PreviewFrame", item), None),)
                frame_count = None
                fps = None
            for frame, index in entries:
                data = self._encode(frame)
                self._context.copy().run(
                    self._emit,
                    data,
                    "image/jpeg",
                    frame.width,
                    frame.height,
                    stream=self._stream_role,
                    frame_index=index,
                    frame_count=frame_count,
                    fps=fps,
                )
        except Exception:
            return
        finally:
            self._slot.release()


def _report_preview_frame(
    data: bytes,
    mime: str,
    width: int,
    height: int,
    *,
    stream: str | None = None,
    frame_index: int | None = None,
    frame_count: int | None = None,
    fps: float | None = None,
) -> None:
    report_preview(
        data,
        mime=mime,
        width=width,
        height=height,
        stream=stream,
        frame_index=frame_index,
        frame_count=frame_count,
        fps=fps,
    )


def _stream_descriptor(
    latent: LatentDescriptor | MultiStreamLatentDescriptor,
    stream_role: str | None,
) -> tuple[LatentDescriptor, str | None] | None:
    # A plain descriptor keeps the caller's role: single-stream families
    # (Wan video) still wrap their sampling states in a one-role
    # MultiStreamLatent, which the emitter unwraps by that role.
    if type(latent) is LatentDescriptor:
        return latent, stream_role
    if type(latent) is MultiStreamLatentDescriptor and stream_role is not None:
        for name, descriptor in latent.streams:
            if name == stream_role:
                return descriptor, stream_role
    return None


# -- TAESD quality decoder ------------------------------------------------------

# Families whose descriptor-named TAE decoder the pinned still-image TAESD
# architecture can decode. 16-channel variants (Flux's taef1) need their own
# architecture and stay unavailable; Wan's video TAEs decode through the
# TAEHV path below.
_TAESD_FAMILIES: dict[str, str] = {
    "dinkster.sd15": "sd15",
    "dinkster.sdxl": "sdxl",
    "dinkster.sdxl_refiner": "sdxl",
}

# Families whose descriptor-named TAE decoder the causal TAEHV video
# architecture (Wan's lighttaew2_*) can decode.
_TAEHV_FAMILIES = frozenset(
    {"dinkster.anima", "dinkster.krea2", "dinkster.wan21", "dinkster.wan22"}
)

# Families whose own codec asset (declared in this pack's manifest) decodes
# sampling states that no generic projection can: TripoSplat's gaussian
# decoder turns shape-code tokens into a rasterized splat frame.
_FAMILY_CODEC_ASSETS: dict[str, str] = {
    "dinkster.triposplat": "triposplat_vae_decoder",
}


class _PreviewUnavailable(Exception):
    """Why a quality decoder cannot serve this sampling call."""


@dataclass(frozen=True)
class _CachedTAESDDecoder:
    """One loaded, pool-governed TAESD decoder shared across invocations."""

    resource_id: str
    handle: NativeComponentHandle
    decode: Callable[[object], PreviewFrame]


@dataclass(frozen=True)
class _CachedTAEHVDecoder:
    """One loaded, pool-governed TAEHV decoder shared across invocations.

    The heavy parts (module and config) are shared; each sampling call
    builds its own decode closure so frame pacing starts fresh."""

    resource_id: str
    handle: NativeComponentHandle
    decoder: Any
    config: Any


@dataclass(frozen=True)
class _CachedTripoSplatDecoder:
    """One loaded, pool-governed TripoSplat gaussian decoder shared
    across invocations."""

    resource_id: str
    handle: NativeComponentHandle
    decode: Callable[[object], PreviewFrame]


_taesd_cache: dict[tuple[str, str], _CachedTAESDDecoder] = {}
_taehv_cache: dict[tuple[str, str], _CachedTAEHVDecoder] = {}
_triposplat_cache: dict[tuple[str, str], _CachedTripoSplatDecoder] = {}
_taesd_cache_lock = threading.Lock()
# Serializes build+insert so one asset loads once. Never held by the
# invalidator, so pool release (which holds the pool's coordination lock
# while invalidating) cannot deadlock against a build (which takes the
# coordination lock through pool.label).
_taesd_build_lock = threading.Lock()
_taesd_invalidator_installed = False


def _drop_preview_decoders(resource_id: str) -> None:
    with _taesd_cache_lock:
        for cache in (_taesd_cache, _taehv_cache, _triposplat_cache):
            for key in [k for k, v in cache.items() if v.resource_id == resource_id]:
                del cache[key]


def _install_preview_invalidator(pool: Any) -> None:
    global _taesd_invalidator_installed  # noqa: PLW0603 - one-shot registration
    if not _taesd_invalidator_installed:
        pool.register_invalidator(_drop_preview_decoders)
        _taesd_invalidator_installed = True


def _build_taesd_entry(asset_id: str, family: str, load_device: Any) -> _CachedTAESDDecoder:
    ref = declared_asset(asset_id)
    inference_torch = importlib.import_module("dinkster_inference_torch")
    torch = importlib.import_module("torch")
    source = load_safetensors_header(ref.local_path())
    plan = plan_taesd_decoder(source, family=cast("Any", family))
    decoder = inference_torch.assemble_taesd_decoder(plan)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        decoder,
        load_device=load_device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )
    handle = NativeComponentHandle(
        decoder,
        mechanism,
        load_device,
        resource_identity=f"taesd-preview:{ref.digest}",
        coordinator=coordinator,
    )
    pool = default_pool()
    _install_preview_invalidator(pool)
    rid = pool.label(handle, ref.name)
    handle.attach_pool(pool)
    return _CachedTAESDDecoder(
        resource_id=resident_resource_id(rid),
        handle=handle,
        decode=inference_torch.taesd_preview_decoder(decoder, plan.config),
    )


def _resolve_taesd_decoder(asset_id: str, family: str, load_device: Any) -> _CachedTAESDDecoder:
    key = (asset_id, str(load_device))
    with _taesd_build_lock:
        with _taesd_cache_lock:
            cached = _taesd_cache.get(key)
        if cached is not None and not cached.handle.released:
            return cached
        entry = _build_taesd_entry(asset_id, family, load_device)
        with _taesd_cache_lock:
            _taesd_cache[key] = entry
        return entry


def _build_taehv_entry(
    asset_id: str, descriptor: LatentDescriptor, load_device: Any
) -> _CachedTAEHVDecoder:
    ref = declared_asset(asset_id)
    inference_torch = importlib.import_module("dinkster_inference_torch")
    torch = importlib.import_module("torch")
    source = load_safetensors_header(ref.local_path())
    plan = plan_taehv_decoder(source)
    config = plan.config
    if config.latent_channels != descriptor.channels:
        raise _PreviewUnavailable(
            f"the decoder expects {config.latent_channels} latent channels, "
            f"the latent space has {descriptor.channels}"
        )
    if config.spatial_upscale != descriptor.spatial_downscale:
        raise _PreviewUnavailable(
            f"the decoder upscales {config.spatial_upscale}x spatially, "
            f"the latent space downscales {descriptor.spatial_downscale}x"
        )
    if config.temporal_upscale != descriptor.temporal_downscale:
        raise _PreviewUnavailable(
            f"the decoder upscales {config.temporal_upscale}x temporally, "
            f"the latent space downscales {descriptor.temporal_downscale}x"
        )
    decoder = inference_torch.assemble_taehv_decoder(plan)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        decoder,
        load_device=load_device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )
    handle = NativeComponentHandle(
        decoder,
        mechanism,
        load_device,
        resource_identity=f"taehv-preview:{ref.digest}",
        coordinator=coordinator,
    )
    pool = default_pool()
    _install_preview_invalidator(pool)
    rid = pool.label(handle, ref.name)
    handle.attach_pool(pool)
    return _CachedTAEHVDecoder(
        resource_id=resident_resource_id(rid),
        handle=handle,
        decoder=decoder,
        config=config,
    )


def _resolve_taehv_decoder(
    asset_id: str, descriptor: LatentDescriptor, load_device: Any
) -> _CachedTAEHVDecoder:
    key = (asset_id, str(load_device))
    with _taesd_build_lock:
        with _taesd_cache_lock:
            cached = _taehv_cache.get(key)
        if cached is not None and not cached.handle.released:
            return cached
        entry = _build_taehv_entry(asset_id, descriptor, load_device)
        with _taesd_cache_lock:
            _taehv_cache[key] = entry
        return entry


def _build_triposplat_entry(asset_id: str, load_device: Any) -> _CachedTripoSplatDecoder:
    ref = declared_asset(asset_id)
    inference_torch = importlib.import_module("dinkster_inference_torch")
    torch = importlib.import_module("torch")
    path = ref.local_path()
    source = load_safetensors_header(path, asset_digest=ref.digest, asset_size=ref.size)
    planned = plan_triposplat_split_component(source, role="gaussian-decoder", path=path)
    identity = triposplat_component_runtime_identity(planned, BFLOAT16)
    loaded = inference_torch.load_triposplat_component(
        path,
        asset=ref,
        expected_role="gaussian-decoder",
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        loaded.module,
        load_device=load_device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )
    handle = NativeComponentHandle(
        loaded.module,
        mechanism,
        load_device,
        resource_identity=f"triposplat-preview:{ref.digest}",
        coordinator=coordinator,
    )
    pool = default_pool()
    _install_preview_invalidator(pool)
    rid = pool.label(handle, ref.name)
    handle.attach_pool(pool)
    return _CachedTripoSplatDecoder(
        resource_id=resident_resource_id(rid),
        handle=handle,
        decode=inference_torch.triposplat_preview_decoder(loaded.module),
    )


def _resolve_triposplat_decoder(asset_id: str, load_device: Any) -> _CachedTripoSplatDecoder:
    key = (asset_id, str(load_device))
    with _taesd_build_lock:
        with _taesd_cache_lock:
            cached = _triposplat_cache.get(key)
        if cached is not None and not cached.handle.released:
            return cached
        entry = _build_triposplat_entry(asset_id, load_device)
        with _taesd_cache_lock:
            _triposplat_cache[key] = entry
        return entry


def _taesd_emitter(
    descriptor: LatentDescriptor,
    *,
    family_id: str,
    role: str | None,
    load_device: object,
    quality: bool,
) -> SamplingPreviewEmitter | None:
    """A quality emitter for the descriptor's named TAE decoder, or None.

    Every unavailability degrades: in ``quality`` mode one info log names
    the reason (per sampling invocation - the worker sees no run identity),
    and resolution falls through to the next provider; ``auto`` degrades
    silently."""
    asset_id = descriptor.taesd_decoder
    if asset_id is None:
        return None
    try:
        family = _TAESD_FAMILIES.get(family_id)
        if family is None:
            raise _PreviewUnavailable(
                f"the still-image TAESD decoder does not cover family {family_id}"
            )
        if load_device is None:
            raise _PreviewUnavailable("the sampling handle names no load device")
        entry = _resolve_taesd_decoder(asset_id, family, load_device)
    except Exception as error:
        if quality:
            with contextlib.suppress(Exception):
                report_log(
                    "info",
                    f"Quality sampling preview is unavailable ({error}); "
                    "a cheaper preview is used when one exists.",
                    data={"previewProvider": TAESD_PROVIDER.id, "assetId": asset_id},
                )
        return None
    handle = entry.handle
    return SamplingPreviewEmitter(
        entry.decode,
        stream_role=role,
        stage=lambda: handle.stage(observer_stage="sample"),
    )


def _taehv_emitter(
    descriptor: LatentDescriptor,
    *,
    family_id: str,
    role: str | None,
    load_device: object,
    quality: bool,
) -> SamplingPreviewEmitter | None:
    """A quality animation emitter for the descriptor's named video TAE
    decoder, or None. Unavailability degrades exactly like the TAESD path:
    one info log in ``quality`` mode, silence in ``auto``, and resolution
    falls through to the next provider either way."""
    asset_id = descriptor.taesd_decoder
    if asset_id is None:
        return None
    try:
        if family_id not in _TAEHV_FAMILIES:
            raise _PreviewUnavailable(f"the TAEHV video decoder does not cover family {family_id}")
        if load_device is None:
            raise _PreviewUnavailable("the sampling handle names no load device")
        entry = _resolve_taehv_decoder(asset_id, descriptor, load_device)
    except Exception as error:
        if quality:
            with contextlib.suppress(Exception):
                report_log(
                    "info",
                    f"Quality sampling preview is unavailable ({error}); "
                    "a cheaper preview is used when one exists.",
                    data={"previewProvider": TAEHV_PROVIDER.id, "assetId": asset_id},
                )
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    handle = entry.handle
    return SamplingPreviewEmitter(
        inference_torch.taehv_preview_decoder(entry.decoder, entry.config, descriptor),
        stream_role=role,
        stage=lambda: handle.stage(observer_stage="sample"),
    )


def _triposplat_emitter(
    descriptor: LatentDescriptor,
    *,
    family_id: str,
    role: str | None,
    load_device: object,
    quality: bool,
) -> SamplingPreviewEmitter | None:
    """A quality splat emitter for the family's gaussian decoder, or None.

    The family-pinned spec matches both TripoSplat token streams; only the
    shape-code stream decodes, so the camera stream declines silently (a
    structural fact, not an unavailability worth logging). Everything else
    degrades exactly like the TAESD path: one info log in ``quality`` mode,
    silence in ``auto``."""
    asset_id = _FAMILY_CODEC_ASSETS.get(family_id)
    if asset_id is None:
        return None
    if descriptor.channels != TRIPOSPLAT_CONFIG.latent_channels:
        return None
    try:
        if load_device is None:
            raise _PreviewUnavailable("the sampling handle names no load device")
        entry = _resolve_triposplat_decoder(asset_id, load_device)
    except Exception as error:
        if quality:
            with contextlib.suppress(Exception):
                report_log(
                    "info",
                    f"Quality sampling preview is unavailable ({error}); "
                    "a cheaper preview is used when one exists.",
                    data={"previewProvider": TRIPOSPLAT_SPLAT_PROVIDER.id, "assetId": asset_id},
                )
        return None
    handle = entry.handle
    return SamplingPreviewEmitter(
        entry.decode,
        stream_role=role,
        stage=lambda: handle.stage(observer_stage="sample"),
    )


class MultiStreamPreviewEmitter:
    """A fan-out over per-stream emitters for one multi-stream sampling
    call. Every solver state event reaches each child, which unwraps its
    own stream role and throttles independently; the combined stage keeps
    every child's decoder weights placed for the sampling call."""

    def __init__(self, emitters: tuple[SamplingPreviewEmitter, ...]) -> None:
        if len(emitters) < 2:
            raise ValueError("a multi-stream preview emitter needs at least two streams")
        self._emitters = emitters

    def stage(self) -> AbstractContextManager[object]:
        @contextlib.contextmanager
        def combined() -> Generator[object]:
            with contextlib.ExitStack() as stack:
                for emitter in self._emitters:
                    stack.enter_context(emitter.stage())
                yield stack

        return combined()

    def on_state(self, event: Any) -> None:
        for emitter in self._emitters:
            emitter.on_state(event)


PreviewEmitter = SamplingPreviewEmitter | MultiStreamPreviewEmitter


def preview_stage(emitter: PreviewEmitter | None) -> AbstractContextManager[object]:
    """The residency stage for one optional emitter's decoder weights (a
    no-op without an emitter or for decoders that own no weights)."""
    if emitter is None:
        return nullcontext()
    return emitter.stage()


def _emitter_for(
    descriptor: LatentDescriptor,
    *,
    family_id: str,
    role: str | None,
    load_device: object,
    mode: PreviewMode,
    animation: PreviewAnimation = "ring",
) -> SamplingPreviewEmitter | None:
    """The emitter for one resolved latent descriptor, or None when no
    registered provider with an available decoder backend serves it."""
    registry = builtin_preview_registry()
    # Video latent spaces prefer animated previews over stills: providers
    # resolve in kind passes so any animation provider (ring-addressed or
    # encoded) outranks every still-image one, regardless of registry
    # ordering within one pass. The run's animation transport breaks the
    # tie between the two animated kinds: "encoded" puts self-contained
    # animations in a pass of their own ahead of ring providers, while the
    # default "ring" keeps one mixed pass in which ring providers win
    # through the registry's deterministic ordering. 1D spaces resolve by
    # stream role: audio streams (and plain roleless 1D descriptors) take
    # audio providers, while non-audio token streams (TripoSplat's shape
    # codes) render as still images.
    kind_passes: tuple[tuple[str, ...], ...]
    if descriptor.dimensions == 3:
        if animation == "encoded":
            kind_passes = (("encoded_animation",), ("animation",), ("image",))
        else:
            kind_passes = (("animation", "encoded_animation"), ("image",))
    elif descriptor.dimensions == 1:
        if role is None or role == "audio":
            kind_passes = (("audio",),)
        else:
            kind_passes = (("image",),)
    else:
        kind_passes = (("image",),)
    resolutions = [
        registry.resolve(
            descriptor,
            family_id=family_id,
            mode=mode,
            kinds=cast("Any", kinds),
            family_codec=family_id in _FAMILY_CODEC_ASSETS,
        )
        for kinds in kind_passes
    ]
    if not any(resolutions):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    builders = inference_torch.preview_decoder_builders()
    quality = mode == "quality"
    for specs in resolutions:
        for spec in specs:
            if spec.id == TAESD_PROVIDER.id:
                emitter = _taesd_emitter(
                    descriptor,
                    family_id=family_id,
                    role=role,
                    load_device=load_device,
                    quality=quality,
                )
                if emitter is not None:
                    return emitter
                continue
            if spec.id == TAEHV_PROVIDER.id:
                emitter = _taehv_emitter(
                    descriptor,
                    family_id=family_id,
                    role=role,
                    load_device=load_device,
                    quality=quality,
                )
                if emitter is not None:
                    return emitter
                continue
            if spec.id == TRIPOSPLAT_SPLAT_PROVIDER.id:
                emitter = _triposplat_emitter(
                    descriptor,
                    family_id=family_id,
                    role=role,
                    load_device=load_device,
                    quality=quality,
                )
                if emitter is not None:
                    return emitter
                continue
            builder = builders.get(spec.id)
            if builder is None:
                continue
            if spec.kind == "encoded_animation":
                # The decoder yields raw clip frames on the sampling
                # thread; the container encode joins JPEG encoding on the
                # background slot.
                return SamplingPreviewEmitter(
                    builder(descriptor),
                    stream_role=role,
                    encode_animation=_encode_webp_animation,
                )
            return SamplingPreviewEmitter(builder(descriptor), stream_role=role)
    return None


def sampling_preview_emitter(
    handle: Any, *, stream_role: str | None = None
) -> SamplingPreviewEmitter | None:
    """The preview state callback for one native sampling call, or None.

    None whenever previews cost nothing to skip: the invocation's mode is
    ``off``, the family declares nothing a registered provider can use, or
    no decoder backend is available. ``stream_role`` names the latent
    stream a multi-stream family previews (for example ``"video"``)."""
    context = current_execution_context()
    if context is None or context.preview_mode == "off":
        return None
    family = getattr(handle.runtime, "family", None)
    if family is None:
        return None
    resolved = _stream_descriptor(family.latent, stream_role)
    if resolved is None:
        return None
    descriptor, role = resolved
    return _emitter_for(
        descriptor,
        family_id=family.id,
        role=role,
        load_device=getattr(handle, "load_device", None),
        mode=context.preview_mode,
        animation=context.preview_animation,
    )


def multistream_sampling_preview_emitter(handle: Any) -> PreviewEmitter | None:
    """The preview state callback for one multi-stream sampling call, or
    None. Every stream of the family's latent resolves its own provider
    (H3's audio stream previews as a waveform even while its video stream
    declares nothing decodable); a single previewable stream skips the
    fan-out, and a plain single-descriptor family resolves as usual."""
    context = current_execution_context()
    if context is None or context.preview_mode == "off":
        return None
    family = getattr(handle.runtime, "family", None)
    if family is None:
        return None
    latent = family.latent
    if type(latent) is not MultiStreamLatentDescriptor:
        return sampling_preview_emitter(handle)
    load_device = getattr(handle, "load_device", None)
    emitters: list[SamplingPreviewEmitter] = []
    for name, descriptor in latent.streams:
        emitter = _emitter_for(
            descriptor,
            family_id=family.id,
            role=name,
            load_device=load_device,
            mode=context.preview_mode,
            animation=context.preview_animation,
        )
        if emitter is not None:
            emitters.append(emitter)
    if not emitters:
        return None
    if len(emitters) == 1:
        return emitters[0]
    return MultiStreamPreviewEmitter(tuple(emitters))


# -- ComfyUI-arm translation ----------------------------------------------------

# ComfyUI latent-format class names whose latent space the Dinkster catalog
# already describes. The catalog descriptor carries what the comfy instance
# lacks (calibrated rgb factors, content fps, the TAE decoder asset name),
# so both arms resolve identical providers and wire semantics.
_COMFY_CATALOG_FAMILIES: dict[str, str] = {
    "SD15": "dinkster.sd15",
    "SDXL": "dinkster.sdxl",
    "Wan21": "dinkster.wan21",
    "Wan22": "dinkster.wan22",
}


def _catalog_latent(family_id: str) -> LatentDescriptor | None:
    for family in builtin_families():
        if family.id == family_id and type(family.latent) is LatentDescriptor:
            return family.latent
    return None


def _coerce_rgb_factors(
    value: object, channels: int
) -> tuple[tuple[float, float, float], ...] | None:
    """The comfy ``latent_rgb_factors`` as descriptor factors, or None.

    Comfy stores factors as nested lists or torch tensors (sometimes
    reshaped, as Mochi does); anything that does not coerce to exactly one
    RGB triple per latent channel is treated as absent."""
    if value is None:
        return None
    try:
        raw: Any = cast("Any", value)
        listed: Any = raw.tolist() if hasattr(raw, "tolist") else raw
        factors: list[tuple[float, float, float]] = []
        for row in cast("list[Any]", listed):
            r, g, b = cast("tuple[Any, Any, Any]", row)
            factors.append((float(r), float(g), float(b)))
    except Exception:
        return None
    if len(factors) != channels:
        return None
    return tuple(factors)


def _coerce_rgb_bias(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        raw: Any = cast("Any", value)
        listed: Any = raw.tolist() if hasattr(raw, "tolist") else raw
        r, g, b = cast("tuple[Any, Any, Any]", listed)
        return (float(r), float(g), float(b))
    except Exception:
        return None


def _comfy_generic_descriptor(fmt: Any) -> LatentDescriptor | None:
    """A cheap-preview descriptor for an untranslated comfy latent format,
    or None when its attributes do not describe a previewable space."""
    try:
        channels = int(fmt.latent_channels)
        dimensions = int(getattr(fmt, "latent_dimensions", 2))
        if dimensions not in (1, 2, 3):
            return None
        rgb_factors = _coerce_rgb_factors(getattr(fmt, "latent_rgb_factors", None), channels)
        rgb_bias: tuple[float, float, float] | None = None
        if rgb_factors is not None:
            raw_bias = getattr(fmt, "latent_rgb_factors_bias", None)
            if raw_bias is not None:
                rgb_bias = _coerce_rgb_bias(raw_bias)
                if rgb_bias is None:
                    # A bias that exists but does not coerce would tint
                    # every frame wrong; no preview beats a wrong one.
                    return None
        return LatentDescriptor(
            channels=channels,
            dimensions=cast("Any", dimensions),
            scale_factor=float(getattr(fmt, "scale_factor", 1.0)),
            rgb_factors=rgb_factors,
            rgb_bias=rgb_bias,
        )
    except Exception:
        return None


def comfy_sampling_preview_emitter(model: Any) -> SamplingPreviewEmitter | None:
    """The preview state callback for one ComfyUI-arm sampling call, or None.

    The comfy arm never touches ComfyUI's own preview machinery (its
    ``latent_preview`` module stays unused and no comfy global is mutated):
    the model's latent format translates to a Dinkster descriptor - the
    catalog's when the class name maps to a known family, otherwise a
    cheap-only generic one built from the instance's attributes - and
    provider resolution then matches the native arm exactly. Comfy solver
    callbacks report states in the processed sampler space, the same space
    the catalog's rgb factors and TAE decoders are calibrated for."""
    context = current_execution_context()
    if context is None or context.preview_mode == "off":
        return None
    fmt = getattr(getattr(model, "model", None), "latent_format", None)
    if fmt is None:
        return None
    name = type(fmt).__name__
    family_id = _COMFY_CATALOG_FAMILIES.get(name)
    descriptor = None if family_id is None else _catalog_latent(family_id)
    if descriptor is None or family_id is None:
        family_id = f"comfy.{name}"
        descriptor = _comfy_generic_descriptor(fmt)
    if descriptor is None:
        return None
    # Video spaces tag frames with the same stream role the native arm
    # uses; multi-stream sampling states unwrap by that role, and plain
    # tensor states decode directly regardless of it.
    role = "video" if descriptor.dimensions == 3 else None
    return _emitter_for(
        descriptor,
        family_id=family_id,
        role=role,
        load_device=getattr(model, "load_device", None),
        mode=context.preview_mode,
        animation=context.preview_animation,
    )


__all__ = [
    "PREVIEW_DECODE_FAILURE_LIMIT",
    "PREVIEW_JPEG_QUALITY",
    "PREVIEW_MIN_INTERVAL_S",
    "PREVIEW_WEBP_FALLBACK_FPS",
    "PREVIEW_WEBP_QUALITY",
    "MultiStreamPreviewEmitter",
    "PreviewEmitter",
    "SamplingPreviewEmitter",
    "comfy_sampling_preview_emitter",
    "multistream_sampling_preview_emitter",
    "preview_stage",
    "sampling_preview_emitter",
]
