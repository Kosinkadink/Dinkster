"""Digest-addressed media thumbnails shared by the engine and always-on process.

Media dependencies load on demand, never when the library helper starts.
"""

from __future__ import annotations

import io
import json
import tempfile
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import BinaryIO, cast

from .asset_identity import new_hasher

PREVIEW_EDGE = 512
PREVIEW_MEDIA_TYPE = "image/webp"
THUMBNAIL_SIZES = (128, 256, 512)
THUMBNAIL_VERSION = "media-v1"


class ContentMismatch(Exception):
    """The stored bytes no longer hash to their digest."""


class ThumbnailUnavailable(Exception):
    """The asset cannot be drawn by the installed media renderers."""


@dataclass(frozen=True)
class Thumbnail:
    image: bytes
    duration: float | None = None
    peaks: tuple[tuple[float, float], ...] | None = None

    @property
    def media_type(self) -> str:
        return "application/json" if self.peaks is not None else PREVIEW_MEDIA_TYPE

    @property
    def body(self) -> bytes:
        if self.peaks is None:
            return self.image
        return json.dumps(
            {"duration": self.duration, "peaks": self.peaks},
            separators=(",", ":"),
            allow_nan=False,
        ).encode()


@dataclass
class _VerifiedSource:
    path: Path
    digest: str
    size: int

    def open(self) -> BinaryIO:
        return self.path.open("rb")

    def to_wire(self) -> dict[str, object]:
        return {"digest": self.digest, "size": self.size}


_cache: OrderedDict[tuple[str, int, str], Thumbnail] = OrderedDict()
_cache_lock = Lock()


def thumbnail_etag(digest: str, size: int, *, preview: bool = False) -> str:
    representation = "preview" if preview else "thumbnail"
    return f'"{digest}-{representation}-{size}-{THUMBNAIL_VERSION}"'


def _webp(image: bytes, size: int) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as frame:
        frame.thumbnail((size, size), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        frame.save(out, "WEBP", quality=82)
    return out.getvalue()


def _render(source: _VerifiedSource, size: int) -> Thumbnail:
    from dinkster_values.image_codec import decode_image_file, render_image_png
    from dinkster_values.registry import RenditionUnavailable
    from PIL import Image

    try:
        with source.open() as handle, Image.open(handle):
            pass
    except (OSError, ValueError, Image.DecompressionBombError):
        pass
    else:
        try:
            return Thumbnail(_webp(render_image_png(decode_image_file(source)), size))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ThumbnailUnavailable("this image cannot be drawn") from error

    from dinkster_values.video_codec import video_from_source

    try:
        video = video_from_source(source)
    except ImportError as error:
        raise ThumbnailUnavailable("the media thumbnail decoder is unavailable") from error
    except (OSError, ValueError):
        video = None
    if video is not None:
        try:
            from dinkster_values.video_edits import effective_video_facts
            from dinkster_video_runtime import iter_video_pixels

            duration = effective_video_facts(video)["duration"]
            if duration is None:
                raise ThumbnailUnavailable("this video's duration is unavailable")
            with closing(iter_video_pixels(video)) as frames:
                _, frame = next(frames)
                return Thumbnail(
                    _webp(render_image_png(frame), size), float(cast(Fraction, duration))
                )
        except ImportError as error:
            raise ThumbnailUnavailable("the video thumbnail runtime is unavailable") from error
        except (OSError, ValueError, StopIteration) as error:
            raise ThumbnailUnavailable("this video cannot be drawn") from error

    try:
        from dinkster_values.audio_codec import audio_waveform_peaks, render_audio_waveform
        from dinkster_values.audio_lazy import audio_from_source, effective_audio_facts

        audio = audio_from_source(source)
        duration = effective_audio_facts(audio)["duration"]
        if duration is None:
            raise ThumbnailUnavailable("this audio's duration is unavailable")
        peaks = tuple(audio_waveform_peaks(audio, size))
        image = render_audio_waveform(audio, {"waveform": f"{size}x{size // 4}", "batch": "0"})
        return Thumbnail(_webp(image, size), float(duration), peaks)
    except (ImportError, OSError, ValueError, RenditionUnavailable) as error:
        raise ThumbnailUnavailable("this asset has no drawable image, video or audio") from error


def file_thumbnail(path: Path, digest: str, size: int = 256) -> Thumbnail:
    """Verify a bounded-memory snapshot before rendering or reusing immutable bytes."""
    if size not in THUMBNAIL_SIZES:
        raise ValueError("thumbnail size must be 128, 256 or 512")
    key = (digest, size, THUMBNAIL_VERSION)
    # Decode the verified snapshot, not a reopened pathname that could have changed.
    with tempfile.TemporaryDirectory(prefix="dinkster-thumbnail-") as directory:
        snapshot = Path(directory) / "asset"
        hasher = new_hasher()
        length = 0
        with path.open("rb") as original, snapshot.open("wb") as target:
            while chunk := original.read(1024 * 1024):
                hasher.update(chunk)
                target.write(chunk)
                length += len(chunk)
        if "blake3:" + hasher.hexdigest() != digest:
            raise ContentMismatch(digest)
        with _cache_lock:
            cached = _cache.get(key)
            if cached is not None:
                _cache.move_to_end(key)
                return cached
        result = _render(_VerifiedSource(snapshot, digest, length), size)
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > 128:
            _cache.popitem(last=False)
    return result
