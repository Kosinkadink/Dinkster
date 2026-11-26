"""Output previews for the projects page.

The engine and the always-on Dinkster process both answer
``GET /api/assets/{digest}/preview`` with this rendering, so a tile looks the
same whether or not an engine is open. Pillow is imported on first use: the
library helper itself never renders.
"""

from __future__ import annotations

import io
from pathlib import Path

from .asset_identity import digest_bytes

__all__ = ["PREVIEW_EDGE", "PREVIEW_MEDIA_TYPE", "ContentMismatch", "file_preview"]

PREVIEW_EDGE = 512
PREVIEW_MEDIA_TYPE = "image/webp"


class ContentMismatch(Exception):
    """The stored bytes no longer hash to their digest."""


def file_preview(path: Path, digest: str) -> bytes | None:
    """A WebP of the first frame of the still or animated image stored at
    ``path`` under ``digest``, its longest edge at most PREVIEW_EDGE with the
    aspect kept and never upscaled; None when it is not an image Pillow
    decodes. Raises ContentMismatch, and OSError when the file is unreadable."""
    from PIL import Image

    try:
        # Only the header is read, so a large video is refused without hashing it.
        with Image.open(path):
            pass
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    data = path.read_bytes()
    if digest_bytes(data) != digest:
        raise ContentMismatch(digest)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(0)
            image.draft("RGB", (PREVIEW_EDGE, PREVIEW_EDGE))
            if image.mode.startswith("I;16"):
                # Pillow clips 16-bit samples when converting; scale them to 8 bits.
                frame = image.convert("I").point(lambda value: value / 256).convert("L")
            else:
                frame = image.convert(
                    "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                )
        frame.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        frame.save(out, "WEBP", quality=82)
    except (OSError, ValueError, EOFError, Image.DecompressionBombError):
        return None
    return out.getvalue()
