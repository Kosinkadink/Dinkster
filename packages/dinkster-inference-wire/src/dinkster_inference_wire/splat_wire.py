"""Source-shaped Gaussian tensors at the SPLAT consumer boundary."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

from .media_wire import materialize_image, materialize_latent


def materialize_splat(value: object) -> SimpleNamespace:
    """Expose attributes while retaining live tensor device, dtype and storage."""
    fields = cast("Mapping[str, object]", value if isinstance(value, Mapping) else vars(value))
    return SimpleNamespace(
        **{
            "counts": None,
            **{
                key: None if item is None else materialize_image(materialize_latent(item))
                for key, item in fields.items()
            },
        }
    )
