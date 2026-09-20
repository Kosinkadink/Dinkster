from __future__ import annotations

from ..family_registry import load_component as load_registered_component
from ..native_residency import NativeComponentHandle


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    role_labels = {
        "qwen3vl-32b-conditioner": "conditioner",
        "video-vae": "video VAE",
        "audio-vae": "audio VAE",
    }
    expected = "component" if role is None else role_labels.get(role, "component")
    try:
        return load_registered_component(value, name, role)
    except TypeError as error:
        raise TypeError(f"{name} must be a native MiniMax H3 {expected} component") from error
