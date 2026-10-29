"""Public seam for consuming a loaded native runtime handle.

Node bodies outside the inference packages receive loaded models, text
encoders, and codecs as opaque resident values. This protocol names the
narrow surface such a value guarantees - runtime access, reconstruction
identity, an opaque device token, and the lease lifecycle - without
binding consumers to any concrete residency implementation. Residency
policy (what loads, unloads, or pages when) stays with the handle owner;
consumers only lease a component role for the duration of one execution.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from .recipe import ReconstructionRecipe
from .runtime import CustomSamplingRuntime


@runtime_checkable
class InferenceRuntimeHandle(Protocol):
    """One resident identity for a loaded native runtime.

    The handle outlives individual executions; ``runtime`` raises once the
    handle is released or its materialization was dropped, so consumers
    re-read it per use instead of caching the runtime object.
    """

    @property
    def runtime(self) -> CustomSamplingRuntime[Any]:
        """The live family runtime; raises after release."""
        ...

    @property
    def recipe(self) -> ReconstructionRecipe:
        """The reconstruction recipe identifying this runtime."""
        ...

    @property
    def load_device(self) -> object:
        """Opaque backend device token accepted by tensors used with the
        runtime; never None."""
        ...

    def require_active(self) -> None:
        """Raise if the handle was terminally released."""
        ...

    def stage(self, role: str) -> AbstractContextManager[None]:
        """Lease the named component role (for example ``"text"``,
        ``"diffusion"``, ``"vae"``) for one execution."""
        ...


def require_inference_runtime_handle(value: object, input_id: str) -> InferenceRuntimeHandle:
    """Fail-closed narrowing of one node input to a runtime handle.

    Structural protocol membership alone proves only attribute presence, so
    this also checks the invariants concrete handles establish at
    construction: an active lifecycle, a real reconstruction recipe, a
    runtime satisfying an execution protocol, and a runtime identity that
    matches the recipe.
    """

    if not isinstance(value, InferenceRuntimeHandle):
        raise TypeError(f"{input_id} must be a native runtime handle, got {type(value).__name__}")
    for name in ("require_active", "stage"):
        if not callable(getattr(value, name)):
            raise TypeError(f"{input_id} {name} must be callable")
    value.require_active()
    recipe = value.recipe
    if not isinstance(recipe, ReconstructionRecipe):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{input_id} recipe must be a ReconstructionRecipe")
    runtime = value.runtime
    if not isinstance(runtime, CustomSamplingRuntime):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{input_id} runtime does not satisfy the custom sampling protocol")
    identity = runtime.runtime_identity
    if not isinstance(identity, str) or not identity:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{input_id} runtime identity must be a non-empty string")
    if identity != recipe.runtime_identity:
        raise ValueError(f"{input_id} runtime identity does not match its reconstruction recipe")
    return value


__all__ = [
    "InferenceRuntimeHandle",
    "require_inference_runtime_handle",
]
