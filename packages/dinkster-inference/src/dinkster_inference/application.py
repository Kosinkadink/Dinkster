"""Torch-free bindings for applying resident components to model execution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from .component_handle import InferenceComponentHandle, require_inference_component_handle
from .runtime_handle import require_inference_runtime_handle

_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")

_ApplicationKwargsMaterializer = Callable[
    [object, object, object],
    Mapping[str, object],
]
"""Build sample kwargs from ``(runtime, live component, runtime latent)``."""


def _identity_parts(identity: object, *, field_name: str) -> tuple[str, str]:
    if not isinstance(identity, str):
        raise TypeError(f"{field_name} must be a string")
    parts = identity.split(":")
    if len(parts) != 3 or parts[0] != "native":
        raise ValueError(f"{field_name} must be a 3-part native identity")
    if not parts[1]:
        raise ValueError(f"{field_name} family must be non-empty")
    digest = parts[2]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} digest must be a lowercase sha256 hex digest")
    return parts[1], digest


@dataclass(frozen=True, eq=False)
class ComponentApplication:
    """One resident component and its family-owned sample-kwarg builder.

    ``application_identity`` must derive from ``handle.resource_identity``
    and every output-affecting application parameter or resident dependency.
    The materializer may use its staged live-component argument only for that
    call; it must not retain the component across a staging boundary.
    """

    family_id: str
    role: str
    handle: InferenceComponentHandle
    application_identity: str
    materialize_application_kwargs: _ApplicationKwargsMaterializer
    resident_dependencies: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        family_id = cast("object", self.family_id)
        if not isinstance(family_id, str):
            raise TypeError("component application family_id must be a string")
        if not family_id:
            raise ValueError("component application family_id must be non-empty")

        role = cast("object", self.role)
        if not isinstance(role, str):
            raise TypeError("component application role must be a string")
        if _ROLE_RE.fullmatch(role) is None:
            raise ValueError("component application role must be a canonical id")

        handle = require_inference_component_handle(self.handle, "component application handle")
        component_family, _component_digest = _identity_parts(
            handle.resource_identity,
            field_name="component application handle identity",
        )
        identity_family, _digest = _identity_parts(
            self.application_identity,
            field_name="component application identity",
        )
        if identity_family != family_id:
            raise ValueError("component application identity family must match family_id")
        if component_family != family_id:
            raise ValueError("component application handle identity family must match family_id")
        if not callable(self.materialize_application_kwargs):
            raise TypeError("component application materialize_application_kwargs must be callable")
        dependencies = cast("object", self.resident_dependencies)
        if not isinstance(dependencies, tuple):
            raise TypeError("component application resident_dependencies must be a tuple")
        if any(dependency is None for dependency in cast("tuple[object, ...]", dependencies)):
            raise TypeError("component application resident dependencies must not be None")


@dataclass(frozen=True, eq=False)
class ApplicationChain:
    """An opaque model plus a non-empty ordered component application chain."""

    model: object
    applications: tuple[ComponentApplication, ...]
    _base_model_identity: str = field(init=False, repr=False)
    _chain_identity: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.model, ApplicationChain):
            raise TypeError("application chains must not be nested; use append()")
        applications = cast("object", self.applications)
        if not isinstance(applications, tuple) or not all(
            isinstance(application, ComponentApplication)
            for application in cast("tuple[object, ...]", applications)
        ):
            raise TypeError(
                "application chain applications must be a tuple of ComponentApplication"
            )
        resolved = cast("tuple[ComponentApplication, ...]", applications)
        if not resolved:
            raise ValueError("application chain applications must be non-empty")

        handle = require_inference_runtime_handle(self.model, "application chain model")
        base_identity = handle.recipe.runtime_identity
        base_family, _base_digest = _identity_parts(
            base_identity,
            field_name="application chain base model identity",
        )
        invocation_identity = getattr(self.model, "_dinkster_application_identity", base_identity)
        if not isinstance(invocation_identity, str):
            raise TypeError("_dinkster_application_identity must be a string")
        invocation_family, _invocation_digest = _identity_parts(
            invocation_identity,
            field_name="application chain invocation identity",
        )
        if invocation_family != base_family:
            raise ValueError(
                "application chain invocation identity must share the base model family"
            )
        for index, application in enumerate(resolved):
            if application.family_id != base_family:
                raise ValueError(
                    f"application chain item {index} family_id must match the base model family"
                )
            _application_family, _application_digest = _identity_parts(
                application.application_identity,
                field_name="component application identity",
            )

        hasher = hashlib.sha256()
        hasher.update(f"base={invocation_identity}\n".encode())
        for application in resolved:
            hasher.update(f"application={application.application_identity}\n".encode())
        object.__setattr__(self, "_base_model_identity", base_identity)
        object.__setattr__(
            self,
            "_chain_identity",
            f"native:{base_family}:{hasher.hexdigest()}",
        )

    @property
    def base_model_identity(self) -> str:
        """The runtime identity captured when the chain was constructed."""

        return self._base_model_identity

    @property
    def chain_identity(self) -> str:
        """Identity of the base model and ordered application sequence."""

        return self._chain_identity

    @property
    def _dinkster_resident_owner(self) -> object:
        return getattr(self.model, "_dinkster_resident_owner", self.model)

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        refs: list[object] = []
        for application in self.applications:
            refs.append(application.handle)
            refs.extend(
                getattr(dependency, "_dinkster_resident_owner", dependency)
                for dependency in application.resident_dependencies
            )
        return tuple(refs)

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.chain_identity

    def append(self, application: ComponentApplication) -> ApplicationChain:
        """Return a normalized chain with one application appended."""

        if not isinstance(cast("object", application), ComponentApplication):
            raise TypeError("application must be a ComponentApplication")
        return ApplicationChain(self.model, (*self.applications, application))


__all__ = [
    "ApplicationChain",
    "ComponentApplication",
]
