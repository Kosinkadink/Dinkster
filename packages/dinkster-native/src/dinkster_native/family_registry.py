"""Registered native node callables for model-family components."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dinkster_inference.component_registry import execution_symbol

from .native_residency import NativeComponentHandle


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    native = importlib.import_module("dinkster_native.native_arm")
    if not isinstance(value, native.NativeComponentHandle):
        raise TypeError(f"{name} must be a native component")
    handle = cast("NativeComponentHandle", value)
    handle.require_active()
    recipe = handle.recipe
    descriptor = (
        None
        if recipe is None
        else native._active_inference_registries().components.get(recipe.family_id)
    )
    source_roles = () if recipe is None else tuple(source.role for source in recipe.sources)
    expected_roles = () if descriptor is None else descriptor.roles
    selected_role = source_roles[0] if role is None and len(source_roles) == 1 else role
    if (
        recipe is None
        or descriptor is None
        or selected_role is None
        or source_roles != (selected_role,)
        or selected_role not in expected_roles
        or recipe.runtime_identity != handle.resource_identity
    ):
        label = "component" if descriptor is None else descriptor.family.display_name
        raise TypeError(
            f"{name} must be a native {label} {selected_role or 'component'} component "
            "with matching source and identity"
        )
    assert selected_role is not None
    try:
        importlib.import_module("dinkster_inference").ComponentBinding(
            selected_role.replace("-", "_"),
            recipe.family_id,
            handle.resource_identity,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a native {descriptor.family.display_name} {selected_role} component "
            "with a valid source identity digest"
        ) from error
    return handle


def encode_component_text(
    value: object,
    text: str,
    options: Any,
    *,
    runtime: str,
    carrier: str,
    role: str | None = None,
    t5_options: bool = False,
    omit_empty_t5_options: bool = False,
    carrier_within_stage: bool = False,
) -> object:
    handle = load_component(value, "clip", role)
    recipe = handle.recipe
    assert recipe is not None
    selected_role = recipe.sources[0].role
    runtime_type = execution_symbol(runtime)
    encode_options: dict[str, object] = {}
    if t5_options and (
        not omit_empty_t5_options
        or options.t5_min_padding is not None
        or options.t5_min_length is not None
    ):
        encode_options["min_padding"] = options.t5_min_padding
        encode_options["min_length"] = options.t5_min_length
    torch = importlib.import_module("dinkster_native.native_arm")._torch()
    to_carrier = execution_symbol(carrier)
    carrier_value = None
    with handle.stage():
        text_runtime = runtime_type(handle.component)
        with torch.inference_mode():
            conditioning = text_runtime.encode_text(text, **encode_options)
            if carrier_within_stage:
                carrier_value = to_carrier(conditioning)
    if not carrier_within_stage:
        carrier_value = to_carrier(conditioning)
    assert carrier_value is not None
    inference = importlib.import_module("dinkster_inference")
    return inference.bind_component_conditioning(
        carrier_value,
        inference.ComponentBinding(selected_role, recipe.family_id, handle.resource_identity),
    )


def decode_component(codec: Any, latent: Any) -> Any:
    return codec.decode_latent(latent)


def encode_component(codec: Any, content: Any) -> Any:
    return codec.encode_content(content)


def registered_callable(value: object, attribute: str) -> Any:
    recipe = getattr(value, "recipe", None)
    family_id = getattr(recipe, "family_id", None)
    native = importlib.import_module("dinkster_native.native_arm")
    descriptor = (
        None
        if family_id is None
        else native._active_inference_registries().components.get(family_id)
    )
    reference = None if descriptor is None else getattr(descriptor, attribute)
    if reference is None:
        roles = tuple(source.role for source in getattr(recipe, "sources", ()))
        raise TypeError(f"no declared {attribute}; detected family={family_id!r}, roles={roles!r}")
    return execution_symbol(reference)
