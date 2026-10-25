"""Hook value codecs retain canonical runtime identities without Torch."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from dinkster_api.v1 import (
    FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    TIMESTEPS_RANGE_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor, ResidencyTable, ResidentLookupError


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_inference_types(registry)
    return registry


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ((0.75, -2), (0.75, -2)),
        ([0.75, -2], (0.75, -2)),
        ((-3.0, 1.25), (-3.0, 1.25)),
    ),
)
def test_timesteps_range_preserves_endpoint_order_without_clamping(
    value: object, expected: tuple[int | float, int | float]
) -> None:
    spec = _registry().spec(TIMESTEPS_RANGE_TYPE)
    encoded = spec.encode(value)

    assert spec.decode(encoded) == expected
    assert spec.encode(spec.decode(encoded)) == encoded


@pytest.mark.parametrize(
    "value",
    (True, "0.5", object(), (0.0,), (0.0, 1.0, 2.0), (0.0, float("nan"))),
)
def test_timesteps_range_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _registry().spec(TIMESTEPS_RANGE_TYPE).encode(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, 1),
        (-1.25, -1.25),
        ([1, -2.5], (1, -2.5)),
        ((1, -2.5), (1, -2.5)),
        (range(-2, 3, 2), (-2, 0, 2)),
        ([], ()),
    ),
)
def test_floats_preserves_scalars_or_ordered_sequences(value: object, expected: object) -> None:
    spec = _registry().spec(FLOATS_TYPE)
    encoded = spec.encode(value)

    assert spec.decode(encoded) == expected
    assert spec.encode(spec.decode(encoded)) == encoded


def test_floats_preserves_numpy_tensor_dtype_shape_and_bytes() -> None:
    source = np.array([np.float32(0.5), np.float32(-3.25)], dtype=np.float32)
    spec = _registry().spec(FLOATS_TYPE)
    decoded = spec.decode(spec.encode(source))

    assert type(decoded) is EncodedLatentTensor
    assert decoded.dtype == "float32"
    assert decoded.shape == (2,)
    assert decoded.data == source.tobytes()
    assert spec.fingerprint is not None
    fingerprint = spec.fingerprint
    assert fingerprint(source) == fingerprint(decoded)


@pytest.mark.parametrize(
    "value",
    (
        True,
        "1.0",
        object(),
        (item for item in (1, 2)),
        np.float32(1.0),
        np.array(1.0, dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        [1, float("inf")],
        [False],
    ),
)
def test_floats_rejects_impostors_unsized_and_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _registry().spec(FLOATS_TYPE).encode(value)


def test_hook_values_are_resident_and_release_invalidates_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
    registry = _registry()
    for type_id in (HOOKS_TYPE, HOOK_KEYFRAMES_TYPE):
        spec = registry.spec(type_id)
        source = object()
        encoded = spec.encode(source)

        assert spec.decode(encoded) is source
        assert spec.fingerprint is not None
        fingerprint = spec.fingerprint
        assert fingerprint(source) == fingerprint(spec.decode(encoded))
        resident_id = json.loads(encoded)["residentId"]
        assert pool.remove(resident_id) is source
        with pytest.raises(ResidentLookupError):
            spec.decode(encoded)


def test_hook_values_reject_foreign_resident_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    source_pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: source_pool)
    source_registry = _registry()
    target_pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: target_pool)
    target_registry = _registry()
    encoded = source_registry.spec(HOOKS_TYPE).encode(object())

    with pytest.raises(ResidentLookupError):
        target_registry.spec(HOOKS_TYPE).decode(encoded)


def test_hook_type_registration_is_torch_free() -> None:
    script = """
import sys
from dinkster_api.v1 import (
    CLIP_TYPE, FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE,
    TypeRegistry, register_inference_types,
)
from dinkster_native.native import register_native_types

for first, second in ((register_native_types, register_inference_types),
                      (register_inference_types, register_native_types)):
    registry = TypeRegistry()
    first(registry)
    clip = registry.spec(CLIP_TYPE)
    second(registry)
    type_ids = (FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE)
    assert all(type_id in registry for type_id in type_ids)
    assert registry.spec(CLIP_TYPE) is clip
    assert clip.coerce is not None
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("native_first", (False, True))
def test_clip_hook_replacement_preserves_only_current_declared_dependencies(
    monkeypatch: pytest.MonkeyPatch, native_first: bool
) -> None:
    from dinkster_api.v1 import CLIP_TYPE
    from dinkster_native.native import register_native_types
    from dinkster_values import RESOURCE_ID_META_KEY, RESOURCE_REFS_META_KEY, resident_resource_id

    pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
    monkeypatch.setattr("dinkster_native.native.default_pool", lambda: pool)
    registry = TypeRegistry()
    first, second = (
        (register_native_types, register_inference_types)
        if native_first
        else (register_inference_types, register_native_types)
    )
    first(registry)
    spec = registry.spec(CLIP_TYPE)
    second(registry)
    first(registry)
    assert registry.spec(CLIP_TYPE) is spec
    primary, explicit, group_owner, group_ref = (object() for _ in range(4))
    old_keyframes, new_keyframes = object(), object()
    old_group = SimpleNamespace(hooks=[SimpleNamespace(hook_keyframe=old_keyframes)])
    new_group = SimpleNamespace(
        hooks=[SimpleNamespace(hook_keyframe=new_keyframes)],
        _dinkster_resident_owner=group_owner,
        _dinkster_resident_refs=(group_ref, group_ref),
    )
    clip = SimpleNamespace(
        apply_hooks_to_conds=old_group,
        patcher=SimpleNamespace(forced_hooks=old_group),
        _dinkster_resident_owner=primary,
        _dinkster_resident_refs=(explicit, explicit),
    )
    assert spec.coerce is not None
    spec.coerce(clip)
    clip.apply_hooks_to_conds = new_group
    value = registry.wrap(CLIP_TYPE, clip)
    expected = (
        explicit,
        new_group,
        group_owner,
        group_ref,
        new_keyframes,
        old_group,
        old_keyframes,
    )
    assert value.meta.get(RESOURCE_ID_META_KEY) == resident_resource_id(pool.rid_for(primary))
    assert value.meta.get(RESOURCE_REFS_META_KEY) == tuple(
        resident_resource_id(pool.rid_for(owner)) for owner in expected
    )
    assert registry.wrap(CLIP_TYPE, clip).meta == value.meta
    clip.patcher.forced_hooks = new_group
    replaced = registry.wrap(CLIP_TYPE, clip)
    assert replaced.meta.get(RESOURCE_REFS_META_KEY) == tuple(
        resident_resource_id(pool.rid_for(owner)) for owner in expected[:-2]
    )
    clip.apply_hooks_to_conds = None
    clip.patcher.forced_hooks = None
    cleared = registry.wrap(CLIP_TYPE, clip)
    assert cleared.meta.get(RESOURCE_REFS_META_KEY) == (
        resident_resource_id(pool.rid_for(explicit)),
    )
    assert clip._dinkster_resident_owner is primary
