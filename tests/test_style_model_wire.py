from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("native_first", (True, False))
def test_style_model_registration_and_schema_discovery_are_runtime_free(native_first: bool) -> None:
    script = f"""
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import STYLE_MODEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import STYLE_MODEL_TYPE as values_type
from dinkster_values.builtin_types import STYLE_MODEL_TYPE as builtin_type
from dinkster_memory.resident import default_pool
from dinkster_native.native import register_native_types
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.style_model import StyleModelApply
assert StyleModelApply.schema().display_name == 'Apply Style Model'
assert STYLE_MODEL_TYPE == values_type == builtin_type == 'STYLE_MODEL'
registry = TypeRegistry()
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
for registration in registrations:
    registration(registry)
spec = registry.spec(STYLE_MODEL_TYPE)
register_inference_types(registry)
assert registry.spec(STYLE_MODEL_TYPE) is spec
assert default_pool() in registry.resident_tables
assert 'comfy.STYLE_MODEL' not in registry
for node in COMFY_NODES:
    node.schema()
source = object()
assert spec.decode(spec.encode(source)) is source
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
assert not any(name.startswith('dinkster_inference.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("native_first", (True, False))
def test_moge_registration_is_cold_idempotent_and_uses_holding_pool(native_first: bool) -> None:
    script = f"""
import subprocess
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import (
    MOGE_MODEL_TYPE, MOGE_GEOMETRY_TYPE, TypeRegistry, register_inference_types)
from dinkster_memory.resident import default_pool
from dinkster_native.native import register_native_types
from dinkster_nodes_comfy import register_inference_types as register_comfy_types
from dinkster_values import ResidentLookupError
registry = TypeRegistry()
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
registrations[0](registry)
model = registry.spec(MOGE_MODEL_TYPE)
geometry = registry.spec(MOGE_GEOMETRY_TYPE)
for registration in (*registrations, register_comfy_types, *registrations):
    registration(registry)
    assert registry.spec(MOGE_MODEL_TYPE) is model
    assert registry.spec(MOGE_GEOMETRY_TYPE) is geometry
pool = default_pool()
assert pool in registry.resident_tables
source = object()
encoded = model.encode(source)
assert model.decode(encoded) is source
assert model.encode(source) == encoded
foreign = '''
from dinkster_api.v1 import MOGE_MODEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import ResidentLookupError
registry = TypeRegistry()
register_inference_types(registry)
try:
    registry.spec(MOGE_MODEL_TYPE).decode(bytes.fromhex({{wire!r}}))
except ResidentLookupError:
    pass
else:
    raise AssertionError('foreign model resolved outside its holding process')
'''
result = subprocess.run([sys.executable, '-c', foreign.format(wire=encoded.hex())],
                        capture_output=True, text=True)
assert result.returncode == 0, result.stderr
assert pool.release_resident(source)
try:
    model.decode(encoded)
except ResidentLookupError:
    pass
else:
    raise AssertionError('released model still resolves')
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
assert not any(name.startswith('dinkster_inference.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("type_id", ("MOGE_MODEL", "MOGE_GEOMETRY"))
def test_moge_registration_preserves_existing_authority(type_id: str) -> None:
    from dinkster_api.v1 import TypeRegistry, register_inference_types
    from dinkster_native.native import register_native_types
    from dinkster_nodes_comfy import register_inference_types as register_comfy_types

    registry = TypeRegistry()
    prior = registry.register(type_id)
    for registration in (register_inference_types, register_native_types, register_comfy_types):
        registration(registry)
        assert registry.spec(type_id) is prior


@pytest.mark.parametrize("native_first", (True, False))
@pytest.mark.parametrize("type_name", ("MODEL_PATCH_TYPE", "LATENT_UPSCALE_MODEL_TYPE"))
def test_model_patch_registration_and_schema_discovery_are_runtime_free(
    native_first: bool, type_name: str
) -> None:
    script = f"""
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import {type_name} as resident_type, TypeRegistry, register_inference_types
from dinkster_values import {type_name} as values_type
from dinkster_values.builtin_types import {type_name} as builtin_type
from dinkster_memory.resident import default_pool
from dinkster_native.native import register_native_types
from dinkster_nodes_comfy import COMFY_NODES
assert resident_type == values_type == builtin_type == {type_name.removesuffix("_TYPE")!r}
registry = TypeRegistry()
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
registrations[0](registry)
spec = registry.spec(resident_type)
registrations[1](registry)
assert registry.spec(resident_type) is spec
register_inference_types(registry)
assert registry.spec(resident_type) is spec
assert default_pool() in registry.resident_tables
assert 'comfy.' + resident_type not in registry
for node in COMFY_NODES:
    node.schema()
source = object()
assert spec.decode(spec.encode(source)) is source
assert sys.modules['torch'] is None
assert sys.modules['dinkster_inference'] is None
assert not any(name.startswith('dinkster_inference.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
