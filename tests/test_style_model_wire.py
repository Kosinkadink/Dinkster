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
def test_model_patch_registration_and_schema_discovery_are_runtime_free(native_first: bool) -> None:
    script = f"""
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import MODEL_PATCH_TYPE, TypeRegistry, register_inference_types
from dinkster_values import MODEL_PATCH_TYPE as values_type
from dinkster_values.builtin_types import MODEL_PATCH_TYPE as builtin_type
from dinkster_memory.resident import default_pool
from dinkster_native.native import register_native_types
from dinkster_nodes_comfy import COMFY_NODES
assert MODEL_PATCH_TYPE == values_type == builtin_type == 'MODEL_PATCH'
registry = TypeRegistry()
registrations = (register_native_types, register_inference_types)
if not {native_first!r}:
    registrations = registrations[::-1]
for registration in registrations:
    registration(registry)
spec = registry.spec(MODEL_PATCH_TYPE)
register_inference_types(registry)
assert registry.spec(MODEL_PATCH_TYPE) is spec
assert default_pool() in registry.resident_tables
assert 'comfy.MODEL_PATCH' not in registry
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
