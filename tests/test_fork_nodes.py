from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import MODEL_PATCH_TYPE
from dinkster_inference_wire import ResidentConditioningCarrier
from dinkster_native.native import register_native_types
from dinkster_native.pool import default_pool
from dinkster_protocol import Invocation
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import ResidentLookupError, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

CONDITIONING = TypeExpr.concrete("CONDITIONING")
INT = TypeExpr.concrete("INT")


class _ConditioningConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.conditioning-consumer",
            inputs=(InputSpec("conditioning", CONDITIONING),),
            outputs=(OutputSpec("count", INT),),
        )

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        values = list(cast("Any", conditioning))
        return cls.outputs(count=len(values))


def test_native_types_preserve_resident_conditioning_codec() -> None:
    class ResidentPayload:
        _dinkster_resident_fingerprint = "test-resident-conditioning"
        _dinkster_input_value = "native-conditioning"

    registry = TypeRegistry()
    register_native_types(registry)
    carrier = ResidentConditioningCarrier(ResidentPayload())

    spec = registry.spec("CONDITIONING")
    assert spec.decode(spec.encode(carrier)) is carrier
    assert registry.input_object("CONDITIONING", carrier) == "native-conditioning"


def test_hook_conditioning_release_invalidates_only_dependent_values() -> None:
    from dinkster_caches import MemoryLRUCache
    from dinkster_inference_wire.conditioning_wire import register_conditioning_type
    from dinkster_memory.resident import ResidentPool
    from dinkster_protocol import CacheKey
    from dinkster_values import RESOURCE_REFS_META_KEY, ResidentCodec, resident_resource_id

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        registry = TypeRegistry()
        spec = register_conditioning_type(registry, resident_table=pool)
        cache = MemoryLRUCache(max_bytes=1024)
        pool.register_invalidator(cache.drop_referencing)
        first_hook, replacement_hook = object(), object()
        embedding = object()
        first = [[embedding, {"hooks": first_hook, "strength": 0.3}]]
        replacement = [[embedding, {"hooks": replacement_hook, "strength": 1.7}]]
        plain = [[embedding, {"strength": 0.6}]]
        first_value = registry.wrap("CONDITIONING", first)
        replacement_value = registry.wrap("CONDITIONING", replacement)
        plain_value = registry.wrap("CONDITIONING", plain)
        assert first_value.fingerprint == registry.wrap("CONDITIONING", first).fingerprint
        assert first_value.fingerprint != replacement_value.fingerprint
        assert first_value.meta.get(RESOURCE_REFS_META_KEY) == (
            resident_resource_id(pool.rid_for(first_hook)),
        )
        first_bytes = spec.encode(first_value.resolve())
        replacement_bytes = spec.encode(replacement_value.resolve())
        assert registry.input_object("CONDITIONING", spec.decode(first_bytes)) is first
        assert registry.input_object("CONDITIONING", spec.decode(replacement_bytes)) is replacement
        assert plain_value.resolve() is plain
        assert spec.encode(plain) == b"DMFR\x01" + ResidentCodec(pool).encode(plain)
        assert spec.decode(spec.encode(plain)) is plain
        first_key, replacement_key = CacheKey("first"), CacheKey("replacement")
        await cache.put(first_key, {"conditioning": first_value})
        await cache.put(replacement_key, {"conditioning": replacement_value})
        assert await cache.get(first_key) is not None
        assert pool.release_resident(first_hook)
        assert await cache.get(first_key) is None
        assert await cache.get(replacement_key) is not None
        with pytest.raises(ResidentLookupError):
            spec.decode(first_bytes)
        assert registry.input_object("CONDITIONING", spec.decode(replacement_bytes)) is replacement
        foreign = TypeRegistry()
        foreign_spec = register_conditioning_type(foreign, resident_table=ResidentPool())
        with pytest.raises(ResidentLookupError):
            foreign_spec.decode(replacement_bytes)
        assert first[0][1] == {"hooks": first_hook, "strength": 0.3}
        assert replacement[0][1] == {"hooks": replacement_hook, "strength": 1.7}
        assert pool.release_resident(replacement_hook)
        assert await cache.get(replacement_key) is None
        with pytest.raises(ResidentLookupError):
            spec.decode(replacement_bytes)

    run_scenario(scenario())


def test_native_types_register_resident_model_patches() -> None:
    registry = TypeRegistry()
    register_native_types(registry)
    model_patch = object()

    spec = registry.spec(MODEL_PATCH_TYPE)

    assert spec.decode(spec.encode(model_patch)) is model_patch


def test_native_latent_operation_full_release_invalidates_encoded_closure() -> None:
    registry = TypeRegistry()
    register_native_types(registry)
    storage = np.zeros(1)
    storage_reference = weakref.ref(storage)

    def make_operation(captured: np.ndarray[Any, Any]) -> object:
        def operation(value: object) -> np.ndarray[Any, Any]:
            del value
            return captured

        return operation

    operation = make_operation(storage)
    spec = registry.spec("LATENT_OPERATION")
    encoded = spec.encode(operation)

    assert run_scenario(default_pool().full_release()).status == "complete"
    del operation, storage
    gc.collect()

    with pytest.raises(ResidentLookupError):
        spec.decode(encoded)
    assert storage_reference() is None


def test_worker_unwraps_resident_conditioning_for_node_consumers() -> None:
    class ResidentPayload:
        _dinkster_resident_fingerprint = "test-worker-conditioning"
        _dinkster_input_value = (("embedding", {"pooled_output": "pooled"}),)

    registry = TypeRegistry()
    register_core_types(registry)
    register_native_types(registry)
    value = registry.wrap(
        "CONDITIONING",
        ResidentConditioningCarrier(ResidentPayload()),
    )
    schema = _ConditioningConsumer.schema()
    worker = InProcessWorker({schema.node_type: _ConditioningConsumer}, registry)
    result = run_scenario(
        worker.invoke(
            Invocation(
                "invocation",
                "basic-guider",
                schema.node_type,
                {"conditioning": value},
                schema,
            )
        )
    )

    assert result.error is None, result.error
    assert result.outputs is not None
    assert result.outputs["count"].resolve() == 1
