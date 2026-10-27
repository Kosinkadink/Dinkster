"""Public schemas for native sampling cache nodes remain Torch-free."""

from dinkster_api.v1 import CORE_BOOLEAN, CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.cache import CACHE_NODES, EasyCache, FluxKVCache, LazyCache


def test_cache_nodes_publish_the_fixed_core_interfaces() -> None:
    assert CACHE_NODES == (EasyCache, LazyCache, FluxKVCache)
    assert set(CACHE_NODES).issubset(COMFY_NODES)
    flux = FluxKVCache.define_schema()
    assert flux.category == "experimental"
    assert [(item.id, item.type) for item in flux.inputs] == [
        ("model", TypeExpr.concrete(MODEL_TYPE))
    ]
    assert [(item.id, item.type) for item in flux.outputs] == [
        ("MODEL", TypeExpr.concrete(MODEL_TYPE))
    ]
    for node in (EasyCache, LazyCache):
        schema = node.define_schema()
        assert schema.node_type == node.__name__
        assert schema.category == "advanced/debug"
        assert [item.id for item in schema.inputs] == [
            "model",
            "reuse_threshold",
            "start_percent",
            "end_percent",
            "verbose",
        ]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert [item.type for item in schema.inputs[1:4]] == [TypeExpr.concrete(CORE_FLOAT)] * 3
        assert schema.inputs[4].type == TypeExpr.concrete(CORE_BOOLEAN)
        assert [item.default for item in schema.inputs[1:]] == [0.2, 0.15, 0.95, False]
        assert [item.widget for item in schema.inputs[1:4]] == [
            NumberWidget(min=0.0, max=3.0, step=0.01),
            NumberWidget(min=0.0, max=1.0, step=0.01),
            NumberWidget(min=0.0, max=1.0, step=0.01),
        ]
        assert all(item.advanced for item in schema.inputs[1:])
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
