"""Public schemas for native sampling cache nodes remain Torch-free."""

from dinkster_api.v1 import CORE_BOOLEAN, CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_flux import NODES as FLUX_NODES
from dinkster_flux.cache import FluxKVCache
from dinkster_models import NODES as MODEL_NODES
from dinkster_models.cache import EasyCache, LazyCache


def test_cache_nodes_publish_the_fixed_core_interfaces() -> None:
    assert EasyCache in MODEL_NODES
    assert LazyCache in MODEL_NODES
    assert FluxKVCache in FLUX_NODES
    flux = FluxKVCache.define_schema()
    assert flux.node_type == "dinkster.flux.flux_kv_cache"
    assert flux.category == "experimental"
    assert [(item.id, item.type) for item in flux.inputs] == [
        ("model", TypeExpr.concrete(MODEL_TYPE))
    ]
    assert [(item.id, item.type) for item in flux.outputs] == [
        ("MODEL", TypeExpr.concrete(MODEL_TYPE))
    ]
    for node, node_type in (
        (EasyCache, "dinkster.models.easy_cache"),
        (LazyCache, "dinkster.models.lazy_cache"),
    ):
        schema = node.define_schema()
        assert schema.node_type == node_type
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
