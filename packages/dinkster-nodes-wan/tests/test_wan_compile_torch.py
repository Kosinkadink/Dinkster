"""Wan block swap remains a deprecated identity operation."""

from dinkster_wan.compile import wanBlockSwap


def test_wan_block_swap_is_a_deprecated_identity_node() -> None:
    model = object()
    schema = wanBlockSwap.schema()

    assert schema.node_type == "dinkster.wan.wan_block_swap"
    assert schema.deprecation is not None
    assert schema.search_visibility == "deprecated"
    assert wanBlockSwap.execute(model=model)["MODEL"] is model
