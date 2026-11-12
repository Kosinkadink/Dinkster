"""Wan block swap remains a deprecated identity operation."""

from dinkster_wan.compile import wanBlockSwap


def test_wan_block_swap_is_a_deprecated_identity_node() -> None:
    model = object()
    assert wanBlockSwap.execute(model=model)["MODEL"] is model
