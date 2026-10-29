from __future__ import annotations

from typing import Literal

import pytest
from dinkster_workers import SingleJobMultiGpuConfig


@pytest.mark.parametrize("mode", ("auto", "guidance", "sequence", "window"))
def test_fixed_configuration_preserves_order_and_mode(
    mode: Literal["auto", "guidance", "sequence", "window"],
) -> None:
    config = SingleJobMultiGpuConfig((3, 1, 2), mode)
    assert config.cuda_indices == (3, 1, 2)
    assert config.mode == mode


@pytest.mark.parametrize("indices", ((), (0,), (0, 0), (-1, 0)))
def test_config_requires_two_or_more_unique_non_negative_logical_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="at least two"):
        SingleJobMultiGpuConfig(indices, "auto")


def test_config_rejects_removed_model_mode() -> None:
    with pytest.raises(ValueError, match="mode is invalid"):
        SingleJobMultiGpuConfig((0, 1), "model")  # type: ignore[arg-type]
