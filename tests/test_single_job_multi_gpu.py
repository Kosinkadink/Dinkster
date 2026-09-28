from __future__ import annotations

from typing import Literal

import pytest
from dinkster_protocol import (
    SingleJobMultiGpuConfig,
    SingleJobMultiGpuExecution,
    single_job_multi_gpu_config_from_wire,
    single_job_multi_gpu_config_to_wire,
    single_job_multi_gpu_execution_from_wire,
    single_job_multi_gpu_execution_to_wire,
)


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


def test_config_rejects_four_way_mode() -> None:
    with pytest.raises(ValueError, match="mode is invalid"):
        SingleJobMultiGpuConfig((0, 1), "four-way")  # type: ignore[arg-type]


def test_config_and_rank_execution_round_trip_strict_wire_contracts() -> None:
    config = SingleJobMultiGpuConfig((3, 1), "sequence")
    execution = SingleJobMultiGpuExecution(1, 2, "sequence")
    assert single_job_multi_gpu_config_from_wire(
        single_job_multi_gpu_config_to_wire(config)
    ) == config
    assert single_job_multi_gpu_execution_from_wire(
        single_job_multi_gpu_execution_to_wire(execution)
    ) == execution
    with pytest.raises(ValueError, match="must contain"):
        single_job_multi_gpu_config_from_wire({"cudaIndices": [0, 1]})
    with pytest.raises(ValueError, match="rank and world size"):
        single_job_multi_gpu_execution_from_wire(
            {"rank": 2, "worldSize": 2, "mode": "sequence"}
        )
