from __future__ import annotations

import pytest
from dinkster_protocol import (
    SingleJobMultiGpuConfig,
    SingleJobMultiGpuExecution,
    single_job_multi_gpu_config_from_wire,
    single_job_multi_gpu_config_to_wire,
    single_job_multi_gpu_execution_from_wire,
    single_job_multi_gpu_execution_to_wire,
)


def test_config_and_rank_execution_round_trip_strict_wire_contracts() -> None:
    config = SingleJobMultiGpuConfig((3, 1), "sequence")
    execution = SingleJobMultiGpuExecution(1, 2, "sequence")
    assert (
        single_job_multi_gpu_config_from_wire(single_job_multi_gpu_config_to_wire(config)) == config
    )
    assert (
        single_job_multi_gpu_execution_from_wire(single_job_multi_gpu_execution_to_wire(execution))
        == execution
    )
    with pytest.raises(ValueError, match="must contain"):
        single_job_multi_gpu_config_from_wire({"cudaIndices": [0, 1]})
    with pytest.raises(ValueError, match="rank and world size"):
        single_job_multi_gpu_execution_from_wire({"rank": 2, "worldSize": 2, "mode": "sequence"})
