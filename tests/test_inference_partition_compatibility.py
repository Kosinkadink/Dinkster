from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from dinkster_inference.devices import BFLOAT16, FLOAT16, DType
from dinkster_inference.partition_compatibility import (
    ContiguousShard,
    FullSequence,
    PartitionCompatibility,
    PartitionCompatibilityError,
    Replicated,
    RingSequenceShard,
    UlyssesHeadScatter,
    UlyssesRingHybrid,
    require_attention_partition_feasibility,
)
from dinkster_inference.process_mesh import ProcessMesh
from dinkster_inference.sequence_partition import plan_sequence_partition


def compatibility(
    *modes: Replicated | UlyssesHeadScatter | RingSequenceShard | UlyssesRingHybrid,
    expectation: FullSequence | ContiguousShard | None = None,
    normalization: bool = True,
) -> PartitionCompatibility:
    return PartitionCompatibility(
        revision=1,
        modes=modes,
        tensor_expectation=expectation or ContiguousShard(1),
        supported_dtypes=(FLOAT16,),
        device_kinds=("cuda",),
        provides_matching_block_normalization=normalization,
    )


@pytest.mark.parametrize(
    ("mode", "ulysses", "ring"),
    [
        (Replicated(), 1, 1),
        (UlyssesHeadScatter(), 2, 1),
        (RingSequenceShard(), 1, 2),
        (UlyssesRingHybrid(), 2, 2),
    ],
)
def test_each_partition_mode_uses_existing_planner(
    mode: Replicated | UlyssesHeadScatter | RingSequenceShard | UlyssesRingHybrid,
    ulysses: int,
    ring: int,
) -> None:
    mesh = ProcessMesh(guidance=1, tp=1, sp_ulysses=ulysses, sp_ring=ring)
    result = require_attention_partition_feasibility(
        8, 8, 1, mesh, compatibility(mode), FLOAT16, "cuda"
    )
    assert result == plan_sequence_partition(8, ulysses * ring)


def test_guidance_and_tp_do_not_multiply_sequence_shards() -> None:
    mesh = ProcessMesh(guidance=3, tp=5, sp_ulysses=2, sp_ring=2)
    result = require_attention_partition_feasibility(
        8, 8, 1, mesh, compatibility(UlyssesRingHybrid()), FLOAT16, "cuda"
    )
    assert result.shard_count == 4


@pytest.mark.parametrize(
    ("mesh", "message"),
    [
        (ProcessMesh(1, 1, 2, 1), "head_count"),
        (ProcessMesh(1, 1, 2, 2), "head_count"),
    ],
)
def test_ulysses_requires_head_divisibility(mesh: ProcessMesh, message: str) -> None:
    mode = UlyssesHeadScatter() if mesh.sp_ring == 1 else UlyssesRingHybrid()
    with pytest.raises(PartitionCompatibilityError, match=message):
        require_attention_partition_feasibility(3, 8, 1, mesh, compatibility(mode), FLOAT16, "cuda")


@pytest.mark.parametrize("length", [3, 5])
def test_existing_planner_rejects_short_or_empty_tail_partitions(length: int) -> None:
    mesh = ProcessMesh(1, 1, 2, 2)
    with pytest.raises(PartitionCompatibilityError, match="sequence partition is infeasible"):
        require_attention_partition_feasibility(
            8, length, 1, mesh, compatibility(UlyssesRingHybrid()), FLOAT16, "cuda"
        )


def test_supported_padded_length_is_feasible() -> None:
    mesh = ProcessMesh(1, 1, 2, 2)
    result = require_attention_partition_feasibility(
        8, 7, 1, mesh, compatibility(UlyssesRingHybrid()), FLOAT16, "cuda"
    )
    assert result == plan_sequence_partition(7, 4)
    assert result.padded_length == 8


@pytest.mark.parametrize(
    ("dtype", "device_kind", "message"),
    [(BFLOAT16, "cuda", "dtype"), (FLOAT16, "cpu", "device kind")],
)
def test_dtype_and_device_must_be_declared(dtype: DType, device_kind: str, message: str) -> None:
    with pytest.raises(PartitionCompatibilityError, match=message):
        require_attention_partition_feasibility(
            8,
            8,
            1,
            ProcessMesh(1, 1, 1, 1),
            compatibility(Replicated()),
            dtype,
            device_kind,
        )


def test_required_mode_must_be_declared() -> None:
    with pytest.raises(PartitionCompatibilityError, match="UlyssesHeadScatter"):
        require_attention_partition_feasibility(
            8,
            8,
            1,
            ProcessMesh(1, 1, 2, 1),
            compatibility(Replicated()),
            FLOAT16,
            "cuda",
        )


def test_full_sequence_supports_replicated_without_block_normalization() -> None:
    result = require_attention_partition_feasibility(
        3,
        1,
        1,
        ProcessMesh(2, 3, 1, 1),
        compatibility(Replicated(), expectation=FullSequence(), normalization=False),
        FLOAT16,
        "cuda",
    )
    assert result == plan_sequence_partition(1, 1)


def test_full_sequence_refuses_sequence_parallel_placement() -> None:
    with pytest.raises(PartitionCompatibilityError, match="full-sequence"):
        require_attention_partition_feasibility(
            8,
            8,
            1,
            ProcessMesh(1, 1, 2, 1),
            compatibility(UlyssesHeadScatter(), expectation=FullSequence()),
            FLOAT16,
            "cuda",
        )


@pytest.mark.parametrize(
    ("expectation", "message"),
    [(ContiguousShard(2), "dimension"), (ContiguousShard(1, False), "equal chunks")],
)
def test_contiguous_shard_expectation_must_match_candidate(
    expectation: ContiguousShard, message: str
) -> None:
    with pytest.raises(PartitionCompatibilityError, match=message):
        require_attention_partition_feasibility(
            8,
            8,
            1,
            ProcessMesh(1, 1, 1, 1),
            compatibility(Replicated(), expectation=expectation),
            FLOAT16,
            "cuda",
        )


@pytest.mark.parametrize(
    ("mesh", "mode", "shard_count"),
    [
        (ProcessMesh(1, 1, 1, 2), RingSequenceShard(), 2),
        (ProcessMesh(1, 1, 2, 2), UlyssesRingHybrid(), 4),
    ],
)
def test_ring_requires_matching_block_normalization(
    mesh: ProcessMesh, mode: RingSequenceShard | UlyssesRingHybrid, shard_count: int
) -> None:
    with pytest.raises(PartitionCompatibilityError, match="same local block callable"):
        require_attention_partition_feasibility(
            8,
            8,
            1,
            mesh,
            compatibility(mode, normalization=False),
            FLOAT16,
            "cuda",
        )
    assert require_attention_partition_feasibility(
        8, 8, 1, mesh, compatibility(mode), FLOAT16, "cuda"
    ) == plan_sequence_partition(8, shard_count)


def test_non_ring_does_not_require_block_normalization() -> None:
    mesh = ProcessMesh(1, 1, 2, 1)
    assert require_attention_partition_feasibility(
        8,
        8,
        1,
        mesh,
        compatibility(UlyssesHeadScatter(), normalization=False),
        FLOAT16,
        "cuda",
    ) == plan_sequence_partition(8, 2)


@pytest.mark.parametrize("revision", [True, 0, 2])
def test_declaration_rejects_unknown_or_bool_revision(revision: object) -> None:
    with pytest.raises(PartitionCompatibilityError, match="revision"):
        PartitionCompatibility(
            revision=revision,  # type: ignore[arg-type]
            modes=(Replicated(),),
            tensor_expectation=FullSequence(),
            supported_dtypes=(FLOAT16,),
            device_kinds=("cuda",),
            provides_matching_block_normalization=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modes", [Replicated()], "modes"),
        ("modes", (), "nonempty"),
        ("modes", (Replicated(), Replicated()), "duplicates"),
        ("modes", (object(),), "exact supported mode"),
        ("tensor_expectation", object(), "tensor_expectation"),
        ("supported_dtypes", [FLOAT16], "supported_dtypes"),
        ("supported_dtypes", (object(),), "exact DType"),
        ("device_kinds", ["cuda"], "device_kinds"),
        ("device_kinds", ("",), "nonempty strings"),
        ("device_kinds", (1,), "nonempty strings"),
        ("provides_matching_block_normalization", 1, "exact bool"),
    ],
)
def test_declaration_rejects_mutable_or_malformed_values(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "revision": 1,
        "modes": (Replicated(),),
        "tensor_expectation": FullSequence(),
        "supported_dtypes": (FLOAT16,),
        "device_kinds": ("cuda",),
        "provides_matching_block_normalization": False,
    }
    arguments[field] = value
    with pytest.raises(PartitionCompatibilityError, match=message):
        PartitionCompatibility(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("axis", "equal_chunk"),
    [(-1, True), (True, True), (1, 1)],
)
def test_contiguous_shard_rejects_malformed_values(axis: object, equal_chunk: object) -> None:
    with pytest.raises(PartitionCompatibilityError):
        ContiguousShard(axis, equal_chunk)  # type: ignore[arg-type]


def test_declaration_values_are_frozen() -> None:
    declaration = compatibility(Replicated())
    with pytest.raises(FrozenInstanceError):
        declaration.revision = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        declaration.tensor_expectation.sequence_dimension = 2  # type: ignore[union-attr,misc]
    mode = Replicated()
    # Python 3.12 rejects new slots with TypeError rather than FrozenInstanceError.
    with pytest.raises((FrozenInstanceError, TypeError)):
        mode.extra = True  # type: ignore[attr-defined]
    assert not hasattr(mode, "extra")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_count", True),
        ("sequence_length", 0),
        ("sequence_dimension", True),
        ("mesh", object()),
        ("compatibility", object()),
        ("dtype", object()),
        ("device_kind", ""),
    ],
)
def test_feasibility_rejects_malformed_candidate_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "head_count": 8,
        "sequence_length": 8,
        "sequence_dimension": 1,
        "mesh": ProcessMesh(1, 1, 1, 1),
        "compatibility": compatibility(Replicated()),
        "dtype": FLOAT16,
        "device_kind": "cuda",
    }
    arguments[field] = value
    with pytest.raises(PartitionCompatibilityError):
        require_attention_partition_feasibility(**arguments)  # type: ignore[arg-type]


def test_module_import_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dinkster_inference.partition_compatibility as module; "
                "assert 'torch' not in sys.modules; print(module.__file__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()).resolve() == (
        Path(__file__).resolve().parents[1]
        / "packages/dinkster-inference/src/dinkster_inference/partition_compatibility.py"
    )
