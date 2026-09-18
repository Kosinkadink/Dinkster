from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import get_ident

import pytest
import torch
from dinkster_inference import (
    Conditioning,
    GuidanceCondition,
    GuidanceEvaluationPlan,
    GuidanceEvaluationRequest,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    SamplingCancelled,
    sampling_execution_context,
)
from dinkster_inference_torch import distributed


def test_attempt_scoped_configuration_changes_rendezvous_and_resets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RANK", "0")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_MULTI_GPU_MODE", "guidance")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RENDEZVOUS", f"file://{tmp_path / 'group'}")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_TOKEN", "1" * 32)

    distributed.activate_distributed_sampling_attempt("first", 1)
    first = distributed.distributed_sampling_config()
    assert first is not None
    destroyed: list[bool] = []
    monkeypatch.setattr(distributed, "_config", first)
    monkeypatch.setattr(distributed, "_destroy_process_group", lambda: destroyed.append(True))
    distributed.release_distributed_sampling_attempt("first", 1)
    assert distributed.distributed_sampling_config() is None

    distributed.activate_distributed_sampling_attempt("second", 1)
    second = distributed.distributed_sampling_config()
    assert second is not None
    distributed.release_distributed_sampling_attempt("second", 1)

    assert first.attempt == "first:1"
    assert second.attempt == "second:1"
    assert first.rendezvous != second.rendezvous
    assert first.token != second.token
    assert destroyed == [True, True]


def _set_rank_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RANK", "0")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "4")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_MULTI_GPU_MODE", mode)
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RENDEZVOUS", f"file://{tmp_path / 'group'}")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_TOKEN", "1" * 32)


def test_configuration_rejects_removed_model_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "model")
    distributed.activate_distributed_sampling_attempt("removed-model-mode", 1)
    try:
        with pytest.raises(RuntimeError, match="environment is malformed"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("removed-model-mode", 1)


@pytest.mark.parametrize(
    ("guidance", "ulysses", "ring"),
    ((None, "2", "2"), ("1", "2", "2"), ("2", "2", "1")),
)
def test_sequence_configuration_resolves_exact_geometry(
    guidance: str | None,
    ulysses: str,
    ring: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "sequence")
    if guidance is not None:
        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE", guidance)
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES", ulysses)
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_RING", ring)
    distributed.activate_distributed_sampling_attempt("sequence", 1)
    try:
        config = distributed.distributed_sampling_config()
        assert config is not None
        assert (
            config.sequence_guidance,
            config.sequence_ulysses,
            config.sequence_ring,
        ) == (int(guidance or "1"), int(ulysses), int(ring))
    finally:
        distributed.release_distributed_sampling_attempt("sequence", 1)


@pytest.mark.parametrize(
    ("guidance", "ulysses", "ring"),
    (
        (None, None, "4"),
        (None, "2", None),
        (None, "0", "4"),
        (None, "two", "2"),
        (None, "2", "3"),
        ("0", "2", "2"),
        ("-1", "2", "2"),
        ("two", "2", "2"),
        ("2", "2", "2"),
    ),
)
def test_sequence_configuration_rejects_bad_geometry_before_collectives(
    guidance: str | None,
    ulysses: str | None,
    ring: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "sequence")
    if guidance is not None:
        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE", guidance)
    if ulysses is not None:
        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES", ulysses)
    if ring is not None:
        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_RING", ring)
    distributed.activate_distributed_sampling_attempt("invalid-sequence", 1)
    try:
        with pytest.raises(RuntimeError, match="sequence geometry"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("invalid-sequence", 1)


@pytest.mark.parametrize(
    ("transport", "expected"),
    ((None, "nccl"), ("nccl", "nccl"), ("peer-copy", "peer-copy")),
)
def test_sequence_transport_resolves_and_defaults_to_nccl(
    transport: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "sequence")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_RING", "2")
    if transport is not None:
        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", transport)
    distributed.activate_distributed_sampling_attempt("sequence-transport", 1)
    try:
        config = distributed.distributed_sampling_config()
        assert config is not None
        assert config.sequence_transport == expected
    finally:
        distributed.release_distributed_sampling_attempt("sequence-transport", 1)


@pytest.mark.parametrize("transport", ("", "peer", "nvlink", "PEER-COPY"))
def test_sequence_transport_rejects_unknown_values(
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "sequence")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_RING", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", transport)
    distributed.activate_distributed_sampling_attempt("bad-transport", 1)
    try:
        with pytest.raises(RuntimeError, match="nccl or peer-copy"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("bad-transport", 1)


def test_sequence_transport_rejects_other_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "guidance")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", "peer-copy")
    distributed.activate_distributed_sampling_attempt("transport-mode", 1)
    try:
        with pytest.raises(RuntimeError, match="requires sequence mode"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("transport-mode", 1)


@pytest.mark.parametrize(
    "variable",
    (
        "DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE",
        "DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES",
        "DINKSTER_SINGLE_JOB_SEQUENCE_RING",
    ),
)
def test_sequence_geometry_rejects_other_modes(
    variable: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_rank_environment(monkeypatch, tmp_path, "guidance")
    monkeypatch.setenv(variable, "2")
    distributed.activate_distributed_sampling_attempt("conflicting-sequence", 1)
    try:
        with pytest.raises(RuntimeError, match="requires sequence mode"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("conflicting-sequence", 1)


def test_collective_fence_identity_binds_attempt_and_operation() -> None:
    first = distributed.DistributedSamplingConfig(0, 2, "auto", "file:///a", "1" * 32, "a:1")
    second = distributed.DistributedSamplingConfig(0, 2, "auto", "file:///b", "2" * 32, "b:1")

    assert distributed._fence_identity(first, "guidance") != distributed._fence_identity(  # pyright: ignore[reportPrivateUsage]
        first,
        "sampling",  # pyright: ignore[reportPrivateUsage]
    )
    assert distributed._fence_identity(first, "guidance") != distributed._fence_identity(  # pyright: ignore[reportPrivateUsage]
        second,
        "guidance",  # pyright: ignore[reportPrivateUsage]
    )


def _run_sequence_fence_disagreement(
    rank: int,
    rendezvous: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            2,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=2,
            sequence_ring=1,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="sequence-parallel sideband control disagrees"):
            distributed.fence_sequence_evaluation(
                3,
                torch.zeros(1, 2 + rank, 4),
                lane_identity="conditional",
                sequence_identity="packed-facts-a",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_sequence_fence_rejects_real_cross_rank_contract_disagreement() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "sequence-fence-rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_sequence_fence_disagreement,
                args=(rank, rendezvous),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def _run_grouped_sequence_fence(
    rank: int,
    rendezvous: str,
    disagree_within_group: bool,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=10),
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            4,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=2,
            sequence_ring=1,
            sequence_guidance=2,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        sequence_identity = (
            "packed-facts-b" if disagree_within_group and rank == 1 else "packed-facts-a"
        )
        lane_identity = "conditional" if rank < 2 else "unconditional"
        if disagree_within_group and rank < 2:
            with pytest.raises(RuntimeError, match="sequence-parallel sideband control disagrees"):
                distributed.fence_sequence_evaluation(
                    3,
                    torch.zeros(1, 2, 4),
                    lane_identity=lane_identity,
                    sequence_identity=sequence_identity,
                )
        else:
            distributed.fence_sequence_evaluation(
                3,
                torch.zeros(1, 2, 4),
                lane_identity=lane_identity,
                sequence_identity=sequence_identity,
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("disagree_within_group", (False, True))
def test_sequence_fence_scopes_lane_facts_to_guidance_groups(
    disagree_within_group: bool,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "grouped-sequence-fence-rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_grouped_sequence_fence,
                args=(rank, rendezvous, disagree_within_group),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def _transport_config(
    world_size: int,
    mode: str = "guidance",
    *,
    sequence_ulysses: int = 1,
    sequence_ring: int = 1,
) -> distributed.DistributedSamplingConfig:
    return distributed.DistributedSamplingConfig(
        0,
        world_size,
        mode,
        "file:///group",
        "1" * 32,
        sequence_ulysses=sequence_ulysses,
        sequence_ring=sequence_ring,
    )


def test_two_ranks_on_amd_preserve_nccl_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

    def read_text(_path: Path) -> str:
        return "vendor_id: AuthenticAMD"

    monkeypatch.setattr(Path, "read_text", read_text)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(2)
    )

    assert "NCCL_P2P_LEVEL" not in os.environ


def test_three_ranks_on_non_amd_preserve_nccl_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

    def read_text(_path: Path) -> str:
        return "vendor_id: GenuineIntel"

    monkeypatch.setattr(Path, "read_text", read_text)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(3)
    )

    assert "NCCL_P2P_LEVEL" not in os.environ


def test_three_ranks_on_amd_default_nccl_p2p_to_phb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

    def read_text(_path: Path) -> str:
        return "vendor_id: AuthenticAMD"

    monkeypatch.setattr(Path, "read_text", read_text)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(3)
    )

    assert os.environ["NCCL_P2P_LEVEL"] == "PHB"


@pytest.mark.parametrize(
    ("ulysses", "ring"),
    ((4, 1), (1, 4), (2, 2)),
)
def test_sequence_parallel_on_amd_preserves_nccl_default_without_host_probe(
    ulysses: int,
    ring: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

    def unexpected_read(_path: Path) -> str:
        raise AssertionError("sequence-parallel transport must avoid probing the host")

    monkeypatch.setattr(Path, "read_text", unexpected_read)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(4, "sequence", sequence_ulysses=ulysses, sequence_ring=ring)
    )

    assert "NCCL_P2P_LEVEL" not in os.environ


def test_unreadable_cpuinfo_preserves_nccl_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)

    def read_text(_path: Path) -> str:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(Path, "read_text", read_text)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(3)
    )

    assert "NCCL_P2P_LEVEL" not in os.environ


def test_explicit_nccl_p2p_level_is_preserved_without_host_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NCCL_P2P_LEVEL", "unusual-caller-value")

    def unexpected_read(_path: Path) -> str:
        raise AssertionError("caller override must avoid probing the host")

    monkeypatch.setattr(Path, "read_text", unexpected_read)

    distributed._configure_nccl_p2p_level(  # pyright: ignore[reportPrivateUsage]
        _transport_config(3, "sequence", sequence_ulysses=3)
    )

    assert os.environ["NCCL_P2P_LEVEL"] == "unusual-caller-value"


_SEQUENCE_TEST_FACTS = (
    "topology=sequence",
    "execution_provider=bf16-linear",
    "sequence_ulysses=2",
    "sequence_ring=1",
    "sequence_guidance=1",
)
_MINIMAX_H3_BF16_SDPA_U2_FACTS = (
    "topology=sequence",
    "execution_provider=bf16-linear",
    "attention_provider=torch-sdpa-priority-v1",
    "attention_provider_version=builtin",
    "torch_version=2.13.0+cu130",
    "sequence_ulysses=2",
    "sequence_ring=1",
    "sequence_guidance=1",
)
_MINIMAX_H3_INT8_ATTENTION_U2_FACTS = (
    "topology=sequence",
    "execution_provider=bf16-linear",
    "attention_provider=comfy-kitchen-int8-attention-v1",
    "attention_provider_version=0.2.31",
    "torch_version=2.13.0+cu130",
    "sequence_ulysses=2",
    "sequence_ring=1",
    "sequence_guidance=1",
)


def test_sequence_receipt_identity_is_domain_separated_from_guidance() -> None:
    identity = distributed.sequence_receipt_identity(
        "dinkster.minimax_h3", _SEQUENCE_TEST_FACTS, torch.bfloat16, 2
    )
    assert identity == distributed.receipt_identity_for_domain(
        "domain=dinkster.distributed.sequence-receipt.v2",
        "dinkster.minimax_h3",
        _SEQUENCE_TEST_FACTS,
        torch.bfloat16,
        2,
        prefix="distributed",
    )
    assert identity != distributed.guidance_receipt_identity(
        "dinkster.minimax_h3", _SEQUENCE_TEST_FACTS, torch.bfloat16, 2
    )


@pytest.mark.parametrize("mode", ("guidance", "sequence", "window"))
def test_distributed_identity_excludes_software_builds(mode: str) -> None:
    from dinkster_inference_torch.flux_window_distributed import window_receipt_identity

    build_identity = {
        "guidance": distributed.guidance_receipt_identity,
        "sequence": distributed.sequence_receipt_identity,
        "window": window_receipt_identity,
    }[mode]
    facts = (
        f"topology={mode}",
        "attention_provider=comfy-kitchen-int8-attention-v1",
        "sequence_ulysses=2",
    )
    expected = build_identity("dinkster.synthetic", facts, torch.bfloat16, 2)
    for version in ("0.2.31", "unseen-version"):
        observed = build_identity(
            "dinkster.synthetic",
            (*facts, f"attention_provider_version={version}", f"torch_version={version}"),
            torch.bfloat16,
            2,
        )
        assert observed == expected
    for changed_facts, dtype, world_size in (
        ((*facts[:-1], "sequence_ulysses=4"), torch.bfloat16, 2),
        (facts, torch.float32, 2),
        (facts, torch.bfloat16, 3),
    ):
        assert build_identity("dinkster.synthetic", changed_facts, dtype, world_size) != expected


def _run_data_plane(
    rank: int, world_size: int, rendezvous: str, fail_rank_zero: bool = False
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            world_size,
            "guidance",
            f"file://{rendezvous}",
            "1" * 32,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        lanes = (
            GuidanceCondition(
                "positive",
                GuidanceRole.CONDITIONAL,
                Conditioning(torch.tensor([[[1.0]]])),
            ),
            GuidanceCondition(
                "negative",
                GuidanceRole.UNCONDITIONAL,
                Conditioning(torch.tensor([[[2.0]]])),
            ),
        )
        request = GuidanceEvaluationRequest(
            torch.zeros(1, 1, 2, 2),
            torch.tensor(1.0),
            GuidanceEvaluationPlan(lanes, "positive", "negative"),
            sampling_execution_context((1.0, 0.0), 7),
        )

        def evaluate(
            x: torch.Tensor,
            _sigma: float,
            local: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            if fail_rank_zero and rank == 0:
                raise RuntimeError("local guidance failed")
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(
                        lane.id,
                        x + lane.conditioning.embeddings.item(),
                        GuidancePredictionSource.MODEL,
                    )
                    for lane in local.plan.lanes
                    if lane.conditioning is not None
                )
            )

        evaluator = distributed.DistributedGuidanceEvaluator(evaluate)
        if fail_rank_zero:
            with pytest.raises(RuntimeError, match="guidance .*failed"):
                evaluator.evaluate_request(request.input, 1.0, request)
            return
        predictions = evaluator.evaluate_request(request.input, 1.0, request)
        assert [item.lane_id for item in predictions.items] == ["positive", "negative"]
        assert torch.equal(predictions.items[0].value, torch.ones_like(request.input))
        assert torch.equal(predictions.items[1].value, torch.full_like(request.input, 2.0))

    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_guidance_data_plane_covers_every_rank(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_data_plane,
                args=(rank, world_size, rendezvous),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def test_guidance_failure_reaches_every_rank_without_a_collective_hang() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_data_plane,
                args=(rank, 2, rendezvous, True),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def _run_rank_zero_sampling(rank: int, rendezvous: str, outcome: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=1),
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            2,
            "guidance",
            f"file://{rendezvous}",
            "1" * 32,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        callbacks: list[int] = []
        template = torch.zeros(2, 3)
        caller_thread = get_ident()

        def action() -> torch.Tensor:
            assert distributed.rank_zero_sampling_active()
            assert get_ident() == caller_thread
            callbacks.append(rank)
            if outcome == "cancel":
                raise SamplingCancelled("cancelled by callback")
            if outcome == "fail":
                raise RuntimeError("failed by callback")
            if outcome == "slow":
                time.sleep(2.0)
            return torch.full_like(template, 7.0)

        if outcome == "cancel":
            with pytest.raises(SamplingCancelled, match="cancel"):
                distributed.run_rank_zero_sampling(action, template, config)
        elif outcome == "fail":
            with pytest.raises(RuntimeError, match="fail"):
                distributed.run_rank_zero_sampling(action, template, config)
        else:
            output = distributed.run_rank_zero_sampling(action, template, config)
            assert torch.equal(output, torch.full_like(template, 7.0))
        assert callbacks == ([0] if rank == 0 else [])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("outcome", ("success", "slow", "cancel", "fail"))
def test_rank_zero_sampling_completes_after_results_cancellation_and_failure(
    outcome: str,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_rank_zero_sampling,
                args=(rank, rendezvous, outcome),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def test_rank_zero_sampling_hides_distributed_configuration_from_local_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RANK", "0")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_MULTI_GPU_MODE", "guidance")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RENDEZVOUS", f"file://{tmp_path / 'group'}")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_TOKEN", "1" * 32)
    distributed.activate_distributed_sampling_attempt("rank-zero", 1)
    try:
        config = distributed.distributed_sampling_config()
        assert config is not None
        monkeypatch.setattr(distributed, "ensure_process_group", lambda: config)
        monkeypatch.setattr(distributed, "_rank_zero_control_group_for_sampling", lambda: None)

        def broadcast(_tensor: torch.Tensor, src: int, group: object | None = None) -> None:
            assert src == 0
            assert group is None

        monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
        template = torch.zeros(1)

        def action() -> torch.Tensor:
            assert distributed.distributed_sampling_config() is None
            return template

        assert distributed.run_rank_zero_sampling(action, template, config) is template
    finally:
        distributed.release_distributed_sampling_attempt("rank-zero", 1)


def _run_rank_zero_sampling_peer_failure(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=1),
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            2,
            "guidance",
            f"file://{rendezvous}",
            "1" * 32,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        template = torch.zeros(1)
        if rank == 1:
            broadcast = torch.distributed.broadcast

            def exit_after_first_heartbeat(
                tensor: torch.Tensor,
                src: int,
                group: torch.distributed.ProcessGroup | None = None,
            ) -> None:
                broadcast(tensor, src=src, group=group)
                if tensor.device.type == "cpu" and tensor.numel() == 2:
                    os._exit(17)

            torch.distributed.broadcast = exit_after_first_heartbeat  # type: ignore[assignment]
            distributed.run_rank_zero_sampling(lambda: template, template, config)
            raise AssertionError("peer did not exit during rank-zero work")

        stopped = False
        started = time.monotonic()
        execution = sampling_execution_context((1.0, 0.0), 7)

        def action() -> torch.Tensor:
            nonlocal stopped
            try:
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    execution.cancellation.check()
                    time.sleep(0.05)
                raise AssertionError("peer failure did not cancel rank-zero work")
            finally:
                stopped = True

        with pytest.raises(RuntimeError, match="rank-zero sampling control failed"):
            distributed.run_rank_zero_sampling(action, template, config)
        assert stopped
        assert time.monotonic() - started < 8.0
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_rank_zero_sampling_stops_when_a_peer_fails_during_work() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_rank_zero_sampling_peer_failure,
                args=(rank, rendezvous),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=20)
            assert processes[0].exitcode == 0
            assert processes[1].exitcode == 17
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)


def _run_step_callback_desync(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        config = distributed.DistributedSamplingConfig(
            rank,
            2,
            "guidance",
            f"file://{rendezvous}",
            "1" * 32,
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        lanes = (
            GuidanceCondition(
                "positive",
                GuidanceRole.CONDITIONAL,
                Conditioning(torch.tensor([[[1.0]]])),
            ),
            GuidanceCondition(
                "negative",
                GuidanceRole.UNCONDITIONAL,
                Conditioning(torch.tensor([[[2.0]]])),
            ),
        )
        execution = sampling_execution_context((1.0, 0.5, 0.0), 7)

        def evaluate(
            x: torch.Tensor,
            _sigma: float,
            local: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(
                        lane.id,
                        x + lane.conditioning.embeddings.item(),
                        GuidancePredictionSource.MODEL,
                    )
                    for lane in local.plan.lanes
                    if lane.conditioning is not None
                )
            )

        evaluator = distributed.DistributedGuidanceEvaluator(evaluate)

        def request_at(ordinal: int) -> GuidanceEvaluationRequest[torch.Tensor]:
            return GuidanceEvaluationRequest(
                torch.zeros(1, 1, 2, 2),
                torch.tensor(1.0),
                GuidanceEvaluationPlan(lanes, "positive", "negative"),
                replace(execution, model_evaluation=ordinal),
            )

        def on_step(step: int) -> None:
            if rank == 0 and step == 0:
                raise RuntimeError("step callback failed")

        completed_steps: list[int] = []

        def run_lockstep_sampler_loop() -> None:
            # Guidance mode is SPMD: every rank runs this same sampler loop
            # and serves its share of each guidance collective from within
            # evaluate_request; there is no separate service loop. The step
            # callback therefore fires between fences, never inside one.
            for step in range(2):
                request = request_at(step)
                evaluator.evaluate_request(request.input, 1.0, request)
                completed_steps.append(step)
                on_step(step)

        if rank == 0:
            with pytest.raises(RuntimeError, match="step callback failed"):
                run_lockstep_sampler_loop()
            # The raising callback stopped the loop before the next fence.
            assert completed_steps == [0]
        else:
            with pytest.raises(RuntimeError) as failure:
                run_lockstep_sampler_loop()
            # The peer rank fails closed inside the abandoned fence instead
            # of completing a desynchronized collective.
            assert "step callback failed" not in str(failure.value)
            assert completed_steps == [0]
    finally:
        torch.distributed.destroy_process_group()


def test_step_callback_failure_cannot_desync_a_guidance_collective() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "step-callback-rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=_run_step_callback_desync,
                args=(rank, rendezvous),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0


def _run_shared_sampler_engine(rank: int, rendezvous: str, failure: str, cfg_scale: float) -> None:
    from dinkster_inference import (
        FLUX_DEV,
        Parameterization,
        SamplingCancelled,
        SamplingGuidance,
        use_sampling_environment,
    )
    from dinkster_inference_torch.denoise import run_sampler_engine
    from dinkster_inference_torch.guidance import ConditioningEvaluation
    from dinkster_inference_torch.sampling_execution import compile_guidance_plan, guided_denoiser
    from dinkster_inference_torch.solvers import torch_sampler_registry

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    config = distributed.DistributedSamplingConfig(
        rank, 2, "guidance", f"file://{rendezvous}", "1" * 32
    )
    distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
    distributed.distributed_sampling_config = lambda: config  # type: ignore[assignment]
    try:
        sampler = torch_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        cond = Conditioning(torch.ones((1, 2, 3)))
        cfg = SamplingGuidance(Conditioning(torch.zeros((1, 2, 3))), cfg_scale)
        evaluations: list[int] = []
        steps: list[object] = []
        states: list[object] = []
        cancellation = [False]

        def step(event: object) -> None:
            steps.append(event)
            if failure == "step":
                raise RuntimeError("step callback failed")
            if failure == "cancel":
                cancellation[0] = True

        def state(event: object) -> None:
            states.append(event)
            if failure == "state":
                raise RuntimeError("state callback failed")

        def evaluate(x: torch.Tensor, sigma: float, condition: Conditioning[torch.Tensor]):
            evaluations.append(rank)
            return x * 0.5 + condition.embeddings.mean()

        def sample() -> torch.Tensor:
            denoiser = guided_denoiser(
                ConditioningEvaluation(lambda value, _role: value, evaluate),
                input=torch.zeros((1, 16, 2, 2)),
                executor=None,
                plan=compile_guidance_plan(cond, cfg, sampler, None),
                execution=sampling_execution_context((1.0, 0.5, 0.0), 7),
            )
            return run_sampler_engine(
                denoiser,
                sampler.build(),
                latent=torch.zeros((1, 16, 2, 2)),
                noise=torch.ones((1, 16, 2, 2)),
                sigmas=(1.0, 0.5, 0.0),
                parameterization=Parameterization.FLOW,
                sigma_max=FLUX_DEV.sampling.sigma_max,
                process_in=lambda value: value,
                process_out=lambda value: value,
                on_step=step if rank == 0 else None,
                on_state=state if rank == 0 else None,
                denoise_mask=torch.tensor([0.0, 1.0]).expand(1, 16, 2, 2),
            )

        with use_sampling_environment((), cancelled=lambda: cancellation[0]):
            if failure:
                expected = SamplingCancelled if failure == "cancel" else RuntimeError
                with pytest.raises(expected, match="cancelled|callback.*failed"):
                    sample()
            else:
                output = sample()
                distributed.distributed_sampling_config = lambda: None  # type: ignore[assignment]
                assert torch.equal(output, sample())
                assert torch.count_nonzero(output[..., 0]) == 0
                if rank == 0:
                    assert len(steps) == len(states) == 4
                else:
                    assert not steps and not states
        if failure:
            assert len(steps) <= 1
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("failure", ("", "step", "state", "cancel"))
@pytest.mark.parametrize("cfg_scale", (1.0, 2.0))
def test_shared_engine_callbacks_masks_and_cancellation_across_ranks(
    tmp_path: Path, failure: str, cfg_scale: float
) -> None:
    processes = [
        multiprocessing.get_context("spawn").Process(
            target=_run_shared_sampler_engine,
            args=(rank, str(tmp_path / "engine-rendezvous"), failure, cfg_scale),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
