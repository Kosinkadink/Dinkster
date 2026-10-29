"""Distributed window scatter: consensus gating, fencing, bitwise parity.

Window scatter is circumstantial acceleration of windowed sampling and
is never counted toward the core multi-GPU goals (#121/#298). These
tests pin the distributed-safety contract: no scatter collective
without a minted manifest consensus token, divergent plans refuse
before any collective, symmetric failure propagation, and bitwise
equality with the serial windowed merge.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import tempfile
from datetime import timedelta

import pytest
import torch
from dinkster_inference import (
    AccumulationDType,
    CanonicalManifest,
    Conditioning,
    IntegerAffineIndexMap,
    KindAxisMap,
    LayerWindow,
    ManifestConsensusToken,
    ManifestRefusal,
    ManifestRefusalCode,
    MediaAxis,
    MergeDeclaration,
    OptionValue,
    Parameterization,
    SamplerInfo,
    StepEvent,
    WindowIndexList,
    WindowKind,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    compile_window_plan,
    prove_manifest_consensus,
)
from dinkster_inference_torch import distributed, torch_euler
from dinkster_inference_torch import flux_window_distributed as window_distributed
from dinkster_inference_torch._conditioning_layout import (
    bind_flux_layout,
    declare_text_conditioning,
)
from dinkster_inference_torch.distributed import (
    DistributedSamplingConfig,
    collective_fence_identity,
    guidance_receipt_identity,
)
from dinkster_inference_torch.flux_window import (
    FluxWindowConditioningEvaluation,
    FluxWindowError,
    PreparedFluxWindowConditioning,
    PreparedFluxWindowPlan,
    prepare_flux_window_plan,
)
from dinkster_inference_torch.flux_window_distributed import (
    DistributedFluxWindowEvaluation,
    WindowDigestConsensusTransport,
    build_flux_window_manifest,
    flux_window_plan_binding,
    window_preflight_failed,
    window_receipt_identity,
    window_route_mismatch,
)
from dinkster_inference_torch.guidance import (
    evaluate_conditioning_batch as _engine_evaluate_conditioning_batch,
)

_RUNTIME_IDENTITY = f"native:dinkster.flux-test:{hashlib.sha256(b'window').hexdigest()}"
_FACTS = ("topology=window", "window_evaluation=full-replica-window-scatter")


def _kind_declarations(
    axes: tuple[MediaAxis, ...],
) -> tuple[WindowKind, WindowKind]:
    names = tuple(axis.name for axis in axes)
    return (
        WindowKind(
            "latent_image",
            tuple(KindAxisMap(axis.name, axis.extent, IntegerAffineIndexMap(1)) for axis in axes),
        ),
        WindowKind("text", invariant_axes=names),
    )


def _prepared_plan(
    *,
    window_count: int = 2,
    weighted: bool = False,
) -> PreparedFluxWindowPlan:
    axes = (
        MediaAxis("height", 2, wrappable=True),
        MediaAxis("width", 3, wrappable=True),
    )
    if window_count == 1:
        windows = [LayerWindow((WindowIndexList((0, 1)), WindowIndexList((0, 1, 2))))]
    else:
        windows = [
            LayerWindow(
                (
                    WindowIndexList((0, 1)),
                    WindowIndexList((2, 3, 5), modular=True),
                )
            ),
            LayerWindow((WindowIndexList((0, 1)), WindowIndexList((1,)))),
            LayerWindow((WindowIndexList((0, 1)), WindowIndexList((0,)))),
        ][:window_count]
    layer = WindowPlanLayer(
        ("height", "width"),
        tuple(windows),
        (
            WindowWeightProfile(WindowWeightKind.FLAT),
            WindowWeightProfile(
                WindowWeightKind.OVERLAP_LINEAR if weighted else WindowWeightKind.FLAT,
                overlap=1 if weighted else 0,
            ),
        ),
        MergeDeclaration(AccumulationDType.FLOAT64),
    )
    plan = compile_window_plan(axes=axes, kinds=_kind_declarations(axes), layers=(layer,))
    return prepare_flux_window_plan(plan, latent_height=4, latent_width=6, patch_size=2)


class _ScaledEvaluator:
    """Deterministic per-window toy denoiser distinguishing window and lane."""

    def __init__(self, index: int) -> None:
        self._index = index

    @staticmethod
    def prepare_conditioning(
        conditioning: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(conditioning, Conditioning)
        return conditioning.embeddings, conditioning.pooled

    @staticmethod
    def batchable(
        conditions: tuple[tuple[torch.Tensor, torch.Tensor | None], ...],
    ) -> bool:
        return bool(conditions)

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        return x * float(self._index + 2) + sigma + condition[0].sum()

    evaluate_conditioning_batch = _engine_evaluate_conditioning_batch

    def _evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[tuple[torch.Tensor, torch.Tensor | None], ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.evaluate_conditioning(x, sigma, condition) for condition in conditions)


class _ExplodingEvaluator(_ScaledEvaluator):
    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        raise ValueError("window evaluator exploded")


class _LoopbackTransport:
    """In-memory unanimous transport for minting a token without a group."""

    def __init__(self, group_size: int) -> None:
        self._group_size = group_size

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        del rank
        return (digest,) * self._group_size


def _manifest(
    config: DistributedSamplingConfig,
    prepared_plan: PreparedFluxWindowPlan,
    *,
    seed: int = 11,
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
    pre_offset_sigmas: tuple[float, ...] | None = None,
    text_token_counts: tuple[int, ...] = (5,),
    sampler_options: tuple[tuple[str, OptionValue], ...] = (),
) -> CanonicalManifest:
    return build_flux_window_manifest(
        runtime_identity=_RUNTIME_IDENTITY,
        config=config,
        prepared_plan=prepared_plan,
        text_token_counts=text_token_counts,
        sampler_id="euler",
        sampler_options=sampler_options,
        seed=seed,
        pre_offset_sigmas=sigmas if pre_offset_sigmas is None else pre_offset_sigmas,
        sigmas=sigmas,
    )


def _config(rank: int = 0, world_size: int = 2, mode: str = "window") -> DistributedSamplingConfig:
    return DistributedSamplingConfig(rank, world_size, mode, "file:///group", "1" * 32, "a:1")


def test_window_receipt_identity_is_domain_separated() -> None:
    identity = window_receipt_identity("dinkster.flux", _FACTS, torch.bfloat16, 2)

    prefix, family, digest = identity.split(":")
    assert (prefix, family) == ("distributed", "dinkster.flux")
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert identity != guidance_receipt_identity("dinkster.flux", _FACTS, torch.bfloat16, 2)
    assert identity != window_receipt_identity("dinkster.flux", _FACTS, torch.bfloat16, 4)
    assert identity != window_receipt_identity("dinkster.flux", _FACTS, torch.float16, 2)


def test_window_fence_identity_binds_operation_and_attempt() -> None:
    first = _config()
    second = DistributedSamplingConfig(0, 2, "window", "file:///group", "1" * 32, "a:2")

    assert collective_fence_identity(first, "window") != collective_fence_identity(
        first, "guidance"
    )
    assert collective_fence_identity(first, "window") != collective_fence_identity(second, "window")


def test_flux_window_plan_binding_pins_layouts_and_absence_markers() -> None:
    prepared = _prepared_plan()

    binding = flux_window_plan_binding(prepared, text_token_count=5)

    assert binding.plan is prepared.declaration
    assert len(binding.window_token_layout_digests) == 2
    assert binding.window_token_layout_digests[0] != binding.window_token_layout_digests[1]
    absent_partition = hashlib.sha256(
        b"dinkster.flux.window.partition-plan.absent.v1\n"
    ).hexdigest()
    absent_rows = hashlib.sha256(b"dinkster.flux.window.structural-rows.absent.v1\n").hexdigest()
    assert set(binding.window_partition_plan_digests) == {absent_partition}
    assert set(binding.window_structural_row_digests) == {absent_rows}

    with pytest.raises(FluxWindowError, match="exact PreparedFluxWindowPlan"):
        flux_window_plan_binding(object(), text_token_count=5)  # pyright: ignore[reportArgumentType]


def test_flux_window_manifest_binds_invocation_identity() -> None:
    config = _config()
    prepared = _prepared_plan()

    base = _manifest(config, prepared)

    assert base.group_size == 2
    assert base.slots[0].slot == "windowed-evaluation"
    assert base.rank_plan_digests == ((prepared.declaration.digest,),) * 2

    variants = (
        _manifest(config, prepared, seed=12),
        _manifest(config, prepared, sigmas=(1.0, 0.25, 0.0)),
        _manifest(config, prepared, pre_offset_sigmas=(1.25, 0.5, 0.0)),
        _manifest(config, prepared, text_token_counts=(5, 7)),
        _manifest(config, prepared, sampler_options=(("s_churn", 0.1),)),
        _manifest(config, _prepared_plan(weighted=True)),
        _manifest(_config(world_size=4), prepared),
    )
    digests = {base.digest, *(variant.digest for variant in variants)}
    assert len(digests) == 8

    with pytest.raises(FluxWindowError, match="at least one declared token count"):
        _manifest(config, prepared, text_token_counts=())


def test_window_consensus_transport_validates_configuration() -> None:
    config = _config()

    transport = WindowDigestConsensusTransport(config, torch.device("cpu"))
    assert transport.rank == 0
    assert transport.physical_ranks == (0, 1)

    with pytest.raises(ValueError, match="window-eligible"):
        WindowDigestConsensusTransport(_config(mode="guidance"), torch.device("cpu"))
    with pytest.raises(TypeError, match="exact torch.device"):
        WindowDigestConsensusTransport(config, "cpu")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="does not match the configured window rank"):
        transport.exchange_digest(1, "a" * 64)
    with pytest.raises(ValueError, match="lowercase sha256"):
        transport.exchange_digest(0, "not-a-digest")


def test_window_scatter_demands_the_groups_consensus_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(window_distributed, "ensure_process_group", lambda: config)
    prepared = _prepared_plan()
    inner = FluxWindowConditioningEvaluation(
        prepared, tuple(_ScaledEvaluator(index) for index in range(2))
    )

    with pytest.raises(ManifestRefusal) as forged:
        ManifestConsensusToken(
            manifest_digest="a" * 64, group_size=2, rank=0, mint_authority=object()
        )
    assert forged.value.code is ManifestRefusalCode.FORGED_CONSENSUS_TOKEN

    with pytest.raises(RuntimeError, match="manifest consensus token"):
        DistributedFluxWindowEvaluation(inner, object())  # pyright: ignore[reportArgumentType]

    token = prove_manifest_consensus(
        _manifest(config, prepared), rank=0, transport=_LoopbackTransport(2)
    )

    with pytest.raises(TypeError, match="exact FluxWindowConditioningEvaluation"):
        DistributedFluxWindowEvaluation(object(), token)  # pyright: ignore[reportArgumentType]

    single = _prepared_plan(window_count=1)
    single_inner = FluxWindowConditioningEvaluation(single, (_ScaledEvaluator(0),))
    with pytest.raises(FluxWindowError, match="at least two joint windows"):
        DistributedFluxWindowEvaluation(single_inner, token)

    foreign = _config(rank=1)
    monkeypatch.setattr(window_distributed, "ensure_process_group", lambda: foreign)
    with pytest.raises(RuntimeError, match="does not cover this process group"):
        DistributedFluxWindowEvaluation(inner, token)


def test_window_mode_parses_and_refuses_sequence_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RANK", "0")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_MULTI_GPU_MODE", "window")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_RENDEZVOUS", f"file://{tmp_path}/group")
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_TOKEN", "1" * 32)
    distributed.activate_distributed_sampling_attempt("window-parse", 1)
    try:
        config = distributed.distributed_sampling_config()
        assert config is not None
        assert config.mode == "window"
        assert config.world_size == 2

        monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES", "2")
        with pytest.raises(RuntimeError, match="sequence geometry requires sequence mode"):
            distributed.distributed_sampling_config()
    finally:
        distributed.release_distributed_sampling_attempt("window-parse", 1)


def _init_group(
    rank: int, rendezvous: str, world_size: int = 2, backend: str = "gloo"
) -> DistributedSamplingConfig:
    torch.distributed.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    config = DistributedSamplingConfig(
        rank, world_size, "window", f"file://{rendezvous}", "1" * 32, "a:1"
    )
    window_distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
    return config


def _minted_scatter(
    config: DistributedSamplingConfig,
    prepared: PreparedFluxWindowPlan,
    evaluators: tuple[_ScaledEvaluator, ...],
    device: torch.device | None = None,
) -> DistributedFluxWindowEvaluation[tuple[torch.Tensor, torch.Tensor | None]]:
    manifest = _manifest(config, prepared)
    transport = WindowDigestConsensusTransport(config, device or torch.device("cpu"))
    assert transport.physical_ranks == tuple(range(config.world_size))
    token = prove_manifest_consensus(manifest, rank=config.rank, transport=transport)
    return DistributedFluxWindowEvaluation(
        FluxWindowConditioningEvaluation(prepared, evaluators), token
    )


_PreparedCondition = PreparedFluxWindowConditioning[tuple[torch.Tensor, torch.Tensor | None]]


def _prepared_conditions(
    evaluation: FluxWindowConditioningEvaluation[tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[_PreparedCondition, _PreparedCondition]:
    first = declare_text_conditioning(Conditioning(torch.zeros(1, 5, 3)), 5)
    second = declare_text_conditioning(Conditioning(torch.full((1, 5, 3), 2.0)), 5)
    return (
        evaluation.prepare_conditioning(
            bind_flux_layout(first, latent_height=4, latent_width=6, patch_size=2)
        ),
        evaluation.prepare_conditioning(
            bind_flux_layout(second, latent_height=4, latent_width=6, patch_size=2)
        ),
    )


def test_rank_zero_sampling_uses_serial_window_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_plan(window_count=2)
    evaluators = tuple(_ScaledEvaluator(index) for index in range(2))
    serial = FluxWindowConditioningEvaluation(prepared, evaluators)
    scatter = object.__new__(DistributedFluxWindowEvaluation)
    scatter.inner = serial
    condition, second = _prepared_conditions(serial)
    x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    monkeypatch.setattr(window_distributed, "rank_zero_sampling_active", lambda: True)

    assert torch.equal(
        scatter.evaluate_conditioning(x, 0.5, condition),
        serial.evaluate_conditioning(x, 0.5, condition),
    )
    actual_batch = scatter.evaluate_conditioning_batch(x, 0.5, (condition, second))
    expected_batch = serial.evaluate_conditioning_batch(x, 0.5, (condition, second))
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_batch, expected_batch, strict=True)
    )


def _run_scatter_parity(rank: int, rendezvous: str, window_count: int) -> None:
    config = _init_group(rank, rendezvous)
    try:
        prepared = _prepared_plan(window_count=window_count)
        evaluators = tuple(_ScaledEvaluator(index) for index in range(window_count))
        serial = FluxWindowConditioningEvaluation(prepared, evaluators)
        scatter = _minted_scatter(config, prepared, evaluators)
        condition, second = _prepared_conditions(serial)
        x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)

        single = scatter.evaluate_conditioning(x, 0.75, condition)
        assert torch.equal(single, serial.evaluate_conditioning(x, 0.75, condition))

        batch = scatter.evaluate_conditioning_batch(x, 0.5, (condition, second))
        expected = serial.evaluate_conditioning_batch(x, 0.5, (condition, second))
        assert len(batch) == 2
        for actual, reference in zip(batch, expected, strict=True):
            assert torch.equal(actual, reference)

        stale = scatter._last_ordinal  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="stale or duplicate"):
            scatter._begin_fence(stale, 1, x)  # pyright: ignore[reportPrivateUsage]
    finally:
        torch.distributed.destroy_process_group()


def _run_divergent_plan_refusal(rank: int, rendezvous: str) -> None:
    config = _init_group(rank, rendezvous)
    try:
        assert not window_preflight_failed(False, torch.device("cpu"))
        prepared = _prepared_plan(weighted=rank == 1)
        manifest = _manifest(config, prepared)
        transport = WindowDigestConsensusTransport(config, torch.device("cpu"))
        peer = 1 - rank
        with pytest.raises(ManifestRefusal, match=f"divergent ranks {peer}") as refusal:
            prove_manifest_consensus(manifest, rank=rank, transport=transport)
        assert refusal.value.code is ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH
    finally:
        torch.distributed.destroy_process_group()


def _run_preflight_failure(rank: int, rendezvous: str) -> None:
    _init_group(rank, rendezvous)
    try:
        assert window_preflight_failed(rank == 0, torch.device("cpu"))
    finally:
        torch.distributed.destroy_process_group()


def _run_route_agreement(rank: int, rendezvous: str, routes: tuple[bool, bool]) -> None:
    _init_group(rank, rendezvous)
    try:
        assert window_route_mismatch(routes[rank], torch.device("cpu")) is (routes[0] != routes[1])
    finally:
        torch.distributed.destroy_process_group()


def _run_fence_geometry_disagreement(rank: int, rendezvous: str) -> None:
    config = _init_group(rank, rendezvous)
    try:
        prepared = _prepared_plan()
        evaluators = tuple(_ScaledEvaluator(index) for index in range(2))
        scatter = _minted_scatter(config, prepared, evaluators)
        condition, _ = _prepared_conditions(scatter.inner)
        x = torch.zeros(1, 1 + rank, 4, 6)
        with pytest.raises(RuntimeError, match="window sideband control disagrees"):
            scatter.evaluate_conditioning(x, 1.0, condition)
    finally:
        torch.distributed.destroy_process_group()


def _run_local_failure_symmetry(rank: int, rendezvous: str) -> None:
    config = _init_group(rank, rendezvous)
    try:
        prepared = _prepared_plan()
        evaluators = (_ExplodingEvaluator(0), _ScaledEvaluator(1))
        scatter = _minted_scatter(config, prepared, evaluators)
        condition, _ = _prepared_conditions(scatter.inner)
        x = torch.zeros(1, 1, 4, 6)
        if rank == 0:
            with pytest.raises(ValueError, match="window evaluator exploded"):
                scatter.evaluate_conditioning(x, 1.0, condition)
        else:
            with pytest.raises(RuntimeError, match="peer window evaluation failed"):
                scatter.evaluate_conditioning(x, 1.0, condition)
    finally:
        torch.distributed.destroy_process_group()


def _run_lane_count_disagreement(rank: int, rendezvous: str) -> None:
    config = _init_group(rank, rendezvous)
    try:
        prepared = _prepared_plan()
        evaluators = tuple(_ScaledEvaluator(index) for index in range(2))
        scatter = _minted_scatter(config, prepared, evaluators)
        condition, _ = _prepared_conditions(scatter.inner)
        x = torch.zeros(1, 1, 4, 6)
        lanes = () if rank == 0 else (condition,)
        with pytest.raises(RuntimeError, match="window sideband control disagrees"):
            scatter.evaluate_conditioning_batch(x, 1.0, lanes)
    finally:
        torch.distributed.destroy_process_group()


def _run_empty_batch_failure_symmetry(rank: int, rendezvous: str) -> None:
    config = _init_group(rank, rendezvous)
    try:
        prepared = _prepared_plan()
        evaluators = tuple(_ScaledEvaluator(index) for index in range(2))
        scatter = _minted_scatter(config, prepared, evaluators)
        x = torch.zeros(1, 1, 4, 6)
        with pytest.raises(FluxWindowError, match="non-empty condition tuple"):
            scatter.evaluate_conditioning_batch(x, 1.0, ())
    finally:
        torch.distributed.destroy_process_group()


def _run_scatter_parity_nccl(
    rank: int, rendezvous: str, world_size: int, window_count: int
) -> None:
    if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
        raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    config = _init_group(rank, rendezvous, world_size=world_size, backend="nccl")
    try:
        prepared = _prepared_plan(window_count=window_count)
        evaluators = tuple(_ScaledEvaluator(index) for index in range(window_count))
        serial = FluxWindowConditioningEvaluation(prepared, evaluators)
        scatter = _minted_scatter(config, prepared, evaluators, device=device)
        condition, second = _prepared_conditions(serial)
        x = torch.arange(24, dtype=torch.float32, device=device).reshape(1, 1, 4, 6)

        single = scatter.evaluate_conditioning(x, 0.75, condition)
        assert single.device == device
        assert torch.equal(single, serial.evaluate_conditioning(x, 0.75, condition))

        batch = scatter.evaluate_conditioning_batch(x, 0.5, (condition, second))
        expected = serial.evaluate_conditioning_batch(x, 0.5, (condition, second))
        assert len(batch) == 2
        for actual, reference in zip(batch, expected, strict=True):
            assert torch.equal(actual, reference)
    finally:
        torch.distributed.destroy_process_group()


def _run_window_step_callback_desync(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    config = DistributedSamplingConfig(rank, 2, "window", f"file://{rendezvous}", "1" * 32, "a:1")
    window_distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
    try:
        prepared = _prepared_plan()
        evaluators = tuple(_ScaledEvaluator(index) for index in range(2))
        scatter = _minted_scatter(config, prepared, evaluators)
        condition, _ = _prepared_conditions(scatter.inner)
        x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        fenced_evaluations: list[float] = []

        def denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
            result = scatter.evaluate_conditioning(x, sigma, condition)
            fenced_evaluations.append(sigma)
            return result

        emitted_steps: list[int] = []

        def on_step(event: StepEvent) -> None:
            emitted_steps.append(event.step)
            if rank == 0 and event.step == 0:
                raise RuntimeError("step callback failed")

        def run_lockstep_solver() -> None:
            # Window scatter is SPMD like guidance: every rank runs the
            # same solver loop and the fenced collective sequence is
            # issued inside evaluate_conditioning, so the real solver
            # emits on_step between fences, never inside one.
            torch_euler()(
                denoiser,
                x,
                (1.0, 0.5, 0.25),
                SamplerInfo(Parameterization.EPS),
                on_step=on_step,
            )

        if rank == 0:
            with pytest.raises(RuntimeError, match="step callback failed"):
                run_lockstep_solver()
            # The raising callback stopped the solver before the next fence.
            assert emitted_steps == [0]
            assert fenced_evaluations == [1.0]
        else:
            with pytest.raises(RuntimeError) as failure:
                run_lockstep_solver()
            # The peer rank fails closed inside the abandoned fence instead
            # of completing a desynchronized collective.
            assert "step callback failed" not in str(failure.value)
            assert emitted_steps == [0]
            assert fenced_evaluations == [1.0]
    finally:
        torch.distributed.destroy_process_group()


def _spawn_group(target: object, *args: object, world_size: int = 2, timeout: float = 60) -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "window-rendezvous")
        processes = [
            multiprocessing.get_context("spawn").Process(
                target=target,  # pyright: ignore[reportArgumentType]
                args=(rank, rendezvous, *args),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=timeout)
            assert process.exitcode == 0


@pytest.mark.parametrize("window_count", (2, 3))
def test_window_scatter_matches_the_serial_merge_bitwise(window_count: int) -> None:
    _spawn_group(_run_scatter_parity, window_count)


def test_divergent_window_plans_refuse_before_any_collective() -> None:
    _spawn_group(_run_divergent_plan_refusal)


def test_one_rank_preflight_failure_fails_the_whole_group() -> None:
    _spawn_group(_run_preflight_failure)


@pytest.mark.parametrize("routes", ((False, False), (True, True), (True, False)))
def test_window_route_requires_rank_wide_agreement(routes: tuple[bool, bool]) -> None:
    _spawn_group(_run_route_agreement, routes)


def test_window_fence_refuses_cross_rank_geometry_disagreement() -> None:
    _spawn_group(_run_fence_geometry_disagreement)


def test_one_rank_evaluation_failure_propagates_symmetrically() -> None:
    _spawn_group(_run_local_failure_symmetry)


def test_lane_count_disagreement_refuses_at_the_fence() -> None:
    _spawn_group(_run_lane_count_disagreement)


def test_group_wide_empty_batch_fails_after_the_fence() -> None:
    _spawn_group(_run_empty_batch_failure_symmetry)


def test_step_callback_failure_cannot_desync_a_window_collective() -> None:
    _spawn_group(_run_window_step_callback_desync)


_GPU_TESTS_ENABLED = os.environ.get("DINKSTER_ENABLE_GPU_TESTS") == "1"


@pytest.mark.skipif(
    not _GPU_TESTS_ENABLED or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs DINKSTER_ENABLE_GPU_TESTS=1 and two CUDA devices",
)
@pytest.mark.parametrize("window_count", (2, 3))
def test_window_scatter_matches_the_serial_merge_bitwise_on_nccl(window_count: int) -> None:
    _spawn_group(_run_scatter_parity_nccl, 2, window_count, timeout=300)


@pytest.mark.skipif(
    not _GPU_TESTS_ENABLED or not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="needs DINKSTER_ENABLE_GPU_TESTS=1 and four CUDA devices",
)
def test_window_scatter_covers_idle_ranks_on_nccl() -> None:
    _spawn_group(_run_scatter_parity_nccl, 4, 3, world_size=4, timeout=300)
