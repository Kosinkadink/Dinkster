from __future__ import annotations

import asyncio
import random
from contextlib import ExitStack

import pytest
import torch
from dinkster_inference_torch import (
    DeviceMemory,
    PagedKVCache,
    PagedKVCapacityError,
    PagedKVGeometry,
    PagedKVSessionBusyError,
    PagedKVSessionMissingError,
    PagedKVShapeError,
    ResidencyManager,
)
from dinkster_memory import DetailedConsumer, MemoryGovernor, PressureSignal, Shedder

GEOMETRY = PagedKVGeometry(
    block_tokens=4,
    layer_count=2,
    kv_heads=2,
    head_dim=3,
    dtype=torch.float32,
)


def _cache(*, max_device_blocks: int = 64) -> PagedKVCache:
    return PagedKVCache(
        "qwen-kv",
        "qwen-test.safetensors",
        GEOMETRY,
        load_device="cpu",
        max_device_blocks=max_device_blocks,
    )


def _kv(start: int, token_count: int, *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(
        start,
        start + GEOMETRY.layer_count * token_count * GEOMETRY.kv_heads * GEOMETRY.head_dim,
        dtype=GEOMETRY.dtype,
        device=device,
    ).reshape(GEOMETRY.layer_count, token_count, GEOMETRY.kv_heads, GEOMETRY.head_dim)
    return values, values + 10_000


def _read(cache: PagedKVCache, session_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    with cache.pin(session_id) as lease:
        keys = []
        values = []
        for layer in range(GEOMETRY.layer_count):
            views = lease.block_views(layer)
            shape = (0, GEOMETRY.kv_heads, GEOMETRY.head_dim)
            keys.append(
                torch.cat([view.key for view in views], dim=0) if views else torch.empty(shape)
            )
            values.append(
                torch.cat([view.value for view in views], dim=0) if views else torch.empty(shape)
            )
        return torch.stack(keys), torch.stack(values)


def _commit(cache: PagedKVCache, session_id: str, key: torch.Tensor, value: torch.Tensor) -> None:
    with cache.pin(session_id) as lease:
        lease.append(key, value)
        lease.commit()


def test_geometry_and_cache_validate_exact_contract() -> None:
    assert GEOMETRY.tensor_shape == (2, 4, 2, 3)
    assert GEOMETRY.block_nbytes == 2 * 2 * 4 * 2 * 3 * 4

    with pytest.raises(TypeError, match="exact integer"):
        PagedKVGeometry(True, 1, 1, 1, torch.float32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        PagedKVGeometry(0, 1, 1, 1, torch.float32)
    with pytest.raises(ValueError, match="floating-point"):
        PagedKVGeometry(1, 1, 1, 1, torch.int64)
    with pytest.raises(ValueError, match="explicit device index"):
        PagedKVCache("kv", "model", GEOMETRY, load_device="cuda", max_device_blocks=1)
    with pytest.raises(ValueError, match="same host tier"):
        PagedKVCache(
            "kv",
            "model",
            GEOMETRY,
            load_device="cpu",
            max_device_blocks=1,
            max_host_blocks=1,
        )


def test_append_exposes_exact_blocks_bytes_and_values() -> None:
    cache = _cache()
    cache.create_session("session")

    rejected = cache.pin("session")
    with pytest.raises(PagedKVShapeError, match="shape"):
        rejected.append(torch.ones(2), torch.ones(2))
    assert cache.session_states()[0].active is False

    key, value = _kv(0, 5)
    key.requires_grad_()
    value.requires_grad_()
    with cache.pin("session") as lease:
        lease.append(key, value)
        assert lease.token_count == 5
        views = lease.block_views(0)
        assert [view.token_count for view in views] == [4, 1]
        assert torch.equal(torch.cat([view.key for view in views]), key[0])
        assert torch.equal(torch.cat([view.value for view in views]), value[0])
        assert not any(view.key.requires_grad or view.value.requires_grad for view in views)
        lease.commit()
        lease.commit()

    state = cache.session_states()[0]
    assert state.token_count == 5
    assert state.block_ids == (0, 1)
    assert state.device_block_count == 2
    assert state.host_block_count == 0
    assert cache.loaded_bytes() == cache.total_bytes() == 2 * GEOMETRY.block_nbytes
    assert cache.offloaded_bytes() == 0
    assert all(block.reference_count == 1 for block in cache.block_states())
    assert all(block.nbytes == GEOMETRY.block_nbytes for block in cache.block_states())
    pointers = []
    with cache.pin("session") as lease:
        for view in lease.block_views(0):
            pointers.append(view.key.untyped_storage().data_ptr())
            pointers.append(view.value.untyped_storage().data_ptr())
    assert len(set(pointers)) == len(pointers)


def test_close_and_failed_append_restore_committed_state() -> None:
    cache = _cache(max_device_blocks=3)
    cache.create_session("session")
    key, value = _kv(0, 3)
    _commit(cache, "session", key, value)
    committed_state = cache.session_states()[0]

    lease = cache.pin("session")
    with pytest.raises(PagedKVSessionBusyError):
        cache.pin("session")
    with pytest.raises(PagedKVSessionBusyError):
        cache.close_session("session")
    appended_key, appended_value = _kv(100, 5)
    lease.append(appended_key, appended_value)
    assert lease.token_count == 8
    lease.close()
    lease.close()

    assert cache.session_states()[0].block_ids == committed_state.block_ids
    assert torch.equal(_read(cache, "session")[0], key)
    assert torch.equal(_read(cache, "session")[1], value)

    failed = cache.pin("session")
    with pytest.raises(PagedKVCapacityError):
        failed.append(*_kv(200, 10))
    assert cache.session_states()[0].active is False
    assert torch.equal(_read(cache, "session")[0], key)


def test_partial_block_append_needs_no_spare_block_when_prefix_is_unshared() -> None:
    cache = _cache(max_device_blocks=1)
    cache.create_session("session")
    original_key, original_value = _kv(0, 2)
    _commit(cache, "session", original_key, original_value)

    with cache.pin("session") as lease:
        lease.append(*_kv(100, 1))
    assert torch.equal(_read(cache, "session")[0], original_key)
    assert cache.session_states()[0].block_ids == (0,)

    appended_key, appended_value = _kv(200, 2)
    _commit(cache, "session", appended_key, appended_value)
    assert torch.equal(_read(cache, "session")[0], torch.cat((original_key, appended_key), dim=1))
    assert torch.equal(
        _read(cache, "session")[1], torch.cat((original_value, appended_value), dim=1)
    )
    assert cache.session_states()[0].block_ids == (0,)


def test_partial_prefix_fork_uses_copy_on_write_without_contamination() -> None:
    cache = _cache()
    cache.create_session("parent")
    parent_key, parent_value = _kv(0, 3)
    _commit(cache, "parent", parent_key, parent_value)
    parent_block = cache.session_states()[0].block_ids[0]

    cache.fork_session("parent", "child", token_count=2)
    assert cache.block_states()[0].reference_count == 2
    child_key, child_value = _kv(100, 4)
    _commit(cache, "child", child_key, child_value)

    parent_after = _read(cache, "parent")
    child_after = _read(cache, "child")
    assert torch.equal(parent_after[0], parent_key)
    assert torch.equal(parent_after[1], parent_value)
    assert torch.equal(child_after[0], torch.cat((parent_key[:, :2], child_key), dim=1))
    assert torch.equal(child_after[1], torch.cat((parent_value[:, :2], child_value), dim=1))
    states = {state.session_id: state for state in cache.session_states()}
    assert states["parent"].block_ids == (parent_block,)
    assert states["child"].block_ids[0] != parent_block

    cache.close_session("parent")
    cache.close_session("parent")
    cache.close_session("child")
    assert cache.session_states() == ()
    assert cache.block_states() == ()
    assert cache.total_bytes() == 0
    cache.create_session("reused")
    _commit(cache, "reused", *_kv(200, 1))
    assert cache.session_states()[0].block_ids == (parent_block,)


def test_partial_prefix_copy_failure_releases_unowned_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    cache.create_session("parent")
    _commit(cache, "parent", *_kv(0, 3))
    cache.fork_session("parent", "child", token_count=2)

    copy_count = 0
    original_copy = torch.Tensor.copy_

    def fail_second_copy(
        target: torch.Tensor,
        source: torch.Tensor,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise RuntimeError("injected copy failure")
        return original_copy(target, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        with cache.pin("child") as lease:
            lease.append(*_kv(100, 1))

    states = {state.session_id: state for state in cache.session_states()}
    assert states["parent"].block_ids == states["child"].block_ids == (0,)
    assert all(state.active is False for state in states.values())
    assert [(block.block_id, block.reference_count) for block in cache.block_states()] == [(0, 2)]
    assert cache.total_bytes() == GEOMETRY.block_nbytes
    cache.unload()
    assert cache.total_bytes() == 0


def test_restore_protects_shared_target_blocks_from_reclamation() -> None:
    cache = _cache(max_device_blocks=2)
    cache.max_host_blocks = 2
    cache.create_session("target")
    _commit(cache, "target", *_kv(0, 8))
    cache.fork_session("target", "shared-prefix", token_count=4)

    with cache.pin("shared-prefix"):
        assert (
            asyncio.run(cache.shed(PressureSignal("ram", GEOMETRY.block_nbytes)))
            == GEOMETRY.block_nbytes
        )
    target = {state.session_id: state for state in cache.session_states()}["target"]
    assert target.device_block_count == target.host_block_count == 1

    cache.create_session("newer")
    _commit(cache, "newer", *_kv(100, 4))
    with cache.pin("target") as lease:
        target = {state.session_id: state for state in cache.session_states()}["target"]
        assert target.active is True
        assert target.device_block_count == 2
        assert target.host_block_count == 0
        assert len(lease.block_views(0)) == 2


def test_partial_unload_refuses_active_session_without_dropping_idle_peers() -> None:
    cache = _cache(max_device_blocks=2)
    for index, session_id in enumerate(("old", "new")):
        cache.create_session(session_id)
        _commit(cache, session_id, *_kv(index * 100, 4))

    cache.create_session("third")
    _commit(cache, "third", *_kv(200, 4))
    assert [state.session_id for state in cache.session_states()] == ["new", "third"]

    active = cache.pin("new")
    before_sessions = cache.session_states()
    before_blocks = cache.block_states()
    with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
        cache.partially_unload(2 * GEOMETRY.block_nbytes)
    assert cache.session_states() == before_sessions
    assert cache.block_states() == before_blocks
    with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
        cache.unload()
    active.close()
    assert cache.partially_load(-(2 * GEOMETRY.block_nbytes)) == -(2 * GEOMETRY.block_nbytes)
    assert cache.session_states() == ()


def test_residency_manager_reclaims_idle_cache_and_fails_on_active_cache() -> None:
    cache = _cache(max_device_blocks=4)
    manager = ResidencyManager(
        free_memory=lambda _device: DeviceMemory(
            free_total=4 * GEOMETRY.block_nbytes - cache.loaded_bytes(),
            free_torch=0,
        ),
        total_memory=lambda _device: 4 * GEOMETRY.block_nbytes,
        empty_cache=lambda _device: None,
    )
    manager.load((cache,))
    cache.create_session("idle")
    _commit(cache, "idle", *_kv(0, 4))

    manager.free(4 * GEOMETRY.block_nbytes, torch.device("cpu"))
    assert cache.loaded_bytes() == 0
    assert cache.session_states() == ()
    assert manager.registered() == ()

    manager.load((cache,))
    cache.create_session("active")
    _commit(cache, "active", *_kv(100, 4))
    lease = cache.pin("active")
    with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
        manager.free(4 * GEOMETRY.block_nbytes, torch.device("cpu"))
    assert cache.loaded_bytes() == GEOMETRY.block_nbytes
    lease.close()


def test_manager_refuses_mixed_active_pressure_without_mutating_idle_session() -> None:
    cache = _cache(max_device_blocks=3)
    manager = ResidencyManager(
        free_memory=lambda _device: DeviceMemory(
            free_total=3 * GEOMETRY.block_nbytes - cache.loaded_bytes(),
            free_torch=0,
        ),
        total_memory=lambda _device: 3 * GEOMETRY.block_nbytes,
        empty_cache=lambda _device: None,
    )
    manager.load((cache,))
    cache.create_session("active")
    _commit(cache, "active", *_kv(0, 8))
    cache.create_session("idle")
    _commit(cache, "idle", *_kv(100, 4))
    lease = cache.pin("active")
    before_sessions = cache.session_states()
    before_blocks = cache.block_states()
    assert cache.partial_unload_capacity() == 0

    with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
        manager.free(GEOMETRY.block_nbytes // 2, torch.device("cpu"))

    assert cache.session_states() == before_sessions
    assert cache.block_states() == before_blocks
    assert manager.registered() == (cache,)
    lease.close()


def test_manager_refuses_session_activated_after_capacity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(max_device_blocks=2)
    manager = ResidencyManager(
        free_memory=lambda _device: DeviceMemory(
            free_total=2 * GEOMETRY.block_nbytes - cache.loaded_bytes(),
            free_torch=0,
        ),
        total_memory=lambda _device: 2 * GEOMETRY.block_nbytes,
        empty_cache=lambda _device: None,
    )
    manager.load((cache,))
    for index, session_id in enumerate(("first", "second")):
        cache.create_session(session_id)
        _commit(cache, session_id, *_kv(index * 100, 4))
    expected_sessions = ()
    expected_blocks = ()
    original_capacity = cache.partial_unload_capacity

    with ExitStack() as active_leases:

        def activate_after_capacity_check() -> int:
            nonlocal expected_blocks, expected_sessions
            capacity = original_capacity()
            active_leases.enter_context(cache.pin("first"))
            expected_sessions = cache.session_states()
            expected_blocks = cache.block_states()
            return capacity

        monkeypatch.setattr(cache, "partial_unload_capacity", activate_after_capacity_check)
        with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
            manager.free(GEOMETRY.block_nbytes // 2, torch.device("cpu"))

        assert cache.session_states() == expected_sessions
        assert cache.block_states() == expected_blocks
        assert manager.registered() == (cache,)


def test_governor_surfaces_exact_footprint_details_and_targeted_shedding() -> None:
    cache = _cache()
    cache.create_session("session")
    _commit(cache, "session", *_kv(0, 5))
    assert isinstance(cache, Shedder)
    assert isinstance(cache, DetailedConsumer)
    governor = MemoryGovernor({"ram": 4 * GEOMETRY.block_nbytes})
    governor.register_shedder(cache, name="llm-kv")

    detail = governor.details()["llm-kv"][0]
    assert detail.item_id == "qwen-kv"
    assert detail.display_name == "qwen-test.safetensors"
    assert detail.bytes_by_residency == {"ram": 2 * GEOMETRY.block_nbytes}
    assert detail.pages is not None
    assert detail.pages.page_bytes == GEOMETRY.block_nbytes
    assert detail.pages.flags == (1, 1)
    assert governor.footprint("ram") == 2 * GEOMETRY.block_nbytes

    ignored = asyncio.run(
        governor.shed(
            "ram",
            GEOMETRY.block_nbytes,
            consumers=("llm-kv",),
            items=("other",),
        )
    )
    assert ignored == 0
    freed = asyncio.run(
        governor.shed(
            "ram",
            GEOMETRY.block_nbytes,
            consumers=("llm-kv",),
            items=("qwen-kv",),
        )
    )
    assert freed == 2 * GEOMETRY.block_nbytes
    assert cache.session_states() == ()


def test_randomized_transactions_never_leak_or_cross_contaminate_sessions() -> None:
    rng = random.Random(0x525)
    cache = _cache(max_device_blocks=512)
    expected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    next_session = 0
    next_value = 0

    def create() -> str:
        nonlocal next_session
        session_id = f"session-{next_session}"
        next_session += 1
        cache.create_session(session_id)
        empty = torch.empty(
            (GEOMETRY.layer_count, 0, GEOMETRY.kv_heads, GEOMETRY.head_dim),
            dtype=GEOMETRY.dtype,
        )
        expected[session_id] = (empty, empty.clone())
        return session_id

    create()
    for _ in range(120):
        operation = rng.choices(("append", "fork", "close"), weights=(7, 2, 1))[0]
        session_id = rng.choice(sorted(expected))
        if operation == "append":
            count = rng.randint(1, 6)
            key, value = _kv(next_value, count)
            next_value += key.numel()
            should_commit = rng.random() < 0.75
            with cache.pin(session_id) as lease:
                lease.append(key, value)
                if should_commit:
                    lease.commit()
            if should_commit:
                old_key, old_value = expected[session_id]
                expected[session_id] = (
                    torch.cat((old_key, key), dim=1),
                    torch.cat((old_value, value), dim=1),
                )
        elif operation == "fork":
            target = create()
            source_key, source_value = expected[session_id]
            count = rng.randint(0, source_key.shape[1])
            cache.close_session(target)
            cache.fork_session(session_id, target, token_count=count)
            expected[target] = (source_key[:, :count].clone(), source_value[:, :count].clone())
        elif len(expected) > 1:
            cache.close_session(session_id)
            del expected[session_id]

        states = {state.session_id: state for state in cache.session_states()}
        assert set(states) == set(expected)
        for check_id, (expected_key, expected_value) in expected.items():
            assert states[check_id].token_count == expected_key.shape[1]
            actual_key, actual_value = _read(cache, check_id)
            assert torch.equal(actual_key, expected_key)
            assert torch.equal(actual_value, expected_value)
        references = sum(len(state.block_ids) for state in states.values())
        assert sum(block.reference_count for block in cache.block_states()) == references
        assert cache.total_bytes() == cache.loaded_bytes() + cache.offloaded_bytes()

    for session_id in tuple(expected):
        cache.close_session(session_id)
    assert cache.total_bytes() == 0
    with pytest.raises(PagedKVSessionMissingError):
        cache.pin("missing")
