from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import dinkster_nodes_vision.hed.mlsd as mlsd_module
import torch
from dinkster_api.v1 import PressureSignal
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.hed import register_types
from dinkster_nodes_vision.hed.anyline import _remove_small_objects, execute_anyline
from dinkster_nodes_vision.hed.cache import MODEL_CACHE, ModelCache
from dinkster_nodes_vision.hed.lineart import execute_realistic
from dinkster_nodes_vision.hed.mlsd import execute_mlsd
from dinkster_nodes_vision.hed.model import execute_hed
from dinkster_nodes_vision.hed.teed import execute_teed
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker
from worker_model_fixtures import worker_patch_launcher

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_hed_pack/dinkster-pack.toml"


def test_hed_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = np.zeros((1, 3, 5, 3), dtype=np.float32)
        source[0, :, :, 0] = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            launcher=worker_patch_launcher(
                tmp_path,
                """
import torch
from dinkster_nodes_vision.hed import model

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        height, width = image.shape[-2:]
        gradient = torch.linspace(0.0, 1.0, width).reshape(1, 1, 1, width)
        return tuple(gradient.expand(1, 1, height, width) for _ in range(5))

model._load_model = FakeModel
""",
            ),
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "hed": GraphNode(
                        "dinkster.preprocess.model_edges",
                        {
                            "image": TypedLiteral("IMAGE", source.tolist()),
                            "provider": "dinkster-vision-hed",
                            "safe": False,
                            "scribble": False,
                            "resolution": 64,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["hed"])
            output = np.asarray(result.outputs["hed"]["image"].resolve())
            assert output.shape == (1, 64, 107, 3)
            np.testing.assert_allclose(output[0, 0, 0], 127.0 / 255.0, rtol=0.0, atol=1e-7)
            np.testing.assert_allclose(output[0, 0, -1], 177.0 / 255.0, rtol=0.0, atol=1e-7)
            assert float(output[0, 0, 0, 0]) < float(output[0, 0, -1, 0])
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_mlsd_cpu_oom_propagates_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    used_devices: list[str] = []

    @contextmanager
    def always_oom(
        key: str,
        factory: Callable[[], torch.nn.Module],
        *,
        device: torch.device | None = None,
    ) -> Iterator[torch.nn.Module]:
        del key, factory
        assert device is not None
        used_devices.append(device.type)
        raise torch.OutOfMemoryError("forced CPU OOM")
        yield torch.nn.Identity()

    monkeypatch.setattr(MODEL_CACHE, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(MODEL_CACHE, "use", always_oom)
    with pytest.raises(torch.OutOfMemoryError, match="forced CPU OOM"):
        execute_mlsd(
            np.zeros((1, 64, 64, 3), dtype=np.float32),
            score_threshold=0.1,
            distance_threshold=0.1,
            resolution=64,
        )
    assert used_devices == ["cpu"]


def test_mlsd_discards_primary_and_fallback_models_when_retry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MODEL_CACHE.discard("M-LSD")
    MODEL_CACHE.discard("M-LSD CPU fallback")
    original_use = MODEL_CACHE.use
    frame_calls = 0

    @contextmanager
    def redirect_cuda_cache_to_cpu(
        key: str,
        factory: Callable[[], torch.nn.Module],
        *,
        device: torch.device | None = None,
    ) -> Iterator[torch.nn.Module]:
        assert device is not None
        target = torch.device("cpu") if device.type == "cuda" else device
        with original_use(key, factory, device=target) as model:
            yield model

    def fail_both_attempts(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal frame_calls
        frame_calls += 1
        if frame_calls == 1:
            raise torch.OutOfMemoryError("forced CUDA OOM")
        raise RuntimeError("forced CPU failure")

    monkeypatch.setattr(MODEL_CACHE, "device", lambda: torch.device("cuda"))
    monkeypatch.setattr(MODEL_CACHE, "use", redirect_cuda_cache_to_cpu)
    monkeypatch.setattr(mlsd_module, "_load_mlsd", lambda: torch.nn.Linear(1, 1))
    monkeypatch.setattr(mlsd_module, "_mlsd_frame", fail_both_attempts)
    with pytest.raises(RuntimeError, match="forced CPU failure"):
        execute_mlsd(
            np.zeros((1, 64, 64, 3), dtype=np.float32),
            score_threshold=0.1,
            distance_threshold=0.1,
            resolution=64,
        )
    assert frame_calls == 2
    assert all(
        item.item_id not in {"M-LSD", "M-LSD CPU fallback"} for item in MODEL_CACHE.details()
    )


def test_anyline_small_object_threshold_matches_pinned_scikit_image_behavior() -> None:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[1, 1] = 1.0
    assert np.count_nonzero(_remove_small_objects(image, minimum_size=2, connectivity=1)) == 3
    assert np.count_nonzero(_remove_small_objects(image, minimum_size=3, connectivity=1)) == 0

    image[0, 0] = 1.0
    assert np.count_nonzero(_remove_small_objects(image, minimum_size=4, connectivity=1)) == 0
    assert np.count_nonzero(_remove_small_objects(image, minimum_size=4, connectivity=2)) == 6
    assert np.count_nonzero(_remove_small_objects(image, minimum_size=4, connectivity=16_384)) == 6


@pytest.mark.parametrize(
    ("execute", "inputs", "error"),
    (
        (execute_realistic, {"coarse": 1, "resolution": 64}, TypeError),
        (execute_hed, {"safe": 1, "scribble": False, "resolution": 64}, TypeError),
        (execute_teed, {"safe_steps": 11, "resolution": 64}, ValueError),
        (
            execute_mlsd,
            {"score_threshold": 0.0, "distance_threshold": 0.1, "resolution": 64},
            ValueError,
        ),
        (
            execute_anyline,
            {
                "merge_with_lineart": "unknown",
                "resolution": 64,
                "lineart_lower_bound": 0.0,
                "lineart_upper_bound": 1.0,
                "object_min_size": 36,
                "object_connectivity": 1,
            },
            ValueError,
        ),
        (
            execute_anyline,
            {
                "merge_with_lineart": "lineart_standard",
                "resolution": 64,
                "lineart_lower_bound": -0.1,
                "lineart_upper_bound": 1.0,
                "object_min_size": 36,
                "object_connectivity": 1,
            },
            ValueError,
        ),
        (
            execute_anyline,
            {
                "merge_with_lineart": "lineart_standard",
                "resolution": 64,
                "lineart_lower_bound": 0.0,
                "lineart_upper_bound": 1.0,
                "object_min_size": 0,
                "object_connectivity": 1,
            },
            ValueError,
        ),
    ),
)
def test_preprocessors_reject_invalid_parameters(
    execute: Callable[..., np.ndarray], inputs: dict[str, object], error: type[Exception]
) -> None:
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    with pytest.raises(error):
        execute(image, **inputs)


def test_model_cache_reuses_idle_models_and_sheds_governed_memory() -> None:
    cache = ModelCache()
    loads = 0

    def factory() -> torch.nn.Module:
        nonlocal loads
        loads += 1
        return torch.nn.Linear(4, 3)

    with cache.use("detector", factory) as first:
        assert cache.footprint("ram") == 60
        assert asyncio.run(cache.shed(PressureSignal(device="ram", bytes_needed=1))) == 0
    with cache.use("detector", factory) as second:
        assert second is first
    assert loads == 1
    details = cache.details()
    assert len(details) == 1
    assert details[0].item_id == "detector"
    assert details[0].bytes_by_residency == {"ram": 60}
    assert asyncio.run(cache.shed(PressureSignal(device="ram", bytes_needed=1))) == 60
    assert cache.details() == []


def test_model_cache_only_discards_idle_models() -> None:
    cache = ModelCache()
    with cache.use("detector", lambda: torch.nn.Linear(4, 3)):
        assert cache.discard("detector") == 0
    assert cache.discard("detector") == 60
    assert cache.details() == []


def test_model_cache_full_release_clears_idle_models_while_retaining_active_models() -> None:
    async def scenario() -> None:
        cache = ModelCache()
        with cache.use("idle", lambda: torch.nn.Linear(4, 3)):
            pass
        with cache.use("active", lambda: torch.nn.Linear(4, 3)):
            assert (await cache.full_release()).status == "busy"
            assert [item.item_id for item in cache.details()] == ["active"]
        assert (await cache.full_release()).status == "complete"
        assert cache.details() == []

    asyncio.run(scenario())


def test_model_cache_full_release_clears_idle_models_while_retaining_parked_models() -> None:
    async def scenario() -> None:
        cache = ModelCache()
        for key in ("parked", "idle"):
            with cache.use(key, lambda: torch.nn.Linear(4, 3), device=torch.device("cpu")):
                pass

        with cache.park("parked", torch.device("cpu")) as parked:
            assert parked
            assert (await cache.full_release()).status == "busy"
            assert [item.item_id for item in cache.details()] == ["parked"]
        assert (await cache.full_release()).status == "complete"
        assert cache.details() == []

    asyncio.run(scenario())


def test_model_cache_full_release_clears_all_idle_models_and_reloads_on_reuse() -> None:
    async def scenario() -> None:
        cache = ModelCache()
        loads = 0

        def factory() -> torch.nn.Module:
            nonlocal loads
            loads += 1
            return torch.nn.Linear(4, 3)

        with cache.use("detector", factory) as first:
            pass
        assert (await cache.full_release()).status == "complete"
        assert cache.details() == []
        with cache.use("detector", factory) as second:
            assert second is not first
        assert loads == 2
        assert (await cache.full_release()).status == "complete"

    asyncio.run(scenario())


def test_model_cache_full_release_returns_cuda_allocator_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_cache_calls = 0

    def empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.nn.Linear, "to", lambda self, _device: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    cache = ModelCache()
    with cache.use(
        "detector",
        lambda: torch.nn.Linear(4, 3),
        device=torch.device("cuda:0"),
    ):
        pass

    assert asyncio.run(cache.full_release()).status == "complete"
    assert cache.details() == []
    assert empty_cache_calls == 1


@pytest.mark.parametrize("failure_call", (1, 2))
def test_model_cache_evicts_failed_parking_without_blocking_reload(
    monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    cache = ModelCache()
    with cache.use("detector", lambda: torch.nn.Linear(4, 3), device=torch.device("cpu")) as model:
        pass
    original_to = cast("Callable[..., torch.nn.Module]", model.to)
    calls = 0

    def fail_selected_move(*args: object, **kwargs: object) -> torch.nn.Module:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise torch.OutOfMemoryError("forced relocation failure")
        return original_to(*args, **kwargs)

    monkeypatch.setattr(model, "to", fail_selected_move)
    with pytest.raises(torch.OutOfMemoryError, match="forced relocation failure"):
        with cache.park("detector", torch.device("cpu")):
            pass
    assert cache.details() == []
    with cache.use("detector", lambda: torch.nn.Linear(4, 3), device=torch.device("cpu")):
        pass
