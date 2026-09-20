from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import zlib
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
from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.hed import register_types
from dinkster_nodes_vision.hed.anyline import _remove_small_objects, execute_anyline
from dinkster_nodes_vision.hed.cache import MODEL_CACHE, ModelCache
from dinkster_nodes_vision.hed.lineart import execute_anime, execute_manga, execute_realistic
from dinkster_nodes_vision.hed.mlsd import execute_mlsd
from dinkster_nodes_vision.hed.model import execute_hed
from dinkster_nodes_vision.hed.teed import execute_teed
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_hed_pack/dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "hed_controlnet_aux_59b1fc4.json"
LINE_EDGE_GOLDEN_PATH = ROOT / "tests" / "goldens" / "line_edge_controlnet_aux_59b1fc4.json"
MODEL_DIGEST = "blake3:36ea9a81b5e5f69c9f98b81eacce0c70b7bb444af4d821201b8a910e05792da9"
MODEL_SHA256 = "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa"

ARTIFACTS = {
    "hed-model": (
        "DINKSTER_HED_TEST_MODEL",
        MODEL_DIGEST,
        MODEL_SHA256,
    ),
    "lineart-realistic-model": (
        "DINKSTER_LINEART_REALISTIC_TEST_MODEL",
        "blake3:9849b9af40a35a8af12709de87d9d82abe8443b30b8ff37b86db47f912b7f933",
        "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    ),
    "lineart-realistic-coarse-model": (
        "DINKSTER_LINEART_REALISTIC_COARSE_TEST_MODEL",
        "blake3:94cca6f565259a25238844327987e1eca0a98194eab82bd4af56932d9662cd89",
        "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
    ),
    "lineart-anime-model": (
        "DINKSTER_LINEART_ANIME_TEST_MODEL",
        "blake3:147873fae2d83761e678eea074fdf8f2e8e0e99a2745f26b2e4711880f91210a",
        "ccabdcc3f5cf3c07cf65d58776acb21df7dfda825cdc70c9766a93fd62bfc488",
    ),
    "lineart-manga-model": (
        "DINKSTER_LINEART_MANGA_TEST_MODEL",
        "blake3:938c4173935eafac5a9d88f53ed7ca0041e278c6094da2b26098844107e8855e",
        "badbd6baf013cefbd98993307b02cc14a26c770d067416e4fdecc8720b88feeb",
    ),
    "mlsd-model": (
        "DINKSTER_MLSD_TEST_MODEL",
        "blake3:b770f3458f83a5be2065d89703fc53db8c3cf4c60fd6e5a49032ab28c9a644e9",
        "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
    ),
    "teed-model": (
        "DINKSTER_TEED_TEST_MODEL",
        "blake3:3f9dae7af1da3156f2fe3f72e0d74b170c7bb7a76a32105c110365dc9e53bd00",
        "b9037964149c55156c6adbffdfbd7e8ca7d2ef2a4d90573520efa7f3a1aacf06",
    ),
    "mteed-model": (
        "DINKSTER_MTEED_TEST_MODEL",
        "blake3:bbcff8e81d853e788b06190e371f14207662b5d7a524c0eeb43da9d45060b3db",
        "a3c2d8a8ce9422555c787160bd46362d761325a565333c0e3f6a53e0bae2abdb",
    ),
}


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_HED_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_HED_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_HED_TEST_MODEL does not exist: {path}")
    return path


def _decode(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _decode_compressed(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    dtype = np.dtype(cast("str", payload["dtype"]))
    encoded = cast("str", payload["zlibBase64"])
    array = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=dtype).reshape(shape)
    assert hashlib.sha256(array.tobytes()).hexdigest() == payload["sha256"]
    return array


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def _vault(tmp_path: Path) -> AssetVault:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        with _model_path().open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                writer.write(chunk)
        writer.commit()
    return vault


def _artifact_path(asset_id: str) -> Path:
    environment, _, _ = ARTIFACTS[asset_id]
    value = os.environ.get(environment)
    if not value:
        pytest.skip(f"{environment} is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{environment} does not exist: {path}")
    return path


def _all_assets_vault(tmp_path: Path) -> AssetVault:
    vault = AssetVault(tmp_path / "all-assets-vault")
    for asset_id, (_, digest, _) in ARTIFACTS.items():
        with vault.writer(digest) as writer:
            with _artifact_path(asset_id).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    writer.write(chunk)
            writer.commit()
    return vault


def test_hed_outputs_match_pinned_controlnet_aux_vectors(tmp_path: Path) -> None:
    golden = _golden()
    assert golden["baseline"] == "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    assert golden["modelBlake3"] == MODEL_DIGEST
    assert golden["modelSha256"] == MODEL_SHA256
    assert golden["numpy"] == "2.5.1"
    assert golden["opencv"] == "5.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _vault(tmp_path))
    source = _decode(golden["source"])[None].astype(np.float32) / 255.0
    cases = cast("dict[str, object]", golden["cases"])
    with use_declared_asset_pack(manifest.name):
        actual = {
            "soft": execute_hed(source, safe=False, scribble=False, resolution=64),
            "safe": execute_hed(source, safe=True, scribble=False, resolution=64),
            "scribble": execute_hed(source, safe=True, scribble=True, resolution=64),
        }
    for name, output in actual.items():
        expected = _decode(cases[name])
        assert output.shape == (1, *expected.shape)
        assert output.dtype == np.float32
        np.testing.assert_array_equal(
            np.rint(output[0] * 255.0).astype(np.uint8),
            expected,
        )


def test_hed_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        golden = _golden()
        source = _decode(golden["source"])[None].astype(np.float32) / 255.0
        expected = _decode(cast("dict[str, object]", golden["cases"])["scribble"])
        vault = _vault(tmp_path)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            extra_env={"DINKSTER_ASSET_VAULT": str(vault.root)},
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
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "provider": "dinkster-vision-hed",
                            "safe": True,
                            "scribble": True,
                            "resolution": 64,
                        },
                    )
                }
            )
            result = await engine.run(graph, ["hed"])
            output = np.asarray(result.outputs["hed"]["image"].resolve())
            np.testing.assert_array_equal(
                np.clip(output[0] * 255.0, 0.0, 255.0).astype(np.uint8),
                expected,
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_learned_preprocessors_match_pinned_controlnet_aux_vectors(tmp_path: Path) -> None:
    golden = cast(
        "dict[str, object]", json.loads(LINE_EDGE_GOLDEN_PATH.read_text(encoding="utf-8"))
    )
    assert golden["controlnetAuxCommit"] == "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    assert golden["comfyuiCommit"] == "8a33128f2f8c5585c57486c07de481241e70a39c"
    assert golden["device"] == "cpu"
    assert golden["resolution"] == 256
    assert cast("dict[str, str]", golden["environment"])["scikitImage"] == "0.26.0"
    source = _decode_compressed(golden["source"])[None].astype(np.float32) / 255.0
    cases = cast("dict[str, object]", golden["cases"])
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _all_assets_vault(tmp_path))
    asyncio.run(MODEL_CACHE.shed(PressureSignal(device="ram", bytes_needed=2**63 - 1)))

    with use_declared_asset_pack(manifest.name):
        actual = {
            "lineart-realistic": execute_realistic(source, coarse=False, resolution=256),
            "lineart-realistic-coarse": execute_realistic(source, coarse=True, resolution=256),
            "lineart-anime": execute_anime(source, resolution=256),
            "lineart-manga": execute_manga(source, resolution=256),
            "hed-soft": execute_hed(source, safe=False, scribble=False, resolution=256),
            "hed-safe": execute_hed(source, safe=True, scribble=False, resolution=256),
            "hed-scribble": execute_hed(source, safe=True, scribble=True, resolution=256),
            "hed-scribble-unsafe": execute_hed(source, safe=False, scribble=True, resolution=256),
            "teed": execute_teed(source, safe_steps=2, resolution=256),
            "teed-unquantized": execute_teed(source, safe_steps=0, resolution=256),
            "mlsd": execute_mlsd(
                source, score_threshold=0.1, distance_threshold=0.1, resolution=256
            ),
            "mlsd-empty": execute_mlsd(
                source, score_threshold=2.0, distance_threshold=20.0, resolution=256
            ),
        }
    for name, output in actual.items():
        assert output.dtype == np.float32
        expected = _decode_compressed(cases[name])
        if name == "lineart-manga":
            # Float32 CPU convolutions may cross one uint8 truncation boundary
            # across AVX2 kernels; a second level is a parity failure.
            np.testing.assert_allclose(output, expected, rtol=0, atol=1.0 / 255.0 + 1e-7)
        else:
            np.testing.assert_array_equal(output, expected)


def test_mlsd_cuda_oom_retries_on_cpu_without_retaining_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = cast(
        "dict[str, object]", json.loads(LINE_EDGE_GOLDEN_PATH.read_text(encoding="utf-8"))
    )
    source = _decode_compressed(golden["source"])[None].astype(np.float32) / 255.0
    expected = _decode_compressed(cast("dict[str, object]", golden["cases"])["mlsd"])
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _all_assets_vault(tmp_path))
    MODEL_CACHE.discard("M-LSD")
    original_use = MODEL_CACHE.use
    original_frame = mlsd_module._mlsd_frame  # pyright: ignore[reportPrivateUsage]
    used_devices: list[str] = []
    frame_calls = 0

    @contextmanager
    def redirect_cuda_cache_to_cpu(
        key: str,
        factory: Callable[[], torch.nn.Module],
        *,
        device: torch.device | None = None,
    ) -> Iterator[torch.nn.Module]:
        assert device is not None
        used_devices.append(device.type)
        target = torch.device("cpu") if device.type == "cuda" else device
        with original_use(key, factory, device=target) as model:
            yield model

    def oom_during_first_frame(
        model: torch.nn.Module,
        frame: np.ndarray,
        resolution: int,
        score_threshold: float,
        distance_threshold: float,
    ) -> np.ndarray:
        nonlocal frame_calls
        frame_calls += 1
        if frame_calls == 1:
            raise torch.OutOfMemoryError("forced CUDA OOM")
        return original_frame(model, frame, resolution, score_threshold, distance_threshold)

    monkeypatch.setattr(MODEL_CACHE, "device", lambda: torch.device("cuda"))
    monkeypatch.setattr(MODEL_CACHE, "use", redirect_cuda_cache_to_cpu)
    monkeypatch.setattr(mlsd_module, "_mlsd_frame", oom_during_first_frame)
    with use_declared_asset_pack(manifest.name):
        actual = execute_mlsd(source, score_threshold=0.1, distance_threshold=0.1, resolution=256)

    np.testing.assert_array_equal(actual, expected)
    assert used_devices == ["cuda", "cpu"]
    assert all(
        item.item_id not in {"M-LSD", "M-LSD CPU fallback"} for item in MODEL_CACHE.details()
    )


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


def test_anyline_merge_arms_match_pinned_controlnet_aux_vectors(tmp_path: Path) -> None:
    golden = cast(
        "dict[str, object]", json.loads(LINE_EDGE_GOLDEN_PATH.read_text(encoding="utf-8"))
    )
    source = _decode_compressed(golden["source"])[None].astype(np.float32) / 255.0
    cases = cast("dict[str, object]", golden["cases"])
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _all_assets_vault(tmp_path))

    with use_declared_asset_pack(manifest.name):
        for merge in (
            "lineart_standard",
            "lineart_realisitic",
            "lineart_anime",
            "manga_line",
        ):
            output = execute_anyline(
                source,
                merge_with_lineart=merge,
                resolution=256,
                lineart_lower_bound=0.0,
                lineart_upper_bound=1.0,
                object_min_size=36,
                object_connectivity=1,
            )
            suffix = {
                "lineart_standard": "standard",
                "lineart_realisitic": "realistic",
                "lineart_anime": "anime",
                "manga_line": "manga",
            }[merge]
            np.testing.assert_array_equal(output, _decode_compressed(cases[f"anyline-{suffix}"]))


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


def test_model_artifact_is_the_expected_bytes() -> None:
    assert hashlib.sha256(_model_path().read_bytes()).hexdigest() == MODEL_SHA256


@pytest.mark.parametrize("asset_id", tuple(ARTIFACTS))
def test_line_edge_artifacts_are_the_expected_bytes(asset_id: str) -> None:
    _, _, expected = ARTIFACTS[asset_id]
    assert hashlib.sha256(_artifact_path(asset_id).read_bytes()).hexdigest() == expected
