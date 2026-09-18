"""CPU file/preprocessing parity and the two native-node import boundaries."""

from __future__ import annotations

import pickle
import subprocess
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import safetensors.torch
import torch
from dinkster_inference_torch.checkpoint import load_checkpoint, load_checkpoint_with_metadata
from dinkster_inference_torch.image_preprocess import clip_preprocess
from dinkster_inference_torch.sources import tensor_file_slice


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "torch": torch,
        "safetensors": safetensors,
        "comfy": SimpleNamespace(memory_management=SimpleNamespace(aimdo_enabled=False)),
        "DISABLE_MMAP": False,
        "MMAP_TORCH_FILES": False,
    }
    for name in ("clip_preprocess", "load_torch_file"):
        path = Path(__file__).with_name(f"{name}_reference.txt")
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 4, 4, 3), (2, 5, 9, 4), (2, 9, 5, 3), (1, 1, 7, 3)])
def test_preprocess_matches_upstream(
    reference: dict[str, Any], crop: bool, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    count = torch.Size(shape).numel()
    image = torch.linspace(-0.25, 1.25, count, dtype=dtype).reshape(shape).transpose(1, 2)
    original = image.clone()
    kwargs: dict[str, Any] = dict(size=4, crop=crop, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    try:
        expected = reference["clip_preprocess"](image, **kwargs)
    except (RuntimeError, NotImplementedError) as error:
        # CPU antialiased bicubic support is dtype-dependent in torch.
        with pytest.raises(type(error)) as actual:
            clip_preprocess(image, **kwargs)
        assert str(actual.value) == str(error)
    else:
        actual = clip_preprocess(image, **kwargs)
        assert torch.equal(actual, expected)
        assert actual.dtype == dtype
        assert actual.shape == (shape[0], 3, 4, 4)
    assert torch.equal(image, original)


def test_preprocess_preserves_grayscale_broadcast_and_nonfinite_values(
    reference: dict[str, Any],
) -> None:
    image = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5]).reshape(1, 2, 2, 1)
    expected = reference["clip_preprocess"](image, size=2)
    actual = clip_preprocess(image, size=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual.shape == (1, 3, 2, 2)


@pytest.mark.parametrize(
    "filename",
    [
        "weights.safetensors",
        "weights.SAFETENSORS",
        "weights.sft",
        "weights.SFT",
        ".safetensors",
        ".sft",
    ],
)
def test_checkpoint_reuses_native_mapped_reader(
    tmp_path: Path, reference: dict[str, Any], filename: str
) -> None:
    path = tmp_path / filename
    state = {
        "float": torch.arange(12, dtype=torch.float16).reshape(3, 4),
        "integer": torch.tensor([2, 7], dtype=torch.int64),
        "empty": torch.empty((0, 4), dtype=torch.bfloat16),
        "scalar": torch.tensor(True),
        "packed": torch.arange(6, dtype=torch.uint8).reshape(2, 3).view(torch.float4_e2m1fn_x2),
        "scale": torch.tensor([0, 127, 255], dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    safetensors.torch.save_file(state, path, metadata={"test": "metadata is not tensor state"})
    original = path.read_bytes()
    expected = reference["load_torch_file"](str(path), safe_load=True)
    actual = cast("dict[str, torch.Tensor]", load_checkpoint(path))
    assert actual.keys() == expected.keys() == state.keys()
    for key, value in actual.items():
        assert torch.equal(
            value.reshape(-1).view(torch.uint8), expected[key].reshape(-1).view(torch.uint8)
        )
        assert value.shape == expected[key].shape == state[key].shape
        assert value.dtype == state[key].dtype
        assert value.device.type == "cpu"
        assert value.layout == torch.strided
        if value.numel():
            assert tensor_file_slice(value) is not None
    actual["float"].add_(5)
    assert path.read_bytes() == original


def test_checkpoint_returns_safetensors_metadata(tmp_path: Path, reference: dict[str, Any]) -> None:
    path = tmp_path / "weights.safetensors"
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    safetensors.torch.save_file(state, path, metadata={"format": "pt", "model": "test"})
    expected = reference["load_torch_file"](str(path), safe_load=True, return_metadata=True)

    actual, metadata = load_checkpoint_with_metadata(path)

    assert isinstance(actual, dict)
    assert torch.equal(actual["weight"], expected[0]["weight"])
    assert metadata == expected[1] == {"format": "pt", "model": "test"}
    assert tensor_file_slice(actual["weight"]) is not None


def test_checkpoint_returns_no_metadata_for_plain_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    safetensors.torch.save_file({"weight": torch.ones(1)}, path)

    _, metadata = load_checkpoint_with_metadata(path)

    assert metadata is None


def test_checkpoint_returns_no_metadata_for_legacy_wrapped_state(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    state = {"weight": torch.arange(3, dtype=torch.float32)}
    torch.save({"state_dict": state, "epoch": 7}, path)

    actual, metadata = load_checkpoint_with_metadata(path)

    assert isinstance(actual, dict)
    assert torch.equal(actual["weight"], state["weight"])
    assert metadata is None
    unchanged = cast("dict[str, torch.Tensor]", load_checkpoint(path))
    assert torch.equal(unchanged["weight"], state["weight"])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("wrapper", ["plain", "state_dict", "model", "single", "empty", "invalid"])
def test_torch_checkpoint_wrapper_and_layout_parity(
    tmp_path: Path, reference: dict[str, Any], legacy: bool, wrapper: str
) -> None:
    path = tmp_path / "weights.pth"
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    state = OrderedDict(weight=base, transposed=base.T)
    payload: Any = state
    if wrapper == "state_dict":
        payload = {"state_dict": state, "epoch": 3}
    elif wrapper == "model":
        payload = {"model": state}
    elif wrapper == "single":
        payload = {"weight": base}
    elif wrapper == "empty":
        payload = {}
    elif wrapper == "invalid":
        payload = {"state_dict": None}
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
    expected = reference["load_torch_file"](str(path), safe_load=True)
    actual = load_checkpoint(path)
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            assert torch.equal(value, expected[key])
            assert value.dtype == expected[key].dtype
            assert value.stride() == expected[key].stride()
            assert value.device.type == "cpu"
        if "transposed" in actual:
            assert actual["weight"].untyped_storage() is actual["transposed"].untyped_storage()
    else:
        assert actual is expected is None


class _UntrustedTrainingObject:
    pass


def test_torch_checkpoint_refuses_unregistered_objects_without_leaking_globals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {}, "training": _UntrustedTrainingObject()}, path)
    before = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_checkpoint(path)
    assert set(torch.serialization.get_safe_globals()) == before


@pytest.mark.parametrize("preexisting", [False, True])
def test_torch_checkpoint_accepts_reference_legacy_metadata(
    tmp_path: Path, preexisting: bool
) -> None:
    import numpy as np

    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "dtype": np.dtype("float64")}, path)
    with torch.serialization.safe_globals([np.dtype] if preexisting else []):
        before = set(torch.serialization.get_safe_globals())
        actual = cast("dict[str, torch.Tensor]", load_checkpoint(path))
        assert torch.equal(actual["weight"], torch.ones(1))
        assert set(torch.serialization.get_safe_globals()) == before


def test_concurrent_legacy_loads_do_not_remove_each_others_safe_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_entered = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    before = set(torch.serialization.get_safe_globals())

    def load(path: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"map_location": torch.device("cpu"), "weights_only": True}
        if path.name == "first.pt":
            first_entered.set()
            assert release_first.wait(10)
        else:
            second_entered.set()
        added = set(torch.serialization.get_safe_globals()) - before
        assert all(callable(item) for item in added)
        names = {f"{cast(Any, item).__module__}.{cast(Any, item).__name__}" for item in added}
        assert {
            "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "numpy.core.multiarray.scalar",
            "_codecs.encode",
        } <= names
        return {}

    def second() -> object:
        second_started.set()
        return load_checkpoint(tmp_path / "second.pt")

    monkeypatch.setattr(torch, "load", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(load_checkpoint, tmp_path / "first.pt")
        try:
            assert first_entered.wait(10)
            other = executor.submit(second)
            assert second_started.wait(10)
            assert not second_entered.wait(0.1)
        finally:
            release_first.set()
        assert first.result(timeout=10) == other.result(timeout=10) == {}
    assert second_entered.is_set()
    assert set(torch.serialization.get_safe_globals()) == before


@pytest.mark.parametrize("batch", [1, 2])
def test_native_utility_execution_without_upstream_imports(tmp_path: Path, batch: int) -> None:
    script = r"""
import importlib.abc
import sys

class BlockUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"comfy", "comfy_extras", "nodes"}:
            raise AssertionError("upstream import: " + fullname)

sys.meta_path.insert(0, BlockUpstream())
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_compat_comfy import native_arm as arm
from dinkster_inference_torch.image_preprocess import clip_preprocess

path = Path(sys.argv[1]) / "weights.pt"
torch.save({"model": {"weight": torch.tensor([3.0])}}, path)
asset = AssetRef(digest=digest_file(path), name=path.name, size=path.stat().st_size,
                 resolver=SimpleNamespace(resolve=lambda digest: path))
assert torch.equal(arm._load_comfy_state_dict(asset)["weight"], torch.tensor([3.0]))
events = []
pixels_seen = []
@contextmanager
def stage(*, observer_stage):
    assert observer_stage == "encode"
    events.append("enter")
    try:
        yield
    finally:
        events.append("exit")

def component(*, pixel_values):
    assert events == ["enter"]
    assert torch.is_inference_mode_enabled()
    assert pixel_values.shape == (1, 3, 4, 4)
    assert pixel_values.dtype == torch.float64
    pixels_seen.append(pixel_values.clone())
    return pixel_values.mean(dim=1, keepdim=True)

batch = int(sys.argv[2])
image = torch.linspace(-0.1, 1.1, batch * 5 * 7 * 4).reshape(batch, 5, 7, 4)
handle = SimpleNamespace(stage=stage, module=component, load_device=torch.device("cpu"))
model = arm._NativeBackgroundRemovalModel(handle, torch.float64, 4,
                                          (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
result = arm.GenerationRemoveBackground.execute(model=model, image=image)["mask"]
expected_pixels = clip_preprocess(image, size=4, mean=model.image_mean,
                                  std=model.image_std, crop=False).to(torch.float64)
assert torch.equal(torch.cat(pixels_seen), expected_pixels)
expected = torch.nn.functional.interpolate(expected_pixels.mean(dim=1, keepdim=True),
                  size=(5, 7), mode="bicubic", antialias=False).sigmoid().float().squeeze(1)
assert torch.equal(result, expected)
assert result.shape == (batch, 5, 7) and result.device.type == "cpu"
assert events == ["enter", "exit"]
assert not any(name.split(".")[0] in {"comfy", "comfy_extras", "nodes"} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(batch)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
