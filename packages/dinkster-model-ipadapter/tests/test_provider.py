"""First-party SD1.5 IP-Adapter node and provider tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from dinkster_inference import InferenceComponentHandle, extend_runtime_identity
from dinkster_model_ipadapter import IPADAPTER_MODEL_NODE_IDS, IPADAPTER_MODEL_NODES, provider


class _Asset:
    def __init__(self, digest: str, size: int, path: str) -> None:
        self.digest = digest
        self.size = size
        self.path = Path(path)
        self.path_reads = 0

    def local_path(self) -> Path:
        self.path_reads += 1
        return self.path


class _Handle:
    load_device = torch.device("cpu")

    def __init__(self, identity: str, component: object) -> None:
        self.resource_identity = identity
        self._component = component
        self.staged = False

    @property
    def component(self) -> object:
        if not self.staged:
            raise RuntimeError("component read outside its lease")
        return self._component

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self):  # noqa: ANN201
        self.staged = True
        try:
            yield
        finally:
            self.staged = False

    @contextmanager
    def stage_with(self, _runtime_handle: object, _role: str):  # noqa: ANN201
        with self.stage():
            yield


def _assets() -> tuple[_Asset, _Asset]:
    return (
        _Asset(
            provider._ADAPTER_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._ADAPTER_ASSET_SIZE,  # pyright: ignore[reportPrivateUsage]
            "ip-adapter_sd15.safetensors",
        ),
        _Asset(
            provider._CLIP_VISION_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._CLIP_VISION_ASSET_SIZE,  # pyright: ignore[reportPrivateUsage]
            "model.safetensors",
        ),
    )


def test_pack_exposes_only_the_native_standard_sd15_surface() -> None:
    assert IPADAPTER_MODEL_NODE_IDS == (
        "dinkster.load_sd15_ipadapter",
        "dinkster.apply_sd15_ipadapter",
    )
    schemas = {node.schema().node_type: node.schema() for node in IPADAPTER_MODEL_NODES}
    assert schemas["dinkster.load_sd15_ipadapter"].outputs[0].type.types == (
        "dinkster.sd15-ipadapter",
    )
    apply = schemas["dinkster.apply_sd15_ipadapter"]
    assert tuple(item.id for item in apply.inputs) == (
        "model",
        "ipadapter",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "mask",
    )


def test_loader_refuses_nonofficial_artifacts_before_opening_them() -> None:
    adapter, vision = _assets()
    adapter.digest = "blake3:" + "0" * 64

    with pytest.raises(ValueError, match="pinned official"):
        provider.execute_load_sd15_ipadapter(adapter=adapter, clip_vision=vision)
    assert adapter.path_reads == vision.path_reads == 0


def test_loader_plans_assembles_and_publishes_both_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_asset, vision_asset = _assets()
    adapter_source = object()
    vision_source = object()
    plan = SimpleNamespace(adapter=object(), clip_vision=object())
    adapter_module = torch.nn.Identity()
    vision_module = torch.nn.Identity()
    assembled = SimpleNamespace(adapter=adapter_module, clip_vision=vision_module)
    adapter_identity = "native:dinkster.sd15:" + "1" * 64
    vision_identity = "native:dinkster.sd15:" + "2" * 64
    calls: list[tuple[object, ...]] = []

    def load(path: Path, *, asset_digest: str, asset_size: int) -> object:
        calls.append(("header", path, asset_digest, asset_size))
        return adapter_source if path == adapter_asset.path else vision_source

    def plan_components(
        adapter: object,
        vision: object,
        *,
        adapter_asset_digest: str,
        clip_vision_asset_digest: str,
    ) -> object:
        calls.append(
            (
                "plan",
                adapter,
                vision,
                adapter_asset_digest,
                clip_vision_asset_digest,
            )
        )
        return plan

    def assemble(value: object, *, adapter_dtype: torch.dtype) -> object:
        calls.append(("assemble", value, adapter_dtype))
        return assembled

    def identity(value: object, **kwargs: object) -> str:
        calls.append(("identity", value, kwargs))
        return adapter_identity if value is plan.adapter else vision_identity

    class Publisher:
        def publish(self, module: torch.nn.Module, *, resource_identity: str) -> _Handle:
            calls.append(("publish", module, resource_identity))
            return _Handle(resource_identity, module)

    monkeypatch.setattr(provider, "load_safetensors_header", load)
    monkeypatch.setattr(provider, "plan_sd15_ipadapter", plan_components)
    monkeypatch.setattr(provider, "assemble_sd15_ipadapter", assemble)
    monkeypatch.setattr(provider, "_component_identity", identity)
    monkeypatch.setattr(provider, "component_publisher", lambda: Publisher())

    output = provider.execute_load_sd15_ipadapter(
        adapter=adapter_asset,
        clip_vision=vision_asset,
    )

    resource = output["ipadapter"]
    assert type(resource) is provider.SD15IPAdapterResource
    assert resource.identity == extend_runtime_identity(
        adapter_identity, (f"clip_vision={vision_identity}",)
    )
    assert resource._dinkster_resident_refs == (  # pyright: ignore[reportPrivateUsage]
        resource.clip_vision,
    )
    assert calls == [
        (
            "header",
            adapter_asset.path,
            provider._ADAPTER_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._ADAPTER_ASSET_SIZE,  # pyright: ignore[reportPrivateUsage]
        ),
        (
            "header",
            vision_asset.path,
            provider._CLIP_VISION_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._CLIP_VISION_ASSET_SIZE,  # pyright: ignore[reportPrivateUsage]
        ),
        (
            "plan",
            adapter_source,
            vision_source,
            provider._ADAPTER_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._CLIP_VISION_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
        ),
        ("assemble", plan, torch.float16),
        (
            "identity",
            plan.adapter,
            {
                "role": "ipadapter",
                "asset_digest": provider._ADAPTER_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
                "compute_dtype": "float16",
            },
        ),
        (
            "identity",
            plan.clip_vision,
            {
                "role": "ipadapter_clip_vision",
                "asset_digest": provider._CLIP_VISION_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
                "compute_dtype": "float32",
            },
        ),
        ("publish", adapter_module, adapter_identity),
        ("publish", vision_module, vision_identity),
    ]


def test_image_and_mask_inputs_are_snapshotted() -> None:
    image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
    mask = torch.ones((1, 4, 4), dtype=torch.float32)

    image_snapshot = provider._image(image)  # pyright: ignore[reportPrivateUsage]
    mask_snapshot = provider._mask(mask)  # pyright: ignore[reportPrivateUsage]
    image.zero_()
    mask.zero_()

    assert torch.count_nonzero(image_snapshot).item() == image_snapshot.numel()
    assert mask_snapshot is not None
    assert torch.count_nonzero(mask_snapshot).item() == mask_snapshot.numel()


def test_resource_identity_must_bind_both_resident_components() -> None:
    adapter_identity = "native:dinkster.sd15:" + "1" * 64
    vision_identity = "native:dinkster.sd15:" + "2" * 64
    adapter = cast("InferenceComponentHandle", _Handle(adapter_identity, object()))
    vision = cast("InferenceComponentHandle", _Handle(vision_identity, object()))

    with pytest.raises(ValueError, match="does not bind"):
        provider.SD15IPAdapterResource(
            adapter,
            vision,
            provider._ADAPTER_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            provider._CLIP_VISION_ASSET_DIGEST,  # pyright: ignore[reportPrivateUsage]
            adapter_identity,
        )
