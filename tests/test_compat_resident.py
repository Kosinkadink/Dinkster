"""Worker-resident values (compat pack): loaded models never cross the
boundary, stubs do. Pack-level proof of the identity-crosses/content-stays
shape using only public TypeRegistry codec hooks - no engine changes."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib
import json
import sys
import threading
import types
import weakref
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    DEFAULT_RESIDENT_V1_TYPES,
    ResidencyTable,
    ResidentLookupError,
    comfy_execution,
    register_resident_type,
    translate_mappings,
)
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCE_REFS_META_KEY,
    TypeRegistry,
    UnresolvablePayload,
    register_core_types,
    value_resource_ids,
)
from dinkster_workers import IsolatedWorker

TESTS_DIR = Path(__file__).parent


class Opaque:
    """Unpicklable stand-in for a loaded model."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


# --- residency table and codec, pure ------------------------------------


def test_compat_resident_names_are_the_shared_dinkster_values_mechanism() -> None:
    import dinkster_values

    assert ResidencyTable is dinkster_values.ResidencyTable
    assert ResidentLookupError is dinkster_values.ResidentLookupError
    assert register_resident_type is dinkster_values.register_resident_type


def test_table_reuses_id_per_object_and_rejects_unknown() -> None:
    table = ResidencyTable()
    a, b = Opaque(), Opaque()
    rid_a = table.rid_for(a)
    assert table.rid_for(a) == rid_a  # idempotent per object
    assert table.rid_for(b) != rid_a  # distinct per object
    assert table.get(rid_a) is a
    assert len(table) == 2
    with pytest.raises(ResidentLookupError, match="another worker"):
        table.get("no-such-rid")


def test_resident_codec_sends_stub_and_resolves_locally() -> None:
    table = ResidencyTable()
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", table)
    spec = registry.spec("comfy.MODEL")

    model = Opaque()  # pickling this would raise: the codec must not try
    data = spec.encode(model)
    stub = json.loads(data)
    assert set(stub) == {"residentId"}  # small JSON stub, no object bytes
    assert len(data) < 128
    assert spec.decode(data) is model  # same process: exact object back

    value = registry.wrap("comfy.MODEL", model)
    assert value.fingerprint.startswith("resident:")
    assert value.fingerprint == registry.wrap("comfy.MODEL", model).fingerprint


def test_resident_envelope_stamps_owner_provenance() -> None:
    from dinkster_values import RESOURCE_OWNER_META_KEY, process_instance_token

    table = ResidencyTable()
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", table)
    # This process HOLDS the object, so the envelope carries this process's
    # lifetime token (stage 6 dispatch: consumers route back to the owner,
    # and a token from an earlier worker lifetime is known-dead).
    value = registry.wrap("comfy.MODEL", Opaque())
    assert value.meta.get(RESOURCE_OWNER_META_KEY) == process_instance_token()


def test_stub_routed_to_wrong_table_fails_clearly() -> None:
    ours, theirs = TypeRegistry(), TypeRegistry()
    register_resident_type(ours, "comfy.MODEL", ResidencyTable())
    register_resident_type(theirs, "comfy.MODEL", ResidencyTable())
    data = ours.spec("comfy.MODEL").encode(Opaque())
    with pytest.raises(ResidentLookupError, match="another worker"):
        theirs.spec("comfy.MODEL").decode(data)


def test_malformed_stub_fails_clearly() -> None:
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", ResidencyTable())
    with pytest.raises(ResidentLookupError, match="malformed"):
        registry.spec("comfy.MODEL").decode(b'{"other": 1}')


# --- translator routing --------------------------------------------------


class V1Loader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LATENT")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206 - v1 shape, deliberately untyped
        return {"required": {"name": ("STRING", {"default": "x"})}}

    def load(self, name):  # noqa: ANN001, ANN201
        return (object(), object(), object(), {"samples": None})


def test_register_types_routes_resident_v1_types_to_resident_codec() -> None:
    assert {"MODEL", "MOGE_MODEL", "BACKGROUND_REMOVAL"} <= DEFAULT_RESIDENT_V1_TYPES
    translation = translate_mappings({"Loader": V1Loader})
    registry = TypeRegistry()
    translation.register_types(registry)
    # Loaded hardware state gets the resident codec (declared)...
    for resident in ("comfy.MODEL", "comfy.CLIP", "comfy.VAE"):
        assert registry.spec(resident).declared_codec is True
    # ...latents use the declared cross-process structural codec.
    assert registry.spec("comfy.LATENT").declared_codec is True
    assert registry.spec("comfy.LATENT").validate_encoded is not None


# --- device residency and cost meta ---------------------------------------


class FakePatcher:
    """Duck-typed ModelPatcher: load_device + model_size."""

    def __init__(self, device: str, size: int) -> None:
        self.load_device = device
        self._size = size

    def model_size(self) -> int:
        return self._size


class FakeClip:
    """CLIP-shaped: the patcher rides at .patcher."""

    def __init__(self, device: str, size: int) -> None:
        self.patcher = FakePatcher(device, size)


def test_resident_meta_reads_patcher_devices_and_cost() -> None:
    from dinkster_compat_comfy import comfy_resident_meta

    meta = comfy_resident_meta(FakePatcher("cuda:1", 1000))
    assert meta == {
        "resources": {"gpu": "cuda:1"},
        # "ram" is the offload copy comfy keeps for the model's lifetime.
        "cost": {"vram:cuda:1": 1000, "ram": 1000},
    }
    # CLIP/VAE shape: patcher at .patcher.
    assert comfy_resident_meta(FakeClip("cuda:0", 10))["resources"] == {"gpu": "cuda:0"}
    # CPU-resident and unreadable objects contribute nothing - admission
    # falls back to the abstract lane instead of guessing.
    assert comfy_resident_meta(FakePatcher("cpu", 1000)) == {}
    assert comfy_resident_meta(object()) == {}


def test_resident_meta_spanning_devices_splits_cost() -> None:
    from dinkster_compat_comfy import comfy_resident_meta

    patcher = FakePatcher("cuda:0", 1000)
    patcher.load_devices = ["cuda:0", "cuda:1"]  # type: ignore[attr-defined]
    meta = comfy_resident_meta(patcher)
    assert meta["resources"] == {"gpu": ("cuda:0", "cuda:1")}
    # vram split across devices; ram is the single full-size offload copy.
    assert meta["cost"] == {"vram:cuda:0": 500, "vram:cuda:1": 500, "ram": 1000}


def test_resident_meta_honors_declared_non_model_cost() -> None:
    from dinkster_compat_comfy import comfy_resident_meta

    class Resident:
        _dinkster_resident_cost = {"ram": 256}

    assert comfy_resident_meta(Resident()) == {"cost": {"ram": 256}}

    Resident._dinkster_resident_cost = {"ram": -1}
    with pytest.raises(TypeError, match="nonnegative"):
        comfy_resident_meta(Resident())


def test_resident_meta_rides_the_envelope() -> None:
    from dinkster_compat_comfy import comfy_resident_meta

    registry = TypeRegistry()
    table = ResidencyTable()
    register_resident_type(registry, "comfy.MODEL", table, meta=comfy_resident_meta)
    obj = FakePatcher("cuda:1", 42)
    value = registry.wrap("comfy.MODEL", obj)
    assert value.meta.get("resources") == {"gpu": "cuda:1"}
    assert value.meta.get("cost") == {"vram:cuda:1": 42, "ram": 42}
    # Every resident envelope declares what it references: holders use
    # this to exclude the cost from their accounting (the table owns it)
    # and to invalidate entries when the resident is released.
    assert value.meta.get("resourceId") == "resident:" + table.rid_for(obj)


def test_dependent_resident_uses_owner_resource_without_duplicate_pool_cost() -> None:
    from dinkster_compat_comfy.pool import ResidentPool

    class Dependent:
        def __init__(self, owner: object) -> None:
            self.owner = owner

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

    pool = ResidentPool(cost_of=lambda _obj: {"ram": 42})
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", pool)
    owner = Opaque()
    dependent = Dependent(owner)

    value = registry.wrap("comfy.MODEL", dependent)
    owner_rid = pool.rid_for(owner)
    encoded = registry.spec("comfy.MODEL").encode(dependent)
    dependent_rid = json.loads(encoded)["residentId"]

    assert dependent_rid != owner_rid
    assert value.meta.get("resourceId") == "resident:" + owner_rid
    assert value.fingerprint == "resident:" + dependent_rid
    assert registry.spec("comfy.MODEL").decode(encoded) is dependent
    assert len(pool.details()) == 1
    pool.remove(owner_rid)
    with pytest.raises(ResidentLookupError):
        registry.spec("comfy.MODEL").decode(encoded)
    assert len(pool) == 0
    assert pool.details() == []


def test_resident_conditioning_is_reclaimed_with_its_released_owner() -> None:
    from dinkster_compat_comfy.pool import ResidentPool
    from dinkster_inference import (
        CONDITIONING_TYPE_ID,
        ResidentConditioningCarrier,
        register_conditioning_type,
    )

    class Payload:
        def __init__(self, owner: object, iteration: int) -> None:
            self.owner = owner
            self.iteration = iteration
            self.large_map = bytearray(1024 * 1024)

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.owner

        @property
        def _dinkster_resident_fingerprint(self) -> str:
            return f"conditioning:{self.iteration}"

    events: list[tuple[str, object]] = []

    def terminal_release(obj: object) -> bool:
        events.append(("release", id(obj)))
        return True

    pool = ResidentPool(
        cost_of=lambda _obj: {"cost": {"ram": 1024 * 1024}},
        terminal_release=terminal_release,
    )
    pool.register_invalidator(lambda resource_id: events.append(("invalidate", resource_id)))
    registry = TypeRegistry()
    spec = register_conditioning_type(registry, resident_table=pool)

    for iteration in range(3):
        owner = Opaque()
        payload = Payload(owner, iteration)
        carrier = ResidentConditioningCarrier(payload)
        value = registry.wrap(CONDITIONING_TYPE_ID, carrier)
        encoded = spec.encode(value.resolve())
        resource_id = value.meta.get(RESOURCE_ID_META_KEY)
        payload_ref = weakref.ref(payload)

        assert resource_id == "resident:" + pool.rid_for(owner)
        assert len(pool) == 2
        del payload, carrier, value
        gc.collect()
        assert payload_ref() is not None

        events.clear()
        assert pool.release_resident(owner) is True
        assert events == [("invalidate", resource_id), ("release", id(owner))]
        assert len(pool) == 0
        with pytest.raises(ResidentLookupError):
            spec.decode(encoded)
        gc.collect()
        assert payload_ref() is None


def test_composite_resident_names_every_owner_and_uses_declared_fingerprint() -> None:
    from dinkster_compat_comfy.pool import ResidentPool

    class Composite:
        def __init__(self, primary: object, extra: object) -> None:
            self.primary = primary
            self.extra = extra

        @property
        def _dinkster_resident_owner(self) -> object:
            return self.primary

        @property
        def _dinkster_resident_refs(self) -> tuple[object, ...]:
            return (self.extra,)

        @property
        def _dinkster_resident_fingerprint(self) -> str:
            return "composite:stable"

    pool = ResidentPool(cost_of=lambda _obj: {"ram": 42})
    registry = TypeRegistry()
    register_resident_type(registry, "comfy.MODEL", pool)
    primary = Opaque()
    extra = Opaque()
    composite = Composite(primary, extra)

    value = registry.wrap("comfy.MODEL", composite)
    primary_id = "resident:" + pool.rid_for(primary)
    extra_id = "resident:" + pool.rid_for(extra)

    assert value.fingerprint == "composite:stable"
    assert value.meta.get(RESOURCE_REFS_META_KEY) == (extra_id,)
    assert value_resource_ids(value) == (primary_id, extra_id)

    encoded = registry.spec("comfy.MODEL").encode(composite)
    pool.remove(extra_id.removeprefix("resident:"))
    with pytest.raises(ResidentLookupError):
        registry.spec("comfy.MODEL").decode(encoded)
    assert pool.get(primary_id.removeprefix("resident:")) is primary
    assert len(pool) == 1


def test_nodes_with_resident_inputs_occupy_gpu() -> None:
    class V1Sampler:
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "sample"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"model": ("MODEL",), "latent": ("LATENT",)}}

        def sample(self, model, latent):  # noqa: ANN001, ANN201
            return (latent,)

    translation = translate_mappings({"Sampler": V1Sampler, "Loader": V1Loader})
    schemas = {c.schema().node_type: c.schema() for c in translation.node_classes}
    # Consuming loaded hardware state occupies a GPU lane; producing it
    # (the loader) does not - loading targets the offload device.
    assert schemas["comfy.Sampler"].occupies == ("gpu",)
    assert schemas["comfy.Loader"].occupies == ()


# --- native asset loader registration ------------------------------------


def test_native_loader_schema_is_asset_in_resident_out() -> None:
    from dinkster_assets import ASSET_TYPE
    from dinkster_compat_comfy import LoadCheckpoint
    from dinkster_schema import AssetWidget

    schema = LoadCheckpoint.schema()
    assert schema.node_type == "dinkster.load_checkpoint"
    # The legacy-name claim (2026-07-26 model-loader porting): translated
    # comfy.CheckpointLoaderSimple is evicted, API prompts naming the old
    # class reach this port through the ckpt_name adapter.
    assert schema.aliases == ("CheckpointLoaderSimple",)
    assert [spec.id for spec in schema.inputs] == ["checkpoint"]
    assert schema.inputs[0].type.types == (ASSET_TYPE,)
    assert schema.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/checkpoint"
    )
    assert {out.id: out.type.types for out in schema.outputs} == {
        "model": ("comfy.MODEL",),
        "clip": ("comfy.CLIP",),
        "vae": ("comfy.VAE",),
    }


def test_media_image_loader_owns_legacy_alias_outside_compat() -> None:
    import dinkster_compat_comfy
    from dinkster_compat_comfy.native import MEDIA_IO_CLAIMED_V1_NAMES, NATIVE_NODES
    from dinkster_nodes_media_io import LoadImage
    from dinkster_schema import AssetWidget, SourceFilenameSpec, TypeExpr
    from dinkster_workers import load_manifest

    assert not hasattr(dinkster_compat_comfy, "LoadImage")
    assert not hasattr(dinkster_compat_comfy, "SaveImage")
    schema = LoadImage.schema()
    assert LoadImage not in NATIVE_NODES
    assert MEDIA_IO_CLAIMED_V1_NAMES == ("LoadImage", "SaveImage")
    assert schema.node_type == "dinkster.load_image"
    assert schema.aliases == ("LoadImage",)
    assert schema.inputs[0].type == TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
    assert schema.inputs[0].widget == AssetWidget(
        ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff"),
        kind="media/image",
        allow_upload=True,
    )
    assert schema.inputs[0].source_filename == SourceFilenameSpec("media/image", "input")
    manifest = load_manifest(
        Path(__file__).parents[1] / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"
    )
    assert {dependency.pack: dependency.version for dependency in manifest.dependencies}[
        "dinkster-nodes-media-io"
    ] == "<1,>=0.0.1"
    assert [output.type.types for output in schema.outputs] == [
        ("dinkster.image",),
        ("dinkster.mask",),
        ("core.string",),
    ]


def test_image_resize_claim_is_scoped_to_compat_merge() -> None:
    from dinkster_compat_comfy.native import IMAGE_CLAIMED_V1_NAMES
    from dinkster_nodes_image import ImageResize

    assert IMAGE_CLAIMED_V1_NAMES == ("ResizeImageMaskNode",)
    assert ImageResize.schema().aliases == ()


def test_native_types_resolve_upload_vault_before_indexed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker binding understands the upload CAS layout directly. The
    ref proves the registry-bound resolver reaches bytes by digest, without
    relying on or copying through a ComfyUI model-library index."""
    from dinkster_assets import AssetRef, AssetVault, digest_bytes
    from dinkster_compat_comfy import register_native_types

    payload = b"uploaded image bytes"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    monkeypatch.delenv("DINKSTER_ASSET_ROOT", raising=False)
    registry = TypeRegistry()
    register_native_types(registry)
    wrapped = registry.wrap(
        "dinkster.asset", {"digest": digest, "name": "upload.png", "size": len(payload)}
    )
    ref = wrapped.payload.load()
    assert isinstance(ref, AssetRef)
    assert ref.read_bytes() == payload


def test_media_image_loader_decodes_asset_png(tmp_path: Path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    from dinkster_assets import AssetRef, AssetVault, digest_bytes
    from dinkster_nodes_media_io import LoadImage

    path = tmp_path / "pixel.png"
    pil_image.new("RGBA", (2, 1), (128, 64, 32, 128)).save(path)
    payload = path.read_bytes()
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    result = LoadImage.execute(
        image=AssetRef(digest, "pixel.png", len(payload), "image/png", resolver=vault)
    )
    pixels = cast("Any", result["image"])
    mask = cast("Any", result["mask"])
    assert pixels.dtype == np.float32
    assert mask.dtype == np.float32
    assert tuple(pixels.shape) == (1, 1, 2, 3)
    assert pixels[0, 0, 0].tolist() == pytest.approx([128 / 255, 64 / 255, 32 / 255])
    assert tuple(mask.shape) == (1, 1, 2)
    assert mask[0, 0, 0].item() == pytest.approx(1 - 128 / 255)


def test_native_empty_latent_schema_is_ints_in_latent_out() -> None:
    from dinkster_compat_comfy import EmptyLatentImage
    from dinkster_values import CORE_INT

    schema = EmptyLatentImage.schema()
    assert schema.node_type == "dinkster.empty_latent_image"
    assert schema.aliases == ("EmptyLatentImage",)
    assert "EmptyLatentImage" in schema.search_terms
    assert {spec.id: spec.default for spec in schema.inputs} == {
        "width": 512,
        "height": 512,
        "batch_size": 1,
    }
    assert all(spec.type.types == (CORE_INT,) for spec in schema.inputs)
    assert all(not spec.required for spec in schema.inputs)
    assert [out.type.types for out in schema.outputs] == [("comfy.LATENT",)]


def test_native_empty_latent_uses_native_device_and_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from dinkster_compat_comfy import EmptyLatentImage

    imported = _record_dynamic_imports(monkeypatch)
    result = EmptyLatentImage.execute(width=64, height=32, batch_size=2)
    latent = cast("Any", result["latent"])
    assert latent["downscale_ratio_spacial"] == 8
    samples = latent["samples"]
    assert samples.dtype == torch.float32
    assert samples.device.type == "cpu"
    assert tuple(samples.shape) == (2, 4, 4, 8)  # [batch, 4, h//8, w//8]
    assert samples.abs().sum().item() == 0.0
    assert "comfy.model_management" not in imported
    # The v1 min/max contract survives the port as execute-side checks.
    with pytest.raises(ValueError, match="width"):
        EmptyLatentImage.execute(width=8, height=32, batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        EmptyLatentImage.execute(width=64, height=64, batch_size=0)


@pytest.mark.parametrize("comfy_installed", (False, True))
def test_native_empty_hunyuan_video_latent_matches_wan_shape(
    monkeypatch: pytest.MonkeyPatch,
    comfy_installed: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from dinkster_compat_comfy import EmptyHunyuanLatentVideo
    from dinkster_values import CORE_INT

    schema = EmptyHunyuanLatentVideo.schema()
    assert schema.node_type == "dinkster.empty_hunyuan_latent_video"
    assert schema.aliases == ("EmptyHunyuanLatentVideo",)
    assert schema.category == "model/latent/hunyuan video"
    assert {spec.id: spec.default for spec in schema.inputs} == {
        "width": 848,
        "height": 480,
        "length": 25,
        "batch_size": 1,
    }
    assert all(spec.type.types == (CORE_INT,) for spec in schema.inputs)

    if comfy_installed:
        monkeypatch.setitem(
            sys.modules, "comfy.model_management", types.ModuleType("comfy.model_management")
        )
    else:
        monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    imported = _record_dynamic_imports(monkeypatch)
    result = EmptyHunyuanLatentVideo.execute(width=64, height=32, length=9, batch_size=2)
    latent = cast("Any", result["latent"])
    assert latent["downscale_ratio_spacial"] == 8
    samples = latent["samples"]
    assert samples.dtype == torch.float32
    assert samples.device.type == "cpu"
    assert tuple(samples.shape) == (2, 16, 3, 4, 8)
    assert samples.abs().sum().item() == 0.0
    assert "comfy.model_management" not in imported
    with pytest.raises(ValueError, match="length"):
        EmptyHunyuanLatentVideo.execute(width=64, height=32, length=8, batch_size=1)


def test_native_empty_latent_honors_fp16_intermediates_without_model_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from dinkster_compat_comfy import EmptyLatentImage

    model_management = types.ModuleType("comfy.model_management")
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setattr(sys, "argv", ["worker", "--fp16-intermediates"])
    imported = _record_dynamic_imports(monkeypatch)

    samples = cast("Any", EmptyLatentImage.execute(width=64, height=32, batch_size=1)["latent"])[
        "samples"
    ]
    assert samples.dtype == torch.float16
    assert samples.device.type == "cpu"
    assert "comfy.model_management" not in imported


def test_native_wan22_start_image_uses_native_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from dinkster_compat_comfy import Wan22ImageToVideoLatent

    resize = types.ModuleType("dinkster_inference_torch.resize")
    resize_calls: list[tuple[object, ...]] = []

    def common_upscale(samples, width, height, method, crop):  # noqa: ANN001, ANN202
        resize_calls.append((tuple(samples.shape), width, height, method, crop))
        return torch.zeros((samples.shape[0], samples.shape[1], height, width))

    setattr(resize, "common_upscale", common_upscale)  # noqa: B010
    latent_formats = types.ModuleType("comfy.latent_formats")
    setattr(latent_formats, "Wan22", lambda: types.SimpleNamespace(process_out=lambda value: value))  # noqa: B010
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.resize", resize)
    monkeypatch.setitem(sys.modules, "comfy.latent_formats", latent_formats)
    imported = _record_dynamic_imports(monkeypatch)

    class Vae:
        @staticmethod
        def encode(image):  # noqa: ANN001, ANN205
            assert tuple(image.shape) == (1, 32, 64, 3)
            return torch.ones((1, 48, 1, 2, 4))

    latent = cast(
        "Mapping[str, Any]",
        Wan22ImageToVideoLatent.execute(
            vae=Vae(),
            width=64,
            height=32,
            length=5,
            batch_size=1,
            start_image=torch.ones((1, 16, 32, 3)),
        )["latent"],
    )
    assert resize_calls == [((1, 3, 16, 32), 64, 32, "bilinear", "center")]
    assert latent["samples"].shape == (1, 48, 2, 2, 4)
    assert latent["noise_mask"].shape == (1, 1, 2, 2, 4)
    assert "dinkster_inference_torch.resize" in imported
    assert "comfy.utils" not in imported


def test_native_clip_text_encode_schema_is_text_clip_in_conditioning_out() -> None:
    from dinkster_compat_comfy import CLIPTextEncode
    from dinkster_schema import (
        StringWidget,
        WidgetRepresentation,
        WidgetRepresentations,
        schema_signature,
        schema_to_wire,
    )
    from dinkster_values import CORE_STRING

    schema = CLIPTextEncode.schema()
    assert schema.node_type == "dinkster.clip_text_encode"
    assert schema.aliases == ("CLIPTextEncode",)
    assert "CLIPTextEncode" in schema.search_terms
    assert {spec.id: spec.type.types for spec in schema.inputs} == {
        "text": (CORE_STRING,),
        "clip": ("comfy.CLIP",),
    }
    assert all(spec.required for spec in schema.inputs)
    assert [out.type.types for out in schema.outputs] == [("comfy.CONDITIONING",)]
    text_widget = schema.inputs[0].widget
    assert text_widget == WidgetRepresentations(
        representations=(
            WidgetRepresentation(
                "single-line",
                StringWidget(multiline=False, dynamic_prompts=True),
                display_name="Single line",
            ),
            WidgetRepresentation(
                "multiline",
                StringWidget(multiline=True, dynamic_prompts=True),
                display_name="Multiline",
            ),
        ),
        default="multiline",
        user_switchable=True,
    )
    wire19 = schema_to_wire(schema)
    text19 = next(entry for entry in wire19["interface"] if entry["id"] == "text")  # type: ignore[union-attr]
    assert text19["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "multiline",
        "userSwitchable": True,
        "representations": [
            {
                "id": "single-line",
                "displayName": "Single line",
                "widget": {
                    "type": "STRING",
                    "multiline": False,
                    "dynamicPrompts": True,
                },
            },
            {
                "id": "multiline",
                "displayName": "Multiline",
                "widget": {
                    "type": "STRING",
                    "multiline": True,
                    "dynamicPrompts": True,
                },
            },
        ],
    }
    wire18 = schema_to_wire(schema, wire_version=18)
    text18 = next(entry for entry in wire18["interface"] if entry["id"] == "text")  # type: ignore[union-attr]
    assert text18["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "multiline",
        "userSwitchable": True,
        "representations": [
            {
                "id": "single-line",
                "displayName": "Single line",
                "widget": {"type": "STRING", "multiline": False},
            },
            {
                "id": "multiline",
                "displayName": "Multiline",
                "widget": {"type": "STRING", "multiline": True},
            },
        ],
    }
    assert schema_signature(schema) != "40d9a01d8cbfcb2965e38d468014dc119da21fe9"


def test_native_clip_text_encode_tokenizes_then_encodes() -> None:
    """The port preserves ComfyUI's two-step contract - tokenize, then
    encode_from_tokens_scheduled - and names the checkpoint-without-text-
    encoder failure instead of exploding on None."""
    from dinkster_compat_comfy import CLIPTextEncode

    class FakeClip:
        def tokenize(self, text: str) -> dict[str, str]:
            return {"tokens": text}

        def encode_from_tokens_scheduled(self, tokens: dict[str, str]) -> list[object]:
            return [["embedding", {"from": tokens["tokens"]}]]

    literal = "a {fennec|fox} girl"
    result = CLIPTextEncode.execute(text=literal, clip=FakeClip())
    assert result["conditioning"] == [["embedding", {"from": literal}]]
    with pytest.raises(ValueError, match="text encoder"):
        CLIPTextEncode.execute(text="anything", clip=None)


def test_native_vae_decode_schema_is_latent_vae_in_image_out() -> None:
    from dinkster_compat_comfy import VAEDecode

    schema = VAEDecode.schema()
    assert schema.node_type == "dinkster.vae_decode"
    assert schema.aliases == ("VAEDecode",)
    assert "VAEDecode" in schema.search_terms
    assert {spec.id: spec.type.types for spec in schema.inputs} == {
        "samples": ("comfy.LATENT",),
        "vae": ("comfy.VAE",),
    }
    assert all(spec.required for spec in schema.inputs)
    assert [out.type.types for out in schema.outputs] == [("comfy.IMAGE",)]


def test_native_vae_decode_ports_v1_decode_semantics() -> None:
    """The port preserves ComfyUI's decode path exactly: plain latents go
    to vae.decode as-is, nested latents unbind to their first tensor
    first, and a 5-dim (video-batch) decode flattens batch dims."""
    from dinkster_compat_comfy import VAEDecode

    class FakeImages:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def reshape(self, *dims: int) -> tuple[str, tuple[int, ...]]:
            return ("reshaped", dims)

    class FakeVAE:
        def __init__(self, images: FakeImages) -> None:
            self.images = images
            self.decoded: list[object] = []

        def decode(self, latent: object) -> FakeImages:
            self.decoded.append(latent)
            return self.images

    class FakeLatent:
        is_nested = False

    plain = FakeLatent()
    vae = FakeVAE(FakeImages((2, 512, 512, 3)))
    result = VAEDecode.execute(samples={"samples": plain}, vae=vae)
    assert result["image"] is vae.images  # 4-dim: untouched
    assert vae.decoded == [plain]

    class NestedLatent:
        is_nested = True

        def __init__(self, inner: object) -> None:
            self.inner = inner

        def unbind(self) -> list[object]:
            return [self.inner]

    inner = FakeLatent()
    vae = FakeVAE(FakeImages((2, 8, 512, 512, 3)))
    result = VAEDecode.execute(samples={"samples": NestedLatent(inner)}, vae=vae)
    assert vae.decoded == [inner]  # nested: first tensor decoded
    assert result["image"] == ("reshaped", (-1, 512, 512, 3))  # 5-dim: flattened


def test_native_vae_encode_schema_and_latent_dict_shape() -> None:
    from dinkster_compat_comfy import VAEEncode

    schema = VAEEncode.schema()
    assert schema.node_type == "dinkster.vae_encode"
    assert schema.aliases == ("VAEEncode",)
    assert "VAEEncode" in schema.search_terms
    assert {spec.id: spec.type.types for spec in schema.inputs} == {
        "pixels": ("comfy.IMAGE",),
        "vae": ("comfy.VAE",),
    }
    assert all(spec.required for spec in schema.inputs)
    assert [out.type.types for out in schema.outputs] == [("comfy.LATENT",)]

    class FakeVAE:
        def encode(self, pixels: object) -> tuple[str, object]:
            return ("encoded", pixels)

    pixels = object()
    result = VAEEncode.execute(pixels=pixels, vae=FakeVAE())
    # the exact {"samples": ...} dict every sampler consumes, as v1 emits
    assert result["latent"] == {"samples": ("encoded", pixels)}


# --- native model loaders (LoRA/VAE/CLIP/UNET asset ports) ----------------


def _vault_ref(tmp_path: Path, name: str, payload: bytes) -> Any:  # AssetRef
    from dinkster_assets import AssetRef, AssetVault, digest_bytes

    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    return AssetRef(digest, name, len(payload), resolver=vault)


def _clear_lora_cache() -> None:
    import dinkster_native.native as native_module

    with native_module._lora_sd_cache_lock:  # noqa: SLF001 - test reset
        native_module._lora_sd_cache.clear()  # noqa: SLF001
        native_module._lora_sd_cache_bytes = 0  # noqa: SLF001


def _record_dynamic_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    imported: list[str] = []
    import_module = importlib.import_module

    def record(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", record)
    return imported


def test_native_model_loader_schemas_are_asset_in_resident_out() -> None:
    """The five loader ports are public exports whose schemas follow the
    checkpoint recipe: first input is a bare dinkster.asset, outputs are
    worker-resident model types, and the alias claims the legacy name."""
    from dinkster_assets import ASSET_TYPE
    from dinkster_compat_comfy import (
        LoadClip,
        LoadDiffusionModel,
        LoadLora,
        LoadLoraModelOnly,
        LoadVae,
    )
    from dinkster_schema import AssetWidget, NumberWidget, schema_to_wire
    from dinkster_values import CORE_COMBO

    expectations = [
        (
            LoadLora,
            "dinkster.load_lora",
            "LoraLoader",
            "lora",
            "model/lora",
            {"model": ("comfy.MODEL",), "clip": ("comfy.CLIP",)},
        ),
        (
            LoadLoraModelOnly,
            "dinkster.load_lora_model_only",
            "LoraLoaderModelOnly",
            "lora",
            "model/lora",
            {"model": ("comfy.MODEL",)},
        ),
        (LoadVae, "dinkster.load_vae", "VAELoader", "vae", "model/vae", {"vae": ("dinkster.vae",)}),
        (
            LoadClip,
            "dinkster.load_clip",
            "CLIPLoader",
            "text_encoder",
            "model/text-encoder",
            {"clip": ("comfy.CLIP",)},
        ),
        (
            LoadDiffusionModel,
            "dinkster.load_diffusion_model",
            "UNETLoader",
            "diffusion_model",
            "model/diffusion",
            {"model": ("comfy.MODEL",)},
        ),
    ]
    for node, node_type, legacy, asset_input, asset_kind, outputs in expectations:
        schema = node.schema()
        assert schema.node_type == node_type
        assert schema.aliases == (legacy,)
        assert legacy in schema.search_terms
        specs = {spec.id: spec for spec in schema.inputs}
        assert specs[asset_input].type.types == (ASSET_TYPE,)
        assert specs[asset_input].required is (node is not LoadVae)
        assert specs[asset_input].widget == AssetWidget(
            accept=("application/octet-stream",), kind=asset_kind
        )
        assert {out.id: out.type.types for out in schema.outputs} == outputs
    # The strength inputs keep v1's optional-with-default shape.
    lora_specs = {spec.id: spec for spec in LoadLora.schema().inputs}
    assert lora_specs["strength_model"].default == 1.0
    assert lora_specs["strength_clip"].default == 1.0
    strength_widget = NumberWidget(min=-100.0, max=100.0, step=0.01)
    assert lora_specs["strength_model"].widget == strength_widget
    assert lora_specs["strength_clip"].widget == strength_widget
    assert "strength_clip" not in {spec.id for spec in LoadLoraModelOnly.schema().inputs}
    model_only_specs = {spec.id: spec for spec in LoadLoraModelOnly.schema().inputs}
    assert model_only_specs["strength_model"].default == 1.0
    assert model_only_specs["strength_model"].widget == strength_widget
    for schema in (LoadLora.schema(), LoadLoraModelOnly.schema()):
        wire_interface = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])
        wire_inputs = {item["id"]: item for item in wire_interface}
        for input_id in ("strength_model", "strength_clip"):
            if input_id in wire_inputs:
                assert wire_inputs[input_id]["default"] == 1.0
                assert wire_inputs[input_id]["widget"] == {
                    "type": "NUMBER",
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }
    clip_specs = {spec.id: spec for spec in LoadClip.schema().inputs}
    assert clip_specs["type"].type.types == (CORE_COMBO,)
    assert clip_specs["device"].type.types == (CORE_COMBO,)
    diffusion_specs = {spec.id: spec for spec in LoadDiffusionModel.schema().inputs}
    assert diffusion_specs["weight_dtype"].type.types == (CORE_COMBO,)


def test_native_lora_loader_caches_by_digest_and_forwards_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 parity: the parsed LoRA is reused across executions (strength
    tweaks never re-read the file), keyed by digest so the same bytes hit
    the same entry under any filename. Metadata forwards to installs whose
    load_lora_for_models accepts it."""
    from dinkster_compat_comfy import LoadLora

    _clear_lora_cache()
    ref_a = _vault_ref(tmp_path / "a", "style.safetensors", b"lora bytes A")
    ref_b = _vault_ref(tmp_path / "b", "style.safetensors", b"lora bytes B")

    file_loads: list[str] = []
    native_checkpoint = types.ModuleType("dinkster_inference_torch.checkpoint")

    def load_checkpoint_with_metadata(path: Path):  # noqa: ANN202
        file_loads.append(str(path))
        return ({"sd_from": str(path)}, {"meta_from": str(path)})

    setattr(native_checkpoint, "load_checkpoint_with_metadata", load_checkpoint_with_metadata)  # noqa: B010

    applied: list[dict[str, Any]] = []
    comfy_sd = types.ModuleType("comfy.sd")

    def load_lora_for_models(  # noqa: ANN202
        model,
        clip,
        lora_sd,
        strength_model,
        strength_clip,
        *,
        lora_metadata=None,  # noqa: ANN001
    ):
        applied.append(
            {
                "model": model,
                "clip": clip,
                "lora_sd": lora_sd,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
                "lora_metadata": lora_metadata,
            }
        )
        return (("patched", model), ("patched", clip))

    setattr(comfy_sd, "load_lora_for_models", load_lora_for_models)  # noqa: B010
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.checkpoint", native_checkpoint)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    imported = _record_dynamic_imports(monkeypatch)

    model, clip = object(), object()
    result = LoadLora.execute(
        model=model, clip=clip, lora=ref_a, strength_model=0.8, strength_clip=0.5
    )
    assert result["model"] == ("patched", model)
    assert result["clip"] == ("patched", clip)
    assert len(file_loads) == 1
    assert applied[-1]["strength_model"] == 0.8
    assert applied[-1]["strength_clip"] == 0.5
    assert applied[-1]["lora_metadata"] is not None  # forwarded when accepted
    assert "comfy.utils" not in imported

    # Same digest: cache hit, no second file read.
    LoadLora.execute(model=model, clip=clip, lora=ref_a, strength_model=0.2, strength_clip=0.2)
    assert len(file_loads) == 1
    # Different digest: cache miss, file read again.
    LoadLora.execute(model=model, clip=clip, lora=ref_b, strength_model=1.0, strength_clip=1.0)
    assert len(file_loads) == 2

    # v1 range contract survives as execute-side validation...
    with pytest.raises(ValueError, match="strength_model"):
        LoadLora.execute(
            model=model,
            clip=clip,
            lora=ref_a,
            strength_model=101.0,
            strength_clip=1.0,
        )
    with pytest.raises(ValueError, match="requires native model execution"):
        LoadLora.execute(
            model=model,
            clip=clip,
            lora=ref_a,
            strength_model=1.0,
            strength_clip=1.0,
            execution_mode="attach",
        )
    # ...and zero strengths short-circuit without touching the file.
    _clear_lora_cache()
    file_loads.clear()
    result = LoadLora.execute(
        model=model, clip=clip, lora=ref_a, strength_model=0.0, strength_clip=0.0
    )
    assert result["model"] is model and result["clip"] is clip
    assert file_loads == []
    _clear_lora_cache()


def test_native_lora_cache_holds_full_chains_without_thrash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream keeps one parsed lora per live node instance, so an
    N-lora chain never reparses on rerun. The digest cache is bounded
    by BYTES, not entry count: five distinct loras cache fully and a
    second pass with changed strengths performs zero new parses."""
    from dinkster_compat_comfy import LoadLora

    _clear_lora_cache()
    refs = [
        _vault_ref(tmp_path / f"l{i}", f"lora{i}.safetensors", b"lora %d" % i) for i in range(5)
    ]
    file_loads: list[str] = []
    native_checkpoint = types.ModuleType("dinkster_inference_torch.checkpoint")

    def load_checkpoint_with_metadata(path: Path):  # noqa: ANN202
        file_loads.append(str(path))
        return ({"sd": str(path)}, None)

    setattr(native_checkpoint, "load_checkpoint_with_metadata", load_checkpoint_with_metadata)  # noqa: B010
    comfy_sd = types.ModuleType("comfy.sd")
    setattr(  # noqa: B010
        comfy_sd,
        "load_lora_for_models",
        lambda model, clip, lora_sd, sm, sc: (model, clip),
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.checkpoint", native_checkpoint)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    model, clip = object(), object()
    for ref in refs:  # first pass: five parses
        LoadLora.execute(model=model, clip=clip, lora=ref, strength_model=1.0, strength_clip=1.0)
    assert len(file_loads) == 5
    for ref in refs:  # second pass, strengths changed: all hits, no parses
        LoadLora.execute(model=model, clip=clip, lora=ref, strength_model=0.5, strength_clip=0.5)
    assert len(file_loads) == 5
    _clear_lora_cache()


def test_native_checkpoint_loader_drops_path_reload_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1's load_checkpoint_guess_config installs path-retaining reload
    factories on the model, clip.patcher, and vae.patcher (comfy/sd.py @
    947c2749); the asset port drops all three so no mount path survives
    the invocation."""
    from dinkster_compat_comfy import LoadCheckpoint

    ref = _vault_ref(tmp_path, "sd15.safetensors", b"checkpoint bytes")
    comfy_sd = types.ModuleType("comfy.sd")

    class FakeModel:
        def __init__(self, path: str) -> None:
            self.cached_patcher_init = ("factory", (path,))

    class FakeWrapped:
        def __init__(self, path: str) -> None:
            self.patcher = types.SimpleNamespace(cached_patcher_init=("factory", (path,)))

    def load_checkpoint_guess_config(  # noqa: ANN202
        path: str,
        *,
        output_vae,
        output_clip,
        embedding_directory,  # noqa: ANN001
    ):
        return (FakeModel(path), FakeWrapped(path), FakeWrapped(path), None)

    setattr(  # noqa: B010
        comfy_sd, "load_checkpoint_guess_config", load_checkpoint_guess_config
    )
    folder_paths = types.ModuleType("folder_paths")
    setattr(  # noqa: B010
        folder_paths, "get_folder_paths", lambda name: [f"/models/{name}"]
    )
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    result = LoadCheckpoint.execute(checkpoint=ref)
    assert cast("FakeModel", result["model"]).cached_patcher_init is None
    assert cast("FakeWrapped", result["clip"]).patcher.cached_patcher_init is None
    assert cast("FakeWrapped", result["vae"]).patcher.cached_patcher_init is None


def test_native_lora_loader_omits_metadata_for_older_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installs whose load_lora_for_models predates the lora_metadata
    kwarg get the positional-only call instead of a TypeError."""
    from dinkster_compat_comfy import LoadLoraModelOnly

    _clear_lora_cache()
    ref = _vault_ref(tmp_path, "style.safetensors", b"lora bytes")
    native_checkpoint = types.ModuleType("dinkster_inference_torch.checkpoint")
    setattr(  # noqa: B010
        native_checkpoint,
        "load_checkpoint_with_metadata",
        lambda path: ({"sd": 1}, {"meta": 1}),
    )
    applied: list[tuple[object, ...]] = []
    comfy_sd = types.ModuleType("comfy.sd")

    def load_lora_for_models(model, clip, lora_sd, strength_model, strength_clip):  # noqa: ANN001, ANN202
        applied.append((model, clip, lora_sd, strength_model, strength_clip))
        return (("patched", model), None)

    setattr(comfy_sd, "load_lora_for_models", load_lora_for_models)  # noqa: B010
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.checkpoint", native_checkpoint)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    model = object()
    result = LoadLoraModelOnly.execute(model=model, lora=ref, strength_model=0.7)
    # Model-only semantics: clip rides through as None with zero strength.
    assert applied == [(model, None, {"sd": 1}, 0.7, 0.0)]
    assert result["model"] == ("patched", model)
    assert "clip" not in result
    _clear_lora_cache()


def test_native_vae_loader_verifies_and_refuses_path_reload_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The VAE port parses through the native checkpoint loader, runs
    v1's validity check, and deliberately does NOT install upstream's
    path-backed cached_patcher_init reload factory (a mount path retained
    beyond the invocation would bypass digest verification on reload)."""
    from dinkster_compat_comfy import LoadVae

    ref = _vault_ref(tmp_path, "fixed.vae.safetensors", b"vae bytes")
    native_checkpoint = types.ModuleType("dinkster_inference_torch.checkpoint")
    seen_paths: list[str] = []

    def load_checkpoint_with_metadata(path: Path):  # noqa: ANN202
        seen_paths.append(str(path))
        return ({"vae_sd": 1}, {"vae_meta": 1})

    setattr(native_checkpoint, "load_checkpoint_with_metadata", load_checkpoint_with_metadata)  # noqa: B010

    class FakePatcherSlot:
        pass

    class FakeVAE:
        def __init__(self, *, sd: object, metadata: object) -> None:
            self.sd = sd
            self.metadata = metadata
            self.patcher = FakePatcherSlot()
            self.validated = False

        def throw_exception_if_invalid(self) -> None:
            self.validated = True

    comfy_sd = types.ModuleType("comfy.sd")
    setattr(comfy_sd, "VAE", FakeVAE)  # noqa: B010
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.checkpoint", native_checkpoint)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    imported = _record_dynamic_imports(monkeypatch)

    result = LoadVae.execute(vae=ref)
    loaded = cast("FakeVAE", result["vae"])
    assert loaded.sd == {"vae_sd": 1}
    assert loaded.metadata == {"vae_meta": 1}
    assert loaded.validated
    assert len(seen_paths) == 1
    assert "comfy.utils" not in imported
    # The deliberate non-port: no path-based reload factory on the patcher.
    assert not hasattr(loaded.patcher, "cached_patcher_init")


def test_native_diffusion_model_loader_maps_weight_dtype_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference parity for UNETLoader's dtype vocabulary: exactly the
    three fp8 spellings set model_options; anything else (including
    unknown strings) means default."""
    from dinkster_compat_comfy import LoadDiffusionModel

    ref = _vault_ref(tmp_path, "unet.safetensors", b"unet bytes")
    fp8_e4m3fn, fp8_e5m2 = object(), object()
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "float8_e4m3fn", fp8_e4m3fn)  # noqa: B010
    setattr(fake_torch, "float8_e5m2", fp8_e5m2)  # noqa: B010
    loads: list[dict[str, Any]] = []
    comfy_sd = types.ModuleType("comfy.sd")

    class FakeModelPatcher:
        def __init__(self, path: str, options: dict[str, Any]) -> None:
            # v1's load_diffusion_model installs a path-retaining reload
            # factory exactly like this (comfy/sd.py @ 947c2749).
            self.cached_patcher_init = (load_diffusion_model, (path, options))

    def load_diffusion_model(path: str, *, model_options: dict[str, Any]):  # noqa: ANN202
        loads.append({"path": path, "options": model_options})
        return FakeModelPatcher(path, model_options)

    setattr(comfy_sd, "load_diffusion_model", load_diffusion_model)  # noqa: B010
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    for dtype, expected in [
        ("default", {}),
        ("fp8_e4m3fn", {"dtype": fp8_e4m3fn}),
        ("fp8_e4m3fn_fast", {"dtype": fp8_e4m3fn, "fp8_optimizations": True}),
        ("fp8_e5m2", {"dtype": fp8_e5m2}),
        ("something_new", {}),  # unknown means default, exactly as v1
    ]:
        result = LoadDiffusionModel.execute(diffusion_model=ref, weight_dtype=dtype)
        assert loads[-1]["options"] == expected
        # The path-retaining reload factory v1 installs is dropped.
        model = cast("FakeModelPatcher", result["model"])
        assert model.cached_patcher_init is None
    assert LoadDiffusionModel.WEIGHT_DTYPES[0] == "default"


def test_native_clip_loader_maps_type_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference parity for CLIPLoader: type strings resolve through
    CLIPType with v1's getattr fallback to STABLE_DIFFUSION, and
    device=cpu pins both load and offload devices."""
    from dinkster_compat_comfy import LoadClip

    ref = _vault_ref(tmp_path, "clip_l.safetensors", b"clip bytes")
    cpu_device = object()
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "device", lambda spec: cpu_device)  # noqa: B010
    clip_types = types.SimpleNamespace(STABLE_DIFFUSION=object(), WAN=object())
    loads: list[dict[str, Any]] = []
    comfy_sd = types.ModuleType("comfy.sd")
    setattr(comfy_sd, "CLIPType", clip_types)  # noqa: B010

    class FakeClipWithPatcher:
        def __init__(self, ckpt_paths: list[str]) -> None:
            # v1's load_clip installs a path-retaining reload factory on
            # clip.patcher exactly like this (comfy/sd.py @ 947c2749).
            self.patcher = types.SimpleNamespace(cached_patcher_init=(load_clip, (ckpt_paths,)))

    def load_clip(  # noqa: ANN202
        *,
        ckpt_paths,
        embedding_directory,
        clip_type,
        model_options,  # noqa: ANN001
    ):
        loads.append(
            {
                "ckpt_paths": ckpt_paths,
                "embedding_directory": embedding_directory,
                "clip_type": clip_type,
                "model_options": model_options,
            }
        )
        return FakeClipWithPatcher(ckpt_paths)

    setattr(comfy_sd, "load_clip", load_clip)  # noqa: B010
    folder_paths = types.ModuleType("folder_paths")
    setattr(  # noqa: B010
        folder_paths, "get_folder_paths", lambda name: [f"/models/{name}"]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    result = LoadClip.execute(text_encoder=ref, type="wan", device="default")
    assert loads[-1]["clip_type"] is clip_types.WAN
    assert loads[-1]["model_options"] == {}
    assert len(loads[-1]["ckpt_paths"]) == 1
    assert loads[-1]["embedding_directory"] == ["/models/embeddings"]
    # The path-retaining reload factory v1 installs is dropped.
    clip = cast("FakeClipWithPatcher", result["clip"])
    assert clip.patcher.cached_patcher_init is None

    # v1's getattr fallback: unrecognized types mean STABLE_DIFFUSION.
    LoadClip.execute(text_encoder=ref, type="not_a_family", device="default")
    assert loads[-1]["clip_type"] is clip_types.STABLE_DIFFUSION

    LoadClip.execute(text_encoder=ref, type="stable_diffusion", device="cpu")
    assert loads[-1]["model_options"] == {
        "load_device": cpu_device,
        "offload_device": cpu_device,
    }


def test_compat_ksampler_schema_is_generation_owner_schema() -> None:
    from dinkster_compat_comfy import KSampler
    from dinkster_nodes_generation.nodes import (
        SAMPLER_CHOICES,
        SCHEDULER_CHOICES,
    )
    from dinkster_nodes_generation.nodes import (
        KSampler as GenerationKSampler,
    )
    from dinkster_schema import (
        ComboOption,
        ComboWidget,
        NumberWidget,
        schema_signature,
        schema_to_wire,
    )
    from dinkster_values import CORE_COMBO, CORE_FLOAT, CORE_INT

    schema = KSampler.schema()
    assert schema == GenerationKSampler.schema()
    assert schema.node_type == "dinkster.ksampler"
    assert schema.aliases == ("KSampler",)
    assert schema.search_terms == ("sample", "denoise", "generate")
    assert [spec.id for spec in schema.inputs] == [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "conditioning_batching",
        "max_fused_lanes",
    ]
    by_id = {spec.id: spec for spec in schema.inputs}
    assert by_id["model"].type.types == ("dinkster.model",)
    assert by_id["positive"].type.types == ("dinkster.conditioning",)
    assert by_id["negative"].type.types == ("dinkster.conditioning",)
    assert by_id["latent_image"].type.types == ("dinkster.latent",)
    assert by_id["seed"].type.types == (CORE_INT,)
    assert by_id["cfg"].type.types == (CORE_FLOAT,)
    assert by_id["seed"].widget == NumberWidget(
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        step=1,
        control_after_generate="randomize",
    )
    assert by_id["steps"].widget == NumberWidget(min=1, max=10_000)
    assert by_id["cfg"].widget == NumberWidget(min=0.0, max=100.0, step=0.1)
    assert by_id["denoise"].widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    defaulted_ids = {
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "conditioning_batching",
        "max_fused_lanes",
    }
    assert {spec.id: spec.default for spec in schema.inputs if spec.id in defaulted_ids} == {
        "seed": 0,
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "denoise": 1.0,
        "conditioning_batching": "auto",
        "max_fused_lanes": 2,
    }
    for input_id, choices in (
        ("sampler_name", SAMPLER_CHOICES),
        ("scheduler", SCHEDULER_CHOICES),
    ):
        spec = by_id[input_id]
        assert spec.type.types == (CORE_COMBO,)
        widget = spec.widget
        assert isinstance(widget, ComboWidget)
        assert widget.options == choices
        assert widget.remote_route == ""
        assert not widget.refresh_button
        assert spec.default in {
            option.value if isinstance(option, ComboOption) else option for option in widget.options
        }
    assert [out.type.types for out in schema.outputs] == [("dinkster.latent",)]

    wire = schema_to_wire(schema)
    assert wire["schemaVersion"] == 40
    interface = {item["id"]: item for item in wire["interface"]}  # type: ignore[index]
    assert interface["seed"]["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "max": str(0xFFFFFFFFFFFFFFFF),
        "step": 1,
        "controlAfterGenerate": "randomize",
    }
    assert interface["steps"]["widget"] == {"type": "NUMBER", "min": 1, "max": 10_000}
    assert interface["cfg"]["widget"] == {
        "type": "NUMBER",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
    }
    assert interface["denoise"]["widget"] == {
        "type": "NUMBER",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    }

    numeric_ids = {"seed", "steps", "cfg", "denoise"}
    without_numeric_widgets = dataclasses.replace(
        schema,
        inputs=tuple(
            dataclasses.replace(spec, widget=None) if spec.id in numeric_ids else spec
            for spec in schema.inputs
        ),
    )
    assert schema_signature(schema) == schema_signature(without_numeric_widgets)


def test_ksampler_advanced_schema_and_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import KSamplerAdvanced
    from dinkster_nodes_generation.nodes import KSamplerAdvanced as GenerationKSamplerAdvanced
    from dinkster_schema import NumberWidget, schema_to_wire

    schema = KSamplerAdvanced.schema()
    assert schema.node_type == "dinkster.ksampler_advanced"
    assert schema.aliases == ("KSamplerAdvanced",)
    assert [spec.id for spec in schema.inputs] == [
        "model",
        "add_noise",
        "noise_seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise",
        "conditioning_batching",
        "max_fused_lanes",
    ]
    assert schema == GenerationKSamplerAdvanced.schema()
    noise_seed = next(spec for spec in schema.inputs if spec.id == "noise_seed")
    assert noise_seed.default == 0
    assert noise_seed.widget == NumberWidget(
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        step=1,
        control_after_generate="randomize",
    )
    wire_interface = cast("list[dict[str, Any]]", schema_to_wire(schema)["interface"])
    wire_seed = next(item for item in wire_interface if item["id"] == "noise_seed")
    assert wire_seed["default"] == 0
    assert wire_seed["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "max": str(0xFFFFFFFFFFFFFFFF),
        "step": 1,
        "controlAfterGenerate": "randomize",
    }

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_compat_sampler(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return {"samples": "continued"}

    monkeypatch.setattr(comfy_execution, "common_ksampler", fake_compat_sampler)
    result = KSamplerAdvanced.execute(
        model="model",
        add_noise="disable",
        noise_seed=42,
        steps=20,
        cfg=1.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.simple",
        positive="positive",
        negative="negative",
        latent_image={"samples": "stage-one"},
        start_at_step=8,
        end_at_step=10_000,
        return_with_leftover_noise="enable",
    )
    assert result == {"latent": {"samples": "continued"}}
    assert calls[0][0][4:6] == ("euler", "simple")
    assert calls[0][1] == {
        "denoise": 1.0,
        "disable_noise": True,
        "start_step": 8,
        "last_step": 10_000,
        "force_full_denoise": False,
    }

    KSamplerAdvanced.execute(
        model="model",
        add_noise="enable",
        noise_seed=43,
        steps=21,
        cfg=2.0,
        sampler_name="res4lyf.res_2m",
        scheduler="res4lyf.beta57",
        positive="positive",
        negative="negative",
        latent_image={"samples": "stage-two"},
        start_at_step=3,
        end_at_step=12,
        return_with_leftover_noise="disable",
    )
    assert calls[1][0][4:6] == ("res_2m", "beta57")

    advanced_kwargs = {
        "model": "model",
        "add_noise": "enable",
        "noise_seed": 44,
        "steps": 22,
        "cfg": 3.0,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "positive": "positive",
        "negative": "negative",
        "latent_image": {"samples": "stage-three"},
        "start_at_step": 2,
        "end_at_step": 11,
        "return_with_leftover_noise": "disable",
    }
    with pytest.raises(ValueError, match="conditioning_batching='auto'"):
        KSamplerAdvanced.execute(**advanced_kwargs, conditioning_batching="force-separate")
    with pytest.raises(ValueError, match="max_fused_lanes=2"):
        KSamplerAdvanced.execute(**advanced_kwargs, max_fused_lanes=3)
    assert len(calls) == 2


def test_translated_ksampler_preserves_empty_sigma_tensor(monkeypatch: pytest.MonkeyPatch) -> None:

    class Tensor:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values
            self.shape = (len(values),)

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: slice) -> Tensor:
            return Tensor(self.values[index])

        def to(self, **_kwargs: object) -> Tensor:
            return self

    captured: list[Tensor] = []

    class ComfyKSampler:
        def __init__(self, _model: object, **kwargs: object) -> None:
            self.sampler = kwargs["sampler"]
            self.sigmas = Tensor((1.0, 0.5, 0.0))

    latent = Tensor((0.0,))

    def capture_sample(*args: object, **_kwargs: object) -> Tensor:
        sigmas = args[7]
        assert isinstance(sigmas, Tensor)
        captured.append(sigmas)
        return latent

    modules = {
        "comfy.sample": types.SimpleNamespace(
            fix_empty_latent_channels=lambda _model, value, *_ratios: value,
            prepare_empty_noise=lambda value: value,
            prepare_noise=lambda value, _seed, _indices: value,
        ),
        "comfy.samplers": types.SimpleNamespace(
            KSampler=ComfyKSampler,
            sampler_object=lambda _name: types.SimpleNamespace(
                sampler_function=lambda *_args, **_kwargs: latent
            ),
            sample=capture_sample,
        ),
        "comfy.model_management": types.SimpleNamespace(
            intermediate_device=lambda: "cpu",
            intermediate_dtype=lambda: "float32",
        ),
        "comfy.utils": types.SimpleNamespace(PROGRESS_BAR_ENABLED=True),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, cast("Any", module))

    model = types.SimpleNamespace(load_device="cpu", model_options={})
    result = comfy_execution.common_ksampler(
        model,
        1,
        2,
        1.0,
        "euler",
        "simple",
        object(),
        object(),
        {"samples": latent},
        denoise=1.0,
        start_step=2,
    )

    assert result == {"samples": latent}
    assert len(captured) == 1
    assert isinstance(captured[0], Tensor)
    assert captured[0].shape == (0,)


def test_native_ksampler_advanced_builds_exact_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_compat_comfy.native_arm as native_arm
    from dinkster_inference import SamplingSegment

    calls: list[dict[str, object]] = []

    def fake_native_execute(cls: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"latent": "continued"}

    monkeypatch.setattr(
        native_arm.NativeKSampler,
        "execute",
        classmethod(fake_native_execute),
    )
    result = native_arm.NativeKSamplerAdvanced.execute(
        model="model",
        add_noise="disable",
        noise_seed=42,
        steps=20,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive="positive",
        negative="negative",
        latent_image={"samples": "stage-one"},
        start_at_step=8,
        end_at_step=10_000,
        return_with_leftover_noise="enable",
    )
    assert result == {"latent": "continued"}
    assert calls[0]["segment"] == SamplingSegment(20, 8, 20, False, True)
    assert calls[0]["seed"] == 42
    assert calls[0]["denoise"] == 1.0

    calls.clear()
    unchanged = {"samples": "stage-one", "downscale_ratio_spacial": 16}
    result = native_arm.NativeKSamplerAdvanced.execute(
        model="model",
        add_noise="enable",
        noise_seed=42,
        steps=20,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive="positive",
        negative="negative",
        latent_image=unchanged,
        start_at_step=20,
        end_at_step=20,
        return_with_leftover_noise="disable",
    )
    assert result == {"latent": {"samples": "stage-one"}}
    assert not calls


def test_compat_ksampler_delegates_to_callback_aware_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combo options are UI vocabulary, not execution identity: an
    off-list sampler_name goes through verbatim."""
    from dinkster_compat_comfy import KSampler

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_compat_sampler(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return {"samples": "denoised"}

    monkeypatch.setattr(comfy_execution, "common_ksampler", fake_compat_sampler)
    model, positive, negative = Opaque(), object(), object()
    latent = {"samples": "noisy"}
    result = KSampler.execute(
        model=model,
        seed=KSampler.MAX_SEED,
        steps=20,
        cfg=8.0,
        sampler_name="res_2s_rf",  # off-list: RES4LYF-style registered name
        scheduler="dinkster.karras",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=0.5,
    )
    assert result["latent"] == {"samples": "denoised"}
    (args, kwargs) = calls[0]
    assert args == (
        model, KSampler.MAX_SEED, 20, 8.0, "res_2s_rf", "karras",
        positive, negative, latent,
    )  # fmt: skip
    assert kwargs == {"denoise": 0.5}

    KSampler.execute(
        model=model,
        seed=41,
        steps=21,
        cfg=7.0,
        sampler_name="res4lyf.res_2m",
        scheduler="res4lyf.beta57",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=0.75,
    )
    assert calls[1][0][4:6] == ("res_2m", "beta57")

    standard_kwargs = {
        "model": model,
        "seed": 42,
        "steps": 22,
        "cfg": 6.0,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "positive": positive,
        "negative": negative,
        "latent_image": latent,
        "denoise": 1.0,
    }
    with pytest.raises(ValueError, match="conditioning_batching='auto'"):
        KSampler.execute(**standard_kwargs, conditioning_batching="max-fused-lanes")
    with pytest.raises(ValueError, match="max_fused_lanes=2"):
        KSampler.execute(**standard_kwargs, max_fused_lanes=1)
    # The v1 min/max contract survives the port as execute-side checks.
    with pytest.raises(ValueError, match="steps"):
        KSampler.execute(
            model=model,
            seed=0,
            steps=0,
            cfg=8.0,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )
    with pytest.raises(ValueError, match="denoise"):
        KSampler.execute(
            model=model,
            seed=0,
            steps=20,
            cfg=8.0,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.5,
        )
    assert len(calls) == 2  # validation failures never reach the sampler


def test_compat_ksampler_routes_native_runtime_handle_to_native_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import sampling
    from dinkster_native.native_arm import GenerationKSampler
    from dinkster_native.native_residency import NativeRuntimeHandle

    handle = object.__new__(NativeRuntimeHandle)
    calls: list[dict[str, object]] = []

    def native_execute(cls: type[object], **kwargs: object) -> Mapping[str, object]:
        del cls
        calls.append(kwargs)
        return {"latent": {"samples": "native"}}

    def refuse_compat(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("native handles must not reach compatibility sampling")

    monkeypatch.setattr(
        GenerationKSampler,
        "execute",
        classmethod(native_execute),
    )
    monkeypatch.setattr(comfy_execution, "common_ksampler", refuse_compat)

    result = sampling.KSampler.execute(
        model=handle,
        seed=91,
        steps=20,
        cfg=8.0,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image={"samples": "noise"},
        denoise=1.0,
    )

    assert result == {"latent": {"samples": "native"}}
    assert calls == [
        {
            "model": handle,
            "seed": 91,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "positive": "positive",
            "negative": "negative",
            "latent_image": {"samples": "noise"},
            "denoise": 1.0,
            "conditioning_batching": "auto",
            "max_fused_lanes": 2,
            "segment": None,
        }
    ]


def test_compat_ksampler_maps_owned_multistream_callback_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import (
        LatentStream,
        MultiStreamLatent,
        ProgressScope,
        SamplingCancelled,
    )
    from dinkster_workers import ExecutionContext
    from dinkster_workers.execution import use_execution_context

    class Tensor:
        def __init__(
            self,
            value: int,
            shape: tuple[int, ...] = (1, 2),
            parts: tuple[Tensor, ...] = (),
        ) -> None:
            self.value = value
            self.shape = shape
            self.parts = parts

        def detach(self) -> Tensor:
            return self

        def clone(self) -> Tensor:
            return Tensor(
                self.value,
                self.shape,
                tuple(part.clone() for part in self.parts),
            )

        def to(self, **_kwargs: object) -> Tensor:
            return self

    class NestedTensor:
        def __init__(self, values: tuple[Tensor, ...]) -> None:
            self.values = values

        def unbind(self) -> tuple[Tensor, ...]:
            return self.values

        def to(self, **_kwargs: object) -> NestedTensor:
            return self

    current = Tensor(0, (1, 1, 4), (Tensor(1), Tensor(2)))
    denoised = Tensor(0, (1, 1, 4), (Tensor(3), Tensor(4)))
    callback_step = 0
    callback_has_sigma = True
    emitter_states: list[object] = []
    sample_callbacks: list[object] = []
    wrapper_callbacks: list[tuple[int, object, object, int]] = []
    wrapper_fails = False
    states: list[object | None] = []

    class ComfyKSampler:
        def __init__(self, model: object, **kwargs: object) -> None:
            del model
            self.sampler = kwargs["sampler"]
            self.sigmas = (1.0, 0.5, 0.0)

    def sampler_function(
        _model: object,
        _noise: object,
        _sigmas: object,
        *,
        callback: Any,
        **_kwargs: object,
    ) -> Tensor:
        info = {"i": callback_step, "x": current, "denoised": denoised}
        if callback_has_sigma:
            info["sigma"] = 1.0
        callback(info)
        current.parts[0].value = 99
        denoised.parts[0].value = 98
        return current

    def sample(
        _model: object,
        noise: object,
        _positive: object,
        _negative: object,
        _cfg: float,
        _device: object,
        sampler: object,
        sigmas: object,
        _model_options: object,
        **kwargs: object,
    ) -> NestedTensor:
        sample_callbacks.append(kwargs["callback"])

        def augmented_callback(step: int, x0: object, x: object, total: int) -> None:
            wrapper_callbacks.append((step, x0, x, total))
            if wrapper_fails:
                raise RuntimeError("wrapper callback failed")

        def invocation_callback(info: Mapping[str, object]) -> None:
            augmented_callback(
                cast("int", info["i"]),
                info["denoised"],
                info["x"],
                len(cast("tuple[float, ...]", sigmas)) - 1,
            )

        packed = cast("Any", sampler).sampler_function(
            object(), noise, sigmas, extra_args={}, callback=invocation_callback, disable=True
        )
        return NestedTensor(packed.parts)

    shared_sampler = types.SimpleNamespace(sampler_function=sampler_function)
    modules = {
        "comfy.sample": types.SimpleNamespace(
            fix_empty_latent_channels=lambda _model, value, *_ratios: value,
            prepare_noise=lambda value, _seed, _indices: value,
        ),
        "comfy.samplers": types.SimpleNamespace(
            KSampler=ComfyKSampler,
            sampler_object=lambda _name: shared_sampler,
            sample=sample,
        ),
        "comfy.model_management": types.SimpleNamespace(
            intermediate_device=lambda: "cpu",
            intermediate_dtype=lambda: "float32",
        ),
        "comfy.utils": types.SimpleNamespace(
            PROGRESS_BAR_ENABLED=True,
            unpack_latents=lambda value, _shapes: value.parts,
        ),
        "comfy.nested_tensor": types.SimpleNamespace(NestedTensor=NestedTensor),
        "torch": types.SimpleNamespace(Tensor=Tensor),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, cast("Any", module))

    # The compat arm builds its own emitter instead of comfy's preview
    # callback; a fake one records the states it is fed.
    fake_emitter = types.SimpleNamespace(
        on_state=emitter_states.append,
        stage=lambda: nullcontext(),
    )
    monkeypatch.setattr(
        comfy_execution, "comfy_sampling_preview_emitter", lambda _model: fake_emitter
    )

    original_report = ProgressScope.report

    def capture_report(self: ProgressScope, event: object, state: object | None = None) -> None:
        states.append(state)
        original_report(self, cast("Any", event), cast("Any", state))

    monkeypatch.setattr(ProgressScope, "report", capture_report)
    latent = {
        "samples": MultiStreamLatent(
            (LatentStream("video", Tensor(10)), LatentStream("audio", Tensor(20)))
        )
    }
    model = types.SimpleNamespace(load_device="cpu", model_options={})

    result = comfy_execution.common_ksampler(
        model,
        7,
        1,
        1.0,
        "euler",
        "simple",
        object(),
        object(),
        latent,
        denoise=1.0,
    )

    state = cast("Any", states[0])
    assert state.current.roles == ("video", "audio")
    assert state.denoised.roles == ("video", "audio")
    assert state.current.by_role("video").value == 1
    assert state.denoised.by_role("video").value == 3
    assert state.current.by_role("video") is not current.parts[0]
    assert state.phase == "pre_update"
    # No comfy preview callback rides samplers.sample; the Dinkster emitter
    # receives the same state object the progress scope reports.
    assert sample_callbacks == [None]
    assert emitter_states == [state]
    assert wrapper_callbacks == [(0, denoised, current, 2)]
    assert shared_sampler.sampler_function is sampler_function
    assert cast("Any", result)["samples"].roles == ("video", "audio")

    callback_step = 1
    callback_has_sigma = False
    comfy_execution.common_ksampler(
        model,
        7,
        2,
        1.0,
        "uni_pc",
        "simple",
        object(),
        object(),
        latent,
        denoise=1.0,
    )
    assert cast("Any", states[-1]).phase == "post_update"
    assert cast("Any", states[-1]).sigma == 0.5
    assert len(wrapper_callbacks) == 2

    callback_step = 0
    callback_has_sigma = True
    comfy_execution.common_ksampler(
        model,
        7,
        2,
        1.0,
        "dpm_adaptive",
        "simple",
        object(),
        object(),
        latent,
        denoise=1.0,
    )
    assert cast("Any", states[-1]).phase == "post_update"
    assert len(wrapper_callbacks) == 3

    monkeypatch.setattr(ProgressScope, "report", original_report)
    with use_execution_context(ExecutionContext(None, None, cancelled=lambda: True)):
        with pytest.raises(SamplingCancelled, match="sampling cancelled"):
            comfy_execution.common_ksampler(
                model,
                7,
                1,
                1.0,
                "euler",
                "simple",
                object(),
                object(),
                latent,
                denoise=1.0,
            )
    # Cancellation raises before the state callback: neither the emitter
    # nor the comfy forward saw the cancelled step.
    assert len(emitter_states) == 3
    assert len(wrapper_callbacks) == 3

    wrapper_fails = True
    with pytest.raises(RuntimeError, match="wrapper callback failed"):
        comfy_execution.common_ksampler(
            model,
            7,
            1,
            1.0,
            "euler",
            "simple",
            object(),
            object(),
            latent,
            denoise=1.0,
        )
    # The emitter runs before the comfy forward, so it saw the state the
    # failing wrapper interrupted.
    assert len(emitter_states) == 4
    assert len(wrapper_callbacks) == 4
    assert all(entry is None for entry in sample_callbacks)


@pytest.mark.parametrize("shape", ((0,), (1, 0, 2)))
def test_torch_latent_decoder_constructs_zero_element_tensors(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    from dinkster_compat_comfy.latent import _decode_tensor
    from dinkster_values import EncodedLatentTensor

    calls: list[tuple[tuple[int, ...], object]] = []
    empty = object()
    torch = types.SimpleNamespace(
        float32=object(),
        empty=lambda tensor_shape, *, dtype: calls.append((tensor_shape, dtype)) or empty,
        frombuffer=lambda *_args, **_kwargs: pytest.fail("empty tensors must not use frombuffer"),
    )
    monkeypatch.setitem(sys.modules, "torch", cast("Any", torch))

    result = _decode_tensor(EncodedLatentTensor("float32", shape, b""))

    assert result is empty
    assert calls == [(shape, torch.float32)]


def test_compat_ksampler_materializes_native_scheduled_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_compat_comfy.native as native_module
    import dinkster_compat_comfy.sampling as sampling

    class HookKeyframe:
        def __init__(self, strength: float, start_percent: float, guarantee_steps: int) -> None:
            self.strength = strength
            self.start_percent = start_percent
            self.guarantee_steps = guarantee_steps

    class HookKeyframeGroup:
        def __init__(self) -> None:
            self.keyframes: list[HookKeyframe] = []

        def add(self, keyframe: HookKeyframe) -> None:
            self.keyframes.append(keyframe)

    class HookGroup:
        def __init__(self, hooks: list[dict[str, object]] | None = None) -> None:
            self.hooks = [] if hooks is None else hooks

        def clone_and_combine(self, other: HookGroup) -> HookGroup:
            return HookGroup(self.hooks + other.hooks)

        def set_keyframes_on_hooks(self, keyframes: HookKeyframeGroup) -> None:
            for hook in self.hooks:
                hook["keyframes"] = keyframes

    def create_hook_lora(*, lora: object, strength_model: float, strength_clip: float) -> HookGroup:
        return HookGroup(
            [
                {
                    "lora": lora,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                }
            ]
        )

    comfy_hooks = types.ModuleType("comfy.hooks")
    comfy_hooks.HookGroup = HookGroup  # type: ignore[attr-defined]
    comfy_hooks.HookKeyframe = HookKeyframe  # type: ignore[attr-defined]
    comfy_hooks.HookKeyframeGroup = HookKeyframeGroup  # type: ignore[attr-defined]
    comfy_hooks.create_hook_lora = create_hook_lora  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.hooks", comfy_hooks)
    monkeypatch.setattr(native_module, "_load_lora_file", lambda _ref: ("lora-state", {}))

    calls: list[tuple[Any, ...]] = []

    def fake_compat_sampler(*args: Any, **_kwargs: object) -> object:
        calls.append(args)
        return {"samples": "denoised"}

    monkeypatch.setattr(comfy_execution, "common_ksampler", fake_compat_sampler)

    ref = _vault_ref(tmp_path, "scheduled.safetensors", b"lora")
    keyframes = native_module.ScheduledHookKeyframes(((0.0, 0.25), (0.75, 1.0)))
    hooks = native_module.ScheduledHooks(
        (native_module.ScheduledLoraHook(ref, 0.8, 0.4, keyframes),)
    )
    conditioning = [[object(), {native_module.SCHEDULED_HOOKS_KEY: hooks}]]
    sampling.KSampler.execute(
        model=object(),
        seed=1,
        steps=2,
        cfg=3.0,
        sampler_name="euler",
        scheduler="simple",
        positive=conditioning,
        negative=conditioning,
        latent_image={"samples": object()},
        denoise=1.0,
    )

    lowered = cast("list[list[object]]", calls[0][6])
    metadata = cast("dict[str, object]", lowered[0][1])
    assert native_module.SCHEDULED_HOOKS_KEY not in metadata
    group = cast("HookGroup", metadata["hooks"])
    assert group.hooks[0]["lora"] == "lora-state"
    assert group.hooks[0]["strength_model"] == 0.8
    points = cast("HookKeyframeGroup", group.hooks[0]["keyframes"]).keyframes
    assert [(point.start_percent, point.strength) for point in points] == [
        (0.0, 0.25),
        (0.75, 1.0),
    ]
    assert [point.guarantee_steps for point in points] == [0, 0]
    with pytest.raises(ValueError, match="both native and compatibility hooks"):
        sampling._compat_conditioning_hooks(  # noqa: SLF001 - call-site contract
            [[object(), {native_module.SCHEDULED_HOOKS_KEY: hooks, "hooks": HookGroup()}]],
            "positive",
        )


def test_media_save_image_schema_is_save_target_widget_assets_out() -> None:
    from dinkster_compat_comfy.native import NATIVE_NODES
    from dinkster_nodes_media_io import SaveImage
    from dinkster_schema import SaveTargetWidget, TypeExpr

    assert SaveImage not in NATIVE_NODES
    schema = SaveImage.schema()
    assert schema.node_type == "dinkster.save_image"
    assert schema.aliases == ("SaveImage",)
    assert schema.output_node is True
    assert schema.idempotent is False
    assert [spec.id for spec in schema.inputs] == [
        "images",
        "target",
        "format",
        "quality",
        "compression",
        "lossless",
        "metadata_json",
    ]
    target = schema.inputs[1]
    assert target.type.types == ("dinkster.save_target",)
    assert target.required is False
    assert target.default == {"mount": "comfy-output", "prefix": "ComfyUI"}
    assert target.widget == SaveTargetWidget()
    images_out, assets_out = schema.outputs
    assert images_out.id == "images"
    assert assets_out.id == "assets"
    assert assets_out.type == TypeExpr.list_of(
        TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
    )


def test_native_comfy_equivalents_keep_upstream_names_and_aliases() -> None:
    from dinkster_compat_comfy.native import NATIVE_NODES

    schemas = {node.schema().node_type: node.schema() for node in NATIVE_NODES}
    assert {
        "comfy.ControlNetLoader",
        "comfy.ControlNetApply",
        "comfy.ControlNetApplyAdvanced",
    }.isdisjoint(schemas)
    expected = {
        "dinkster.load_latent": ("Load Latent", ("LoadLatent",)),
        "dinkster.apply_z_image_control_patch": (
            "Apply Z-Image Fun ControlNet",
            ("ZImageFunControlnet",),
        ),
        "dinkster.empty_minimax_h3_av": (
            "Empty MiniMax H3 AV Latent",
            ("EmptyMiniMaxH3LatentAV",),
        ),
        "dinkster.trim_video_latent": ("Trim Video Latent", ("TrimVideoLatent",)),
        "dinkster.wan21_image_to_video": ("WanImageToVideo", ("WanImageToVideo",)),
        "dinkster.wan22_fun_control_to_video": (
            "Wan22FunControlToVideo",
            ("Wan22FunControlToVideo",),
        ),
        "dinkster.wan22_image_to_video_latent": (
            "Wan22ImageToVideoLatent",
            ("Wan22ImageToVideoLatent",),
        ),
        "dinkster.wan_camera_embedding": ("WanCameraEmbedding", ("WanCameraEmbedding",)),
        "dinkster.wan_camera_image_to_video": (
            "WanCameraImageToVideo",
            ("WanCameraImageToVideo",),
        ),
        "dinkster.wan_move_concat_track": (
            "WanMoveConcatTrack",
            ("WanMoveConcatTrack",),
        ),
        "dinkster.wan_move_generate_tracks": (
            "Generate Video Tracks",
            ("GenerateTracks",),
        ),
        "dinkster.wan_move_track_to_video": (
            "WanMoveTrackToVideo",
            ("WanMoveTrackToVideo",),
        ),
        "dinkster.wan_move_tracks_from_coords": (
            "WanMoveTracksFromCoords",
            ("WanMoveTracksFromCoords",),
        ),
        "dinkster.wan_move_visualize_tracks": (
            "WanMoveVisualizeTracks",
            ("WanMoveVisualizeTracks",),
        ),
        "dinkster.wan_first_last_frame_to_video": (
            "WanFirstLastFrameToVideo",
            ("WanFirstLastFrameToVideo",),
        ),
        "dinkster.wan_fun_control_to_video": (
            "WanFunControlToVideo",
            ("WanFunControlToVideo",),
        ),
        "dinkster.wan_fun_inpaint_to_video": (
            "WanFunInpaintToVideo",
            ("WanFunInpaintToVideo",),
        ),
        "dinkster.wan_phantom_subject_to_video": (
            "WanPhantomSubjectToVideo",
            ("WanPhantomSubjectToVideo",),
        ),
        "dinkster.wan_track_to_video": ("WanTrackToVideo", ("WanTrackToVideo",)),
        "dinkster.wan_vace_to_video": ("WanVaceToVideo", ("WanVaceToVideo",)),
    }

    assert {
        node_type: (schemas[node_type].display_name, schemas[node_type].aliases)
        for node_type in expected
    } == expected


def test_native_trim_video_latent_copies_metadata_and_slices_only_samples() -> None:
    from dinkster_compat_comfy.native import TrimVideoLatent

    trimmed = object()

    class Samples:
        def __init__(self) -> None:
            self.keys: list[object] = []

        def __getitem__(self, key: object) -> object:
            self.keys.append(key)
            return trimmed

    tensor = Samples()
    noise_mask = object()
    custom = object()
    source = {"samples": tensor, "noise_mask": noise_mask, "custom": custom}

    result = TrimVideoLatent.execute(samples=source, trim_amount=2)["latent"]

    assert result is not source
    assert cast("dict[str, object]", result)["samples"] is trimmed
    assert cast("dict[str, object]", result)["noise_mask"] is noise_mask
    assert cast("dict[str, object]", result)["custom"] is custom
    assert source["samples"] is tensor
    assert tensor.keys == [(slice(None), slice(None), slice(2, None))]


def test_native_trim_video_latent_schema_matches_comfy_contract() -> None:
    from dinkster_compat_comfy.native import TrimVideoLatent
    from dinkster_schema import NumberWidget

    schema = TrimVideoLatent.schema()
    assert schema.node_type == "dinkster.trim_video_latent"
    assert schema.display_name == "Trim Video Latent"
    assert schema.category == "model/latent"
    assert schema.aliases == ("TrimVideoLatent",)
    assert [spec.id for spec in schema.inputs] == ["samples", "trim_amount"]
    assert schema.inputs[1].default == 0
    assert schema.inputs[1].widget == NumberWidget(min=0, max=99999)
    assert [spec.id for spec in schema.outputs] == ["latent"]


def test_native_node_types_covered_by_manifest_namespace_claims() -> None:
    """Every native node's dinkster.* id must fall under the compat
    manifest's declared namespace claims, or composition refuses the
    whole pack at serve time (compose.py enforces claims over every
    announced type). This is the check that would have caught the
    dinkster.* move shipping while the manifest still claimed only
    "comfy"."""
    from dinkster_compat_comfy.native import NATIVE_NODES
    from dinkster_schema import claim_covers
    from dinkster_workers import load_manifest

    manifest = load_manifest(
        Path(__file__).parents[1] / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"
    )
    for node in NATIVE_NODES:
        node_type = node.schema().node_type
        assert any(claim_covers(claim, node_type) for claim in manifest.namespaces), (
            f"{node_type} is outside the manifest claims {manifest.namespaces}"
        )


def test_merge_native_nodes_evicts_by_claimed_legacy_name() -> None:
    """Native and separately composed claims evict translated legacy nodes."""
    from dataclasses import replace

    from dinkster_compat_comfy import merge_native_nodes
    from dinkster_compat_comfy.native import NATIVE_NODES
    from dinkster_compat_comfy.native_arm import GenerationEmptyLatentImage
    from dinkster_compat_comfy.prompt import translate_prompt
    from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr

    string = TypeExpr.concrete("core.string")

    def stub(v1_name: str) -> type[Node]:
        class Stub(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    node_type=f"comfy.{v1_name}",
                    outputs=(OutputSpec("out", string),),
                    aliases=(v1_name,),
                    output_node=v1_name == "Other",
                )

            @classmethod
            def execute(cls) -> dict[str, object]:
                return {"out": v1_name}

        return Stub

    load_image, save_image = stub("LoadImage"), stub("SaveImage")
    resize_image_mask = stub("ResizeImageMaskNode")
    trim_video_latent = stub("TrimVideoLatent")
    load_checkpoint = stub("CheckpointLoaderSimple")
    empty_latent, encode = stub("EmptyLatentImage"), stub("CLIPTextEncode")
    ksampler = stub("KSampler")
    ksampler_advanced = stub("KSamplerAdvanced")
    ultimate_upscale = stub("UltimateSDUpscale")
    ultimate_no_upscale = stub("UltimateSDUpscaleNoUpscale")
    upscale_model_loader = stub("UpscaleModelLoader")
    other = stub("Other")
    loaders = [
        stub(name)
        for name in (
            "LoraLoader",
            "LoraLoaderModelOnly",
            "VAELoader",
            "CLIPLoader",
            "UNETLoader",
        )
    ]
    merged = merge_native_nodes(
        [
            load_image,
            save_image,
            resize_image_mask,
            trim_video_latent,
            load_checkpoint,
            empty_latent,
            encode,
            ksampler,
            ksampler_advanced,
            ultimate_upscale,
            ultimate_no_upscale,
            upscale_model_loader,
            other,
            *loaders,
        ]
    )
    assert load_image not in merged  # evicted: media pack claims "LoadImage"
    assert save_image not in merged  # evicted: media pack claims "SaveImage"
    assert resize_image_mask not in merged
    assert trim_video_latent not in merged  # evicted: native claims "TrimVideoLatent"
    assert load_checkpoint not in merged  # evicted: "CheckpointLoaderSimple"
    legacy_empty = next(n for n in merged if n.schema().node_type == "comfy.EmptyLatentImage")
    assert issubclass(legacy_empty, empty_latent)
    assert legacy_empty.schema() == replace(empty_latent.schema(), aliases=())
    assert empty_latent.schema().aliases == ("EmptyLatentImage",)
    assert legacy_empty.execute() == {"out": "EmptyLatentImage"}
    schemas = {n.schema().node_type: n.schema() for n in (*merged, GenerationEmptyLatentImage)}
    for source, target in (
        ("EmptyLatentImage", "dinkster.empty_latent_image"),
        ("comfy.EmptyLatentImage", "comfy.EmptyLatentImage"),
    ):
        graph = translate_prompt(
            {
                "e": {"class_type": source, "inputs": {}},
                "sink": {"class_type": "Other", "inputs": {}},
            },
            schemas,
        ).graph
        resolved = graph.nodes["e"]
        assert isinstance(resolved, GraphNode)
        assert resolved.node_type == target
    assert encode not in merged  # evicted: native claims "CLIPTextEncode"
    assert ksampler not in merged  # evicted: native claims "KSampler"
    assert ksampler_advanced not in merged  # evicted: native claims "KSamplerAdvanced"
    assert ultimate_upscale not in merged
    assert ultimate_no_upscale not in merged
    assert upscale_model_loader not in merged
    for loader in loaders:  # evicted: the asset-native model loaders
        assert loader not in merged
    assert other in merged  # unclaimed: survives
    native_ksamplers = [node for node in merged if node.schema().node_type == "dinkster.ksampler"]
    assert native_ksamplers == []
    native_advanced = [
        node for node in merged if node.schema().node_type == "dinkster.ksampler_advanced"
    ]
    assert native_advanced == []
    assert merged[-len(NATIVE_NODES) :] == NATIVE_NODES


def test_merge_native_nodes_evicts_foundation_twins() -> None:
    """Always-composed foundation nodes claim their legacy names, so the
    compat worker's translated twins are evicted the same way NATIVE_NODES
    claims are - otherwise
    aliases such as "PrimitiveInt" and "CreateList" would be ambiguous. The
    natives themselves are NOT added here (they live in dinkster-nodes-foundation,
    composed on the core surface, not in the compat worker). PreviewAny
    is deliberately unclaimed: the translated node runs beside torch and
    can stringify resident values the engine-process native cannot."""
    from dinkster_compat_comfy import merge_native_nodes
    from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr

    string = TypeExpr.concrete("core.string")

    def stub(v1_name: str) -> type[Node]:
        class Stub(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    node_type=f"comfy.{v1_name}",
                    outputs=(OutputSpec("out", string),),
                    aliases=(v1_name,),
                )

        return Stub

    foundation_twins = [
        stub(name)
        for name in (
            "PrimitiveInt",
            "PrimitiveFloat",
            "PrimitiveString",
            "PrimitiveStringMultiline",
            "PrimitiveBoolean",
            "CreateList",
        )
    ]
    preview_any = stub("PreviewAny")
    merged = merge_native_nodes([*foundation_twins, preview_any])
    for twin in foundation_twins:
        assert twin not in merged  # evicted: a foundation node claims the name
    assert preview_any in merged  # unclaimed: torch-side preview survives
    merged_types = {node.schema().node_type for node in merged}
    assert "dinkster.int" not in merged_types  # natives stay in their own pack


def test_native_save_image_requires_mounts_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_assets import AssetError
    from dinkster_compat_comfy.native import mount_writer

    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(AssetError, match="DINKSTER_MOUNTS_SNAPSHOT"):
        mount_writer()


def test_media_save_image_writes_png_batch_and_honors_revoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2-frame batch lands as counter-named PNGs inside the granted
    mount, returns AssetRefs whose digests match the bytes on disk, and a
    snapshot revoke fails the very next save - no process restart."""
    pil_image = pytest.importorskip("PIL.Image")
    from dinkster_assets import MOUNT_NAMESPACE, AssetError, digest_bytes
    from dinkster_nodes_media_io import SaveImage

    root = tmp_path / "outdir"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}]}),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    # Two solid-color frames: red and green, 2x3 so orientation survives.
    batch = np.zeros((2, 2, 3, 3), dtype=np.float32)
    batch[0, :, :, 0] = 1.0
    batch[1, :, :, 1] = 0.5
    result = SaveImage.execute(images=batch, target={"mount": "out", "prefix": "renders/scene"})
    refs = cast("Any", result["assets"])
    assert [ref.name for ref in refs] == ["scene_00001.png", "scene_00002.png"]
    for index, ref in enumerate(refs):
        landed = root / "renders" / ref.name
        payload = landed.read_bytes()
        assert ref.digest == digest_bytes(payload)
        assert ref.size == len(payload)
        assert ref.media_type == "image/png"
        assert ref.virtual_path == f"{MOUNT_NAMESPACE}/out/renders/{ref.name}"
        decoded = pil_image.open(landed).convert("RGB")
        assert decoded.size == (3, 2)  # PIL reports (width, height)
        expected = (255, 0, 0) if index == 0 else (0, 127, 0)
        assert decoded.getpixel((0, 0)) == expected
    # Revoke the grant by republishing the snapshot without the mount:
    # the next save through the same env refuses.
    snapshot.write_text(json.dumps({"mounts": []}), "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        SaveImage.execute(images=batch, target={"mount": "out", "prefix": "renders/scene"})


def test_register_native_types_fills_gaps_without_clobbering() -> None:
    from dinkster_assets import ASSET_TYPE
    from dinkster_compat_comfy import register_native_types

    # Empty registry: asset + resident model types all come from us.
    registry = TypeRegistry()
    register_native_types(registry)
    for type_id in (
        ASSET_TYPE,
        "comfy.MODEL",
        "comfy.CLIP",
        "comfy.VAE",
        "dinkster.sampler",
        "dinkster.sigmas",
        "dinkster.guider",
        "dinkster.noise",
        "dinkster.image",
        "dinkster.mask",
        "dinkster.model3d",
    ):
        assert registry.spec(type_id).declared_codec is True
    for type_id in ("comfy.LATENT", "comfy.CONDITIONING", "comfy.TRACKS"):
        assert type_id in registry  # data types: default codec

    # After a translation that already registered the model types
    # (resident), a second pass must not raise on duplicates.
    translation = translate_mappings({"Loader": V1Loader})
    registry = TypeRegistry()
    translation.register_types(registry)
    register_native_types(registry)
    assert ASSET_TYPE in registry


def test_native_sampler_settings_round_trip_every_builtin_without_worker_ownership() -> None:
    from dinkster_compat_comfy import register_native_types
    from dinkster_compat_comfy.native_arm import GenerationKSamplerSelect
    from dinkster_inference import builtin_samplers, register_inference_types
    from dinkster_inference.sampling_wire import SamplerSelection

    worker, host = TypeRegistry(), TypeRegistry()
    register_native_types(worker)
    register_inference_types(host)
    worker_spec = worker.spec("dinkster.sampler")
    host_spec = host.spec("dinkster.sampler")
    for descriptor in builtin_samplers():
        selected = GenerationKSamplerSelect.execute(sampler_name=descriptor.id)["sampler"]
        encoded = worker_spec.encode(selected)
        decoded = host_spec.decode(encoded)
        assert type(decoded) is SamplerSelection
        assert decoded.sampler_id == descriptor.id
        assert host_spec.encode(decoded) == encoded
        assert worker_spec.decode(encoded) == decoded
        assert worker_spec.meta is None
        assert host_spec.meta is None


def test_compat_inference_residents_use_governed_pool_and_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import CompatTranslation, bootstrap
    from dinkster_compat_comfy.pool import default_pool

    monkeypatch.setattr(
        bootstrap,
        "load_comfyui_nodes",
        lambda *, required=(): CompatTranslation(),
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    entry = importlib.import_module("dinkster_compat_comfy.entry")
    try:
        registry = TypeRegistry()
        entry.register_types(registry)
        resident = FakePatcher("cuda:1", 42)

        for type_id in ("dinkster.model", "dinkster.clip", "dinkster.vae"):
            spec = registry.spec(type_id)
            encoded = spec.encode(resident)
            rid = cast("str", json.loads(encoded)["residentId"])
            assert default_pool().get(rid) is resident
            value = registry.wrap(type_id, resident)
            assert value.meta.get("resources") == {"gpu": "cuda:1"}
            assert value.meta.get("cost") == {"vram:cuda:1": 42, "ram": 42}
    finally:
        sys.modules.pop("dinkster_compat_comfy.entry", None)


# --- across a real process boundary --------------------------------------


@pytest.mark.parametrize("use_shm", [False, True])
def test_flux_sampling_patch_survives_worker_roundtrip_and_replay(
    tmp_path: Path, use_shm: bool
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "flux-patch-test"\n\n[pack.entry]\n'
            'nodes = "flux_sampling_pack_nodes:NODES"\n'
            'types = "flux_sampling_pack_nodes:register_types"\n'
        )
        graph = Graph(
            nodes={
                "load": GraphNode("test.flux_model", {}),
                "patch": GraphNode(
                    "dinkster.model_sampling_flux",
                    {
                        "model": Link("load", "model"),
                        "max_shift": 1.15,
                        "base_shift": 0.5,
                        "width": 768,
                        "height": 1024,
                    },
                ),
                "read": GraphNode("test.flux_shift", {"model": Link("patch", "model")}),
            }
        )
        wire = graph_to_wire(graph)
        fingerprints = []
        for _ in range(2):
            graph = graph_from_wire(wire)
            worker = IsolatedWorker(
                manifest,
                registry,
                extra_env={"PYTHONPATH": str(TESTS_DIR)},
                use_shm=use_shm,
                shm_threshold=1,
                transport="tcp",
            )
            await worker.start()
            try:
                engine = Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                )
                first = await engine.run(graph, ["patch", "read"])
                value = first.outputs["patch"]["model"]
                assert value.fingerprint.startswith("resident:")
                fingerprints.append(value.fingerprint)
                assert first.outputs["read"]["shift"].resolve() == 0.9766666666666666
                second = await engine.run(graph, ["patch", "read"])
                assert second.outputs["patch"]["model"].fingerprint == value.fingerprint
                patch = cast(GraphNode, graph.nodes["patch"])
                changed_graph = Graph(
                    nodes={
                        **graph.nodes,
                        "patch": dataclasses.replace(patch, inputs={**patch.inputs, "width": 1024}),
                    }
                )
                changed = await engine.run(changed_graph, ["patch", "read"])
                assert changed.outputs["read"]["shift"].resolve() == 1.15
                assert changed.outputs["patch"]["model"].fingerprint != value.fingerprint
            finally:
                await worker.close()
        assert fingerprints[0] != fingerprints[1]

    asyncio.run(scenario())


def test_resident_value_stays_in_worker_and_round_trips(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)  # res.heavy deliberately not registered
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "respack"\n\n[pack.entry]\n'
            'nodes = "respack_nodes:NODES"\ntypes = "respack_nodes:register_types"\n'
        )
        worker = IsolatedWorker(manifest, registry, extra_env={"PYTHONPATH": str(TESTS_DIR)})
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
                    "load": GraphNode("res.load", {"token": "s3cret"}),
                    "use": GraphNode(
                        "res.use",
                        {"heavy": Link("load", "heavy"), "oid": Link("load", "oid")},
                    ),
                }
            )
            result = await engine.run(graph, ["load", "use"])

            # The parent holds an interrogable stub, not the object: the
            # payload is a tiny JSON resident id (a HeavyThing contains a
            # threading.Lock, so pickling it would have raised in the
            # worker), and it cannot be resolved here.
            heavy = result.outputs["load"]["heavy"]
            assert heavy.type_id == "res.heavy"
            assert heavy.fingerprint.startswith("resident:")
            data = heavy.payload.data  # type: ignore[attr-defined]
            assert set(json.loads(data)) == {"residentId"}
            with pytest.raises(UnresolvablePayload, match="not registered"):
                heavy.resolve()

            # Back in the owning worker, the stub resolved to the exact
            # object the loader created.
            assert result.outputs["use"]["same"].resolve() is True
            assert result.outputs["use"]["token"].resolve() == "s3cret"
        finally:
            await worker.close()

    asyncio.run(scenario())
