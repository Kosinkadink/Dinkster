from __future__ import annotations

import asyncio
import hashlib
import tomllib
from pathlib import Path

from dinkster_assets import RemoteSource
from dinkster_nodes_image import (
    DETECT_PROVIDER_CHOICE,
    IMAGE_NODES,
    SEGMENT_PROVIDER_CHOICE,
    TEXT_SEGMENT_PROVIDER_CHOICE,
    TRACK_PROVIDER_CHOICE,
    vision_choices,
)
from dinkster_nodes_image import DetectObjects as OwnerDetectObjects
from dinkster_nodes_image import SegmentByText as OwnerSegmentByText
from dinkster_nodes_image import SegmentDetections as OwnerSegmentDetections
from dinkster_nodes_image import TrackObjects as OwnerTrackObjects
from dinkster_nodes_vision.sam31 import DetectObjects as SAM31DetectObjects
from dinkster_nodes_vision.sam31 import SegmentByText as SAM31SegmentByText
from dinkster_nodes_vision.sam31 import SegmentDetections as SAM31SegmentDetections
from dinkster_nodes_vision.sam31 import TrackObjects as SAM31TrackObjects
from dinkster_schema import ComboWidget, schema_signature, schema_to_wire
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
SIDECAR = PACKAGE / "dinkster_vision_sam31_pack"
MANIFEST = SIDECAR / "dinkster-pack.toml"
LICENSE = SIDECAR / "SAM_LICENSE"
CLIP_LICENSE = SIDECAR / "CLIP_LICENSE"
TOKENIZER = PACKAGE / "src/dinkster_nodes_vision/sam31/data/bpe_simple_vocab_16e6.txt.gz"
LICENSE_SHA256 = "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be"
SOURCE_LICENSE_SHA256 = "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448"
CLIP_LICENSE_SHA256 = "893951b3bf94db8df1b13e05da5cdeb499400960e4d44a3962a8b33ed0b4f28e"
SOURCE_CLIP_LICENSE_SHA256 = "987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db"
TOKENIZER_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
MODEL_DIGEST = "blake3:1c8d5762dbaf238bc9a2f10de07e0c476d1b68e75453566feb8dc9bcf2cb41c5"


def test_detect_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerDetectObjects.schema()
    provider_schema = SAM31DetectObjects.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{DETECT_PROVIDER_CHOICE}"
    assert vision_choices()[DETECT_PROVIDER_CHOICE] == ()
    assert OwnerDetectObjects in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_segment_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerSegmentDetections.schema()
    provider_schema = SAM31SegmentDetections.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{SEGMENT_PROVIDER_CHOICE}"
    assert vision_choices()[SEGMENT_PROVIDER_CHOICE] == ()
    assert OwnerSegmentDetections in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_text_segment_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerSegmentByText.schema()
    provider_schema = SAM31SegmentByText.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{TEXT_SEGMENT_PROVIDER_CHOICE}"
    assert vision_choices()[TEXT_SEGMENT_PROVIDER_CHOICE] == ()
    assert OwnerSegmentByText in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_track_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerTrackObjects.schema()
    provider_schema = SAM31TrackObjects.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{TRACK_PROVIDER_CHOICE}"
    assert vision_choices()[TRACK_PROVIDER_CHOICE] == ()
    assert OwnerTrackObjects in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_sam31_pack_declares_cpu_providers_and_pinned_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-sam31"
    assert manifest.namespaces == ()
    assert manifest.executes == (
        "dinkster.detection.detect",
        "dinkster.detection.segment",
        "dinkster.detection.segment_text",
        "dinkster.detection.track",
    )
    assert manifest.sandbox.gpu is False
    assert manifest.sandbox.network is False
    assert manifest.sandbox.writable_mounts is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "opencv-python-headless==5.0.0.93",
        "safetensors==0.8.0",
        "torch==2.13.0",
    )
    assert [provider.choice for provider in manifest.vision_providers] == [
        DETECT_PROVIDER_CHOICE,
        SEGMENT_PROVIDER_CHOICE,
        TEXT_SEGMENT_PROVIDER_CHOICE,
        TRACK_PROVIDER_CHOICE,
    ]
    assert [provider.node for provider in manifest.vision_providers] == [
        "dinkster.detection.detect",
        "dinkster.detection.segment",
        "dinkster.detection.segment_text",
        "dinkster.detection.track",
    ]
    assert [provider.batching for provider in manifest.vision_providers] == [
        "per-image",
        "per-image",
        "per-image",
        "batch",
    ]
    assert [provider.model for provider in manifest.vision_providers] == [
        "sam-3.1",
        "sam-3.1",
        None,
        None,
    ]
    for provider in manifest.vision_providers:
        assert provider.devices == ("cpu",)
        assert provider.dtypes == ("float32",)
        assert provider.artifacts == ("sam31-multiplex-fp16",)
    assert len(manifest.assets) == 1
    asset = manifest.assets[0]
    assert asset.need.digest == MODEL_DIGEST
    assert asset.need.size == 1_745_546_848
    assert "SAM License" in asset.need.name
    source = asset.need.sources[0]
    assert isinstance(source, RemoteSource)
    assert source.url == (
        "https://huggingface.co/Comfy-Org/sam3.1/resolve/"
        "f38cd62b71494b53ac2b56ca36e24f3c8d565581/checkpoints/"
        "sam3.1_multiplex_fp16.safetensors"
    )


def test_sam31_wheel_contains_pack_runtime_manifest_and_license() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    assert "dinkster_vision_sam31_pack/SAM_LICENSE" in project["project"]["license-files"]
    assert "dinkster_vision_sam31_pack/CLIP_LICENSE" in project["project"]["license-files"]
    license_bytes = LICENSE.read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == LICENSE_SHA256
    assert hashlib.sha256(license_bytes.removesuffix(b"\n")).hexdigest() == SOURCE_LICENSE_SHA256
    clip_license = CLIP_LICENSE.read_bytes()
    assert hashlib.sha256(clip_license).hexdigest() == CLIP_LICENSE_SHA256
    assert hashlib.sha256(clip_license + b"\n").hexdigest() == SOURCE_CLIP_LICENSE_SHA256
    assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == TOKENIZER_SHA256
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_sam31_pack"] == "dinkster_vision_sam31_pack"
    assert (
        included["src/dinkster_nodes_vision/sam31"]
        == "dinkster_vision_sam31_pack/src/dinkster_nodes_vision/sam31"
    )


def test_sam31_provider_populates_owner_choice_and_scopes_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[DETECT_PROVIDER_CHOICE] == (
                "dinkster-vision-detr",
                "dinkster-vision-rtdetr",
                "dinkster-vision-sam31",
            )
            assert composition.choices[SEGMENT_PROVIDER_CHOICE] == (
                "dinkster-vision-efficient-sam",
                "dinkster-vision-sam31",
            )
            assert composition.choices[TEXT_SEGMENT_PROVIDER_CHOICE] == ("dinkster-vision-sam31",)
            assert composition.choices[TRACK_PROVIDER_CHOICE] == ("dinkster-vision-sam31",)
            assert tuple(composition.asset_catalog.needs_for_nodes(())) == ()
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.segment", "dinkster-vision-sam31"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
            detection_needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.detect", "dinkster-vision-sam31"),
                },
            )
            assert tuple(detection_needs) == (MODEL_DIGEST,)
            tracking_needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.track", "dinkster-vision-sam31"),
                },
            )
            assert tuple(tracking_needs) == (MODEL_DIGEST,)
            text_segment_needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.segment_text", "dinkster-vision-sam31"),
                },
            )
            assert tuple(text_segment_needs) == (MODEL_DIGEST,)
            combined_needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.detect", "dinkster-vision-sam31"),
                    ("dinkster.detection.segment", "dinkster-vision-sam31"),
                    ("dinkster.detection.segment_text", "dinkster-vision-sam31"),
                    ("dinkster.detection.track", "dinkster-vision-sam31"),
                },
            )
            assert tuple(combined_needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())
