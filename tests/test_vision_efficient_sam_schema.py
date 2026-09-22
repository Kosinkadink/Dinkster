from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_assets import RemoteSource
from dinkster_nodes_image import (
    IMAGE_NODES,
    SEGMENT_PROVIDER_CHOICE,
    vision_choices,
)
from dinkster_nodes_image import (
    SegmentDetections as OwnerSegmentDetections,
)
from dinkster_nodes_vision.efficient_sam import SegmentDetections as EfficientSamSegmentDetections
from dinkster_schema import ComboWidget, schema_signature, schema_to_wire
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_efficient_sam_pack" / "dinkster-pack.toml"
ENCODER_DIGEST = "blake3:106600f3dd645019eff0d6fabe46cc1359654b03bfaec45316dabb1121cc9e6e"
DECODER_DIGEST = "blake3:41cdd8a75918dbef651ba10a244d22084999fc6162a265b9d504faacaef14004"


def test_segment_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerSegmentDetections.schema()
    provider_schema = EfficientSamSegmentDetections.schema()
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


def test_efficient_sam_pack_declares_cpu_provider_and_pinned_models() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-efficient-sam"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.detection.segment",)
    assert manifest.sandbox.gpu is False
    assert manifest.sandbox.network is False
    assert manifest.sandbox.writable_mounts is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "onnxruntime==1.29.0",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == SEGMENT_PROVIDER_CHOICE
    assert provider.node == "dinkster.detection.segment"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "per-image"
    assert provider.artifacts == (
        "efficient-sam-vitt-encoder",
        "efficient-sam-vitt-decoder",
    )
    assert provider.model == "efficient-sam-ti"
    assert [asset.need.digest for asset in manifest.assets] == [ENCODER_DIGEST, DECODER_DIGEST]
    assert [asset.need.size for asset in manifest.assets] == [24_799_761, 16_565_728]
    sources = [asset.need.sources[0] for asset in manifest.assets]
    assert all(isinstance(source, RemoteSource) for source in sources)
    assert [source.url for source in sources if isinstance(source, RemoteSource)] == [
        "https://raw.githubusercontent.com/yformer/EfficientSAM/"
        "d525f622e6f640acf5a0fc37c7ca1f243da5bde0/weights/efficient_sam_vitt_encoder.onnx",
        "https://raw.githubusercontent.com/yformer/EfficientSAM/"
        "d525f622e6f640acf5a0fc37c7ca1f243da5bde0/weights/efficient_sam_vitt_decoder.onnx",
    ]


def test_efficient_sam_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_efficient_sam_pack"] == "dinkster_vision_efficient_sam_pack"
    assert (
        included["src/dinkster_nodes_vision/efficient_sam"]
        == "dinkster_vision_efficient_sam_pack/src/dinkster_nodes_vision/efficient_sam"
    )


def test_efficient_sam_provider_populates_owner_choice_and_scopes_models() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[SEGMENT_PROVIDER_CHOICE] == (
                "dinkster-vision-efficient-sam",
                "dinkster-vision-sam31",
            )
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.segment", "dinkster-vision-efficient-sam"),
                },
            )
            assert tuple(needs) == (ENCODER_DIGEST, DECODER_DIGEST)
        finally:
            await composition.close()

    asyncio.run(scenario())
