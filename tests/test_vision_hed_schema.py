from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_nodes_image import (
    IMAGE_NODES,
    MODEL_EDGE_PROVIDER_CHOICE,
    MODEL_EDGE_PROVIDER_CHOICES,
    preprocessor_choices,
)
from dinkster_nodes_image import (
    ModelEdgePreprocessor as OwnerModelEdgePreprocessor,
)
from dinkster_nodes_vision.hed import (
    AnimeLineartPreprocessor,
    AnyLinePreprocessor,
    MangaLineartPreprocessor,
    MLSDPreprocessor,
    RealisticLineartPreprocessor,
    TEEDPreprocessor,
)
from dinkster_nodes_vision.hed import (
    ModelEdgePreprocessor as HedModelEdgePreprocessor,
)
from dinkster_schema import ComboWidget, schema_signature
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_hed_pack" / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:36ea9a81b5e5f69c9f98b81eacce0c70b7bb444af4d821201b8a910e05792da9"


def test_model_edge_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerModelEdgePreprocessor.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{MODEL_EDGE_PROVIDER_CHOICE}"
    assert preprocessor_choices()[MODEL_EDGE_PROVIDER_CHOICE] == ()
    assert OwnerModelEdgePreprocessor in IMAGE_NODES
    assert schema_signature(HedModelEdgePreprocessor.schema()) == schema_signature(schema)


def test_hed_pack_declares_isolated_cpu_cuda_providers_and_pinned_models() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-hed"
    assert manifest.namespaces == ()
    assert manifest.executes == (
        "dinkster.preprocess.model_edges",
        "dinkster.preprocess.lineart_realistic",
        "dinkster.preprocess.lineart_anime",
        "dinkster.preprocess.lineart_manga",
        "dinkster.preprocess.anyline",
        "dinkster.preprocess.teed",
        "dinkster.preprocess.mlsd",
    )
    assert manifest.sandbox.gpu is True
    assert manifest.requires == (
        "numpy==2.5.1",
        "opencv-python-headless==5.0.0.93",
        "torch==2.13.0",
    )
    assert len(manifest.vision_providers) == 7
    assert {provider.node for provider in manifest.vision_providers} == set(manifest.executes)
    schemas = {node.schema().node_type: node.schema() for node in IMAGE_NODES}
    for provider in manifest.vision_providers:
        provider_input = schemas[provider.node].input("provider")
        assert provider_input is not None
        assert isinstance(provider_input.widget, ComboWidget)
        assert provider_input.widget.remote_route == f"/api/choices/{provider.choice}"
        assert provider.devices == ("cpu", "cuda")
        assert provider.dtypes == ("float32",)
        assert provider.batching == "per-image"
    assets = {asset.id: asset.need for asset in manifest.assets}
    assert assets["hed-model"].digest == MODEL_DIGEST
    assert assets["hed-model"].size == 29_444_406
    assert len(assets) == 8
    assert manifest.comfy_aliases is None


def test_hed_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_hed_pack"] == "dinkster_vision_hed_pack"
    assert (
        included["src/dinkster_nodes_vision/hed"]
        == "dinkster_vision_hed_pack/dinkster_nodes_vision/hed"
    )


def test_hed_provider_populates_owner_choice_and_scopes_its_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            for choice in MODEL_EDGE_PROVIDER_CHOICES:
                assert composition.choices[choice] == ("dinkster-vision-hed",)
            expected = {
                "dinkster.preprocess.model_edges": {MODEL_DIGEST},
                "dinkster.preprocess.lineart_realistic": {
                    "blake3:9849b9af40a35a8af12709de87d9d82abe8443b30b8ff37b86db47f912b7f933",
                    "blake3:94cca6f565259a25238844327987e1eca0a98194eab82bd4af56932d9662cd89",
                },
                "dinkster.preprocess.lineart_anime": {
                    "blake3:147873fae2d83761e678eea074fdf8f2e8e0e99a2745f26b2e4711880f91210a"
                },
                "dinkster.preprocess.lineart_manga": {
                    "blake3:938c4173935eafac5a9d88f53ed7ca0041e278c6094da2b26098844107e8855e"
                },
                "dinkster.preprocess.anyline": {
                    "blake3:bbcff8e81d853e788b06190e371f14207662b5d7a524c0eeb43da9d45060b3db",
                    "blake3:9849b9af40a35a8af12709de87d9d82abe8443b30b8ff37b86db47f912b7f933",
                    "blake3:147873fae2d83761e678eea074fdf8f2e8e0e99a2745f26b2e4711880f91210a",
                    "blake3:938c4173935eafac5a9d88f53ed7ca0041e278c6094da2b26098844107e8855e",
                },
                "dinkster.preprocess.teed": {
                    "blake3:3f9dae7af1da3156f2fe3f72e0d74b170c7bb7a76a32105c110365dc9e53bd00"
                },
                "dinkster.preprocess.mlsd": {
                    "blake3:b770f3458f83a5be2065d89703fc53db8c3cf4c60fd6e5a49032ab28c9a644e9"
                },
            }
            for node, digests in expected.items():
                needs = composition.asset_catalog.needs_for_nodes(
                    (), provider_selections={(node, "dinkster-vision-hed")}
                )
                assert set(needs) == digests
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_line_edge_provider_schemas_match_their_owners() -> None:
    from dinkster_nodes_image import (
        AnimeLineartPreprocessor as OwnerAnime,
    )
    from dinkster_nodes_image import AnyLinePreprocessor as OwnerAnyLine
    from dinkster_nodes_image import MangaLineartPreprocessor as OwnerManga
    from dinkster_nodes_image import MLSDPreprocessor as OwnerMLSD
    from dinkster_nodes_image import RealisticLineartPreprocessor as OwnerRealistic
    from dinkster_nodes_image import TEEDPreprocessor as OwnerTEED

    for owner, provider in (
        (OwnerRealistic, RealisticLineartPreprocessor),
        (OwnerAnime, AnimeLineartPreprocessor),
        (OwnerManga, MangaLineartPreprocessor),
        (OwnerAnyLine, AnyLinePreprocessor),
        (OwnerTEED, TEEDPreprocessor),
        (OwnerMLSD, MLSDPreprocessor),
    ):
        assert owner in IMAGE_NODES
        assert schema_signature(provider.schema()) == schema_signature(owner.schema())
