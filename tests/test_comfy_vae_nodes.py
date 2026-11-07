"""VAE interfaces can be inspected without importing Torch."""

from dinkster_nodes_latent import NODES
from dinkster_nodes_latent.vae import VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled


def test_vae_core_interfaces() -> None:
    nodes = (VAEDecode, VAEEncode, VAEDecodeTiled, VAEEncodeTiled)
    assert [node.schema().node_type for node in nodes] == [
        "dinkster.latent.vae_decode",
        "dinkster.latent.vae_encode",
        "dinkster.latent.vae_decode_tiled",
        "dinkster.latent.vae_encode_tiled",
    ]
    for node in nodes:
        assert node in NODES
        schema = node.schema()
        encode = "_encode" in schema.node_type
        tiled = schema.node_type.endswith("_tiled")
        assert [field.id for field in schema.inputs] == [
            "pixels" if encode else "samples",
            "vae",
            *(["tile_size", "overlap", "temporal_size", "temporal_overlap"] if tiled else []),
        ]
        assert [field.id for field in schema.outputs] == ["LATENT" if encode else "IMAGE"]
        assert schema.category == "model/latent"
