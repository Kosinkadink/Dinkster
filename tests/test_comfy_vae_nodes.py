"""VAE interfaces can be inspected without importing Torch."""

from dinkster_nodes_comfy.vae import VAE_NODES


def test_vae_core_interfaces() -> None:
    assert [node.schema().node_type for node in VAE_NODES] == [
        "VAEDecode",
        "VAEEncode",
        "VAEDecodeTiled",
        "VAEEncodeTiled",
    ]
    for node in VAE_NODES:
        schema = node.schema()
        encode = "Encode" in schema.node_type
        tiled = schema.node_type.endswith("Tiled")
        assert [field.id for field in schema.inputs] == [
            "pixels" if encode else "samples",
            "vae",
            *(["tile_size", "overlap", "temporal_size", "temporal_overlap"] if tiled else []),
        ]
        assert [field.id for field in schema.outputs] == ["LATENT" if encode else "IMAGE"]
        assert schema.category == "model/latent"
