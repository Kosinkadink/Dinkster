# Previous-main asset catalog fixture

Source: [Dinkster b5e4bc962635547f2cef0cb4a39b641e5428e943](https://github.com/Kosinkadink/Dinkster/tree/b5e4bc962635547f2cef0cb4a39b641e5428e943), generated with its public `LibraryStore` APIs.

Reproduction inputs:

- `alpha_png_bytes`: `Image.new("RGBA", (3, 2), (22, 111, 203, 255)).save(output, format="PNG")`; `beta_png_bytes`: the same call with `(196, 67, 98, 255)`.
- `grant_media(MediaGrant(scope="studio-alpha", digest="blake3:cb22aacddee1bca16e287d00ed3c9b130be15bcb44051c31423b9c17e12063f4", kind="media/image", media_type="image/png", extension="png", byte_size=76), alpha_png_bytes)`
- `create("studio-alpha", "alpha-reference.png", alpha_digest, "image/png", labels=("reference", "blue"), folder="boards/alpha")`
- `grant_media(MediaGrant(scope="studio-beta", digest="blake3:a468d9eebb557d8c52b51bf99153fca3875430ad1404683b4ebff41ca4fa3f3c", kind="media/image", media_type="image/png", extension="png", byte_size=76), beta_png_bytes)`
- `create("studio-beta", "beta-cover.png", beta_digest, "image/png", labels=("cover", "red"), folder="deliveries/beta")`
- `put_dependencies(alpha_digest, [{"resourceId": "alpha-source", "digest": beta_digest}])` and the reciprocal beta manifest
- `put_derivation(digest_bytes(b"asset-catalog-previous-main:alpha-derivation"), {"sourceDigest": alpha_digest, "operation": "thumbnail", "version": 1})` and the beta `preview` manifest

After close and reopen, expected rows are `records=2`, `media_grants=2`, `asset_dependencies=2`, and `asset_derivations=2`. `PRAGMA integrity_check` returns `ok`; `alembic_version` is absent.
