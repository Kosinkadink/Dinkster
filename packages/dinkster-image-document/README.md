# dinkster-image-document

`dinkster-image-document` owns the closed ImageDocument v2 parser and CPU renderer
shared by workspace assets and the `dinkster.layers` value. Both use canonical
UTF-8 JSON bytes with `format: "dinkster-image"` and `formatVersion: 2`. Decoding
v1 documents migrates structurally, without flattening or copying pixels.

Resources carry content digests, byte sizes, dimensions, media types, and
`alphaMode` (`straight`, `opaque`, or `premultiplied`), never paths. Raster bytes
up to 4096 bytes can use base64 `inline`; larger resources are published through
`DINKSTER_ASSET_VAULT`. Value metadata advertises external `asset_refs` for worker
staging. A bound asset resolver is retained when a document is edited.

Both `dinkster.layers` and `comfy.LAYERS` expose a `document` rendition through
`GET /api/values?...&rendition=document`, with MIME
`application/vnd.dinkster.image-document+json`. It returns canonical document
bytes without rendering or loading raster resources; `dinkster.layers` keeps its
PNG `image` rendition as the default. Retrieval requires read access to the
completed job, not a Save Layers node. It does not adopt library resources or
grant access to referenced assets: the image-document adoption endpoint still
requires `assets:write` and existing `media/image` grants in the target scope
for external resources.

Layers are raster or group records with ordered children, opacity, visibility,
blend mode, clipping, masks, and fixed-point affine transforms. Group `isolation`
defaults to `isolated`; `pass-through` lets child blends see the sibling backdrop,
with group opacity, masks, and transforms retained. Siblings sort
by optional `z_index`, with array order breaking ties. Optional transform
`components` contain x/y, width/height, rotation (radians), flipHorizontal,
flipVertical, and sourceWidth/sourceHeight; their derived affine must agree
exactly with the stored affine. They are not a second transform schema.

Canvas color uses the shared IMAGE FFmpeg integer enums, defaulting to
`{primaries: 1, transfer: 13, range: 2}`; unknown/reserved integers are retained.
Optional `background` is straight RGBA in 0..65535. The renderer supports all
26 Comfy blend modes plus deterministic dissolve, perceptual or linear-light
compositing, and masks with enable/invert/opacity, alpha or luminance channels,
and multiply/add/subtract/intersect combination. Affines use deterministic
nearest-neighbor sampling and 16-bit premultiplied intermediates.

`apply_commands` accepts canvas, remove, reorder, group, layer, mask, transform,
add_mask, and remove_mask operations. `dinkster.layers.edit` exposes those commands
as JSON. `dinkster.compositor` is `{version: 2, documentDigest, commands}`; a null
digest is unbound, a mismatched digest falls back to native placement. Legacy
native stacks and Comfy LAYERS/COMPOSITOR migrate at their boundaries.

Selectors are `composite`, `layer:<id>`, and `mask:<id>`. `flatten` uses the same
canonical PNG quantization as the Render API, returns RGB when opaque or RGBA
when transparency is present, and always returns a transparency mask (1 means
transparent). `dinkster.layers.split` returns raster-layer IMAGE/MASK lists and
full-canvas regions. `dinkster.render_image_document` delegates to this flatten
path and is deprecated in favor of Load Layers plus Flatten Layers.

Load Layers accepts native JSON, OpenRaster, and raster PSD layers. Save Layers
writes native JSON, OpenRaster, or PSD through library save targets. OpenRaster
keeps portable raster visibility, opacity, integer translation, and blend
properties editable and embeds the exact native document and resources;
unchanged files reimport byte-exactly. Masks, source crops, and other affine
transforms are baked into individual standard layers. Modified standard stack
entries are imported instead of silently using stale native data. Unsupported
ORA blend/clipping/background semantics use a merged standard view with a warning,
while the embedded native tree stays editable. PSD import rasterizes text and
smart objects as ordinary layers and converts non-8-bit or non-RGB documents to
8-bit RGBA with warnings. Vector masks are not preserved. Effects and adjustment
layers without raster pixels produce diagnostics, not paint tools. PSD export
preserves straight-alpha raster pixels, integer offsets, layer order and names,
visibility, 8-bit opacity, PSD blend and clipping modes, isolated or pass-through
groups, and one raster mask per raster layer. Opacity is rounded to the nearest
PSD 8-bit value. A mask's channel, inversion, and opacity are baked into grayscale
coverage; its enabled state remains editable. Non-integer or affine transforms,
group masks, transforms, or clipping, combined masks, linear compositing, canvas
backgrounds, and grain extract/merge use one merged standard layer with a warning.
Effects, adjustment layers, smart objects, text, vector masks, 16/32-bit channels,
and CMYK are not export capabilities.
The PSD merged image is generated with the layer tree for non-layer-aware readers.
PSD container bytes are not stable across writer versions; decoded pixels and
layer properties are the interchange contract. A zero-layer native
document uses one transparent standard composite so its ORA stack is not empty.
OpenRaster layer opacity is rounded to the nearest 8-bit value and written just
above its exact ratio so Krita's truncating importer preserves that value. The
embedded native document retains exact Dinkster 16-bit opacity, and the merged image
remains the normative OpenRaster composite.

The `dinkster-image-document-v2-cpu-reference` profile accepts canvases up to
4,194,304 pixels, up to 67,108,864 layer/mask work-pixels, and up to 512 MiB
of declared resource bytes plus decoded RGBA storage. Inputs outside those
bounds fail before raster decoding. These render bounds do not limit transport
or structural editing. The renderer contract includes Pillow/JPEG/WebP decoder
versions so cache identities change with decoding behavior.
