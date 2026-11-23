# Image document ownership and interchange cautions

## Produced document ownership

1. The host acquires `dinkster_assets.scratch.hold_produced_vault(vault.root)`
   before the first lazy decode, edit or render of a produced document. Release
   the lease only after all document operations and spawned work have settled,
   including on exception and cancellation. Short-lived SDK/tools may use
   `contextlib.closing()` around the existing lease.
2. To keep a document beyond that context, save its bytes and every external
   resource to durable storage before releasing the transient lease. A result
   publication alone does not keep temporary bytes. Preserve original legacy
   envelope bytes and fingerprints when deriving a document.
3. Bare decoded documents, including an `ImageDocument` with a retained
   resolver, have no vault ownership or automatic retention. Scalar and
   read-only decoding without a lease remains possible. Maintained callers
   that depend on produced bytes later must own their lease explicitly.

Resolver binding controls lookup, not lifetime. See
[result publications](../packages/dinkster-server/README.md#result-publications)
for saved results, and the
[ownership API](../packages/dinkster-library/src/dinkster_library/produced_scratch.py)
for the host lease boundary.

## Interchange decisions

PSD container bytes are not stable across writer versions; decoded pixels and
layer properties are the interchange contract. OpenRaster's merged image is the
normative composite; its embedded native document retains exact Dinkster
16-bit opacity. Inspect interchange warnings before treating a standard layer
stack as equivalent to the native tree. The
[interchange implementation](../packages/dinkster-image-document/src/dinkster_image_document/interchange.py)
and [round-trip tests](../tests/test_layer_documents.py) are the reference for
which properties remain editable and which require a merged standard view.

Use Load Layers plus Flatten Layers rather than the deprecated
`dinkster.render_image_document` node. Native asset-backed documents are distinct
from structural tensor `LAYERS` values.
