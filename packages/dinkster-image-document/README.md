# dinkster-image-document

Closed ImageDocument parsing, structural editing, layered-file interchange and
CPU reference rendering for workspace assets and image-document values.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_layer_documents.py tests/test_image_document_rendering.py tests/test_image_document_adoption.py
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference and operations

- [Document ownership and interchange cautions](../../docs/image-document-operations.md)
- [Public API](src/dinkster_image_document/__init__.py)
- [Format](src/dinkster_image_document/format.py), [editing](src/dinkster_image_document/document.py) and [rendering](src/dinkster_image_document/render.py)
- [Layered-file interchange](src/dinkster_image_document/interchange.py)
- [Supported image operations](../../docs/supported/image-operations.md)
