# Extension integration cautions

Use [dinkster_api.v1](../packages/dinkster-api/src/dinkster_api/v1.py) for pack
imports, not internal packages. See [pack authoring](pack-authoring.md) for
manifests and node declarations. The
[extension declarations](../packages/dinkster-protocol/src/dinkster_protocol/extensions.py),
[JSON surfaces](../packages/dinkster-protocol/src/dinkster_protocol/pack_surfaces.py)
and [host routes](../packages/dinkster-server/src/dinkster_server/pack_surfaces.py)
are the API reference.

## Invocation and publication

- Remote clients read `current_execution_context()` for the current invocation
  and its cooperative `cancelled()` callback. Packs must not create or replace
  the host's execution context.
- Sampling consumers call `prepare_sampling_model(model)` once before
  constructing their sampler or guider. Let preparation errors propagate.
- Save nodes call `attach_saved_artifact(ref.to_wire())` after successful
  `AssetWriter` publication. Attach every published file in a batch; an ordinary
  ASSET output is not a saved-file attachment. See
  [saved-artifact validation](../packages/dinkster-workers/src/dinkster_workers/saved_artifacts.py).
- Compositor nodes call `publish_image_document(document.data)` with canonical
  original input bytes before replaying edits. This retains the document in the
  vault, not the selected output folder. See
  [produced document ownership](image-document-operations.md) before lazy work.

## Frontend consumers and route handlers

Frontend route consumers must send `If-Match: sha256:<snapshot digest>` and
check the successful response's `X-Dinkster-Extension-Snapshot` echo. Do not
consume a response as belonging to a different selected generation.

Cancellation is cooperative. Async handlers can be cancelled, but Python cannot
forcibly interrupt a synchronous handler already running in a thread; a dispatch
deadline is not proof its work stopped. Ship self-contained ES modules rather
than arbitrary remote modules or undeclared chunks. Frontend privileges do not
grant node execution, filesystem access or arbitrary HTTP access.

The opt-in [video preview pack](../packages/dinkster-video-runtime/README.md#preview-initialization)
provides a policy-route and typed-event integration exercise. Its
[end-to-end test](../tests/test_serve.py) covers the node, route, module and
correlated WebSocket event.

## Schema evolution and conversions

Before public release, follow the
[current-version-only wire policy](hazards.md#h7-pre-10-wire-formats-are-unstable-and-only-the-current-version-decodes).
At public release, a new wire version must carry an explicit compatibility
decoder and documented support window rather than silently breaking consumers.

First-party pack identities declare only their ComfyUI-class conversion rule.
Do not retain predecessor Dinkster-pack rules or historical family migrations.
The [replacement API](../packages/dinkster-schema/src/dinkster_schema/replace.py)
and [identity helper](../packages/dinkster-schema/src/dinkster_schema/identity.py)
are the declaration reference.
