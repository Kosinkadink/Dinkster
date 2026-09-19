# dinkster-api

`dinkster-api` is the one versioned extension door for Dinkster pack authors.
Packs import `dinkster_api.v1` and nothing else; the underlying schema, value,
memory, and asset packages are internal implementation details. The v1 surface
is additive-only: incompatible changes require a new versioned module rather
than edits that break existing packs.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```console
uv sync --all-packages
```

The package is not published separately yet.

## Use

`dinkster_api.v1` exports extension declaration/composition/snapshot contracts,
node schema types, reporting and logging helpers, value type registration,
memory policy interfaces, asset APIs, and `schema_to_wire`.
A minimal node receives and returns plain Python values:

```python
from dinkster_api.v1 import CORE_INT, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

INT = TypeExpr.concrete(CORE_INT)

class Double(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="my-pack.double",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("result", INT),),
        )

    @classmethod
    def execute(cls, *, value: int):
        return cls.outputs(result=value * 2)
```

Import the version explicitly. Do not import pack APIs from `dinkster_schema`,
`dinkster_values`, or other internal packages.

`OutputDescriptorsSpec` declares ordered heterogeneous outputs from a stored
JSON input. Execution receives `OutputInterface.outputs`; use
`output_descriptor_entries` to read node-owned entry fields and return a plain
mapping keyed by the effective IDs. See the
[descriptor contract](../dinkster-schema/README.md#stored-output-descriptors).

## Declarative extensions

Scope-separated extension entry points use one additive manifest table that
mirrors `[pack.entry]`; loading this TOML never imports the references:

```toml
[pack.extension]
schema = "my_pack.extensions:schema"
server = "my_pack.extensions:server"
privileges = ["schema", "server"]
capabilities = ["background-jobs", "routes"]
```

The five privileges are `schema`, `server`, `inference`, `frontend`, and
`training`.
Capabilities use the closed `EXTENSION_CAPABILITIES` vocabulary, so typos fail
manifest loading instead of becoming inert declarations. Privileges and
capabilities never grant node execution ownership or change native arm selection.

Every future contribution surface declares one `CompositionMode` through a
`ContributionSurfaceDescriptor`: `ordered_list`, `wrapper_chain`, `exclusive`,
`observers`, or `keyed_registry`. `ExtensionSnapshot` and `ActiveExtension`
provide the frozen RPC-clean behavior identity model; use
`canonical_extension_snapshot` and `extension_behavior_hash` for its stable
serialization and sha256. Presentation metadata is intentionally absent from
that hash.

### JSON routes and events

`PackRoute`, `PackEvent`, `JsonObjectSchema`, and `JsonField` describe bounded
JSON contracts. Manifests are the source of truth, validated without imports:

```toml
[pack.extension]
privileges = ["schema", "server"]
capabilities = ["routes"]

[[pack.extension.routes]]
id = "preview-policy"
method = "GET"
handler = "my_pack:preview_policy"
request = {}
response = { maxFrames = "integer" }

[[pack.extension.events]]
name = "my-pack.preview-initialized"
payload = { frameCount = "integer" }
```

Routes are installed only under
`/api/extensions/{pack_id}/routes/{route_id}`. IDs are lowercase ASCII slugs,
not arbitrary paths or patterns. GET requires `jobs:read`, accepts no body or
query, and passes `{}` to the handler. POST requires `jobs:submit` and accepts
an `application/json` body. The host owns authentication, methods, headers,
status codes, the 64 KiB JSON limit, and the 10-second dispatch deadline.
Handlers receive a plain mapping and return a mapping, synchronously or
asynchronously, in the owning pack worker. They cannot choose another handler
through request data. Responses are checked in the worker and host; malformed
requests return 400, oversize bodies 413, wrong content types 415, unknown
routes 404, wrong methods 405, worker errors 502, and dispatch timeouts 504.
Clients may send `If-Match: sha256:<snapshot digest>`; a stale generation
returns 412 before worker dispatch. Successful responses echo the selected
digest in `X-Dinkster-Extension-Snapshot`. Frontend route consumers require both.
Cancellation is cooperative: cancelling an async handler is supported; Python
cannot forcibly interrupt a synchronous handler already running in a thread.

Schemas describe closed objects with required `string`, `integer`, `number`,
or `boolean` fields. Field names are sorted in `JsonObjectSchema.fields`.
Booleans are not numbers; nonfinite numbers and extra/missing fields are
rejected. Python integers must be within `-(2**53 - 1)` through `2**53 - 1`,
inclusive, for both `integer` and `number` fields, so JavaScript consumers
preserve their exact values. Finite Python floats remain valid for `number`
fields, including outside that integer range. This applies to route requests,
route responses, and typed events. Nested objects, optional fields, arbitrary
HTTP responses, and binary pack channels are not part of this contract.

```python
from dinkster_api.v1 import JsonField, JsonObjectSchema, PackEvent, report_pack_event

INITIALIZED = PackEvent(
    "my-pack.preview-initialized",
    JsonObjectSchema((JsonField("frameCount", "integer"),)),
)

def preview_policy(request):
    return {"maxFrames": 120}

# Inside a node's execute method:
report_pack_event(INITIALIZED, {"frameCount": 24})
```

Typed events use the existing `node_event` stream and EventHub, not a separate
socket. They retain host-selected `worker`, `pack`, and `executionArm`, plus
run/job/node correlation and sequence. Declared events add `schemaVersion: 1`
and the invocation's pinned `extensionSnapshotDigest`. The engine drops
wrong-owner, malformed, or binary payloads for declared events. Events are
execution-scoped, client-targeted advisory chatter with bounded drop-oldest
delivery, never durable outputs. `report_event` remains compatible for older
packs; undeclared events do not become declared frontend capabilities.

### Frontend module assets

`[[pack.extension.frontend-modules]]` declares `id`, a `./relative.js` module,
`privileges`, and `contributions`. IDs belong to the pack. Contributions name
`id`, `kind`, and, for `eventConsumer`, an event declared by the same pack.
The four independent frontend privileges are `schema-widget`,
`graph-editor-canvas`, `app-workflow`, and `event-consumer`. They do not grant
node execution, filesystem access, or arbitrary HTTP access. Frontend hosts
may deny them before importing modules without affecting backend node use.

The canonical snapshot's optional `routes`, `events`, and `frontend` arrays
bind the generation; absent arrays are omitted, preserving existing digests.
Each effective frontend entry contains `id`, `moduleUrl`, `moduleDigest`,
`authorizedPrivileges`, and `contributions`. Module URLs are same-origin:
`/api/extension-assets/{pack_id}/{sha256:digest}/{entry_id}.js`.
Serving reads only the declared installed-pack file, rejects symlink escapes,
verifies the bytes against the selected digest, and serves UTF-8 JavaScript
with immutable private caching and `nosniff`. Modules are limited to 1 MiB.
Entries must be self-contained ES modules; arbitrary remote modules and
undeclared chunks are not served. Stale, changed, removed, or unselected
modules return 404. Catalog, snapshot, and module reads do not activate a
worker or import pack code in the host.

The opt-in first-party [video preview pack](../dinkster-video/README.md#preview-initialization)
uses the same policy route and typed event to initialize VHS-style video
preview metadata. No ComfyUI JavaScript, PromptServer, or global extension
registry is involved.

## Learn more

- [Pack authoring](../../docs/pack-authoring.md) describes manifests, nodes,
  custom value types, and the compatibility contract.
- [The pack template](../../templates/pack/) is a complete working starting
  point.
- [DESIGN.md section 3.6](../../DESIGN.md#36-extension-api-predictable-no-hacks-required)
  defines the versioned extension surface.
- `tests/test_api_v1.py`, `tests/test_doctor.py`, and
  `tests/test_pack_template.py` exercise the public door and its policy.
