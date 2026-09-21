# Writing a Dinkster pack

A pack is a directory with a `dinkster-pack.toml` manifest and the Python
module(s) it names. Externally supplied packs load in an isolated worker by
default (own process, optionally own venv); the authoring contract is the
same for in-process packs. Everything you need comes through one import:

```python
from dinkster_api.v1 import ...
```

That door only ever grows within v1 - nothing behind it is renamed or
changed incompatibly (breaking changes would mean a `v2` module, not edits).
Internal `dinkster_*` modules stay physically importable because this is
Python, but they carry no promise and `dinkster doctor` flags them.

Placement and sandbox grants remain host policy. A pack declares the OS
resources it needs, but that declaration grants nothing by itself. Dinkster-owned
packs may run in-process in the same environment, where sandbox declarations
do not apply and updating one may require restarting the inference process.

Start from the template in [`templates/pack/`](../templates/pack/) - it is a
complete working pack (manifest, nodes, custom type, rendition, tests, CI)
kept healthy by this repo's own test suite.

## Building a registry archive

Build the deterministic ZIP submitted to a registry with:

```console
dinkster pack archive ./my-pack --output ./my-pack.zip
```

The command validates `dinkster-pack.toml`, writes the archive, and prints its
`blake3:<hex>` digest. Registry archive size, file count, path, and expanded
size limits are checked before output is written. Symlinks are rejected;
version-control, environment, cache, dependency, and build directories are
omitted at every depth.

## The manifest: dinkster-pack.toml

```toml
[pack]
name = "my-pack"                 # the pack id; also the attribution key
# namespaces = ["my-pack"]       # node-type claims; default: the pack name
requires = ["numpy>=1.26"]       # pip requirements; PIN them (doctor warns)

[pack.sandbox]                   # requests only; the host grants authority
gpu = false
network = false
writable-mounts = false

[pack.contracts]
host = "dinkster-pack-host/1"        # composition/hosting contract
api = "dinkster-api/v1"             # author API imported by the pack
# inference = "dinkster-inference/1" # only when consuming inference registries

[pack.dependencies]
# model-provider = ">=2,<3"       # composition order and release compatibility

[pack.requirements.registry]
# "dinkster.samplers" = ["dinkster.euler"] # exact host registry descriptors

[pack.provides.registry]
# "dinkster.samplers" = ["my-pack.guided-euler"] # descriptors this pack registers

[pack.requirements.capabilities]
# "model-provider.video-generation" = ">=2,<3"

[pack.capabilities]
# "my-pack.image-operations" = "1.0.0" # exact provider release version

[pack.entry]
nodes = "my_pack_nodes:NODES"    # module:attr -> sequence of Node classes
types = "my_pack_nodes:register_types"   # optional: fn(TypeRegistry) -> None
# reservations = "my_pack_nodes:PLANNER" # optional: memory policy (advanced)
# consumers = "my_pack_nodes:CONSUMERS"  # optional: governed memory (advanced)

[pack.presentation]              # all optional; badge data, never identity
display_name = "My Pack"
abbr = "MP"                      # <= 8 printable ASCII; search palette chip
mark = "M"                       # one compact glyph (emoji ok); header chip
color = "#4a7d5e"                # chip fill, #rrggbb
icon = "icon.png"                # raster badge, see below

[pack.frontend]                  # optional static files for pack UI
assets = "frontend"

[pack.settings]                  # optional settings shown by the frontend
schema = "settings.schema.json"

[[pack.blueprints]]              # optional starter workflows, see below
id = "upscale"                   # unique within the pack; closed name grammar
name = "Upscale"
description = "Content-level upscale: image in, image out"
tags = ["image", "upscale"]
boundary_inputs = ["core.image"]   # optional: declared boundary type ids
boundary_outputs = ["core.image"]  # (search hints, passed through verbatim)
file = "blueprints/upscale.json" # a workflow document inside the pack dir
```

Entries are `module:attr` strings resolved inside the worker process, with
the pack directory on `sys.path`. Loading the manifest itself never imports
pack code.

## Static frontend assets and settings

`[pack.frontend] assets` names a directory relative to `dinkster-pack.toml`.
The host recursively loads its regular files when the pack loads and serves
each file at:

```text
/packs/{packId}/static/{path-relative-to-assets}
```

For example, `frontend/styles/panel.css` in `my-pack` is available at
`/packs/my-pack/static/styles/panel.css`. The response contains the exact file
bytes and a MIME type inferred from the filename; an unknown extension uses
`application/octet-stream`. Reads require an authenticated principal. There is
no directory listing, and a missing pack, missing file, directory path, or
traversal attempt returns 404.

The assets path must be non-empty, relative to the pack, resolve inside the
pack directory, name a directory, and contain no symlink at the root or below
it. A tree may contain at most 256 files, each at most 4 MiB, with at most
16 MiB total. These bounds keep pack discovery predictable. Bytes are captured
at load time, so changing a file on disk does not change what a running server
returns; reload the pack to publish changed bytes.

`[pack.settings] schema` names a UTF-8 JSON Schema file relative to the
manifest. This example defines every field shape the Settings page supports:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "enabled": {
      "type": "boolean",
      "title": "Enable enhancement",
      "description": "Apply the pack enhancement to new runs.",
      "default": true
    },
    "quality": {
      "type": "integer",
      "title": "Quality level",
      "default": 2,
      "minimum": 1,
      "maximum": 5,
      "multipleOf": 1
    },
    "strength": {
      "type": "number",
      "title": "Strength",
      "default": 0.5,
      "minimum": 0,
      "maximum": 1,
      "multipleOf": 0.05
    },
    "mode": {
      "type": "string",
      "title": "Processing mode",
      "default": "balanced",
      "enum": ["fast", "balanced", "quality"]
    },
    "prefix": {
      "type": "string",
      "title": "Output prefix",
      "description": "Prepended to generated filenames.",
      "default": "enhanced"
    }
  },
  "required": ["enabled", "quality", "strength", "mode", "prefix"]
}
```

This is a deliberately closed JSON Schema subset, not arbitrary JSON Schema:

- The root must be a non-empty object schema with `type: "object"`,
  `additionalProperties: false`, `properties`, and `required`. `$schema` is
  optional. No other root keywords are accepted.
- Every property is required, and `required` lists every property once in the
  same order as `properties`. Setting names start with an ASCII letter and then
  use only ASCII letters, digits, `_`, `.`, or `-`.
- A field has `type`, non-empty `title`, and `default`. Optional `description`
  supplies help text. No other field keywords are accepted.
- Supported types are `boolean`, `integer`, `number`, and `string`. The
  Settings page renders them as a checkbox, integer/number input, and text
  input respectively. A string `enum` renders as a select control.
- `minimum`, `maximum`, and positive `multipleOf` apply only to numbers and
  integers. Numbers must be finite. Integers and integer constraints must be
  within JavaScript's safe integer range. `minimum` cannot exceed `maximum`.
- A string `enum` contains unique strings. Every default must satisfy its type,
  enum, and numeric constraints.

The schema file is limited to 64 KiB. Duplicate JSON object keys, invalid
UTF-8 or JSON, non-finite numbers, unsupported keywords, and inconsistent
defaults or constraints are rejected rather than ignored.

The frontend discovers configured packs through `settings: true` in the
`/api/nodes` packs table, then uses these authenticated endpoints:

```http
GET /api/packs/my-pack/settings
```

```json
{
  "packId": "my-pack",
  "displayName": "My Pack",
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "enabled": {
        "type": "boolean",
        "title": "Enable enhancement",
        "default": true
      }
    },
    "required": ["enabled"]
  },
  "values": {"enabled": true}
}
```

The first GET returns schema defaults. Updates send the complete values object,
not a patch:

```http
PUT /api/packs/my-pack/settings
Content-Type: application/json

{"enabled": false}
```

A successful PUT returns the same response shape as GET with the accepted
values. Omitted, additional, incorrectly typed, out-of-range, non-finite, or
oversized values return 400 and leave the previous object unchanged. The
complete values object is limited to 64 KiB. An unknown pack or a pack without
declared settings returns 404. Unauthenticated requests return 401; PUT also
requires `settings:write` and returns 403 without it. Unreadable stored data or
a persistence failure returns 500.

With `dinkster-serve --library-root <root>`, accepted values are atomically
stored by the host in `<root>/pack-settings/<packId>.json`, one complete JSON
object per pack. Packs do not read or write these files themselves. Without a
library root, values live in memory and reset when the server restarts. A pack
schema change does not silently coerce stored values: incompatible stored data
causes GET to fail until a valid complete object is PUT or the operator removes
the host-owned file.

`dinkster doctor` treats invalid asset or settings declarations as
`manifest.invalid` publish blockers. It rejects missing, absolute, escaping,
symlinked, incorrectly typed, unreadable, or over-budget paths and files. It
also rejects malformed or unsupported settings schemas. These failures stop
the whole declaration instead of advertising assets or controls that the host
cannot serve safely and consistently.

`[pack.sandbox]` is explicit even when all needs are false; `dinkster doctor`
warns when the table is absent. `gpu` and `network` require matching per-pack
host grants. `writable-mounts` allows writes only to mounts the operator
configured as `readwrite`; every other configured mount remains read-only.
`network` declares a need to reach external services, not a transport or
destination; the host grants exact HTTPS origins and routes them through a
per-worker Unix-socket proxy while the worker network namespace stays
unshared. `DINKSTER_EGRESS_PROXY` names that socket; clients send standard HTTP
CONNECT requests over it and perform TLS through the resulting tunnel. The
table never expands filesystem, network, or device authority on its own. A
sandboxed server cannot add or revoke mount binds while workers are running,
so `--allow-mount-changes` is refused with `--sandbox-packs`.

When the server has a library root, every isolated worker receives dedicated
persistent storage in `DINKSTER_PACK_SCRATCH`. Use it for disposable pack caches
or generated state that should survive worker restarts; the operator owns
retention and cleanup. In an OS sandbox, sibling scratch directories are not
visible, and `HOME` and `TMPDIR` resolve to private temporary `/tmp`. Packs
hosted in one worker group share one group scratch directory, so group members
must not treat scratch contents as confidential from each other. Moving a pack
into or out of a group changes its scratch path. In-process packs do not
receive per-pack scratch because they share the serving process.

A provider may implement a schema owned by another pack without taking over
its identity or attribution:

```toml
[pack]
name = "my-provider"
executes = ["schema-owner.generate"]

[pack.arms]
native = ["schema-owner.generate"]

[pack.entry]
nodes = "my_provider:NODES"
arm_nodes = "my_provider:ARM_NODES"
```

`NODES` must expose the exact owner schema for every `executes` claim.
`ARM_NODES` maps each declared arm name to alternate implementations with
that same schema. The default provider body and its alternate bodies share
one worker session, while the schema remains published and attributed by the
owner pack. The default body owns lazy demand decisions; alternate bodies
receive the resolved inputs selected by that hook. The owner hook must account
for any provider selector that changes which lazy inputs execution needs.

Model-backed vision packs additionally declare how they implement stable
provider-selecting schemas:

```toml
[pack]
name = "depth-provider"
namespaces = []
executes = ["dinkster.preprocess.depth"]

[[pack.vision-providers]]
choice = "dinkster.vision.depth"
node = "dinkster.preprocess.depth"
model = "depth-family-v2"
devices = ["cpu", "cuda"]
dtypes = ["float16", "float32"]
batching = "batch"
artifacts = ["depth-model"]

[[pack.assets]]
id = "depth-model"
name = "Depth model"
digest = "blake3:<64 hex characters>"
file = "models/depth.safetensors"

[pack.entry]
nodes = "depth_provider:NODES"
```

The pack name is the compatibility provider id. Owner schemas declare this as
an optional hidden input, so old workflows can retain an exact implementation
pin while new workflows describe only user intent. `choice` must be the remote
combo route used by the schema's `provider` input, `node` must appear in
`executes`, and each artifact must be a digest-pinned `[[pack.assets]]` entry.
When the owner exposes a semantic `model` combo, every provider declares the
matching non-automatic option value. An omitted provider resolves
deterministically from that model and the compatible live arms before graph
compilation and asset preflight. A literal legacy provider remains pinned.
Provider artifact references are conditional: a resolved or literal provider
selection preflights only that provider's models, while a linked model or
provider selection preflights every installed provider for the node. Declared
assets require explicit digest consent before acquisition and are never
downloaded during node execution. Current remote workers advertise equivalent
provider declarations and artifact associations. An explicit empty declaration
is authoritative evidence that a worker cannot provide the capability. Absence
from an older peer is uncertainty, not proof of incompatibility: composition
may use that worker only when its exact compatible pack execution surface
serves the node, and diagnostics and receipts report the actual remote pack.
`devices` accepts
`cpu`, `cuda`, `mps`, `rocm`, and `xpu`; `dtypes` accepts `float16`,
`bfloat16`, and `float32`;
`batching` is `batch` or `per-image`. Composition dispatches each invocation
to the resolved pack without changing the stored workflow.

External text-generation providers use the same pack id selection without
vision artifact or batching metadata:

```toml
[pack]
name = "generation-provider"
namespaces = []
executes = ["schema-owner.generate"]

[[pack.generation-providers]]
choice = "schema-owner.generation.providers"
node = "schema-owner.generate"
label = "Hosted text service"

[pack.entry]
nodes = "generation_provider:NODES"
```

The owner schema declares an optional advanced `provider` combo whose remote
route is `/api/choices/<choice>` and owns that choice as an empty static list.
Composition presents each configured provider by its human-facing `label`.
Omitting the input uses the normal composed execution provider; selecting a
service stores its compatibility pack id and routes execution to that pack.
Each declared `node` must appear in `executes`. One pack may declare multiple
generation nodes against the same choice id. Provider-marked arms are ordered
after normal execution providers and cannot become the default while a normal
provider is live.

A schema-owner pack that intentionally supplies no executable body declares
that boundary explicitly:

```toml
[pack]
name = "schema-owner"
schema-only = ["schema-owner.generate"]
```

The named node must still be present in `NODES` so composition can validate its
schema. The live serving surface withholds it until another pack claims the
exact node id through `executes` with a matching signature; direct internal
planning before then fails plainly, and the placeholder class is never run.
Schema-only nodes cannot declare same-pack body arms or lazy inputs because
there is no owner execution body for those contracts. A
development composer may stage the owner before its provider. Atomic startup
and production activation refuse an incomplete generation; progressive
non-strict startup retracts an incomplete owner and packs that depend on it,
leaving an explicit pack-failure receipt instead of an unroutable node.
Remote sessions do not have a live removal seam: if a remote provides only
part of one schema-only owner's nodes, that incomplete startup is fatal. A
configured remote must provide every otherwise-unimplemented node of an owner
or none of them.

Contracts and requirements are checked before execution. A host/API/
inference contract mismatch, missing exact registry id, missing or
incompatible capability, duplicate registry or capability provider, or
dependency cycle refuses the candidate composition. A pack that provides
model families, samplers, or schedulers lists each exact id under
`[pack.provides.registry]`. Doctor and serving composition require every
listed id to appear in that pack's materialized inference contribution.
Sampler and scheduler contributions currently expose that materialized
provider surface. Model-family declarations participate in contract resolution,
but cannot activate until the pack also materializes a matching family
contribution.
Registry requirements supplied by another pack order that provider before
the consumer and record the provider pack identity in composition provenance.
Dependencies determine provider ordering but do not grant Python imports
between pack implementations; share behavior through registered ids or a
deliberately versioned library.

Every resolved composition has one canonical digest over its mode, pack
versions, artifact digests, and requirement-to-provider mapping. Production
composition requires artifact-pinned packs. Raw local packs compose in
development mode, and that mode remains part of the provenance record. Pack

Names live in one closed grammar: lowercase ASCII letters/digits in
segments joined by single `-`, `_`, or `.`, starting with a letter, at
most 64 characters. The three separators are ONE identity - `foo-bar`,
`foo_bar`, and `foo.bar` are the same name everywhere (this killed a whole
class of ComfyUI "one pack looks like two" drift). `namespaces` declares
the node-type prefixes your pack claims (default: your pack name); every
node type you announce must sit under a claim at a `.` boundary
(`my-pack` covers `my-pack.shout`, never `my-pack-extra.shout`). Claims
are just that - claims: the local host refuses overlapping claims between
installed packs, and the public registry (not your manifest) is the
authority for who owns a namespace. `std`, `comfy`, `core`, and `dinkster`
are reserved roots; don't claim them or names nesting under them.

`[pack.presentation]` is pixels only: it never joins schema signatures,
cache identity, or documents, and malformed fields warn-and-drop (a bad
emoji never stops your pack from loading). The icon must be exactly 64x64,
a static PNG or WebP (sniffed from the bytes - the extension is ignored),
at most 64 KiB, and inside the pack directory. It is served to frontends
by digest with immutable caching; change the image, ship new bytes.

## Frontend modules

`[[pack.extension.frontend-modules]]` publishes a package-relative JavaScript
module with declared privileges and contribution ids. Contribution kinds use
this closed vocabulary: `widgetKind`, `widgetView`, `previewRenderer`,
`textEditorExtension`, `menu`, `command`, `keybinding`, `setting`,
`canvasLayer`, `nodeDecoration`, `hostUi`, `searchProvider`,
`workflowObserver`, `eventConsumer`, `workflowImporter`, `editor`,
`editorBinding`, `panel`, and `virtualNode`. A `virtualNode` contribution needs
the `graph-editor-canvas` privilege and lets a compatible frontend register a
frontend-owned document node that never enters backend execution requests.
Unknown kinds are rejected when the manifest is loaded.

## Blueprints

`[[pack.blueprints]]` entries ship starter workflows with your pack: plain
JSON workflow documents (authored and saved in the frontend - the same
format both directions) exposing content-level inputs/outputs so a freshly
installed pack is usable without visiting a wiki. Blueprints are DATA,
never code: the backend validates only well-formed JSON with a top-level
object, UTF-8, path containment, and size caps (1 MiB per blueprint,
16 MiB total per pack - blueprints are graphs, not assets; reference
models by digest, don't embed bytes), and captures a sha256 digest.
Document semantics belong to the frontend.

Rules that matter: `id` is unique within your pack (the pack id namespaces
it on the wire) and follows the same closed name grammar as pack names;
`description` and `tags` are optional. `boundary_inputs` and
`boundary_outputs` are optional lists of type-id strings declaring the
blueprint's boundary interface - author-declared search hints passed
through VERBATIM so frontends can port-filter blueprints without fetching
bodies. The backend never derives them from the document or checks them
against it; declaring types your document doesn't have makes your
blueprint show up in the wrong searches, nothing more. A malformed entry
warns and drops ALONE - your pack and its sibling blueprints always load.
Frontends see descriptors
(`{id, name, description?, tags?, boundaryInputs?, boundaryOutputs?, digest}`)
inline in the `/api/nodes` packs table and fetch the document lazily from
`GET /api/packs/{packId}/blueprints/{id}` with immutable digest caching -
change the document, ship new bytes, get a new digest. Blueprints never
join schema signatures or execution identity.

## Documentation

Declare documentation explicitly and keep it with the nodes it describes:

```toml
[pack.docs]
dir = "docs"
default_locale = "en"
```

Pages are discovered at `docs/nodes/<node_type>/<locale>.md` and
`docs/guides/<slug>/<locale>.md`. Shared media lives beneath
`docs/assets/`. Dinkster accepts `en` and `zh`; every node or guide must ship
the pack's default locale. Core packs use this same layout.

Each page starts with TOML front matter between `+++` lines:

```markdown
+++
title = "Shout"
summary = "Uppercases text and repeats it."
schema_version = 1
+++
```

`title` and `summary` are required. Node pages may include the current
positive integer `schema_version`; doctor warns when it trails the live
node schema. Keep `NodeSchema.description` to one or two sentences for
search results. Doctor warns above 240 characters; longer explanations
belong in the page. Front matter rides the `/api/docs` descriptor, not the
fetched page body; the page digest addresses the exact body-only Markdown
bytes.

The renderer supports headings, paragraphs, emphasis, inline and fenced
code, ordered and unordered lists, links, images, tables, and block quotes.
Raw HTML is stripped. Links allow only `https:`, `mailto:`, and `dinkster:`.
Markdown images must name an exact `assets/...` path. Videos use:

````markdown
```dinkster-media
asset = "assets/demo.mp4"
caption = "What the example produces"
poster = "assets/demo.webp"
```
````

`asset` is required. `caption` and `poster` are optional, and media never
autoplays. Guides can link to a blueprint or template declared by the same
pack. Declare exactly one target:

````markdown
```dinkster-example
template = "map-and-gather"
caption = "Map values and gather the results"
```
````

Use `blueprint` instead of `template` for an insertable blueprint. `caption`
is optional. Reference a node declared by the pack with:

````markdown
```dinkster-node
node = "dinkster.route.gate"
```
````

Unknown fields, targets, and node types are authoring errors. Optional
`guide.toml` fields beside a guide are `order`, `tags`, and
`guide_kind = "tour"`.

Pages are capped at 256 KiB, images at 2 MiB, videos at 16 MiB, and all
documentation at 32 MiB per pack. Allowed assets are PNG, JPEG, WebP, GIF,
SVG, MP4, and WebM. Invalid paths, locales, pages, references, or assets
warn and drop only the affected content; they never stop the pack.

Wire 42 adds only `hasDocs: true` to documented `/api/nodes` entries.
Descriptors are paged through
`GET /api/docs?q=&kind=&pack=&id=&limit=&cursor=` and bodies are fetched by
digest from `GET /api/packs/{packId}/docs/pages/{digest}` and
`GET /api/packs/{packId}/docs/assets/{digest}`. A matching `If-None-Match`
returns 304; successful bodies are private, immutable, and cached for one
year. Locale fallback is exact locale, base language, pack default, `en`,
then generated help from schema descriptions and port docs.

### Translated node text

Put short translated strings in `locales/<locale>.json` beside the pack
manifest. Locale filenames are canonical lowercase tags such as `en`,
`pt-br`, or `en-x-pseudo`. A catalog has this shape:

```json
{
  "nodes": {
    "my-pack.shout": {
      "displayName": "Shout",
      "description": "Uppercases text.",
      "inputs": { "text": { "displayName": "Text", "doc": "Text to change." } },
      "outputs": { "text": { "displayName": "Text", "doc": "Changed text." } },
      "combos": { "mode": { "strict": "Strict" } }
    }
  },
  "blueprints": { "starter": { "name": "Starter", "description": "Start here." } },
  "guides": { "getting-started": { "title": "Getting started" } },
  "searchTerms": { "my-pack.shout": ["uppercase"] }
}
```

Every section is optional. Node and search-term keys must belong to the
pack's namespace and resolve to a node the pack publishes. Blueprint and
guide keys must resolve within the same pack. Translation strings and
search-term arrays are non-empty. Unknown fields, invalid JSON, symlinks,
and catalogs above 1 MiB warn and drop only that catalog; doctor reports a
`docs.invalid` error. All catalogs together are capped at 16 MiB.

Wire 44 lists surviving catalogs as
`packs[packId].locales[locale] = "sha256:<hex>"`; the field is absent on
older wires and when the pack has no valid catalogs. Fetch exact catalog
bytes from `GET /api/packs/{packId}/locales/{digest}`. The digest is over
the source bytes, which the server returns unchanged with immutable caching
and ETag support. Frontends resolve each translated key independently:
exact locale, base language, another listed regional variant, pack default,
`en`, then the English schema text.

## Authoring nodes

A node is a class with two classmethods: `define_schema()` returning a
`NodeSchema`, and `execute()` taking and returning plain Python values.
You never see envelopes, fingerprints, transports, or placement - the
worker shim does all of that from your schema (hazard H9).

```python
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Shout(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="my-pack.shout",
            display_name="Shout",
            category="text",
            inputs=(
                InputSpec("text", TypeExpr.concrete("core.string")),
                InputSpec("times", TypeExpr.concrete("core.int"),
                          required=False, default=1),
            ),
            outputs=(OutputSpec("shouted", TypeExpr.concrete("core.string")),),
        )

    @classmethod
    def execute(cls, *, text: str, times: int = 1):
        return cls.outputs(shouted=(text.upper() + "!") * times)


NODES = [Shout]
```

Facts that matter:

- **node_type is identity.** Namespace it with your pack name
  (`my-pack.shout`). It keys documents, caching, and replacement rules;
  renaming it is a migration (see replacement rules below).
- **Outputs are a mapping, never a tuple.** `cls.outputs(...)` is sugar
  that validates ids at the return site; returning a plain dict is equally
  valid.
- **execute() may be sync or async.** Declare plain named parameters
  matching your input ids; the shim only ever calls you with
  schema-declared inputs.
- **Type expressions**: `TypeExpr.concrete("core.int")`,
  `TypeExpr.union("core.int", "core.float")`, `TypeExpr.wildcard()`,
  `TypeExpr.variable("T", allowed=(...))` (same variable id unifies across
  ports), and `TypeExpr.list_of(element)` for typed lists, nestable
  (`list<list<core.int>>`). Core atoms: `core.int`, `core.float`,
  `core.bool`, `core.string` (constants `CORE_INT` etc. are exported).
- **Caching is automatic and on by default.** Results are keyed by schema
  signature + input fingerprints, never by node id or process (hazard H4).
  A side-effecting node opts out with `NodeSchema(idempotent=False)` - it
  then re-executes every run and is exempt from result reuse.
- **Absence (optional outputs)**: declare `OutputSpec(..., optional=True)`
  and return `cls.outputs(vae=ABSENT)` (or `AbsentOutput("reason")`) when
  there is deliberately no value. Inputs declare what happens when a linked
  value arrives absent via `InputSpec(on_absent=...)`: `"skip"` (default
  for required inputs - your node becomes absent too), `"omit"` (default
  for optional inputs), `"accept"` (you receive `None`), or `"fail"`.
- **occupies** declares abstract resource kinds (`occupies=("gpu",)`) so
  the scheduler limits concurrency; it is a declaration, never code, and
  never joins cache keys.
- **Dynamic interfaces** (variable-arity ports) go through
  `InputFamilySpec`/`OutputFamilySpec`; membership is document state,
  elaborated once before execution. Most packs never need them - a
  data-dependent number of results is a `list<T>` value on one static
  output, not variable ports.

### Deprecation and replacement

Schemas carry structured lifecycle metadata, all additive and excluded
from signatures:

- `NodeSchema(deprecation=Deprecation(message=..., since=..., replacement=...))`
  renders a badge with your prose and a click-to-replace affordance.
- `NodeSchema(search_visibility="deprecated" | "hidden")` controls search
  listing independently; hidden nodes stay fully valid in existing
  workflows.
- `NodeSchema(replacements=(ReplacementRule(...),))` on the *successor*
  schema ships declarative migration rules (guarded cases, copy/value/link/
  constant mappings, enumRename/scale transforms). Rules are closed data,
  never code; the frontend auto-applies unambiguous ordinary replacements and
  offers ambiguous ones for review. `slot_variants` maps materialized combo or
  slot construct paths to literal declared keys. Source-dependent selections
  use guarded cases; ordinary input mappings carry slot connections. The
  planner resolves choices before elaborating the target, so `inputs` can
  address only paths active in the selected shape. A same-type rule can attach
  `ReplacementMigration` with retired flat input IDs for one-shot stored-
  workflow migration. Doctor and the server validate all source and target
  references.

## Value types and codecs

Register pack-owned types in `register_types(registry)`:

```python
from dinkster_api.v1 import TypeRegistry
import json


def register_types(registry: TypeRegistry) -> None:
    registry.register(
        "my-pack.tally",
        encode=lambda obj: json.dumps(obj, sort_keys=True).encode(),
        decode=lambda data: json.loads(data),
    )
    registry.register_rendition(
        "my-pack.tally", "text", mime="text/plain",
        render=lambda obj: str(obj).encode(), default=True,
    )
```

- **Codec (`encode`/`decode`) is the boundary contract.** With a declared
  codec your values cross process/machine boundaries and cache to disk.
  Without one, a pickle fallback is used and doctor warns
  (`types.fallback-codec`): fine for prototyping, not for shipping.
- **Type ids are atoms** in the closed grammar `name | "list<" id ">"`;
  `<` and `>` are banned in atom names. List values are built structurally
  by the runtime - never register a `list<...>` id yourself.
- **Renditions** are how rich types become browser-renderable bytes for
  previews and frozen-run peeking: register kinds with MIME types, mark
  exactly one `default=True`. (Planned video convention: a static poster
  image is the default; playable containers are additional kinds.)
- **Optional TypeSpec hooks**: `fingerprint` (content hash; defaults to
  hashing encoded bytes), `meta` (small JSON facts like shape/length),
  `coerce` (normalize JSON-shaped literals into your runtime form at wrap
  time; must be idempotent), `inline` (declare a small scalar form for
  descriptors/event summaries - only int/float/bool/str survive the
  central policy).
- **Resource-backed types** (models living in a worker) go through
  `ResourceHandle` / `register_resource_handle_type`: the envelope crosses
  boundaries, the resource stays put.

## Reporting, logging, events

From inside `execute()`:

- `report_progress(step, total, text="decoding")` - progress units are
  whatever you count.
- `report_preview(data, mime="image/jpeg", width=..., height=...)` -
  encoded preview *images*, small (they cross every boundary between your
  node and the frontend).
- `report_event("my-pack.stage", {...}, blob=...)` - pack-defined events;
  names must be dot-namespaced, payloads JSON-representable.
- `report_value_diagnostic(code, data=None)` - persistent, nonblocking facts
  about output values, such as `media_format_fallback` with `requested`,
  `effective`, `outputId`, and `reason` details. Codes and details are
  extensible, not a media allowlist. Details must be JSON-representable and
  are snapshotted; `code` and `nodeId` are reserved keys in `data`.
  Unlike transient events, these records persist on successful outputs even
  without an observer, cross worker boundaries, and replay on cache hits
  through `value_diagnostics` with the current engine-supplied node ID.
  Awaited tasks and `asyncio.to_thread` inherit the invocation scope; calls
  outside execution or after the invocation ends are silent no-ops.
  Invalid arguments raise `ValueError`, and failed invocations retain no
  records. Do not use `report_event` for value diagnostics.
- `pack_logger("my-pack")` - a stdlib logger named `dinkster.pack.my-pack`:
  origin-tagged, host-controlled verbosity (`dinkster-serve --log
  dinkster.pack.my-pack=debug`). Logs are for humans; anything a frontend
  renders goes through `report_*`. Never install handlers, never print.

For a preserving media-format substitution:

```python
from dinkster_api.v1 import report_value_diagnostic

report_value_diagnostic(
    "media_format_fallback",
    {
        "outputId": "video",
        "requested": {
            "container": "mp4",
            "codec": "h264",
            "pixelFormat": "yuv420p",
            "channelLayout": "5.1",
        },
        "effective": {
            "container": "matroska",
            "codec": "ffv1",
            "pixelFormat": "gbrap16le",
            "channelLayout": "5.1",
        },
        "reason": "preserve_alpha_and_precision",
    },
)
```

The four format fields in `requested` and `effective` are strings or null
(`None` in Python). Requested null means unspecified; effective null means
not applicable. The format worker must preserve actual alpha, HDR sample
values, and channels, and use a capability error only when no preserving
output is possible. This helper reports the substitution; it does not
choose formats.

### Mirror declarations (frontend preview estimates)

A schema may declare that a client can compute an instant preview estimate
of the node's transform without a round trip:

```python
NodeSchema(
    ...,
    mirror=MirrorSpec(
        kind="expression",
        precision="bounded",
        tolerance=MirrorTolerance(relative=1e-12),
        grammar_version=1,
    ),
)
```

Facts that matter:

- **Presentation only.** The engine, scheduler, and workers never read
  `mirror`; mirrored results never become outputs or cached values, and the
  authoritative `execute()` result always replaces the estimate. Like
  `emits_previews`, `mirror` is excluded from the schema signature.
- **Kinds**: `"expression"` (the client re-evaluates the deterministic
  expression grammar; declare the `grammar_version` you were verified
  against) and `"glsl"` (inline GLSL ES 3.00 fragment shader in `source`,
  at most 16 KiB UTF-8).
- **Precision is a contract.** `"exact"` promises bit-identical IEEE-754
  binary64 results and is restricted to correctly-rounded operations
  (arithmetic, comparisons, sqrt, integer ops). Anything touching
  transcendentals or GPU rasterization is `"bounded"` and must declare a
  `MirrorTolerance` (`relative` for scalars, `per_channel` for images).
- **Declare only what is verified.** A mirror declaration is a parity
  promise; back it with vector coverage against the authoritative
  implementation (see `tests/fixtures/math_expression_v1.json` for the
  expression corpus shape) before shipping it.

## The import contract (what doctor enforces)

`dinkster doctor` is the pack linter; run it locally and in CI:

```bash
uv run dinkster-doctor path/to/pack        # human output
uv run dinkster-doctor --json path/to/pack # structured, for CI gating
```

It loads your manifest, statically scans sources, then imports your entries
in a disposable subprocess and reports what happened. Errors (publish
blockers): importing `dinkster_*` internals bypassing the api door
(`imports.private-module`, `imports.host-machinery`), importing another
pack implementation (`imports.pack-implementation`), unresolvable entries,
invalid schemas or replacement rules, duplicate node types, invalid
presentation icons (`presentation.icon-invalid`), invalid blueprints
(`blueprints.invalid`, `blueprints.duplicate-id`,
`blueprints.budget-exceeded`), or any unresolved `[pack.extension]` scope
(`extension.entry-unresolvable`). Warnings (drift):
unpinned requirements, import-time side effects (stdout output, thread
spawns), raw import-time logging, slow imports (> 2 s), codec-less types
that your own schemas put on edges, foreign-origin logging, and extension
contribution kinds or capabilities that no runtime consumer implements.

The rules exist because ComfyUI's ecosystem grew the opposite habits and
every host change became a breaking change. A pack that only uses
`dinkster_api.v1`, imports quietly, and declares codecs runs unchanged
in-process, in another venv, and on another machine.

## Running and testing

```bash
uv run dinkster-serve --pack path/to/dinkster-pack.toml   # isolated worker, repeatable flag
```

Each `--pack` runs in its own process and attributes its nodes on
`/api/nodes`. Add `--watch-packs` for live reload, boundary-cost diagnostics,
and cache-miss explanations while developing.

Test your nodes directly - they are plain classmethods over plain values:

```python
def test_shout():
    out = Shout.execute(text="hey", times=2)
    assert out == {"shouted": "HEY!HEY!"}
```

and gate CI on `dinkster-doctor .` as its own step (see the template's
workflow). Don't import the doctor from your tests: tests live in the pack
directory, so they hold to the same one-door rule as the rest of your code,
and `dinkster_workers` is host machinery. For golden-file testing of schema
wire output, encode with `schema_to_wire` (exported through
`dinkster_api.v1`) and compare committed JSON - the wire is self-describing
via its `schemaVersion` field.

## Legacy ComfyUI packs

Unported ComfyUI packs load through the compat layer
(`dinkster-serve --comfy-root <install> --legacy-pack <dir>`), quarantined in
one worker and attributed as `comfy.<pack>`. A legacy pack directory can
ship (or a user can drop in) a `dinkster-pack.toml` containing only
`[pack.presentation]` and/or `[[pack.blueprints]]` to get its own
badge/icon and starter workflows without being a loadable Dinkster pack (a
legacy blueprint references `comfy.<pack>.*` node types, which resolve
only while the compat worker has the pack loaded). Compat is a migration
on-ramp, best-effort by design (hazard H11); porting to a native pack is
the endorsed destination.

## Porting a legacy pack (`dinkster-port`)

`dinkster-port` turns a ComfyUI pack's declared schemas into a native pack
skeleton - the mechanical part of a port, done for you. Both node APIs
port: v1 `NODE_CLASS_MAPPINGS`, V3 `comfy_entrypoint`, and mixed packs
shipping both (the legacy *runtime* loads only a mixed pack's v1 half,
matching upstream ComfyUI's precedence, and flags the ignored V3 half;
porting covers both):

```bash
uv run dinkster-port path/to/legacy_pack --name my-pack \
    --comfyui-root ~/ComfyUI          # or DINKSTER_COMFYUI_ROOT
```

The legacy pack is imported in a disposable probe subprocess, never in
the CLI's process, under the ComfyUI install's own interpreter
(`--comfy-python`, else `$DINKSTER_COMFYUI_PYTHON`, else
`<comfy-root>/venv/bin/python` - packs import torch and friends). For
v1 mappings the probe runs the same loader and translate.py rules the
compat worker runs, so the generated schemas can never drift from what
compat actually serves. For V3 packs it resolves the extension the way
ComfyUI's own loader does (`comfy_entrypoint` -> `on_load` ->
`get_node_list`) and converts each node's `Schema` object directly -
authoritative schema objects, never source regex.

What you get in `--out` (default `./<name>`): a `dinkster-pack.toml`, a
nodes module with one native `Node` class per translatable source node,
a `pyproject.toml`, a README with a porting checklist (listing each
node's source API), and a smoke-test suite. The output is doctor-clean
on day one. Schemas are faithful translations - recursive type
expressions, required/default handling, `list<T>` sockets for
IS_LIST/`is_input_list` nodes, category/display name, `output_node`,
and the source name preserved as an alias so existing API submissions
keep resolving. V3-specific declarations carry over to their Dinkster
equivalents: MatchType templates become type variables, Autogrow inputs
become input families, `is_api_node` becomes `io_bound`,
`is_deprecated`/`is_dev_only` become deprecation metadata and hidden
search visibility, and per-socket tooltips become doc strings. Opaque
`comfy.*` types the schemas reference are registered in a generated
`register_types` (doctor warns until you declare codecs - that warning
is honest, not noise).

What you do NOT get is behavior: every generated `execute()` raises
`NotImplementedError` with a `TODO(port)` marker and carries the source
function's code as a reference comment. The tool refuses false
fidelity - v1 and V3 functions run against ComfyUI's runtime and would
fail in ways a silent copy would hide. Work the TODO markers, delete
the reference comments, run `dinkster-doctor` and the generated tests.

`--nodes A,B` ports a subset (unknown or untranslatable requested
names are an error, never a silent skip); nodes the translator skips
are listed with reasons and recorded in the generated README. A
non-empty output directory is refused without `--force`.
