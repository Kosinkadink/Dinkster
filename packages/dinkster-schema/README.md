# dinkster-schema

Before public release, the schema wire contract is revised in place and only
version 1 is served or accepted. Compatibility across wire versions begins at
public release; after that point, a wire change will add a new version with an
explicit compatibility decoder and a documented support window.

`dinkster-schema` is the typed, V3-native node interface model and node-authoring
surface. It depends only on `dinkster-values`; graph validation, workers, the
engine, the extension API, and protocol layers consume its schemas rather than
maintaining parallel interface descriptions.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```sh
uv sync --all-packages
```

The package is not published separately yet and provides no console script.

## Use

The public surface exports `NodeSchema`, `InputSpec`, `OutputSpec`, `TypeExpr`,
`Node`, elaboration and type-solving helpers, schema wire functions, reporting,
logging, naming rules, and declarative replacement models. A minimal node keeps
its schema and execution contract together:

```python
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Double(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete("core.int")
        return NodeSchema(
            node_type="example.double",
            inputs=(InputSpec("value", integer),),
            outputs=(OutputSpec("value", integer),),
        )

    @classmethod
    def execute(cls, value: int):
        return cls.outputs(value=value * 2)
```

`Double.schema()` caches the declaration, while `Double.outputs(...)` checks
static output IDs at the return site. Execution still receives and returns
plain Python values; workers own envelope and transport details.

## Stored output descriptors

`NodeSchema.output_descriptors` and its `OutputDescriptorsSpec`
names a required top-level `core.string` input containing JSON:

```json
{"entries":[{"id":"answer","name":"Answer","type":"integer","value":42}]}
```

`choices` declares the concrete type for each choice ID (`integer` above).
Elaboration produces ordinary ordered outputs with entry IDs as wire identities
and entry names as display names. IDs match `[A-Za-z0-9_-]+`; IDs and names are
unique. Documents are limited to 1 MiB and 512 entries, further bounded by
`min_entries` and `max_entries`. Linked sources and undeclared type choices
are invalid. Additional fields such as `value` or `expression` belong to the
node and remain in its literal input identity. Output count never depends on
execution results or runtime list length.

Workers independently project the stored source and compare the full ordered
interface before invoking the node. The node receives `output_spec.outputs`
and returns a mapping keyed by those IDs, rather than using the static
`Node.outputs` helper. Pack code can read its entries with
`dinkster_api.v1.output_descriptor_entries`.

`fixed_ids=True` requires entry IDs to equal their semantic choice IDs.
`OutputProbeSpec` associates such declarations with a required asset input
and a revisioned host-owned model probe. The descriptor is catalog metadata,
not a worker activation request or an assertion that a runtime is installed.

Non-default alpha or mask policies are supported on inputs, outputs, descriptor
choices, nested input families, combos, and slots. Default policies remain
omitted from signatures.

## Input-family combo options

A `ComboWidget` can declare an `InputFamilyOptionSource`.
The frontend builds the combo choices from that dynamic input family's member
descriptors: each member suffix is the stable stored option value, while its
occurrence-authored display name is an editable label. Native graphs contain
only the selected suffix and existing family links such as `values.member_id`;
labels never enter execution or cache identity. Elaboration replaces the
source with static suffix options on the effective schema used by workers.
Clients must reject duplicate labels or render a stable visible disambiguation;
they never use labels as selection identity.

`ReplacementRule` is the shared data-only migration language. Ordinary input
mappings address static inputs or paths materialized by a case's dynamic combo
and slot choices. `ReplacementCase.slot_variants` maps materialized construct
paths to literal declared option or variant keys. Source-dependent selections
use guarded cases; ordinary input mappings carry slot connections. The planner
resolves choices before elaborating and validating the target.
Same-type rules can attach `ReplacementMigration` with retired flat input IDs
so stored nodes migrate once into the selected dynamic shape.
`ReplacementCase.input_families` additionally supports copying a top-level
dynamic family while preserving suffixes and authored order, or creating an
explicit ordered member list from static source inputs.

`ComfyAliasRegistry` models maintained ComfyUI import translations separately
from native schemas. Each record carries an exact `ReplacementRule`, confidence
metadata, and an import-only source schema snapshot. Registry wire helpers are
strict and reject unknown or noncanonical fields.

## Pre-execution known values

An `OutputSpec` may declare `known_value=OutputKnownValue(input="value")` when
the output is exactly one of the node's top-level inputs. The input and output
must share one concrete `core.int`, `core.float`, `core.string`, or
`core.boolean` type. Clients can present the resolved value downstream before
execution without treating it as an estimate or cached result. If the input is
connected, its upstream execution provenance remains authoritative.

Known-value declarations are presentation metadata: they do not affect schema
signatures, submitted graphs, or cache keys.

## Pre-execution output representations

An `OutputSpec` may declare `represents=OutputRepresents(...)` when selecting
an asset on one of the node's own asset-widget inputs is enough to estimate
that output before execution. `input` names the top-level input and
`rendition` names the client-side conversion. Clients render only rendition
kinds they understand, estimates never enter submitted graphs or cache keys,
and an executed output always replaces its estimate.

The `decoded-image` rendition means the selected still image after EXIF
orientation, ICC-aware conversion to sRGB, and conversion to contiguous RGB
float values in `[0, 1]`. It excludes alpha masks, metadata, video, and audio.

`applies` may limit the promise to required dynamic-combo options using the
same mapping and validation rules as `MirrorSpec.applies`. Elaboration strips
a covered scope and removes an uncovered representation.

## GLSL mirrors

A node schema may declare `mirror=MirrorSpec(kind="glsl", ...)` carrying one
inline GLSL ES 3.00 fragment shader of at most 16384 UTF-8 bytes. Like every
mirror, it is presentation metadata: a client MAY run it over client-resident
inputs to render an instant preview estimate, and authoritative `execute()`
results always replace estimates.

The binding contract between the schema and the shader:

- Each image input `<id>` binds as `uniform sampler2D u_<id>`. The shader
  reads exact texels with `texelFetch(u_<id>, ivec2(gl_FragCoord.xy), 0)`,
  so no sampler filtering or wrap state participates in parity. The shader
  must declare `precision highp sampler2D;`: GLSL ES 3.00 predeclares
  samplers as lowp, and texel reads inherit the sampler's precision, which
  destroys parity regardless of float precision.
- Each float input binds as `uniform float <id>`, each int input as
  `uniform int <id>`, each boolean input as `uniform bool <id>`, and each
  combo input as `uniform int <id>` holding the index of the selected value
  in the schema's declared options.
- The client draws one batch frame per pass into a framebuffer with the
  frame's pixel dimensions; the fragment output is the estimate for the same
  pixel. Texture channels beyond a value's real channel count are
  unspecified, and parity applies per real channel.
- Mirrors model only the success path of `execute()`. For inputs the node
  would reject, the estimate is unspecified and a client should render none.

A mirror may declare `applies`, a mapping from combo input ids to the option
values the shader covers. Absent `applies`, the mirror covers the node's whole
input space. When `applies` is present, the estimate is specified only for
executions whose named combo inputs hold one of the listed values; for any
other selection a client must render no estimate. Every key must name a
declared required dynamic combo and every value one of its declared option
keys; schema construction rejects anything else. Elaboration resolves the
scope against the consumed combo choices: a covered variant keeps the mirror
with `applies` stripped, an uncovered variant elaborates with no mirror, so
an elaborated schema never carries `applies`.

Estimates render at the preview's pixel dimensions, which may differ from the
authoritative output's. Parameters denominated in output pixels (for example
a blur radius) therefore produce a visually coarser or finer estimate at
preview resolution. That divergence is presentation-only: parity corpora pin
the shader against `execute()` at equal dimensions.

Bounded GLSL mirrors declare `tolerance.per_channel`. GLSL ES 3.00 does not
guarantee correctly rounded arithmetic, so `exact` generally overclaims for
shaders.

`ComfyGroupRegistry` describes bounded, connected ComfyUI subgraphs with static
interfaces that can collapse to import-only group schemas before the same
replacement language is applied. Records classify every source input, preserve
declared boundaries, parameters, constants, modes, and topology, and carry
grouped confidence.

## Learn more

See DESIGN 3.1 for the single schema source of truth, DESIGN 3.13 for generic
and list types, and DESIGN 3.15 for optional outputs. See hazards H1, H7, H9,
H10, H19, and H23 in `docs/hazards.md`.

Focused tests include `tests/test_schema.py`, `tests/test_names.py`,
`tests/test_replace.py`, `tests/test_comfy_alias_registry.py`,
`tests/test_comfy_group_registry.py`, `tests/test_reporting.py`, and
`tests/test_logging.py`.
