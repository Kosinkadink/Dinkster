# Schema wire 15: recursive dynamic entries (co-pinned contract)

Contract of record for the joint backend/frontend schema wire 15 grammar.
Co-pinned 2026-07-28 between backend coordinator
`T-019f9e5d-d2e8-7173-8d6b-88a91bf66880` and frontend coordinator
`T-019f9e58-7a96-7412-89f1-3bb237db21d9`. The frontend proposal of record
is Dinkster-Frontend `docs/wire15-proposal.md`; this document records the
reconciled result (the proposal's Q1-Q11 with the backend's four accepted
deltas). Neither side changes this contract unilaterally.

Grounding sources: ComfyUI `comfy_api/latest/_io.py` (V3 dynamic
expansion: `parse_class_inputs`, `finalize_prefix`, `Autogrow`,
`DynamicCombo`, `DynamicSlot`, `build_nested_inputs`), backend
`dinkster_schema/model.py` / `wire.py` / `elaborate.py` / `solve.py`.

## The DynamicEntry union

A closed discriminated union on the existing `role` field, legal in INPUT
positions only. Outputs keep top-level `output` / `outputFamily` roles;
dynamic output nesting is excluded from wire 15.

### role "input" (ordinary entry)

`id`, `type` (TypeExpr), `required?`, `default?`, `onAbsent?`, `widget?`,
`doc?`, `displayName?`, plus NEW in wire 15: `forceInput?` and `advanced?`
(bool, default false). `forceInput`/`advanced` are presentation-only
(hazard H15, like `displayName`): never schema signature, never execution
identity. Grounded need: upstream forces widget inputs to sockets inside
Autogrow templates (`_AutogrowTemplate`) and on slot sockets
(`DynamicSlot.Input`). `lazy` stays excluded.

### role "inputFamily"

`id`, `template: DynamicEntry[]` (non-empty, RECURSIVE), `minMembers?`,
`maxMembers?`, `memberPrefix?`, `memberNames?`, `required?`, `doc?`,
`displayName?`.

- Wire 15 REPLACES wire 14's family `type` with `template`. An entry
  carrying both is malformed and rejects the node. A wire-14 single-type
  family becomes a one-entry ordinary template in wire 15.
- Naming forms: exactly one of `memberPrefix` / `memberNames`, or
  NEITHER (a native free-suffix family with no ordinal vocabulary and no
  compat lowering).
- `memberPrefix` is emitted EXPLICITLY whenever the family has an ordinal
  prefix vocabulary, INCLUDING when it equals `id`. Absent means "no
  ordinal vocabulary"; the decoder must never infer a prefix from the id.
- Prefix form: keeps `minMembers`/`maxMembers`; upstream bounds are
  min >= 0, max in [1, 100]; ordinal names are ZERO-BASED `{prefix}{i}`
  (`Autogrow.TemplatePrefix`). The first `min` members are required, the
  rest optional (upstream `_expand_schema_for_dynamic` semantics).
- Names form: `memberNames` is an ordered, unique, complete vocabulary;
  `maxMembers` is OMITTED on the wire and capacity is derived as
  `len(memberNames)`; `minMembers <= len(memberNames)`; empty `[]` is a
  legal zero-capacity family and requires `minMembers` 0/omitted.
- Upstream V3 can only produce single-ordinary-member templates today
  (`_AutogrowTemplate` takes one input and asserts it is not a
  DynamicInput); grouped and nested templates are grammar capacity for
  native Dinkster packs and future evolution.

### role "dynamicCombo"

`id`, `options: [{key, inputs: DynamicEntry[]}]` (ordered, recursive),
`default?` (an option key), `required?`, `doc?`, `displayName?`.

- NO widget field: the selector affordance is derived from the ordered
  option keys, the single source of truth, including zero-option combos
  (which simply have no active state; the compat translator refuses a
  REQUIRED zero-option upstream combo loudly as unusable).
- The active choice is DOCUMENT STATE (like slotChoices): signature-
  joining, keyed by the materialized construct path. The selector is NOT
  an input in Dinkster space; compat execution synthesizes the upstream
  selector value from the stored choice.
- Option keys use the single-space exception pinned under Identifier
  grammar (Q7) and retain exact-string identity.

### role "dynamicSlot"

Two MUTUALLY EXCLUSIVE forms, discriminated by which field is present.
Exactly one of `variants` / `slotType`; both or neither rejects the node.

- Variant form (existing native model plus recursion):
  `id`, `variants: [{key, type, inputs: DynamicEntry[], doc?}]`,
  `inputs?: DynamicEntry[]` (shared dependents), `required`, `doc?`,
  `displayName?`. Variant types stay recursively concrete; variant keys
  keep `[A-Za-z0-9_-]+`.
- Open form (ComfyUI compat): `id`, `slotType: TypeExpr` (must NOT be a
  variable; upstream slots are single non-dynamic typed sockets, always
  optional), `inputs: DynamicEntry[]` (shared dependents, recursive),
  `required?`, `forceInput?`, `doc?`, `displayName?`.
- No derived `slotType` is ever emitted for variant slots (single source
  of truth).
- Open-form dependent activation (pinned 2026-07-28, slice B blocker
  resolution): the slot socket itself is always present in the effective
  schema as an optional input. The shared `inputs` dependents - and any
  nested active state and consumed choices under them - materialize ONLY
  when the document stores a value/link at the slot's materialized path.
  An absent open slot contributes the bare optional socket and nothing
  else; stored dependents under an inactive open slot are rejected as
  unknown inputs. This mirrors upstream
  `DynamicSlot._expand_schema_for_dynamic`, which parses dependents only
  when `finalized_id` is present in `live_inputs`. Variant-form
  activation (choice-driven) is unchanged. Node-facing delivery: variant
  slots keep SlotValue-with-choice semantics; an ACTIVE open slot
  delivers the plain connected value under its construct path with no
  choice, and an absent open slot passes nothing.

## Variables at every depth (Q11)

Variable TypeExprs (`{kind: "variable", templateId, allowed?}`) are legal
as the `type` of ANY DynamicEntry at ANY recursive depth: Autogrow
templates, DynamicCombo option entries, dynamicSlot dependents (both
forms). Unification is node-instance-wide by `(nodeId, templateId)`: all
stamped family members and same-template outputs join one equivalence
class. Allowed-set membership is enforced at bind time
(`solve.py _unify`), and binding composes through `list<variable>` and
`asset<variable>` in both directions. The backend's per-invocation
worker-boundary resolution is authoritative; frontend live document
solving layers on top. The wire-14 fast path (single-member MatchType
Autogrow template -> variable-typed inputFamily) shipped 2026-07-28 as
compat phase 1.5 (08542db).

## Identifier grammar (Q7)

Upstream enforces NO grammar on V3 ids/prefixes/names/option keys (plain
strings; a literal dot already corrupts upstream itself via
`build_nested_inputs` splitting on `.`). Wire 15 therefore pins its own:
every STRUCTURAL segment - construct ids, nested entry ids, template leaf
ids, variant keys, `memberPrefix`, each `memberNames` entry - must match
`[A-Za-z0-9_-]+`. Dynamic combo option keys are the single exception and
match `[!-~]+( [!-~]+)*`: printable non-space ASCII tokens separated by single
spaces, with no leading, trailing, or consecutive spaces. The first exception
was pinned jointly with the frontend on 2026-07-29 for ResizeImageMaskNode's
space-containing options. This second widening was pinned jointly the same day
for upstream BFL keys `Flux.2 [pro]` and `Flux.2 [max]`. Because the frontend's
persisted combo branch paths compose `${construct}.[${key}]` and use prefix
matching, sibling option keys may not start with another sibling key followed
by `]`; both sides reject the whole combo. Option keys retain exact-string
identity and are never trimmed, case-folded, or normalized. The backend
enforces these grammars at translation/model validation; a violating V3 node
becomes a loud classified compatSkips entry. NO escaping rule (revival trigger:
a real, working upstream node surfaces with a non-conforming identifier
outside this option-key exception).
Top-level ordinary input and output ids outside dynamic scopes are NOT
structural segments and keep today's wire-14 permissiveness (non-empty
string) under wire 15 as well - working upstream nodes ship ids like
"input_blocks.0." (ModelMerge blocks) and "Audio VAE" (LTXV outputs).
Clarified 2026-07-28 after frontend live verification caught an
over-broad decoder reading; both sides pin the structural-only scope.

## Naming and lowering (Q6)

Upstream compat submission keys are CONSTRUCT-SCOPED: `parse_class_inputs`
pushes every dynamic construct's id onto the prefix and `finalize_prefix`
is a plain `".".join`. Pinned vectors (flat keys accepted by
`/api/compat/comfy/prompt` and upstream `get_finalized_class_inputs`;
nesting into execute()'s dicts is server-side):

- Top-level family `images`, prefix `image`: members `images.image0`,
  `images.image1`, ...; execute() receives
  `values["images"] == {"image0": v, ...}` (`{}` when all-optional and
  empty).
- Combo `mode`, option `batch` containing family `frame` prefix `image`:
  selector key `mode` (value = option key), members `mode.frame.image0`,
  ... (upstream also self-nests the selector value at
  `values["mode"]["mode"]` - backend-internal, no wire impact).
- Combo-in-combo (live proof: DCTestNode, comfy_extras/nodes_logic.py):
  selectors `combo`, `combo.subcombo`; leaves `combo.subcombo.float_x`.
- Slot `source`: socket key `source`, dependents `source.<depId>`,
  recursing identically.
- Grouped templates and Autogrow-in-Autogrow: unproducible upstream, so
  NO compat lowering exists; loud skip if ever encountered.

Dinkster DOCUMENT space (unchanged model contract): family members are
`<family>.<suffix>` with document-chosen stable suffixes, never
renumbered (H10). Grouped templates append the leaf
(`<family>.<suffix>.<leaf>`); a single-ordinary-entry template omits the
leaf segment in BOTH spaces. Option/variant keys NEVER appear in
materialized input ids (mutually exclusive branches may reuse locals).
Member -> upstream-name mapping: prefix form maps ordered document
members to `{prefix}{0..n}` by document order (ordinals deliberately
shift on reorder); names form requires each member suffix to BE one of
`memberNames` (subset allowed, min enforced) because upstream names are
semantic.

## Validation and failure policy (Q10)

Whole-node rejection, both directions: the backend never emits a partial
dynamic interface (translation/validation failures inside a node's
dynamic structure are whole-node classified compatSkips); the frontend
rejects whole nodes on unknown structural roles or TypeExpr kinds and
never mints concrete atoms from unknown markers. Unknown OBJECT FIELDS
stay additive/ignorable on both sides. `template` non-empty; `inputs`
and `options` may be empty; sibling ids unique per recursive list;
`onAbsent: "omit"` invalid on required entries, and all wire-14 strict
type/normalization rules carry over.

## Exclusions and versioning

- Excluded from wire 15: `accept_all_inputs`, `lazy`, `rawLink`, dynamic
  output nesting. All remain loud skips/deferrals with ledgered triggers.
- `SCHEMA_WIRE_VERSION` bumps 14 -> 15 in the backend schema-foundation
  slice. The existing single-version negotiation contract is unchanged:
  the server serves exactly one wire version, a stale `?wire=14` request
  gets the loud machine-readable 406, never a silent downgrade or lossy
  conversion. Like v13 -> v14, the bump rotates all signatures/caches
  once; frontend decoder adoption lands in lockstep before the shared
  server restarts onto the bumped commit.

## Implementation plan

- Slice A (schema foundation): model recursion (`InputFamilySpec`
  template, new `DynamicComboSpec` with document-state choices, two-form
  `DynamicSlotSpec` with shared inputs, `forceInput`/`advanced`
  presentation fields), wire-15 encode/decode under the existing
  single-version negotiation, recursive elaboration and naming
  validation, version bump 14 -> 15.
- Slice B (compat phase 2): V3 DynamicCombo/DynamicSlot/names-form/nested
  translation and upstream execution lowering per the pinned vectors.
