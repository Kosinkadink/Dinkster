# 1->N node replacement (deprecation to a connected subgraph)

Status: IMPLEMENTED 2026-07-29 (frontend ACK same day, all five section-5
questions answered; frontend shipped 53acf7b/7df6599, backend mirror landed
afterward as sequenced; frontend promises row at Dinkster-Frontend 6daab2e).
The frontend's `replace/model.ts` + `replace/plan.ts` are the
reference shape and the executor of replacement rules;
`packages/dinkster-schema/src/dinkster_schema/replace.py` is a
field-for-field mirror and stays one ("we mirror, never dialect").
Sequencing confirmed: the frontend plan/apply extension lands FIRST,
the backend replace.py mirror second. Both sides' docstrings already
reserve merge/split transforms as additive growth, never
reinterpretation.

Motivation (user directive 2026-07-29, template parity program): some
deprecations cannot be expressed as one node -> one node. The driving
example: a successor node adopts a `list<T>` input/output where the
predecessor had scalar ports, so a faithful migration must insert
helper nodes (join/split/construct) around the successor to keep the
document's existing connections meaning the same thing. The custom-pack
RED cohort (84 templates, report-11) will also want 1->N mappings when
native equivalents restructure ports.

## 1. Grounding: what exists today (verified 2026-07-29)

- One `ReplacementRule` migrates one source type; guarded cases
  evaluate top-down, first match wins, last case unconditional.
- A case is strictly 1->1: `to` names ONE target type; `inputs` maps
  target input id -> MappingSource (copy/value/link/constant, closed
  transforms); `outputs` maps target output id -> source output id.
- The frontend `NodeReplacePlan` is 1->1 IN PLACE: the source node id
  SURVIVES as the target node's id. Everything downstream of that
  invariant - staleness guards (`from` type check, `boundaryGuard`),
  same-node `inputRewires`/`outputRewires`, net rewires, subgraph
  boundary bindings "stay valid because the node id survives" - is
  anchored on id survival.
- Unmapped source connections are review-forcing warnings; ambiguous
  or impossible mappings are errors that leave the node untouched.

Any 1->N vocabulary that abandons id survival would force a redesign
of every guard above. So it doesn't.

## 2. Proposed vocabulary (additive fields on ReplacementCase)

The primary target keeps the source node id, exactly as today. Helper
nodes are new document nodes created by the case:

```
{
  "when": { ... },
  "to": "dinkster.new_node",
  "nodes": {
    "join": { "type": "dinkster.list_join", "values": { "count": 2 } }
  },
  "inputs": {
    "strength": { "kind": "copy", "input": "strength" },
    "join:item_0": { "kind": "copy", "input": "image_a" },
    "join:item_1": { "kind": "copy", "input": "image_b" }
  },
  "links": [
    { "from": "join:items", "to": "images" }
  ],
  "outputs": {
    "out": "IMAGE"
  }
}
```

- `nodes` (NEW, optional): local id -> `{ type, values? }`. Local ids
  use the structural id grammar `^[A-Za-z0-9_-]+$`. `values` are
  constants only (JSON), merged over the helper schema's declared
  defaults, all written explicitly at plan time (the existing
  reproducibility rule). When `nodes` is absent every existing field
  keeps byte-identical semantics - the extension is purely additive.
- Port ADDRESSES (NEW grammar, used only inside a case): a bare port
  id addresses the primary target; `localId:port` addresses a helper.
  `:` is outside both the structural id grammar and the widened
  option-key grammar, so the form is unambiguous and needs no escaping.
- `inputs` keys widen from target input ids to target input ADDRESSES:
  a MappingSource can now feed a helper input directly from source
  node state (moved links land on the helper).
- `links` (NEW, optional): internal wiring among the created nodes and
  the primary; `from` is an output address, `to` is an input address.
  The internal graph (helpers + primary) must be acyclic; each input
  address is fed by at most ONE of {an `inputs` entry, a `links`
  entry} - double-feeding is a construction-time error.
- `outputs` values stay SOURCE output ids; keys widen to output
  ADDRESSES, so downstream consumers of a source output can be rewired
  to a helper's output (e.g. a split helper) rather than the primary.
  The existing fan-in rule (one source output feeds at most one entry)
  is unchanged.

Nothing else moves: predicates, transforms, MappingSource kinds, the
fallback contract, and layer precedence are untouched.

## 3. Semantics the executor (frontend) owns - proposed defaults

- Helper DOCUMENT ids: deterministic derivation from the replaced
  node's id + the local id (node ids forbid only `/ [ ]`, so e.g.
  `{sourceId}:{localId}` is representable). Same plan input -> same
  ids, so re-planning is stable and golden corpora stay meaningful.
  Collision with an existing document id is a plan ERROR forcing
  manual review - the planner never renames its way around a
  collision.
- Provenance: the plan records rule/case/local id per created node so
  the review UI can show "inserted by the X migration".
- Recursion/cycles: rules are planned against DOCUMENT nodes only;
  nodes created by a plan are not re-planned inside the same
  scan/apply pass. A later scan may migrate them normally; a rule set
  whose helper types chain further replacements is legal but each hop
  is a separate reviewable step, so A->B->A ping-pong cannot loop
  silently inside one command.
- Subgraph boundary bindings: bindings live on the primary node and
  keep working via id survival, exactly as today. v1 REFUSES (plan
  ERROR, review) a case that would need a binding to follow a port
  onto a HELPER node - bindings are node-anchored and silently
  retargeting them to a new node id is exactly the kind of guess the
  planner never makes. Revive with a dedicated binding-retarget
  vocabulary if a real rule needs it.
- Lossy policy unchanged: source connections no mapping or address
  consumes are review-forcing warnings with explicit dropLinks.
- VIEW placement of created helper nodes (frontend-pinned default,
  2026-07-29): the planner is geometry-blind and the plan carries no
  view data; the FRONTEND owns deterministic placement at APPLY time
  (an offset column derived from the primary node's view position,
  stable for identical inputs so golden corpora and re-planning stay
  meaningful). Placement is executor territory with no vocabulary
  impact; the backend mirror must never grow a competing notion of
  placement.

## 4. Backend mirror scope (lands SECOND, after the frontend plan/apply slice ships)

- `replace.py`: additive `nodes`/`links` fields + address grammar on
  `inputs`/`outputs` keys, wire encode/decode with exact model.ts
  field names, construction-time validation (local id grammar, address
  wellformedness, acyclic internal links, single-feeder rule, fan-in
  rule, helper `values` JSON check).
- `dinkster doctor` cross-schema checks widen to resolve helper types and
  their port ids.
- No engine/server involvement: replacement stays a document-level
  authoring operation executed by the frontend command layer; the
  backend only carries and validates the data. (ComfyUI->Dinkster import
  translation can consume the same rules later - that consumer reads,
  it does not add vocabulary.)

## 5. Resolved by frontend ACK (2026-07-29)

All five questions answered; grounded by the frontend against
replace/model.ts + plan.ts at their current main.

1. Address syntax: string `localId:port` ACCEPTED (inputs/outputs are
   Records, so JSON object keys must be strings; a structured form
   would restructure both fields into arrays for zero gain). Verified:
   `:` is legal in document node ids (their DINKSTER_NODE_ID_FORBIDDEN
   forbids only `/ [ ]`) and outside the structural id grammar, so
   parsing is unambiguous. Validation pins BOTH sides enforce
   identically: at most one `:`, both halves nonempty, localId must
   exist in `nodes`, and the port half stays subject to the existing
   static-only rejection (the plan.ts error on `.` or `#` in target
   input keys applies to the port half of an address).
2. Plan shape: their NodeReplacePlan grows (a) `createdNodes` entries
   `{nodeId (derived doc id), localId, type, values (complete explicit
   defaults+constants, same reproducibility rule), controllers?}` -
   localId doubles as per-node provenance beside the existing
   from/caseIndex; (b) internal `links` become explicit link-create
   entries; (c) rewires gain node-qualified destinations: entries whose
   destination is a helper carry an optional `node` field (today's
   same-node entries imply the surviving id); netSinks already carry
   full PortRefs; netSourceRewires gain the same optional `node`.
   Staleness needs exactly ONE new apply-time guard beyond the
   existing from-type + boundaryGuard checks: every derived
   created-node id must be ABSENT from the document at apply time
   (plan-time collision ERROR re-checked at apply; collision after
   planning = stale refusal). Id survival keeps every other existing
   guard valid.
3. Id derivation `{sourceId}:{localId}`: ACCEPTED, collision = plan
   ERROR forcing review, re-checked at apply. Noted consequence: their
   scan CHAINS hops on the same primary node (A->B->C, scratch-applied
   per hop); two hops of one chain reusing the same localId on the
   same primary will collide and the chain becomes blocked with the
   planned prefix surviving for review - the correct loud outcome, not
   a defect. No discriminator in v1.
4. No-replan-in-pass: ACCEPTED, matches their architecture. Scan
   chains follow the PRIMARY node id only; helpers created by a hop
   exist in the scratch document as ordinary nodes for LATER hops'
   connection state but are never themselves chain-planned in-pass; a
   later scan migrates them normally. Cycle/depth guards unchanged.
5. Boundary-binding-to-helper: v1 refusal ACCEPTED. When a bound
   port's mapping destination is a helper address, they emit a new
   plan ERROR `replace.boundary.helper` instead of retargeting
   (consistent with their replace.boundary.unmapped philosophy).
   Revive with dedicated binding-retarget vocabulary on a real case.

Two additional semantics the backend mirror can rely on (frontend
confirmed): MappingSource kinds on helper addresses keep the same
widget checks evaluated against the HELPER schema (constant/value to
a widget-less helper input = error, same as primary); helper types
must resolve on the connected backend exactly like `to` (unknown
helper type = plan error, `replace.target.unknown` analog).
