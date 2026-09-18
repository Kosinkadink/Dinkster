# Lazy switch lowering (document-time branch pruning)

Status: PROPOSED 2026-07-29 (backend design pass; user approved the
document-time-pruning direction 2026-07-29, including that
API-submitted prompts go through the same lowering with no client
pre-pruning or breadcrumb contract). Revised same day after design
review (target handling, physical cone pruning, literal portability,
carve-out seam, schema-generation pinning). Frontend ACKED the
additive `selector` schema field 2026-07-29 (shape as proposed plus
the two validation pins in section 3; their decoder is
source-verified TOLERANT of unknown additive top-level fields, so
there is NO sequencing constraint - backend may ship first and
ComfySwitchNode decodes as an ordinary node until their consumption
slice lands; client SchemaRegistry hashes rotate for schemas gaining
the field, expected and harmless). IMPLEMENTATION UNBLOCKED.
M1 AMENDMENT 2026-08-08: stored booleans still use this physical
lowering path, while link-fed selectors are intentionally retained
unchanged for runtime lazy demand. Queue admission now re-runs the
lowering as a fail-closed identity check against the pinned execution
schemas. Selector-bearing region bodies and any stored residue that
would still rewrite remain client-safe refusals.
Current-upstream completion follows ComfyUI's executor convention:
`ComfySwitchNode.check_lazy_status` returns a branch-name list while
the selected input is unresolved and implicitly returns `None` once
ready. The in-process worker maps only exact `None` to zero further
demands; list/tuple conversion and malformed non-None refusal stay
unchanged.

Restores the largest single compat skip: upstream `ComfySwitchNode`
(lazy branch selection) blocks 70 of 223 RED official templates
(report-10). Upstream semantics grounded at /home/kosin/ComfyUI
f4b99bc `comfy_extras/nodes_logic.py`: a required boolean `switch`
input plus two required lazy `MatchType` inputs `on_false`/`on_true`
sharing one template with the output; `check_lazy_status` evaluates
only the selected branch and implicitly returns `None` once its value
is present. `ComfySoftSwitchNode` exists upstream but is
deliberately unregistered; out of scope.

## 1. Design summary

A switch with a stored boolean is lowered away at submission time,
before the engine sees the graph:

- Every consumer of the switch's output is rewired to the source of
  the branch selected by the STORED boolean `switch` value.
- The switch node is then dropped from the graph.
- The unselected branch's exclusive upstream cone is PHYSICALLY
  deleted by the lowering itself (section 5, rule 4). Planner
  reachability (`packages/dinkster-graph/src/dinkster_graph/plan.py
  reachable_from`) is NOT sufficient on its own: full-document
  validation (`validate.py`: "errors anywhere in the document fail
  the run, even on branches the targets do not reach") and the
  submission asset preflight (`packages/dinkster-server/src/
  dinkster_server/preflight.py` scans every node and literal at every
  nesting level) both run over the whole graph, so an inactive
  branch left in place could still fail validation or demand assets.

Consequences that fall out with ZERO engine/planner/cache changes:

- Only the executed subgraph is planned and validated, so invocation
  cache identity (dinkster-engine `engine.py _cache_key`: schema
  signature + selection + resolved input fingerprints, never a
  full-graph hash) covers exactly the executed subgraph.
- No worker calling-convention change, no laziness in the node
  contract, no cache-identity story for unevaluated inputs.

This was the complete initial design. M1 subsequently added runtime
lazy demand for a link-fed selector, which `lower_selectors` preserves
exactly instead of eagerly evaluating either branch; see section 6.

## 2. Where the lowering runs (both submission boundaries)

One shared lowering function:

```
lower_selectors(graph, targets, schemas) -> LoweringResult
# LoweringResult carries: graph, targets, problems
```

Targets are part of the contract because submission targets are NODE
IDS (`app.py handle_submit` requires a non-empty list of node ids;
results are keyed by them). v1 REFUSES a selector node that is
itself a target (`prompt.selector_is_target`): retargeting a dropped
switch to its branch source would silently change the result key the
client asked for, and a stored-value branch has no node id to target
at all. Target aliasing is a possible later extension and would need
an output-identity contract - exactly the breadcrumb machinery this
design avoids.

Applied to the native `Graph` before queueing:

- Compat prompt path: inside `translate_prompt`
  (`packages/dinkster-compat-comfy/src/dinkster_compat_comfy/prompt.py`),
  after graph AND targets are constructed (targets are derived late
  in that function), emitting anchored `PromptProblem`s on failure
  (codes below).
- Native document path: in the server submit path
  (`packages/dinkster-server/src/dinkster_server/app.py handle_submit`)
  after `graph_from_wire`, so wire-15 documents and any internal
  submission lower identically. API prompts therefore need no
  pre-pruning and no breadcrumbs.

Exact `handle_submit` sequencing (order matters):

1. Parse submitted graph and targets.
2. Compute the job-content fingerprint from the RAW submitted
   content, exactly as today (`app.py` hashes raw body graph,
   targets, priority, scope, provenance). Fingerprints are
   queue-level submission identity, not execution cache identity; a
   byte-identical duplicate returns the existing job before lowering
   runs again.
3. Handle active duplicate/conflict (existing fingerprint check).
4. Pin ONE execution runtime / schema snapshot. Today
   `JobQueue.submit` pins execution only when the job is finally
   queued, and there is an `await` between preflight and submission,
   so composition/hot-reload could otherwise change schemas between
   lowering and execution. `JobQueue.submit` grows an optional
   pre-pinned execution argument for this.
5. Lower selectors against that snapshot's schemas (rewire, drop,
   physically prune the inactive exclusive cone).
6. Asset-preflight the LOWERED graph (inactive-branch assets must
   not block submission).
7. Queue with the already-pinned execution runtime.

`JobQueue.submit` enforces the invariant rather than relying on an
assertion. Against the already pinned execution schemas it calls
`lower_selectors(graph, targets, schemas)` and admits only an empty
problem list plus exact graph/target identity. This permits already
lowered graphs and M1 link-fed selectors, which lowering intentionally
preserves. Stored-selector residue that would rewrite, malformed stored
selectors, and selectors inside region bodies fail with a typed,
client-safe admission error. The queue never performs the rewrite.

The function iterates to fixpoint so chained switches (a branch fed
by another switch) lower correctly; each pass strictly removes nodes,
so termination is trivial.

## 3. Schema representation (contract-visible, needs frontend ACK)

Additive OPTIONAL node-schema field, tentatively:

```
"selector": {
  "input": "switch",
  "branches": { "false": "on_false", "true": "on_true" }
}
```

Semantics: a stored selector is replaced at compile or submission time
by a rewiring from its single output to the named branch input. Under
the later M1 amendment, a link-fed selector is instead preserved for
runtime lazy demand. `branches` keys are the boolean literals as strings;
values are input ids that must exist on the node, share the output's
type/template, and be distinct.

Frontend-pinned validation rules (ACK 2026-07-29, fail-closed on
both sides, enforced at SelectorSpec/NodeSchema construction and at
wire decode):

- A selector-bearing schema MUST declare exactly one output. The
  server never emits `selector` on a multi-output node; the frontend
  treats selector+multi-output as a malformed schema, never guessing
  which output reroutes. An explicit output id can arrive additively
  with the combo-selector widening if multi-output selectors ever
  exist.
- `selector.input` must be distinct from both branch input ids.

Model shape: a frozen `SelectorSpec` dataclass on `NodeSchema` (not
a raw dict inside the frozen schema). Wire encode OMITS the field
when `None`, so every existing schema's wire bytes - and therefore
its `schema_signature` (dinkster-schema `wire.py`; signature hashes the
wire form minus the presentation/lifecycle exclusion list) - are
unchanged. Only schemas that carry the field get new signatures, and
all of those are new. `selector` is computation-relevant, so it is
deliberately NOT added to the signature exclusion list.

Rationale for schema-carried data over a private compat table: the
frontend needs the same fact for branch-aware display and for its own
document->prompt compile lowering; two hand-maintained tables would
drift. This matches the replacement-rule precedent (`replace.py`:
declarative data carried on schemas, mirrored on both sides).

Wire impact: additive optional field on the node-schema wire; NO wire
version bump (frontend tolerance for unknown additive fields to be
confirmed in their ACK - if their decoder is strict, they land the
mirror first, exactly like the option-key grammar sequencing).

Native packs MAY declare `selector` on their own boolean-selector
nodes; the vocabulary deliberately starts boolean-only (exactly two
branches). Widening to combo-selector switches is additive later.

## 4. Compat translation carve-out

CORRECTION 2026-07-29 (implementation delegate escape-hatch, verified
by the coordinator against source): the original seam placement was
wrong. `translate_v3_schema` is NEVER called on the production path -
its only caller is `port_probe.py`. The runtime catalog path is
`bootstrap.load_comfyui_nodes()` -> `translate.py translate_mappings()`
-> `translate_node()`, whose input loop rejects lazy configs
(`config.get("lazy")` raise under `iter_v1_inputs`) before any
`translate_v3.py` code can run. A carve-out confined to
`translate_v3.py` is dead code at runtime.

Corrected placement: identity + shape detection lives in ONE shared
helper in `translate.py` (the module both paths can import), invoked
at each translator entry point:

- `translate.py translate_node()` (the RUNTIME seam): when detection
  passes, thread the allow-list for exactly the two verified branch
  inputs through its lazy rejection in the input loop.
- `translate_v3.py translate_v3_schema()` (the PROBE seam, port_probe
  only): thread the same allow-list through BOTH of its lazy-rejection
  seams (`_translate_v3_entry` rejects lazy before
  `_ordinary_v3_input` is reached, and `_ordinary_v3_input` has its
  own rejection; threading only one is dead code), so probe and
  runtime translations agree.

Detection itself is evaluated once per node per entry point via the
shared helper; the pins below are single-sourced there.

The carve-out is identity-pinned AND shape-verified:

- Identity: node id `ComfySwitchNode`, inputs named exactly
  `switch` / `on_false` / `on_true`.
- Shape (defense against upstream drift): one required boolean
  input, exactly two required lazy inputs sharing one MatchType
  template that is also the sole output's template, no list
  wrapping. If the shape check fails, the node stays a loud
  classified skip - identity alone never forces a translation.

When both hold, the translator threads an allow-list for exactly the
two verified branch inputs through both rejection seams, emits the
normal schema (lazy flags dropped) PLUS the `selector` field. Any
other lazy usage anywhere stays a loud classified skip. At both
pinned upstreams, ComfySwitchNode is the only registered core node
using lazy semantics (re-verify in the implementation slice).

## 5. Lowering rules (exact)

For each graph node whose schema carries `selector`:

1. Read the value of `selector.input`. A Link marks the M1 runtime
   form: retain the node and graph exactly and continue without
   evaluating either branch.
2. For a stored selector, if the node id is in `targets` ->
   `prompt.selector_is_target` refusal (section 2).
3. Validate the stored value.
   - Absent or not a JSON boolean -> problem
     `prompt.bad_selector_value` (compat) / submission problem
     (native), anchored to the node.
4. Resolve the selected branch input and rewire every consumer of
   this node's output. Consumers include ordinary node inputs AND
   top-level `RegionNode.inputs` (region nodes consume top-level
   outputs; region BODIES are separately scoped and out of reach).
   - Link -> rewire each consumer to that link source.
   - Stored value -> inline that stored value into each consumer's
     input, preserving its exact form (`TypedLiteral` stays
     `TypedLiteral`, plain literal stays plain). This is NOT
     universally valid and is checked by normal validation, not
     assumed: a plain literal requires a runtime-resolvable concrete
     destination type, list inputs require list-shaped literals,
     wildcard/union/variable destinations accept only TypedLiterals,
     and region element ports expect `list<T>`. MatchType relates
     declared edge types; it does not stamp a plain literal. If the
     rewired graph fails validation, that is a loud submission
     problem anchored to the consumer input - never a silent drop.
   - Absent -> problem `prompt.missing_branch` (the selected branch
     must be present; the unselected branch is ignored entirely).
5. Drop the node, then physically delete the unselected branch's
   EXCLUSIVE upstream cone: candidate set = the upstream cone of the
   unselected branch source (accumulated across all lowered switches
   in the pass); iteratively delete candidates that are not targets
   and are no longer referenced by any surviving node input or
   region input. Nodes shared with any retained consumer survive;
   unrelated orphans elsewhere in the document are untouched and
   keep today's validation behavior.
6. Region boundaries: lowering handles only top-level nodes. A selector inside
   a region body remains in that body and executes through the region's
   occurrence-local lazy-demand fixpoint. Each map, fold, or while iteration
   selects and prepares only its demanded branch.
7. Repeat until every remaining selector is a link-fed M1 runtime node.

Validation of the rewired graph is the normal `validate()` pass over
the lowered graph.

## 6. Computed (link-fed) selector: implemented by M1

Census evidence found computed-value lazy in 0.6% of templates. The
M1 trigger fired in 2026-08: a link-fed selector now remains in the
graph and its lazy hook demands only the selected producer cone. The
unselected producer does not start, and warm runs reuse the selected
producer from cache while replaying exact `lazy_demand` attribution.
Stored booleans still take the physical-pruning path above. Region,
list, and dynamic selector widening remains out of scope.

## 7. Failure/edge inventory

- Stored selector node named as a submission target: loud refusal (5.2).
- Stored selector value not boolean / absent: loud problem (5.3).
- Selected stored branch absent: loud problem (5.4).
- Unselected branch absent or dangling: fine; ignored.
- Inlined literal rejected by a consumer's literal rules: loud
  validation problem on the lowered graph (5.4); not silent.
- Switch output unconsumed: node dropped; cone GC still runs.
- Chained switches: fixpoint iteration (5.7).
- Switch in region body: loud refusal, deferred (5.6).
- Shared upstream node feeding both branches or another retained
  consumer: survives cone GC by construction (5.5).
- Schema hot-reload between lowering and execution: prevented by
  pinning one execution snapshot before lowering (section 2).
- Muted/bypassed switch node: CANNOT REACH this boundary. Mute/
  bypass is a document-level concept with no representation in
  either submitted form - the native graph wire's only "mode" field
  is the region-output gather mode (dinkster-graph wire.py), and compat
  API prompts arrive pre-resolved with no node modes. (Correction
  2026-07-29: do NOT cite the frontend's
  compile.wire15.modesUnsupported gate here - it fires only for
  recursive wire-15 documents and protects neither compat documents
  nor direct API submitters. It is irrelevant to this boundary
  precisely because modes are unrepresentable in what the server
  receives.) Mode lowering semantics remain the frontend Option A
  program; if node modes ever become wire-representable, their
  interaction with selector cones is a new joint design and this doc
  gains that trigger.

## 8. Slice plan

One backend implementation slice (frontend ACK received 2026-07-29,
delegable now; no collision with inference S3 - surface is dinkster-schema
model/wire additive field, dinkster-graph or shared lowering module,
compat prompt.py + translate.py + translate_v3.py, server app.py + queue.py,
tests; no compose.py or native_arm.py edits):

1. Frozen `SelectorSpec` on NodeSchema + wire encode/decode
   (omit-when-None) + validation (input exists, boolean, branches
   exist/distinct/type-share, exactly one output, selector.input
   distinct from both branches) + schema_signature coverage proving
   existing signatures are byte-stable.
2. Shared `lower_selectors(graph, targets, schemas)` with the
   section-5 rules, cone GC, and problem codes.
3. Compat carve-out (4) so ComfySwitchNode translates with
   `selector`; catalog skip count drops by one (CustomCombo and
   ResizeImageMaskNode tracked separately).
4. Wire-in at both boundaries with the section-2 handle_submit
   sequencing, including the pre-pinned execution argument on
   `JobQueue.submit` and its lowered-graphs-only invariant.
5. Tests: lowering matrix (link/stored/absent branches, TypedLiteral
   vs plain inlining, literal-rejection loudness, chained,
   unconsumed, shared-cone survival, selector-as-target refusal,
   region refusal, computed-selector identity preservation, non-boolean refusal,
   region-node-input consumers), preflight proof that inactive
   assets no longer block, live compat catalog delta, live
   end-to-end ComfySwitchNode prompt execution proving only the
   selected branch runs (cache/event evidence).
6. PROMISES + ROADMAP flips in the same commit.
