# Implicit list-map lowering (compat prompt translation)

Status: BINDING SPEC 2026-07-30 (backend design pass; user directed
starting the LISTEXP census slices immediately). Evidence base:
docs/research/listexp-conclusions.md - implicit single-list mapping
carries 34.2 percent user-weighted fan-out usage (census slice A) and
is compat's one real feature-set regression risk; zero packs author
against multi-list broadcast (L3 = 0 in 1172 rows).
AMENDED 2026-07-30 (user adjudication): unequal-length lists do NOT
fail loudly. L3 = 0 only covers explicit-flag node code; it cannot
see workflow-level reliance on upstream's implicit alignment
(especially 1-vs-N scalar-as-list), and loud length-mismatch
mechanisms in ComfyUI proved hostile in practice. Compat preserves
upstream broadcast semantics via a new native binding mode
(section 3a).
AMENDED 2026-07-30 (G1 adjudication): mapped OUTPUT_IS_LIST outputs use
typed RegionOutput flatten semantics (section 4), closing slice B's
flatten refusal without changing their effective `list<T>` type.
Upstream semantics pinned at ComfyUI 947c2749 execution.py:157-317:
unless INPUT_IS_LIST, the executor maps the node function over list
inputs (invocation count = longest list, shorter lists repeat their
final element); OUTPUT_IS_LIST outputs are flattened per output flag.

## 1. Design summary

ComfyUI's implicit per-element mapping is not a runtime construct in
Dinkster. It is lowered away at prompt translation time into the
explicit region primitive the engine already has:

- After `translate_prompt()` builds the Graph
  (packages/dinkster-compat-comfy/src/dinkster_compat_comfy/prompt.py:234-435),
  a new lowering pass rewrites every node that ComfyUI would have
  implicitly mapped into a `RegionNode(kind="map", binding="zip")`
  (packages/dinkster-graph/src/dinkster_graph/model.py:90-130) wrapping
  that single node.
- Multi-list consumers align exactly as upstream does: iterate to
  the longest list, shorter lists repeat their final element. This is
  carried by a new native `binding="broadcast"` mode (section 3a),
  not emulated in compat code.
- The pass lives in a new compat module
  (packages/dinkster-compat-comfy/src/dinkster_compat_comfy/listmap.py) and
  is invoked from `translate_prompt()` before returning. One additive
  native change (the broadcast binding mode, section 3a); no wire
  version bump, no schema change, no new dependency edges.

## 2. Cardinality inference (single topological pass)

Nodes are translated from schemas the translator already resolves, so
input/output `TypeExpr`s are available (`TypeExpr.cardinality()`,
dinkster-schema model.py:189).

Effective output cardinality per node, in topological order:

- Base: the declared output type's cardinality.
- A node becomes MAPPED when at least one linked input satisfies ALL
  of: (a) the input's declared type has scalar cardinality, (b) the
  driving source's EFFECTIVE type is `list<E>`, (c) `E` is
  link-compatible with the declared input type under the same
  compatibility rules ordinary links use. Incompatible element types
  are NOT lowered - they fall through to the existing validation
  diagnostics unchanged.
- A mapped node's effective output types become `list<declared>`,
  so mapping propagates down chains in the same pass (DAG order).
- Nodes whose schema declared the input as `list<T>` (INPUT_IS_LIST
  translation, translate.py:1119-1125) consume lists natively and are
  never mapped by that input.

## 3. Lowering (per mapped node)

Replace GraphNode `N` (same node id, so existing `Link`s to it stay
valid) with:

- `RegionNode(kind="map")` with `binding="zip"` when the region has
  exactly one element port (lengths trivially equal; encodes nothing
  extra on the wire) and `binding="broadcast"` when it has two or
  more (upstream alignment semantics, section 3a).
- One ELEMENT port per list-driving scalar input: port type = the
  element type; region input = the original outer link.
- One plain (non-element) port per scalar-driven LINKED input,
  fed by the original outer link. Literal inputs stay inline on the
  body node untouched.
- Body: a one-node Graph containing `N` with lowered inputs rewired
  to `port(...)` references (construction template:
  tests/test_regions.py:203-218, tests/test_combinators.py:313-330).
- Outputs: `RegionOutput` gather per declared output, keyed by the
  SAME output ids, so downstream links keep resolving. Gather order
  is iteration order, matching upstream's per-invocation collection.
- Cache identity needs nothing: region iteration bindings flow into
  ordinary per-node cache keys (engine.py `_run_region` docstring).

## 3a. Native `binding="broadcast"` (additive, backend-owned)

New third `BindingMode` carrying upstream's alignment semantics as a
first-class native primitive (it is a legitimate general semantic,
not a compat-only shim):

- `model.py:72`: `BindingMode = Literal["zip", "cross", "broadcast"]`
  plus a docstring line in `RegionNode` mirroring the zip/cross ones.
- Engine binding-set computation (engine.py:1280-1310, new branch):
  iteration count = max element-list length; iteration `i` binds
  `ch[min(i, len(ch) - 1)]` per element port (repeat final element,
  byte-matching pinned upstream execution.py:157-317). ALL element
  lists empty -> zero iterations (same as zip). SOME empty while
  others are not -> loud runtime error naming the empty ports (there
  is no final element to repeat; upstream would IndexError here, so
  nothing is lost).
- wire.py: add "broadcast" to `BINDING_MODES`. Encoding already
  omits `binding` when it is "zip", so existing graphs are
  byte-unchanged and no wire version bump is needed; the value is
  additive. The coordinator notifies the frontend thread of the new
  enum value after landing (its decoder may validate binding values).
- validate.py: accept "broadcast" at :705 (unknown-binding-mode) and
  give it the same shape rules as zip (>=1 element port for map/fold;
  a while region declaring it is the same shape error as cross at
  :789).
- Native proving tests in tests/test_regions.py: unequal 2-vs-5
  repeat-last, 1-vs-N scalar-as-list promotion, equal lengths
  byte-identical to the zip result, mixed empty/non-empty loud error,
  all-empty zero iterations, while+broadcast shape diagnostic, wire
  round-trip of a broadcast region.

## 4. Mapped OUTPUT_IS_LIST flatten semantics

A mapped node that itself declares `OUTPUT_IS_LIST` on some output
must concatenate each invocation's list rather than gather a nested
`list<list<T>>`. The lowering exports that body output with
`RegionOutput(mode="flatten")`. The engine concatenates the per-iteration
lists in iteration order and exposes the body output's same `list<T>` type;
zero iterations produce an empty `list<T>`. Scalar body outputs continue to
use `mode="gather"`, which exposes `list<T>`. This matches ComfyUI's
OUTPUT_IS_LIST collection semantics and propagates the flattened effective
type through downstream mapping decisions.

## 5. Diagnostics (slice B)

- `validate.py`'s `list-into-scalar` diagnostic stays as the backstop
  for edges the pass deliberately skips (element-type-incompatible).
- Unequal lengths are NOT a diagnostic surface: broadcast alignment
  is silently faithful to upstream (which is also silent). The only
  runtime length error is the mixed empty/non-empty case (section
  3a), which upstream cannot execute either.
- The existing expand/blocker refusals (translate.py, f27c96d) reference
  docs/compat-porting-recipes.md (slice C) so refused users get a rewrite
  recipe, not a dead end. Slice C's alignment recipe
  documents zip-vs-broadcast semantics for native graph authors
  (compat users need no rewrite).

## 6. Proving tests

In tests/test_compat_prompt.py (translation shape) and an execution
test against the engine (fake v1 classes per house style in
tests/test_compat_comfy.py):

- Fan-out (`OUTPUT_IS_LIST` producer) -> unflagged scalar consumer ->
  `INPUT_IS_LIST` aggregator: consumer lowered to a map region,
  end-to-end element ORDER byte-stable, aggregator receives the
  gathered list.
- Chain propagation: two unflagged consumers in a row both lowered;
  the second's element port fed by the first's gather.
- Two equal-length lists into one unflagged node: one region, two
  element ports (broadcast binding), pairwise pairing verified.
- Unequal lengths: end-to-end upstream parity - 2-vs-5 repeats the
  final element, 1-vs-N promotes the singleton across all N
  iterations, results byte-match a hand-computed upstream
  transcription. Cache identity stays per-iteration stable.
- Literal inputs repeat identically per iteration.
- INPUT_IS_LIST consumer is NOT lowered (native list socket).
- Element-type-incompatible list edge is NOT lowered and still emits
  `list-into-scalar`/type diagnostics.
- Mapped OUTPUT_IS_LIST outputs flatten in invocation and element order,
  preserve `list<T>` cardinality, and feed whole-list consumers directly.
- Conformance corpus: mirror the list-semantics cases from ComfyUI
  tests/execution/testing_nodes/testing-pack (pinned 947c2749) that
  fall inside the covered surface, as compat fixtures; expansion/loop
  cases assert loud refusal (expected-refusal fixtures, never silent
  mistranslation).

## 7. Consumers / boundaries

Inference-thread ownership (packages/dinkster-inference*) untouched.
Frontend contract: no wire version bump; regions already carry. The
`binding="broadcast"` enum value is additive on the existing field
(section 3a) - the backend coordinator notifies the frontend thread
after it lands.
ROADMAP "Engine / execution" census-slice entry tracks status;
PROMISES.md row ships with the implementation commit.
