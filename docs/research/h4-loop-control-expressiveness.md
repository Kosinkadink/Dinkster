# H4 loop and control expressiveness proof

Date: 2026-07-31

Status: research only. This document does not choose or implement an
architecture.

## Scope and source pins

This proof compares the current Dinkster selector and region model with the
demand-driven behavior used by ComfyUI and ltdrdata's Impact/Inspire pack
ecosystem. The binding runtime case is:

1. A normal upstream node computes a boolean during execution.
2. That boolean selects one of two expensive model-producing cones.
3. Only the selected model cone may execute any node body.

The code-read pins used here are:

- Dinkster `8bb753d5ffbd1e513c795a1fd64204d24f5ac9e4` at the start of the
  investigation.
- ComfyUI catalog `e651b7bef55a5376343dcb1c0edb79f0142c985e`, read-only at
  `/home/kosin/ComfyUI-catalog`.
- ComfyUI-Impact-Pack `429d0159ad429e64d2b3916e6e7be9c22d025c3c` at
  `/home/kosin/node-analysis/clones/comfyui-impact-pack`, code read only.
- ComfyUI-Inspire-Pack `d23db9aa544de9a6d4c609cb7005fa9e0d42031d` at
  `/home/kosin/node-analysis/clones/comfyui-inspire-pack`, code read only.

The Inspire pin is included because the current Impact Pack has no
GraphBuilder loop-open/close pair. The pair commonly associated with the
same ltdrdata ecosystem is Inspire Pack's `ForeachListBegin` and
`ForeachListEnd`. Attributing those classes to Impact Pack would be
incorrect. Current Impact Pack's historical loop facility instead combines
logic nodes, output-node events, and repeated queue execution.

## Executed selector refusal surfaces

Dinkster's `SelectorSpec` is a lowering instruction, not a runtime control
primitive. `lower_selectors` reads the selector input directly from stored
`GraphNode.inputs`: a `Link` yields `prompt.computed_selector`, and anything
whose exact type is not `bool` yields `prompt.bad_selector_value`
(`packages/dinkster-graph/src/dinkster_graph/lower.py:114-130`).

The following command constructs both minimal graphs and calls the real
lowerer. It then calls the real `JobQueue.submit` guard with both a top-level
selector and a selector nested in a `RegionNode` body. `object.__new__` is
used only to avoid starting a queue dispatcher; the executed method and
assertion are the production method.

```console
$ .venv/bin/python - <<'PY'
from types import SimpleNamespace
from dinkster_graph import Graph, GraphNode, Link, RegionNode, lower_selectors
from dinkster_schema import InputSpec, NodeSchema, OutputSpec, SelectorSpec, TypeExpr
from dinkster_server.queue import JobQueue

integer = TypeExpr.concrete("core.int")
switch = NodeSchema(
    "proof.switch",
    inputs=(
        InputSpec("switch", TypeExpr.concrete("core.boolean")),
        InputSpec("false_value", integer),
        InputSpec("true_value", integer),
    ),
    outputs=(OutputSpec("out", integer),),
    selector=SelectorSpec(
        "switch", {"false": "false_value", "true": "true_value"}
    ),
)
source = NodeSchema("proof.source", outputs=(OutputSpec("out", integer),))
schemas = {schema.node_type: schema for schema in (switch, source)}

cases = {
    "computed": Graph(
        {
            "bool_source": GraphNode("proof.source"),
            "selector": GraphNode(
                "proof.switch",
                {
                    "switch": Link("bool_source", "out"),
                    "false_value": 0,
                    "true_value": 1,
                },
            ),
        }
    ),
    "bad_value": Graph(
        {
            "selector": GraphNode(
                "proof.switch",
                {"switch": 1, "false_value": 0, "true_value": 1},
            )
        }
    ),
}
for name, graph in cases.items():
    result = lower_selectors(graph, (), schemas)
    print(name, result.problems)

queue = object.__new__(JobQueue)
queue._closed = False
runtime = SimpleNamespace(schemas=schemas)
for name, graph in {
    "top_level": cases["computed"],
    "region_body": Graph(
        {"region": RegionNode(kind="map", body=cases["bad_value"])}
    ),
}.items():
    try:
        queue.submit("client", "job", graph, (), execution=runtime)
    except AssertionError as exc:
        print(name, f"AssertionError: {exc}")
PY
computed (LoweringProblem(code='prompt.computed_selector', message='selector input must be a stored boolean, not a link', node_id='selector', input_id='switch'),)
bad_value (LoweringProblem(code='prompt.bad_selector_value', message='selector input must be a JSON boolean', node_id='selector', input_id='switch'),)
top_level AssertionError: JobQueue.submit accepts lowered graphs only
region_body AssertionError: JobQueue.submit accepts lowered graphs only
```

The queue assertion is recursive: `contains_selector` checks each top-level
`GraphNode` and recursively checks every `RegionNode.body`, then asserts that
none remain (`packages/dinkster-server/src/dinkster_server/queue.py:197-208`). The
queue therefore cannot serve as a fallback runtime for either refusal.
Selectors inside region bodies have an earlier dedicated refusal,
`prompt.selector_in_region` (`packages/dinkster-graph/src/dinkster_graph/lower.py:25-47`).

### Stored boolean behavior is compile-time pruning

The stored-boolean case does provide strong non-execution, but only because
lowering deletes graph structure before submission. This executed example
uses two model-producing branches and selects the true branch:

```console
$ .venv/bin/python - <<'PY'
from dinkster_graph import Graph, GraphNode, Link, lower_selectors
from dinkster_schema import InputSpec, NodeSchema, OutputSpec, SelectorSpec, TypeExpr

model = TypeExpr.concrete("proof.model")
switch = NodeSchema(
    "proof.switch",
    inputs=(InputSpec("switch", TypeExpr.concrete("core.boolean")),
            InputSpec("false_model", model), InputSpec("true_model", model)),
    outputs=(OutputSpec("model", model),),
    selector=SelectorSpec("switch", {"false": "false_model", "true": "true_model"}),
)
source = NodeSchema("proof.expensive_model", outputs=(OutputSpec("model", model),))
sink = NodeSchema("proof.sink", inputs=(InputSpec("model", model),))
schemas = {x.node_type: x for x in (switch, source, sink)}
graph = Graph({
    "false_loader": GraphNode(source.node_type),
    "true_loader": GraphNode(source.node_type),
    "selector": GraphNode(switch.node_type, {
        "switch": True,
        "false_model": Link("false_loader", "model"),
        "true_model": Link("true_loader", "model"),
    }),
    "sink": GraphNode(sink.node_type, {"model": Link("selector", "model")}),
})
result = lower_selectors(graph, ("sink",), schemas)
print("problems", result.problems)
print("nodes_before", tuple(graph.nodes))
print("nodes_after", tuple(result.graph.nodes))
print("sink_input", result.graph.nodes["sink"].inputs["model"])
PY
problems ()
nodes_before ('false_loader', 'true_loader', 'selector', 'sink')
nodes_after ('true_loader', 'sink')
sink_input Link(node_id='true_loader', output_id='model')
```

`_cone` walks backward from the inactive branch link through ordinary input
links (`lower.py:50-59`). `_prune` removes inactive candidates only when they
are neither targets nor still referenced (`lower.py:62-76`). The selector is
deleted and its consumers are rewired before this pruning (`lower.py:140-151`).
Consequently, an inactive cone used elsewhere remains. This is correct static
reachability, not runtime branch activation.

It cannot extend to an execution-time boolean: choosing which cone to pass to
`_cone` requires the boolean value at lowering time. Running the boolean node
during lowering would collapse document transformation and execution, and is
not supported by the current pure graph layer. Leaving both links in the
lowered graph also does not defer the choice: normal planning recursively
includes every linked dependency of a target (`packages/dinkster-graph/src/dinkster_graph/plan.py:14-44`),
and the ready-set engine dispatches every dependency once ready
(`packages/dinkster-engine/src/dinkster_engine/engine.py:1120-1186`).

## Runtime selector parity scenario

Consider this graph, where `probe_boolean` computes its result from an image
or model inspection at execution time:

```text
probe_input -> probe_boolean -----> conditional_model -> consumer
                                  /                 \
false_source -> expensive_model_A                   (selected model)
true_source  -> expensive_model_B
```

The contract under examination is not "compute both models, then discard
one." It is:

1. Run the non-lazy upstream cone of `probe_boolean`.
2. Observe its boolean output.
3. Demand exactly one of the two model-producing cones.
4. Run `consumer` with that result.
5. Invoke zero node bodies in the other model cone.

### Pinned ComfyUI demand order

At the pinned ComfyUI catalog revision, this order follows directly from the
execution machinery:

- Execution begins from explicit output targets. Each target is added to an
  `ExecutionList` (`/home/kosin/ComfyUI-catalog/execution.py:777-780`).
- `TopologicalSort.add_node` traverses normal links but omits links whose
  destination input metadata has `lazy=True`
  (`/home/kosin/ComfyUI-catalog/comfy_execution/graph.py:138-166`). Thus the
  condition, a normal input, enters the initial dependency cone, while the
  two lazy model inputs do not.
- When the conditional node becomes ready, missing lazy links appear as
  missing inputs. The executor calls the node's `check_lazy_status`, filters
  its returned input names, promotes each requested input to a strong link,
  and returns `PENDING` (`/home/kosin/ComfyUI-catalog/execution.py:492-520`).
- `make_input_strong_link` adds the selected source, and `add_strong_link`
  recursively adds that source's non-lazy cone
  (`/home/kosin/ComfyUI-catalog/comfy_execution/graph.py:120-166`). The
  unrequested lazy link never enters `pendingNodes`.
- After the dynamically requested cone completes, the staged conditional
  runs again with the selected value available. `ExecutionList` explicitly
  supports returning a staged node after adding dependencies
  (`/home/kosin/ComfyUI-catalog/comfy_execution/graph.py:193-197`).

This is why the boolean cone runs first and only the selected expensive cone
is demanded. Caching may satisfy a demanded source without invoking its body,
but it does not cause the unselected source to become demanded.

### Impact Pack switches

Impact Pack registers `ImpactSwitch` as `GeneralSwitch`
(`/home/kosin/node-analysis/clones/comfyui-impact-pack/__init__.py:199-203`).
Its arbitrary-type dynamic inputs are lazy, `select` defaults to
`select_on_execution`, and the tooltip explicitly describes dynamic execution
selection (`modules/impact/util_nodes.py:15-45`). `check_lazy_status` converts
the one-based integer selection to `inputN` and requests only that input
(`util_nodes.py:50-59`); `doit` returns that input plus its label and index
(`util_nodes.py:61-87`).

`ImpactConditionalBranch` marks both `tt_value` and `ff_value` lazy
(`/home/kosin/node-analysis/clones/comfyui-impact-pack/modules/impact/logics.py:63-71`).
Its `check_lazy_status` asks for `tt_value` only when `cond` is true and
`ff_value` only when false; `doit` returns the same selected value
(`logics.py:79-89`). These nodes use ComfyUI's lazy-input machinery. They do
not use `GraphBuilder`.

### Loop taxonomy and expansion

Current Impact Pack contains no loop-open/close class and imports no
`GraphBuilder`. Its loop-oriented behavior is prompt-lifecycle control:

- `ImpactConditionalStopIteration` is an `OUTPUT_NODE` which emits a
  `stop-iteration` server event when true
  (`/home/kosin/node-analysis/clones/comfyui-impact-pack/modules/impact/logics.py:203-220`).
- `ImpactQueueTrigger` is an `OUTPUT_NODE` which emits `impact-add-queue` to
  enqueue another prompt execution (`logics.py:431-451`).
- `ImpactListBridge` gathers a list before forwarding it, ensuring the prior
  list subworkflow has completed (`logics.py:751-771`).

The open/close GraphBuilder pair in the ltdrdata ecosystem is instead Inspire
Pack's `ForeachListBegin`/`ForeachListEnd`, registered at
`/home/kosin/node-analysis/clones/comfyui-inspire-pack/inspire/list_nodes.py:252-257`.
The begin node splits out the next item, remainder, and accumulator
(`list_nodes.py:82-122`). The end node returns the accumulator when no items
remain (`list_nodes.py:169-179`). Otherwise it discovers the nodes between
open and close (`list_nodes.py:149-167,181-191`), clones and reconnects that
contained graph with `GraphBuilder` (`list_nodes.py:193-209`), feeds the
remainder and current accumulator to the cloned open node
(`list_nodes.py:211-217`), and returns `expand: graph.finalize()`
(`list_nodes.py:219-222`).

Pinned ComfyUI accepts such expansion results by adding ephemeral nodes to
the dynamic prompt, adding expanded output nodes as targets, and adding
expanded result links as strong dependencies before returning the parent
node to `PENDING` (`/home/kosin/ComfyUI-catalog/execution.py:580-613`). This
is runtime graph expansion, a different mechanism from lazy branch demand
and from Impact Pack's repeated queue execution.

Dinkster's current typed regions already express bounded structural repetition:
`RegionNode.kind` is closed to `map`, `fold`, and `while`; map iterations are
independent, fold chains state, and while checks a body-produced boolean with
a mandatory cap (`packages/dinkster-graph/src/dinkster_graph/model.py:75-127`). The
engine runs map iterations concurrently, fold sequentially, and while until
false or its cap (`packages/dinkster-engine/src/dinkster_engine/engine.py:1401-1452`).
This covers a substantial loop outcome without GraphBuilder, but selectors
inside those bodies are explicitly refused and runtime conditional demand is
not represented.

## Zero-execution is an observable

The acceptance criterion for an unselected branch must be measured at the
node-body boundary. A proving harness should use a cold cache and a recording
worker or invocation wrapper which appends `(run_id, node_id, node_type)`
immediately before each worker node body call. Given named disjoint branch
cones `A` and `B`, it should assert:

1. The condition cone has expected body entries before branch activation.
2. The selected cone has its expected body entries, or separately recorded
   cache hits if the test intentionally covers warm-cache behavior.
3. Every node id unique to the unselected cone has body-entry count exactly
   zero.
4. The unselected cone has no `node_started` event and no cache lookup or
   resource admission if the chosen architecture promises that stronger
   form of non-demand.
5. A deliberate exception node in the unselected cone cannot fail the run,
   while the same exception node fails when its cone is selected.

Dinkster already exposes useful corroborating data: `RunResult.executed` records
fresh body completions and `cached` records cache use
(`packages/dinkster-engine/src/dinkster_engine/engine.py:188-208`); `node_started`
is emitted immediately before Invocation construction
(`engine.py:1009-1049`), and a fresh completion appends to `executed`
(`engine.py:1088-1103`). However, the primary proof should be the worker-side
body-entry recorder. An engine event alone could be emitted too early or too
late, and `executed == ()` alone is ambiguous with a cache hit.

Three concepts must remain distinct:

- **Demand-driven branch non-execution:** the inactive cone never enters the
  demanded execution graph. Its body count is zero because no work was
  requested.
- **Lazy input evaluation:** an input edge is initially weak. A node callback
  can promote a named input and its upstream cone after ordinary inputs are
  known. ComfyUI's `check_lazy_status` is one implementation of demand-driven
  activation; "lazy" by itself is only edge metadata.
- **Dinkster ABSENT propagation:** a value envelope says that a producer yielded
  no value. Consumers then apply `fail`, `omit`, `accept`, or `skip` policy.
  A skip avoids that consumer's invocation and emits `node_skipped`, but the
  producer and all normally linked dependencies were already planned
  (`packages/dinkster-engine/src/dinkster_engine/engine.py:723-770,885-915`). ABSENT
  therefore cannot prove that an unselected model cone was never demanded.
  Region gather/flatten also propagates an absent child to an absent whole
  collection (`engine.py:1454-1490`), which is value semantics rather than
  branch activation.

## Candidate demand-driven primitives

The following are candidates for adjudication, not recommendations. None is
implemented by this research.

### 1. Runtime-selector region kind

**Grammar movement:** Add a fourth `RegionKind`, branch bodies or branch entry
links, a boolean selector source, and a typed result interface. Define whether
each branch is a complete nested `Graph` or a set of activation roots.

**Ownership:** Graph owns stored structure and validation. Engine owns
condition-first execution and activation. Graph wire serialization owns the
new region form. Protocol changes are needed only if workers receive a region
or activation unit rather than ordinary per-node invocations.

**Tradeoffs and blast radius:** This keeps control explicit and can compose
with typed regions, but changes model, wire, validation, planning, nested path
grammar, frontend authoring, and compat lowering. Separate branch graphs make
side-effect and target containment explicit but complicate links shared by
both branches. A region kind also risks conflating one-shot choice with the
existing repetition abstraction.

### 2. Engine-level demand scheduling over conditional edges

**Grammar movement:** Add conditional or weak edge metadata to Graph/InputSpec,
plus a selector node contract that maps a runtime value to a set of demanded
input ids. Keep branch producers as normal graph nodes.

**Ownership:** Schema or graph declares the weak edges; engine owns the
two-phase scheduler and only activates selected dependencies. Protocol can
remain unchanged if the engine itself evaluates a built-in selector and then
issues ordinary invocations.

**Tradeoffs and blast radius:** This is close to ComfyUI's topology behavior
without a worker callback, and can preserve ordinary node cache entries. It
moves the planner from a fixed target cone to a mutable demand graph and
therefore touches validation, deadlock/cycle detection, cancellation,
single-flight, resource preparation, event ordering, and run reporting. A
built-in selector contract may be less extensible but has a smaller trust
boundary than arbitrary workers deciding topology.

### 3. Deferred subgraph submission or expansion

**Grammar movement:** Define a typed graph fragment/result, legal links into
and out of it, deterministic runtime node ids, expansion limits, and an
activation result. A selector could submit only the chosen stored fragment;
loop-close nodes could submit the next iteration fragment.

**Ownership:** Graph/protocol own the fragment representation and validation.
Engine owns admission, snapshot identity, recursive planning, and joining the
deferred result. Workers either construct fragments or choose among pre-stored
ones.

**Tradeoffs and blast radius:** This is expressive enough for Inspire-style
GraphBuilder loops and deferred branches, but it has the largest grammar and
security surface. Dynamic topology affects fingerprints, diagnostics,
resource limits, cancellation, extension snapshots, isolated-worker trust,
node-id stability, and replay determinism. Deferring a whole JobQueue job is
not sufficient unless parent/child cache, cancellation, outputs, and errors
gain a defined join contract.

### 4. `check_lazy_status`-style worker callback

**Grammar movement:** Mark selected inputs lazy, allow an invocation with
ordinary inputs present and lazy inputs missing, and define a callback response
containing demanded input ids. The engine then promotes those links and later
performs the full invocation.

**Ownership:** Schema declares lazy inputs. Protocol owns partial-input request
and demand-response frames. Worker owns demand selection. Engine owns the
mutable dependency graph and validates that requests name declared linked lazy
inputs.

**Tradeoffs and blast radius:** This most directly ports current Impact switch
semantics and supports custom selectors. It requires at least two worker
interactions, including isolated hosts, and lets extension code influence
scheduling. It must define callback purity, timeout/error behavior, cache
identity for callback results, repeated requests, cycles, capability limits,
and whether callbacks may allocate resources. It also broadens every worker
implementation and protocol compatibility path.

## Cross-cutting interaction matrix

### Caching

Current Dinkster node cache identity is schema signature, input fingerprints,
and execution selection, shared across iterations and regions
(`packages/dinkster-engine/src/dinkster_engine/engine.py:870-923`). A conditional
primitive must specify whether:

- the selector's key contains the condition and only the demanded branch
  value, not an unavailable unselected value;
- branch nodes retain their ordinary independent keys;
- a changed condition reuses an already-cached newly selected branch while
  still performing zero body calls in the newly unselected branch;
- runtime expansion identity includes the stored fragment, expansion path,
  iteration, and extension snapshot without making equivalent branch work
  unnecessarily miss;
- cache lookup for an unselected node counts as demand. The strongest
  zero-demand observable says no, while the required zero-body criterion alone
  permits a lookup that does not invoke.

### Side effects and output nodes

An unselected branch must not run side-effecting nodes. `OUTPUT_NODE` cannot
override that rule merely because it is globally target-like. In compat prompt
extraction, every schema with `output_node` becomes a target
(`packages/dinkster-compat-comfy/src/dinkster_compat_comfy/prompt.py:421`), while
the schema documentation says output-node status is a target hint, not
scheduling semantics (`packages/dinkster-schema/src/dinkster_schema/model.py:1074-1079`).

A branch-contained save/preview node therefore needs one of two explicit
models: it is a target only after its branch is activated, or it remains a
global target and is consequently outside branch non-execution. Treating it
as both globally targeted and unselected is contradictory. Inspire expansion
currently adds expanded output nodes as targets (`ComfyUI-catalog/execution.py:593-609`),
but only after the expansion itself is demanded.

### Output targets inside branches

If a branch result is consumed by an outer target, branch activation can be
rooted at the selected result. If a branch contains its own output target,
the grammar must preserve branch membership so target discovery is scoped.
Merely leaving both branch output nodes at the top level causes normal
output-driven reachability to demand both. A runtime-selector region naturally
contains them; weak top-level edges need an additional activation membership
rule.

### Failure behavior

An error in the selected condition cone or selected branch must fail the run
through the normal node failure path. An error in the unselected cone must be
unobservable because no body is invoked. Validation errors are a separate
question: structural validation may still reject an invalid unselected branch
before execution, just as a program can reject an unreachable ill-typed arm.
The contract must decide whether missing artifacts, worker preparation, and
resource admission are validation/demand-time effects. Current Dinkster prepares
all node types in the fixed plan before dispatch
(`packages/dinkster-engine/src/dinkster_engine/engine.py:1581-1591`), so a future
strong zero-demand contract may need preparation to follow activation too.

When parallel selected work fails, current `TaskGroup` behavior cancels sibling
work and surfaces the first `ExecutionError` (`engine.py:1178-1186`). Dynamic
activation must retain a deterministic rule for failures racing in the same
selected cone.

### Nesting with loops

The required compositions are:

- A selector inside `map`: evaluate a condition per iteration; only that
  iteration's selected branch runs. Parallel iterations may choose different
  branches, and occurrence ids must make recorder and cache evidence distinct.
- A selector inside `fold`: each iteration's condition can depend on prior
  state. Branch choice must occur after state arrives and before branch bodies.
- A selector inside `while`: the branch may compute state or the continue
  boolean. Maximum-iteration enforcement remains outside branch semantics.
- A `map`, `fold`, or `while` inside either selector branch: no iteration or
  region-body node in the unselected region may execute or prepare resources.
- A GraphBuilder-style recursive loop around a selector: each expanded
  iteration makes a fresh demand decision while retaining deterministic
  expansion ids and bounded growth.
- A selector around a GraphBuilder-style loop: the unselected loop must create
  no ephemeral nodes, not merely create them and skip their bodies.

Current Dinkster recursion already gives each region iteration the same scheduler,
cache, admission lanes, and pins (`engine.py:1120-1136`), which is useful for
composition. It does not supply conditional activation, and
`prompt.selector_in_region` proves that current static selectors cannot fill
that role.

## OPEN ADJUDICATION QUESTIONS

1. Is the required primitive limited to a typed boolean two-way selector, or
   must it support ImpactSwitch-style integer selection and extension-defined
   lazy input choice in the first contract?
2. Is the acceptance bar only zero node-body entries in the unselected cone,
   or the stronger zero-demand bar of no cache lookup, worker preparation,
   artifact preflight, admission, or resource pinning for that cone?
3. Should branch structure be explicit nested graphs/regions, or remain normal
   top-level nodes connected by weak conditional edges?
4. Does worker or extension code get authority to request lazy dependencies,
   or must activation be an engine-interpreted closed selector contract?
5. Are output nodes inside an inactive branch scoped targets, forbidden, or
   still global targets that intentionally defeat branch inactivity?
6. Must structurally invalid or artifact-missing unselected branches fail
   before execution, or should some checks occur only after demand?
7. Which cache identity is canonical for a selector: condition plus selected
   value, a branch identity token, or the complete stored two-branch shape?
8. Should cached unselected branch values be ignored without lookup, and what
   recorder evidence is required to distinguish that from a warm-cache hit?
9. Is Impact Pack's repeated-queue loop lifecycle in scope, or only typed
   in-run map/fold/while and Inspire-style GraphBuilder expansion?
10. If dynamic subgraph expansion is admitted, what deterministic node-id,
    depth, size, cancellation, and capability limits form the minimum safe
    grammar?
11. For selectors nested in map/fold/while, is activation per iteration, and
    may different map iterations demand different branch types concurrently?
12. When a selector branch contains a region, must worker preparation and
    artifact resolution also wait until that branch is selected?
13. Does branch validation require both arms to have one identical concrete
    output type, or may wildcard/union typing preserve Impact's arbitrary-type
    behavior?
14. Which primitive, if any, should also serve loop-close recursion rather
    than keeping conditional demand and repetition as separate mechanisms?
