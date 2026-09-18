# List execution + graph expansion usage: ecosystem conclusions

Directly-referenceable conclusions of the 2026-07-30 feature-usage
census (user directive). Full data, scanner, spec, per-pack
classifications, and the detailed report live in the private
[Kosinkadink/dinkster-research](https://github.com/Kosinkadink/dinkster-research)
repository: `feature-usage/LISTEXP-REPORT.md` at `54f8302`,
`feature-usage/LISTEXP-SCAN-SPEC.md` + `feature-usage/results-listexp/`
(mechanical batches at `43168bf`, classification waves at `b65338a`,
`9f07fa4`, `2012b87`, `f197283`, `e2ee873`). Method: mechanical token
scan over the 747-pack usage-weighted roster (745 covered; registry-CDN
recovery of 9 deleted repos), then code-read classification of all 195
flagged packs (1172 usage rows) by five delegate threads, every wave
coordinator-verified by byte-identical aggregate recomputation. Core
ground truth pinned at ComfyUI `947c2749`. Denominator: 514,802
weighted users.

## Conclusions

1. **List semantics are the second-widest ecosystem feature measured
   so far: 38.1 percent of user-weighted installs.** Fan-out list
   production (L2, 34.2 percent, 132 packs) and whole-list
   aggregation (L1, 21.6 percent, 79 packs) dominate; forward-only
   plumbing (L4) is 6.2 percent.
2. **The load-bearing implicit behavior is single-list mapping, and
   it is Dinkster compat's one real feature-set regression risk.** L2's
   entire value is that a produced list drives per-element invocation
   of downstream unflagged nodes. Dinkster translates declared flags
   into `list<T>` sockets, but a list output wired into an unflagged
   scalar input dies as list-into-scalar. Porting ComfyUI workflows
   requires lowering such edges into the explicit map region
   (`RegionKind "map"`, dinkster-graph model.py) mechanically.
3. **Zero packs author against implicit multi-list zip/broadcast
   (L3 = 0 in 1172 rows).** Every multi-list consumer declares
   INPUT_IS_LIST and aligns explicitly. ComfyUI's
   longest-length-zip-with-last-element-broadcast contract needs NO
   emulation; Dinkster's equal-length `zip` / cartesian `cross` region
   bindings stand. Caveat: workflows (not pack code) can still hit
   executor broadcast by wiring unequal lists into unflagged nodes -
   a loud refusal with a porting hint is the sanctioned disposition.
4. **Blocker-style gating is the dominant real "expansion" feature
   (E3, 10.2 percent, 34 packs) and it is absence semantics, not
   expansion.** Most E3 rows gate downstream execution with
   ExecutionBlocker and generate no graph. Dinkster's absent envelopes +
   per-input `on_absent` policies (skip/omit/accept/fail) are the
   matching primitive; compat refuses raw blockers loudly since
   `f27c96d`. The port recipe is blocker-producer -> absent-producer.
5. **Genuine runtime graph generation is small and splits cleanly.**
   Loops via recursive expansion (E1) are 4.1 percent / 10 packs;
   runtime subgraph compilation (E2) 5.8 percent / 9 packs. Of 24
   non-infrastructure graph-emitting rows, 12 are
   workflow-deterministic (covered by extension-design.md 3.8
   deterministic plan compilation when it ships), 9 are
   runtime-data-dependent or run user code, 3 mixed. The
   data-dependent remainder maps to native map/while regions by
   rewrite, not compat emulation. Pack-shipped expansion frameworks
   (E4) are 0.1 percent - out of scope.
6. **Core ComfyUI ships list nodes but zero expansion nodes.** At the
   pin: 15 `is_input_list` + 25 `is_output_list` production
   declarations (all v3) across 7 comfy_extras files; no shipping
   core node returns `"expand"` or uses ExecutionBlocker. Core-node
   parity is a list-nodes problem only.
7. **Token hits usually mean real usage.** Only 1.8 percent of user
   weight sits in packs whose every hit was a false positive, though
   per-row false positives are common (LD 139 rows / ED 30 rows:
   default-value flag spellings, comments, dead nodes, vendored
   copies, an input literally named "expand").

## Capability / gap matrix

| ComfyUI feature | Real weight | Dinkster today | Disposition |
|---|---|---|---|
| INPUT_IS_LIST whole-list consumption | 21.6% (L1) | compat lowers declared flags to `list<T>` sockets | covered |
| OUTPUT_IS_LIST list production | 34.2% (L2) | `list<T>` socket produced | covered at the socket; consumption gap below |
| Implicit per-element mapping of one list into unflagged nodes | carrier of L2 | list-into-scalar refusal | GAP: compat map-region lowering (slice A) |
| Mapped-node output flatten/concat (OUTPUT_IS_LIST under mapping) | part of L2 | gather collects per-iteration into `list<T>` | explicit flatten in lowering (slice A) |
| Implicit multi-list longest-zip + last-element broadcast | 0% (L3) | equal-length zip / cross bindings | sanctioned refusal + porting diagnostic (slice B) |
| ExecutionBlocker gating | 10.2% (E3) | absent envelopes + on_absent policies; compat refuses raw blockers (f27c96d) | covered semantically; porting recipe doc (slice C) |
| Deterministic subgraph compilation | subset of E2 (5.8%) | extension-design.md 3.8 plan compilation (designed, not shipped) | covered by design; port path when 3.8 ships |
| Runtime-data-dependent expansion, loops | 4.1% (E1) + E2 remainder | native map/while regions | rewrite-on-port, no compat emulation; loud refusal stands |
| Pack-shipped expansion frameworks | 0.1% (E4) | none | out of scope, refuse |
| Core comfy_extras list nodes | 15+25 declarations, 7 files | not yet ported | port with native equivalents (slice D) |

## Recommended slices

- **A. Compat map-region lowering.** When a translated `list<T>`
  output feeds a scalar input of a non-list-declared node, lower the
  edge into a map region wrapping the consumer (zip binding over the
  single list; gather outputs; explicit flatten when the consumer
  itself declares OUTPUT_IS_LIST). Proof: ported two-node
  fan-out->per-element workflows byte-matching ComfyUI element order.
- **B. Broadcast refusal diagnostic.** Detect multi-list-into-scalar
  and unequal-length wiring at compat lowering time; refuse with a
  message naming the L3 census result and the explicit map/zip
  rewrite. Never silently truncate or broadcast.
- **C. Blocker porting recipe.** Document blocker-producer ->
  absent-producer rewrite (docs/comfy-porting); compat refusal
  message points at it.
- **D. Core list-node parity.** Port/verify the 7 comfy_extras
  list-node files' semantics against native list sockets + regions.

Consumers: ROADMAP "Engine / execution" (census-derived compat list
work, added with this document); extension-design.md 3.7/3.8
(expansion analog confirmed correctly scoped by census);
packages/dinkster-compat-comfy translate.py refusal messages.
