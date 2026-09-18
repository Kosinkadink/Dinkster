# Compat porting recipes

Dinkster refuses ComfyUI execution behavior that cannot be translated
faithfully. These recipes show the corresponding explicit Dinkster constructs
for pack and workflow authors.

## ExecutionBlocker to Dinkster absence semantics

ComfyUI nodes may return an `ExecutionBlocker` to stop values from flowing
through downstream nodes. Dinkster represents the same conditional-gating
intent as an optional output that produces a `core.absent` value. The
downstream input declares how absence is handled with `on_absent`:

- `"skip"` (the default for required inputs) skips the downstream node and
  propagates absence through its outputs.
- `"omit"` (the default for optional inputs) calls the downstream node
  without that input.
- `"accept"` calls the downstream node with `None`.
- `"fail"` fails the run and names the absence origin.

Before, a ComfyUI node blocks its output when the condition is false:

```python
def gate(self, value, enabled):
    if not enabled:
        return (ExecutionBlocker("disabled"),)
    return (value,)
```

After, a Dinkster-native node declares an optional output and produces an
absent value. A required downstream input already has `on_absent="skip"`
by default; spell another policy explicitly when that is the intended
behavior.

```python
from dinkster_api.v1 import (
    AbsentOutput,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)


class Gate(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="example.gate",
            inputs=(
                InputSpec("value", TypeExpr.concrete("core.int")),
                InputSpec("enabled", TypeExpr.concrete("core.boolean")),
            ),
            outputs=(
                OutputSpec("value", TypeExpr.concrete("core.int"), optional=True),
            ),
        )

    @classmethod
    def execute(cls, *, value, enabled):
        return cls.outputs(
            value=value if enabled else AbsentOutput("disabled"),
        )
```

This is a source-porting recipe, not automatic compat emulation. Raw
`ExecutionBlocker` payloads remain a loud compat refusal.

## Native list alignment: zip versus broadcast

ComfyUI implicitly maps an unflagged node over linked list inputs. When the
lists have different lengths, it repeats each shorter list's final element
until the longest list is exhausted. Compat translation preserves this
behavior: a mapped node with one element input uses `binding="zip"`, while a
mapped node with two or more element inputs uses `binding="broadcast"`.
Compat workflow users do not need to insert an alignment node.

Native graph authors choose the alignment policy explicitly:

- `binding="zip"` requires all element lists to have equal length.
- `binding="broadcast"` iterates to the longest list and repeats each
  shorter list's final element. Two lists `[1, 2, 3]` and `[10]` therefore
  bind `(1, 10)`, `(2, 10)`, and `(3, 10)`.
- `binding="cross"` binds the cartesian product instead of aligning by
  position.

All-empty broadcast inputs produce zero iterations. A mix of empty and
non-empty element lists fails loudly because an empty list has no final
element to repeat. Use an explicit alignment node before a zip map only when
the intended policy is padding, truncation, cycling, or something other than
repeat-final-element broadcast.

When an implicitly mapped ComfyUI node declares `OUTPUT_IS_LIST`, compat
preserves ComfyUI's flattening behavior: each invocation returns a list and
the map region concatenates those lists in invocation order into one typed
list. Scalar outputs from the same mapped node are gathered normally. Pack
and workflow authors do not need a flatten rewrite for this shape.

## Graph expansion dispositions

ComfyUI's `expand` result can create runtime graph structure. The port
depends on what determines that structure:

- Workflow-deterministic expansion, such as a fixed-count loop, ports to a
  statically unrolled graph or an explicit map region.
- Runtime-data-dependent expansion ports to native `while` or `fold`
  regions when the algorithm can be expressed by their bounded state and
  iteration contracts.

The available region kinds are the `RegionKind` definition in
`packages/dinkster-graph/src/dinkster_graph/model.py`: `map`, `fold`, and `while`.
Compat does not currently translate `expand` payloads, execute arbitrary
runtime-generated subgraphs, or provide native plan compilation for all
workflow-deterministic expansion. Such payloads continue to refuse loudly;
pack authors must rewrite them using supported static graphs or regions.

## Accept-all inputs require a declared family

Compat does not expose arbitrary `accept_all_inputs` kwargs. The one core
exception is ComfyUI's exact unnamespaced V3 `CustomCombo`: its frontend-authored
literal `option1..N` keys become members of a closed `options` family, bounded at
100 strings and required to be contiguous. `choice` and optional `index` remain
ordinary declared inputs. A custom-pack node or a changed `CustomCombo` shape
still refuses.

Pack authors should replace predictable extra keys with an `InputFamilySpec`
whose member vocabulary and bounds describe every accepted key. Runtime-open
names need a separately designed document-state contract; `**kwargs` alone is
not a schema.

## VHS meta-batching to region-windowed loading

VideoHelperSuite's `VHS_BatchManager` processes a long video in chunks of
`frames_per_batch`: it holds the load and encode generators open across
executions and auto-requeues the workflow until the loader is exhausted, so
only one chunk of frames occupies RAM at a time. Dinkster expresses the same
chunked processing with a region over frame windows; no requeue mechanism
exists or is needed, because the region executes every window within one
run. Use a `fold` region to match VHS's strictly sequential
one-window-at-a-time residency; `map` dispatches windows concurrently, which
trades peak memory for throughput.

- Window list: compute per-window `start_time` values as a `list<core.float>`
  (for example `dinkster.curve.sample` over `[0, duration]`, or list and math
  nodes over a window index list). The source duration must be supplied by
  the author: `dinkster.load_video` reports the duration of the frames it
  loaded, not the container, and no node probes source metadata without
  decoding frames. A trailing window whose `start_time` is past the end of
  the source refuses ("video selection contains no frames"), so size the
  window list from the real duration rather than overshooting.
- Windowed load: inside the region body, `dinkster.load_video` takes the bound
  `start_time` plus a fixed `frame_load_cap` (and optional
  `select_every_nth`/`force_rate`). Each binding seeks the container and
  retains only its own window's frames, so each individual load stays under
  the 512 MiB decoded-frames limit no matter how long the source is. Under
  `map`, several windows may be decoded at once; under `fold`, windows are
  decoded one at a time, like the VHS construct (the node cache may retain
  recent results until eviction).
- Per-window processing and gather: the body processes its window and the
  region gathers outputs in index order. Gather latent-space windows and join
  them with the latent concat nodes, or assemble per-window clips with
  `dinkster.video.assemble` and gather a `list` of encoded videos. Gathered
  outputs from every window are held until the region completes, so prefer
  compact per-window outputs (latents or encoded clips) over raw frame
  batches when the window count is large.

Two behavioral differences from the ComfyUI construct:

- Temporal quality: like VHS meta-batching, each window is a hard boundary
  for anything computed inside the region body. For denoising smoothness
  across windows, use sampling context windows (the engine-level windowed
  denoising mechanism), which meta-batching cannot express.
- Single-file output ceiling: `dinkster.save_video`/`dinkster.video.assemble`
  accept one materialized frame batch bounded by the 512 MiB input and
  1 GiB encoded output limits, while `VHS_VideoCombine` under a meta batch
  appends chunks to one open container indefinitely. Outputs beyond those
  limits must remain per-window clips today; Dinkster has no append-to-open-
  container encode surface.
