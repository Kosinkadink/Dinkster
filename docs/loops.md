# Loop regions

Dinkster uses one typed region primitive with three execution profiles:

- **Map** runs one independent body occurrence per element binding. Occurrences
  may execute concurrently, while gathered outputs retain binding order.
- **Fold** runs sequentially and passes each state output into its matching
  state input. Gathering the same body output exposes the intermediate states
  as a scan.
- **While** runs sequentially until its boolean continuation is false. Express
  "until condition" by continuing while the condition is false. Every while
  region has a positive maximum iteration count and fails loudly at the cap.

Element bindings are zip, cross product, or broadcast. Region outputs are
gather, compact, flatten, or state. Gather retains absent values, compact omits
them, and flatten concatenates one list level. Whole-region input absence skips
the body and propagates absence to every output.

Ports are generic. The same region carries images, latents, conditioning,
masks, audio, video, assets, strings, integers, floats, and pack-defined value
types without a loop-specific codec. Native `list<T>` values preserve element
type, order, metadata, and exact payload content. State ports preserve one
typed value unchanged across iterations even when that value's payload is
heterogeneous or list-shaped; only an explicitly declared `flatten` output
concatenates `list<T>` values.

A region is itself the declared loop boundary in the graph contract. Its body,
typed ports, element and state roles, output modes, and while continuation are
serialized inside the region entry and validated structurally. Boundaries are
not inferred from node names, implementation classes, or matching ports.

Runtime body IDs use `region[index]/node`. Nested regions extend the path at
each level. The server emits expansion, per-iteration start and completion,
region completion, and ordinary body-node lifecycle events. Event replay and
job history preserve these facts.

The foundation pack ships executable CPU templates for image map, ordered
gather to an image batch, fold with scan, while/until, and per-item image
spawn. They are listed by `/api/templates` and load as ordinary editable
workflow documents.
