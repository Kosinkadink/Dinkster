## Implemented capabilities with limited or specialized exposure

- Experimental on-demand execution supports static data-only nodes with
  scalars, images, masks, latents, and recursive lists through a caller-supplied
  dispatch and object store. Only local execution is validated.
- Collaboration sessions, operations, snapshots, and session WebSockets are
  mountable but are not mounted by `create_app`.
- The public mount-scoped asset writer publishes bounded, ordered stream batches
  with source-compatible counters and can give an encoder its selected filename
  stem and counter before it produces a bounded output stream.
- Native job submission accepts optional `exportSnapshot` metadata with an object
  `prompt` and optional object `extraPnginfo`. Output nodes can read a detached
  copy through public v1 `current_export_snapshot`; the server does not reconstruct
  an original workflow or prompt from the submitted execution graph.
- MiniMax H3 independently loaded components and attention/residency behavior
  are supplied by the pinned `dinkster_inference` runtime.
- Model-family training and all retired model-family inference surfaces are not
  currently exposed by Dinkster.
