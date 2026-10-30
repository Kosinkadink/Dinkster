## Implemented capabilities with limited or specialized exposure

- Environment management can inspect local metadata, start and stop named
  supervised bindings, and reactivate a retained previous generation on explicit
  confirmation. Activation journals preserve running and stopped assignments
  across recovery; ambiguous process identities refuse mutation. End-to-end
  setup, update and relocation are not supported install paths yet.
- Experimental on-demand execution supports static data-only nodes with
  scalars, images, masks, latents, and recursive lists through a caller-supplied
  dispatch and object store. Only local execution is validated.
- Collaboration sessions, operations, snapshots, and session WebSockets are
  mountable but are not mounted by `create_app`.
- The public mount-scoped asset writer publishes bounded, ordered stream batches
  with source-compatible counters and can give an encoder its selected filename
  stem and counter before it produces a bounded output stream.
- Explicit named imports through the mount writer preserve the requested filename
  when free and number collisions without overwriting existing files. They retain
  writable-grant, containment and bounded-stream checks. This writer API alone does
  not change the destination of browser uploads.
- Native job submission accepts optional `exportSnapshot` metadata with independently
  optional object `prompt` and `extraPnginfo` members. Unavailable metadata remains
  absent, distinct from an explicitly empty object. Output nodes can read a detached
  copy through public v1 `current_export_snapshot`; the server does not reconstruct
  an original workflow or prompt from the submitted execution graph.
- MiniMax H3 independently loaded components and attention/residency behavior
  are supplied by the pinned `dinkster_inference` runtime.
- Model-family training and all retired model-family inference surfaces are not
  currently exposed by Dinkster.
