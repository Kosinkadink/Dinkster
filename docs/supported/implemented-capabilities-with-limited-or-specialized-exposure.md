## Implemented capabilities with limited or specialized exposure

- Experimental on-demand execution supports static data-only nodes with
  scalars, images, masks, latents, and recursive lists through a caller-supplied
  dispatch and object store. Only local execution is validated.
- Collaboration sessions, operations, snapshots, and session WebSockets are
  mountable but are not mounted by `create_app`.
- MiniMax H3 independently loaded components and attention/residency behavior
  are supplied by the pinned `dinkster_comfy` runtime.
- Model-family training and all retired model-family inference surfaces are not
  currently exposed by Dinkster.
