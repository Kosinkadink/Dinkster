# RenderMesh fails for a single triangle

Status: reproduced locally, not reported upstream.

Reference: ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
`comfy_extras/nodes_mesh_postprocess.py` mesh-postprocess BVH helper
`_build_triangle_bvh`.

Calling the helper with one triangle, such as vertices (-1,-1,0),
(1,-1,0), (0,1,0), raises `RuntimeError: max(): Expected reduction dim
to be specified for input.numel() == 0`. A one-triangle BVH has no
internal nodes; the binary range search still reduces the empty internal
index tensor. RenderMesh reaches this helper through its ordinary mesh input.

The native renderer handles this case as one root leaf with the triangle's
bounds and no children. An upstream fix can use the same single-leaf case
before internal-node construction. The public native regression checks both
the rendered triangle color/mask and the uncovered background.
