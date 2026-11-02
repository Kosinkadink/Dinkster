# Normal-map bake indexes shared high-mesh normals by the low batch

Status: reproduced at fixed ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f;
not reported upstream.

`BakeNormalMapFromMesh.execute` supports one shared high-poly mesh for multiple
low-poly meshes. With authored high-poly normals it raises `IndexError` on the
second low-poly item: geometry selects high batch zero, but normals select the
low batch index in `comfy_extras/nodes_mesh_postprocess.py`.

Reproduction: pass two UV-mapped low-poly planes and one high-poly plane with
`normals` shaped `[1, 4, 3]`. At the second item the fixed source reports
`index 1 is out of bounds for dimension 0 with size 1`. Omitting authored high
normals takes the computed-normal path and does not trigger that indexing bug.

The upstream fix is to index high normals with the same selected high batch as
the high vertices and faces. The native bake does this, so authored-normal
broadcast agrees with geometry broadcast. Its public regression also covers an
empty third low-poly item; both nonempty outputs retain the shared authored
normal. No tolerance or accepted fixed-reference receipt is changed.
