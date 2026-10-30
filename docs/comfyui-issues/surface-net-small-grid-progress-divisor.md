# Surface-net extraction fails on small nonempty grids

Status: reproduced in ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f;
not reported upstream.

`voxel_to_mesh_surfnet` in `comfy_extras/nodes_hunyuan3d.py` computes
`vert_progress_mod = round(len(cell_vertices) / 50)` and then uses
`i % vert_progress_mod`. A small nonempty surface rounds the divisor to zero.
The face progress loop has the same problem with its independently rounded
divisor. Geometry extraction raises `ZeroDivisionError` before returning.

CPU reproduction: create a float32 tensor of shape `(2, 2, 2)`, set element
`[0, 0, 0]` to 1, leave all others zero, and call
`voxel_to_mesh_surfnet(grid, threshold=0.5)`. Eight active cells produce a
zero vertex progress divisor. The reference fails with integer modulo by zero.

An upstream fix can clamp both reporting divisors to at least one. Dinkster
reports progress through the native reporting API without modulo divisors;
the extraction calculations and vertex/face order are unchanged. A CPU
regression checks the small surface and an empty surface through `VoxelToMesh`.
