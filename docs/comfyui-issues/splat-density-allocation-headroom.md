# SplatToMesh allocates density grids before checking accelerator headroom

Status: found in pinned source bc5f880dbdd43e426077f42ccec53bb577ffee9f;
not reported upstream. Dinkster selects CPU density construction when the
grids alone cannot fit in reported accelerator memory.

In `comfy_extras/nodes_gaussian_splat.py`, `_splat_density` allocates a
float32 density grid, three-channel float16 color grid and, when sharpening
colors, a float16 normalization grid before `_gaussian_to_mesh` checks free
memory for Surface Nets. These require 10 or 12 bytes per voxel. At a cubic
769x769x769 grid, the buffers alone need 4,547,566,090 bytes without sharpening
and 5,457,079,308 bytes with sharpening. The later extraction fallback cannot
handle an allocation failure that has already occurred.

Reproduce with a valid nonempty splat on an accelerator whose remaining
memory is smaller than the computed grids. The density/color allocation
fails before Surface Nets can select CPU. Dinkster's public-node regression
models the constrained allocator on CPU, including a budget that fits the
unsharpened grids but not the extra normalization grid.

Compute the actual bounded grid dimensions and minimum buffer bytes before
allocating them. Select CPU construction if those buffers cannot fit, then
carry the density device through extraction and color sampling. Keep the
normal accelerator path and the later extraction-headroom check. This is a
minimum-buffer guard, not a reservation or a guarantee against concurrent
allocations and kernel-workspace exhaustion.
