# Self-Attention Guidance cannot reshape a one-token attention grid

Status: found locally; not reported upstream.

Reference: ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
`comfy_extras/nodes_sag.py`, `create_blur_map`.

For one attention token, the reference factor-search loop has no candidate and
leaves its grid width unset. The subsequent mask reshape fails. Dinkster uses
the explicit 1x1 grid for this case, then applies the same threshold-selected
blur behavior. Multi-token factor selection remains unchanged.
