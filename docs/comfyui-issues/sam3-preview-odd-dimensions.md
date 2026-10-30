# SAM3 track previews cannot encode odd dimensions

Status: reproduced locally; not reported upstream.

Reference: ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
`comfy_extras/nodes_sam3.py`, `SAM3_TrackPreview.execute`.

The node assigns source width and height directly to an H264 stream with
`yuv420p` pixel format. That format requires even width and height. Source
images take precedence over `track_data["orig_size"]`; neither path aligns
the dimensions before encoding.

A CPU image batch of shape `(1, 16, 15, 3)` with absent packed masks,
`n_frames=1` and `orig_size=(16, 15)` raises `av.error.ExternalError`, errno
542398533, at `avcodec_open2("libx264", {})`. The AST-extracted reference and
native node produced the same failure with PyAV 18.1.0. No preview artifact
was published. This is an inherited input limitation, not a new port defect;
no claim is made that every producer supplies even dimensions.

An upstream fix could define an explicit crop, pad or resize policy for odd
dimensions. Dinkster preserves the fixed reference behavior instead of
silently changing dimensions. Even-sized previews are unaffected.
