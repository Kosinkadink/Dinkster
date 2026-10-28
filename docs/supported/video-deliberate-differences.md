# Video behavior differences

## Empty trim selections

Trim Video rejects an empty selection with `ValueError`, including a start
exactly at the source duration with duration zero. The shared VIDEO contract
does not represent an empty video. This is a failure-timing difference from
ComfyUI at `bc5f880dbdd43e426077f42ccec53bb577ffee9f`, not an empty-video
compatibility guarantee.

For a retained eight-frame, 4 fps source, the reference's `VideoFromFile`
window at start 2 seconds and duration zero decodes to a float32 frame batch
with shape `(0, 0, 0, 3)`, frame rate 4, and no audio or alpha. Saving that
window to MP4 raises `ValueError` because there are no decodable frames and
writes zero bytes. Native trimming rejects the window before downstream
decode or save. This observation covers component decode and save, not
preview rendering or the advanced trim node.

## Shared encoder policy for saved files

Crop and trim preserve the selected decoded frames. Saving those frames
through the shared writer can produce different decoded pixels from a
ComfyUI export because lossy encoding uses a different encoder policy;
lossy-export pixel identity is not a crop or trim parity guarantee.

The retained fixed-source probe for an eight-frame, 4 fps MP4
(`bc5f880dbdd43e426077f42ccec53bb577ffee9f`, input SHA-256
`446f89029d695bab37b6d4359d4920f6fd30c950251ce9d4e50ad0e5267a91fd`)
selected two frames at start -0.75 seconds for duration 0.5 seconds and at
start 0.25 seconds for duration 0.5 seconds. Both native selections exactly
matched the source windows under the same RGB24 decoder; saved exports
were not pixel-identical. This is a shared encoder-policy difference, not
a node-local encoder or a different trim boundary. The
[source probe receipt and scope ruling](https://github.com/Kosinkadink/comfy-vibe-station/issues/511#issuecomment-6476052610)
record the distinction. Preview rendering was not exercised.
