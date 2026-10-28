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
