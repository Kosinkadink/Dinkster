# LoadImage rejects a small animated GIF

Status: found locally with ComfyUI b78cec879b9460d5cb25228a83a942fb78d2cd24
and e20d433a4966dcc88fa5abbae6ace824cb78b263, PyAV 18.1.0.

Pillow can write and decode a two-frame 3x2 RGB GIF, but ComfyUI LoadImage
raises `av.error.ArgumentError: Invalid argument returned 22` in
`VideoFromFile.get_components_internal` at `g.configure()`. The alignment
filter graph (`pad` and `fillborders`) fails before the Pillow fallback.

Reproduce by saving red and blue 3x2 RGB Pillow images as a GIF with
`save_all=True`, `append_images=[blue]`, and `duration=100`, then calling
`LoadImage().load_image(filename)` from the configured input directory.

Dinkster's image loader uses Pillow's ordered frame iterator and handles this
file. Its small-GIF regression is in `tests/test_image_io.py`. Upstream could
route Pillow-supported animations through the image decoder rather than the
video alignment filter. No upstream report or source change was made.

The pinned PyAV path also changes samples in an otherwise valid 32x2 GIF:
RGB differs from exact Pillow palette values by up to 0.00387579 and an
opaque alpha mask contains values up to 0.000045776367. A two-page TIFF
returns only its first page through that path. Dinkster deliberately retains
exact palette/alpha values and loads every same-size TIFF page. These
differences are asserted explicitly in the pinned image-load replay tests,
not hidden behind widened tolerances. PNG and lossless WebP animation
batches match the source vectors exactly.
