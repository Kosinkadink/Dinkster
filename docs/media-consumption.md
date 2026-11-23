# Media consumer cleanup

For decoded document leases and durable resource retention, follow
[produced document ownership](image-document-operations.md). A decoded object or
retained resolver alone is not ownership of its backing storage.

## Frames and shared streams

Use `with open_image_components(source) as frames:` when consuming image frames.
Account separately for retained arrays: decoder admission guards cover one frame
and its conversion scratch, not an accumulated batch. FFmpeg opening and probing
are not a hard allocation sandbox. See the
[image component API](../packages/dinkster-values/src/dinkster_values/image_components.py)
for admission limits.

Use `MediaStream.retain()` for an independent consumer lease, and close every
lease. Tie source assets or scratch storage to final release with `on_close()`.
See the [stream API](../packages/dinkster-values/src/dinkster_values/streams.py).

## Audio readers and early termination

Use `with AudioWindowReader(obj) as reader:` for repeated reads, or close the
reader explicitly. For loops that can stop early, use
`contextlib.closing(iter_audio_chunks(...))` or call the generator's `close()`;
a bare `break` does not guarantee reader and spool-file cleanup. Argument
validation happens on first advancement, not generator construction.

This is one synchronous consumer. Cancellation from another thread cannot be
promised to preempt an in-progress read. Injected `throw()` cancellation, errors
and exhaustion release the reader; do not mistake those cases for cross-thread
preemption. Bind portable sources at the asset layer before a nonempty read.
Asset resolver, availability and integrity failures remain asset errors rather
than codec errors.

The [audio reader API](../packages/dinkster-values/src/dinkster_values/audio_lazy.py)
and [audio codec and rendition API](../packages/dinkster-values/src/dinkster_values/audio_codec.py)
are the reference for edited-timeline windows, EOF, allocation limits, selectors
and errors. See [asset source binding](../packages/dinkster-assets/src/dinkster_assets/audio.py)
for host resolver setup.
