# dinkster-values

`dinkster-values` defines Dinkster's typed value envelopes, type registry, payload
transports, fingerprints, lists, resource handles, and resource pins. It has no
package dependencies and no torch dependency; `dinkster-schema`, memory and asset
infrastructure, caches, workers, and the engine build on its value conventions.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```sh
uv sync --all-packages
```

The package is not published separately yet and provides no console script.

## Use

The public surface includes `Value`, `ValueMeta`, payload implementations,
`TypeRegistry`, core type registration, list helpers, `ResourceHandle`,
`ResourcePins`, and first-class absence helpers. Register a type and let the
registry construct its envelope:

```python
from dinkster_values import TypeRegistry

types = TypeRegistry()
types.register("example.point")
point = types.wrap("example.point", {"x": 3, "y": 4})

assert point.type_id == "example.point"
assert point.resolve() == {"x": 3, "y": 4}
```

Core scalar types can be installed into the same registry:

```python
from dinkster_values import CORE_INT, TypeRegistry, register_core_types

types = TypeRegistry()
register_core_types(types)
answer = types.wrap(CORE_INT, 42)
```

Node authors normally work with plain Python values. Worker shims construct
the envelopes at the execution boundary.

### Bounded media streams

`MediaStream` exposes replayable half-open frame or sample ranges with a
declared total and per-read chunk limit. `iter_chunks()` is ordered and
demand-driven; it does not prefetch. `retain()` creates an independent consumer
lease, and the shared source closes only after every retained lease closes.
Use `on_close()` to tie source assets or scratch storage to that final release.

### AUDIO windows

`audio_window(obj, start_sample: int, sample_count: int, *, batch_index: int | None = None)`
returns a dict with float32 `[batch, channels, samples]` `waveform` and integer
`sample_rate`. Indices address the edited timeline. Reads clip at known EOF;
zero-length and past-EOF requests return empty windows without reading sources.
`batch_index=None` selects all batches.

For repeated reads, use `with AudioWindowReader(obj) as reader:` and
`reader.read(start_sample, sample_count, *, batch_index=None)`. Sequential reads
reuse bounded decoder state. Call `reader.close()` when not using a context
manager; any read error also closes the reader.

For ordered consumption, use
`iter_audio_chunks(obj, *, start_sample=0, sample_count=None, chunk_samples, batch_index=None)`.
It yields the same waveform/sample_rate carrier, with at most `chunk_samples`
sample frames per chunk, through one retained `AudioWindowReader`. Windows are
contiguous on the effective edited timeline, including trim, gain, concat,
channel maps and chained resamples; codec state and globally aligned filter halos
are reused. Stored int16/fp32 and channel-layout facts are not changed.

`effective_audio_facts(obj)["frames"]` counts edited sample frames, not scalar
samples across channels/batches, and may be `None`. With `sample_count=None`,
iteration runs to known EOF or actual decoded EOF for an unknown-length root.
An explicit nonnegative count limits that range and clips at EOF. Unknown EOF
is detected from decoder exhaustion, not zero-filled windows, and mapped through
the existing edit facts without modifying carried probe metadata. A trim that
already declares a finite length retains its ordinary finite-window semantics,
including source padding; ordinary `audio_window` padding is unchanged.

The generator validates arguments on first advancement, including batch selection
for empty ranges. `chunk_samples` must be a positive integer whose fp32 chunk
fits the 32 MiB budget; intermediate allocation guards also apply. Iteration emits
no empty chunks. Use `contextlib.closing(iter_audio_chunks(...))` around loops
that may exit early, or call the generator's `close()`. Exhaustion, errors, and
cancellation injected with `throw()` release readers and spool files. A bare
`break` does not guarantee closure. This is one synchronous consumer: cancellation
from another thread cannot be promised to preempt an in-progress read.

These APIs expose errors through `dinkster_values` and
`dinkster_values.audio_codec`, without HTTP status handling:

- `AudioRangeError(ValueError)`: a nonnegative integer `batch_index` is outside
  the batch, checked before source decoding (including empty windows).
- `AudioSourceUnavailableError(ValueError)`: a nonempty read reaches an unbound
  portable source descriptor. Bind it at the asset layer before reading.
- Plain `ValueError`: negative or wrong-type start/count/batch arguments
  (including booleans), invalid values, closed readers, or allocation limits.
- Asset-side failures remain `dinkster_assets.AssetError`, including
  `AssetIntegrityError`, after `dinkster_assets.audio.bind_audio_value` constructs
  an `AssetRef`. Missing resolvers, unavailable content, and failed integrity
  checks are not wrapped as codec errors; `dinkster-values` does not import assets.

### AUDIO renditions

Registered AUDIO types advertise versioned `waveform` PNG and `window` WAV
renditions. `waveform=<width>x<height>` accepts positive dimensions up to
2048x512 and 262,144 pixels. It computes the finite min/max span for each
half-open edited-timeline column, mixes channels by their outer extrema, and
draws opaque RGBA with fixed background, waveform, and silent-center colors.
`window=<start>,<duration>` uses nonnegative decimal seconds, limits duration to
30 seconds and 8,388,608 channel sample values, and returns PCM16 WAV. Both use
the separately normalized zero-based `batch` selector, defaulting to 0.

Parameterized rendition cache identity includes the rendition version and every
normalized selector. HTTP routing reports malformed, repeated, unsupported, or
out-of-range parameters as `invalid_rendition_request`, and unavailable bound
source content as `rendition_unavailable`.

## Learn more

See DESIGN 3.2 for value envelopes, DESIGN 3.13 for `list<T>`, and DESIGN
3.15 for typed absence. The load-bearing rules are hazards H2, H7, H9, H19,
and H20 in `docs/hazards.md`.

Focused tests include `tests/test_values.py`, `tests/test_lists.py`,
`tests/test_resources.py`, `tests/test_pins.py`, and `tests/test_absence.py`.
