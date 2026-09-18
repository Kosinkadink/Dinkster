# Video dimension probe exceptions bypass validation

Status: found 2026-08; not yet reported upstream; Dinkster fails loudly.

Pinned reference: ComfyUI-catalog
`e651b7bef55a5376343dcb1c0edb79f0142c985e`,
`comfy_api_nodes/util/validation_utils.py:96-117`.

## Symptom

Kling video inputs whose width or height cannot be probed can pass local
dimension validation and reach the provider. The provider then rejects the
request later, with a less useful error after upload or transport work.

## Root cause

`validate_video_dimensions` catches probe exceptions and returns without
checking either dimension. A failed open, a missing video stream, or an
indeterminate stream dimension therefore acts as successful validation.

## Reproduction

Pass malformed MP4 bytes, an audio-only container, or a video stream reporting
zero width or height to a node that calls `validate_video_dimensions`. Observe
that no local dimension error is raised before the request path continues.

## Suggested upstream fix

Open the container once, require a first video stream, require positive integer
width and height, and let probe failure surface as a local validation error.
When duration and dimensions are both requested, derive both from that same
open and stream.

## Dinkster interim handling

Serialized VIDEO validation probes loudly whenever non-default dimension bounds
are declared. Missing streams and non-positive or non-integer dimensions are
rejected before upload or provider transport. Duration is checked before
dimensions when both use the same probe.
