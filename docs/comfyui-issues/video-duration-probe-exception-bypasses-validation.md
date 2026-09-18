# Video duration probe exceptions silently bypass validation

Status: found 2026-07-31 during partner slice 3.3; not reported
upstream; fixed in Dinkster by failing validation loudly before upload.

Baseline: ComfyUI e651b7bef55a5376343dcb1c0edb79f0142c985e (partner
pack pinned catalog), `comfy_api_nodes/util/validation_utils.py:124-127`.

## Symptom

A malformed or otherwise unprobeable video passes duration validation. The
provider request can then proceed even when the node promises minimum and
maximum duration checks.

## Root cause

`validate_video_duration` catches every exception from
`video.get_duration()`, logs it, and returns successfully. The caller cannot
distinguish a valid duration from a failed probe, so both duration bounds are
silently skipped.

## Repro

Pass an `Input.Video` whose `get_duration()` raises `RuntimeError` to
`validate_video_duration(video, min_duration=2, max_duration=30)`. The
function logs the exception and returns without raising.

## Suggested upstream fix

Raise a `ValueError` that preserves the probe exception as its cause instead
of returning. This keeps invalid media out of provider requests and gives the
workflow a useful validation failure.

## Dinkster handling meanwhile

Dinkster treats probe failure as a validation error before any upload or network
request. `test_wan_l_reference_video_loud_probe_failure_and_zero_upload`
proves the behavior.
