# Image aspect lower-bound diagnostic uses the wrong comparator

Status: found 2026-07-31 during partner slice 3.3; not reported
upstream; fixed in Dinkster's independent validation diagnostic.

Baseline: ComfyUI e651b7bef55a5376343dcb1c0edb79f0142c985e (partner
pack pinned catalog), `comfy_api_nodes/util/validation_utils.py:219-226`.

## Symptom

An image at or below a strict minimum aspect ratio is correctly rejected,
but the error says its aspect ratio "must be <" the minimum. Following that
instruction makes the input remain invalid.

## Root cause

The lower-bound branch rejects `ar <= lo` in strict mode (or `ar < lo` in
inclusive mode), then formats the required comparator as `<` (or `<=`). The
displayed operators are inverted: a lower bound requires `>` in strict mode
and `>=` in inclusive mode. The upper-bound branch uses the expected `<` and
`<=` operators and is not affected.

## Repro

On the pinned checkout, call `_assert_ratio_bounds(0.3,
min_ratio=(2, 5))`. It raises `Aspect ratio 0.3 must be < 0.4`, although
the accepted values are greater than 0.4.

## Suggested upstream fix

At line 225, select `>` for strict bounds and `>=` for inclusive bounds.
Keep the rejection predicates unchanged.

## Dinkster handling meanwhile

Dinkster's partner media constraint reports the valid lower-bound direction and
tests both narrow and wide failures in
`test_wan_j_happyhorse_failed_key_validation_precedes_upload`.
