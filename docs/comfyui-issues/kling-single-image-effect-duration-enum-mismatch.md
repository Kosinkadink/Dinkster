# KlingSingleImageVideoEffectNode exposes duration 10 its request enum rejects

Status: found 2026-07-30 (partner slice 3.2a pinned body audit); not
reported upstream; Dinkster handling: node deferred to partner slice 3.2b
(class-3), where validation parity must pin this exact refusal.

Baseline: ComfyUI e651b7bef55a5376343dcb1c0edb79f0142c985e (partner
pack pinned catalog).

## Symptom

Selecting duration "10" on KlingSingleImageVideoEffectNode (any of its
effect scenes) fails with a pydantic validation error when the request
model is constructed, before any network call. The schema offers a
value the request model cannot accept.

## Root cause

Three definitions disagree:

1. The node schema exposes BOTH durations: the duration combo is built
   from `KlingVideoGenDuration` ("5" and "10") at
   `comfy_api_nodes/nodes_kling.py:2242-2245`
   (`options=[i.value for i in KlingVideoGenDuration]`; enum members at
   `comfy_api_nodes/apis/__init__.py:1328-1330`).
2. The single-image request model requires the RESTRICTIVE enum:
   `KlingSingleImageEffectInput.duration: KlingSingleImageEffectDuration`
   at `comfy_api_nodes/apis/__init__.py:5154-5160`.
3. `KlingSingleImageEffectDuration` has only `field_5 = '5'` at
   `comfy_api_nodes/apis/__init__.py:1290-1291`; there is no "10"
   member.

Reachability: the node's execute passes duration unchanged
(`nodes_kling.py:2271-2286`) into `execute_video_effect`, which selects
the request model solely on `dual_character` (`nodes_kling.py:538-553`);
every single-image scene reaches `KlingSingleImageEffectInput`, so the
mismatch fires for any scene once duration "10" is chosen. No coercion
intervenes, and stringification would not help ("10" is absent from the
restrictive enum). The dual-character node is unaffected: its request
model uses the broad `KlingVideoGenDuration`
(`comfy_api_nodes/apis/__init__.py:5002-5006`).

## Repro

On the pinned checkout, run KlingSingleImageVideoEffectNode with
`effect_scene="squish"`, `model_name` default, `duration="10"`:
pydantic raises a validation error constructing
`KlingSingleImageEffectInput` inside `execute_video_effect`.

## Suggested upstream fix

Either build the node's duration combo from
`KlingSingleImageEffectDuration` (schema stops offering "10"), or add
`field_10 = '10'` to the enum if the Kling API actually accepts it -
whichever matches the provider's documented contract.

## Dinkster handling meanwhile

The node is deferred to partner slice 3.2b. The 3.2b transcription must
either reproduce the upstream refusal behavior verbatim (schema parity
including the dead option, refusal pinned by fixture) or, if upstream
fixes the schema first, follow the fixed shape - decided at 3.2b
adjudication. Recorded in the slice 3.2a audit (see
[partner-node census](https://github.com/Kosinkadink/comfy-vibe-station/blob/main/notes/research/partner-nodes-census.md) correction section and
docs/DELEGATION-LEDGER-PARTNER.md wave 3.2a outcome).
