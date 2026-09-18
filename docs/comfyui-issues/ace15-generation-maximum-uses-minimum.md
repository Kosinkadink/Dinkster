# ACE-Step 1.5 generation maximum uses the minimum token bound

Status: found; not reported upstream.

At [ComfyUI 25dfc16f9ac0, ace15.py lines 333-336](https://github.com/Comfy-Org/ComfyUI/blob/25dfc16f9ac0a87991d34fbf5f02d6c25c844639/comfy/text_encoders/ace15.py#L333-L336),
`ACE15TEModel.encode_token_weights` passes `lm_metadata["min_tokens"]` as
both `min_tokens` and `max_tokens` to `generate_audio_codes`. The tokenizer
independently retains the requested `max_tokens` in `lm_metadata`.

## Reproduction and effect

Tokenize with `duration=2`, `min_tokens=3`, and `max_tokens=10`. Metadata
contains minimum 3 and maximum 10, but the generation call receives maximum
3. Consequently the loop cannot generate more than 3 codes. Its EOS branch
requires `min_tokens < step`, so EOS is unreachable through this composer.
With default bounds both equal `ceil(duration) * 5`, hiding the discrepancy.

## Suggested upstream correction

Pass `lm_metadata["max_tokens"]` for the maximum, then separately decide
whether the strict EOS test should allow stopping immediately at the minimum.
These are behavior changes and need source-side tests.

## Dinkster behavior

The ACE recipe retains both requested bounds but reproduces the pinned
composer call with the minimum as the effective maximum. The standalone
audio generation function honors its explicit maximum and strict EOS test.
Synthetic tests distinguish these contracts; they do not establish
official-artifact numerical or performance parity.
