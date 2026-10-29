# Bare `embedding:` directive crashes tokenization with IndexError

Baseline: ComfyUI @ 947c2749dd04c51ef0e21b069544d8b0b4f9b411

## Symptom

A prompt containing a bare `embedding:` directive with no name after
it (for example `"foo embedding:"` or `"embedding: bar"` where the
directive is followed only by whitespace) crashes prompt tokenization
with an unhandled `IndexError` when an embedding directory is
configured. The whole job errors out instead of the directive being
ignored or reported.

## Root cause

`comfy/sd1_clip.py` `SDTokenizer._try_get_embedding` (lines 548-549):

```python
split_embed = embedding_name.split()
embedding_name = split_embed[0]
```

When the text after the `embedding:` identifier is empty or
whitespace-only, `embedding_name.split()` returns `[]` and
`split_embed[0]` raises `IndexError`. The caller
(`tokenize_with_weights`, line ~603) strips only `"\n"` from the
directive text before calling, so empty/whitespace names reach the
unguarded index.

## Repro

With any embedding directory configured (contents irrelevant):

```python
from comfy import sd1_clip
tok = sd1_clip.SDTokenizer(embedding_directory=".")
tok.tokenize_with_weights("foo embedding:")  # IndexError
```

## Suggested upstream fix

Guard the empty split in `_try_get_embedding` - e.g. return
`(None, "", leftover)` when `split_embed` is empty - so a bare
directive degrades to the existing unresolvable-embedding path
(directive dropped, rest of the prompt tokenizes normally).

## Dinkster handling

Deliberate divergence, not bug-for-bug compatibility:
`dinkster_inference.prompt_tokens.tokenize_prompt` reports the empty
name in `TokenizedPrompt.missing_embeddings` and drops the directive;
the remaining prompt tokenizes identically to the reference without
the directive. Pinned by
`tests/test_inference_clip_tokenize.py::test_bare_directive_is_missing_not_a_crash`.

## Status

found 2026-07; handled in Dinkster (reported-not-crash divergence)
