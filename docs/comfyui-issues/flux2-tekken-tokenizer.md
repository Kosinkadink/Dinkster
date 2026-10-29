# Flux2 Mistral tekken tokenizer loading is fragile and ignores the blob's pattern

- **Area:** ComfyUI `comfy/text_encoders/flux.py` `load_mistral_tokenizer` at
  `947c2749dd04c51ef0e21b069544d8b0b4f9b411`
- **Status:** found 2026-08; crash fixed upstream by `bbb4b04c` (2026-08-11);
  pattern divergence still present; Dinkster self-contained

Two findings in the same loader, which converts the `tekken_model` JSON blob
embedded in the Mistral3-Small Flux2 text-encoder checkpoint into a usable
tokenizer.

## 1. Crash under transformers 5.x

### Symptom

Loading the Flux2 dev text encoder raises a `TypeError` when the installed
transformers is 5.x: `load_mistral_tokenizer` calls
`transformers.convert_slow_tokenizer.MistralConverter(vocab=...,
additional_special_tokens=...)`, but transformers 5.15.1 changed
`MistralConverter.__init__` to take a `vocab_file` path and removed the
keyword API (and removed the pixtral fallback the old code path relied on).

### Root cause

The loader depends on a private, unversioned transformers internal. The
converter's constructor signature changed between 4.57.x and 5.x, so the
pinned reference only runs with transformers 4.57.x.

### Repro

With `transformers==5.15.1` installed, construct
`comfy.text_encoders.flux.Flux2Tokenizer(tokenizer_data={"tekken_model":
<blob bytes>})` at the pinned commit. With `transformers==4.57.1` the same
call succeeds.

### Upstream fix

Already fixed: commit `bbb4b04c` (2026-08-11) removed the transformers
dependency entirely and converts the blob with an in-repo pure-Python BPE
(`comfy/text_encoders/bpe_tokenizer.py` `from_tekken_json`). Its id and
merge derivation is equivalent to the old converter output; it lacks the
whole-piece `ignore_merges` shortcut, but every kept multi-byte token is
reachable through its own merges, so no id-sequence divergence was found.

## 2. The blob's declared split pattern is ignored

### Symptom

The tekken JSON declares its own v11 pre-tokenization pattern, but the
converted tokenizer never uses it: ComfyUI never passes the pattern through,
so the converter's default Llama-style split regex applies (same as Qwen's
except number pieces are `\p{N}{1,3}` instead of `\p{N}`). Long digit runs
therefore split into groups of up to three digits, where the blob's declared
pattern splits differently - a divergence from mistral-common/BFL tokenizer
semantics.

### Root cause

`load_mistral_tokenizer` builds the converter from the blob's vocab and
special tokens only. Both the pinned converter path and master's
`from_tekken_json` hardcode the default pattern instead of reading
`config["pattern"]` from the blob.

### Upstream fix

Not reported; the released Flux2 checkpoints were evidently trained/served
with the default pattern via this exact path, so "fixing" it would change
model inputs. If upstream ever adopts the declared pattern, it must be a
deliberate, evidence-backed change.

## Dinkster handling

Dinkster vendors the exact `tekken_model` bytes and reimplements the conversion
without transformers (`dinkster_inference/tekken_bpe.py`), mirroring the
reference semantics exactly: default Llama-style pattern with three-digit
number pieces, rank+1000 id mapping, rank-ordered tiktoken-style merges,
whole-piece vocabulary hits bypassing the merge loop, no normalizer.
Goldens replay executed reference outputs generated with transformers
4.57.1 at the pinned commit.
