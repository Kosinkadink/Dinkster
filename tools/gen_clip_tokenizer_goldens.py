"""Generate CLIP tokenizer goldens from the ComfyUI reference.

Runs the REFERENCE tokenizers @ the audited baseline - Hugging Face
CLIPTokenizer over comfy/sd1_tokenizer for raw BPE ids, and
comfy.sd1_clip.SDTokenizer / comfy.sdxl_clip.SDXLClipGTokenizer for
tokenize_with_weights - and writes
tests/goldens/clip_tokenizer_goldens.json. dinkster_inference.clip_bpe
and dinkster_inference.prompt_tokens are pinned against these outputs;
the oracle is the executed reference, never a re-derivation.

Embedding cases use fixture safetensors files written to a temp
directory; every row of every fixture is filled with a distinct
constant so the golden can record tensor tokens as
``{"emb": name, "row": r}`` markers by reading the value back.

Usage (needs a torch interpreter with transformers + safetensors that
imports the pinned checkout; the workspace root venv is deliberately
torch-free). comfy.sd1_clip pulls in comfy.ops -> comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv's
copy is older than the baseline needs:

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \
        tools/gen_clip_tokenizer_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402
from comfy import sd1_clip, sdxl_clip  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import CLIPTokenizer  # noqa: E402

OUT = REPO / "tests" / "goldens" / "clip_tokenizer_goldens.json"

#: Raw-BPE corpus: normalization, scanning, and merge coverage.
BPE_TEXTS = [
    "",
    "a photo of a cat",
    "Hello, WORLD!!  multiple   spaces",
    "a (masterpiece:1.2) photo",
    "it's the dog's toy, isn't it?",
    "cafe\u0301 naive\u0308 r\u00e9sum\u00e9",
    "\u4e2d\u6587\u6d4b\u8bd5 mixed CJK\u6c49text",
    "emoji \U0001f600\U0001f680 and \u2603 snowman",
    "numbers 1234567890 42nd 3.14159",
    "<|endoftext|> literal and <|startoftext|> too",
    "!!<|endoftext|> tricky",
    "<|ENDOFTEXT|> upper literal",
    "control\x00chars\x07here\ufffd",
    "tabs\tand\nnewlines\r\nhere",
    "'s 't 're 've 'm 'll 'd standalone",
    "hyphen-ated under_scored dotted.words slash/path",
    "\u00fc\u00f1\u00ee\u00e7\u00f8d\u00e9 accents \u00df sharp",
    "ALLCAPS MiXeD case",
    "a" * 200,
    ("supercalifragilistic " * 30).strip(),
    "trailing space ",
    " leading space",
    "  ",
    "\u3000ideographic space\u3000wide",
    "quotes \u201ccurly\u201d and 'straight'",
    "math \u2211 \u221e \u00b1 5\u00d75",
    "\U0001f3f4\u200d\u2620\ufe0f pirate flag zwj",
    "Ko\u017eu\u0161\u010dek",
    "\u0130stanbul \u0131 dotless",
]

#: Weighting/packing corpus, tokenized by both CLIP-L and CLIP-G.
WEIGHTED_TEXTS = [
    "a (masterpiece:1.2) photo of (a (nested:0.5) thing)",
    "((double)) and (plain) and (last:2.0)",
    "(neg:-1) negative weight",
    "\\(escaped\\) parens (real:1.3) here",
    "(unbalanced (foo:1.1",
    ")stray( parens",
    "(text:abc) colon fallback",
    "(:1.5) colon at start",
    "() empty group (x)",
    ("masterpiece best quality " * 30).strip(),
    "one " + "hyperextraordinarily" * 20 + " word",
    "(" + ("weighted across chunks " * 20).strip() + ":1.4) tail",
    "  leading and trailing  ",
    "(\u4e2d\u6587:1.3) \u00e9t\u00e9 unicode",
    "",
]

#: (params, text) variant cases exercising non-default packing knobs
#: on the CLIP-L tokenizer.
VARIANT_CASES = [
    ({"pad_to_max_length": False}, "short (weighted:1.2) prompt"),
    ({"pad_to_max_length": False, "min_length": 20}, "short prompt"),
    ({"min_padding": 5, "pad_left": True}, "left padded (x:0.9)"),
    ({"min_length": 90}, "min length beyond max (y:1.1)"),
    ({"pad_to_max_length": False}, ("word " * 80).strip()),
]

#: Fixture embeddings: name -> row count. Row r of fixture k is filled
#: with 100*(k+1)+r so tensors map back to (name, row) exactly.
EMBED_FIXTURES = {"emb1": 1, "emb2": 2, "emb9": 9}

EMBED_TEXTS = [
    "photo of embedding:emb2 style",
    "embedding:emb2 at start",
    "(embedding:emb2:1.4) weighted",
    "a embedding:emb2, comma leftover",
    "a embedding:nosuch b",
    "x embedding:emb9 y",
    ("word " * 70).strip() + " embedding:emb9 tail",
    "pre (padding embedding:emb2 leftover:0.7) post",
    "z embedding:emb2<lora:x> angle-stop",
    "b ( embedding:emb2:1.2) c",
    "rank one embedding:emb1 vector",
    "double embedding:emb2 embedding:emb1 refs",
]


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def chunk_to_json(chunk, marker_of):
    out = []
    for token, weight, word_id in chunk:
        if isinstance(token, torch.Tensor):
            out.append([marker_of(token), weight, word_id])
        else:
            out.append([int(token), weight, word_id])
    return out


def main() -> None:
    head = git_head(COMFY_ROOT)
    if head != REFERENCE_COMMIT:
        raise SystemExit(f"../ComfyUI is at {head}, expected reference {REFERENCE_COMMIT}")

    hf = CLIPTokenizer.from_pretrained(str(COMFY_ROOT / "comfy" / "sd1_tokenizer"))
    bpe_goldens = [{"text": text, "ids": hf(text)["input_ids"]} for text in BPE_TEXTS]

    tok_l = sd1_clip.SDTokenizer()
    tok_g = sdxl_clip.SDXLClipGTokenizer()

    def no_marker(_tensor):
        raise AssertionError("tensor token outside embedding section")

    weighted_goldens = []
    for text in WEIGHTED_TEXTS:
        weighted_goldens.append(
            {
                "text": text,
                "l": [
                    chunk_to_json(c, no_marker)
                    for c in tok_l.tokenize_with_weights(text, return_word_ids=True)
                ],
                "g": [
                    chunk_to_json(c, no_marker)
                    for c in tok_g.tokenize_with_weights(text, return_word_ids=True)
                ],
            }
        )

    variant_goldens = []
    for params, text in VARIANT_CASES:
        tok = sd1_clip.SDTokenizer(**params)
        variant_goldens.append(
            {
                "params": params,
                "text": text,
                "chunks": [
                    chunk_to_json(c, no_marker)
                    for c in tok.tokenize_with_weights(text, return_word_ids=True)
                ],
            }
        )

    with tempfile.TemporaryDirectory() as embed_dir:
        markers = {}
        for k, (name, rows) in enumerate(EMBED_FIXTURES.items()):
            fills = [100 * (k + 1) + r for r in range(rows)]
            for fill, row in zip(fills, range(rows), strict=True):
                markers[float(fill)] = {"emb": name, "row": row}
            tensor = torch.tensor(fills, dtype=torch.float32).reshape(rows, 1)
            tensor = tensor.expand(rows, 768).contiguous()
            if rows == 1:
                tensor = tensor.reshape(768)
            save_file({"emb": tensor}, str(Path(embed_dir) / f"{name}.safetensors"))

        def marker_of(tensor):
            return markers[float(tensor.reshape(-1)[0].item())]

        tok_emb = sd1_clip.SDTokenizer(embedding_directory=embed_dir)
        embed_goldens = []
        for text in EMBED_TEXTS:
            embed_goldens.append(
                {
                    "text": text,
                    "chunks": [
                        chunk_to_json(c, marker_of)
                        for c in tok_emb.tokenize_with_weights(text, return_word_ids=True)
                    ],
                }
            )

    OUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "comfyui_commit": head,
                    "transformers": transformers.__version__,
                    "torch": torch.__version__,
                    "generator": "tools/gen_clip_tokenizer_goldens.py",
                },
                "embedding_fixtures": EMBED_FIXTURES,
                "bpe": bpe_goldens,
                "weighted": weighted_goldens,
                "variants": variant_goldens,
                "embeddings": embed_goldens,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
