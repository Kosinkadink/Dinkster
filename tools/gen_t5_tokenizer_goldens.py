"""Generate T5 tokenizer goldens from the executed reference.

Runs the REFERENCE tokenizers at the audited ComfyUI baseline -
Hugging Face ``tokenizers`` loading
comfy/text_encoders/t5_tokenizer/tokenizer.json (the byte-identical
file dinkster_inference vendors as t5_tokenizer.json.gz) for raw ids,
and comfy.text_encoders.flux.T5XXLTokenizer.tokenize_with_weights for
the Flux weighting/packing shape (no start token, EOS 1, pad 0,
min_length 256, unbounded chunk) - and writes
tests/goldens/t5_tokenizer_goldens.json. dinkster_inference.t5_spm and
the T5 packing profile are pinned against these outputs; the oracle
is the executed reference, never a re-derivation.

The raw-id corpus is a curated set (specials, normalization,
whitespace, scripts the vocabulary cannot represent, tie/fusion
cases) plus a seeded pseudo-random sweep over seven generator pools
that torture the Precompiled charsmap, grapheme iteration, Metaspace,
and the Viterbi lattice. The seed is fixed so regeneration is
deterministic.

Usage (needs a torch interpreter with transformers that imports the
pinned checkout; the workspace root venv is deliberately free of HF
runtime deps; comfy.sd1_clip pulls in comfy.ops -> comfy_aimdo, so
the sibling comfy-aimdo checkout must be on PYTHONPATH when the
venv's copy is older than the baseline needs):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \
        tools/gen_t5_tokenizer_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(COMFY_ROOT))

import tokenizers  # noqa: E402
from comfy.cli_args import args as _comfy_args  # noqa: E402

_comfy_args.cpu = True

from comfy.text_encoders import flux  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

OUT = REPO / "tests" / "goldens" / "t5_tokenizer_goldens.json"
TOKENIZER_JSON = COMFY_ROOT / "comfy" / "text_encoders" / "t5_tokenizer" / "tokenizer.json"

#: Curated corpus: every behavior class the port must reproduce.
CURATED = [
    "",
    " ",
    "  ",
    "hello",
    "hello world",
    "a photo of a cat",
    "Hey   friend!",
    "trailing space ",
    " leading space",
    "a\tb",
    "tabs\tand\nnewlines\r\nhere",
    "\xa0",
    "\u3000ideographic space\u3000wide",
    "x ",
    # Added/special tokens: raw extraction before normalization.
    "a<extra_id_5>b",
    "</s>x",
    "x</s>",
    "<pad><unk></s>",
    "<extra_id_0><extra_id_99>",
    "a <extra_id_42> b",
    "no<extra_id_100>such",  # not a vendored special: plain text
    # NFKC-ish charsmap normalization.
    "caf\u00e9",
    "cafe\u0301",
    "\ufb01ne",
    "\uff12\uff13",
    "\uff41\uff42\uff43",
    "\u00fcberm\u00e4\u00dfig",
    "quotes \u201ccurly\u201d and 'straight'",
    "math \u2211 \u221e \u00b1 5\u00d75",
    # The metaspace replacement appearing literally in input.
    "\u2581weird",
    "pre\u2581mid",
    # Unknown fusion: scripts outside the (English-heavy) vocab.
    "\ud55c\uad6d\uc5b4",
    "\u4f60\u597d\u4e16\u754c",
    "\u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f",
    "\uff76\uff80\uff76\uff85",
    "\U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466",
    "\U0001f1fa\U0001f1f8\U0001f1eb\U0001f1f7",
    "mixed \u4e2d\u6587 and english",
    # Long inputs: lattice depth and chunking downstream.
    "a" * 200,
    ("photorealistic portrait of an astronaut riding a horse " * 8).strip(),
    "supercalifragilisticexpialidocious",
]

#: Weighting/packing corpus for the Flux T5-XXL tokenizer shape.
WEIGHTED_TEXTS = [
    "",
    "hello world",
    "a (masterpiece:1.2) photo of (a (nested:0.5) thing)",
    "((double)) and (plain) and (last:2.0)",
    "(neg:-1) negative weight",
    "\\(escaped\\) parens (real:1.3) here",
    "(unbalanced (foo:1.1",
    "(text:abc) colon fallback",
    "() empty group (x)",
    "  leading and trailing  ",
    "(\u4e2d\u6587:1.3) \u00e9t\u00e9 unicode",
    "</s> special (inside:1.1) weights",
    # Around and past the min_length=256 pad boundary.
    ("masterpiece best quality " * 40).strip(),
    ("a photograph of a very detailed scene " * 40).strip(),
    "one " + "hyperextraordinarily" * 40 + " word",
]

TOKENIZER_OPTION_CASES = [
    {"text": "hello world", "min_padding": 5, "min_length": 12},
]


def fuzz_corpus(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    pools = [
        lambda: "".join(
            rng.choice("abcdefghijklmnopqrstuvwxyz ABCDEFGH.,!?-_' \t\n")
            for _ in range(rng.randint(0, 40))
        ),
        lambda: "".join(
            rng.choice("caf\u00e9\u00fc\u00df\ufb01\ufb02\uff41\uff42\uff43\uff12 e\u0301a\u0300 ")
            for _ in range(rng.randint(0, 24))
        ),
        lambda: "".join(
            rng.choice(
                "\u4f60\u597d\u4e16\u754c\ud55c\uad6d\uc5b4\u30ab\u30bf\u30ab\u30ca\uff76\uff80 "
            )
            for _ in range(rng.randint(0, 16))
        ),
        lambda: "".join(
            rng.choice(
                [
                    "\U0001f469",
                    "\u200d",
                    "\U0001f467",
                    "\U0001f1fa",
                    "\U0001f1f8",
                    "\u0915",
                    "\u094d",
                    "\u0937",
                    " ",
                    "x",
                ]
            )
            for _ in range(rng.randint(0, 12))
        ),
        lambda: (
            rng.choice(["</s>", "<pad>", "<unk>", "<extra_id_0>", "<extra_id_99>", ""])
            + "".join(rng.choice("ab c") for _ in range(rng.randint(0, 8)))
            + rng.choice(["</s>", "<extra_id_42>", ""])
        ),
        lambda: "".join(
            chr(rng.choice([rng.randint(0x20, 0x2FFF), rng.randint(0x1F000, 0x1FAFF)]))
            for _ in range(rng.randint(0, 10))
        ),
        lambda: "".join(
            rng.choice([" ", "  ", "\t", "\u00a0", "\u3000", "\u2581", "a", "B"])
            for _ in range(rng.randint(0, 15))
        ),
    ]
    return [rng.choice(pools)() for _ in range(count)]


def main() -> None:
    head = subprocess.run(
        ["git", "-C", str(COMFY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != REFERENCE_COMMIT:
        sys.exit(f"ComfyUI checkout at {head}, need {REFERENCE_COMMIT}; refusing")
    if subprocess.run(
        ["git", "-C", str(COMFY_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        sys.exit("ComfyUI checkout must be clean; refusing")
    raw = TOKENIZER_JSON.read_bytes()
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    texts = CURATED + fuzz_corpus(seed=20260724, count=2000)
    cases = [{"text": text, "ids": tok.encode(text).ids} for text in texts]

    flux_t5 = flux.T5XXLTokenizer()
    weighted = [
        {
            "text": text,
            "chunks": [
                [[int(t), w, word] for t, w, word in chunk]
                for chunk in flux_t5.tokenize_with_weights(text, return_word_ids=True)
            ],
        }
        for text in WEIGHTED_TEXTS
    ]
    weighted_options = [
        {
            **case,
            "chunks": [
                [[int(t), w, word] for t, w, word in chunk]
                for chunk in flux_t5.tokenize_with_weights(
                    case["text"],
                    return_word_ids=True,
                    tokenizer_options={
                        "t5xxl_min_padding": case["min_padding"],
                        "t5xxl_min_length": case["min_length"],
                    },
                )
            ],
        }
        for case in TOKENIZER_OPTION_CASES
    ]

    OUT.write_text(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "tokenizers_version": tokenizers.__version__,
                "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest(),
                "cases": cases,
                "weighted": weighted,
                "weighted_options": weighted_options,
            },
            ensure_ascii=True,
            indent=None,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
