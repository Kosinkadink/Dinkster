"""Generate LTX-2 text-embedding connector goldens from pinned ComfyUI.

This runs the reference ``Embeddings1DConnector`` at b78cec87 exactly as
the LTXAV text encoder instantiates it (``split_rope=True``,
``double_precision_rope=True``). The full-size tower is constructed on
the meta device for its strict key/shape layout and cross-checked
against both connector subtrees of the real combined checkpoint header.
A tiny geometry runs the same reference transformer math with
deterministic hash-filled weights, and the reference
``LTXAVTEModel.encode_token_weights`` executes the complete
projection-then-connectors path over a tiny stack.

Run from the Dinkster root with a sibling ``ComfyUI`` checkout detached at
the pinned commit with a clean tree, under the torch build recorded in
the payload. ``PYTHONPATH`` must include comfy-aimdo when the reference
interpreter does not already provide it. One artifact input is
required, verified against its pinned sha256:

    DINKSTER_LTX2_19B_HEADER_JSON     safetensors header JSON of the combined
                                   Lightricks ltx-2-19b-dev checkpoint
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
#: The torch build the goldens are certified under.
GENERATOR_TORCH = "2.13.0+cpu"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402
import torch  # noqa: E402
from clip_fill import fill_value  # noqa: E402

comfy.options.enable_args_parsing()

from comfy import model_management, ops  # noqa: E402
from comfy.ldm.lightricks import embeddings_connector  # noqa: E402
from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy.text_encoders import lt  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). On this CPU-only
# interpreter the module-level selection falls to the manual
# attention_basic kernel, whose softmax rounding drifts from SDPA.
# The connector's CrossAttention calls the selector through its fully
# qualified module attribute, so rebinding the module global suffices.
_attention.optimized_attention = _attention.attention_pytorch
ATTENTION_BACKEND = "attention_pytorch"

OUT = REPO / "tests" / "goldens" / "ltx_embeddings_connector_goldens.json"

LTX2_HEADER_SHA256 = "92f323a94dff45d5cb985917692535a1dc8af5126b9aca7e7d162e535e552fd0"

VIDEO_PREFIX = "model.diffusion_model.video_embeddings_connector."
AUDIO_PREFIX = "model.diffusion_model.audio_embeddings_connector."

#: Tiny geometry in Dinkster LtxConnectorConfig field names; the reference
#: construction arguments are derived from it below.
TINY_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 4,
    "num_layers": 2,
    "num_learnable_registers": 4,
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": 4096,
}
TINY_INNER = TINY_CONFIG["num_attention_heads"] * TINY_CONFIG["attention_head_dim"]

#: Token counts of the executed tiny forwards: one below the reference's
#: hard-coded 1024 register fill floor, one above it.
TINY_TOKEN_COUNTS = (5, 1030)

#: The executed encoder stack: [batch, depth, tokens, hidden] with a
#: left-padded mask, projected to the tiny connector width.
ENCODER_STACK_SHAPE = (1, 3, 6, 4)
ENCODER_ATTENTION_MASK = (0, 0, 1, 1, 1, 1)
ENCODER_FEATURES = ENCODER_STACK_SHAPE[1] * ENCODER_STACK_SHAPE[3]


def required_input(name: str, sha256: str) -> Path:
    configured = os.environ.get(name)
    if configured is None:
        raise SystemExit(f"{name} is not set")
    path = Path(configured)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{name} {path} has sha256 {digest}, expected {sha256}")
    return path


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_dirty(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def enc(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def reference_connector(config: dict[str, object], device: str) -> Embeddings1DConnector:
    return Embeddings1DConnector(
        attention_head_dim=config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        num_layers=config["num_layers"],
        positional_embedding_theta=config["positional_embedding_theta"],
        positional_embedding_max_pos=[config["positional_embedding_max_pos"]],
        num_learnable_registers=config["num_learnable_registers"],
        split_rope=True,
        double_precision_rope=True,
        dtype=torch.float32,
        device=device,
        operations=ops.disable_weight_init,
    )


FULL_CONFIG = {
    "num_attention_heads": 30,
    "attention_head_dim": 128,
    "num_layers": 2,
    "num_learnable_registers": 128,
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": 4096,
}


def full_layout() -> list:
    connector = reference_connector(FULL_CONFIG, "meta")
    return sorted((key, list(value.shape)) for key, value in connector.state_dict().items())


def ltx2_19b_dev(header_path: Path, layout: list) -> dict[str, object]:
    header = json.loads(header_path.read_text())
    subtrees = {}
    for name, prefix in (("video", VIDEO_PREFIX), ("audio", AUDIO_PREFIX)):
        entries = sorted(
            (key.removeprefix(prefix), list(entry["shape"]))
            for key, entry in header.items()
            if key.startswith(prefix)
        )
        if entries != layout:
            raise SystemExit(f"{name} connector subtree does not match the reference layout")
        subtrees[f"{name}_subtree_matches_reference_layout"] = True
    absent = f"{AUDIO_PREFIX}transformer_1d_blocks.2.attn1.to_q.bias"
    trigger = f"{AUDIO_PREFIX}transformer_1d_blocks.0.attn1.to_q.bias"
    connector_keys = sorted(key for key in header if "embeddings_connector" in key)
    dtypes = sorted({header[key]["dtype"] for key in connector_keys})
    return {
        "source": "Lightricks ltx-2-19b-dev.safetensors full header",
        "header_sha256": LTX2_HEADER_SHA256,
        **subtrees,
        "connector_key_count": len(connector_keys),
        "connector_dtypes": dtypes,
        "absent_probe_key": absent,
        "absent_probe_present": absent in header,
        "trigger_key": trigger,
        "trigger_shape": list(header[trigger]["shape"]) if trigger in header else None,
        "fires_reference_compat_mode": absent not in header
        and trigger in header
        and header[trigger]["shape"][0] == 3840,
    }


def tiny_reference_connector(fill_prefix: str = "") -> Embeddings1DConnector:
    connector = reference_connector(TINY_CONFIG, "cpu")
    entries = sorted((key, list(value.shape)) for key, value in connector.state_dict().items())
    connector.load_state_dict(
        {key: fill_value(f"{fill_prefix}{key}", shape) for key, shape in entries}, strict=True
    )
    return connector


def tiny_golden() -> dict[str, object]:
    connector = tiny_reference_connector()
    entries = sorted((key, list(value.shape)) for key, value in connector.state_dict().items())
    cases = []
    for tokens in TINY_TOKEN_COUNTS:
        fill_key = f"connector.case.t{tokens}.input"
        source = fill_value(fill_key, (1, tokens, TINY_INNER))
        with torch.no_grad():
            output, mask = connector(source.clone())
        if mask is not None:
            raise AssertionError("the maskless connector forward must return mask None")
        expected_length = (
            math.ceil(max(1024, tokens) / TINY_CONFIG["num_learnable_registers"])
            * TINY_CONFIG["num_learnable_registers"]
        )
        if list(output.shape) != [1, expected_length, TINY_INNER]:
            raise AssertionError(f"unexpected connector output shape {list(output.shape)}")
        cases.append({"input_fill_key": fill_key, "input": enc(source), "output": enc(output)})
    return {"config": TINY_CONFIG, "state_dict": entries, "cases": cases}


def encoder_golden() -> dict[str, object]:
    stack = fill_value("encoder.stack", ENCODER_STACK_SHAPE)
    mask = torch.tensor(ENCODER_ATTENTION_MASK, dtype=torch.long)

    projection = torch.nn.Linear(ENCODER_FEATURES, TINY_INNER, bias=False)
    projection_fill = [["text_embedding_projection.weight", [TINY_INNER, ENCODER_FEATURES]]]
    with torch.no_grad():
        projection.weight.copy_(fill_value(projection_fill[0][0], tuple(projection_fill[0][1])))

    original = model_management.should_use_bf16
    model_management.should_use_bf16 = lambda *args, **kwargs: False
    try:
        stub = SimpleNamespace(
            gemma3_12b=SimpleNamespace(
                encode_token_weights=lambda pairs: (
                    stack.clone(),
                    None,
                    {"attention_mask": mask.clone()},
                )
            ),
            text_encoder_key="gemma3_12b",
            text_projection_type="single_linear",
            execution_device=torch.device("cpu"),
            text_embedding_projection=projection,
            compat_mode=True,
            video_embeddings_connector=tiny_reference_connector("video_embeddings_connector."),
            audio_embeddings_connector=tiny_reference_connector("audio_embeddings_connector."),
        )
        with torch.no_grad():
            out, pooled, extra = lt.LTXAVTEModel.encode_token_weights(stub, {"gemma3_12b": []})
    finally:
        model_management.should_use_bf16 = original
    if pooled is not None or extra != {}:
        raise AssertionError("unexpected reference encoder extras on the connector path")
    if list(out.shape) != [1, 1024, 2 * TINY_INNER]:
        raise AssertionError(f"unexpected encoder output shape {list(out.shape)}")
    return {
        "stack": enc(stack),
        "attention_mask": list(ENCODER_ATTENTION_MASK),
        "attended_tokens": int(mask.sum().item()),
        "projection_fill": projection_fill,
        "video_fill_prefix": "video_embeddings_connector.",
        "audio_fill_prefix": "audio_embeddings_connector.",
        "output": enc(out),
    }


def main() -> None:
    commit = git_head(COMFY_ROOT)
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    dirty = git_dirty(COMFY_ROOT)
    if dirty:
        raise SystemExit(f"{COMFY_ROOT} has uncommitted changes:\n{dirty}")
    if torch.__version__ != GENERATOR_TORCH:
        raise SystemExit(
            f"interpreter runs torch {torch.__version__}; goldens are certified"
            f" under {GENERATOR_TORCH}"
        )
    for module in (lt, embeddings_connector, _attention):
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"reference module imported from {module_file}")

    ltx2_header_path = required_input("DINKSTER_LTX2_19B_HEADER_JSON", LTX2_HEADER_SHA256)

    layout = full_layout()
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention_backend": ATTENTION_BACKEND,
        },
        "connector_layout": layout,
        "ltx2_19b_dev": ltx2_19b_dev(ltx2_header_path, layout),
        "tiny": tiny_golden(),
        "encoder": encoder_golden(),
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=True, indent=None, separators=(",", ":")) + "\n"
    )
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
