"""Generate EfficientSAM-Ti vectors from pinned official ONNX exports.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_efficient_sam_golden.py \
        /path/to/EfficientSAM-at-d525f62 \
        /path/to/efficient_sam_vitt_encoder.onnx \
        /path/to/efficient_sam_vitt_decoder.onnx

Use Python 3.12 with NumPy 2.5.1 and ONNX Runtime 1.29.0 on an AMD AVX2
CPU without AVX-512.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASELINE = "d525f622e6f640acf5a0fc37c7ca1f243da5bde0"
ENCODER_SHA256 = "84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951"
DECODER_SHA256 = "a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11"
INTRA_OP_NUM_THREADS = 1
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "efficient_sam_vitt_d525f62.json"


def _source() -> np.ndarray:
    height, width = 48, 64
    y, x = np.mgrid[:height, :width]
    frame = np.stack(
        (
            (x * 4 + y) % 256,
            (x + y * 5) % 256,
            ((x // 4) * 29 + (y // 4) * 17) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    disk = (x - 18) ** 2 + (y - 24) ** 2 < 10**2
    frame[disk] = (240, 30, 80)
    frame[10:40, 38:59] = (20, 220, 130)
    return frame


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _float_record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "shape": list(contiguous.shape),
        "float32Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"EfficientSAM must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("EfficientSAM checkout must be clean")


def _check_model(path: Path, expected: str, name: str) -> None:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise SystemExit(f"{name} SHA-256 does not match the pinned EfficientSAM artifact")


def _check_cpu_dispatch() -> None:
    try:
        fields = {
            key.strip(): value.strip()
            for line in Path("/proc/cpuinfo")
            .read_text(encoding="utf-8")
            .split("\n\n", 1)[0]
            .splitlines()
            if ":" in line
            for key, value in (line.split(":", 1),)
        }
    except OSError as error:
        raise SystemExit("EfficientSAM generation requires Linux CPU dispatch metadata") from error
    flags = set(fields.get("flags", "").split())
    if fields.get("vendor_id") != "AuthenticAMD" or "avx2" not in flags or "avx512f" in flags:
        raise SystemExit("EfficientSAM generation requires an AMD AVX2 host without AVX-512")


def _sessions(encoder: Path, decoder: Path) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_NUM_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(encoder, sess_options=options, providers=providers),
        ort.InferenceSession(decoder, sess_options=options, providers=providers),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("decoder", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    encoder_path = args.encoder.resolve()
    decoder_path = args.decoder.resolve()
    _check_reference(reference)
    _check_model(encoder_path, ENCODER_SHA256, "encoder")
    _check_model(decoder_path, DECODER_SHA256, "decoder")
    _check_cpu_dispatch()

    source = _source()
    image = np.ascontiguousarray(source.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    boxes = ((8.0, 7.0, 38.0, 35.0), (35.0, 8.0, 62.0, 44.0))
    encoder, decoder = _sessions(encoder_path, decoder_path)
    embedding = encoder.run(["image_embeddings"], {"batched_images": image})[0]
    size = np.asarray(source.shape[:2], dtype=np.int64)
    labels = np.asarray([[[2.0, 3.0]]], dtype=np.float32)
    logits: list[np.ndarray] = []
    ious: list[np.ndarray] = []
    for left, top, right, bottom in boxes:
        points = np.asarray([[[[left, top], [right, bottom]]]], dtype=np.float32)
        outputs = decoder.run(
            ["output_masks", "iou_predictions"],
            {
                "image_embeddings": embedding,
                "batched_point_coords": points,
                "batched_point_labels": labels,
                "orig_im_size": size,
            },
        )
        logits.append(outputs[0][0, 0])
        ious.append(outputs[1][0, 0])

    document = {
        "baseline": BASELINE,
        "boxes": [list(box) for box in boxes],
        "decoderSha256": DECODER_SHA256,
        "encoderSha256": ENCODER_SHA256,
        "intraOpNumThreads": INTRA_OP_NUM_THREADS,
        "ious": _float_record(np.stack(ious)),
        "logits": _float_record(np.stack(logits)),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "source": _record(source),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
