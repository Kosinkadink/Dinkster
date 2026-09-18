#!/usr/bin/env bash
# Isolated Intel XPU backend environment (Linux), pinned to the PyTorch 2.13
# XPU support cell. Run from the repository root.
#
# Prerequisites: Python 3.12 and uv on PATH, plus the Intel compute runtime
# and Level Zero packages per the PyTorch 2.13 XPU notes.
#
# The pins here must match dinkster_workers.backend_env.BACKEND_ENV_RECIPES
# ("linux-xpu"); tests/test_backend_env.py asserts they stay in sync.
set -euo pipefail

uv venv .venv-xpu --clear --python 3.12

uv pip install --python .venv-xpu/bin/python \
    --index-url https://download.pytorch.org/whl/xpu \
    "torch==2.13.0+xpu"

uv pip install --python .venv-xpu/bin/python \
    pytest numpy scipy torchsde tqdm pillow packaging \
    "comfy-kitchen@https://files.pythonhosted.org/packages/a3/43/ceed9307bf92bccdc420703c3800ed46eafcafbfd764cbd93726f43db2b6/comfy_kitchen-0.2.32-py3-none-any.whl#sha256=6a5fba5224abbb7c9d8248bb7fe607bfab26ee623d311fcfae70066f1c7cfd9b" \
    -e packages/dinkster-schema \
    -e packages/dinkster-graph \
    -e packages/dinkster-values \
    -e packages/dinkster-protocol \
    -e packages/dinkster-caches \
    -e packages/dinkster-assets \
    -e packages/dinkster-memory \
    -e packages/dinkster-workers \
    -e packages/dinkster-inference \
    -e packages/dinkster-inference-torch \
    -e packages/dinkster-image-document \
    -e packages/dinkster-video \
    -e packages/dinkster-api \
    -e packages/dinkster-nodes-generation \
    -e packages/dinkster-native \
    -e packages/dinkster-compat-comfy

.venv-xpu/bin/python -c "import dinkster_compat_comfy.native_arm"

.venv-xpu/bin/python scripts/xpu_smoke.py --json xpu-report.json
