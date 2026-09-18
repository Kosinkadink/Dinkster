#!/usr/bin/env bash
# Isolated ROCm backend environment (Linux), pinned to the ROCm 7.14 /
# PyTorch 2.12 support cell. Run from the repository root.
#
# Prerequisites: Python 3.12 and uv on PATH, plus a ROCm 7.14 host install
# per the ROCm compatibility matrix.
#
# The pins here must match dinkster_workers.backend_env.BACKEND_ENV_RECIPES
# ("linux-rocm"); tests/test_backend_env.py asserts they stay in sync.
set -euo pipefail

uv venv .venv-rocm --clear --python 3.12

uv pip install --python .venv-rocm/bin/python \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-all]==2.12.0+rocm7.14.0"

uv pip install --python .venv-rocm/bin/python \
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

.venv-rocm/bin/python -c "import dinkster_compat_comfy.native_arm"

.venv-rocm/bin/python scripts/rocm_smoke.py --json rocm-report.json
