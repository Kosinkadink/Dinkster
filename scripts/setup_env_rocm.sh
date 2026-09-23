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
    pytest numpy scipy torchsde tqdm pillow packaging "tokenizers==0.23.1" \
    "dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/dinkster_kitchen-0.2.35.post1-py3-none-any.whl#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f" \
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
