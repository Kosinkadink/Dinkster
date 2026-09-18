#!/usr/bin/env bash
# Build the three validation environments the AGENTS.md gates need,
# from a fresh checkout, with no sudo and no system mutation:
#
#   .venv        torch-free root env        (uv sync --all-packages)
#   .venv-torch  CPU torch test env         (torch README "Setup: torch
#                                            environments")
#   .venv-gpu    CUDA torch test env        (torch README "GPU
#                                            validation"; built only when
#                                            nvidia-smi reports a GPU)
#   .venv-gpu-extras/pyheaders              shared local Python headers
#
# This script is the source of truth for exact environment setup commands;
# README.md and packages/dinkster-inference-torch/README.md explain their use.
# Idempotent: existing venvs are kept and their
# installs re-asserted (cheap no-ops when already satisfied); pass
# --force to delete and rebuild all venvs.
#
# The comfy-kitchen 0.2.32 CPU wheel builds from the immutable release commit,
# fetched into a temporary repo when the workspace sibling lacks that tag.
#
# On macOS (Darwin) the torch env installs the native arm64 PyPI wheels
# instead: torch's mac build ships MPS support in the one default wheel,
# and comfy-kitchen's only mac-compatible distribution is its pure-Python
# PyPI wheel (eager/triton backends - the CPU flavor this env wants).
# The setup finishes with scripts/mps_smoke.py, which reports what the
# machine's MPS device can actually do.

set -euo pipefail

cd "$(dirname "$0")/.."

os=$(uname -s)
aimdo_token=${DINKSTER_AIMDO_TOKEN:-${GH_TOKEN:-${GITHUB_TOKEN:-}}}
export -n aimdo_token
unset DINKSTER_AIMDO_TOKEN GH_TOKEN GITHUB_TOKEN

install_dinkster_aimdo() {
    if [ -n "$aimdo_token" ]; then
        DINKSTER_AIMDO_TOKEN="$aimdo_token" "$1" scripts/install_dinkster_aimdo.py
    else
        "$1" scripts/install_dinkster_aimdo.py
    fi
}

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "usage: scripts/setup_envs.sh [--force]" >&2; exit 2 ;;
    esac
done

command -v uv >/dev/null || {
    echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
}

if [ "$FORCE" = 1 ]; then
    rm -rf .venv .venv-torch .venv-gpu .venv-gpu-extras
fi

# ---------------------------------------------------------------- .venv
echo "==> .venv (torch-free root env)"
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD" --all-packages

# ---------------------------------------------------------- .venv-torch
echo "==> .venv-torch (CPU torch test env)"
[ -x .venv-torch/bin/python ] || uv venv .venv-torch --python 3.12
if [ "$os" = "Darwin" ]; then
    # The mac torch wheel is one native arm64 build with MPS included;
    # PyPI is its canonical source (the cpu index mirrors it).
    uv pip install --python .venv-torch/bin/python torch "torchvision==0.28.0"
else
    uv pip install --python .venv-torch/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch "torchvision==0.28.0"
fi
# numpy is declared in the package's torch extra, not its base
# dependencies, and torch 2.13 CPU wheels no longer depend on it.
# The model node packs carry their own torch-side tests. Pillow supports
# TripoSplat preprocessing, OpenCV supports HED preprocessing, and scipy
# supports WanDancer audio features.
# dinkster-workers (and its values/protocol/assets/caches deps) makes
# scripts/benchmark_inference.py importable here, so torch-dependent
# benchmark-harness tests can run in this venv.
uv pip install --python .venv-torch/bin/python pytest packaging "numpy>=1.26" "scipy>=1.11" \
    "simpleeval==1.0.3" \
    "onnxruntime==1.29.0" "opencv-python-headless==5.0.0.93" "pillow==12.0.0" \
    "safetensors==0.8.0" "transformers==5.16.1" \
    -e packages/dinkster-api \
    -e packages/dinkster-schema \
    -e packages/dinkster-values \
    -e packages/dinkster-video \
    -e packages/dinkster-protocol \
    -e packages/dinkster-assets \
    -e packages/dinkster-caches \
    -e packages/dinkster-inference \
    -e packages/dinkster-kernels \
    -e packages/dinkster-memory \
    -e packages/dinkster-graph \
    -e packages/dinkster-engine \
    -e packages/dinkster-native \
    -e packages/dinkster-inference-torch \
    -e packages/dinkster-nodes-generation \
    -e packages/dinkster-compat-comfy \
    -e packages/dinkster-acceptance \
    -e packages/dinkster-model-ipadapter \
    -e packages/dinkster-model-qwen-image \
    -e packages/dinkster-model-triposplat \
    -e packages/dinkster-vision-birefnet \
    -e packages/dinkster-vision-depth-anything-v2 \
    -e packages/dinkster-vision-depth-anything-v3 \
    -e packages/dinkster-vision-detr \
    -e packages/dinkster-vision-efficient-sam \
    -e packages/dinkster-vision-hed \
    -e packages/dinkster-vision-rtdetr \
    -e packages/dinkster-vision-sam31 \
    -e packages/dinkster-vision-upscale \
    -e packages/dinkster-workers \
    -e 'packages/dinkster-training-torch[torch]'

# comfy-kitchen CPU wheel (required by INT8 ConvRot). On Linux, build only
# from an immutable archive of the release commit, never the sibling
# working tree - PyPI would deliver the CUDA wheel there. On macOS no
# platform wheel exists, so PyPI resolves the pure-Python wheel
# (eager/triton backends), which is already the wanted CPU flavor.
if [ "$os" = "Darwin" ]; then
    echo "==> comfy-kitchen 0.2.32 (PyPI pure-Python wheel)"
    uv pip install --python .venv-torch/bin/python "comfy-kitchen==0.2.32"
else
    kitchen_commit=f0092e814e73c0e82e9bfa364d55bad8f84280b6
    kitchen_source=../comfy-kitchen
    sibling_tag=$(git -C "$kitchen_source" rev-parse v0.2.32^{commit} 2>/dev/null || true)
    if ! git -C "$kitchen_source" cat-file -e "$kitchen_commit^{commit}" 2>/dev/null || \
        [ "$sibling_tag" != "$kitchen_commit" ]; then
        kitchen_source=/tmp/ck-source
        rm -rf "$kitchen_source"
        git init -q "$kitchen_source"
        git -C "$kitchen_source" remote add origin https://github.com/Comfy-Org/comfy-kitchen.git
        git -C "$kitchen_source" fetch -q --depth=1 origin tag v0.2.32
    fi
    if [ "$(git -C "$kitchen_source" rev-parse v0.2.32^{commit})" != "$kitchen_commit" ]; then
        echo "error: comfy-kitchen v0.2.32 does not resolve to $kitchen_commit" >&2
        exit 1
    fi
    echo "==> comfy-kitchen CPU wheel (release $kitchen_commit)"
    rm -rf /tmp/ck-build /tmp/ck-venv
    mkdir /tmp/ck-build
    git -C "$kitchen_source" archive "$kitchen_commit" | tar -x -C /tmp/ck-build
    uv venv /tmp/ck-venv -q
    uv pip install --python /tmp/ck-venv/bin/python -q setuptools wheel
    (cd /tmp/ck-build && /tmp/ck-venv/bin/python setup.py bdist_wheel --no-cuda)
    kitchen_wheel=$(find /tmp/ck-build/dist -maxdepth 1 -type f \
        -name 'comfy_kitchen-0.2.32-*.whl' -print -quit)
    if [ -z "$kitchen_wheel" ]; then
        echo "error: comfy-kitchen did not build the pinned 0.2.32 wheel" >&2
        exit 1
    fi
    uv pip install --python .venv-torch/bin/python --reinstall "$kitchen_wheel"
fi
if [ "$os" != "Darwin" ]; then
    install_dinkster_aimdo .venv-torch/bin/python
fi
.venv-torch/bin/python -c "from importlib.metadata import version; assert version('comfy-kitchen') == '0.2.32'"

# ------------------------------------------------ shared Python headers
# Torch inductor CPU and CUDA compilation both include Python.h. The CPU and
# CUDA venvs use the same Python 3.12 base interpreter, so check it once and
# extract one shared local header tree when the base installation is incomplete.
python_include=$(.venv-torch/bin/python - <<'EOF'
import sysconfig
print(sysconfig.get_paths()["include"])
EOF
)
python_minor=$(.venv-torch/bin/python - <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)
python_headers_root="$PWD/.venv-gpu-extras/pyheaders"
extracted_python_include="$python_headers_root/usr/include/python$python_minor"
if [ -f "$python_include/Python.h" ]; then
    echo "==> Python.h present in the Python $python_minor base interpreter"
elif [ -f "$extracted_python_include/Python.h" ]; then
    echo "==> shared extracted Python $python_minor headers already present"
elif command -v apt-get >/dev/null && command -v dpkg >/dev/null; then
    python_dev_package="libpython$python_minor-dev"
    echo "==> extracting $python_dev_package headers for CPU and CUDA compilation (no sudo)"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    if ! (cd "$tmp" && apt-get download "$python_dev_package"); then
        echo "error: Python.h is missing and $python_dev_package could not be downloaded" >&2
        echo "  install matching Python $python_minor development headers or make" >&2
        echo "  Python.h available under $python_include before running the test suites" >&2
        exit 1
    fi
    python_dev_deb=$(find "$tmp" -maxdepth 1 -type f -name "$python_dev_package*.deb" -print -quit)
    if [ -z "$python_dev_deb" ]; then
        echo "error: $python_dev_package download produced no Debian package" >&2
        exit 1
    fi
    mkdir -p "$python_headers_root"
    dpkg -x "$python_dev_deb" "$python_headers_root"
    if [ ! -f "$extracted_python_include/Python.h" ]; then
        echo "error: $python_dev_package did not provide $extracted_python_include/Python.h" >&2
        exit 1
    fi
    rm -rf "$tmp"
    trap - EXIT
else
    echo "error: Python.h is missing for the Python $python_minor base interpreter" >&2
    echo "  install matching Python development headers or make Python.h available" >&2
    echo "  under $python_include; apt-get and dpkg are unavailable for local extraction" >&2
    exit 1
fi

# ------------------------------------------------------------ .venv-gpu
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    echo "==> .venv-gpu (CUDA torch test env)"
    [ -x .venv-gpu/bin/python ] || uv venv .venv-gpu --python 3.12
    uv pip install --python .venv-gpu/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
    # scipy, torchsde, tqdm, and Pillow satisfy the pinned ComfyUI
    # k_diffusion import chain used by the native GPU reference proofs;
    # safetensors and sentencepiece support the package's model fixtures.
    uv pip install --python .venv-gpu/bin/python \
        pytest numpy scipy torchsde tqdm pillow packaging \
        "safetensors==0.8.0" "sentencepiece==0.2.1" \
        comfy-kitchen==0.2.32 \
        -e packages/dinkster-api \
        -e packages/dinkster-schema \
        -e packages/dinkster-values \
        -e packages/dinkster-video \
        -e packages/dinkster-protocol \
        -e packages/dinkster-assets \
        -e packages/dinkster-caches \
        -e packages/dinkster-inference \
        -e packages/dinkster-graph \
        -e packages/dinkster-engine \
        -e packages/dinkster-kernels \
        -e packages/dinkster-memory \
        -e packages/dinkster-native \
        -e packages/dinkster-inference-torch \
        -e packages/dinkster-workers \
        -e packages/dinkster-nodes-generation \
        -e packages/dinkster-compat-comfy \
        -e packages/dinkster-acceptance \
        -e packages/dinkster-model-ipadapter \
        -e packages/dinkster-model-triposplat \
        -e packages/dinkster-model-wan \
        -e 'packages/dinkster-training-torch[torch]'

    install_dinkster_aimdo .venv-gpu/bin/python
else
    echo "==> no NVIDIA GPU detected - skipping .venv-gpu (the GPU gate"
    echo "    applies only on GPU machines, AGENTS.md 'Validation gate')"
fi

# Apple Silicon only: on such machines a missing MPS device means a broken
# torch install, so the smoke report's failure should fail the setup. Intel
# macs have no MPS to probe and skip it.
if [ "$os" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "==> MPS smoke report (scripts/mps_smoke.py)"
    .venv-torch/bin/python scripts/mps_smoke.py
fi

echo "==> done. Gates:"
echo "  .venv/bin/ruff check ."
echo "  .venv/bin/pyright"
echo "  .venv/bin/python -m pytest -q"
echo "  .venv/bin/pyright -p packages/dinkster-inference-torch"
echo '  CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \'
echo "    .venv-torch/bin/python -m pytest -q packages/dinkster-inference-torch/tests"
echo "  .venv/bin/pyright -p packages/dinkster-training-torch"
echo "  .venv-torch/bin/python -m pytest -q packages/dinkster-training-torch/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-hed"
echo "  DINKSTER_HED_TEST_MODEL=/path/to/ControlNetHED.pth .venv-torch/bin/python -m pytest -q packages/dinkster-vision-hed/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-upscale"
echo "  .venv-torch/bin/python -m pytest -q packages/dinkster-vision-upscale/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-depth-anything-v2"
echo "  DINKSTER_DEPTH_ANYTHING_V2_TEST_MODEL=/path/to/model.safetensors .venv-torch/bin/python -m pytest -q packages/dinkster-vision-depth-anything-v2/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-depth-anything-v3"
echo "  DINKSTER_DEPTH_ANYTHING_V3_TEST_MODEL=/path/to/model.safetensors .venv-torch/bin/python -m pytest -q packages/dinkster-vision-depth-anything-v3/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-detr"
echo "  DINKSTER_DETR_TEST_MODEL=/path/to/detr-r50-e632da11.pth .venv-torch/bin/python -m pytest -q packages/dinkster-vision-detr/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-rtdetr"
echo "  DINKSTER_RTDETR_TEST_MODEL=/path/to/rt_detr_v4-x-hgnet_fp16.safetensors .venv-torch/bin/python -m pytest -q packages/dinkster-vision-rtdetr/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-efficient-sam"
echo "  DINKSTER_EFFICIENT_SAM_TEST_ENCODER=/path/to/encoder.onnx DINKSTER_EFFICIENT_SAM_TEST_DECODER=/path/to/decoder.onnx .venv-torch/bin/python -m pytest -q packages/dinkster-vision-efficient-sam/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-birefnet"
echo "  DINKSTER_BIREFNET_TEST_MODEL=/path/to/birefnet.safetensors .venv-torch/bin/python -m pytest -q packages/dinkster-vision-birefnet/tests"
echo "  .venv/bin/pyright -p packages/dinkster-vision-sam31"
echo "  DINKSTER_SAM31_TEST_MODEL=/path/to/sam3.1_multiplex_fp16.safetensors .venv-torch/bin/python -m pytest -q packages/dinkster-vision-sam31/tests"
echo "GPU machines additionally (dinkster-kernels resolves against .venv-gpu):"
echo "  .venv/bin/pyright -p packages/dinkster-kernels"
echo '  CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \'
echo "    .venv-gpu/bin/python -m pytest -q packages/dinkster-kernels/tests"
echo '  DINKSTER_ENABLE_GPU_TESTS=1 DINKSTER_VALIDATE_REFERENCE_GOLDENS=1 \'
echo '  CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \'
echo "    .venv-gpu/bin/python -m pytest -q packages/dinkster-inference-torch/tests"
