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
# On macOS (Darwin) the torch env installs the native arm64 PyPI wheels
# instead: torch's mac build ships MPS support in the one default wheel,
# and dinkster-kitchen's mac-compatible distribution is its pure-Python
# PyPI wheel (eager/triton backends - the CPU flavor this env wants).
# The setup finishes with scripts/mps_smoke.py, which reports what the
# machine's MPS device can actually do.

set -euo pipefail

cd "$(dirname "$0")/.."

os=$(uname -s)
evidence_root=${DINKSTER_EVIDENCE_ROOT:-"$PWD/../dinkster-evidence"}
acceptance_package="$evidence_root/packages/dinkster-acceptance"

if [ -d "$acceptance_package" ]; then
    install_acceptance=1
else
    install_acceptance=0
    echo "==> dinkster-evidence not found - skipping optional dinkster-acceptance"
fi

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
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project "$PWD" --python 3.12 --all-packages \
    --reinstall-package dinkster-nodes-std

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
# the benchmark scripts in dinkster-evidence importable, so torch-dependent
# benchmark-harness tests can run in this venv.
uv pip install --python .venv-torch/bin/python --reinstall-package dinkster-nodes-std \
    pytest packaging "numpy>=1.26" "scipy>=1.11" \
    "simpleeval==1.0.3" \
    "onnxruntime==1.29.0" "opencv-python-headless==5.0.0.93" "pillow==12.0.0" \
    "safetensors==0.8.0" "sentencepiece==0.2.1" "transformers==5.16.1" \
    -e packages/dinkster-api \
    -e packages/dinkster-schema \
    -e packages/dinkster-values \
    -e packages/dinkster-video \
    -e packages/dinkster-protocol \
    -e packages/dinkster-assets \
    -e packages/dinkster-caches \
    -e packages/dinkster-inference \
    -e packages/dinkster-memory \
    -e packages/dinkster-graph \
    -e packages/dinkster-engine \
    -e packages/dinkster-native \
    -e packages/dinkster-inference-torch \
    -e packages/dinkster-nodes-generation \
    -e packages/dinkster-nodes-media-io \
    -e packages/dinkster-compat-comfy \
    -e packages/dinkster-model-ipadapter \
    -e packages/dinkster-model-qwen-image \
    -e packages/dinkster-model-triposplat \
    -e packages/dinkster-nodes-vision \
    -e packages/dinkster-workers \
    -e .
if [ "$install_acceptance" = 1 ]; then
    uv pip install --python .venv-torch/bin/python --no-deps --no-sources \
        -e "$acceptance_package"
fi

# The direct PyPI URL forces the device-agnostic wheel in the CPU environment;
# the platform wheels contain accelerator-specific native extensions.
kitchen_cpu_wheel="dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/dinkster_kitchen-0.2.35.post1-py3-none-any.whl#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f"
uv pip install --python .venv-torch/bin/python "$kitchen_cpu_wheel"
if [ "$os" != "Darwin" ]; then
    uv pip install --python .venv-torch/bin/python "dinkster-aimdo==0.5.5.post2"
fi
.venv-torch/bin/python -I -c "from importlib.metadata import version; import site; import av, dinkster.serve, dinkster_model_triposplat.provider, torch; assert site.ENABLE_USER_SITE is False; assert version('av') == '17.0.0'; assert version('dinkster-kitchen') == '0.2.35.post1'"

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
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 torchvision==0.28.0+cu130
    # scipy, torchsde, tqdm, and Pillow satisfy the pinned ComfyUI
    # k_diffusion import chain used by the native GPU reference proofs;
    # safetensors and sentencepiece support the package's model fixtures.
    uv pip install --python .venv-gpu/bin/python --reinstall-package dinkster-nodes-std \
        pytest numpy scipy torchsde tqdm pillow packaging \
        "safetensors==0.8.0" "sentencepiece==0.2.1" "tokenizers==0.23.1" \
        dinkster-kitchen==0.2.35.post1 dinkster-aimdo==0.5.5.post2 \
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
        -e packages/dinkster-memory \
        -e packages/dinkster-native \
        -e packages/dinkster-inference-torch \
        -e packages/dinkster-workers \
        -e packages/dinkster-nodes-generation \
        -e packages/dinkster-nodes-media-io \
        -e packages/dinkster-compat-comfy \
        -e packages/dinkster-model-ipadapter \
        -e packages/dinkster-model-triposplat \
        -e packages/dinkster-model-wan \
        -e .
    if [ "$install_acceptance" = 1 ]; then
        uv pip install --python .venv-gpu/bin/python --no-deps --no-sources \
            -e "$acceptance_package"
    fi
    .venv-gpu/bin/python -I -c "from importlib.metadata import version; import site; import av, dinkster.serve, dinkster_model_triposplat.provider, torch; assert site.ENABLE_USER_SITE is False; assert version('av') == '17.0.0'; assert torch.__version__ == '2.13.0+cu130'; assert version('torchvision') == '0.28.0+cu130'; assert version('tokenizers') == '0.23.1'"

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
echo "  .venv/bin/pyright -p packages/dinkster-nodes-vision"
echo "  .venv-torch/bin/python -m pytest -q packages/dinkster-nodes-vision/tests"
echo "GPU machines additionally:"
echo '  DINKSTER_ENABLE_GPU_TESTS=1 DINKSTER_VALIDATE_REFERENCE_GOLDENS=1 \'
echo '  CPATH="$PWD/.venv-gpu-extras/pyheaders/usr/include/python3.12:$PWD/.venv-gpu-extras/pyheaders/usr/include${CPATH:+:$CPATH}" \'
echo "    .venv-gpu/bin/python -m pytest -q packages/dinkster-inference-torch/tests"
