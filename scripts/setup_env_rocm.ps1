# Isolated ROCm backend environment (native Windows), pinned to the ROCm 7.14
# / PyTorch 2.12 support cell. Run from the repository root.
#
# Prerequisites: Python 3.12 and uv on PATH, plus the validated AMD driver from
# the ROCm 7.14 compatibility matrix (Adrenalin 26.6.4 for RDNA, or a package
# the machine vendor explicitly identifies as OEM driver 26.10.28).
#
# The pins here must match dinkster_workers.backend_env.BACKEND_ENV_RECIPES
# ("windows-rocm"); tests/test_backend_env.py asserts they stay in sync.
$ErrorActionPreference = "Stop"

uv venv .venv-rocm --clear --python 3.12
if ($LASTEXITCODE -ne 0) { exit 1 }

uv pip install --python .venv-rocm\Scripts\python.exe `
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
    "torch[device-all]==2.12.0+rocm7.14.0"
if ($LASTEXITCODE -ne 0) { exit 1 }

uv pip install --python .venv-rocm\Scripts\python.exe `
    pytest numpy scipy torchsde tqdm pillow packaging `
    "dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/dinkster_kitchen-0.2.35.post1-py3-none-any.whl#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f" `
    -e packages/dinkster-schema `
    -e packages/dinkster-graph `
    -e packages/dinkster-values `
    -e packages/dinkster-protocol `
    -e packages/dinkster-caches `
    -e packages/dinkster-assets `
    -e packages/dinkster-memory `
    -e packages/dinkster-workers `
    -e packages/dinkster-inference `
    -e packages/dinkster-inference-torch `
    -e packages/dinkster-image-document `
    -e packages/dinkster-video `
    -e packages/dinkster-api `
    -e packages/dinkster-nodes-generation `
    -e packages/dinkster-nodes-media-io `
    -e packages/dinkster-model-triposplat `
    -e packages/dinkster-native `
    -e packages/dinkster-compat-comfy
if ($LASTEXITCODE -ne 0) { exit 1 }

.venv-rocm\Scripts\python.exe -c "import dinkster_compat_comfy.native_arm"
if ($LASTEXITCODE -ne 0) { exit 1 }

.venv-rocm\Scripts\python.exe scripts\rocm_smoke.py --json rocm-report.json
if ($LASTEXITCODE -ne 0) { exit 1 }
