# Build the root, CPU Torch, and optional NVIDIA CUDA validation environments
# from a native Windows checkout. Existing environments are reasserted;
# -Force removes only the environments this script owns before rebuilding.

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-Native {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

function Get-EditableArguments {
    param([string[]]$Packages)

    $Arguments = @()
    foreach ($Package in $Packages) {
        $Arguments += @("-e", $Package)
    }
    return $Arguments
}

$RootEnvironment = Join-Path $RepoRoot ".venv"
$TorchEnvironment = Join-Path $RepoRoot ".venv-torch"
$GpuEnvironment = Join-Path $RepoRoot ".venv-gpu"
$SetupExtras = Join-Path $RepoRoot ".venv-gpu-extras"
$TorchPython = Join-Path $TorchEnvironment "Scripts\python.exe"
$GpuPython = Join-Path $GpuEnvironment "Scripts\python.exe"
$EvidenceRoot = $env:DINKSTER_EVIDENCE_ROOT
if (-not $EvidenceRoot) {
    $EvidenceRoot = Join-Path $RepoRoot "../dinkster-evidence"
}
$AcceptancePackage = Join-Path $EvidenceRoot "packages/dinkster-acceptance"
$InstallAcceptance = Test-Path $AcceptancePackage -PathType Container
if (-not $InstallAcceptance) {
    Write-Host "==> dinkster-evidence not found - skipping optional dinkster-acceptance"
}

$CpuEditablePackages = @(
    "packages/dinkster-api",
    "packages/dinkster-schema",
    "packages/dinkster-values",
    "packages/dinkster-video",
    "packages/dinkster-protocol",
    "packages/dinkster-assets",
    "packages/dinkster-caches",
    "packages/dinkster-inference",
    "packages/dinkster-memory",
    "packages/dinkster-graph",
    "packages/dinkster-engine",
    "packages/dinkster-native",
    "packages/dinkster-inference-torch",
    "packages/dinkster-nodes-generation",
    "packages/dinkster-compat-comfy",
    "packages/dinkster-model-ipadapter",
    "packages/dinkster-model-qwen-image",
    "packages/dinkster-model-triposplat",
    "packages/dinkster-model-yue2",
    "packages/dinkster-nodes-vision",
    "packages/dinkster-workers"
)
$GpuEditablePackages = @(
    "packages/dinkster-api",
    "packages/dinkster-schema",
    "packages/dinkster-values",
    "packages/dinkster-video",
    "packages/dinkster-protocol",
    "packages/dinkster-assets",
    "packages/dinkster-caches",
    "packages/dinkster-inference",
    "packages/dinkster-graph",
    "packages/dinkster-engine",
    "packages/dinkster-memory",
    "packages/dinkster-native",
    "packages/dinkster-inference-torch",
    "packages/dinkster-workers",
    "packages/dinkster-nodes-generation",
    "packages/dinkster-compat-comfy",
    "packages/dinkster-model-ipadapter",
    "packages/dinkster-model-triposplat",
    "packages/dinkster-model-yue2",
    "packages/dinkster-model-wan"
)
$KitchenCpuWheel = "dinkster-kitchen@https://files.pythonhosted.org/packages/2e/20/84e29ca1dedcd51eb5edd297d3c2f6c665cf2e30bb9237892f0f8d108d0d/dinkster_kitchen-0.2.35.post1-py3-none-any.whl#sha256=31458547cdcf9ff26974a4955cf79e83ebdf50077666720d3bb3255786c5fc4f"
$PreviousProject = [Environment]::GetEnvironmentVariable("UV_PROJECT", "Process")
$PreviousProjectEnvironment = [Environment]::GetEnvironmentVariable(
    "UV_PROJECT_ENVIRONMENT", "Process"
)

Push-Location $RepoRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required (https://docs.astral.sh/uv/)"
    }

    if ($Force) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            $RootEnvironment, $TorchEnvironment, $GpuEnvironment, $SetupExtras
    }

    Write-Host "==> .venv (torch-free root env)"
    [Environment]::SetEnvironmentVariable("UV_PROJECT", $null, "Process")
    [Environment]::SetEnvironmentVariable(
        "UV_PROJECT_ENVIRONMENT", $RootEnvironment, "Process"
    )
    Invoke-Native "uv" @(
        "sync", "--project", $RepoRoot, "--python", "3.12", "--all-packages"
    )
    [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $null, "Process")

    Write-Host "==> .venv-torch (CPU torch test env)"
    if (-not (Test-Path $TorchPython)) {
        Invoke-Native "uv" @("venv", $TorchEnvironment, "--python", "3.12")
    }
    Invoke-Native "uv" @(
        "pip", "install", "--python", $TorchPython,
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.13.0+cpu", "torchvision==0.28.0+cpu"
    )
    $CpuDependencies = @(
        "pytest", "packaging", "numpy>=1.26", "scipy>=1.11",
        "simpleeval==1.0.3", "onnxruntime==1.29.0",
        "opencv-python-headless==5.0.0.93", "pillow==12.0.0",
        "safetensors==0.8.0", "sentencepiece==0.2.1", "transformers==5.16.1",
        $KitchenCpuWheel,
        "dinkster-aimdo==0.5.5.post2"
    ) + (Get-EditableArguments $CpuEditablePackages)
    Invoke-Native "uv" (@("pip", "install", "--python", $TorchPython) + $CpuDependencies)
    if ($InstallAcceptance) {
        Invoke-Native "uv" @(
            "pip", "install", "--python", $TorchPython,
            "--no-deps", "--no-sources", "-e", $AcceptancePackage
        )
    }
    Invoke-Native $TorchPython @(
        "-c",
        "from importlib.metadata import version; import torch; assert torch.__version__ == '2.13.0+cpu'; assert version('torchvision') == '0.28.0+cpu'; assert version('dinkster-kitchen') == '0.2.35.post1'; assert version('dinkster-aimdo') == '0.5.5.post2'"
    )

    $PythonInclude = (& $TorchPython -c "import sysconfig; print(sysconfig.get_paths()['include'])")
    if ($LASTEXITCODE -ne 0) {
        throw "$TorchPython failed while locating Python headers"
    }
    if (-not (Test-Path (Join-Path $PythonInclude.Trim() "Python.h"))) {
        throw "Python.h is missing from the Python 3.12 interpreter at $PythonInclude"
    }
    Write-Host "==> Python.h present in the Python 3.12 base interpreter"

    $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $HasNvidiaGpu = $false
    if ($NvidiaSmi) {
        & $NvidiaSmi.Source -L *> $null
        $HasNvidiaGpu = $LASTEXITCODE -eq 0
    }

    if ($HasNvidiaGpu) {
        Write-Host "==> .venv-gpu (CUDA torch test env)"
        if (-not (Test-Path $GpuPython)) {
            Invoke-Native "uv" @("venv", $GpuEnvironment, "--python", "3.12")
        }
        Invoke-Native "uv" @(
            "pip", "install", "--python", $GpuPython,
            "--index-url", "https://download.pytorch.org/whl/cu130",
            "torch==2.13.0+cu130"
        )
        $GpuDependencies = @(
            "pytest", "numpy", "scipy", "torchsde", "tqdm", "pillow", "packaging",
            "safetensors==0.8.0", "sentencepiece==0.2.1",
            "dinkster-kitchen==0.2.35.post1", "dinkster-aimdo==0.5.5.post2",
            "triton-windows==3.7.1.post27"
        ) + (Get-EditableArguments $GpuEditablePackages)
        Invoke-Native "uv" (@("pip", "install", "--python", $GpuPython) + $GpuDependencies)
        if ($InstallAcceptance) {
            Invoke-Native "uv" @(
                "pip", "install", "--python", $GpuPython,
                "--no-deps", "--no-sources", "-e", $AcceptancePackage
            )
        }
        Invoke-Native $GpuPython @(
            "-c",
            "import torch, triton; assert torch.__version__ == '2.13.0+cu130'; assert triton.__version__ == '3.7.1'"
        )
    }
    else {
        Write-Host "==> no NVIDIA GPU detected - skipping .venv-gpu"
    }

    Write-Host "==> done. Gates:"
    Write-Host "  .venv\Scripts\ruff.exe check ."
    Write-Host "  .venv\Scripts\pyright.exe"
    Write-Host "  .venv\Scripts\python.exe -m pytest -q"
    Write-Host "  .venv\Scripts\pyright.exe -p packages\dinkster-inference-torch"
    Write-Host "  .venv-torch\Scripts\python.exe -m pytest -q packages\dinkster-inference-torch\tests"
    if ($HasNvidiaGpu) {
        Write-Host '  $env:DINKSTER_ENABLE_GPU_TESTS = "1"'
        Write-Host '  $env:DINKSTER_VALIDATE_REFERENCE_GOLDENS = "1"'
        Write-Host "  .venv-gpu\Scripts\python.exe -m pytest -q packages\dinkster-inference-torch\tests"
        Write-Host "  Remove-Item Env:\DINKSTER_ENABLE_GPU_TESTS"
        Write-Host "  Remove-Item Env:\DINKSTER_VALIDATE_REFERENCE_GOLDENS"
    }
}
finally {
    [Environment]::SetEnvironmentVariable("UV_PROJECT", $PreviousProject, "Process")
    [Environment]::SetEnvironmentVariable(
        "UV_PROJECT_ENVIRONMENT", $PreviousProjectEnvironment, "Process"
    )
    Pop-Location
}
