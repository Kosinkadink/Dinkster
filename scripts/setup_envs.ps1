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

function Get-FirstCredential {
    param([string[]]$Names)

    foreach ($Name in $Names) {
        $Value = [Environment]::GetEnvironmentVariable($Name, "Process")
        if ($Value) {
            return $Value
        }
    }
    return $null
}

function Invoke-PrivateInstaller {
    param(
        [string]$Python,
        [string[]]$Arguments,
        [string]$CredentialName,
        [AllowNull()][string]$Credential
    )

    try {
        if ($Credential) {
            [Environment]::SetEnvironmentVariable($CredentialName, $Credential, "Process")
        }
        Invoke-Native $Python $Arguments
    }
    finally {
        [Environment]::SetEnvironmentVariable($CredentialName, $null, "Process")
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

$CredentialNames = @("DINKSTER_AIMDO_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
$CredentialEnvironment = @{}
foreach ($Name in $CredentialNames) {
    $CredentialEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}
$AimdoToken = Get-FirstCredential @("DINKSTER_AIMDO_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
foreach ($Name in $CredentialNames) {
    [Environment]::SetEnvironmentVariable($Name, $null, "Process")
}

$RootEnvironment = Join-Path $RepoRoot ".venv"
$TorchEnvironment = Join-Path $RepoRoot ".venv-torch"
$GpuEnvironment = Join-Path $RepoRoot ".venv-gpu"
$SetupExtras = Join-Path $RepoRoot ".venv-gpu-extras"
$TorchPython = Join-Path $TorchEnvironment "Scripts\python.exe"
$GpuPython = Join-Path $GpuEnvironment "Scripts\python.exe"

$CpuEditablePackages = @(
    "packages/dinkster-api",
    "packages/dinkster-schema",
    "packages/dinkster-values",
    "packages/dinkster-video",
    "packages/dinkster-protocol",
    "packages/dinkster-assets",
    "packages/dinkster-caches",
    "packages/dinkster-inference",
    "packages/dinkster-kernels",
    "packages/dinkster-memory",
    "packages/dinkster-graph",
    "packages/dinkster-engine",
    "packages/dinkster-native",
    "packages/dinkster-inference-torch",
    "packages/dinkster-nodes-generation",
    "packages/dinkster-compat-comfy",
    "packages/dinkster-acceptance",
    "packages/dinkster-model-ipadapter",
    "packages/dinkster-model-qwen-image",
    "packages/dinkster-model-triposplat",
    "packages/dinkster-vision-birefnet",
    "packages/dinkster-vision-depth-anything-v2",
    "packages/dinkster-vision-depth-anything-v3",
    "packages/dinkster-vision-detr",
    "packages/dinkster-vision-efficient-sam",
    "packages/dinkster-vision-hed",
    "packages/dinkster-vision-rtdetr",
    "packages/dinkster-vision-sam31",
    "packages/dinkster-vision-upscale",
    "packages/dinkster-workers",
    "packages/dinkster-training-torch[torch]"
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
    "packages/dinkster-kernels",
    "packages/dinkster-memory",
    "packages/dinkster-native",
    "packages/dinkster-inference-torch",
    "packages/dinkster-workers",
    "packages/dinkster-nodes-generation",
    "packages/dinkster-compat-comfy",
    "packages/dinkster-acceptance",
    "packages/dinkster-model-ipadapter",
    "packages/dinkster-model-triposplat",
    "packages/dinkster-model-wan",
    "packages/dinkster-training-torch[torch]"
)
$KitchenWheel = "comfy-kitchen@https://files.pythonhosted.org/packages/a3/43/ceed9307bf92bccdc420703c3800ed46eafcafbfd764cbd93726f43db2b6/comfy_kitchen-0.2.32-py3-none-any.whl#sha256=6a5fba5224abbb7c9d8248bb7fe607bfab26ee623d311fcfae70066f1c7cfd9b"
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
    Invoke-Native "uv" @("sync", "--project", $RepoRoot, "--all-packages")
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
        "safetensors==0.8.0", "transformers==5.16.1", $KitchenWheel
    ) + (Get-EditableArguments $CpuEditablePackages)
    Invoke-Native "uv" (@("pip", "install", "--python", $TorchPython) + $CpuDependencies)
    Invoke-PrivateInstaller $TorchPython @("scripts\install_dinkster_aimdo.py") `
        "DINKSTER_AIMDO_TOKEN" $AimdoToken
    Invoke-Native $TorchPython @(
        "-c",
        "from importlib.metadata import version; import torch; assert torch.__version__ == '2.13.0+cpu'; assert version('torchvision') == '0.28.0+cpu'; assert version('comfy-kitchen') == '0.2.32'"
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
            "safetensors==0.8.0", "sentencepiece==0.2.1", $KitchenWheel,
            "triton-windows==3.7.1.post27"
        ) + (Get-EditableArguments $GpuEditablePackages)
        Invoke-Native "uv" (@("pip", "install", "--python", $GpuPython) + $GpuDependencies)
        Invoke-PrivateInstaller $GpuPython @("scripts\install_dinkster_aimdo.py") `
            "DINKSTER_AIMDO_TOKEN" $AimdoToken
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
    Write-Host "  .venv\Scripts\pyright.exe -p packages\dinkster-training-torch"
    Write-Host "  .venv-torch\Scripts\python.exe -m pytest -q packages\dinkster-training-torch\tests"
    if ($HasNvidiaGpu) {
        Write-Host '  $env:DINKSTER_ENABLE_GPU_TESTS = "1"'
        Write-Host '  $env:DINKSTER_VALIDATE_REFERENCE_GOLDENS = "1"'
        Write-Host "  .venv\Scripts\pyright.exe -p packages\dinkster-kernels"
        Write-Host "  .venv-gpu\Scripts\python.exe -m pytest -q packages\dinkster-kernels\tests"
        Write-Host "  .venv-gpu\Scripts\python.exe -m pytest -q packages\dinkster-inference-torch\tests"
        Write-Host "  .venv-gpu\Scripts\python.exe -m pytest -q packages\dinkster-training-torch\tests"
        Write-Host "  Remove-Item Env:\DINKSTER_ENABLE_GPU_TESTS"
        Write-Host "  Remove-Item Env:\DINKSTER_VALIDATE_REFERENCE_GOLDENS"
    }
}
finally {
    [Environment]::SetEnvironmentVariable("UV_PROJECT", $PreviousProject, "Process")
    [Environment]::SetEnvironmentVariable(
        "UV_PROJECT_ENVIRONMENT", $PreviousProjectEnvironment, "Process"
    )
    foreach ($Name in $CredentialNames) {
        [Environment]::SetEnvironmentVariable(
            $Name, $CredentialEnvironment[$Name], "Process"
        )
    }
    Pop-Location
}
