$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $UvDirectory = Join-Path $HOME ".local\bin"
    $UvExecutable = Join-Path $UvDirectory "uv.exe"
    if (-not (Test-Path $UvExecutable -PathType Leaf)) {
        Write-Host "Installing uv in your user directory (no administrator or profile changes)..."
        $PreviousInstallDirectory = $env:UV_INSTALL_DIR
        $PreviousModifyPath = $env:UV_NO_MODIFY_PATH
        try {
            $env:UV_INSTALL_DIR = $UvDirectory
            $env:UV_NO_MODIFY_PATH = "1"
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
            if (-not (Test-Path $UvExecutable -PathType Leaf)) {
                throw "uv installer did not produce $UvExecutable"
            }
        }
        catch {
            Write-Error "uv installation failed; check Internet access to https://astral.sh/uv/install.ps1: $_"
            exit 1
        }
        finally {
            $env:UV_INSTALL_DIR = $PreviousInstallDirectory
            $env:UV_NO_MODIFY_PATH = $PreviousModifyPath
        }
    }
    $env:PATH = "$UvDirectory;$env:PATH"
}
Push-Location $PSScriptRoot
try {
    & uv run --no-project --python 3.12 scripts/run.py @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
