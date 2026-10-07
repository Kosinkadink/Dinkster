$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
Push-Location $PSScriptRoot
try {
    & uv run --no-project --python 3.12 scripts/run.py @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
