<#
.SYNOPSIS
  clawde one-shot installer (Windows / PowerShell 5.1+).
.DESCRIPTION
  Installs uv + Python 3.12, syncs the workspace, installs the pre-commit hook,
  and creates .env from .env.example. Run once after cloning.
.EXAMPLE
  .\setup.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host "    [OK]   $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "    [WARN] $Msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg) Write-Host "`n    [FAIL] $Msg`n" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  clawde -- Setup" -ForegroundColor White
Write-Host "  ===============" -ForegroundColor DarkGray

# -- 1. uv -------------------------------------------------------------------
Write-Step "Checking uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warn "uv not found -- installing..."
    try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }
    catch { Write-Fail "Could not auto-install uv. See https://docs.astral.sh/uv/getting-started/installation/" }
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Fail "uv installed but not on PATH yet. Open a new terminal and re-run .\setup.ps1"
    }
}
Write-Ok "uv $(uv --version)"

# -- 2. Python 3.12 ----------------------------------------------------------
Write-Step "Checking Python 3.12"
$null = uv python find "3.12" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Installing Python 3.12 via uv..."
    uv python install "3.12"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install Python 3.12" }
}
Write-Ok "Python 3.12 ready"

# -- 3. Workspace ------------------------------------------------------------
Write-Step "Syncing the workspace (uv sync --all-packages)"
uv sync --all-packages
if ($LASTEXITCODE -ne 0) { Write-Fail "uv sync failed -- check the output above" }
Write-Ok "Python packages installed"

# -- 4. pre-commit -----------------------------------------------------------
Write-Step "Installing the pre-commit hook"
uv run pre-commit install | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Warn "pre-commit install failed -- run 'uv run pre-commit install' manually" }
else { Write-Ok "pre-commit hook installed" }

# -- 5. .env -----------------------------------------------------------------
Write-Step "Configuring environment"
$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $PSScriptRoot ".env.example") $envPath
    Write-Ok ".env created from .env.example -- add a provider key (BYOM)"
} else {
    Write-Ok ".env already exists"
}

Write-Host ""
Write-Host "  ===============" -ForegroundColor DarkGray
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Try it:" -ForegroundColor White
Write-Host "    uv run clawde version" -ForegroundColor Cyan
Write-Host "    uv run --directory packages/core pytest" -ForegroundColor Cyan
Write-Host ""
