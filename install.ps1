$ErrorActionPreference = "Stop"

Write-Host "HoloForge installer"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "uv is not installed and winget is unavailable. Install uv from https://docs.astral.sh/uv/ then run this script again."
    }
    Write-Host "Installing the uv Python manager..."
    winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    $uvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path -LiteralPath $uvPath)) {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uv) { throw "uv installed, but this shell cannot find it. Open a new terminal and rerun install.ps1." }
        $uvPath = $uv.Source
    }
} else {
    $uvPath = $uv.Source
}

& $uvPath python install 3.12
& $uvPath sync --python 3.12

Write-Host ""
Write-Host "Installation complete. Run .\run.ps1 to open HoloForge."
Write-Host "AI model weights download on first use and are then cached locally."
