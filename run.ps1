$ErrorActionPreference = "Stop"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    $candidate = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path -LiteralPath $candidate) { $uvPath = $candidate }
    else { throw "uv was not found. Run .\install.ps1 first." }
} else {
    $uvPath = $uv.Source
}
& $uvPath run holoforge-ui @args

