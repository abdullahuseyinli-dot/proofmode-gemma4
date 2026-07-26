[CmdletBinding()]
param([switch]$Live)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ProofMode virtual environment is missing: $python"
}

& $python -m compileall -q (Join-Path $projectRoot 'app.py') (Join-Path $projectRoot 'proofmode') (Join-Path $projectRoot 'desktop_launcher.py')
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
& $python -m pytest -q (Join-Path $projectRoot 'tests')
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

if ($Live) {
    $gemma = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 3
    $app = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 3
    if ($gemma.status -ne 'ok' -or $app.StatusCode -ne 200) { throw 'A live service is unhealthy.' }
}

Write-Output 'ProofMode verification passed.'

