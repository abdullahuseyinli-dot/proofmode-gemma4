[CmdletBinding()]
param(
    [string]$PythonPath,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (-not $PythonPath) {
        $gemmaPython = Join-Path (Split-Path $projectRoot -Parent) 'gemma4\.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $gemmaPython -PathType Leaf) {
            $PythonPath = $gemmaPython
        } else {
            $PythonPath = (Get-Command python -ErrorAction Stop).Source
        }
    }
    & $PythonPath -m venv (Join-Path $projectRoot '.venv')
}

if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
}

& $venvPython -m pytest -q (Join-Path $projectRoot 'tests')
Write-Output 'ProofMode setup and verification complete. Run .\launch-proofmode.cmd'

