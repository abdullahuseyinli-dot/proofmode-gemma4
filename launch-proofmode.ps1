[CmdletBinding()]
param(
    [switch]$NoBubble,
    [switch]$NoOpen,
    [switch]$DefaultBrowser,
    [ValidateRange(1024, 131072)]
    [int]$ContextSize = 8192,
    [switch]$Wait
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$launcher = Join-Path $projectRoot 'desktop_launcher.py'

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "ProofMode launcher is missing: $launcher"
}

$configuredPython = [Environment]::GetEnvironmentVariable('PROOFMODE_PYTHON')
if ($configuredPython) {
    $python = $configuredPython
} else {
    $pythonw = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
    $pythonConsole = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $pythonw -PathType Leaf) {
        $python = $pythonw
    } elseif (Test-Path -LiteralPath $pythonConsole -PathType Leaf) {
        $python = $pythonConsole
    } else {
        throw "ProofMode's virtual environment is missing. Expected: $pythonConsole"
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Configured ProofMode Python executable does not exist: $python"
}

$launcherArguments = @('"' + $launcher + '"', '--context-size', $ContextSize)
if ($NoBubble) { $launcherArguments += '--no-bubble' }
if ($NoOpen) { $launcherArguments += '--no-open' }
if ($DefaultBrowser) { $launcherArguments += '--default-browser' }

$process = Start-Process `
    -FilePath $python `
    -ArgumentList ($launcherArguments -join ' ') `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

if ($Wait) {
    $process.WaitForExit()
    exit $process.ExitCode
}

Write-Output "ProofMode is starting in the background (launcher PID $($process.Id))."
