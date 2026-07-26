[CmdletBinding()]
param(
    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'ProofMode.lnk')
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$launcher = Join-Path $projectRoot 'launch-proofmode.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "ProofMode launcher is missing: $launcher"
}

$shortcutTarget = [System.IO.Path]::GetFullPath($ShortcutPath)
$shortcutDirectory = Split-Path -Parent $shortcutTarget
if (-not (Test-Path -LiteralPath $shortcutDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $shortcutDirectory -Force | Out-Null
}

$powershell = (Get-Command 'powershell.exe' -ErrorAction Stop).Source
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutTarget)
$shortcut.TargetPath = $powershell
$shortcut.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'Launch the local ProofMode study companion'
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
$shortcut.Save()

Write-Output "Created ProofMode shortcut: $shortcutTarget"
