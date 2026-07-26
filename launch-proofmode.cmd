@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launch-proofmode.ps1" %*
exit /b %ERRORLEVEL%
