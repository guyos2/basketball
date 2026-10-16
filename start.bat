@echo off
rem Windows: double-click this file. The first run installs everything.
setlocal
cd /d "%~dp0"
title Basketball - shoot or pass

set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo Python 3.12 was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  echo Then double-click this file again.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the environment with %PY% ...
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
  echo Installing - the first run takes 5-10 minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

".venv\Scripts\python.exe" -m courtvision serve --open
pause
exit /b 0

:fail
echo.
echo Installation failed - see the messages above.
pause
exit /b 1
