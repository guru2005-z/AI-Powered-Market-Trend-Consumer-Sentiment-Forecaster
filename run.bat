@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   AI Market Trend ^& Consumer Sentiment Forecaster
echo ============================================================

set "PY_EXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
) else (
    set "PY_EXE=python"
)

echo Using Python: %PY_EXE%
"%PY_EXE%" main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with code %ERRORLEVEL%.
    pause
)

