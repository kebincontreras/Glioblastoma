@echo off
REM =============================================================================
REM Complete Troubleshooting Script - Windows
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo   Complete Python Environment Troubleshooting
echo ============================================

set ENV_NAME=gbm_env

echo [Step 1] Deactivating any active environments...
call deactivate >nul 2>&1 || echo   No active environment to deactivate

echo [Step 2] Removing corrupted environment...
if exist "%ENV_NAME%" (
    echo   Removing %ENV_NAME%...
    rmdir /s /q "%ENV_NAME%" >nul 2>&1
    if exist "%ENV_NAME%" (
        echo   Forcing removal with administrative privileges...
        powershell -Command "Remove-Item -Path '%ENV_NAME%' -Recurse -Force" >nul 2>&1
    )
    if exist "%ENV_NAME%" (
        echo   Warning: Could not remove %ENV_NAME%. Please remove manually.
    ) else (
        echo   Environment removed successfully
    )
) else (
    echo   No environment to remove
)

echo [Step 3] Cleaning Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1
del /s /q *.pyc >nul 2>&1
echo   Cache files cleaned

echo [Step 4] Checking and fixing pip...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   pip not working, attempting to fix...
    python -m ensurepip --upgrade --default-pip >nul 2>&1
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo   Downloading get-pip.py...
        curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
        if exist "get-pip.py" (
            python get-pip.py --user >nul 2>&1
            del "get-pip.py" >nul 2>&1
        )
    )
) else (
    echo   pip is working correctly
)

echo [Step 5] Checking venv module...
python -m venv --help >nul 2>&1
if %errorlevel% equ 0 (
    echo   venv module is available
) else (
    echo   Warning: venv module not available - please reinstall Python
)

echo [Step 6] Cleaning temporary files...
if exist "*.tmp" del "*.tmp" >nul 2>&1
if exist "temp" rmdir /s /q "temp" >nul 2>&1
echo   Temporary files cleaned

echo [Step 7] Checking disk space...
for /f "tokens=3" %%i in ('dir /-c . ^| findstr /C:"bytes free"') do set FREE_SPACE=%%i
if defined FREE_SPACE (
    echo   Disk space available: %FREE_SPACE% bytes
) else (
    echo   Could not check disk space
)

echo.
echo ============================================
echo   Troubleshooting completed!
echo ============================================
echo.

REM Only pause if script is run manually (not called from another script)
if "%1"=="auto" (
    echo Returning to main script...
) else (
    echo You can now try running the main script again.
    echo If problems persist, run: scripts\diagnose.bat
    pause
)
