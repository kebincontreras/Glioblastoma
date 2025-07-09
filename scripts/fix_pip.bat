@echo off
REM =============================================================================
REM Fix pip installation - Windows
REM =============================================================================

echo ============================================
echo   Fixing pip installation
echo ============================================

echo Checking current Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo Attempting to fix pip installation...

REM Method 1: Try ensurepip
echo Method 1: Using ensurepip...
python -m ensurepip --upgrade --default-pip
if %errorlevel% equ 0 (
    echo pip installed successfully via ensurepip!
    goto :verify_pip
)

REM Method 2: Download and use get-pip.py
echo Method 2: Downloading get-pip.py...
if exist "get-pip.py" del "get-pip.py"

curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
if %errorlevel% neq 0 (
    echo Failed to download get-pip.py. Trying with PowerShell...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
    if %errorlevel% neq 0 (
        echo Error: Failed to download get-pip.py
        echo Please check your internet connection.
        pause
        exit /b 1
    )
)

echo Installing pip using get-pip.py...
python get-pip.py
if %errorlevel% equ 0 (
    echo pip installed successfully!
    del "get-pip.py"
) else (
    echo Error: Failed to install pip using get-pip.py
    pause
    exit /b 1
)

:verify_pip
echo Verifying pip installation...
python -m pip --version
if %errorlevel% equ 0 (
    echo ============================================
    echo   pip is now working correctly!
    echo ============================================
    python -m pip --version
) else (
    echo Error: pip is still not working
    echo Please reinstall Python with pip included from python.org
    pause
    exit /b 1
)

echo You can now run run_project.bat
pause
