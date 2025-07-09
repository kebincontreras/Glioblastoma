@echo off
REM =============================================================================
REM System Health Check Script - Windows
REM =============================================================================

echo ============================================
echo   System Health Check for GBM Project
echo ============================================

set ALL_OK=1

echo [1] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Python is installed
    python --version
) else (
    echo   ✗ Python NOT installed
    set ALL_OK=0
)

echo.
echo [2] Checking pip...
python -m pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ pip is working
) else (
    echo   ✗ pip NOT working
    set ALL_OK=0
)

echo.
echo [3] Checking venv module...
python -m venv --help >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ venv module available
) else (
    echo   ✗ venv module NOT available
    set ALL_OK=0
)

echo.
echo [4] Checking project files...
if exist "main.py" (
    echo   ✓ main.py found
) else (
    echo   ✗ main.py NOT found
    set ALL_OK=0
)

if exist "requirements.txt" (
    echo   ✓ requirements.txt found
) else (
    echo   ✗ requirements.txt NOT found
    set ALL_OK=0
)

if exist "src\utils.py" (
    echo   ✓ src\utils.py found
) else (
    echo   ✗ src\utils.py NOT found
    set ALL_OK=0
)

echo.
echo [5] Checking directories...
if exist "models" (
    echo   ✓ models directory exists
) else (
    echo   ! models directory missing (will be created)
)

if exist "figures" (
    echo   ✓ figures directory exists
) else (
    echo   ! figures directory missing (will be created)
)

if exist "data" (
    echo   ✓ data directory exists
) else (
    echo   ! data directory missing (will be created)
)

echo.
echo [6] Checking disk space...
for /f "tokens=3" %%a in ('dir /-c "%cd%" ^| find "bytes free"') do (
    echo   Available space: %%a bytes
)

echo.
echo [7] Checking network connectivity...
ping -n 1 8.8.8.8 >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Internet connection available
) else (
    echo   ✗ No internet connection
    set ALL_OK=0
)

echo.
echo ============================================
if %ALL_OK% equ 1 (
    echo   System Status: READY ✓
    echo   You can run the main script safely.
) else (
    echo   System Status: ISSUES DETECTED ✗
    echo   Please fix the issues above before running.
    echo   You can run 'scripts\troubleshoot.bat' to auto-fix some issues.
)
echo ============================================

pause
