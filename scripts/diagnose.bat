@echo off
REM =============================================================================
REM Python Environment Diagnostic Script - Windows
REM =============================================================================

echo ============================================
echo   Python Environment Diagnostic
echo ============================================

echo Checking Python installation...
echo.

REM Check if Python is installed
echo [1] Python executable:
python --version 2>nul
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT FOUND
    echo   Please install Python from python.org
    goto :end
)

REM Check Python path
echo.
echo [2] Python path:
where python 2>nul
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT IN PATH
)

REM Check pip
echo.
echo [3] pip availability:
python -m pip --version 2>nul
if %errorlevel% equ 0 (
    echo   Status: OK
    python -m pip --version
) else (
    echo   Status: NOT AVAILABLE
)

REM Check pip path
echo.
echo [4] pip path:
where pip 2>nul
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT IN PATH
)

REM Check venv module
echo.
echo [5] venv module:
python -m venv --help >nul 2>&1
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT AVAILABLE
)

REM Check ensurepip
echo.
echo [6] ensurepip module:
python -m ensurepip --help >nul 2>&1
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT AVAILABLE
)

REM Check internet connectivity
echo.
echo [7] Internet connectivity:
ping -n 1 google.com >nul 2>&1
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NO INTERNET
)

REM Check curl availability
echo.
echo [8] curl availability:
curl --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT AVAILABLE
)

REM Check PowerShell
echo.
echo [9] PowerShell availability:
powershell -Command "Write-Host 'PowerShell OK'" 2>nul
if %errorlevel% equ 0 (
    echo   Status: OK
) else (
    echo   Status: NOT AVAILABLE
)

REM Check current directory permissions
echo.
echo [10] Current directory permissions:
echo %CD%
echo. > test_write.tmp 2>nul
if %errorlevel% equ 0 (
    echo   Status: WRITABLE
    del test_write.tmp >nul 2>&1
) else (
    echo   Status: NOT WRITABLE
)

:end
echo.
echo ============================================
echo   Diagnostic completed
echo ============================================
echo.
echo If you see any "NOT AVAILABLE" or "NOT FOUND" statuses,
echo those are likely causing the script failures.
echo.
pause
