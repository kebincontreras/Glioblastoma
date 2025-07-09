@echo off
REM =============================================================================
REM Cleanup Script for GBM Detection Project - Windows
REM =============================================================================

echo ============================================
echo   GBM Detection Project Cleanup
echo ============================================

set ENV_NAME=gbm_env

echo Cleaning up project environment...

REM Force stop any Python processes that might be using the environment
echo Stopping any Python processes...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1

REM Wait for processes to terminate
timeout /t 2 /nobreak >nul

REM Remove virtual environment with multiple attempts
if exist "%ENV_NAME%" (
    echo Removing virtual environment: %ENV_NAME%
    
    REM First attempt: normal removal
    rmdir /s /q "%ENV_NAME%" 2>nul
    
    if exist "%ENV_NAME%" (
        echo First attempt failed, trying force removal...
        
        REM Second attempt: force delete files then remove directory
        del /f /s /q "%ENV_NAME%" >nul 2>&1
        timeout /t 1 /nobreak >nul
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
        
        if exist "%ENV_NAME%" (
            echo Force removal also failed. Manual cleanup required.
            echo Please delete the '%ENV_NAME%' folder manually and run the script again.
            pause
            exit /b 1
        )
    )
    
    echo Virtual environment removed successfully!
) else (
    echo No virtual environment found to remove.
)

REM Clean up any temporary files
echo Cleaning temporary files...
if exist "*.pyc" del /f /q "*.pyc" >nul 2>&1
if exist "__pycache__" rmdir /s /q "__pycache__" >nul 2>&1
if exist "src\__pycache__" rmdir /s /q "src\__pycache__" >nul 2>&1
if exist "get-pip.py" del /f /q "get-pip.py" >nul 2>&1

REM Clean pip cache
echo Cleaning pip cache...
python -m pip cache purge >nul 2>&1 || echo Pip cache not available or already clean

echo ============================================
echo   Cleanup completed successfully!
echo ============================================
echo You can now run run_project.bat again.
pause
