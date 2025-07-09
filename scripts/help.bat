@echo off
REM =============================================================================
REM Help Script - Shows all available scripts and their functions
REM =============================================================================

echo ============================================
echo   GBM Detection Project - Available Scripts
echo ============================================
echo.

echo MAIN SCRIPTS (in project root):
echo   run_project.bat     - Complete setup and execution (Windows)
echo   run_project.sh      - Complete setup and execution (Linux/macOS)
echo.

echo TROUBLESHOOTING SCRIPTS (in scripts folder):
echo   troubleshoot.bat    - Complete environment troubleshooting
echo   troubleshoot.sh     - Complete environment troubleshooting (Linux/macOS)
echo   fix_pip.bat         - Fix pip installation issues
echo   fix_pip.sh          - Fix pip installation issues (Linux/macOS)
echo   health_check.bat    - System health check
echo   health_check.sh     - System health check (Linux/macOS)
echo   diagnose.bat        - Detailed system diagnostics
echo.

echo UTILITY SCRIPTS (in scripts folder):
echo   cleanup.bat         - Clean environment and temporary files
echo   cleanup.sh          - Clean environment and temporary files (Linux/macOS)
echo   download_dataset.bat - Download dataset separately
echo   download_dataset.sh  - Download dataset separately (Linux/macOS)
echo   run_simple.bat     - Step-by-step execution
echo.

echo USAGE EXAMPLES:
echo   Normal execution:
echo     run_project.bat
echo.
echo   If you have problems:
echo     scripts\troubleshoot.bat
echo     run_project.bat
echo.
echo   Check system health:
echo     scripts\health_check.bat
echo.
echo   Clean everything:
echo     scripts\cleanup.bat
echo.
echo   Get detailed diagnostics:
echo     scripts\diagnose.bat
echo.

echo ============================================
echo   For more information, see: SCRIPT_USAGE.md
echo ============================================

pause
