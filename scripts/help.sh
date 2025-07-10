#!/bin/bash

# =============================================================================
# Help Script - Shows all available scripts and their functions
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GBM Detection Project - Available Scripts${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${GREEN}MAIN SCRIPTS (in project root):${NC}"
echo "  ./run_project.sh      - Complete setup and execution (Linux/macOS)"
echo "  ./run_project.bat     - Complete setup and execution (Windows)"
echo ""

echo -e "${YELLOW}TROUBLESHOOTING SCRIPTS (in scripts folder):${NC}"
echo "  ./scripts/troubleshoot.sh    - Complete environment troubleshooting"
echo "  ./scripts/troubleshoot.bat   - Complete environment troubleshooting (Windows)"
echo "  ./scripts/fix_pip.sh         - Fix pip installation issues"
echo "  ./scripts/fix_pip.bat        - Fix pip installation issues (Windows)"
echo "  ./scripts/health_check.sh    - System health check"
echo "  ./scripts/health_check.bat   - System health check (Windows)"
echo "  ./scripts/diagnose.sh        - Detailed system diagnostics"
echo "  ./scripts/diagnose.bat       - Detailed system diagnostics (Windows)"
echo ""

echo -e "${BLUE}UTILITY SCRIPTS (in scripts folder):${NC}"
echo "  ./scripts/cleanup.sh         - Clean environment and temporary files"
echo "  ./scripts/cleanup.bat        - Clean environment and temporary files (Windows)"
echo "  ./scripts/download_dataset.sh - Download dataset separately"
echo "  ./scripts/download_dataset.bat - Download dataset separately (Windows)"
echo "  ./scripts/run_simple.bat     - Step-by-step execution (Windows)"
echo ""

echo -e "${GREEN}USAGE EXAMPLES:${NC}"
echo "  Normal execution:"
echo "    ./run_project.sh"
echo ""
echo "  If you have problems:"
echo "    ./scripts/troubleshoot.sh"
echo "    ./run_project.sh"
echo ""
echo "  Check system health:"
echo "    ./scripts/health_check.sh"
echo ""
echo "  Clean everything:"
echo "    ./scripts/cleanup.sh"
echo ""
echo "  Get detailed diagnostics:"
echo "    ./scripts/diagnose.sh"
echo ""

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  For more information, see: SCRIPT_USAGE.md${NC}"
echo -e "${BLUE}============================================${NC}"
