#!/bin/bash

# =============================================================================
# System Health Check Script - Linux/macOS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  System Health Check for GBM Project${NC}"
echo -e "${BLUE}============================================${NC}"

ALL_OK=1

echo -e "${YELLOW}[1] Checking Python installation...${NC}"
if command -v python3 >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Python is installed"
    python3 --version
else
    echo -e "  ${RED}✗${NC} Python NOT installed"
    ALL_OK=0
fi

echo ""
echo -e "${YELLOW}[2] Checking pip...${NC}"
if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} pip is working"
else
    echo -e "  ${RED}✗${NC} pip NOT working"
    ALL_OK=0
fi

echo ""
echo -e "${YELLOW}[3] Checking venv module...${NC}"
if command -v python3 >/dev/null 2>&1 && python3 -m venv --help >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} venv module available"
else
    echo -e "  ${RED}✗${NC} venv module NOT available"
    ALL_OK=0
fi

echo ""
echo -e "${YELLOW}[4] Checking project files...${NC}"
if [ -f "main.py" ]; then
    echo -e "  ${GREEN}✓${NC} main.py found"
else
    echo -e "  ${RED}✗${NC} main.py NOT found"
    ALL_OK=0
fi

if [ -f "requirements.txt" ]; then
    echo -e "  ${GREEN}✓${NC} requirements.txt found"
else
    echo -e "  ${RED}✗${NC} requirements.txt NOT found"
    ALL_OK=0
fi

if [ -f "src/utils.py" ]; then
    echo -e "  ${GREEN}✓${NC} src/utils.py found"
else
    echo -e "  ${RED}✗${NC} src/utils.py NOT found"
    ALL_OK=0
fi

echo ""
echo -e "${YELLOW}[5] Checking directories...${NC}"
if [ -d "models" ]; then
    echo -e "  ${GREEN}✓${NC} models directory exists"
else
    echo -e "  ${YELLOW}!${NC} models directory missing (will be created)"
fi

if [ -d "figures" ]; then
    echo -e "  ${GREEN}✓${NC} figures directory exists"
else
    echo -e "  ${YELLOW}!${NC} figures directory missing (will be created)"
fi

if [ -d "data" ]; then
    echo -e "  ${GREEN}✓${NC} data directory exists"
else
    echo -e "  ${YELLOW}!${NC} data directory missing (will be created)"
fi

echo ""
echo -e "${YELLOW}[6] Checking disk space...${NC}"
if command -v df >/dev/null 2>&1; then
    DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}')
    echo "  Available space: $DISK_SPACE"
else
    echo "  Could not check disk space"
fi

echo ""
echo -e "${YELLOW}[7] Checking network connectivity...${NC}"
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Internet connection available"
else
    echo -e "  ${RED}✗${NC} No internet connection"
    ALL_OK=0
fi

echo ""
echo -e "${BLUE}============================================${NC}"
if [ $ALL_OK -eq 1 ]; then
    echo -e "  ${GREEN}System Status: READY ✓${NC}"
    echo "  You can run the main script safely."
else
    echo -e "  ${RED}System Status: ISSUES DETECTED ✗${NC}"
    echo "  Please fix the issues above before running."
    echo "  You can run './scripts/troubleshoot.sh' to auto-fix some issues."
fi
echo -e "${BLUE}============================================${NC}"
