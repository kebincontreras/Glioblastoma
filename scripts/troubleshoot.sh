#!/bin/bash

# =============================================================================
# Complete Troubleshooting Script - Linux/macOS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENV_NAME="gbm_env"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Complete Python Environment Troubleshooting${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "${YELLOW}[Step 1] Deactivating any active environments...${NC}"
# Deactivate conda environments
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    conda deactivate 2>/dev/null || true
    echo "  Conda environment deactivated"
else
    echo "  No conda environment active"
fi

# Deactivate virtualenv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate 2>/dev/null || true
    echo "  Virtual environment deactivated"
else
    echo "  No virtual environment active"
fi

echo -e "${YELLOW}[Step 2] Removing corrupted environment...${NC}"
if [ -d "$ENV_NAME" ]; then
    echo "  Removing $ENV_NAME..."
    rm -rf "$ENV_NAME"
    if [ -d "$ENV_NAME" ]; then
        echo "  Warning: Could not remove $ENV_NAME. Please remove manually."
    else
        echo "  Environment removed successfully"
    fi
else
    echo "  No environment to remove"
fi

echo -e "${YELLOW}[Step 3] Cleaning Python cache files...${NC}"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Cache files cleaned"

echo -e "${YELLOW}[Step 4] Checking and fixing pip...${NC}"
if command -v python3 >/dev/null 2>&1; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo "  pip not working, attempting to fix..."
        
        # Try ensurepip first
        python3 -m ensurepip --upgrade --user >/dev/null 2>&1 || true
        
        # Check if pip is now available
        if ! python3 -m pip --version >/dev/null 2>&1; then
            # Try downloading get-pip.py
            if command -v curl >/dev/null 2>&1; then
                curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1 || true
            elif command -v wget >/dev/null 2>&1; then
                wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1 || true
            fi
            
            if [ -f "get-pip.py" ]; then
                python3 get-pip.py --user >/dev/null 2>&1 || true
                rm -f get-pip.py
            fi
        fi
        
        if python3 -m pip --version >/dev/null 2>&1; then
            echo "  pip fixed successfully"
        else
            echo "  Warning: Could not fix pip"
        fi
    else
        echo "  pip is working correctly"
    fi
else
    echo "  Error: Python3 not found"
fi

echo -e "${YELLOW}[Step 5] Checking venv module...${NC}"
if command -v python3 >/dev/null 2>&1 && python3 -m venv --help >/dev/null 2>&1; then
    echo "  venv module is available"
else
    echo "  Warning: venv module not available"
fi

echo -e "${YELLOW}[Step 6] Cleaning temporary files...${NC}"
rm -f *.tmp 2>/dev/null || true
rm -rf temp 2>/dev/null || true
echo "  Temporary files cleaned"

echo -e "${YELLOW}[Step 7] Checking disk space...${NC}"
if command -v df >/dev/null 2>&1; then
    DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}')
    echo "  Available disk space: $DISK_SPACE"
else
    echo "  Could not check disk space"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Troubleshooting completed!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "You can now try running the main script again."
echo "If problems persist, check the requirements and documentation."
