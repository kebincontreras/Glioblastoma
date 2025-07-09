#!/bin/bash

# =============================================================================
# Fix pip installation - Linux/macOS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Fixing pip installation${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "${YELLOW}Checking current Python installation...${NC}"
if command -v python3 >/dev/null 2>&1; then
    python3 --version
else
    echo -e "${RED}Error: Python3 is not installed or not in PATH.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Attempting to fix pip installation...${NC}"

# Method 1: Try ensurepip
echo -e "${BLUE}Method 1: Using ensurepip...${NC}"
if python3 -m ensurepip --upgrade --user >/dev/null 2>&1; then
    echo -e "${GREEN}pip installed successfully via ensurepip!${NC}"
    python3 -m pip --version
    exit 0
fi

# Method 2: Download and use get-pip.py
echo -e "${BLUE}Method 2: Downloading get-pip.py...${NC}"
rm -f get-pip.py

if command -v curl >/dev/null 2>&1; then
    curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py 2>/dev/null
elif command -v wget >/dev/null 2>&1; then
    wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py 2>/dev/null
else
    echo -e "${RED}Error: Neither curl nor wget is available for downloading get-pip.py${NC}"
    exit 1
fi

if [ -f "get-pip.py" ]; then
    echo -e "${BLUE}Installing pip using get-pip.py...${NC}"
    if python3 get-pip.py --user; then
        echo -e "${GREEN}pip installed successfully!${NC}"
        rm -f get-pip.py
    else
        echo -e "${RED}Failed to install pip using get-pip.py${NC}"
        rm -f get-pip.py
        exit 1
    fi
else
    echo -e "${RED}Failed to download get-pip.py${NC}"
    exit 1
fi

# Method 3: Try package manager (as last resort)
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo -e "${BLUE}Method 3: Trying package manager...${NC}"
    
    if command -v apt-get >/dev/null 2>&1; then
        echo "Trying apt-get (requires sudo)..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    elif command -v yum >/dev/null 2>&1; then
        echo "Trying yum (requires sudo)..."
        sudo yum install -y python3-pip
    elif command -v dnf >/dev/null 2>&1; then
        echo "Trying dnf (requires sudo)..."
        sudo dnf install -y python3-pip
    elif command -v brew >/dev/null 2>&1; then
        echo "Trying homebrew..."
        brew install python3
    else
        echo -e "${YELLOW}No supported package manager found.${NC}"
    fi
fi

# Verify installation
echo ""
echo -e "${BLUE}Verifying pip installation...${NC}"
if python3 -m pip --version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ pip is now working correctly!${NC}"
    python3 -m pip --version
    echo ""
    echo -e "${GREEN}You can now run the main script.${NC}"
else
    echo -e "${RED}✗ pip installation failed.${NC}"
    echo ""
    echo -e "${YELLOW}Manual installation required:${NC}"
    echo "1. Make sure Python 3.8+ is installed"
    echo "2. Install pip using your system's package manager"
    echo "3. Or visit: https://pip.pypa.io/en/stable/installation/"
    exit 1
fi
