#!/bin/bash

# =============================================================================
# Cleanup Script for GBM Detection Project - Linux/macOS
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENV_NAME="gbm_env"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GBM Detection Project Cleanup${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "${YELLOW}Cleaning up project environment...${NC}"

# Deactivate conda environment if active
if command -v conda >/dev/null 2>&1; then
    if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
        echo -e "${YELLOW}Deactivating conda environment...${NC}"
        conda deactivate 2>/dev/null || true
    fi
fi

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${YELLOW}Deactivating virtual environment...${NC}"
    deactivate 2>/dev/null || true
fi

# Remove virtual environment
if [ -d "$ENV_NAME" ]; then
    echo -e "${YELLOW}Removing virtual environment: $ENV_NAME${NC}"
    rm -rf "$ENV_NAME"
    echo -e "${GREEN}Virtual environment removed successfully!${NC}"
else
    echo -e "${BLUE}No virtual environment found to remove.${NC}"
fi

# Remove conda environment if it exists
if command -v conda >/dev/null 2>&1; then
    if conda info --envs | grep -q "^$ENV_NAME"; then
        echo -e "${YELLOW}Removing conda environment: $ENV_NAME${NC}"
        conda env remove -n "$ENV_NAME" -y
        echo -e "${GREEN}Conda environment removed successfully!${NC}"
    fi
fi

# Clean up Python cache files
echo -e "${YELLOW}Cleaning Python cache files...${NC}"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Cleanup completed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${BLUE}You can now run ./run_project.sh again.${NC}"
