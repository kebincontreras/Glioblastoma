#!/bin/bash

# =============================================================================
# GBM Detection Project - Linux/macOS Main Script with Auto-Troubleshooting
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="GBM_Detection"
ENV_NAME="gbm_env"
PYTHON_VERSION="3.8"
SCRIPTS_DIR="scripts"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect Python command (python3 or python)
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python" # fallback
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GBM Detection Project Setup & Run${NC}"
echo -e "${BLUE}============================================${NC}"

# =============================================================================
# FUNCTIONS
# =============================================================================

# Function to run troubleshooting
troubleshoot() {
    echo -e "${YELLOW}Running automatic troubleshooting...${NC}"
    
    if [ -f "$SCRIPTS_DIR/troubleshoot.sh" ]; then
        bash "$SCRIPTS_DIR/troubleshoot.sh"
    else
        # Fallback to basic troubleshooting
        # Deactivate any active environments
        if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
            conda deactivate 2>/dev/null || true
        fi
        if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
            deactivate 2>/dev/null || true
        fi
        
        # Remove corrupted environment
        if [ -d "$ENV_NAME" ]; then
            rm -rf "$ENV_NAME"
        fi
        
        # Clean Python cache files
        find . -name "*.pyc" -delete 2>/dev/null || true
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    fi
    
    sleep 1
}

# Function to fix pip
fix_pip() {
    echo -e "${YELLOW}Attempting to fix pip installation...${NC}"
    
    if [ -f "$SCRIPTS_DIR/fix_pip.sh" ]; then
        bash "$SCRIPTS_DIR/fix_pip.sh"
    else
        # Fallback to basic pip fix
        # Try ensurepip first
        $PYTHON_CMD -m ensurepip --upgrade --user >/dev/null 2>&1 || true
        
        # Check if pip is now available
        if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
            # Try downloading get-pip.py
            curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1 || true
            if [ -f "get-pip.py" ]; then
                $PYTHON_CMD get-pip.py --user >/dev/null 2>&1 || true
                rm -f get-pip.py
            fi
        fi
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Main execution starts here
main() {
    # Optional: Run health check first (uncomment the next 2 lines to enable)
    # echo -e "${BLUE}Running system health check...${NC}"
    # [ -f "$SCRIPTS_DIR/health_check.sh" ] && bash "$SCRIPTS_DIR/health_check.sh"

    # Check if Python is installed
    if ! command_exists "$PYTHON_CMD"; then
        echo -e "${RED}Error: Python is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi

    # Display Python version
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${BLUE}Found Python version: ${PYTHON_VER}${NC}"

    # Check and fix pip if needed
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: pip not available. Auto-fixing...${NC}"
        fix_pip
        if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
            echo -e "${RED}Error: Could not fix pip. Please install pip manually.${NC}"
            exit 1
        fi
    fi

    # Check if conda is available, otherwise use venv
    # if command_exists conda; then
    #     echo -e "${GREEN}Using Conda for environment management${NC}"
    #     USE_CONDA=true
    # else
    #     echo -e "${GREEN}Using venv for environment management${NC}"
    #     USE_CONDA=false
        
    #     # Check venv module availability
    #     if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    #         echo -e "${RED}Error: venv module not available. Please install python3-venv.${NC}"
    #         exit 1
    #     fi
    # fi

    # Forzar uso de venv siempre, nunca conda
    echo -e "${GREEN}Using venv for environment management${NC}"
    USE_CONDA=false
    # Check venv module availability
    if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
        echo -e "${RED}Error: venv module not available. Please install python3-venv.${NC}"
        exit 1
    fi

    # Remove existing environment if it exists (only if corrupted)
    if [ -d "$ENV_NAME" ]; then
        echo -e "${YELLOW}Testing existing environment...${NC}"
        if [ "$USE_CONDA" = true ]; then
            if conda activate "$ENV_NAME" >/dev/null 2>&1; then
                # Check if required packages are installed
                echo -e "${YELLOW}Checking if required packages are installed...${NC}"
                if $PYTHON_CMD -c "import torch, torchvision, numpy, sklearn, matplotlib, pandas, pydicom, cv2, tqdm" >/dev/null 2>&1; then
                    echo -e "${GREEN}Existing conda environment is healthy. Using it...${NC}"
                    conda deactivate >/dev/null 2>&1
                    CREATE_NEW_ENV=false
                    REINSTALL_PACKAGES=false
                else
                    echo -e "${YELLOW}Required packages missing. Will reinstall dependencies...${NC}"
                    conda deactivate >/dev/null 2>&1
                    CREATE_NEW_ENV=false
                    REINSTALL_PACKAGES=true
                fi
            else
                echo -e "${YELLOW}Conda environment corrupted. Removing...${NC}"
                conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1
                CREATE_NEW_ENV=true
                REINSTALL_PACKAGES=true
            fi
        else
            if source "$ENV_NAME/bin/activate" >/dev/null 2>&1; then
                # Check if required packages are installed
                echo -e "${YELLOW}Checking if required packages are installed...${NC}"
                if $PYTHON_CMD -c "import torch, torchvision, numpy, sklearn, matplotlib, pandas, pydicom, cv2, tqdm" >/dev/null 2>&1; then
                    echo -e "${GREEN}Existing virtual environment is healthy. Using it...${NC}"
                    deactivate >/dev/null 2>&1
                    CREATE_NEW_ENV=false
                    REINSTALL_PACKAGES=false
                else
                    echo -e "${YELLOW}Required packages missing. Will reinstall dependencies...${NC}"
                    deactivate >/dev/null 2>&1
                    CREATE_NEW_ENV=false
                    REINSTALL_PACKAGES=true
                fi
            else
                echo -e "${YELLOW}Virtual environment corrupted. Removing...${NC}"
                rm -rf "$ENV_NAME"
                CREATE_NEW_ENV=true
                REINSTALL_PACKAGES=true
            fi
        fi
    else
        echo -e "${YELLOW}No existing environment found.${NC}"
        CREATE_NEW_ENV=true
        REINSTALL_PACKAGES=true
    fi

    # Create and activate environment only if needed
    if [ "$CREATE_NEW_ENV" = true ]; then
        if [ "$USE_CONDA" = true ]; then
            echo -e "${YELLOW}Creating conda environment: ${ENV_NAME}${NC}"
            
            # Create new environment
            conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}Conda creation failed. Trying troubleshooting...${NC}"
                troubleshoot
                conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
                if [ $? -ne 0 ]; then
                    echo -e "${RED}Error: Failed to create conda environment after troubleshooting.${NC}"
                    exit 1
                fi
            fi
        else
            echo -e "${YELLOW}Creating virtual environment: ${ENV_NAME}${NC}"
            
            # Create new environment
            $PYTHON_CMD -m venv ${ENV_NAME}
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}venv creation failed. Trying troubleshooting...${NC}"
                troubleshoot
                $PYTHON_CMD -m venv ${ENV_NAME}
                if [ $? -ne 0 ]; then
                    echo -e "${RED}Error: Failed to create virtual environment after troubleshooting.${NC}"
                    exit 1
                fi
            fi
            
            # Verify pip is available in the new environment
            if [ -f "${ENV_NAME}/bin/activate" ]; then
                source ${ENV_NAME}/bin/activate
                if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
                    echo -e "${YELLOW}pip not found in new environment, attempting to install...${NC}"
                    curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1 || true
                    if [ -f "get-pip.py" ]; then
                        $PYTHON_CMD get-pip.py >/dev/null 2>&1 || true
                        rm -f get-pip.py
                    fi
                fi
                # Verify pip is now working
                if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
                    echo -e "${RED}Error: Could not install pip in the new environment${NC}"
                    exit 1
                fi
                deactivate
            fi
        fi
    else
        echo -e "${GREEN}Using existing environment: ${ENV_NAME}${NC}"
    fi

    # Activate the environment (new or existing)
    if [ "$USE_CONDA" = true ]; then
        echo -e "${YELLOW}Activating conda environment: ${ENV_NAME}${NC}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate ${ENV_NAME}
        # Redefine PYTHON_CMD to conda env python
        PYTHON_CMD="$(which python)"
        # Prevent pip from using user site-packages
        export PIP_USER=no
    else
        echo -e "${YELLOW}Activating virtual environment: ${ENV_NAME}${NC}"
        source ${ENV_NAME}/bin/activate
        # Redefine PYTHON_CMD to venv python
        PYTHON_CMD="$(pwd)/${ENV_NAME}/bin/python"
        # Prevent pip from using user site-packages
        export PIP_USER=no
    fi

    # Upgrade pip (always use the environment's python)
    echo -e "${YELLOW}Upgrading pip...${NC}"
    $PYTHON_CMD -m pip install --upgrade pip >/dev/null 2>&1

    # Check for CUDA/GPU availability before installing packages
    echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}NVIDIA GPU detected. Will install PyTorch with CUDA support.${NC}"
        INSTALL_CUDA=true
    else
        echo -e "${YELLOW}No NVIDIA GPU detected. Will install CPU-only PyTorch.${NC}"
        INSTALL_CUDA=false
    fi

    # Install requirements
    echo -e "${YELLOW}Installing project requirements...${NC}"
    if [ -f "requirements.txt" ]; then
        if [ "$REINSTALL_PACKAGES" = true ]; then
            echo -e "${YELLOW}Installing/updating requirements...${NC}"
            
            # Install PyTorch first with appropriate CUDA support
            if [ "$INSTALL_CUDA" = true ]; then
                echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
                $PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            else
                echo -e "${YELLOW}Installing CPU-only PyTorch...${NC}"
                $PYTHON_CMD -m pip install torch torchvision torchaudio
            fi

            # Verify PyTorch installation
            if ! $PYTHON_CMD -c "import torch; print('PyTorch version:', torch.__version__)" >/dev/null 2>&1; then
                echo -e "${RED}Error: PyTorch installation failed${NC}"
                exit 1
            fi

            # Install other requirements with dependency management
            echo -e "${YELLOW}Installing other packages from requirements.txt...${NC}"
            # First install without dependencies to avoid conflicts
            $PYTHON_CMD -m pip install -r requirements.txt --no-deps >/dev/null 2>&1 || true
            # Then install with dependencies to ensure everything is properly linked
            $PYTHON_CMD -m pip install -r requirements.txt
            if [ $? -ne 0 ]; then
                echo -e "${RED}Error: Failed to install requirements${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}Requirements already satisfied. Skipping installation...${NC}"
        fi
    else
        echo -e "${RED}Error: requirements.txt not found!${NC}"
        exit 1
    fi

    # Check CUDA availability
    echo -e "${YELLOW}Checking CUDA availability...${NC}"
    $PYTHON_CMD -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

    # Create necessary directories
    echo -e "${YELLOW}Creating necessary directories...${NC}"
    mkdir -p models figures data

    # Download dataset
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Downloading Dataset${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}Downloading RSNA-MICCAI Brain Tumor Dataset...${NC}"

    if [ -f "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" ]; then
        echo -e "${GREEN}Dataset archive already exists. Skipping download...${NC}"
    else
        echo -e "${YELLOW}Starting download from HuggingFace...${NC}"
        
        if command_exists curl; then
            curl -L -o "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" \
                "https://huggingface.co/datasets/kebincontreras/Glioblastoma_t1w/resolve/main/rsna-miccai-brain-tumor-radiogenomic-classification.rar"
        elif command_exists wget; then
            wget -O "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" \
                "https://huggingface.co/datasets/kebincontreras/Glioblastoma_t1w/resolve/main/rsna-miccai-brain-tumor-radiogenomic-classification.rar"
        else
            echo -e "${RED}Error: Neither curl nor wget available for download${NC}"
            exit 1
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Dataset download completed successfully!${NC}"
        else
            echo -e "${RED}Error: Failed to download dataset${NC}"
            exit 1
        fi
    fi

    # Extract dataset
    if [ -d "data/rsna-miccai-brain-tumor-radiogenomic-classification" ]; then
        echo -e "${GREEN}Dataset already extracted. Skipping extraction...${NC}"
    else
        echo -e "${YELLOW}Extracting dataset...${NC}"
        
        EXTRACTED=false
        
        # Method 1: Try Python packages for extraction
        echo -e "${BLUE}Trying Python-based extraction packages...${NC}"
        
        # Try rarfile package
        if ! $PYTHON_CMD -c "import rarfile" >/dev/null 2>&1; then
            echo -e "${YELLOW}Installing rarfile package...${NC}"
            $PYTHON_CMD -m pip install rarfile --quiet
        fi
        
        echo -e "${YELLOW}Extracting dataset with rarfile (this may take a while)...${NC}"
        # Show a simple progress bar in English while extracting
        $PYTHON_CMD -c "
import rarfile, sys, time
rf = rarfile.RarFile('data/rsna-miccai-brain-tumor-radiogenomic-classification.rar')
total = len(rf.infolist())
print(f'Extracting {total} files...')
count = 0
start = time.time()
last_update = 0
for info in rf.infolist():
    rf.extract(info, 'data')
    count += 1
    now = time.time()
    if now - last_update > 0.5 or count == total:
        done = int(50 * count / total)
        sys.stdout.write('\r[' + '#' * done + '-' * (50-done) + f'] {count}/{total}')
        sys.stdout.flush()
        last_update = now
end = time.time()
rf.close()
print(f'\nExtraction completed in {end-start:.1f} seconds.')
" 2>/dev/null
        
        # Check if extraction was successful
        if [ -d "data/rsna-miccai-brain-tumor-radiogenomic-classification" ]; then
            echo -e "${GREEN}Dataset extracted successfully with Python rarfile!${NC}"
            EXTRACTED=true
        else
            echo -e "${YELLOW}Python rarfile extraction failed, trying patoolib...${NC}"
            
            # Try patoolib package
            if ! $PYTHON_CMD -c "import patoolib" >/dev/null 2>&1; then
                echo -e "${YELLOW}Installing patoolib package...${NC}"
                $PYTHON_CMD -m pip install patoolib --quiet
            fi
            
            if $PYTHON_CMD -c "import patoolib; patoolib.extract_archive('data/rsna-miccai-brain-tumor-radiogenomic-classification.rar', outdir='data'); print('Python patoolib extraction successful')" 2>/dev/null; then
                echo -e "${GREEN}Dataset extracted successfully with Python patoolib!${NC}"
                EXTRACTED=true
            else
                echo -e "${YELLOW}Python patoolib extraction failed, trying system tools...${NC}"
            fi
        fi
        
        # Method 2: Try unrar (most common)
        if [ "$EXTRACTED" = false ] && command_exists unrar; then
            echo -e "${BLUE}Trying unrar...${NC}"
            if unrar x "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" "data/" >/dev/null 2>&1; then
                echo -e "${GREEN}Dataset extracted successfully with unrar!${NC}"
                EXTRACTED=true
            else
                echo -e "${YELLOW}unrar extraction failed, trying next method...${NC}"
            fi
        elif [ "$EXTRACTED" = false ]; then
            echo -e "${YELLOW}unrar not found, trying next method...${NC}"
        fi
        
        # Method 3: Try 7z
        if [ "$EXTRACTED" = false ] && command_exists 7z; then
            echo -e "${BLUE}Trying 7z...${NC}"
            if 7z x "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" -o"data/" -y >/dev/null 2>&1; then
                echo -e "${GREEN}Dataset extracted successfully with 7z!${NC}"
                EXTRACTED=true
            else
                echo -e "${YELLOW}7z extraction failed, trying next method...${NC}"
            fi
        elif [ "$EXTRACTED" = false ]; then
            echo -e "${YELLOW}7z not found, trying next method...${NC}"
        fi
        
        # Method 4: Try 7za (alternative 7-zip command)
        if [ "$EXTRACTED" = false ] && command_exists 7za; then
            echo -e "${BLUE}Trying 7za...${NC}"
            if 7za x "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" -o"data/" -y >/dev/null 2>&1; then
                echo -e "${GREEN}Dataset extracted successfully with 7za!${NC}"
                EXTRACTED=true
            else
                echo -e "${YELLOW}7za extraction failed, trying next method...${NC}"
            fi
        elif [ "$EXTRACTED" = false ]; then
            echo -e "${YELLOW}7za not found, trying next method...${NC}"
        fi
        
        # Method 5: Try rar (if available)
        if [ "$EXTRACTED" = false ] && command_exists rar; then
            echo -e "${BLUE}Trying rar...${NC}"
            if rar x "data/rsna-miccai-brain-tumor-radiogenomic-classification.rar" "data/" >/dev/null 2>&1; then
                echo -e "${GREEN}Dataset extracted successfully with rar!${NC}"
                EXTRACTED=true
            else
                echo -e "${YELLOW}rar extraction failed...${NC}"
            fi
        elif [ "$EXTRACTED" = false ]; then
            echo -e "${YELLOW}rar not found...${NC}"
        fi
        
        # Check if extraction was successful
        if [ -d "data/rsna-miccai-brain-tumor-radiogenomic-classification" ]; then
            echo -e "${GREEN}Dataset extracted successfully!${NC}"
            # Check if key files exist
            if [ -f "data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv" ]; then
                echo -e "${GREEN}Key dataset files found!${NC}"
            else
                echo -e "${YELLOW}Warning: train_labels.csv not found. Extraction may be incomplete.${NC}"
            fi
        else
            echo -e "${YELLOW}============================================${NC}"
            echo -e "${YELLOW}  Manual extraction required${NC}"
            echo -e "${YELLOW}============================================${NC}"
            echo -e "${YELLOW}The dataset could not be extracted automatically.${NC}"
            echo -e "${YELLOW}All extraction methods failed (Python packages and system tools).${NC}"
            echo -e "${YELLOW}Please manually extract:${NC}"
            echo -e "${YELLOW}  'data/rsna-miccai-brain-tumor-radiogenomic-classification.rar'${NC}"
            echo -e "${YELLOW}To:${NC}"
            echo -e "${YELLOW}  'data/' folder${NC}"
            echo ""
            echo -e "${YELLOW}Install extraction tools:${NC}"
            echo -e "${YELLOW}  Ubuntu/Debian: sudo apt-get install unrar p7zip-full${NC}"
            echo -e "${YELLOW}  CentOS/RHEL/Fedora: sudo dnf install unrar p7zip${NC}"
            echo -e "${YELLOW}  macOS: brew install unrar p7zip${NC}"
            echo ""
            echo -e "${YELLOW}Alternative commands:${NC}"
            echo -e "${YELLOW}  unrar x data/rsna-miccai-brain-tumor-radiogenomic-classification.rar data/${NC}"
            echo -e "${YELLOW}  7z x data/rsna-miccai-brain-tumor-radiogenomic-classification.rar -odata/${NC}"
            echo ""
            echo -e "${YELLOW}After extraction, you should have:${NC}"
            echo -e "${YELLOW}  data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv${NC}"
            echo -e "${YELLOW}  data/rsna-miccai-brain-tumor-radiogenomic-classification/train/${NC}"
            echo -e "${YELLOW}  data/rsna-miccai-brain-tumor-radiogenomic-classification/test/${NC}"
            echo ""
            echo -e "${YELLOW}Press Enter to continue anyway...${NC}"
            read -r
        fi
    fi

    # Run the main script
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Starting GBM Detection Training${NC}"
    echo -e "${BLUE}============================================${NC}"

    $PYTHON_CMD main.py

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}  Training completed successfully!${NC}"
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}Results saved in:${NC}"
        echo -e "${GREEN}  - Models: ./models/${NC}"
        echo -e "${GREEN}  - Figures: ./figures/${NC}"
    else
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}  Training failed!${NC}"
        echo -e "${RED}============================================${NC}"
        exit 1
    fi

    # Deactivate environment
    echo -e "${YELLOW}Deactivating environment...${NC}"
    if [ "$USE_CONDA" = true ]; then
        conda deactivate 2>/dev/null || true
    else
        deactivate 2>/dev/null || true
    fi

    echo -e "${GREEN}Script completed successfully!${NC}"
}

# =============================================================================
# EXECUTE MAIN FUNCTION
# =============================================================================
main

# End of script
