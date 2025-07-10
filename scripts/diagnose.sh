#!/bin/bash
# =============================================================================
# Python Environment Diagnostic Script - Linux
# =============================================================================

echo "============================================"
echo "   Python Environment Diagnostic"
echo "============================================"

echo "Checking Python installation..."
echo

# Check if Python is installed
echo "[1] Python executable:"
if command -v python3 &> /dev/null; then
    python3 --version
    echo "   Status: OK"
elif command -v python &> /dev/null; then
    python --version
    echo "   Status: OK"
else
    echo "   Status: NOT FOUND"
    echo "   Please install Python from python.org or your package manager"
    exit 1
fi

# Check Python path
echo
echo "[2] Python path:"
if command -v python3 &> /dev/null; then
    which python3
    echo "   Status: OK"
elif command -v python &> /dev/null; then
    which python
    echo "   Status: OK"
else
    echo "   Status: NOT IN PATH"
fi

# Check pip
echo
echo "[3] pip availability:"
if command -v python3 &> /dev/null; then
    python3 -m pip --version 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   Status: OK"
        python3 -m pip --version
    else
        echo "   Status: NOT AVAILABLE"
    fi
elif command -v python &> /dev/null; then
    python -m pip --version 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   Status: OK"
        python -m pip --version
    else
        echo "   Status: NOT AVAILABLE"
    fi
else
    echo "   Status: NOT AVAILABLE"
fi

# Check pip path
echo
echo "[4] pip path:"
if command -v pip3 &> /dev/null; then
    which pip3
    echo "   Status: OK"
elif command -v pip &> /dev/null; then
    which pip
    echo "   Status: OK"
else
    echo "   Status: NOT IN PATH"
fi

# Check conda
echo
echo "[5] conda availability:"
if command -v conda &> /dev/null; then
    conda --version 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   Status: OK"
        conda --version
    else
        echo "   Status: NOT AVAILABLE"
    fi
else
    echo "   Status: NOT AVAILABLE"
fi

# Check virtual environment
echo
echo "[6] Virtual environment:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "   Active environment: $VIRTUAL_ENV"
    echo "   Status: ACTIVE"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "   Active environment: $CONDA_DEFAULT_ENV"
    echo "   Status: ACTIVE (conda)"
else
    echo "   Status: NOT ACTIVE"
fi

# Check PyTorch availability
echo
echo "[7] PyTorch availability:"
if command -v python3 &> /dev/null; then
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null
else
    python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null
fi

if [ $? -eq 0 ]; then
    echo "   Status: OK"
else
    echo "   Status: NOT AVAILABLE"
fi

# Check CUDA availability
echo
echo "[8] CUDA availability:"
if command -v python3 &> /dev/null; then
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null
else
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null
fi

if [ $? -eq 0 ]; then
    echo "   Status: OK"
else
    echo "   Status: NOT AVAILABLE"
fi

# Check required packages
echo
echo "[9] Required packages:"
PACKAGES=("numpy" "pandas" "matplotlib" "scikit-learn" "opencv-python" "pydicom" "tqdm")
for package in "${PACKAGES[@]}"; do
    if command -v python3 &> /dev/null; then
        python3 -c "import $package" 2>/dev/null
    else
        python -c "import $package" 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        echo "   $package: OK"
    else
        echo "   $package: NOT AVAILABLE"
    fi
done

# Check current directory permissions
echo
echo "[10] Current directory permissions:"
echo "$(pwd)"
touch test_write.tmp 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   Status: WRITABLE"
    rm -f test_write.tmp 2>/dev/null
else
    echo "   Status: NOT WRITABLE"
fi

echo
echo "============================================"
echo "   Diagnostic completed"
echo "============================================"
echo
echo "If you see any 'NOT AVAILABLE' or 'NOT FOUND' statuses,"
echo "those are likely causing the script failures."
echo
