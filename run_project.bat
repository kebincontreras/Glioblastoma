@echo off
REM =============================================================================
REM GBM Detection Project - Windows Main Script with Auto-Troubleshooting
REM =============================================================================

setlocal enabledelayedexpansion

REM Project configuration
set PROJECT_NAME=GBM_Detection
set ENV_NAME=gbm_env
set PYTHON_VERSION=3.8
set SCRIPTS_DIR=scripts

echo ============================================
echo   GBM Detection Project Setup ^& Run
echo ============================================

REM Main execution starts here
goto :main

REM ============================================================================
REM FUNCTIONS
REM ============================================================================

REM Function to run complete troubleshooting
:troubleshoot
echo Running automatic troubleshooting...
if exist "%SCRIPTS_DIR%\troubleshoot.bat" (
    call "%SCRIPTS_DIR%\troubleshoot.bat" auto
) else (
    call "%SCRIPTS_DIR%\cleanup.bat" >nul 2>&1
)
ping 127.0.0.1 -n 2 >nul
goto :eof

REM Function to fix pip
:fix_pip
echo Attempting to fix pip installation...
if exist "%SCRIPTS_DIR%\fix_pip.bat" (
    call "%SCRIPTS_DIR%\fix_pip.bat"
) else (
    python -m ensurepip --upgrade --default-pip --user >nul 2>&1
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Downloading and installing pip...
        if exist "get-pip.py" del "get-pip.py"
        curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
        if exist "get-pip.py" (
            python get-pip.py --user >nul 2>&1
            del "get-pip.py" >nul 2>&1
        )
    )
)
goto :eof

REM ============================================================================
REM MAIN EXECUTION
REM ============================================================================

:main

REM Optional: Run health check first (uncomment the next 2 lines to enable)
REM echo Running system health check...
REM if exist "%SCRIPTS_DIR%\health_check.bat" call "%SCRIPTS_DIR%\health_check.bat"

REM Check if Python is installed
echo ============================================
echo   Python Environment Check
echo ============================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Error: Python is not installed or not in PATH.
    echo.
    echo SOLUTIONS:
    echo 1. Install Python from: https://python.org/downloads/
    echo 2. Make sure to check "Add Python to PATH" during installation
    echo 3. Restart Command Prompt after installation
    echo.
    pause
    exit /b 1
)

echo Found Python version:
python --version

REM Get detailed Python info for troubleshooting
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Python executable: %PYTHON_PATH%

REM Check if this is a problematic Python installation (like some conda installations)
echo %PYTHON_PATH% | findstr /i "conda\|anaconda" >nul
if %errorlevel% equ 0 (
    echo.
    echo  WARNING: Detected Conda/Anaconda Python
    echo This may cause virtual environment issues.
    echo.
    echo RECOMMENDATION:
    echo 1. Use 'conda create' instead of venv, OR
    echo 2. Install standalone Python from python.org
    echo.
    echo Continuing anyway...
    timeout /t 3 /nobreak >nul
)

REM Check Python architecture
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.architecture()[0])"') do set PYTHON_ARCH=%%i
echo Python architecture: %PYTHON_ARCH%

REM Check and fix pip if needed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: pip not available. Auto-fixing...
    call :fix_pip
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Could not fix pip. Please reinstall Python with pip included.
        pause
        exit /b 1
    )
)

REM Check venv module
python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: venv module not available. Please reinstall Python.
    pause
    exit /b 1
)

REM ============================================
REM   Virtual Environment Management
REM ============================================

set NEED_NEW_ENV=0
set NEED_INSTALL_PACKAGES=0

echo Checking virtual environment: %ENV_NAME%

REM Step 1: Check if environment exists and is functional
if exist "%ENV_NAME%\Scripts\python.exe" (
    echo Environment directory found
    
    REM Test if environment can be activated
    call "%ENV_NAME%\Scripts\activate.bat" >nul 2>&1
    if %errorlevel% neq 0 (
        echo  Environment cannot be activated - will recreate
        set NEED_NEW_ENV=1
    ) else (
        echo Environment can be activated
        
        REM Step 2: Check if all required packages are installed (simple check)
        echo Checking required packages...
        "%ENV_NAME%\Scripts\python.exe" -c "import torch, torchvision, numpy, sklearn, matplotlib, pandas, pydicom, cv2, tqdm; print('All packages available')" >nul 2>&1
        if %errorlevel% neq 0 (
            echo  Some required packages are missing
            set NEED_INSTALL_PACKAGES=1
        ) else (
            echo All required packages are installed
            set NEED_INSTALL_PACKAGES=0
        )
        
        call deactivate >nul 2>&1
    )
) else (
    echo  Environment not found - will create new one
    set NEED_NEW_ENV=1
    set NEED_INSTALL_PACKAGES=1
)

REM Step 3: Create environment if needed
if %NEED_NEW_ENV%==1 (
    echo.
    echo Creating new virtual environment...
    
    REM Remove existing broken environment
    if exist "%ENV_NAME%" (
        echo Removing broken environment...
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
    )
    
    REM Create new environment
    python -m venv "%ENV_NAME%"
    if %errorlevel% neq 0 (
        echo  Failed to create environment with venv, trying alternative...
        python -m venv "%ENV_NAME%" --without-pip
        if %errorlevel% neq 0 (
            echo  Environment creation failed completely
            echo.
            echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for automatic problem solving
            pause
            exit /b 1
        )
    )

    if exist "%ENV_NAME%\Scripts\python.exe" (
        echo Virtual environment created, checking pip...
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  pip not found in new environment, attempting to install with get-pip.py...
            if exist "get-pip.py" del "get-pip.py"
            curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
            if exist "get-pip.py" (
                "%ENV_NAME%\Scripts\python.exe" get-pip.py >nul 2>&1
                del "get-pip.py" >nul 2>&1
            )
        )
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  Error: Could not install pip in the new environment. Please check your Python installation.
            echo  Try reinstalling Python from python.org and ensure pip is included.
            pause
            exit /b 1
        )
        echo Virtual environment created and pip is available
        set NEED_INSTALL_PACKAGES=1
    ) else (
        echo  Environment creation verification failed
        echo.
        echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for automatic problem solving
        pause
        exit /b 1
    )
)

REM Step 4: Activate environment
echo.
echo Activating virtual environment...
call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo  Failed to activate environment
    echo.
    echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for problem solving
    pause
    exit /b 1
)
echo Environment activated successfully

REM Step 5: Install packages if needed
if %NEED_INSTALL_PACKAGES%==1 (
    echo.
    echo Installing required packages...
    
    REM Upgrade pip first
    python -m pip install --upgrade pip --quiet
    
    REM Check for CUDA availability before installing PyTorch
    echo Checking for CUDA support on your system...
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        echo CUDA detected! Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo CUDA not detected. Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
    )

    REM Install other requirements
    if exist "requirements.txt" (
        echo Installing other packages from requirements.txt...
        REM Use --no-deps to avoid reinstalling torch, then install deps
        pip install -r requirements.txt --no-deps
        pip install -r requirements.txt
    ) else (
        echo  requirements.txt not found, installing other essential packages...
        pip install numpy pandas scikit-learn matplotlib tqdm pydicom opencv-python
    )
) else (
    echo Packages already installed, skipping installation
)

REM Check CUDA availability
echo Checking CUDA availability...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

REM Create necessary directories
echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "figures" mkdir figures
if not exist "data" mkdir data

REM Download dataset
echo ============================================
echo   Downloading Dataset
echo ============================================
echo Downloading RSNA-MICCAI Brain Tumor Dataset...

if exist "data\rsna-miccai-brain-tumor-radiogenomic-classification.rar" (
    echo Dataset archive already exists. Skipping download...
) else (
    echo Starting download from HuggingFace...
    curl -L -o "data\rsna-miccai-brain-tumor-radiogenomic-classification.rar" "https://huggingface.co/datasets/kebincontreras/Glioblastoma_t1w/resolve/main/rsna-miccai-brain-tumor-radiogenomic-classification.rar"
    
    if %errorlevel% neq 0 (
        echo Error: Failed to download dataset
        pause
        exit /b 1
    )
    echo Dataset download completed successfully!
)

REM Extract dataset
set DATASET_DIR=data\rsna-miccai-brain-tumor-radiogenomic-classification
set KEY_FILE=%DATASET_DIR%\train_labels.csv

if exist "%KEY_FILE%" (
    echo Dataset already extracted and verified. Skipping extraction...
    goto :after_extraction
)

echo Extracting dataset...
set EXTRACTION_SUCCESS=0

REM Method 1: Try 7-Zip first (most common on Windows)
echo Trying 7-Zip extraction...
    
    REM Check for 7-Zip in common locations
    set SEVENZIP_FOUND=0
    if exist "C:\Program Files\7-Zip\7z.exe" (
        set SEVENZIP_PATH="C:\Program Files\7-Zip\7z.exe"
        set SEVENZIP_FOUND=1
    ) else if exist "C:\Program Files (x86)\7-Zip\7z.exe" (
        set SEVENZIP_PATH="C:\Program Files (x86)\7-Zip\7z.exe"
        set SEVENZIP_FOUND=1
    ) else (
        where 7z >nul 2>&1
        if %errorlevel% equ 0 (
            set SEVENZIP_PATH=7z
            set SEVENZIP_FOUND=1
        )
    )
    
    if %SEVENZIP_FOUND%==1 (
        echo Found 7-Zip, attempting extraction...
        %SEVENZIP_PATH% x "data\rsna-miccai-brain-tumor-radiogenomic-classification.rar" -o"data" -y >nul 2>&1
        if exist "%KEY_FILE%" (
            echo Dataset extracted successfully with 7-Zip!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo 7-Zip extraction failed, trying next method...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo 7-Zip not found, trying WinRAR...
    )
    
    REM Method 2: Try WinRAR
    echo Trying WinRAR extraction...
    
    REM Check for WinRAR in common locations
    set WINRAR_FOUND=0
    if exist "C:\Program Files\WinRAR\WinRAR.exe" (
        set WINRAR_PATH="C:\Program Files\WinRAR\WinRAR.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files (x86)\WinRAR\WinRAR.exe" (
        set WINRAR_PATH="C:\Program Files (x86)\WinRAR\WinRAR.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files\WinRAR\unrar.exe" (
        set WINRAR_PATH="C:\Program Files\WinRAR\unrar.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files (x86)\WinRAR\unrar.exe" (
        set WINRAR_PATH="C:\Program Files (x86)\WinRAR\unrar.exe"
        set WINRAR_FOUND=1
    ) else (
        where winrar >nul 2>&1
        if %errorlevel% equ 0 (
            set WINRAR_PATH=winrar
            set WINRAR_FOUND=1
        ) else (
            where unrar >nul 2>&1
            if %errorlevel% equ 0 (
                set WINRAR_PATH=unrar
                set WINRAR_FOUND=1
            )
        )
    )
    
    if %WINRAR_FOUND%==1 (
        echo Found WinRAR/unrar, attempting extraction...
        %WINRAR_PATH% x "data\rsna-miccai-brain-tumor-radiogenomic-classification.rar" "data\" >nul 2>&1
        if exist "%KEY_FILE%" (
            echo Dataset extracted successfully with WinRAR!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo WinRAR extraction failed, trying Python packages...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo WinRAR not found, trying Python packages...
    )
    
    REM Method 3: Try Python packages for extraction
    echo Trying Python-based extraction packages...
    
    REM Try rarfile package
    python -c "import rarfile" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Installing rarfile package...
        pip install rarfile --quiet >nul 2>&1
    )
    
    echo Testing rarfile extraction...
    python -c "import rarfile; rf = rarfile.RarFile('data\\rsna-miccai-brain-tumor-radiogenomic-classification.rar'); rf.extractall('data'); rf.close(); print('Python rarfile extraction completed')" 2>nul
    if %errorlevel% equ 0 (
        REM Verify extraction worked by checking for key file
        if exist "%KEY_FILE%" (
            echo Dataset extracted successfully with Python rarfile!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo rarfile extraction created directory but no files found. Trying patoolib...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo Python rarfile extraction failed (missing unrar tool), trying patoolib...
        if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        
        REM Try patoolib package
        python -c "import patoolib" >nul 2>&1
        if %errorlevel% neq 0 (
            echo Installing patoolib package...
            pip install patoolib --quiet >nul 2>&1
        )
        
        echo Testing patoolib extraction...
        python -c "import patoolib; patoolib.extract_archive('data\\rsna-miccai-brain-tumor-radiogenomic-classification.rar', outdir='data'); print('Python patoolib extraction completed')" 2>nul
        if %errorlevel% equ 0 (
            REM Verify extraction worked by checking for key file
            if exist "%KEY_FILE%" (
                echo Dataset extracted successfully with Python patoolib!
                set EXTRACTION_SUCCESS=1
                goto :extraction_complete
            ) else (
                echo patoolib extraction created directory but no files found. Trying PowerShell...
                if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
            )
        ) else (
            echo Python patoolib extraction failed, trying PowerShell...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    )
    
    REM Method 4: Try PowerShell as last resort
    echo Trying PowerShell method (rename to .zip)...
    copy "data\rsna-miccai-brain-tumor-radiogenomic-classification.rar" "data\dataset_temp.zip" >nul 2>&1
    if exist "data\dataset_temp.zip" (
        powershell -Command "try { Expand-Archive -Path 'data\dataset_temp.zip' -DestinationPath 'data\' -Force; exit 0 } catch { exit 1 }" >nul 2>&1
        if exist "%KEY_FILE%" (
            echo Dataset extracted successfully with PowerShell!
            set EXTRACTION_SUCCESS=1
            del "data\dataset_temp.zip" >nul 2>&1
            goto :extraction_complete
        ) else (
            echo PowerShell extraction failed...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
        del "data\dataset_temp.zip" >nul 2>&1
    )
    
    REM All methods failed
    echo.
    echo ============================================
    echo   Manual extraction required
    echo ============================================
    echo All automatic extraction methods failed.
    echo.
    echo The issue: Python packages like rarfile need unrar.exe tool
    echo to extract RAR files, but it's not installed on your system.
    echo.
    echo SOLUTION OPTIONS:
    echo.
    echo Option 1 - Install 7-Zip (Recommended):
    echo   1. Download from: https://www.7-zip.org/
    echo   2. Install and add to PATH
    echo   3. Run this script again
    echo.
    echo Option 2 - Manual extraction:
    echo   1. Right-click: data\rsna-miccai-brain-tumor-radiogenomic-classification.rar
    echo   2. Choose "Extract All" or use any extraction tool
    echo   3. Extract to: data\ folder
    echo.
    echo Option 3 - Install WinRAR:
    echo   1. Download from: https://www.winrar.es/
    echo   2. Install and add to PATH
    echo   3. Run this script again
    echo.
    echo After extraction, you should have:
    echo   data\rsna-miccai-brain-tumor-radiogenomic-classification\train_labels.csv
    echo   data\rsna-miccai-brain-tumor-radiogenomic-classification\train\
    echo   data\rsna-miccai-brain-tumor-radiogenomic-classification\test\
    echo.
    echo Press any key to continue with simulated data...
    pause >nul
    
    :extraction_complete
    if exist "%KEY_FILE%" (
        echo Dataset extraction verified - train_labels.csv found!
        
        REM Check for main directories to confirm extraction quality
        if exist "%DATASET_DIR%\train" (
            echo Training data directory found
        )
        if exist "%DATASET_DIR%\test" (
            echo Test data directory found  
        )
        
        echo Dataset extraction completed successfully!
    ) else (
        echo  Warning: Extraction verification failed - train_labels.csv not found.
    )

:after_extraction

REM Check dependencies before running main script
echo ============================================
echo   Checking Project Dependencies
echo ============================================

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

REM Check Python dependencies
echo Verifying Python packages...
"%ENV_NAME%\Scripts\python.exe" -c "import torch, torchvision, numpy, sklearn, matplotlib, pandas, pydicom, cv2, tqdm; print('All packages available')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ============================================
    echo   Dependency check failed!
    echo ============================================
    echo Please install missing dependencies with:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
) else (
    echo All dependencies are satisfied!
)

REM Run the main script
echo ============================================
echo   Starting GBM Detection Training
echo ============================================

python main.py

REM Check if training completed successfully
if %errorlevel% equ 0 (
    echo ============================================
    echo   Training completed successfully!
    echo ============================================
    echo Results saved in:
    echo   - Models: .\models\
    echo   - Figures: .\figures\
) else (
    echo ============================================
    echo   Training failed!
    echo ============================================
    pause
    exit /b 1
)

REM Deactivate environment
echo Deactivating environment...
call deactivate

echo Script completed successfully!
pause

:end
