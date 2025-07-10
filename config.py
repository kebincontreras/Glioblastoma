"""
Configuration file for GBM Detection Project
Contains paths, hyperparameters, and other settings
"""

import os

# Data paths
DATA_PATH = './data/rsna-miccai-brain-tumor-radiogenomic-classification/'
MODELS_PATH = './models/'
FIGURES_PATH = './figures/'

# Data preprocessing parameters
IMAGE_SIZE = 512
SCALE = 1.0
NUM_SLICES = 8
MRI_TYPE = "T1w"

# Model training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 120
NUM_FOLDS = 5

# Model architecture parameters
NEGATIVE_SLOPE = 0.01  # For LeakyReLU

# Problematic samples to exclude (from original project)
# EXCLUDE_SAMPLES = [109, 123, 709] (for example)
EXCLUDE_SAMPLES = []  # No samples excluded by default

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
USE_CUDA = True  # Set to False to force CPU usage

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [MODELS_PATH, FIGURES_PATH, os.path.dirname(DATA_PATH)]
    for directory in directories:
        if directory:  # Only create non-empty directory paths
            os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Data path: {DATA_PATH}")
    print(f"Models path: {MODELS_PATH}")
    print(f"Figures path: {FIGURES_PATH}")
