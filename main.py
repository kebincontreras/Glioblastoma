import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Import configuration
from config import *

# Import utilities
from src.utils import (
    load_rsna_data, 
    load_rsna_data_with_cache,
    preprocess_images, 
    train_model, 
    plot_training_results,
    get_training_summary,
    check_dependencies
)

# Main Execution
if __name__ == "__main__":
    # Print training configuration and check dependencies
    deps_ok = get_training_summary()
    
    if not deps_ok:
        print("\n Missing dependencies detected. Please install requirements first:")
        print("   pip install -r requirements.txt")
        exit(1)
    
    # Ensure directories exist
    ensure_directories()
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    print("="*60)
    print("GBM Detection using CNN - RSNA-MICCAI Dataset")
    print("="*60)
    
    # Load RSNA-MICCAI data with caching support
    X_rsna_processed, y_rsna, patient_ids = load_rsna_data_with_cache(DATA_PATH, MRI_TYPE)
    
    if X_rsna_processed is None:
        print("Real data not found. Using simulated data for testing...")
        # Fallback to simulated data if real data is not available
        num_patients = 100  # Reduced for testing
        X_rsna = np.random.rand(num_patients, NUM_SLICES, IMAGE_SIZE, IMAGE_SIZE)
        y_rsna = np.random.randint(0, 2, num_patients)
        print(f"Generated simulated data: {X_rsna.shape}")
        
        # Apply preprocessing to simulated data
        print("Applying preprocessing to simulated data...")
        X_rsna_processed = preprocess_images(X_rsna, slices_per_patient=NUM_SLICES)
    else:
        print(f"Successfully loaded preprocessed RSNA data: {X_rsna_processed.shape}")
        print(f"Labels distribution: {np.bincount(y_rsna)}")
    
    print(f"Final preprocessed data shape: {X_rsna_processed.shape}")
    
    # Train Model
    print("\nStarting model training...")
    model, histories = train_model(X_rsna_processed, y_rsna)
    
    # Save the model in the models directory (matching project structure)
    model_save_path = os.path.join(MODELS_PATH, 'gbm_cnn_pytorch.pth')
    model_complete_path = os.path.join(MODELS_PATH, 'gbm_cnn_complete.pth')
    
    torch.save(model.state_dict(), model_save_path)
    torch.save(model, model_complete_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Complete model saved to: {model_complete_path}")
    
    # Plot training results using utility function
    plot_training_results(histories)
    
    # Clean up GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nGPU memory cleared.")
        
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)