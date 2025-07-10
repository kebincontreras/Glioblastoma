"""
Inference script for GBM Detection
Loads trained models and performs predictions on new data
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils import (
    load_trained_model,
    predict_from_dicom,
    visualize_prediction,
    batch_inference,
    load_dicom_images_sequence,
    preprocess_images
)
from config import *

def main():
    """
    Main inference function.
    """
    print("="*60)
    print("GBM Detection - Inference")
    print("="*60)
    
    try:
        # Load trained model
        model, device = load_trained_model()
        
        # Run batch inference
        results = batch_inference(model, device)
        
        # Example: Visualize one patient
        if results:
            sample_patient = results[0]['patient_id']
            print(f"\nVisualizing sample patient: {sample_patient}")
            
            try:
                pred_prob = predict_from_dicom(
                    model, sample_patient, device
                )
                
                # Load patient data for visualization
                patient_images = load_dicom_images_sequence(
                    sample_patient, num_imgs=NUM_SLICES
                )
                processed_images = preprocess_images(
                    np.expand_dims(patient_images, 0)
                )[0]
                
                fig = visualize_prediction(
                    processed_images, pred_prob, sample_patient
                )
                
                # Save visualization
                viz_path = os.path.join(FIGURES_PATH, f'sample_prediction_{sample_patient}.png')
                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Visualization saved to: {viz_path}")
                
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
