"""
Inference script for GBM Detection
Loads trained models and performs predictions on new data
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.utils import GBMCNN, load_dicom_images_sequence, preprocess_images
from config import *
import pandas as pd

def load_trained_model(model_path=None):
    """
    Load a trained CNN model for inference.
    """
    if model_path is None:
        # Try different model paths
        possible_paths = [
            os.path.join(MODELS_PATH, 'gbm_cnn_pytorch.pth'),
            os.path.join(MODELS_PATH, 'gbm_v5.pt'),
            os.path.join(MODELS_PATH, 'gbm_cnn_complete.pth')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    # Determine device
    device = torch.device('cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu')
    
    # Load model
    try:
        if model_path.endswith('_complete.pth'):
            # Load complete model
            model = torch.load(model_path, map_location=device)
        else:
            # Load state dict
            model = GBMCNN()
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model = model.to(device)
        model.eval()
        print(f"Successfully loaded model from: {model_path}")
        return model, device
        
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict_patient(model, patient_data, device):
    """
    Make prediction for a single patient.
    
    Args:
        model: Trained CNN model
        patient_data: Preprocessed patient data (num_slices, height, width)
        device: Computing device
    
    Returns:
        prediction_prob: Probability of GBM (float)
        slice_predictions: Individual predictions for each slice
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare data (add batch and channel dimensions)
        patient_tensor = torch.FloatTensor(patient_data).unsqueeze(1).to(device)  # (num_slices, 1, H, W)
        
        # Get predictions for each slice
        slice_predictions = model(patient_tensor).cpu().numpy().flatten()
        
        # Aggregate predictions (average)
        prediction_prob = np.mean(slice_predictions)
        
    return prediction_prob, slice_predictions

def predict_from_dicom(model, patient_id, device, data_path=DATA_PATH, mri_type=MRI_TYPE):
    """
    Make prediction directly from DICOM files.
    
    Args:
        model: Trained CNN model
        patient_id: Patient ID (string, e.g., "00046")
        device: Computing device
        data_path: Path to RSNA dataset
        mri_type: Type of MRI sequence
    
    Returns:
        prediction_prob: Probability of GBM
        slice_predictions: Individual predictions for each slice
    """
    # Load DICOM images
    patient_images = load_dicom_images_sequence(
        patient_id, num_imgs=NUM_SLICES, mri_type=mri_type, 
        split="train", data_path=data_path
    )
    
    # Preprocess
    processed_images = preprocess_images(
        np.expand_dims(patient_images, 0), 
        slices_per_patient=NUM_SLICES
    )[0]  # Remove batch dimension
    
    # Predict
    return predict_patient(model, processed_images, device)

def visualize_prediction(patient_data, slice_predictions, prediction_prob, patient_id=None):
    """
    Visualize patient slices with predictions.
    """
    num_slices = len(patient_data)
    cols = 4
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_slices):
        row = i // cols
        col = i % cols
        
        axes[row, col].imshow(patient_data[i], cmap='gray')
        axes[row, col].set_title(f'Slice {i+1}\nProb: {slice_predictions[i]:.3f}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    title = f'Patient {patient_id} - ' if patient_id else ''
    title += f'Overall GBM Probability: {prediction_prob:.3f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def batch_inference(model, device, data_path=DATA_PATH, mri_type=MRI_TYPE, max_patients=10):
    """
    Perform inference on multiple patients and save results.
    """
    # Try to load patient IDs from train labels
    try:
        labels_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
        labels_df = labels_df[~labels_df['BraTS21ID'].isin(EXCLUDE_SAMPLES)]
        patient_ids = labels_df['BraTS21ID'].apply(lambda x: f"{x:05d}").values[:max_patients]
        true_labels = labels_df['MGMT_value'].values[:max_patients]
    except:
        print("Could not load patient labels. Using sample patient IDs.")
        patient_ids = [f"{i:05d}" for i in range(46, 46 + max_patients)]  # Sample IDs
        true_labels = None
    
    results = []
    print(f"Running inference on {len(patient_ids)} patients...")
    
    for i, patient_id in enumerate(patient_ids):
        try:
            pred_prob, slice_preds = predict_from_dicom(
                model, patient_id, device, data_path, mri_type
            )
            
            result = {
                'patient_id': patient_id,
                'prediction_probability': pred_prob,
                'prediction_binary': int(pred_prob > 0.5),
                'slice_predictions': slice_preds.tolist()
            }
            
            if true_labels is not None:
                result['true_label'] = int(true_labels[i])
                result['correct'] = (result['prediction_binary'] == result['true_label'])
            
            results.append(result)
            print(f"Patient {patient_id}: {pred_prob:.3f} (predicted: {'GBM' if pred_prob > 0.5 else 'No GBM'})")
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue
    
    # Save results
    ensure_directories()
    results_path = os.path.join(FIGURES_PATH, 'inference_results.json')
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference results saved to: {results_path}")
    
    # Calculate accuracy if true labels available
    if true_labels is not None:
        correct_predictions = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_predictions / len(results)
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{len(results)})")
    
    return results

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
                pred_prob, slice_preds = predict_from_dicom(
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
                    processed_images, slice_preds, pred_prob, sample_patient
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
