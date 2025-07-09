import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import pydicom as dicom
import cv2
import glob
import re
from tqdm import tqdm
import json
import pickle
import hashlib

# Import configuration
from config import *

# Cached Data Management
def get_cache_filename(mri_type, num_slices, image_size, exclude_samples):
    """
    Generate a unique cache filename based on preprocessing parameters.
    """
    # Create a hash of the parameters to ensure uniqueness
    params_str = f"{mri_type}_{num_slices}_{image_size}_{sorted(exclude_samples)}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    cache_filename = f"preprocessed_data_{mri_type}_{num_slices}slices_{image_size}px_{params_hash}.pkl"
    cache_path = os.path.join('./data', cache_filename)
    
    return cache_path

def save_preprocessed_data(X_processed, y, patient_ids, cache_path):
    """
    Save preprocessed data to cache file.
    """
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cache_data = {
            'X_processed': X_processed,
            'y': y,
            'patient_ids': patient_ids,
            'preprocessing_params': {
                'mri_type': MRI_TYPE,
                'num_slices': NUM_SLICES,
                'image_size': IMAGE_SIZE,
                'exclude_samples': EXCLUDE_SAMPLES
            },
            'cache_version': '1.0'
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Preprocessed data saved to cache: {cache_path}")
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"   Cache file size: {file_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error saving preprocessed data to cache: {e}")

def load_preprocessed_data(cache_path):
    """
    Load preprocessed data from cache file.
    """
    try:
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify cache compatibility
        if cache_data.get('cache_version') != '1.0':
            print("⚠️ Cache version mismatch. Will regenerate preprocessed data.")
            return None
        
        cached_params = cache_data.get('preprocessing_params', {})
        current_params = {
            'mri_type': MRI_TYPE,
            'num_slices': NUM_SLICES,
            'image_size': IMAGE_SIZE,
            'exclude_samples': EXCLUDE_SAMPLES
        }
        
        if cached_params != current_params:
            print("⚠️ Preprocessing parameters changed. Will regenerate preprocessed data.")
            print(f"   Cached: {cached_params}")
            print(f"   Current: {current_params}")
            return None
        
        print(f"✅ Loaded preprocessed data from cache: {cache_path}")
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"   Cache file size: {file_size_mb:.1f} MB")
        
        return cache_data['X_processed'], cache_data['y'], cache_data['patient_ids']
        
    except Exception as e:
        print(f"❌ Error loading preprocessed data from cache: {e}")
        return None

def clear_cache(mri_type=None):
    """
    Clear preprocessed data cache files.
    
    Args:
        mri_type: If specified, only clear cache for this MRI type. 
                 If None, clear all cache files.
    """
    cache_dir = './data'
    if not os.path.exists(cache_dir):
        return
    
    pattern = "preprocessed_data_*.pkl"
    if mri_type:
        pattern = f"preprocessed_data_{mri_type}_*.pkl"
    
    cache_files = glob.glob(os.path.join(cache_dir, pattern))
    
    if not cache_files:
        print("🧹 No cache files found to clear.")
        return
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"🗑️ Removed cache file: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"❌ Error removing {cache_file}: {e}")
    
    print(f"✅ Cache cleanup completed. Removed {len(cache_files)} files.")

def load_rsna_data_with_cache(data_path=DATA_PATH, mri_type=MRI_TYPE, split="train"):
    """
    Load RSNA-MICCAI dataset with caching support for preprocessed data.
    """
    # Generate cache filename
    cache_path = get_cache_filename(mri_type, NUM_SLICES, IMAGE_SIZE, EXCLUDE_SAMPLES)
    
    print(f"🔍 Checking for cached preprocessed data...")
    
    # Try to load from cache first
    cached_data = load_preprocessed_data(cache_path)
    if cached_data is not None:
        X_processed, y, patient_ids = cached_data
        print(f"📦 Using cached data - Shape: {X_processed.shape}")
        return X_processed, y, patient_ids
    
    print(f"💾 No valid cache found. Processing data from scratch...")
    
    # Load raw data (original function)
    X_raw, y, patient_ids = load_rsna_data(data_path, mri_type, split)
    
    if X_raw is None:
        return None, None, None
    
    # Apply 4-phase preprocessing
    print(f"🔄 Applying 4-phase RSNA-MICCAI preprocessing...")
    X_processed = preprocess_images(X_raw, slices_per_patient=NUM_SLICES)
    
    # Save to cache for future use
    print(f"💾 Saving preprocessed data to cache...")
    save_preprocessed_data(X_processed, y, patient_ids, cache_path)
    
    return X_processed, y, patient_ids

# DICOM Data Loading Functions (adapted from existing project)
def load_dicom_image(path, img_size=IMAGE_SIZE, scale=SCALE):
    """
    Load DICOM image, crop and apply preprocessing.
    Adapted from the existing project's load_dicom_image function.
    """
    try:
        # Load DICOM image
        img = dicom.dcmread(path).pixel_array
        
        # Crop image to reduce black borders
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]

        # Resize to target size
        img = cv2.resize(img, (img_size, img_size))
        
        # Normalize image
        if np.std(img) > 0:
            img = (img - np.mean(img)) / np.std(img)
        
        return img
    except Exception as e:
        print(f"Error loading DICOM file {path}: {e}")
        return np.zeros((img_size, img_size))

def load_dicom_images_sequence(scan_id, num_imgs=NUM_SLICES, img_size=IMAGE_SIZE, 
                              mri_type=MRI_TYPE, split="train", data_path=DATA_PATH):
    """
    Load a sequence of DICOM images for a patient.
    Adapted from the existing project's load_dicom_images_3d function.
    """
    try:
        # Build the DICOM directory path using os.path.join for Windows compatibility
        dicom_dir = os.path.join(data_path, split, scan_id, mri_type)
        # Debug: print the directory being searched
        # if not os.path.exists(dicom_dir):
        #     print(f"[DEBUG] DICOM directory does not exist: {dicom_dir}")
        # else:
        #     print(f"[DEBUG] Searching DICOM files in: {dicom_dir}")

        # Find all DICOM files for the patient
        files = sorted(
            glob.glob(os.path.join(dicom_dir, "*.dcm")),
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]
        )

        # Debug: print number of files found
        # print(f"[DEBUG] Found {len(files)} DICOM files for patient {scan_id} in {dicom_dir}")

        if not files:
            # List directory contents for debugging
            # if os.path.exists(dicom_dir):
            #     print(f"[DEBUG] Directory contents: {os.listdir(dicom_dir)}")
            print(f"No DICOM files found for patient {scan_id}")
            return np.zeros((num_imgs, img_size, img_size))

        # Select middle slices to avoid empty ones
        middle = len(files) // 2
        num_imgs_half = num_imgs // 2
        start_idx = max(0, middle - num_imgs_half)
        end_idx = min(len(files), middle + num_imgs_half)

        # Load and process images
        images = []
        for f in files[start_idx:end_idx]:
            img = load_dicom_image(f, img_size)
            images.append(img)

        # Ensure we have exactly num_imgs images
        if len(images) < num_imgs:
            # Pad with zeros or duplicate the last image
            while len(images) < num_imgs:
                if images:
                    images.append(images[-1])  # Duplicate last image
                else:
                    images.append(np.zeros((img_size, img_size)))
        elif len(images) > num_imgs:
            images = images[:num_imgs]

        return np.array(images)

    except Exception as e:
        print(f"Error loading sequence for patient {scan_id}: {e}")
        return np.zeros((num_imgs, img_size, img_size))

def load_rsna_data(data_path=DATA_PATH, mri_type=MRI_TYPE, split="train"):
    """
    Load RSNA-MICCAI dataset with proper DICOM image loading.
    """
    try:
        # Load labels
        if split == "train":
            labels_df = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
            # Exclude problematic samples as done in original project
            labels_df = labels_df[~labels_df['BraTS21ID'].isin(EXCLUDE_SAMPLES)]
            # Format IDs with leading zeros
            labels_df['BraTS21ID_formatted'] = labels_df['BraTS21ID'].apply(lambda x: f"{x:05d}")
            X_ids = labels_df['BraTS21ID_formatted'].values
            y = labels_df['MGMT_value'].values
        else:  # test split
            test_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
            test_df['BraTS21ID_formatted'] = test_df['BraTS21ID'].apply(lambda x: f"{x:05d}")
            X_ids = test_df['BraTS21ID_formatted'].values
            y = None
        # Load images for each patient
        X_images = []
        print(f"Loading {len(X_ids)} patients from {split} set...")
        for patient_id in tqdm(X_ids):
            patient_images = load_dicom_images_sequence(
                patient_id, num_imgs=NUM_SLICES, mri_type=mri_type, 
                split=split, data_path=data_path
            )
            X_images.append(patient_images)
        X_images = np.array(X_images)
        print(f"Loaded {len(X_images)} patients with shape: {X_images.shape}")
        return X_images, y, X_ids
    except Exception as e:
        print(f"Error loading RSNA data: {e}")
        print("No se pudo cargar el dataset. Abortando ejecución.")
        import sys
        sys.exit(1)

# Data Preprocessing - 4 Phase RSNA-MICCAI Methodology
def is_completely_black_image(img, black_threshold=0.01):
    """
    Phase 1: Determine if an image is completely black (no diagnostic information).
    
    Args:
        img: 2D MRI slice
        black_threshold: Threshold for considering image as black
        
    Returns:
        bool: True if image is completely black/non-informative
    """
    # Check if image is essentially all zeros/black
    non_zero_ratio = np.count_nonzero(img) / img.size
    mean_intensity = np.mean(np.abs(img))
    
    # An image is considered black if it has very few non-zero pixels and low intensity
    return non_zero_ratio < black_threshold or mean_intensity < black_threshold

def resize_with_padding(img, target_size=512):
    """
    Phase 2: Resize image to 512x512 with proper padding for smaller images.
    
    Args:
        img: 2D image array
        target_size: Target size (512x512)
        
    Returns:
        resized_img: Image resized to target_size x target_size
    """
    h, w = img.shape[:2]
    
    if h == target_size and w == target_size:
        return img
    
    # If image is smaller, pad it
    if h < target_size or w < target_size:
        # Calculate padding
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        # Add symmetric padding
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        img_padded = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=0)
        
        # If still not exactly target_size due to odd dimensions, resize
        if img_padded.shape[0] != target_size or img_padded.shape[1] != target_size:
            img_padded = cv2.resize(img_padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        return img_padded
    else:
        # If image is larger, resize to maintain uniformity
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def calculate_information_content(img):
    """
    Phase 4: Calculate information content using binary mask methodology.
    
    Args:
        img: 2D MRI slice (normalized values)
        
    Returns:
        float: Information content score (area of non-zero pixels after binary conversion)
    """
    # Convert image to 0-255 range for binary mask application
    img_normalized = img.copy()
    
    # Normalize to 0-255 range
    if np.max(img_normalized) > np.min(img_normalized):
        img_255 = ((img_normalized - np.min(img_normalized)) / 
                  (np.max(img_normalized) - np.min(img_normalized)) * 255).astype(np.uint8)
    else:
        img_255 = np.zeros_like(img_normalized, dtype=np.uint8)
    
    # Apply binary mask: pixels > 1 → 1, pixels ≤ 1 → 0
    binary_mask = (img_255 > 1).astype(np.uint8)
    
    # Calculate area of non-zero pixels (information content)
    information_content = np.sum(binary_mask) / binary_mask.size
    
    return information_content

def preprocess_images(images, slices_per_patient=8):
    """
    4-Phase RSNA-MICCAI preprocessing methodology:
    
    Phase 1: Remove completely black images (target: 27.7% removal)
    Phase 2: Resize all images to 512×512 (padding smaller, resizing larger)  
    Phase 3: Standardize to exactly 8 images per patient
    Phase 4: For >8 images: select 8 most informative using binary mask area
             For <8 images: duplicate maximum diagnostic information image
    
    Args:
        images: Array of shape (num_patients, num_slices, height, width)
    Returns:
        processed_images: Array of shape (num_patients, 8, 512, 512)
    """
    processed_images = []
    total_original_images = 0
    total_black_images_removed = 0
    
    print(f"Starting 4-Phase RSNA-MICCAI preprocessing for {len(images)} patients...")
    
    for patient_idx, patient_imgs in enumerate(images):
        # Handle different input shapes
        if len(patient_imgs.shape) == 4:  # Has channel dimension
            patient_imgs = patient_imgs[:, :, :, 0]  # Take first channel
        
        total_original_images += len(patient_imgs)
        
        # PHASE 1: Remove completely black images (27.7% target removal)
        non_black_slices = []
        black_count = 0
        
        for img in patient_imgs:
            if not is_completely_black_image(img):
                non_black_slices.append(img)
            else:
                black_count += 1
        
        total_black_images_removed += black_count
        
        # If all images are black, keep at least one for processing
        if len(non_black_slices) == 0:
            print(f"Warning: All images black for patient {patient_idx}. Keeping middle slice.")
            middle_idx = len(patient_imgs) // 2
            non_black_slices = [patient_imgs[middle_idx]]
        
        # PHASE 2: Resize all images to 512×512 with proper padding/resizing
        resized_slices = []
        for img in non_black_slices:
            # Normalize image first (as done in original DICOM loading)
            if np.std(img) > 0:
                normalized_img = (img - np.mean(img)) / np.std(img)
            else:
                normalized_img = img
            
            # Resize with padding methodology
            resized_img = resize_with_padding(normalized_img, target_size=512)
            resized_slices.append(resized_img)
        
        # PHASE 3 & 4: Standardize to exactly 8 images per patient
        if len(resized_slices) == slices_per_patient:
            # Perfect number - no adjustment needed
            final_slices = resized_slices
            
        elif len(resized_slices) > slices_per_patient:
            # PHASE 4a: More than 8 images - select 8 most informative using binary mask
            information_scores = []
            for img in resized_slices:
                info_score = calculate_information_content(img)
                information_scores.append(info_score)
            
            # Select indices of 8 most informative images
            top_indices = np.argsort(information_scores)[-slices_per_patient:]
            final_slices = [resized_slices[i] for i in sorted(top_indices)]
            
        else:
            # PHASE 4b: Less than 8 images - duplicate maximum diagnostic information image
            if len(resized_slices) > 0:
                # Find image with maximum diagnostic information
                information_scores = []
                for img in resized_slices:
                    info_score = calculate_information_content(img)
                    information_scores.append(info_score)
                
                max_info_idx = np.argmax(information_scores)
                max_info_image = resized_slices[max_info_idx]
                
                # Start with existing slices and duplicate the most informative one
                final_slices = resized_slices.copy()
                while len(final_slices) < slices_per_patient:
                    final_slices.append(max_info_image.copy())
            else:
                # Fallback: create zero images
                final_slices = [np.zeros((512, 512)) for _ in range(slices_per_patient)]
        
        processed_images.append(final_slices)
    
    # Calculate and report removal statistics
    if total_original_images > 0:
        removal_percentage = (total_black_images_removed / total_original_images) * 100
        print(f"\n4-Phase Preprocessing Results:")
        print(f"  Phase 1 - Black image removal:")
        print(f"    - Original images: {total_original_images}")
        print(f"    - Black images removed: {total_black_images_removed}")
        print(f"    - Removal percentage: {removal_percentage:.1f}% (target: 27.7%)")
        print(f"  Phase 2 - All images resized to 512×512 with padding")
        print(f"  Phase 3 - Standardized to {slices_per_patient} images per patient")
        print(f"  Phase 4 - Applied binary mask for information content selection")
        print(f"  Final shape: {np.array(processed_images).shape}")
    
    return np.array(processed_images)  # Shape: (num_patients, 8, 512, 512)

# CNN Model Definition
class GBMCNN(nn.Module):
    """
    5-layer CNN for GBM detection with Leaky ReLU, ReLU, and Sigmoid activations.
    """
    def __init__(self):
        super(GBMCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Flatten and FC will be set dynamically
        self._feature_dim = None
        self.fc1 = None
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def _get_flattened_size(self, x):
        # Pass a dummy tensor through conv layers to get output size
        with torch.no_grad():
            x = self.pool1(F.leaky_relu(self.conv1(x), negative_slope=NEGATIVE_SLOPE))
            x = self.pool2(F.leaky_relu(self.conv2(x), negative_slope=NEGATIVE_SLOPE))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            x = self.pool5(F.relu(self.conv5(x)))
            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x), negative_slope=NEGATIVE_SLOPE))
        x = self.pool2(F.leaky_relu(self.conv2(x), negative_slope=NEGATIVE_SLOPE))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)

        # Set FC dynamically on first forward
        if self.fc1 is None or self._feature_dim != x.shape[1]:
            self._feature_dim = x.shape[1]
            self.fc1 = nn.Linear(self._feature_dim, 1).to(x.device)

        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))
        return x

def build_cnn_model():
    """
    Build and return the CNN model.
    """
    return GBMCNN()

def load_existing_model(model_path=None):
    """
    Load existing model if available.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_PATH, 'gbm_v5.pt')
    
    try:
        if os.path.exists(model_path):
            model = build_cnn_model()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded existing model from: {model_path}")
            return model
        else:
            print(f"No existing model found at: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading existing model: {e}")
        return None

# Training with 5-Fold Cross-Validation
def train_model(X_train, y_train):
    """
    Train the CNN model using 5-fold cross-validation with CUDA support.
    X_train: Preprocessed images of shape (num_patients, 8, 512, 512, 1)
    y_train: Binary labels (0 or 1) for GBM detection
    """
    # Check for CUDA availability
    device = torch.device('cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available() and USE_CUDA:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    histories = []
    
    # Reshape to treat each slice as a sample (num_patients * 8, 1, 512, 512)
    X_flat = X_train.reshape(-1, 1, 512, 512)  # PyTorch format: (N, C, H, W)
    y_flat = np.repeat(y_train, 8)  # Repeat labels for each slice
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining Fold {fold + 1}/5")
        
        # Index into flattened arrays
        train_slice_idx = np.concatenate([np.arange(i * 8, (i + 1) * 8) for i in train_idx])
        val_slice_idx = np.concatenate([np.arange(i * 8, (i + 1) * 8) for i in val_idx])
        
        X_tr, X_val = X_flat[train_slice_idx], X_flat[val_slice_idx]
        y_tr, y_val = y_flat[train_slice_idx], y_flat[val_slice_idx]
        
        # Convert to PyTorch tensors and move to device
        X_tr_tensor = torch.FloatTensor(X_tr).to(device)
        y_tr_tensor = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = build_cnn_model().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Store history for this fold
        fold_history = {
            'loss': train_losses,
            'val_loss': val_losses,
            'accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }
        histories.append(fold_history)
    
    return model, histories

def plot_training_results(histories):
    """
    Plot training history and save comprehensive results.
    
    Args:
        histories: List of training histories from cross-validation folds
    
    Returns:
        dict: Dictionary containing training results summary
    """
    if not histories:
        print("No training histories available for plotting.")
        return None
    
    last_fold = histories[-1]
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(last_fold['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(last_fold['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy (Last Fold)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(last_fold['loss'], label='Training Loss', linewidth=2)
    plt.plot(last_fold['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss (Last Fold)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy across all folds
    plt.subplot(1, 3, 3)
    fold_val_accs = [fold['val_accuracy'][-1] for fold in histories]
    plt.bar(range(1, len(fold_val_accs) + 1), fold_val_accs, alpha=0.7)
    plt.title('Final Validation Accuracy per Fold', fontsize=14, fontweight='bold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure in figures directory
    figure_path = os.path.join(FIGURES_PATH, 'training_history_pytorch.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training plots saved to: {figure_path}")
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    final_train_acc = last_fold['accuracy'][-1]
    final_val_acc = last_fold['val_accuracy'][-1]
    print(f"Final Training Accuracy (Last Fold): {final_train_acc:.4f}")
    print(f"Final Validation Accuracy (Last Fold): {final_val_acc:.4f}")
    
    # Calculate comprehensive statistics across all folds
    all_val_accs = [fold['val_accuracy'][-1] for fold in histories]
    all_val_losses = [fold['val_loss'][-1] for fold in histories]
    
    print(f"\nCross-Validation Results ({NUM_FOLDS}-Fold):")
    print(f"Average Validation Accuracy: {np.mean(all_val_accs):.4f} ± {np.std(all_val_accs):.4f}")
    print(f"Average Validation Loss: {np.mean(all_val_losses):.4f} ± {np.std(all_val_losses):.4f}")
    print(f"Best Fold Accuracy: {np.max(all_val_accs):.4f}")
    print(f"Worst Fold Accuracy: {np.min(all_val_accs):.4f}")
    
    # Save results to file
    results_dict = {
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'cv_mean_accuracy': np.mean(all_val_accs),
        'cv_std_accuracy': np.std(all_val_accs),
        'cv_mean_loss': np.mean(all_val_losses),
        'cv_std_loss': np.std(all_val_losses),
        'fold_accuracies': all_val_accs,
        'fold_losses': all_val_losses,
        'configuration': {
            'num_slices': NUM_SLICES,
            'image_size': IMAGE_SIZE,
            'mri_type': MRI_TYPE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_folds': NUM_FOLDS
        }
    }
    
    results_path = os.path.join(FIGURES_PATH, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    return results_dict
