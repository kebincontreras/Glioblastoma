# Deep Learning for Glioblastoma Multiforme Detection from MRI: A Statistical Analysis for Demographic Bias

This repository contains the codebase, model architecture, and supplementary materials for our research on non-invasive detection of Glioblastoma Multiforme (GBM) using Convolutional Neural Networks (CNNs) applied to T1-weighted MRI scans.

The study investigates the model’s generalisation capacity across anatomical differences and evaluates demographic fairness using statistical hypothesis testing.

⚠️ **Note:** This repository is part of an ongoing peer-review process. The final code release and associated post-processing scripts will be published once the manuscript is accepted in the prestigious journal *Applied Sciences (MDPI)*.

## Repository Contents

- **`main.py`**: Main training script with integrated DICOM data loading and CNN training
- **`inference.py`**: Inference script for making predictions on new data
- **`config.py`**: Configuration file with all hyperparameters and paths
- **`src/`**: Source code modules containing statistical analysis functions and utilities
  - **`stats.py`**: Statistical analysis functions for demographic bias evaluation (applied exclusively to EGD dataset in the paper)
  - **`utils.py`**: Utility functions for data processing and analysis
- **`notebooks/`**: Jupyter notebooks and analysis scripts
  - **`GBM.py`**: Code used for Kaggle challenge submissions and analysis
- **`models/`**: Saved model weights and configurations
- **`figures/`**: Visualizations, performance plots, and results
- **`scripts/`**: Automation and troubleshooting scripts for project setup
- **`run_project.bat/.sh`**: Main scripts to execute the complete project (Windows/Linux-macOS)
- **`requirements.txt`**: Python dependencies

## Key Features

- CNN optimised for GBM detection using RSNA-MICCAI dataset.
- External validation with the Erasmus Glioma Database (EGD).
- Statistical evaluation of fairness using Shapiro-Wilk, Mann-Whitney U, and Chi-squared tests.
- Performance metrics include: AUC-ROC, F1-score, Precision, Accuracy, and False Negative Rate.

## Dataset

The **RSNA-MICCAI Brain Tumor Radiogenomic Classification** dataset, which provides annotated pre-operative T1-weighted MRI scans for glioblastoma analysis.

🧠 **Access the dataset here:**  
[https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification)

Note: You must be logged in to Kaggle and accept the competition rules to download the dataset.

### External Validation Dataset

For external validation, this study uses the **Erasmus Glioblastoma Database (EGD)**.

*Note: Access to the EGD dataset requires approval from the data custodians and adherence to their usage terms and conditions.*

## Usage & Automation Scripts

All project setup, troubleshooting, and maintenance is fully automated. Use the following scripts as needed:

### Main Execution
- **Windows:**
  ```bat
  run_project.bat
  ```
- **Linux/macOS:**
  ```bash
  ./run_project.sh
  ```
  
  > **Note:** If you encounter permission errors on Linux/macOS, first make the script executable by running:
  > ```bash
  > chmod +x run_project.sh
  > ```

### GPU Memory Optimization

⚠️ **VRAM Memory Issues:** If you encounter out-of-memory errors during training, you can reduce the batch size in the `config.py` file:

- For example, **For 12GB GPUs:** Recommended batch size is **16**.

To modify the batch size, edit the `BATCH_SIZE` parameter in `config.py`:
```python
BATCH_SIZE = 16  # Adjust based on your GPU memory
```

### Troubleshooting & Utilities (in `scripts/`)
- **Check system health:**
  ```bat
  scripts\health_check.bat     # Windows
  ./scripts/health_check.sh    # Linux/macOS
  ```
- **Full troubleshooting:**
  ```bat
  scripts\troubleshoot.bat     # Windows
  ./scripts/troubleshoot.sh    # Linux/macOS
  ```
- **Repair pip installation:**
  ```bat
  scripts\fix_pip.bat          # Windows
  ./scripts/fix_pip.sh         # Linux/macOS
  ```
- **Detailed diagnostics:**
  ```bat
  scripts\diagnose.bat         # Windows only
  ```
- **Cleanup environments and temp files:**
  ```bat
  scripts\cleanup.bat          # Windows
  ./scripts/cleanup.sh         # Linux/macOS
  ```
- **Get help and usage info:**
  ```bat
  scripts\help.bat             # Windows
  ./scripts/help.sh            # Linux/macOS
  ```

> All scripts are self-healing and provide clear instructions if any issue is detected. For most users, simply running `run_project.bat` or `run_project.sh` is enough for a complete setup and execution.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

## Contact

For academic or technical inquiries, please contact:

**Kebin Contreras**  
Department of Biology, Universidad del Cauca  
📧 [kacontreras@unicauca.edu.co](mailto:kacontreras@unicauca.edu.co)
