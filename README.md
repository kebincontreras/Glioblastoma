# Deep Learning for Glioblastoma Multiforme Detection from MRI: A Statistical Analysis for Demographic Bias

This repository contains the codebase, model architecture, and supplementary materials for our research on non-invasive detection of Glioblastoma Multiforme (GBM) using Convolutional Neural Networks (CNNs) applied to T1-weighted MRI scans.

The study investigates the model’s generalisation capacity across anatomical differences and evaluates demographic fairness using statistical hypothesis testing.

⚠️ **Note:** This repository is part of an ongoing peer-review process. The final code release and associated post-processing scripts will be published once the manuscript is accepted in the prestigious journal *Applied Sciences (MDPI)*.

## Repository Contents

- `src/`: Source code for data preprocessing, CNN architecture, training, and inference.
- `notebooks/`: Jupyter notebooks used for analysis and statistical tests.
- `models/`: Saved model weights and configurations (to be uploaded post-acceptance).
- `figures/`: Visualisations and performance plots.
- `docs/`: Supplementary documentation.

## Key Features

- CNN optimised for GBM detection using RSNA-MICCAI dataset.
- External validation with the Erasmus Glioma Database (EGD).
- Statistical evaluation of fairness using Shapiro-Wilk, Mann-Whitney U, and Chi-squared tests.
- Performance metrics include: AUC-ROC, F1-score, Precision, Accuracy, and False Negative Rate.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

## Contact

For academic or technical inquiries, please contact:

**Kebin Contreras**  
Department of Biology, Universidad del Cauca  
📧 [kacontreras@unicauca.edu.co](mailto:kacontreras@unicauca.edu.co)
