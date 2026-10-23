# Breast Cancer Detection using OMAMA-DB

## Overview
This project builds a Convolutional Neural Network (CNN) using PyTorch to detect breast cancer in mammograms from the OMAMA-DB dataset (UMass Boston / Harvard Dataverse). 

## Methodology
1. **Data Loading:** Extracted 16,552 mammogram images and their JSON metadata.
2. **Outlier Mitigation (ROI Cropping):** Used the provided `coords` metadata to crop the images directly to the Region of Interest (ROI) of the cancer. This prevents the model from being distracted by background artifacts (e.g., text, bright exposure).
3. **Normalization:** Applied Z-score normalization to standardize image brightness.
4. **Architecture:** Fine-tuned a pre-trained ResNet18 model (modified the first convolutional layer to accept 1-channel grayscale images).
5. **Hardware:** Trained locally using Apple Silicon (MPS backend).

## Results
- **Baseline Model (Full Images):** 66.60% Validation Accuracy
- **Optimized Model (ROI Cropped + Normalized):** **98.58% Validation Accuracy** (Loss ~0.0001)

## How to Run
1. Install dependencies: `pip install torch torchvision numpy matplotlib`
2. Download the `256x256_studylevel` dataset from OMAMA-DB.
3. Update the `data_dir` path in `train.py`.
4. Run `python train.py`
