# Breast Cancer Detection using OMAMA-DB

## Overview
This project builds a Convolutional Neural Network (CNN) using PyTorch to detect breast cancer in mammograms from the OMAMA-DB dataset (UMass Boston / Harvard Dataverse). 

## Methodology (Study-Level Multiview Pipeline)
1. **Data Grouping:** Grouped the dataset by Study ID. Each study contains 4 views (Left/Right, CC/MLO). A study is labeled positive (1.0) if *any* of its 4 views contain an `IndexCancer`.
2. **Early Fusion Architecture:** Stacked the 4 views into a 4-channel tensor. Modified a pre-trained ResNet18 to accept 4 channels in its first convolutional layer (`nn.Conv2d(4, 64, ...)`).
3. **Outlier Mitigation (ROI Cropping):** For positive views, used the `coords` metadata to crop to the Region of Interest (ROI), preventing background artifacts from confusing the model.
4. **Normalization:** Applied Z-score normalization to standardize image brightness across all views.
5. **Hardware:** Trained locally using Apple Silicon (MPS backend).

## Results
- **Baseline Model (Single-View, Full Images):** 66.60% Validation Accuracy
- **Optimized Model (Single-View, ROI Cropped):** 98.58% Validation Accuracy
- **Final Model (Study-Level Multiview, 4-Channel Fusion):** **99.40% Validation Accuracy** 

## How to Run
1. Install dependencies: `pip install torch torchvision numpy matplotlib`
2. Download the `256x256_studylevel` dataset from OMAMA-DB.
3. Update the `data_dir` path in `train.py`.
4. Run `python train.py`
