# NeuroBench

## Overview
NeuroBench is an EEG motor imagery classification framework built using the BCI Competition IV Dataset 2a. The project aims to develop and compare classical machine learning and deep learning approaches for EEG decoding, with a long-term focus on cross-subject generalization.

## Dataset
- BCI Competition IV Dataset 2a
- 9 subjects
- 22 EEG channels
- 3 EOG channels
- 4 motor imagery classes

## Pipeline Implemented

Raw EEG (.gdf)
↓
Event Extraction
↓
Epoch Creation
↓
EEG Channel Selection
↓
Bandpass Filtering (8–30 Hz)
↓
NumPy Conversion
↓
CSP Feature Extraction
↓
SVM Classification

## Current Results

- Dataset Shape: (288, 22, 1001)
- Baseline Model: CSP + SVM
- Accuracy: 70.69%

## Technologies Used

- Python
- MNE
- NumPy
- Scikit-learn
- Matplotlib

## Future Work

- EEGNet implementation
- Cross-subject evaluation
- Transformer-based models
- Performance comparison