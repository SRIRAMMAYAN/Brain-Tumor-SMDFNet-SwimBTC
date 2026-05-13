# MRI-CT Tumor Classification System

A Gradio-based web app for brain tumor classification using MRI and CT images with fusion modeling.

## Features

- **Multi-Modal Prediction**: Classify tumors using MRI-only, CT-only, and MRI+CT fusion models
- **Visual Analysis**: Heatmaps, fused images, and probability distributions
- **Confidence Comparison**: Bar chart showing model confidence levels
- **Validation Metrics**: Accuracy, sensitivity, specificity, F1-score, and AUC

## Requirements

- Python 3.11+
- PyTorch
- Gradio
- NumPy, OpenCV, Matplotlib
- PyDICOM, Torchvision, Timm

## Installation

1. Install dependencies:
   ```bash
   pip install torch torchvision timm gradio pydicom numpy opencv-python matplotlib
   ```

2. (Optional) Train models using `maya/train_classifier.py` and `maya/fusion_train.py`

## Usage

Run the app:
```bash
cd maya
python app.py
```

Upload DICOM files for MRI and CT, then click "Analyze" to get predictions and visualizations.

## Models

- **MRI Model**: Hybrid Swin Transformer + EfficientNet
- **CT Model**: Same architecture as MRI model
- **Fusion Model**: CNN-based fusion of MRI and CT features

Note: App uses randomly initialized models if trained weights are unavailable.