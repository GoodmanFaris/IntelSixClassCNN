# Intel Image Classification — CNN with TFLite Quantization

## Overview

This project explores image classification on the Intel Image Classification dataset using Convolutional Neural Networks (CNNs). The primary goal is to investigate how **model size reduction through quantization affects performance**, and to evaluate the trade-off between accuracy and model footprint in the context of resource-constrained deployment.

## Dataset

**Source:** [Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

Download and unzip the dataset into a folder named `Images`. The expected structure is:

```
Images/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── ...
└── seg_pred/
    └── seg_pred/
```

- **Train set:** 14,034 images across 6 classes
- **Test set:** 3,000 images across 6 classes
- **Classes:** buildings, forest, glacier, mountain, sea, street

## Project Structure

```
├── README.md
├── requirements.txt
└── classification.ipynb     # Main notebook
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Average brightness per class
- RGB channel analysis per class
- Visualization of augmented images

Key findings from EDA:
- Dataset is well-balanced (2,191–2,512 images per class)
- Glacier has the highest brightness (132.5) due to ice and snow reflectance
- Forest has the lowest brightness (87.8) due to dense vegetation
- Glacier has a dominant B channel (151.6), Forest a dominant G channel (93.2)

### 2. Data Preprocessing
- Image resize to 150×150
- Normalization to [0, 1] range
- Data augmentation on training set only (rotation, flip, zoom, shift)

### 3. Models

#### CNN v1 — Baseline (32→64→128)
Standard CNN with three convolutional blocks, increasing filter depth.

#### CNN v2 — Lightweight (16→32→64)
Hypothesis: since the 6 classes are visually distinct (confirmed by EDA), a smaller architecture may achieve comparable accuracy with fewer parameters.

#### TFLite Quantization
Post-training quantization (float32 → int8) applied to CNN v1 to evaluate model compression for edge deployment.

### 4. Results

| Model | Accuracy | Size |
|---|---|---|
| CNN v1 (32→64→128) | 89% | 109.5 MB |
| CNN v2 (16→32→64) | 87% | ~30 MB |
| CNN v1 Quantized (TFLite) | 87% | 9.1 MB |

### 5. Key Finding

Quantization reduced the model size by **12x** (109.5 MB → 9.1 MB) with only a **2% accuracy drop**. The quantized model achieves the same accuracy as CNN v2 at a fraction of the size — demonstrating that post-training quantization is a highly effective compression technique.

### 6. Error Analysis

Confusion matrix analysis revealed that misclassifications are not random — the model struggles most with visually similar pairs:
- **Mountain ↔ Glacier** — share rocky terrain, snow, and sky
- **Buildings ↔ Street** — share the same urban gray palette

This is consistent with the RGB analysis performed during EDA, where both glacier and mountain showed a dominant B channel.

## Setup

```bash
git clone https://github.com/your-username/intel-image-classification
cd intel-image-classification
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Then download the dataset from Kaggle and unzip into `Images/`.

## Requirements

See `requirements.txt`.
