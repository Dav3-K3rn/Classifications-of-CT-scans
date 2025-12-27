# Classifications-of-CT-scans
# Medical CT Scan Classification using Deep Learning

## 📋 Project Overview
This project implements a complete deep learning pipeline for multi-class classification of medical CT scans, specifically focused on lung cancer detection. The system uses Convolutional Neural Networks (CNNs) with transfer learning and comprehensive model interpretability techniques.

## 🎯 Project Objectives
- Develop an accurate deep learning model for CT scan classification
- Implement comprehensive model evaluation and interpretability
- Create a reproducible pipeline from data preparation to deployment-ready analysis
- Provide medical data mining insights and real-world applicability assessment

## 🏗️ Project Structure
├── prepare_lung_dataset.py # Phase 0: Data preparation and preprocessing
├── training_lung_cnn.py # Phases 1-3: Model training and evaluation
├── interpretation.py # Phase 4: Comprehensive analysis and reporting
├── output_dataset/ # Processed dataset (generated)
├── raw_kaggle/ # Raw Kaggle dataset (generated)
├── extracted/ # Extracted files (generated)
└── README.md # This file


## 📊 Dataset Information
The project uses the **LIDC-IDRI** dataset from Kaggle, containing:
- Lung CT scan images in DICOM format
- Binary classification: Cancer vs. Non-Cancer
- Approximately 3,000 balanced images (after processing)
- Automatically labeled using heuristic folder name analysis

## 🔧 Installation Requirements

### Prerequisites
- Python 3.8+
- Kaggle API credentials (for dataset download)

### Install Dependencies
```bash
pip install -r requirements.txt
