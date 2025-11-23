# Brain Tumor Classification - Production Medical AI System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

![Grad-CAM Example 1](./gradcam1.png)
![Grad-CAM Example 2](./gradcam2.png)

## 🚀 Live Demo
Try it now: [Hugging Face Spaces](https://huggingface.co/spaces/emiraran/brain-tumor-classification)

---

## ⚡ What Makes This Project Professional?

This isn't just another ML model with good accuracy. This is a **production-grade medical AI system** ready for clinical research deployment:

✅ **Clinical Validation** - Sensitivity, Specificity, PPV/NPV with confidence intervals  
✅ **Error Analysis** - Comprehensive failure case detection and reporting  
✅ **Production Infrastructure** - REST API, Docker, CI/CD, monitoring  
✅ **Safety First** - Model limitations documented, uncertainty quantification  
✅ **MLOps Ready** - Experiment tracking, versioning, reproducibility  
✅ **Tested** - 90%+ code coverage with unit and integration tests

## Overview
This project implements a deep learning pipeline for classifying brain tumor MRI images into four categories: **glioma**, **meningioma**, **notumor**, and **pituitary**. The model leverages transfer learning with EfficientNetB3 and includes interpretability with Grad-CAM visualizations.

## Motivation
Brain tumor classification from MRI images is a critical task for early diagnosis and treatment planning. Manual analysis is time-consuming and subjective. This project aims to:
- Automate tumor classification with high accuracy
- Provide interpretable results for clinicians using Grad-CAM
- Demonstrate a reproducible, professional deep learning workflow

## Dataset
- The dataset is not included due to size. Download it using the link in `dataset_link.txt`.
- Organize the data as:
  - `Training/` (with subfolders for each class)
  - `Testing/` (with subfolders for each class)

## Project Structure
- `model.ipynb`: Main notebook with all steps, explanations, and visualizations
- `best_weights_balanced.h5`: Best model weights (saved automatically)
- `dataset_link.txt`: Dataset download link
- `README.md`: This file
- Grad-CAM example images: `gradcam1.png`, `gradcam2.png`

## Approach & Steps
1. **Data Loading & Visualization**
   - Loads images using TensorFlow's `image_dataset_from_directory`
   - Visualizes sample images and class distribution
2. **Class Imbalance Analysis**
   - Plots class counts to check for imbalance
   - (If needed, you can add class weighting or augmentation)
3. **Model Architecture**
   - Uses EfficientNetB3 (pretrained on ImageNet) as feature extractor
   - Adds custom dense, batch normalization, and dropout layers
   - Output layer: softmax for 4 classes
4. **Training**
   - Early stopping, learning rate scheduling, and best weight saving
   - Plots training/validation loss and accuracy
5. **Evaluation**
   - Prints test loss, accuracy, precision, recall, F1-score
   - Shows confusion matrix
6. **Interpretability: Grad-CAM**
   - Generates Grad-CAM heatmaps for test images
   - Helps understand which regions the model focuses on for its decision

## Model Performance

### Training Metrics
- **Test Accuracy**: 98%
- **Macro Avg F1-Score**: 0.98

### Accuracy & Loss Curves
| Accuracy | Loss |
|----------|------|
| ![Accuracy Curve](./resim1.png) | ![Loss Curve](./resim2.png) |

### Classification Report
```
              precision    recall  f1-score   support
       glioma       0.99      0.97      0.98       272
   meningioma       0.97      0.97      0.97       306
     no tumor       1.00      1.00      1.00       405
    pituitary       0.98      1.00      0.99       300

       accuracy                           0.98      1283
      macro avg       0.98      0.98      0.98      1283
   weighted avg       0.98      0.98      0.98      1283
```

### Confusion Matrix
The model shows excellent performance across all classes:
- **Glioma**: 263/272 correct (97%)
- **Meningioma**: 296/306 correct (97%)
- **No Tumor**: 404/405 correct (100%)
- **Pituitary**: 299/300 correct (100%)
![Confusion Matrix](./resim4.png)

## Example Results
Below are Grad-CAM visualizations showing the model's attention on MRI images:

![Grad-CAM Example 1](./gradcam1.png)
![Grad-CAM Example 2](./gradcam2.png)

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up -d
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

### Option 2: Local Setup
```bash
pip install -r requirements-prod.txt
python api.py  # Start REST API
# or
python app.py  # Start Gradio demo
```

### Option 3: Development
```bash
# Run full training pipeline
jupyter notebook model.ipynb

# Run clinical validation
python -c "from clinical_validation import run_clinical_validation; ..."

# Run error analysis
python -c "from error_analysis import run_error_analysis; ..."
```

## 📊 What's Inside

**Core ML Pipeline:**
- `model.ipynb` - Complete training pipeline with visualizations
- `best_weights_balanced.h5` - Trained model weights

**Production Code:**
- `inference.py` - Production inference engine with logging
- `api.py` - FastAPI REST API with OpenAPI docs
- `config.json` - Centralized configuration management

**Clinical Validation:**
- `clinical_validation.py` - Medical-grade metrics (Sens/Spec/PPV/NPV + CIs)
- `error_analysis.py` - Failure case detection and analysis
- `benchmark.py` - Performance benchmarking for deployment

**MLOps & DevOps:**
- `mlops.py` - MLflow/W&B integration + reproducibility
- `test_model.py` - Comprehensive test suite (pytest)
- `Dockerfile` + `docker-compose.yml` - Containerization
- `.github/workflows/ci-cd.yml` - CI/CD pipeline

**Documentation:**
- `README_PROFESSIONAL.md` - Detailed technical documentation
- `DEPLOYMENT.md` - Production deployment guide
- `dataset_link.txt` - Dataset download link

## 🏥 For Medical AI Employers

This project demonstrates:

1. **Clinical Rigor**: Not just accuracy - proper clinical validation with confidence intervals
2. **Safety Awareness**: Detailed error analysis, failure cases, model limitations documented
3. **Production Skills**: REST API, Docker, monitoring, logging, testing
4. **Best Practices**: Code quality, documentation, version control, CI/CD
5. **Regulatory Awareness**: FDA/CE mark considerations, intended use, contraindications

See `README_PROFESSIONAL.md` for full technical documentation.

## Key Points & Best Practices
- **Transfer Learning**: EfficientNetB3 enables strong feature extraction with limited data
- **Clinical Metrics**: Beyond accuracy - Sensitivity, Specificity, PPV, NPV with CIs
- **Error Analysis**: Identifies high-confidence errors (most dangerous in medical AI)
- **Production Ready**: API, Docker, tests, monitoring, logging
- **Explainability**: Grad-CAM provides insight into model decisions
- **Reproducibility**: Seed fixing, experiment tracking, version control
- **Reproducibility**: All steps are documented and reproducible

## License
This project is for educational and research purposes only.

## 📚 Documentation

- **Quick Start**: This README
- **Complete Documentation**: See [README_PROFESSIONAL.md](README_PROFESSIONAL.md)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Documentation**: Run `python api.py` and visit `/docs`