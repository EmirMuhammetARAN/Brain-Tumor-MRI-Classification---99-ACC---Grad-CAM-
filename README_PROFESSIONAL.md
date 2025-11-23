# Brain Tumor Classification - Professional Medical AI System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Live Demo
**Try it now:** [Hugging Face Spaces](https://huggingface.co/spaces/emiraran/brain-tumor-classification)

## 📋 Project Overview

Professional-grade deep learning system for automated brain tumor classification from MRI scans. This is a **production-ready** implementation suitable for clinical research environments, featuring comprehensive validation, error analysis, and deployment infrastructure.

### Key Features

✅ **Clinical-Grade Validation**
- Sensitivity, Specificity, PPV, NPV with 95% confidence intervals
- ROC-AUC curves and Precision-Recall analysis
- Per-class performance metrics

✅ **Explainable AI**
- Grad-CAM visualizations showing model attention
- Uncertainty quantification via Monte Carlo Dropout
- Confidence thresholding for high-risk cases

✅ **Production-Ready Infrastructure**
- RESTful API with FastAPI
- Docker containerization
- Comprehensive testing suite (90%+ coverage)
- MLflow/W&B integration for experiment tracking

✅ **Clinical Safety**
- Detailed error analysis and failure case detection
- Model limitations documentation
- Regulatory compliance considerations

## 📊 Model Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Accuracy** | 98.44% | [97.89%, 98.99%] |
| **Macro AUC** | 0.994 | - |

### Per-Class Metrics

| Class | Sensitivity | Specificity | PPV | NPV | F1-Score |
|-------|------------|-------------|-----|-----|----------|
| **Glioma** | 0.968 | 0.994 | 0.986 | 0.989 | 0.977 |
| **Meningioma** | 0.967 | 0.996 | 0.989 | 0.987 | 0.978 |
| **No Tumor** | 0.998 | 0.993 | 0.995 | 0.998 | 0.997 |
| **Pituitary** | 0.997 | 0.998 | 0.994 | 0.999 | 0.995 |

### Performance Benchmarks

- **Inference Time**: <200ms (median), <250ms (P95)
- **Throughput**: 5-8 inferences/second (single GPU)
- **Memory**: ~800MB model + 2GB runtime

## 🏗️ Architecture

```
Input (224x224x3 MRI)
    ↓
EfficientNetB3 (pretrained)
    ↓
Global Max Pooling
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(4, softmax)
    ↓
Output: [Glioma, Meningioma, No Tumor, Pituitary]
```

## 📁 Project Structure

```
├── model.ipynb                    # Training notebook with full pipeline
├── best_weights_balanced.h5       # Trained model weights
├── config.json                    # Configuration management
│
├── inference.py                   # Production inference engine
├── api.py                        # FastAPI REST API
├── app.py                        # Gradio demo interface
│
├── clinical_validation.py        # Clinical metrics & validation
├── error_analysis.py             # Failure case detection
├── benchmark.py                  # Performance benchmarking
├── mlops.py                      # Experiment tracking
│
├── test_model.py                 # Comprehensive test suite
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-service orchestration
│
├── requirements.txt              # Basic dependencies
├── requirements-prod.txt         # Production dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# API available at: http://localhost:8000
# MLflow UI at: http://localhost:5000
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-prod.txt

# Run API server
python api.py

# Or run Gradio demo
python app.py
```

## 🔬 Clinical Validation

Run comprehensive clinical validation:

```python
from clinical_validation import run_clinical_validation
from inference import ModelInference

# Load model
model_inference = ModelInference()

# Run validation
metrics = run_clinical_validation(
    model=model_inference.model,
    test_dataset=test_dataset,
    class_names=["glioma", "meningioma", "notumor", "pituitary"],
    output_dir="validation_results"
)
```

**Outputs:**
- `roc_curves.png` - ROC curves for all classes
- `pr_curves.png` - Precision-Recall curves
- `confidence_analysis.png` - Prediction confidence distribution
- `clinical_report.txt` - Detailed metrics with confidence intervals

## 🔍 Error Analysis

Identify and visualize failure cases:

```python
from error_analysis import run_error_analysis

failures = run_error_analysis(
    model=model_inference.model,
    test_dataset=test_dataset,
    class_names=class_names,
    output_dir="error_analysis"
)

# Outputs:
# - confusion_analysis.png
# - misclassification_patterns.png
# - error_analysis_report.txt
# - failure_cases/ (directory with visualizations)
```

## 📈 Performance Benchmarking

```python
from benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark(model_inference.model)
report = benchmark.generate_benchmark_report()

# Outputs:
# - benchmark_report.json
# - Inference time statistics
# - Memory usage analysis
# - Batch processing performance
```

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@mri_scan.jpg" \
  -F "return_gradcam=true"
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@scan1.jpg" \
  -F "files=@scan2.jpg"
```

### Python Client
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("mri_scan.jpg", "rb")}
params = {"return_gradcam": True}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Prediction: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
print(f"Recommendation: {result['clinical_recommendation']}")
```

## 🧪 Testing

Run comprehensive test suite:

```bash
# All tests
pytest test_model.py -v

# With coverage
pytest test_model.py --cov=. --cov-report=html

# Integration tests only
pytest test_model.py -m integration
```

## 📊 MLOps Integration

### MLflow Tracking

```python
from mlops import MLOpsManager, setup_reproducibility

# Setup reproducibility
setup_reproducibility(seed=42)

# Initialize tracking
mlops = MLOpsManager()
run_id = mlops.start_run("training_v2")

# Log config and params
mlops.log_training_config()

# During training
callback = TrainingCallback(mlops)
model.fit(train_data, callbacks=[callback], ...)

# Log clinical metrics
mlops.log_clinical_metrics(validation_metrics)

# Register model
mlops.log_model(model)
mlops.end_run()
```

### Weights & Biases

```python
from mlops import WandbLogger

logger = WandbLogger(project_name="brain-tumor-classification", config=config)
logger.log_metrics({"accuracy": 0.98, "loss": 0.05})
logger.finish()
```

## ⚠️ Model Limitations

**Critical for Clinical Use:**

1. **Dataset Limitations**: Trained on specific MRI protocols (T1/T2). Performance may degrade with:
   - Different MRI equipment or acquisition parameters
   - Non-standard imaging protocols
   - Different patient populations

2. **Not Validated For**:
   - Pediatric patients (<18 years)
   - Emergency triage decisions
   - Rare tumor subtypes
   - Imaging artifacts or low-quality scans

3. **Requires**:
   - Radiologist confirmation of all predictions
   - Institution-specific validation before deployment
   - Quality control for input images

4. **High-Risk Failure Cases**: 
   - See `error_analysis_report.txt` for detailed failure analysis
   - Model can be overconfident in some misclassifications

## 🏥 Clinical Deployment Guidelines

### Recommended Workflow

```
1. MRI Scan Acquired
    ↓
2. Quality Check (human)
    ↓
3. AI Model Inference
    ↓
4. IF confidence < 80% → Flag for review
    ↓
5. Radiologist Review (MANDATORY)
    ↓
6. Final Diagnosis
```

### Regulatory Considerations

- **Current Status**: Research Use Only (RUO)
- **For FDA 510(k) or CE Mark**: Additional validation required
  - Multi-site clinical trials
  - Prospective validation study
  - Risk management documentation (ISO 14971)
  - Software as Medical Device (SaMD) documentation

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{brain_tumor_classifier_2025,
  author = {Emir Aran},
  title = {Professional Brain Tumor Classification System},
  year = {2025},
  url = {https://github.com/emiraran/BrainTumor-Classification-CNN}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**IMPORTANT**: This software is for research purposes only. Not approved for clinical diagnosis.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📞 Contact

- **Author**: Emir Aran
- **LinkedIn**: [Your LinkedIn]
- **Email**: [Your Email]
- **Project**: [GitHub Repository]

## 🔗 Resources

- [Model on Hugging Face](https://huggingface.co/spaces/emiraran/brain-tumor-classification)
- [Dataset](dataset_link.txt)
- [Research Paper](link-to-paper) (if applicable)

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- Brain tumor dataset contributors
- TensorFlow and Keras teams

---

**⚠️ MEDICAL DISCLAIMER**: This software is provided for educational and research purposes only. It is not intended for use in medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.
