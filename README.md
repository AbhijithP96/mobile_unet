# MobileNet-UNet for Lane Segmentation

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-green.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Hub-blue.svg)](https://hub.docker.com/)
[![Model Size](https://img.shields.io/badge/Model%20Size-27MB-brightgreen.svg)]()
[![Inference](https://img.shields.io/badge/Inference-78ms-brightgreen.svg)]()
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AbhijithP96/.github/workflows/main.yml?label=CI/CD)](https://github.com/AbhijithP96/mobile_unet/.github/workflows/main.yml)

**A lightweight, efficient MobileNetV2-based U-Net for real-time lane segmentation, optimized for fast, accurate performance on mobile and edge devices.**

## 🎯 Project Overview

This project implements a production-ready lane segmentation model that combines the efficiency of MobileNetV2 with the accuracy of U-Net architecture. The solution achieves **68.36% Dice coefficient** with only **27MB model size** and **78ms inference time**, making it suitable for real-time autonomous driving applications.

### Key Achievements
- **⚡ Fast Inference**: 78ms total pipeline (29ms preprocessing + 49ms neural network)
- **📱 Lightweight**: 27MB model size optimized for deployment
- **🎯 High Accuracy**: 68.36% Dice coefficient, 97.35% binary accuracy
- **🔬 Systematic Research**: 62+ experiments across 10 categories with MLflow tracking
- **🚀 Production Ready**: FastAPI deployment with Docker containerization

## 📊 Performance Metrics

| Metric | Value | 
|--------|-------|
| **Dice Coefficient** | **68.36%** |
| **Binary Accuracy** | **97.35%** |
| **Precision** | **69.11%** |
| **Recall** | **70.84%** | 
| **Model Size** | **27MB** | 
| **Inference Time** | **78ms** |

### Training Results (Best Model - Epoch 159/300)
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Dice Coefficient** | 0.5557 | 0.6641 | **0.6836** |
| **Binary Accuracy** | 0.9644 | 0.9715 | **0.9735** |
| **Loss** | 0.3405 | 0.2550 | **0.2398** |

## 🏗️ Architecture

### MobileNet-UNet Design
```
Input (224×224×3)
    ↓
MobileNetV2 Encoder (ImageNet, Frozen)
├── Feature Maps: [112×112, 56×56, 28×28, 14×14, 7×7]
├── Channels: [96, 144, 192, 576, 1280]
└── Skip Connections: 4 levels
    ↓
Custom Depthwise Decoder
├── Filters: [128, 64, 32, 16] (optimized)
├── Depthwise Separable Convolutions
├── Transposed Convolution Upsampling
└── Skip Connection Integration
    ↓
Output (224×224×1) - Lane Probability Mask
```
### Optimized Configuration
- **Encoder**: MobileNetV2 with ImageNet weights (frozen)
- **Decoder**: Custom depthwise separable convolutions
- **Loss Function**: Combined Focal-Dice Loss (0.3:0.7 ratio)
- **Optimizer**: AdamW with cosine decay scheduler
- **Data Augmentation**: Comprehensive Albumentations pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV - Python package and project manager
- Docker (for containerized deployment)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AbhijithP96/mobile_unet.git
    cd mobile_unet
    ```
2. Install dependencies using UV:
    ```bash
    uv sync --all-extras --dev
    ```
### Dataset

To download the TuSimple lane segmentation dataset from Kaggle, download the ```kaggle.json``` to your machine and then run
```bash
uv run data_manager --update --credentials path/to/kaggle.json --cache path/to/download/dir
uv run data_manager --kaggle
```

### Training

To train the model, run:
```bash
# to run the model using the best configuration
uv run train.py

# to run experiments with different configurations
# set up mlflow server
uv run mlflow server --host 0.0.0.0 --port 8080
# then run experiments using the json file
uv run run_exp.py --json path/to/your_experiments.json
```
Refer to ```data/exp.json``` for sample experiment configurations.

### Testing

To test on the Tu Simple test set, run:
```bash
uv run test.py --model name/of/mlflow/model --version version_number
```

### Inference
To perform inference on a single image, run:
```bash
uv run infer.py path/to/image.jpg path/to/saved_model/mobile_unet.keras
```

## Sample Inference

### Using TuSimple Test Images

| Augment | Input Image | Ground Truth | Prediction |
|---------|-------------|--------------|------------|
| No | ![Input](/data/samples/test01.jpg) | ![GT](/data/samples/test01_gt.jpg) | ![Pred](/data/samples/test01_predict.jpg) |
|Yes (Random Fog)| ![Input](/data/samples/test02_augmented.jpg) | ![GT](/data/samples/test02_gt.jpg) | ![Pred](/data/samples/test02_predict.jpg) |

### Using images from the web
| Input Image | Prediction |
|-------------|------------|
| ![Input](./data/samples/test03.jpg) | ![Pred](/data/samples/test03_predict.jpg) |
| ![Input](./data/samples/test04.jpg) | ![Pred](/data/samples/test04_predict.jpg) |


## 🐳 Docker Deployment

### Pre-built Image
```bash
# Pull from Docker Hub
docker pull basilisk96/mobile_unet:latest

# Run inference server
docker run -p 8000:8000 basilisk96/mobile_unet:latest
```

### Build Locally
```bash
# Build image
docker build -t mobile_unet .

# Run container
docker run -p 8000:8000 mobile_unet
```

### API Usage for Inference

Once the Docker container is running, you can access the FastAPI inference endpoint at ```http://0.0.0.0:8000/segment```, use either of the following methods to perform inference:

1. **Using the provided `predict.py` script:**
```bash
uv run predict.py path/to/image.jpg
``` 

2. **Using the streamlit app:**
```bash
uv run streamlit run app.py
```
Open the provided URL in the terminal to access the web interface.
Upload an image and view the predicted lane mask and overlay.

## 🔬 Systematic Experiments

### 10 Comprehensive Experiment Categories
1. **🔧 Optimizer Comparison** - Adam, AdamW, RMSprop, SGD analysis
2. **📈 Learning Rate Scheduling** - Exponential, cosine, piecewise strategies  
3. **🎯 Loss Function Study** - Dice, Focal, BCE, and combined approaches
4. **⚡ Architecture Optimization** - Filter reduction and efficiency studies
5. **🎨 Data Augmentation** - Comprehensive augmentation pipeline ablation
6. **🤝 Hyperparameter Grid Search** - Systematic parameter optimization
7. **🔄 Scheduler-Loss Combinations** - Advanced interaction studies
8. **📱 Mobile Architecture** - Depthwise vs standard convolution comparison
9. **🎯 Best Model Selection** - Top configuration validation
10. **📊 Performance Analysis** - Detailed metrics and efficiency studies

### Key Findings
- **AdamW + Cosine LR**: Optimal optimizer-scheduler combination
- **Focal-Dice Loss (0.3:0.7)**: Best performance for lane segmentation
- **Filter Reduction**: 75% parameter reduction with <3% accuracy loss
- **Depthwise Convolutions**: 40% size reduction maintaining performance


## 🧪 Unit Tests

To run unit tests, execute:
```bash
uv run pytest
```

These tests ensure the loss functions and model architecture are functioning correctly.

### Code Quality
- **Black**: Code formatting
- **pytest**: Comprehensive testing
- **GitHub Actions**: Automated CI (formatting checks via Black, Unit tests via pytest on push)
- **MLflow**: Experiment reproducibility

## 📈 MLflow Experiments

All the conducted experiments are tracked using MLflow. To view the experiment results, download the mlruns and mlartifacts directories from [here](https://drive.google.com/drive/folders/1cn6fQ2zLlXp6XkuvJRPCItd9Guun5kI8?usp=drive_link) and run:
```bash
mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlartifacts
```
Click the url provided in the terminal to access the MLflow UI.

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

*Production-ready • Lane Segmentation Model*

**🚀 68.36% Dice • 78ms Inference • 27MB Model 🚀**

</div>