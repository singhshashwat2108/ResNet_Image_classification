# Image Classification using CNN and ResNet (CIFAR-10)

A comparative study of baseline CNN, advanced CNN, and ResNet architectures, demonstrating how residual connections improve deep learning performance on image classification tasks.

## 📌 Overview
This project explores different deep learning architectures for image classification using the CIFAR-10 dataset. The objective is to analyze how architectural improvements such as batch normalization, dropout, and residual connections impact model performance.

The project progresses from a simple baseline CNN to a more advanced CNN and finally to a ResNet-inspired architecture.

## 📊 Dataset
- CIFAR-10 benchmark dataset
- 60,000 images (32x32 RGB)
- 10 classes:
  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

- Balanced dataset (6,000 images per class)

  ## 🧠 Models Implemented

### 1. Baseline CNN
- Simple convolution + pooling architecture
- Used as performance baseline

### 2. Advanced CNN
- Added Batch Normalization and Dropout
- Improved generalization and reduced overfitting

### 3. ResNet (Custom Implementation)
- Implemented residual blocks with skip connections
- Used projection shortcuts (1x1 convolution) for dimension matching
- Improved gradient flow and training stability

  ## 🔬 Methodology
- Data normalization (pixel scaling)
- Data augmentation:
  - Random rotation
  - Horizontal flipping
  - Width/height shifting
- Optimizer: Adam
- Learning rate scheduling using ReduceLROnPlateau
- Early stopping to prevent overfitting

## 📈 Results

## 📊 Visualizations

## 🔍 Key Insights
- Residual connections significantly improve training stability
- Advanced CNN reduces overfitting compared to baseline
- Data augmentation improves generalization
- Most misclassifications occur between visually similar classes (e.g., cat vs dog)

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn


## 🚀 How to Run

```bash
pip install -r requirements.txt

python data_analysis.py
python baseline_cnn.py
python advanced_cnn.py
python resnet_model.py

## 🚀 How to Run

```bash
pip install -r requirements.txt

python data_analysis.py
python baseline_cnn.py
python advanced_cnn.py
python resnet_model.py
