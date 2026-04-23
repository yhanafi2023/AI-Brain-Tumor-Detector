# AI Brain Tumor Image Classification (ANN vs CNN)

## HIGH LEVEL OVERVIEW

This project is a machine learning-based brain tumor classification system that uses MRI scan images to detect and classify brain tumors into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor
## Goals
The main goal of this project is to assist in early and accurate detection of brain tumors using deep learning techniques. The system compares two different neural network approaches:
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

The CNN model is the primary focus due to its superior performance in image-based classification tasks, while the ANN model is included strictly for performance comparison.

---

## SUCCESS CRITERIA

The models are evaluated based on classification performance metrics:

- Target performance: Accuracy above 80% (CNN achieves this target)
- ANN model serves as a baseline comparison

Final achieved results:
- CNN outperforms ANN in all evaluation metrics

---

## REQUIREMENTS

Before running this project, ensure you have the following installed:

- Python  
  https://www.python.org/downloads/

- Node.js  
  https://nodejs.org/en/download/

---

## DATASET

Dataset used:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Dataset Breakdown:
- Training Images: 5600 total
  - Split equally into 4 classes:
    - Glioma
    - Meningioma
    - Pituitary
    - No tumor

- Testing Images: 1600 total
  - Split equally into the same 4 classes

---

## DATA PREPROCESSING (BOTH MODELS)

All images undergo the following preprocessing steps:
- Resize images to 128 x 128
- Convert images to grayscale
- Normalize pixel values by dividing by 255

---

## ANN MODEL ARCHITECTURE

The Artificial Neural Network is defined as follows:

- Input Layer:
  - Flatten layer
  - Input shape: (128, 128, 3)

- Hidden Layers:
  - Dense layer with 3000 neurons
    - Activation: ReLU
  - Dense layer with 1000 neurons
    - Activation: ReLU

- Output Layer:
  - Dense layer with 4 neurons
    - Activation: Softmax
    - Outputs probability distribution across 4 classes

### ANN Training Configuration:
- Optimizer: Stochastic Gradient Descent (SGD)
- Loss Function: sparse_categorical_crossentropy
- Utilized 5 Epochs

---

## CNN MODEL ARCHITECTURE (PRIMARY MODEL)

The Convolutional Neural Network is defined as follows:

- Input Layer:
  - Conv2D
    - Filters: 32
    - Kernel size: 3 x 3
    - Activation: ReLU
    - Input shape: (128, 128, 3)

- Pooling Layer:
  - MaxPooling2D
    - Pool size: 2 x 2

- Second Convolution Block:
  - Conv2D
    - Filters: 64
    - Kernel size: 3 x 3
    - Activation: ReLU
  - MaxPooling2D
    - Pool size: 2 x 2

- Classification Head:
  - Flatten layer
  - Dense layer
    - 64 neurons
    - Activation: ReLU
  - Output Dense layer
    - 4 neurons
    - Activation: Softmax

### CNN Training Configuration:
- Optimizer: Adam
- Loss Function: sparse_categorical_crossentropy
- Utilzied 10 Epochs

---

## MODEL EVALUATION RESULTS

### ANN Performance:
- Precision: 74%
- Recall: 74%
- F1 Score: 74%
- Validation Accuracy: 74%

### CNN Performance:
- Precision: 85%
- Recall: 85%
- F1 Score: 85%
- Validation Accuracy: 85%

---

## HOW TO RUN THIS PROJECT LOCALLY

### 1. Clone the Repository
```bash
git clone https://github.com/yhanafi2023/AI-Brain-Tumor-Detector.git
cd AI-Brain-Tumor-Detector
```
### 2. Install Required Packages
```bash
pip install -r requirements.txt
```
### 3. Train the Models
```bash
cd backend
python modeltraining.py
```
This will Generate Two Model Files:
  - ann_model.keras
  - cnn_model.keras
### 4. Start Flask Backend Server
```bash
python server.py
```
### 5. Launch Website Locally
```bash
npx serve
```



# **Last Updated:** April 2026
