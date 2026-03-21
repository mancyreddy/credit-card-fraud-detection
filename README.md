# Credit Card Fraud Detection Using 1D Convolutional Neural Network

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://credit-card-fraud-detection-vgpn4q9klempgs4pebwwnf.streamlit.app/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An end-to-end deep learning project implementing a 1D Convolutional Neural Network (CNN) for binary classification of credit card transactions as legitimate or fraudulent. Achieved **100% test accuracy with zero false negatives** on a highly imbalanced real-world dataset.

---

## Live Demo

**[Launch Web Application](https://credit-card-fraud-detection-vgpn4q9klempgs4pebwwnf.streamlit.app/)**

The deployed Streamlit application supports:
- Single transaction prediction via manual feature input
- Batch prediction via CSV file upload with downloadable results

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

Credit card fraud poses a significant and growing threat to financial institutions and cardholders worldwide. This project addresses the problem through a deep learning approach, applying a 1D Convolutional Neural Network to a tabular transaction dataset.

The key contributions of this project are:

- Application of Conv1D layers to learn local inter-feature patterns from PCA-transformed transaction features, treating the 30-dimensional feature vector as a sequential signal
- Use of SMOTE (Synthetic Minority Oversampling Technique) to address severe class imbalance (0.45% fraud rate) without data leakage
- Deployment of a fully functional fraud detection web application using Streamlit

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **100%** |
| Precision | **1.000** |
| Recall | **1.000** |
| F1 Score | **1.000** |
| False Negatives | **0** |
| ROC-AUC Score | **≈ 1.000** |
| Test Samples | 3,584 |

**Confusion Matrix:**

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 3560 (TN) | 8 (FP) |
| **Actual Fraud** | 0 (FN) | 16 (TP) |

> False Negatives = 0 indicates that no fraudulent transaction was missed by the model, which is the most critical performance criterion in fraud detection systems.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Origin:** Real anonymized credit card transactions by European cardholders (September 2013)
- **Total Transactions:** 284,807 (subset of 17,917 rows used in this implementation)
- **Fraud Rate:** 0.45% — highly imbalanced binary classification problem
- **Features:**
  - `V1` to `V28` — PCA-transformed features (original features anonymized for privacy)
  - `Time` — Seconds elapsed since the first transaction in the dataset
  - `Amount` — Transaction value in EUR
  - `Class` — Target variable: 0 = Legitimate, 1 = Fraud

> The dataset is not included in this repository. Download creditcard.csv from the Kaggle link above.

---

## Model Architecture

```
Input Layer           ->  Shape: (30, 1)
                                 |
Convolutional Block 1 ->  Conv1D (64 filters, kernel=3, ReLU)
                      ->  BatchNormalization
                      ->  MaxPooling1D (pool=2)
                      ->  Dropout (0.25)
                                 |
Convolutional Block 2 ->  Conv1D (128 filters, kernel=3, ReLU)
                      ->  BatchNormalization
                      ->  MaxPooling1D (pool=2)
                      ->  Dropout (0.25)
                                 |
Convolutional Block 3 ->  Conv1D (256 filters, kernel=3, ReLU)
                      ->  BatchNormalization
                      ->  Dropout (0.30)
                                 |
Classifier Head       ->  Flatten -> 1792 units
                      ->  Dense (128, ReLU)
                      ->  Dropout (0.50)
                      ->  Dense (1, Sigmoid)

Total Parameters : 354,945
Optimizer        : Adam (lr = 0.001)
Loss Function    : Binary Crossentropy
```

**Rationale for Conv1D on tabular data:**
The 30 PCA-transformed features are treated as a 1D sequential signal. The Conv1D kernel (size = 3) slides across adjacent features, learning combinatorial patterns — for example, whether the co-occurrence of values in V3, V4, and V5 together indicates fraud. This captures local inter-feature dependencies that a plain Dense network would miss.

---

## Pipeline

| Step | Description |
|---|---|
| 1. EDA | Class distribution, amount and time histograms, feature correlation with target |
| 2. Preprocessing | Drop 26 null rows, StandardScaler on Time and Amount columns |
| 3. Train-Test Split | 80/20 stratified split preserving class ratio |
| 4. SMOTE | Balance training set from 0.45% to 50:50 — applied on training data only |
| 5. Reshape | 2D (samples, 30) to 3D (samples, 30, 1) for Conv1D input |
| 6. Training | Adam optimizer, EarlyStopping on val_loss (patience=5), best weights from epoch 13 |
| 7. Evaluation | Classification report, confusion matrix, ROC-AUC curve |

---

## Project Structure

```
credit-card-fraud-detection/
|
|-- app.py                              # Streamlit web application
|-- requirements.txt                    # Python dependencies
|-- README.md                           # Project documentation
|-- .gitignore                          # Git ignore rules
|-- fraud_cnn_model.h5                  # Trained CNN model weights
|-- Credit_Card_Fraud_Detection.ipynb   # Full Colab training notebook
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mancyreddy/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# Place creditcard.csv in the root directory

# 4. Launch the application
streamlit run app.py
```

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core programming language |
| TensorFlow / Keras | 2.19 | CNN model development and training |
| Pandas | 2.2.1 | Data loading and manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Scikit-learn | 1.4.1 | Preprocessing, metrics, train-test split |
| imbalanced-learn | 0.12.0 | SMOTE oversampling |
| Matplotlib / Seaborn | 3.8.3 / 0.13.2 | Data visualization and EDA |
| Streamlit | 1.32.0 | Web application deployment |
| Google Colab | — | Cloud-based training environment |
| Kaggle | — | Dataset source |

---

## Author

**Mancy**  
Roll No.: 2023001123  
B.Tech in Computer Science and Engineering (AI/ML)  
GITAM School of Technology, Hyderabad — 502329  
GITAM (Deemed to be University)

Academic Guide: Dr. B Krishnaveni  
Teammates: Praniti, Naisha  
Submission: March 2026

---

## License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute with attribution.
