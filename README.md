# 🔍 Credit Card Fraud Detection Using CNN

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> A deep learning project using a 1D Convolutional Neural Network (CNN) to detect fraudulent credit card transactions with **100% accuracy and zero missed fraud cases**.

---

## 🚀 Live Demo

👉 [**Try the Streamlit App**](credit-card-fraud-detection-vgpn4q9klempgs4pebwwnf.streamlit.app)  
_(Deploy on [Streamlit Cloud](https://streamlit.io/cloud) for free — see deployment section below)_

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **100%** |
| Precision | **1.000** |
| Recall | **1.000** |
| F1 Score | **1.000** |
| False Negatives | **0** ✅ |
| ROC-AUC | **~1.000** |

> **FN = 0** means no fraudulent transaction was missed — the most critical metric in fraud detection.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── notebook/
│   └── Credit_Card_Fraud_Detection.ipynb   # Full Colab notebook
├── model/
│   └── fraud_cnn_model.h5          # Trained CNN model
├── data/
│   └── .gitkeep                    # Dataset not included (see below)
└── README.md
```

> **Note:** The dataset is not included in this repo due to size.  
> Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

---

## 🧠 Model Architecture

```
Input (30, 1)
    ↓
Conv1D 64  → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv1D 128 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv1D 256 → BatchNorm → Dropout(0.3)
    ↓
Flatten → Dense(128) → Dropout(0.5) → Sigmoid Output
```

**Why CNN on tabular data?**  
The 30 PCA-transformed features are treated as a 1D sequence. Conv1D kernels learn local inter-feature patterns — e.g. combinations of V3+V4+V5 that are characteristic of fraud.

---

## 🗃️ Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 17,917 (after preprocessing)
- **Fraud Rate:** 0.45% — highly imbalanced
- **Features:** V1–V28 (PCA-transformed), Time, Amount
- **Target:** Class (0 = Legitimate, 1 = Fraud)

---

## ⚙️ Pipeline

1. **EDA** — Class distribution, amount histogram, feature correlation
2. **Preprocessing** — Drop nulls, StandardScaler on Time & Amount
3. **Train-Test Split** — 80/20 stratified
4. **SMOTE** — Balance training set from 0.45% → 50:50 (applied on train only)
5. **CNN Training** — Adam optimizer, EarlyStopping on val_loss
6. **Evaluation** — Confusion matrix, ROC-AUC, classification report

---

## 🛠️ Installation & Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model
# Place fraud_cnn_model.h5 in the root directory

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done! 🎉

> Upload your `fraud_cnn_model.h5` to the repo or use Streamlit secrets for larger files.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| TensorFlow / Keras | CNN model |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing & metrics |
| imbalanced-learn | SMOTE |
| Matplotlib / Seaborn | Visualization |
| Streamlit | Web application |
| Google Colab | Training environment |

---

## 👩‍💻 Author

**Mancy** — Roll No. 2023001123  
B.Tech CSE · GITAM School of Technology, Hyderabad  
Guide: Dr. B Krishnaveni  
Teammates: Praniti & Naisha

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
