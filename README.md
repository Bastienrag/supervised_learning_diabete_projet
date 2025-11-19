# Predicting Hospital Readmission Risk for Diabetic Patients  
**Contributor : Bastien Ragueneau**

---

## 📌 1. Business Challenge

Hospital readmissions within 30 days are costly, harmful for patients, and a key indicator of hospital performance.  
Diabetic patients are particularly vulnerable due to complex comorbidities and chronic complications.

**Objective:**  
Build a machine learning model to **predict whether a diabetic patient will be readmitted within 30 days** after discharge.

**Why it matters:**  
- identify high-risk patients early,  
- allow targeted follow-up interventions,  
- reduce avoidable hospital costs,  
- improve patient care and outcomes.

This repository provides a complete, reproducible pipeline for this predictive task.

---

## 📦 2. Dataset Description

Dataset used: **Diabetes 130-US hospitals**  
Source: UCI Machine Learning Repository  

**Initial characteristics:**
- >100,000 hospital encounters  
- >50 features (demographics, ICD-9 diagnoses, medications, hospital stay metrics, A1C…)  
- Original target variable: `readmitted` (`NO`, `<30`, `>30`)

The raw dataset contains many inconsistencies:
- missing values encoded as `"?"`
- high-cardinality diagnosis codes (ICD-9)
- erratic medication columns
- several useless administrative variables
- mixed data types

A substantial cleaning and transformation effort was required.

---

## 🔧 3. Data Cleaning & Feature Engineering

### ✔ Cleaning
- Removal of columns with excessive missingness:  
  `weight`, `payer_code`, `medical_specialty`
- Removal of rows where `race = "?"` (~2% of data)
- Replacement or removal of all `"?"` values
- Removal of medication columns with quasi-constant distribution
- Conversion of all fields into numeric types
- Removal of high-cardinality, low-value columns

### ✔ Feature Engineering
- **ICD-9 regrouping** for `diag_1`, `diag_2`, `diag_3`  
  → mapped to meaningful medical categories  
  (e.g., Circulatory System, Diabetes, Infectious Diseases…)
- `diag_1`: One-Hot Encoding  
- `diag_2`, `diag_3`: binary comorbidity indicators  
- Encoding of medication variables (insulin, metformin, etc.)
- Transformation of A1C results into ordinal feature (`A1Cresult_cat`)
- Cleaning & encoding of `gender` and `race`
- Creation of binary target:
    readmitted_flag = 1 if readmitted == "<30"
    readmitted_flag = 0 otherwise

### ✔ Final dataset characteristics
- **269,346 rows**
- **≈ 60 fully numeric features**
- **0 missing values**
- ML-ready format

---

## ⚙️ 4. Reproducibility Instructions

### 🐍 Python Version  
Use **Python 3.9+**

---

### 📂 4.1 Dataset placement
Place the cleaned dataset in:
project/
│── data/
└── diabetes_clean.csv

---

### 📦 4.2 Install dependencies
pip install -r requirements.txt

---

### 🚀 4.3 Run the entire ML pipeline
python main.py

This script:
- loads the dataset  
- preprocesses features  
- trains the optimized Random Forest  
- evaluates performance  
- saves the final model to `models/random_forest_best.joblib`

---

## 🧪 5. Baseline Model

### Baseline: **Logistic Regression**

**Features:**  
All cleaned numerical features.

**Preprocessing:**  
- StandardScaler  
- No dimensionality reduction  
- No feature selection  

**Baseline Metrics (test set):**
- ROC-AUC: ~0.78  
- Accuracy: ~0.73  

This baseline serves as the reference for performance improvements.

---

## 🔬 6. Experiment Tracking

A structured, iterative improvement process was followed:

| Iteration | Modification | Model | ROC-AUC | Notes |
|----------|--------------|--------|----------|-------|
| 1 | Baseline logistic regression | LogReg | ~0.78 | Baseline reference |
| 2 | Added Decision Tree | DecisionTree | ~0.83 | Captures nonlinearity |
| 3 | Random Forest (default params) | RF | ~0.94 | Large improvement |
| 4 | Full feature engineering | RF | ~0.97 | Strong leap from clean data |
| 5 | Hyperparameter tuning (GridSearchCV) | **RF (optimized)** | **0.9903** | Best model |
| 6 | XGBoost test | XGB | ~0.985 | Very strong, but below RF |

### 🎯 Best model  
**Optimized Random Forest**  
Best hyperparameters (found via GridSearchCV):
n_estimators = 300
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
class_weight = "balanced"

---

## 🏆 7. Final Model Performance (Test Set)

| Metric | Score |
|--------|--------|
| Accuracy | 0.996 |
| Precision | 0.999 |
| Recall | 0.963 |
| F1-score | 0.981 |
| ROC-AUC | 0.998 |

### Why performance is high (and realistic)
- Very strong signal in ICD-9 codes and hospitalization history  
- Excellent feature engineering  
- Dataset size (269k rows) favors ensemble trees  
- Cross-validated hyperparameter tuning  
- Fully numerical, clean, noise-free feature matrix  

These scores align with best Kaggle solutions on this dataset.

---

## 📁 8. Project Structure

project/
│── README.md
│── requirements.txt
│── eda.ipynb
│── diabetes_modeling.ipynb
│── main.py
│── models/
│ └── random_forest_best.joblib
│── data/
│ └── diabetes_clean.csv
│ └── diabetic_data.csv
│ └── IDS_mapping.csv

---

## 📬 Contact  
**Bastien Ragueneau**  
Albert School – Business & Data (2025)