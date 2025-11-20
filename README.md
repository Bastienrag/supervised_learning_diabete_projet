# Predicting Hospital Readmission Risk for Diabetic Patients  
**Contributor : Bastien Ragueneau**

---

##  1. Business Challenge

Hospital readmissions within 30 days are costly and often preventable.  
Diabetic patients represent a high-risk group due to complex comorbidities and difficulties in glycemic control.

**Objective:**  
Build a Machine Learning model capable of predicting whether a diabetic patient will be **readmitted within 30 days** after hospital discharge.

**Why this matters:**  
- Helps identify high-risk patients early  
- Enables targeted follow-up interventions  
- Reduces avoidable readmission costs  
- Improves patient care and clinical outcomes

This repository provides a complete, reproducible ML pipeline addressing this challenge.

---

##  2. Dataset Description

Dataset: **Diabetes 130-US Hospitals**  
Source: UCI Machine Learning Repository

**Initial characteristics:**
- 100,000+ hospital encounters  
- 50+ features  
- Demographics, ICD-9 diagnoses, medications, A1C results, prior hospital utilizations  
- Target variable (`readmitted`): `NO`, `<30`, `>30`

The raw dataset contains:  
- Missing values encoded as `"?"`  
- High-cardinality ICD-9 codes  
- Inconsistent or useless administrative features  
- Mixed categorical/numerical variables  
- Medication columns with very low variance

Extensive cleaning and transformation were required.

---

## 🔧 3. Data Cleaning & Feature Engineering

### ✔ Cleaning
- Removal of columns with extreme missingness:  
  `weight`, `payer_code`, `medical_specialty`
- Removal of ~2% of rows with `race = "?"`  
- Cleanup of all `"?"` entries across the dataset  
- Removal of medication features with near-constant distribution  
- Conversion of all remaining features to numeric types

### ✔ Feature Engineering
- **ICD-9 diagnosis grouping** (`diag_1`, `diag_2`, `diag_3`) → medically meaningful categories  
- `diag_1` encoded via One-Hot Encoding  
- `diag_2`, `diag_3` converted to binary comorbidity flags  
- Clinical encoding of medications (`insulin`, `metformin`, etc.)  
- Transformation of A1C results into an ordinal feature (`A1Cresult_cat`)  
- Encoding of demographic features (`gender`, `race`)  
- Creation of binary target:
readmitted_flag = 1 if readmitted == "<30"
readmitted_flag = 0 otherwise

### ✔ Final dataset
- **269,346 rows**  
- **≈ 60 fully numerical features**  
- **0 missing values**  
- **ML-ready tabular dataset**

---

## ⚙️ 4. Reproducibility Instructions

### 🐍 Python version
Use **Python 3.9+**

---

### 📂 4.1 Dataset placement

Place the cleaned dataset here:
project/
│── data/
└── diabetes_clean.csv

---

### 📦 4.2 Install dependencies
pip install -r requirements.txt

---

### 🚀 4.3 Run the full training pipeline
python main.py

This script will:
- Load the dataset  
- Split into train/test  
- Train the **optimized Random Forest model**  
- Evaluate performance  
- Save the model under:  
  `models/random_forest_best.joblib`

---

## 🧪 5. Baseline Model – Logistic Regression

The baseline provides a simple reference point to measure improvements.

### How it works  
Logistic Regression is a **linear probabilistic model** widely used in healthcare.  
Each feature contributes proportionally to the log-odds of readmission.

### Why it’s useful  
- Easy to train, interpret, and benchmark  
- Establishes a **performance baseline**  
- Shows whether raw features already contain predictive signal

Baseline results:  
- **ROC-AUC ~0.78**, **Accuracy ~0.73**  
(These scores serve only as reference, no tuning applied.)

---

## 🔬 6. Experiment Tracking

A structured iterative approach was followed to improve model performance.

| Iteration | Modification | Model | ROC-AUC | Comment |
|----------|--------------|--------|---------|---------|
| 1 | Baseline with raw cleaned features | Logistic Regression | ~0.78 | Starting point |
| 2 | Non-linear modeling | Decision Tree | ~0.83 | Captures interactions |
| 3 | Ensemble method (default params) | Random Forest | ~0.94 | Strong improvement |
| 4 | Full feature engineering (ICD9 grouping, encoding, cleaning) | Random Forest | ~0.97 | Major gain due to high-quality data prep |
| 5 | Hyperparameter tuning (GridSearchCV, CV=5) | **Random Forest (final)** | **0.9903** | Best model |

### 🎯 Final Model Used  
**Optimized Random Forest**

Hyperparameters:
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

### Why such high performance?
- Clean, fully numerical tabular dataset  
- Strong predictive features (diagnoses, hospital history, A1C…)  
- Large sample size (269k rows)  
- Random Forest excels on tabular data  
- Cross-validated hyperparameter optimization  
- No leakage and consistent preprocessing

---

## 📁 8. Project Structure

project/
├── README.md
├── requirements.txt
├── eda.ipynb
├── diabetes_modeling.ipynb
├── main.py
├── app.py  # Streamlit app
├── models/
│   └── random_forest_best.joblib
└── data/
    └── diabetes_clean.csv


---

##  Contact  
**Bastien Ragueneau**  
Bachelor Business & Data – Albert School  
bragueneau@albertschool.com
  









