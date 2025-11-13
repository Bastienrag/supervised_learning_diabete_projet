Predicting Hospital Readmission Risk (Diabetic Patients)


Authors:

Bastien Ragueneau
Meimoun Mathis


Business Use Case (BUC)

Hospital readmissions within 30 days are a major concern in healthcare systems, both medically and financially.
The goal of this project is to predict whether a diabetic patient will be readmitted within 30 days after discharge.

Early identification of high-risk patients enables hospitals to:

Implement targeted follow-up care,

Reduce costs associated with readmissions,

And improve patient outcomes through preventive interventions.



Problem Definition

Task: Supervised Learning — Binary Classification

Target variable: readmitted → encoded as:

1 = Patient readmitted within 30 days (<30)

0 = Not readmitted (>30 or NO)

Business Objective: Minimize false negatives (patients predicted as "safe" but actually at high risk), maximizing recall while maintaining acceptable precision.



Dataset Description

Source: UCI Machine Learning Repository — Diabetes 130-US hospitals for years 1999–2008
Link: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

Overview

Rows: ~100,000 hospital encounters

Columns: 50 (after cleaning ≈ 25–30 used)

Main features: demographics, hospital stay data, diagnoses, medications, and lab results

Target: readmitted (<30, >30, NO)

Preprocessing & Feature Engineering

Removal of irrelevant or constant columns (e.g. weight, examide, citoglipton, max_glu_serum, A1Cresult, encounter_id, patient_nbr, etc.)

Conversion of age groups into numerical midpoints for modeling

Mapping of diag_1, diag_2, diag_3 ICD-9 codes into broad medical categories (e.g., Diabetes, Circulatory system, Respiratory system…)

One-Hot Encoding of categorical variables

Standard scaling of numerical features


Methodology
Baseline Model

A first Logistic Regression without hyperparameter tuning was used as a baseline to set a performance reference.


Feature Preprocessing

A unified scikit-learn Pipeline was built:

ColumnTransformer for One-Hot Encoding + Standard Scaling

Integrated into every model (Decision Tree, Random Forest, Logistic Regression)


Models Tested
Model	Technique	CV Strategy	Tuning Method
Logistic Regression	Linear	5-fold CV	GridSearchCV
Decision Tree	Non-linear	5-fold CV	GridSearchCV
Random Forest	Ensemble	5-fold CV	GridSearchCV

Each model was evaluated using ROC-AUC, precision, recall, and F1-score.

Key Results
Model	ROC-AUC (CV)	ROC-AUC (Test)	Recall	Comment
Logistic Regression	~0.73	~0.72	Good recall, interpretable	
Decision Tree	~0.75	~0.74	Easy to explain, moderate overfit	
Random Forest	~0.80	~0.79	Best trade-off between recall & AUC	

Final model: Random Forest (GridSearchCV tuned)
Chosen for its robustness, interpretability, and better recall balance on imbalanced data.

Experiment Tracking

Iteration 1 → Baseline Logistic Regression

Iteration 2 → Decision Tree (tuning depth & leaf params)

Iteration 3 → Random Forest (tuning n_estimators, max_depth)

Iteration 4 → Feature cleaning (diagnoses mapping, age transformation)

Iteration 5 → Model comparison & ROC analysis

Each iteration was validated using cross-validation ROC-AUC and business-oriented recall analysis.


Reproducibility
Environment

Python version: 3.10+

Main dependencies:

pandas
numpy
scikit-learn
matplotlib



🖥️ Folder structure
📂 project_root/
 ┣ 📄 README.md
 ┣ 📄 main.py
 ┣ 📄 requirements.txt
 ┣ 📊 eda.ipynb
 ┣ 📊 diabetes_modeling.ipynb
 ┣ 📁 data/
 ┃ ┣ diabetic_data.csv
 ┃ ┗ IDS_mapping.csv


▶️ Run instructions
# Create environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the training script
python main.py

Metrics to Monitor

ROC-AUC → Global discrimination ability

Recall (Sensitivity) → Key business metric

Precision / F1-score → Prediction stability

Confusion Matrix → Visual performance check

Next Steps

Model interpretability (feature importances)

Test threshold optimization to maximize recall

(Optional) API serving with FastAPI for hospital integration




Credits: 

Project completed as part of the Supervised Learning course at Albert School (2025)