"""
main.py
Script d'entraînement et d'évaluation du meilleur modèle Random Forest
pour prédire la réadmission (<30 jours) de patients diabétiques.

Usage :
    python main.py
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

DATA_PATH = "data/diabetes_clean.csv"    # adapter si besoin
MODEL_PATH = "models/random_forest_best.joblib"
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────
# Chargement des données
# ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    df = pd.read_csv(path)
    return df


# ──────────────────────────────────────────────────────────────
# Entraînement du modèle
# ──────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    # Séparation features / cible
    y = df["readmitted_flag"]
    X = df.drop(columns=["readmitted_flag"])

    # Train/test split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ─────────────────────────────────────
    # Modèle : Random Forest optimisé (GridSearchCV)
    # Hyperparamètres trouvés :
    # max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300
    # ─────────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Pipeline (scaler non obligatoire mais homogène)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", rf_clf)
    ])

    # Entraînement
    pipe.fit(X_train, y_train)

    # ─────────────────────────────────────
    # Évaluation sur le test set
    # ─────────────────────────────────────
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Evaluation du Random Forest optimisé ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    return pipe


# ──────────────────────────────────────────────────────────────
# Sauvegarde du modèle
# ──────────────────────────────────────────────────────────────

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n Modèle sauvegardé sous : {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print(" Chargement du dataset...")
    df = load_data(DATA_PATH)

    print(" Entraînement du modèle Random Forest optimisé...")
    model = train_model(df)

    print(" Sauvegarde du modèle...")
    save_model(model, MODEL_PATH)

    print("\n Pipeline terminée avec succès !")


if __name__ == "__main__":
    main()
