import os
import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Active public mirrors of the 32,581-row Kaggle Credit Risk Dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/Bulut-Bilisimciler/machine-learning-tools/main/credit_risk_dataset.csv",
    "https://philchodrow.github.io/ml-notes/assets/data/credit-risk/credit_risk_dataset.csv"
]

HOME_MAP = {"RENT": 0, "OTHER": 1, "MORTGAGE": 2, "OWN": 3}
INTENT_MAP = {
    "DEBTCONSOLIDATION": 0, "MEDICAL": 1, "PERSONAL": 2,
    "HOMEIMPROVEMENT": 3, "VENTURE": 4, "EDUCATION": 5
}
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
DEFAULT_MAP = {"N": 0, "Y": 1}


def load_and_clean_real_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "credit_risk_dataset.csv")

    if not os.path.exists(csv_path):
        print("Downloading real 32,581-row Credit Risk Dataset...")
        df = None
        for url in DATASET_URLS:
            try:
                df = pd.read_csv(url)
                df.to_csv(csv_path, index=False)
                print(f"Successfully downloaded and saved raw dataset to {csv_path}")
                break
            except Exception as e:
                print(f"Mirror failed ({e}), trying backup mirror...")
    else:
        print(f"Loading cached dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

    print(f"Raw dataset shape: {df.shape}")

    # 1. Outlier Removal (Real data cleaning talking point for your CV!)
    df = df[df["person_age"] <= 90]
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)]

    # 2. Missing Value Imputation (Median imputation)
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # 3. Encode categorical banking variables
    df["person_home_ownership"] = df["person_home_ownership"].map(HOME_MAP)
    df["loan_intent"] = df["loan_intent"].map(INTENT_MAP)
    df["loan_grade"] = df["loan_grade"].map(GRADE_MAP)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(DEFAULT_MAP)

    # Drop any remaining unexpected nulls
    df = df.dropna()

    # Save a 50-row clean batch sample for testing our dashboard uploader later
    sample_batch = df.drop(columns=["loan_status"]).head(50)
    sample_batch.to_csv(os.path.join(DATA_DIR, "sample_batch_applicants.csv"), index=False)

    print(f"Cleaned dataset shape: {df.shape}")
    return df


def train_and_export():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_clean_real_data()

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTraining XGBoost Classifier on 32k+ real loan records...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = (preds_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, preds_proba)
    acc = accuracy_score(y_test, preds_binary)

    print("\n✅ Real-World Model Trained Successfully!")
    print(f"Validation ROC-AUC Score : {auc:.4f}")
    print(f"Validation Accuracy      : {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds_binary))

    print("Building SHAP Explainable AI TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_credit_model.pkl"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    print(f"Saved model & SHAP explainer to: {MODELS_DIR}")

if __name__ == "__main__":
    train_and_export()