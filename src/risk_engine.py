import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HOME_MAP = {"RENT": 0, "OTHER": 1, "MORTGAGE": 2, "OWN": 3}
INTENT_MAP = {
    "DEBTCONSOLIDATION": 0, "MEDICAL": 1, "PERSONAL": 2,
    "HOMEIMPROVEMENT": 3, "VENTURE": 4, "EDUCATION": 5
}
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
DEFAULT_MAP = {"N": 0, "Y": 1}

FEATURE_ORDER = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

FEATURE_LABELS = {
    "person_age": "Applicant Age",
    "person_income": "Annual Income ($)",
    "person_home_ownership": "Home Ownership Status",
    "person_emp_length": "Employment Tenure (Years)",
    "loan_intent": "Loan Purpose / Intent",
    "loan_grade": "Assigned Credit Grade (A-G)",
    "loan_amnt": "Requested Loan Amount ($)",
    "loan_int_rate": "Interest Rate (%)",
    "loan_percent_income": "Loan-to-Income (LTI) Ratio",
    "cb_person_default_on_file": "Prior Bureau Default on File",
    "cb_person_cred_hist_length": "Credit History Length (Years)"
}


class CreditRiskEngine:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "xgb_credit_model.pkl")
        explainer_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
        self.model = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path)

    def _encode_inputs(self, raw_dict: dict) -> pd.DataFrame:
        """Converts human-readable inputs into the exact encoded format expected by XGBoost."""
        encoded = raw_dict.copy()
        if isinstance(encoded["person_home_ownership"], str):
            encoded["person_home_ownership"] = HOME_MAP[encoded["person_home_ownership"]]
        if isinstance(encoded["loan_intent"], str):
            encoded["loan_intent"] = INTENT_MAP[encoded["loan_intent"]]
        if isinstance(encoded["loan_grade"], str):
            encoded["loan_grade"] = GRADE_MAP[encoded["loan_grade"]]
        if isinstance(encoded["cb_person_default_on_file"], str):
            encoded["cb_person_default_on_file"] = DEFAULT_MAP[encoded["cb_person_default_on_file"]]

        df = pd.DataFrame([encoded])[FEATURE_ORDER]
        return df

    @staticmethod
    def assign_risk_tier(pd_score: float):
        if pd_score < 0.20:
            return "Low Risk", "Auto-Approve (Standard Rate)", "#22c55e"
        elif pd_score < 0.50:
            return "Medium Risk", "Manual Underwriter Review", "#eab308"
        else:
            return "High Risk", "Decline / Require Collateral", "#ef4444"

    def evaluate_applicant(self, raw_applicant: dict) -> dict:
        df_encoded = self._encode_inputs(raw_applicant)
        pd_score = float(self.model.predict_proba(df_encoded)[0, 1])
        tier, action, color = self.assign_risk_tier(pd_score)

        # Calculate SHAP values for local explainability
        shap_vals = self.explainer.shap_values(df_encoded)
        if isinstance(shap_vals, list):
            shap_row = shap_vals[1][0]
        elif len( np.shape(shap_vals) ) == 2:
            shap_row = shap_vals[0]
        else:
            shap_row = shap_vals

        contributions = pd.DataFrame({
            "feature": FEATURE_ORDER,
            "label": [FEATURE_LABELS[col] for col in FEATURE_ORDER],
            "raw_value": [raw_applicant[col] for col in FEATURE_ORDER],
            "shap_impact": shap_row
        }).sort_values(by="shap_impact", key=abs, ascending=False)

        # Generate top 4 plain-English underwriting explanations
        reasons = []
        for _, row in contributions.head(4).iterrows():
            direction = "increased" if row["shap_impact"] > 0 else "reduced"
            icon = "🔺" if row["shap_impact"] > 0 else "🟢"
            reasons.append(
                f"{icon} **{row['label']}** (`{row['raw_value']}`) {direction} default risk "
                f"*(SHAP impact: {row['shap_impact']:+.2f})*"
            )

        return {
            "pd_score": round(pd_score, 4),
            "risk_tier": tier,
            "recommended_action": action,
            "tier_color": color,
            "top_reasons": reasons,
            "contributions_df": contributions
        }

    def evaluate_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Scores an entire uploaded CSV of loan applications at once."""
        df_work = df_batch[FEATURE_ORDER].copy()
        probs = self.model.predict_proba(df_work)[:, 1]
        results = df_batch.copy()
        results["pd_score"] = np.round(probs, 4)
        results["risk_tier"] = [self.assign_risk_tier(p)[0] for p in probs]
        results["recommended_action"] = [self.assign_risk_tier(p)[1] for p in probs]
        return results


# Quick self-test when you run risk_engine.py directly
if __name__ == "__main__":
    engine = CreditRiskEngine()
    sample_applicant = {
        "person_age": 24,
        "person_income": 32000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "D",
        "loan_amnt": 15000,
        "loan_int_rate": 15.2,
        "loan_percent_income": round(15000 / 32000, 2),
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
    res = engine.evaluate_applicant(sample_applicant)
    print(f"Predicted PD: {res['pd_score'] * 100:.2f}% | Tier: {res['risk_tier']}")
    print("Top Underwriting Reasons:")
    for r in res["top_reasons"]:
        print(" ", r)