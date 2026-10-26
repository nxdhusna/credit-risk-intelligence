# credit-risk-intelligence
AI-powered credit risk scoring; explainable underwriting platform

# 🏦 AI-Powered Credit Risk Intelligence Platform

An enterprise-grade **Credit Risk Scoring & Explainable AI (XAI) Underwriting System** built with **XGBoost**, **SHAP**, **SQLite**, and **Streamlit**. 

Designed to help financial institutions evaluate loan applications in real time, predict **Probability of Default (PD)**, classify applicants into automated risk tiers, and generate human-readable regulatory explanations for every credit decision.

---

## 🚀 Key Features

* **🎯 Real-Time Applicant Underwriter:** Evaluates 11 financial and credit bureau indicators to compute calibrated Probability of Default ($PD$) and Loan-to-Income ($LTI$) ratios.
* **🚦 Automated 3-Tier Risk Routing:**
  * 🟢 **Low Risk ($PD < 20\%$):** Auto-Approve at standard rates.
  * 🟡 **Medium Risk ($20\% \le PD < 50\%$):** Route to manual underwriter review.
  * 🔴 **High Risk ($PD \ge 50\%$):** Decline or require collateral.
* **🔍 Explainable AI (SHAP Waterfall & Reason Codes):** Decomposes black-box gradient boosted trees using `SHAP TreeExplainer` to display exact log-odds feature attributions and plain-English underwriting rationale.
* **📂 Batch CSV Portfolio Analyzer:** Allows credit managers to upload bulk CSV loan portfolios, visualize risk distribution via interactive Plotly charts, and export graded risk reports.
* **🛡️ SQLite Regulatory Audit Trail:** Logs timestamped underwriting decisions and primary risk drivers into a relational database for compliance auditing.

---

## 📊 Model Performance & Dataset

Trained and validated on **32,573 real-world banking loan records** (*Credit Risk Benchmark Dataset*) with automated outlier filtering and median missing-value imputation:

| Metric | Validation Score |
| :--- | :--- |
| **ROC-AUC Score** | **0.9449** |
| **Overall Accuracy** | **93.28%** |
| **Defaulter Precision (Class 1)** | **0.97** |
| **Macro F1-Score** | **0.89** |

---

## 🛠️ Tech Stack & Architecture

* **Machine Learning:** `XGBoost`, `Scikit-Learn`, `Pandas`, `NumPy`
* **Explainable AI (XAI):** `SHAP (SHapley Additive exPlanations)`
* **Frontend & Visualization:** `Streamlit`, `Plotly`
* **Database / Persistence:** `SQLite3`, `Joblib`

```text
credit-risk-intelligence/
├── data/
│   ├── credit_risk_dataset.csv       # 32,581-row benchmark dataset
│   └── sample_batch_applicants.csv   # 50-row test file for Batch Analyzer
├── models/
│   ├── xgb_credit_model.pkl          # Trained XGBoost Classifier (0.9449 AUC)
│   └── shap_explainer.pkl            # Serialized SHAP TreeExplainer
├── database/
│   └── audit_log.db                  # SQLite compliance audit trail
├── src/
│   ├── train_pipeline.py             # Data cleaning, feature encoding & training
│   └── risk_engine.py                # PD scoring, tiering & SHAP reason generator
├── app.py                            # Interactive Streamlit Underwriter UI
└── requirements.txt


# 1. Clone the repository
git clone [https://github.com/nxdhusna/credit-risk-intelligence.git](https://github.com/nxdhusna/credit-risk-intelligence.git)
cd credit-risk-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model & generate SHAP explainer (optional - pre-trained models included)
python src/train_pipeline.py

# 4. Launch the interactive web dashboard
streamlit run app.py