import os
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.risk_engine import CreditRiskEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "audit_log.db")

st.set_page_config(
    page_title="CrediSight AI | Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)


def init_and_log_db(applicant: dict, result: dict):
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            timestamp TEXT,
            person_age INT,
            person_income INT,
            loan_amnt INT,
            loan_grade TEXT,
            pd_score REAL,
            risk_tier TEXT,
            recommended_action TEXT,
            top_driver TEXT
        )
    """)
    c.execute(
        "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            applicant["person_age"],
            applicant["person_income"],
            applicant["loan_amnt"],
            applicant["loan_grade"],
            result["pd_score"],
            result["risk_tier"],
            result["recommended_action"],
            result["top_reasons"][0].replace("**", "").replace("`", "")
        )
    )
    conn.commit()
    conn.close()


def fetch_audit_logs() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)
    conn.close()
    return df


@st.cache_resource
def load_engine():
    return CreditRiskEngine()


engine = load_engine()

st.title("🏦 AI-Powered Credit Risk Intelligence Platform")
st.markdown(
    "**Enterprise Underwriting System** | Powered by **XGBoost (`0.9449 ROC-AUC`)** & **SHAP Explainable AI** on 32,500+ loan records."
)

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Applicant Underwriter",
    "📂 Batch Portfolio CSV Analyzer",
    "🛡️ SQL Compliance Audit Log"
])

with tab1:
    with st.sidebar:
        st.header("📋 Applicant Financial Profile")
        person_age = st.slider("Applicant Age", 18, 75, 28)
        person_income = st.number_input("Annual Income ($)", 10000, 500000, 65000, step=2500)
        person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"])
        person_emp_length = st.slider("Employment Tenure (Years)", 0.0, 40.0, 5.0, 1.0)
        st.divider()
        st.header("💳 Loan & Credit Bureau Details")
        loan_amnt = st.number_input("Requested Loan Amount ($)", 1000, 100000, 12000, step=1000)
        loan_intent = st.selectbox(
            "Loan Purpose",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        loan_grade = st.selectbox("Credit Bureau Grade (A = Best, G = Worst)", ["A", "B", "C", "D", "E", "F", "G"], index=1)
        loan_int_rate = st.slider("Proposed Interest Rate (%)", 5.0, 25.0, 9.5, 0.1)
        cb_person_default_on_file = st.selectbox("Prior Default on Credit File?", ["N", "Y"])
        cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 1, 30, 6)

    loan_percent_income = round(loan_amnt / person_income, 2)
    applicant = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    res = engine.evaluate_applicant(applicant)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Default (PD)", f"{res['pd_score'] * 100:.2f}%")
    c2.metric("Assigned Risk Category", res["risk_tier"])
    c3.metric("Recommended Decision", res["recommended_action"])
    c4.metric("Loan-to-Income (LTI) Ratio", f"{loan_percent_income * 100:.0f}%")

    st.divider()
    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("📊 Real-Time Risk Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["pd_score"] * 100,
            number={"suffix": "%"},
            title={"text": f"<b>{res['risk_tier']}</b> ({res['recommended_action']})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": res["tier_color"]},
                "steps": [
                    {"range": [0, 20], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [20, 50], "color": "rgba(234, 179, 8, 0.2)"},
                    {"range": [50, 100], "color": "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("📝 Why This Decision Was Made (Top Drivers)")
        for r in res["top_reasons"]:
            st.markdown(f"- {r}")

        if st.button("💾 Save Evaluation to SQL Audit Log", type="primary"):
            init_and_log_db(applicant, res)
            st.success("Saved to SQLite Compliance Audit Database!")

    with col_right:
        st.subheader("🔍 SHAP Feature Impact Breakdown (Explainable AI)")
        st.caption("Red bars push the applicant toward Default (Higher Risk). Green bars reduce risk (Safer).")
        contrib = res["contributions_df"].sort_values(by="shap_impact", ascending=True)
        contrib["Direction"] = contrib["shap_impact"].apply(
            lambda x: "Increases Default Risk (+)" if x > 0 else "Reduces Default Risk (-)"
        )
        fig = px.bar(
            contrib,
            x="shap_impact",
            y="label",
            color="Direction",
            orientation="h",
            color_discrete_map={
                "Increases Default Risk (+)": "#ef4444",
                "Reduces Default Risk (-)": "#22c55e"
            },
            hover_data=["raw_value"],
            labels={"shap_impact": "SHAP Impact on Default Risk (Log-Odds)", "label": "Financial Indicator"}
        )
        fig.update_layout(height=480, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📂 Batch Loan Portfolio Risk Scoring")
    st.write("Upload a CSV file of applicants (or test with the auto-generated `data/sample_batch_applicants.csv`).")
    uploaded = st.file_uploader("Upload Applicant CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        scored_df = engine.evaluate_batch(batch_df)
        st.success(f"Successfully scored {len(scored_df)} loan applications!")

        tier_counts = scored_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Category", "Applicant Count"]
        fig_pie = px.pie(
            tier_counts, names="Risk Category", values="Applicant Count",
            color="Risk Category",
            color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#eab308", "High Risk": "#ef4444"},
            title="Portfolio Risk Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(scored_df, use_container_width=True)

        csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Graded Risk Report (CSV)", csv_bytes, "graded_loan_portfolio.csv", "text/csv")

with tab3:
    st.subheader("🛡️ Regulatory Compliance SQL Audit Trail")
    logs_df = fetch_audit_logs()
    if logs_df.empty:
        st.info("No evaluations logged yet. Click 'Save Evaluation to SQL Audit Log' in Tab 1 to record decisions.")
    else:
        st.dataframe(logs_df, use_container_width=True)