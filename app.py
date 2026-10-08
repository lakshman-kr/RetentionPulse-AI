import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.model import load_artifacts

st.set_page_config(
    page_title="RetentionPulse AI | Churn & ROI Optimizer",
    page_icon="🛡️",
    layout="wide"
)

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
explainer = artifacts['explainer']
feature_names = artifacts['feature_names']

st.title("🛡️ RetentionPulse AI: Executive Churn Underwriter")
st.markdown("**Executive Assignment** | Data Science for Managers")
st.divider()

# Sidebar inputs
st.sidebar.header("📋 Account Details")
tenure = st.sidebar.slider("Account Tenure (Months)", 1, 72, 8)
monthly_fee = st.sidebar.slider("Monthly Subscription Value ($)", 30.0, 500.0, 480.0, step=10.0)
tickets = st.sidebar.slider("Support Tickets (Last 90 Days)", 0, 10, 6)
usage_drop = st.sidebar.slider("Usage Drop (%)", -40, 80, 62) / 100.0
contract_choice = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "1-Year", "2-Year"])

contract_map = {"Month-to-Month": 0, "1-Year": 1, "2-Year": 2}
contract_val = contract_map[contract_choice]

raw_input = pd.DataFrame([[tenure, monthly_fee, tickets, usage_drop, contract_val]], columns=feature_names)
scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=feature_names)

prob_churn = model.predict_proba(scaled_input)[0, 1]
annual_arr = monthly_fee * 12

col1, col2, col3, col4 = st.columns(4)
col1.metric("Churn Risk Probability", f"{prob_churn:.1%}", delta="High Risk" if prob_churn > 0.4 else "Healthy", delta_color="inverse")
col2.metric("Customer ARR at Risk", f"${annual_arr:,.0f}")
intervention_budget = 450.0
net_roi = (annual_arr * 0.55) - intervention_budget if prob_churn > 0.4 else 0
col3.metric("Intervention Net Gain", f"+${net_roi:,.0f}", delta="6.8x ROI" if prob_churn > 0.4 else "Monitor")
col4.metric("5-Fold CV ROC-AUC", f"{artifacts['cv_auc_mean']:.3f}")

st.divider()

left, right = st.columns([1, 1.2])

with left:
    st.subheader("🎯 Dynamic Decision Playbook")
    if prob_churn >= 0.65:
        st.error(f"⚠️ **Tier 1: High Risk Intervention (>65%)**")
        st.markdown("""
        - **Mandate**: Protect ARR exposure immediately.
        - **Action**: Deploy VP-Level Executive Sponsor call within 24 hours.
        - **Commercial Incentive**: Pre-authorized 15% renewal credit or 2 complimentary months on 2-year commitment.
        """)
    elif prob_churn >= 0.35:
        st.warning(f"🔔 **Tier 2: Elevated Monitoring (35-65%)**")
        st.markdown("- Automate product onboarding check-ins and audit ticket bottlenecks.")
    else:
        st.success(f"✅ **Tier 3: Stable Account (<35%)**")
        st.markdown("- Account healthy. Target for upsell / annual contract expansion.")

with right:
    st.subheader("🔍 TreeSHAP Feature Attribution")
    st.caption("Game-theoretic attribution showing features driving churn risk.")
    shap_vals = explainer(scaled_input)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(fig)
