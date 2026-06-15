import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from io import BytesIO
from utils.scoring_engine import load_model, score_single, score_portfolio


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Collections Agent | Bajaj Finance DMS",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.header-bar {
    background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #1565c0 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.header-bar h1 { color: white; font-size: 1.4rem; font-weight: 600; margin: 0; }
.header-bar p  { color: rgba(255,255,255,0.75); font-size: 0.8rem; margin: 0; }

.score-card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    border-top: 4px solid var(--bucket-color, #1565c0);
}
.score-value {
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--bucket-color, #1565c0);
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.score-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #78909c; margin-top: 0.3rem; }

.badge { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; }
.badge-critical { background: #ffebee; color: #c62828; }
.badge-high     { background: #fff3e0; color: #e65100; }
.badge-medium   { background: #fffde7; color: #f57f17; }
.badge-low      { background: #e8f5e9; color: #2e7d32; }

.action-box {
    background: #f8f9ff;
    border-left: 4px solid #1565c0;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #1a237e;
    margin-top: 1rem;
}
.metric-row { display: flex; gap: 12px; margin-bottom: 1rem; }
.metric-mini { flex: 1; background: #f5f7ff; border-radius: 10px; padding: 0.8rem 1rem; text-align: center; }
.metric-mini .val { font-size: 1.5rem; font-weight: 600; color: #1a237e; font-family: 'DM Mono', monospace; }
.metric-mini .lbl { font-size: 0.7rem; color: #78909c; text-transform: uppercase; letter-spacing: 0.08em; }

.section-title {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.12em; color: #90a4ae;
    margin: 1.2rem 0 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid #eceff1;
}
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None, None


model, scaler, features = get_model()

# ── Header ─────────────────────────────────────────────────────────────────────
today = datetime.date.today().strftime("%d %b %Y")
st.markdown(f"""
<div class="header-bar">
  <h1>🏦 Smart Collections Prioritization Agent</h1>
  <p>Bajaj Finance Limited · Debt Management Services · {today}</p>
</div>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
if model is None:
    st.error("Model files not found. Run these commands first:")
    st.code("python -c \"from credit_risk import *\" # Run notebook cell 13\npython -c \"from credit_risk import *\" # Run notebook cell 14")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", [
        "🔍 Score Single Account",
        "📋 Bulk Portfolio Scoring",
        "📊 Portfolio Dashboard",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.success("✅ PD Model loaded")
    st.caption(f"Features: {len(features)}")
    st.markdown("---")
    st.caption("For internal use only.\nDo not share outside DMS.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – SCORE SINGLE ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Score Single Account":
    st.subheader("Score a Single Customer Account")
    st.caption("Enter the customer's details. The model returns a PD score and recommended action instantly.")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        with st.form("single_score_form"):
            st.markdown('<div class="section-title">Loan Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            emi_amount   = c1.number_input("EMI Amount (₹)",    min_value=500.0,    max_value=100000.0,  value=8000.0,   step=500.0)
            loan_amount  = c2.number_input("Loan Amount (₹)",   min_value=5000.0,   max_value=2000000.0, value=150000.0, step=5000.0)
            tenure       = c1.number_input("Tenure (months)",   min_value=1,        max_value=120,       value=24)
            credit_score = c2.number_input("Credit Score",      min_value=300,      max_value=900,       value=650)

            st.markdown('<div class="section-title">Delinquency Info</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            dpd_options = [0] + list(range(1, 31)) + [40, 50, 60, 75, 90, 120, 150, 180]
            dpd          = c3.selectbox("DPD (Days Past Due)", dpd_options)
            bounce_count = c4.number_input("EMI Bounce Count",           min_value=0, max_value=12, value=1)
            last_payment = c3.number_input("Last Payment (days ago)",    min_value=0, max_value=365, value=45)

            st.markdown('<div class="section-title">Contact History</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            contact_att  = c5.number_input("Contact Attempts",    min_value=0, max_value=20, value=3)
            successful_c = c6.number_input("Successful Contacts", min_value=0, max_value=20, value=1)
            ptp = st.selectbox("Promise to Pay (PTP) Given?", [0, 1],
                                format_func=lambda x: "Yes" if x else "No")

            submitted = st.form_submit_button("▶  Calculate PD Score", use_container_width=True, type="primary")

    with col_right:
        if submitted:
            input_data = {
                "DPD":                   dpd,
                "Bounce_Count":          bounce_count,
                "Contact_Attempts":      contact_att,
                "Successful_Contacts":   min(successful_c, contact_att),
                "Promise_to_Pay":        ptp,
                "Last_Payment_Days":     last_payment,
                "Credit_Score":          credit_score,
                "EMI_Amount":            emi_amount,
                "Loan_Amount":           loan_amount,
                "Tenure_Amount":         tenure,
            }
            result  = score_single(input_data, model, scaler, features)
            pd_pct  = result["pd_score"]
            bucket  = result["risk_bucket"]
            priority= result["priority"]
            action  = result["call_action"]

            color_map = {
                "Critical": "#c62828",
                "High":     "#e65100",
                "Medium":   "#f57f17",
                "Low":      "#2e7d32",
            }
            color     = color_map.get(bucket, "#1565c0")
            badge_cls = f"badge-{bucket.lower()}"

            st.markdown(f"""
            <div class="score-card" style="--bucket-color:{color}">
                <div class="score-value">{pd_pct}%</div>
                <div class="score-label">Probability of Default</div>
                <br>
                <span class="badge {badge_cls}">{bucket} Risk</span>
                <br><br>
                <div style="font-size:0.85rem;color:#455a64;font-weight:500">{priority}</div>
            </div>
            <div class="action-box"><strong>Recommended Action</strong><br>{action}</div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Score breakdown**")
            dpd_c     = min(dpd / 180 * 40, 40)
            bounce_c  = min(bounce_count / 5 * 15, 15)
            payment_c = min(last_payment / 200 * 20, 20)
            credit_c  = min((850 - credit_score) / 550 * 15, 15)
            ratio     = 1 - min(successful_c, contact_att) / (contact_att + 1)
            contact_c = min(ratio * 10, 10)

            for label, val in [
                ("DPD weight",       dpd_c),
                ("Bounce weight",    bounce_c),
                ("Payment recency",  payment_c),
                ("Credit score",     credit_c),
                ("Contact failure",  contact_c),
            ]:
                st.progress(val / 100, text=f"{label}: {val:.1f} pts")
        else:
            st.info("Fill in the form and click **Calculate PD Score** to see results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – BULK PORTFOLIO SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Bulk Portfolio Scoring":
    st.subheader("Bulk Portfolio Scoring")
    st.caption("Upload your Excel file. Every account gets scored and the result is returned as a ranked call list.")

    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

    if uploaded:
        try:
            df_raw = pd.read_excel(uploaded)
            st.success(f"✅ Loaded {len(df_raw)} accounts")

            with st.spinner("Scoring portfolio..."):
                df_scored = score_portfolio(df_raw, model, scaler, features)

            total    = len(df_scored)
            critical = (df_scored["Risk_Bucket"] == "Critical").sum()
            high     = (df_scored["Risk_Bucket"] == "High").sum()
            avg_pd   = df_scored["PD_Score_%"].mean()

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-mini"><div class="val">{total}</div><div class="lbl">Total accounts</div></div>
              <div class="metric-mini"><div class="val" style="color:#c62828">{critical}</div><div class="lbl">Critical</div></div>
              <div class="metric-mini"><div class="val" style="color:#e65100">{high}</div><div class="lbl">High risk</div></div>
              <div class="metric-mini"><div class="val">{avg_pd:.1f}%</div><div class="lbl">Avg PD score</div></div>
            </div>
            """, unsafe_allow_html=True)

            display_cols = [c for c in [
                "Priority_Rank", "Customer_ID", "Customer_Name",
                "Loan_Type", "EMI_Amount", "DPD",
                "PD_Score_%", "Risk_Bucket", "Call_Action"
            ] if c in df_scored.columns]

            st.dataframe(df_scored[display_cols].head(100), use_container_width=True, hide_index=True)

            output = BytesIO()
            df_scored.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)
            fname = f"prioritized_call_list_{datetime.date.today().strftime('%Y%m%d')}.xlsx"
            st.download_button(
                "⬇  Download Prioritized Call List",
                data=output, file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary",
            )
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Upload an Excel file to begin. Required columns:")
        st.code(", ".join(features))

        st.markdown("---")
        if st.button("Load sample dataset (data/loan_accounts.xlsx)"):
            if os.path.exists("data/loan_accounts.xlsx"):
                df_sample = pd.read_excel("data/loan_accounts.xlsx")
                df_scored = score_portfolio(df_sample, model, scaler, features)
                st.success(f"Loaded {len(df_scored)} sample accounts")
                st.dataframe(df_scored[[c for c in [
                    "Priority_Rank", "Customer_ID", "EMI_Amount", "DPD",
                    "PD_Score_%", "Risk_Bucket", "Call_Action"
                ] if c in df_scored.columns]].head(50),
                use_container_width=True, hide_index=True)
                out2 = BytesIO()
                df_scored.to_excel(out2, index=False, engine="openpyxl")
                out2.seek(0)
                st.download_button("⬇  Download sample call list", data=out2,
                    file_name="sample_call_list.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error("Run notebook cells 13-14 first to generate data and train model.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PORTFOLIO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Portfolio Dashboard":
    st.subheader("Portfolio Risk Dashboard")

    @st.cache_data
    def load_portfolio():
        if os.path.exists("data/loan_accounts.xlsx"):
            df = pd.read_excel("data/loan_accounts.xlsx")
            return score_portfolio(df, model, scaler, features)
        return None

    df_dash = load_portfolio()

    if df_dash is None:
        st.warning("Run notebook cells 13-14 first to generate data and train model.")
    else:
        total    = len(df_dash)
        critical = (df_dash["Risk_Bucket"] == "Critical").sum()
        high     = (df_dash["Risk_Bucket"] == "High").sum()
        avg_pd   = df_dash["PD_Score_%"].mean()
        at_risk_emi = df_dash[df_dash["Risk_Bucket"].isin(["Critical","High"])]["EMI_Amount"].sum()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Accounts",  f"{total:,}")
        k2.metric("Critical",        f"{critical}")
        k3.metric("High Risk",       f"{high}")
        k4.metric("Avg PD Score",    f"{avg_pd:.1f}%")
        k5.metric("EMI at Risk (₹)", f"₹{at_risk_emi:,.0f}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Risk bucket distribution**")
            bucket_counts = df_dash["Risk_Bucket"].value_counts().reindex(
                ["Critical","High","Medium","Low"], fill_value=0)
            st.bar_chart(pd.DataFrame({"Count": bucket_counts}))

        with c2:
            st.markdown("**PD score distribution**")
            hist_data = pd.cut(df_dash["PD_Score_%"], bins=10).value_counts().sort_index()
            st.bar_chart(pd.DataFrame({"Frequency": hist_data.values}, index=range(len(hist_data))))

        st.markdown("**Average PD score by DPD bucket**")
        df_dash["DPD_Bucket"] = pd.cut(
            df_dash["DPD"], bins=[-1,0,1,30,60,90,120,181],
            labels=["0","1","1-30","31-60","61-90","91-120","121+"])
        dpd_avg = df_dash.groupby("DPD_Bucket", observed=True)["PD_Score_%"].mean()
        st.bar_chart(pd.DataFrame({"Avg PD Score": dpd_avg.values}, index=dpd_avg.index))

        st.markdown("**Top 20 accounts to call today**")
        top20 = df_dash.head(20)[[c for c in [
            "Priority_Rank","Customer_ID","Loan_Type","EMI_Amount",
            "DPD","PD_Score_%","Risk_Bucket","Call_Action"
        ] if c in df_dash.columns]]
        st.dataframe(top20, use_container_width=True, hide_index=True)
