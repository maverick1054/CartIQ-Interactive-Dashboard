"""
CartIQ Dashboard: Module 7 - Production API & Data Drift Monitoring (M6)
"""

import streamlit as st
import pandas as pd
import numpy as np
from drift_monitor import (
    run_dataset_drift_report, plot_drift_psi_summary,
    plot_feature_drift_distribution
)

def render_module7(cust_df, order_df):
    st.title("🚀 Module 7: ML Engineering — Production Serving & Drift Monitoring (M6)")
    st.markdown("Enterprise MLOps & Production Serving for CartIQ: Live Data & Concept Drift Engine (PSI & KS-Test), Production FastAPI REST Gateway, and Dockerized Container Deployment Architecture.")
    st.markdown("---")

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "🛡️ Live E-Commerce Data Drift Engine",
        "🔌 Production FastAPI REST Microservice",
        "📦 Containerization & MLOps Blueprint"
    ])

    # 1. Drift
    with sub_t1:
        st.markdown("### 🛡️ Live E-Commerce Data & Customer Drift Monitoring Engine")
        st.markdown(r"""
        Customer shopping patterns shift over time due to inflation, seasonality, and macroeconomic dynamics.
        This engine continuously tracks:
        - **Population Stability Index (PSI)**: Bucket-level shift magnitude ($\text{PSI} < 0.1$ stable, $0.1 \le \text{PSI} < 0.25$ warning, $\ge 0.25$ alert).
        - **Two-Sample Kolmogorov-Smirnov (KS) Test**: Continuous distribution differences ($p < 0.05$).
        """)

        st.markdown("#### 🎛️ Simulate Customer Cohort Shift (Production Scenario)")
        d_c1, d_c2, d_c3 = st.columns(3)
        with d_c1:
            shift_spend = st.slider("Simulate Total Spend Shift (Multiplier):", 0.5, 2.0, 1.0, 0.1)
        with d_c2:
            shift_orders = st.slider("Simulate Order Count Shift (Δ Orders):", -5, 5, 0, 1)
        with d_c3:
            shift_delivery = st.slider("Simulate Delivery Days Delay (Δ Days):", -5.0, 10.0, 0.0, 0.5)

        # Base reference vs current production batch
        ref_df = cust_df.copy()
        curr_batch = cust_df.sample(min(len(cust_df), 150), random_state=42).copy()

        if shift_spend != 1.0 and "total_spend" in curr_batch.columns:
            curr_batch["total_spend"] = curr_batch["total_spend"] * shift_spend
        if shift_orders != 0 and "total_orders" in curr_batch.columns:
            curr_batch["total_orders"] = np.clip(curr_batch["total_orders"] + shift_orders, 1, 50)
        if shift_delivery != 0.0 and "avg_delivery_days" in curr_batch.columns:
            curr_batch["avg_delivery_days"] = np.clip(curr_batch["avg_delivery_days"] + shift_delivery, 1.0, 30.0)

        with st.spinner("Analyzing Population Stability Index across customer features..."):
            drift_cols = ['age', 'total_orders', 'total_items', 'avg_delivery_days', 'total_spend', 'avg_order_value', 'gender', 'state']
            drift_cols = [c for c in drift_cols if c in ref_df.columns and c in curr_batch.columns]
            report = run_dataset_drift_report(ref_df, curr_batch, features=drift_cols)

        m1, m2, m3 = st.columns(3)
        m1.metric("Features Monitored", report["total_features"])
        m2.metric("Features with Shift (PSI ≥ 0.1)", report["drifted_count"])
        status_color = "red" if "CRITICAL" in report["overall_status"] else ("orange" if "WARNING" in report["overall_status"] else "green")
        m3.markdown(f"**Drift Alert:** <span style='color:{status_color}; font-weight:800; font-size:1.15rem;'>{report['overall_status']}</span>", unsafe_allow_html=True)

        fig_psi = plot_drift_psi_summary(report)
        st.plotly_chart(fig_psi, use_container_width=True)

        st.markdown("#### 🔬 Detailed Feature Drift Distribution Inspection")
        feat_choice = st.selectbox("Select Feature to Inspect:", [r["feature"] for r in report["feature_reports"]], key="drift_cartiq_feat")
        fig_feat = plot_feature_drift_distribution(ref_df, curr_batch, feat_choice)
        st.plotly_chart(fig_feat, use_container_width=True)

    # 2. REST API
    with sub_t2:
        st.markdown("### 🔌 Production REST API (FastAPI) Gateway")
        st.markdown("CartIQ includes an enterprise-ready FastAPI backend (`api.py`) exposing production endpoints with Pydantic validation, healthchecks, and drift detection.")

        st.info("💡 **FastAPI Server Running File**: `CartIQ_Dashboard/api.py` | Local URL: `http://localhost:8001` | Interactive Docs: `http://localhost:8001/docs`")

        api_col1, api_col2 = st.columns(2)
        with api_col1:
            st.markdown("#### 1. Predict Customer Value Tier (`POST /predict/customer-tier`)")
            st.code("""
curl -X 'POST' \\
  'http://localhost:8001/predict/customer-tier' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "age": 32,
    "total_orders": 8,
    "total_items": 15,
    "avg_delivery_days": 4.2,
    "avg_order_value": 1450.0,
    "gender": "Female",
    "state": "California",
    "model_name": "xgboost"
  }'
            """, language="bash")

        with api_col2:
            st.markdown("#### 2. Sample Response (`200 OK`)")
            st.code("""
{
  "customer_tier": "High Value (VIP)",
  "is_high_value": 1,
  "high_value_probability": 0.8924,
  "expected_annual_spend": 11600.0,
  "model_used": "xgboost",
  "latency_ms": 2.85
}
            """, language="json")

        st.markdown("#### 3. Available CartIQ Endpoints")
        st.markdown("""
        | HTTP Method | Route | Description |
        | :--- | :--- | :--- |
        | `GET` | `/health` | Service uptime, health status & active models |
        | `POST` | `/predict/customer-tier` | Predicts VIP/High-Value customer classification & expected spend |
        | `POST` | `/predict/delivery-delay` | Predicts delivery delay probability for an e-commerce order |
        | `POST` | `/drift/detect` | Computes PSI & KS-test drift metrics comparing batch to baseline |
        """)

    # 3. Docker Containerization
    with sub_t3:
        st.markdown("### 📦 Containerization & MLOps Architecture")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### `Dockerfile` (Production FastAPI + Streamlit Image)")
            st.code("""
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY . /app/
EXPOSE 8001 8502
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
            """, language="dockerfile")

        with c2:
            st.markdown("#### `docker-compose.yml` (Orchestrated Stack)")
            st.code("""
version: '3.8'
services:
  cartiq-api:
    build: .
    command: uvicorn api:app --host 0.0.0.0 --port 8001
    ports: ["8001:8001"]
  cartiq-dashboard:
    build: .
    command: streamlit run main.py --server.port=8502
    ports: ["8502:8502"]
    depends_on: [cartiq-api]
            """, language="yaml")
