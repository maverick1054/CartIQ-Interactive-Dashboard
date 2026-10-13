import streamlit as st
import pandas as pd
import numpy as np

from data_loader import get_cleaned_merged_data, get_customer_summary, get_order_summary
from module1_eda import render_module1
from module2_regression import render_module2
from module3_trees_reg import render_module3
from module4_ensemble import render_module4
from module5_placement import render_module5
from module6_evaluation import render_module6
from module7_production import render_module7

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="CartIQ - E-Commerce Analytics & ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# HIGH-END PROFESSIONAL CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A !important;
    }

    /* Light Theme Main Background */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 95%;
    }

    /* Professional Royal Blue Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.25);
    }
    
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    .hero-banner p {
        color: #DBEAFE !important;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        font-weight: 500;
    }

    /* High Contrast Metric Cards Styling (Light Theme) */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.25s ease-in-out !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        border-color: #2563EB !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.12), 0 4px 6px -2px rgba(37, 99, 235, 0.05) !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        font-size: 0.84rem !important;
        font-weight: 800 !important;
        color: #334155 !important; /* High Contrast Dark Slate Label */
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important; /* Rich Navy Blue Value */
    }

    /* Tab Navigation Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 700;
        font-size: 0.92rem;
        color: #475569 !important;
        border: none !important;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }

    /* Sidebar Customization (Light Professional Theme) */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 14px;
        border-radius: 8px;
        color: #1E293B !important;
    }

    /* Titles & Headings */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* DataFrame Container */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #CBD5E1;
    }

    /* Alerts and Cards */
    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA CACHING
# -------------------------------------------------------------
@st.cache_data
def load_all_data():
    master_df = get_cleaned_merged_data()
    cust_df = get_customer_summary()
    order_df = get_order_summary()
    return master_df, cust_df, order_df

master_df, cust_df, order_df = load_all_data()

# -------------------------------------------------------------
# TOP HERO BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🛒 CartIQ E-Commerce Intelligence Dashboard</h1>
    <p>Comprehensive Sales Analytics, Customer Segmentation, and Predictive Machine Learning Models</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION & DYNAMIC FILTERS
# -------------------------------------------------------------
st.sidebar.markdown("## 🛒 CartIQ Navigation")
st.sidebar.caption("Modular E-Commerce Analytics Platform")
st.sidebar.markdown("---")

selected_module = st.sidebar.radio(
    "Select Module",
    [
        "📦 Module 1: Data Cleaning & EDA",
        "📈 Module 2: Linear & Logistic Regression",
        "🌲 Module 3: Regularization & Decision Trees",
        "⚡ Module 4: Bagging & Boosting Ensembles",
        "🎯 Module 5: Placement Prediction & PCA Analytics",
        "📊 Module 6: Model Evaluation, Calibration & CV (M5)",
        "🚀 Module 7: Production API & Drift Monitoring (M6)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Data Filters")

# State filter
all_states = sorted(master_df['state'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Filter by State", options=all_states, default=all_states)

# Gender filter
all_genders = sorted(cust_df['gender'].dropna().unique().tolist())
selected_genders = st.sidebar.multiselect("Filter by Gender", options=all_genders, default=all_genders)

# Apply filters
filtered_master = master_df[
    (master_df['state'].isin(selected_states)) & 
    (master_df['gender'].isin(selected_genders))
]
filtered_cust = cust_df[
    (cust_df['state'].isin(selected_states)) & 
    (cust_df['gender'].isin(selected_genders))
]
filtered_order = order_df[
    (order_df['state'].isin(selected_states)) & 
    (order_df['gender'].isin(selected_genders))
]

st.sidebar.markdown("---")
st.sidebar.caption("🚀 CartIQ Dashboard | Powered by Streamlit & Scikit-Learn")

# -------------------------------------------------------------
# MODULE ROUTING
# -------------------------------------------------------------
if filtered_master.empty or filtered_cust.empty or filtered_order.empty or len(filtered_cust) < 5 or len(filtered_order) < 5:
    st.warning("⚠️ Insufficient data available for the selected filters (minimum 5 samples required for modeling). Please expand your State and Gender filter selections in the sidebar.")
else:
    if selected_module == "📦 Module 1: Data Cleaning & EDA":
        render_module1(filtered_master, filtered_cust, filtered_order)
    elif selected_module == "📈 Module 2: Linear & Logistic Regression":
        render_module2(filtered_cust, filtered_order)
    elif selected_module == "🌲 Module 3: Regularization & Decision Trees":
        render_module3(filtered_cust, filtered_order)
    elif selected_module == "⚡ Module 4: Bagging & Boosting Ensembles":
        render_module4(filtered_cust, filtered_order)
    elif selected_module == "🎯 Module 5: Placement Prediction & PCA Analytics":
        render_module5(filtered_cust, filtered_order)
    elif selected_module == "📊 Module 6: Model Evaluation, Calibration & CV (M5)":
        render_module6(filtered_cust, filtered_order)
    elif selected_module == "🚀 Module 7: Production API & Drift Monitoring (M6)":
        render_module7(filtered_cust, filtered_order)

