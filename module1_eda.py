import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import get_raw_data

def format_currency(val):
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    elif val >= 1e6:
        return f"${val/1e6:.2f}M"
    elif val >= 1e3:
        return f"${val/1e3:.1f}K"
    else:
        return f"${val:,.0f}"

def format_days_hours(days_val):
    if pd.isna(days_val) or days_val <= 0:
        return "0 Days"
    d = int(days_val)
    h = int(round((days_val - d) * 24))
    if h >= 24:
        d += 1
        h = 0
    if d > 0 and h > 0:
        return f"{d}d {h}h"
    elif d > 0:
        return f"{d} Days"
    else:
        return f"{h} Hours"

def render_module1(master_df, cust_df, order_df):
    st.title("📦 Module 1: Data Cleaning, Visualization & EDA Process")
    st.markdown("---")
    
    if master_df.empty or cust_df.empty or order_df.empty:
        st.warning("⚠️ Insufficient data for EDA under current filter selection. Please broaden sidebar filters.")
        return

    # Overview high-level KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Sales Lines", f"{len(master_df):,}")
    with col2:
        st.metric("Unique Customers", f"{cust_df['customer_id'].nunique():,}")
    with col3:
        st.metric("Total Orders", f"{order_df['order_id'].nunique():,}")
    with col4:
        st.metric("Total Revenue", format_currency(master_df['total_price'].sum()))
    with col5:
        st.metric("Avg Delivery", format_days_hours(order_df['delivery_days'].mean()))

    st.markdown("---")
    st.markdown("### 1️⃣ Data Cleaning & Comprehensive Dataset Details")
    
    tab_master, tab_raw, tab_stats, tab_schema, tab_clean, tab_missing = st.tabs([
        "📋 Master Dataset Preview",
        "📁 4 Raw Source Datasets",
        "📊 Statistical Summary (describe)",
        "🧬 Column Schema & Types",
        "🧹 Data Cleaning Logic",
        "❓ Missing Value Audit"
    ])
    
    # --- TAB 1: MASTER DATASET PREVIEW ---
    with tab_master:
        st.markdown("#### 📋 Cleaned Master Transactional Dataset")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.info(f"**Total Records (Rows)**: `{len(master_df):,}`")
        m_c2.info(f"**Total Features (Columns)**: `{master_df.shape[1]}`")
        mem_mb = master_df.memory_usage(deep=True).sum() / (1024 * 1024)
        m_c3.info(f"**Memory Usage**: `{mem_mb:.2f} MB`")

        num_rows = st.slider("Select sample size to display", 5, 50, 10, key="master_sample_size")
        st.dataframe(master_df.head(num_rows), width='stretch')
        st.caption("Sample of merged Master Dataset (Sales + Orders + Customers + Products)")

    # --- TAB 2: 4 RAW SOURCE DATASETS ---
    with tab_raw:
        st.markdown("#### 📁 Raw Unprocessed Dataset Tables")
        try:
            raw_cust, raw_ord, raw_prod, raw_sales = get_raw_data()
            
            raw_sel = st.radio("Select Raw Table to Inspect", [
                "👥 Customers Table (customers.csv)",
                "🛒 Orders Table (orders.csv)",
                "📦 Products Table (products.csv)",
                "💰 Sales Table (sales.csv)"
            ], horizontal=True)

            if "Customers" in raw_sel:
                st.markdown(f"**Shape**: `{raw_cust.shape[0]}` rows × `{raw_cust.shape[1]}` columns")
                st.dataframe(raw_cust.head(10), width='stretch')
            elif "Orders" in raw_sel:
                st.markdown(f"**Shape**: `{raw_ord.shape[0]}` rows × `{raw_ord.shape[1]}` columns")
                st.dataframe(raw_ord.head(10), width='stretch')
            elif "Products" in raw_sel:
                st.markdown(f"**Shape**: `{raw_prod.shape[0]}` rows × `{raw_prod.shape[1]}` columns")
                st.dataframe(raw_prod.head(10), width='stretch')
            else:
                st.markdown(f"**Shape**: `{raw_sales.shape[0]}` rows × `{raw_sales.shape[1]}` columns")
                st.dataframe(raw_sales.head(10), width='stretch')
        except Exception as e:
            st.error(f"Could not load raw dataset files: {e}")

    # --- TAB 3: STATISTICAL SUMMARY (.describe()) ---
    with tab_stats:
        st.markdown("#### 📊 Descriptive Summary Statistics (`df.describe()`)")
        stat_mode = st.radio("Select Summary Type", ["Numerical Features (describe)", "Categorical Features (describe)"], horizontal=True)
        
        if "Numerical" in stat_mode:
            st.dataframe(master_df.describe().T.style.format("{:.2f}"), width='stretch')
        else:
            cat_df = master_df.select_dtypes(include=['object', 'category'])
            if not cat_df.empty:
                st.dataframe(cat_df.describe().T, width='stretch')
            else:
                st.info("No categorical columns found.")

    # --- TAB 4: COLUMN SCHEMA & TYPES AUDIT ---
    with tab_schema:
        st.markdown("#### 🧬 Detailed Column Schema Audit")
        schema_data = []
        for col in master_df.columns:
            dtype = str(master_df[col].dtype)
            non_null = master_df[col].notnull().sum()
            null_count = master_df[col].isnull().sum()
            null_pct = round((null_count / len(master_df)) * 100, 2)
            n_unique = master_df[col].nunique()
            sample_val = str(master_df[col].dropna().iloc[0]) if not master_df[col].dropna().empty else "N/A"
            schema_data.append({
                "Column Name": col,
                "Data Type": dtype,
                "Non-Null Count": non_null,
                "Missing Count": null_count,
                "Missing %": f"{null_pct}%",
                "Unique Values": n_unique,
                "Sample Value": sample_val
            })
        st.dataframe(pd.DataFrame(schema_data), width='stretch')

    # --- TAB 5: DATA CLEANING & PROCESSING LOGIC ---
    with tab_clean:
        st.markdown("""
        **Data Cleaning & Feature Engineering Audit Steps:**
        - **Date Standardization**: Converted string `order_date` & `delivery_date` into pandas `datetime64[ns]` objects.
        - **Delivery Duration Calculation**: Computed `delivery_days = (delivery_date - order_date).dt.days`. Cleaned negative/invalid records with median imputation.
        - **Target Binary Variable Creation**: Engineered `is_delayed` (1 if `delivery_days > median`, 0 otherwise) for classification algorithms.
        - **Demographic Binning**: Created binned `age_group` categories (`<25`, `25-40`, `41-60`, `60+`).
        - **Customer Summary Aggregation**: Aggregated transactional items to customer level (`total_spend`, `total_orders`, `avg_order_value`, `is_high_value`).
        - **Order Level Aggregation**: Aggregated items to order level (`total_order_price`, `item_count`, `is_delayed`).
        """)

    # --- TAB 6: MISSING VALUE AUDIT ---
    with tab_missing:
        st.markdown("#### ❓ Missing Value Audit Across Columns")
        missing_df = pd.DataFrame({
            "Column": master_df.columns,
            "Missing Count": master_df.isnull().sum(),
            "Missing Percentage (%)": (master_df.isnull().sum() / len(master_df) * 100).round(2)
        }).reset_index(drop=True)
        
        has_missing = missing_df[missing_df["Missing Count"] > 0]
        if not has_missing.empty:
            st.dataframe(has_missing, width='stretch')
        else:
            st.success("✅ Clean Dataset Verified! Zero unhandled missing values detected across all columns.")

    st.markdown("---")
    st.markdown("### 2️⃣ Exploratory Data Analysis (EDA) & Visualizations")
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "📊 Sales & Revenue Analysis", 
        "👥 Customer Demographics", 
        "📈 Feature Distributions", 
        "🔥 Correlation Matrix"
    ])
    
    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            rev_type = master_df.groupby('product_type')['total_price'].sum().reset_index()
            fig_type = px.bar(rev_type, x='product_type', y='total_price', color='product_type',
                              title="Total Revenue by Product Type", labels={'total_price': 'Revenue ($)', 'product_type': 'Product Type'},
                              text_auto='.2s')
            st.plotly_chart(fig_type, width='stretch')
            
        with c2:
            rev_state = master_df.groupby('state')['total_price'].sum().reset_index()
            fig_state = px.pie(rev_state, names='state', values='total_price', hole=0.4,
                               title="Revenue Share by Australian State")
            st.plotly_chart(fig_state, width='stretch')
            
        c3, c4 = st.columns(2)
        with c3:
            sales_color = master_df.groupby(['colour', 'product_type'])['quantity_sales'].sum().reset_index()
            fig_color = px.bar(sales_color, x='colour', y='quantity_sales', color='product_type',
                               title="Units Sold by Color & Category", barmode='stack')
            st.plotly_chart(fig_color, width='stretch')
            
        with c4:
            monthly_rev = master_df.groupby('order_month')['total_price'].sum().reset_index()
            fig_month = px.line(monthly_rev, x='order_month', y='total_price', markers=True,
                                title="Monthly Revenue Trend (2021)", labels={'order_month': 'Month', 'total_price': 'Revenue ($)'})
            st.plotly_chart(fig_month, width='stretch')

    with eda_tab2:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            gender_counts = cust_df['gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            fig_gen = px.bar(gender_counts, x='Gender', y='Count', color='Gender',
                             title="Customer Gender Distribution", text_auto=True)
            st.plotly_chart(fig_gen, width='stretch')
            
        with col_d2:
            fig_age = px.histogram(cust_df, x='age', nbins=20, color='gender',
                                   title="Customer Age Distribution", labels={'age': 'Age'})
            st.plotly_chart(fig_age, width='stretch')
            
        col_d3, col_d4 = st.columns(2)
        with col_d3:
            fig_spend = px.box(cust_df, x='state', y='total_spend', color='state',
                               title="Customer Spend Boxplot by State")
            st.plotly_chart(fig_spend, width='stretch')
            
        with col_d4:
            fig_orders = px.box(cust_df, x='gender', y='total_orders', color='gender',
                                title="Order Frequency Boxplot by Gender")
            st.plotly_chart(fig_orders, width='stretch')

    with eda_tab3:
        st.markdown("#### Distribution & Outlier Detection")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_del = px.histogram(order_df, x='delivery_days', nbins=25, marginal='box',
                                   title="Delivery Days Distribution & Outliers",
                                   color_discrete_sequence=['teal'])
            st.plotly_chart(fig_del, width='stretch')
            
        with col_dist2:
            fig_pay = px.histogram(order_df, x='payment', nbins=30, marginal='box',
                                   title="Order Payment Value Distribution",
                                   color_discrete_sequence=['coral'])
            st.plotly_chart(fig_pay, width='stretch')

    with eda_tab4:
        st.markdown("#### Feature Correlation Heatmap")
        num_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = [c for c in ['age', 'price_per_unit', 'quantity_sales', 'total_price', 'payment', 'delivery_days', 'order_month'] if c in num_cols]
        corr_matrix = master_df[selected_cols].corr().fillna(0).round(2)
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis',
                             title="Correlation Matrix of Numerical Variables", aspect='auto')
        st.plotly_chart(fig_corr, width='stretch')
