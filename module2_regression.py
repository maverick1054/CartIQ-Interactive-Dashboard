import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

def render_module2(cust_df, order_df):
    st.title("📈 Module 2: Linear Regression & Logistic Regression")
    st.markdown("---")
    
    if cust_df.empty or len(cust_df) < 5 or order_df.empty or len(order_df) < 5:
        st.warning("⚠️ Insufficient data for regression modeling under current filter selection. Please broaden sidebar filters.")
        return

    sub_tab1, sub_tab2 = st.tabs([
        "📐 Linear Regression (Customer Spend Prediction)", 
        "🎯 Logistic Regression (Delivery Delay Classification)"
    ])
    
    # -------------------------------------------------------------
    # SUB TAB 1: LINEAR REGRESSION
    # -------------------------------------------------------------
    with sub_tab1:
        st.subheader("1️⃣ Linear Regression Model: Predict Customer Spend")
        st.markdown("Predict **Total Customer Spend ($)** based on customer age, state, gender, and purchase frequency.")
        
        # Prepare feature matrix X and target y
        feature_cols_num = ['age', 'total_orders', 'avg_delivery_days']
        feature_cols_cat = ['gender', 'state']
        
        X_raw = cust_df[feature_cols_num + feature_cols_cat]
        y = cust_df['total_spend']
        
        # One hot encoding for categoricals
        X = pd.get_dummies(X_raw, columns=feature_cols_cat, drop_first=True, dtype=int)
        
        test_size = st.slider("Train-Test Split Ratio (% Test Set)", 10, 40, 20, step=5, key="lin_split") / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train Linear Regression model
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        
        y_pred_train = lin_model.predict(X_train)
        y_pred_test = lin_model.predict(X_test)
        
        # Metrics
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R² Score (Test)", f"{r2_test:.3f}")
        col_m2.metric("MAE ($)", f"{mae_test:.2f}")
        col_m3.metric("RMSE ($)", f"{rmse_test:.2f}")
        col_m4.metric("R² Score (Train)", f"{r2_score(y_train, y_pred_train):.3f}")
        
        # Plots
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
            fig_scat = px.scatter(pred_df, x='Actual', y='Predicted',
                                  title="Actual vs Predicted Customer Spend ($)",
                                  labels={'Actual': 'Actual Spend ($)', 'Predicted': 'Predicted Spend ($)'})
            fig_scat.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                               line=dict(color="Red", dash="dash"), name="Ideal 1:1 Fit")
            st.plotly_chart(fig_scat, width='stretch')
            
        with col_p2:
            coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lin_model.coef_})
            coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values(by='AbsCoef', ascending=True).tail(12)
            fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                               title="Linear Regression Coefficients (Feature Impact)",
                               color='Coefficient', color_continuous_scale='RdBu', color_continuous_midpoint=0)
            st.plotly_chart(fig_coef, width='stretch')
            
        # Interactive Simulator
        st.markdown("#### 🎮 Interactive Customer Spend Simulator")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            input_age = st.number_input("Customer Age", min_value=18, max_value=90, value=35, key="lin_age")
        with col_s2:
            input_orders = st.number_input("Total Orders", min_value=1, max_value=20, value=3, key="lin_ord")
        with col_s3:
            gen_opts = cust_df['gender'].dropna().unique().tolist()
            input_gender = st.selectbox("Gender", options=gen_opts if gen_opts else ['Female', 'Male'], key="lin_gen")
        with col_s4:
            st_opts = cust_df['state'].dropna().unique().tolist()
            input_state = st.selectbox("State", options=st_opts if st_opts else ['NSW'], key="lin_st")
            
        # Construct single input row matching dummy columns
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data['age'] = input_age
        input_data['total_orders'] = input_orders
        avg_del = cust_df['avg_delivery_days'].mean()
        input_data['avg_delivery_days'] = avg_del if pd.notna(avg_del) else 14.0
        
        gen_col = f"gender_{input_gender}"
        if gen_col in input_data.columns:
            input_data[gen_col] = 1
            
        st_col = f"state_{input_state}"
        if st_col in input_data.columns:
            input_data[st_col] = 1
            
        predicted_spend = lin_model.predict(input_data)[0]
        st.success(f"💰 **Predicted Customer Spend**: **${max(0, predicted_spend):,.2f}**")

    # -------------------------------------------------------------
    # SUB TAB 2: LOGISTIC REGRESSION
    # -------------------------------------------------------------
    with sub_tab2:
        st.subheader("2️⃣ Logistic Regression Model: Predict Delivery Delay Risk")
        st.markdown("Classify whether an order will experience **Delivery Delay (`is_delayed = 1`)**.")
        
        feature_cols_num_log = ['payment', 'total_order_price', 'item_count', 'age', 'order_month']
        feature_cols_cat_log = ['gender', 'state']
        
        X_log_raw = order_df[feature_cols_num_log + feature_cols_cat_log]
        y_log = order_df['is_delayed']
        
        X_log = pd.get_dummies(X_log_raw, columns=feature_cols_cat_log, drop_first=True, dtype=int)
        
        test_size_log = st.slider("Train-Test Split Ratio (% Test Set)", 10, 40, 20, step=5, key="log_split") / 100.0
        
        # Safely determine stratify parameter
        min_class_cnt = y_log.value_counts().min() if y_log.nunique() > 1 else 0
        use_stratify = y_log if (y_log.nunique() > 1 and (min_class_cnt * test_size_log) >= 1) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_log, y_log, test_size=test_size_log, random_state=42, stratify=use_stratify)
        
        if y_tr.nunique() < 2:
            st.warning("⚠️ The filtered dataset does not contain both target classes (Delayed and On Time) for model training. Broaden your sidebar filters.")
        else:
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            log_model = LogisticRegression(max_iter=1000)
            log_model.fit(X_tr_scaled, y_tr)
            
            y_pred_log = log_model.predict(X_te_scaled)
            if hasattr(log_model, "predict_proba") and len(log_model.classes_) > 1 and 1 in log_model.classes_:
                class_idx = np.where(log_model.classes_ == 1)[0][0]
                y_prob_log = log_model.predict_proba(X_te_scaled)[:, class_idx]
            else:
                y_prob_log = np.zeros(len(y_te))
            
            acc = accuracy_score(y_te, y_pred_log)
            prec = precision_score(y_te, y_pred_log, zero_division=0)
            rec = recall_score(y_te, y_pred_log, zero_division=0)
            f1 = f1_score(y_te, y_pred_log, zero_division=0)
            
            c_l1, c_l2, c_l3, c_l4 = st.columns(4)
            c_l1.metric("Accuracy", f"{acc * 100:.1f}%")
            c_l2.metric("Precision", f"{prec * 100:.1f}%")
            c_l3.metric("Recall", f"{rec * 100:.1f}%")
            c_l4.metric("F1 Score", f"{f1:.3f}")
            
            col_lg1, col_lg2 = st.columns(2)
            with col_lg1:
                cm = confusion_matrix(y_te, y_pred_log)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   x=['On Time (0)', 'Delayed (1)'], y=['On Time (0)', 'Delayed (1)'],
                                   title="Confusion Matrix (Logistic Regression)",
                                   labels={'x': 'Predicted Label', 'y': 'Actual Label'})
                st.plotly_chart(fig_cm, width='stretch')
                
            with col_lg2:
                if len(np.unique(y_te)) > 1:
                    fpr, tpr, _ = roc_curve(y_te, y_prob_log)
                    roc_auc = auc(fpr, tpr)
                else:
                    fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Logistic Regression (AUC = {roc_auc:.3f})',
                                             line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess (AUC = 0.5)',
                                             line=dict(color='navy', width=2, dash='dash')))
                fig_roc.update_layout(title="Receiver Operating Characteristic (ROC) Curve",
                                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, width='stretch')
                
            # Interactive Simulator for Logistic Regression
            st.markdown("#### 🎮 Interactive Delivery Delay Risk Simulator")
            cl1, cl2, cl3, cl4, cl5 = st.columns(5)
            with cl1:
                sim_pay = st.number_input("Order Payment ($)", min_value=100, max_value=100000, value=30000, key="log_pay")
            with cl2:
                avg_price_default = int(order_df['total_order_price'].mean()) if not order_df['total_order_price'].empty else 1000
                sim_order_price = st.number_input("Total Order Item Price ($)", min_value=10, max_value=50000, value=max(10, avg_price_default), key="log_ord_price")
            with cl3:
                sim_items = st.number_input("Item Count", min_value=1, max_value=20, value=3, key="log_itm")
            with cl4:
                g_log_opts = order_df['gender'].dropna().unique().tolist()
                sim_gender = st.selectbox("Gender", options=g_log_opts if g_log_opts else ['Female', 'Male'], key="log_gen")
            with cl5:
                st_log_opts = order_df['state'].dropna().unique().tolist()
                sim_state = st.selectbox("State", options=st_log_opts if st_log_opts else ['NSW'], key="log_st")
                
            sim_month = st.slider("Order Month", 1, 12, 6, key="log_mth")

            sim_input = pd.DataFrame(0, index=[0], columns=X_log.columns)
            sim_input['payment'] = sim_pay
            sim_input['total_order_price'] = sim_order_price
            sim_input['item_count'] = sim_items
            avg_age = order_df['age'].mean()
            sim_input['age'] = avg_age if pd.notna(avg_age) else 35
            sim_input['order_month'] = sim_month
            
            sim_g_col = f"gender_{sim_gender}"
            if sim_g_col in sim_input.columns:
                sim_input[sim_g_col] = 1

            st_c = f"state_{sim_state}"
            if st_c in sim_input.columns:
                sim_input[st_c] = 1
                
            sim_scaled = scaler.transform(sim_input)
            if hasattr(log_model, "predict_proba") and len(log_model.classes_) > 1 and 1 in log_model.classes_:
                class_idx = np.where(log_model.classes_ == 1)[0][0]
                prob_delay = log_model.predict_proba(sim_scaled)[0, class_idx]
            else:
                prob_delay = 0.0
            pred_class = "Delayed ⚠️" if prob_delay >= 0.5 else "On Time ✅"
            
            st.info(f"🚚 **Predicted Status**: **{pred_class}** (Delay Risk Probability: **{prob_delay * 100:.1f}%**)")
