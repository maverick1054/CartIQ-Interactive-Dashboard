import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

from sklearn.preprocessing import StandardScaler

def render_module3(cust_df, order_df):
    st.title("🌲 Module 3: Regularization & Decision Trees")
    st.markdown("---")
    
    if cust_df.empty or len(cust_df) < 5 or order_df.empty or len(order_df) < 5:
        st.warning("⚠️ Insufficient data for regularization & decision tree modeling under current filter selection. Please broaden sidebar filters.")
        return

    sub_t1, sub_t2 = st.tabs([
        "🛡️ Regularization (Ridge & Lasso Regression)", 
        "🌳 Decision Tree Models (Regressor & Classifier)"
    ])
    
    # -------------------------------------------------------------
    # SUB TAB 1: REGULARIZATION (RIDGE & LASSO)
    # -------------------------------------------------------------
    with sub_t1:
        st.subheader("1️⃣ Regularization: Ridge (L2) & Lasso (L1) Regression")
        st.markdown("""
        Regularization prevents model overfitting by penalizing large model coefficients:
        - **Ridge ($L_2$ Regularization)**: Shrinks coefficients smoothly towards zero without setting them strictly to zero.
        - **Lasso ($L_1$ Regularization)**: Performs feature selection by driving non-essential feature coefficients to exactly zero.
        """)
        
        feature_cols_num = ['age', 'total_orders', 'avg_delivery_days']
        feature_cols_cat = ['gender', 'state']
        
        X_raw = cust_df[feature_cols_num + feature_cols_cat]
        y = cust_df['total_spend']
        X = pd.get_dummies(X_raw, columns=feature_cols_cat, drop_first=True, dtype=int)
        
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            alpha_val = st.select_slider("Alpha Regularization Strength (α)",
                                         options=[0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0],
                                         value=1.0, key="reg_alpha")
        with col_reg2:
            st.info(f"Currently testing penalty **α = {alpha_val}** for Ridge & Lasso vs OLS Linear Regression.")
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for proper regularization penalty application
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit models
        ols_model = LinearRegression().fit(X_train_scaled, y_train)
        ridge_model = Ridge(alpha=alpha_val, random_state=42).fit(X_train_scaled, y_train)
        lasso_model = Lasso(alpha=alpha_val, random_state=42, max_iter=5000).fit(X_train_scaled, y_train)
        
        # Compare metrics
        models_dict = {'OLS Linear': ols_model, 'Ridge (L2)': ridge_model, 'Lasso (L1)': lasso_model}
        results = []
        for name, m in models_dict.items():
            preds = m.predict(X_test_scaled)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            non_zero = np.sum(np.abs(m.coef_) > 1e-4)
            results.append({'Model': name, 'R² Score': f"{r2:.3f}", 'MAE ($)': f"{mae:.2f}", 'RMSE ($)': f"{rmse:.2f}", 'Non-Zero Coefs': int(non_zero)})
            
        st.table(pd.DataFrame(results))
        
        # Coefficient comparison chart
        coef_comp = pd.DataFrame({
            'Feature': X.columns,
            'OLS': ols_model.coef_,
            'Ridge (L2)': ridge_model.coef_,
            'Lasso (L1)': lasso_model.coef_
        })
        
        coef_melt = coef_comp.melt(id_vars=['Feature'], var_name='Model', value_name='Coefficient')
        fig_comp = px.bar(coef_melt, x='Feature', y='Coefficient', color='Model', barmode='group',
                          title=f"Feature Coefficient Comparison (α = {alpha_val})")
        st.plotly_chart(fig_comp, width='stretch')

    # -------------------------------------------------------------
    # SUB TAB 2: DECISION TREES
    # -------------------------------------------------------------
    with sub_t2:
        st.subheader("2️⃣ Decision Tree Models: Regressor & Classifier")
        
        tree_mode = st.radio("Select Decision Tree Type", ["Decision Tree Regressor (Spend)", "Decision Tree Classifier (Delay Risk)"], horizontal=True)
        
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            max_depth = st.slider("Max Tree Depth (max_depth)", 1, 20, 5, key="dt_depth")
        with col_hp2:
            min_samples = st.slider("Min Samples Split (min_samples_split)", 2, 20, 5, key="dt_samples")
            
        if tree_mode == "Decision Tree Regressor (Spend)":
            st.markdown("#### 🪵 Decision Tree Regressor: Customer Spend")
            X_raw = cust_df[['age', 'total_orders', 'avg_delivery_days', 'gender', 'state']]
            y = cust_df['total_spend']
            X = pd.get_dummies(X_raw, columns=['gender', 'state'], drop_first=True, dtype=int)
            
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            
            dt_reg = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
            dt_reg.fit(X_tr, y_tr)
            
            y_pred = dt_reg.predict(X_te)
            
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("R² Score", f"{r2_score(y_te, y_pred):.3f}")
            cm2.metric("MAE ($)", f"{mean_absolute_error(y_te, y_pred):.2f}")
            cm3.metric("RMSE ($)", f"{np.sqrt(mean_squared_error(y_te, y_pred)):.2f}")
            
            # Feature Importances
            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt_reg.feature_importances_}).sort_values(by='Importance', ascending=True)
            fig_imp = px.bar(imp_df.tail(10), x='Importance', y='Feature', orientation='h',
                             title="Decision Tree Feature Importances (Regressor)", color='Importance', color_continuous_scale='Greens')
            st.plotly_chart(fig_imp, width='stretch')
            
        else:
            st.markdown("#### 🍃 Decision Tree Classifier: Delivery Delay Risk")
            X_raw = order_df[['payment', 'total_order_price', 'item_count', 'age', 'order_month', 'gender', 'state']]
            y = order_df['is_delayed']
            X = pd.get_dummies(X_raw, columns=['gender', 'state'], drop_first=True, dtype=int)
            
            min_class_cnt = y.value_counts().min() if y.nunique() > 1 else 0
            use_stratify = y if (y.nunique() > 1 and (min_class_cnt * 0.2) >= 1) else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=use_stratify)
            
            if y_tr.nunique() < 2:
                st.warning("⚠️ The filtered dataset does not contain both target classes (Delayed and On Time) for tree classifier training. Broaden your sidebar filters.")
            else:
                dt_clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
                dt_clf.fit(X_tr, y_tr)
                
                y_pred = dt_clf.predict(X_te)
                
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Accuracy", f"{accuracy_score(y_te, y_pred)*100:.1f}%")
                cc2.metric("Precision", f"{precision_score(y_te, y_pred, zero_division=0)*100:.1f}%")
                cc3.metric("Recall", f"{recall_score(y_te, y_pred, zero_division=0)*100:.1f}%")
                cc4.metric("F1 Score", f"{f1_score(y_te, y_pred, zero_division=0):.3f}")
                
                col_tree1, col_tree2 = st.columns(2)
                with col_tree1:
                    cm = confusion_matrix(y_te, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Greens',
                                       x=['On Time (0)', 'Delayed (1)'], y=['On Time (0)', 'Delayed (1)'],
                                       title="Confusion Matrix (Decision Tree Classifier)")
                    st.plotly_chart(fig_cm, width='stretch')
                    
                with col_tree2:
                    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt_clf.feature_importances_}).sort_values(by='Importance', ascending=True)
                    fig_imp = px.bar(imp_df.tail(10), x='Importance', y='Feature', orientation='h',
                                     title="Decision Tree Feature Importances (Classifier)", color='Importance', color_continuous_scale='Greens')
                    st.plotly_chart(fig_imp, width='stretch')

