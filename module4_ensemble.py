import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              BaggingRegressor, BaggingClassifier,
                              AdaBoostRegressor, AdaBoostClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

def render_module4(cust_df, order_df):
    st.title("⚡ Module 4: Bagging & Boosting Ensemble Learning")
    st.markdown("---")
    
    if cust_df.empty or len(cust_df) < 5 or order_df.empty or len(order_df) < 5:
        st.warning("⚠️ Insufficient data for ensemble modeling under current filter selection. Please broaden sidebar filters.")
        return

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "🎒 Bagging Ensembles (Random Forest & Bagging)", 
        "🚀 Boosting Ensembles (AdaBoost, GBM, XGBoost & LightGBM)",
        "📊 Model Comparison Leaderboard"
    ])
    
    # -------------------------------------------------------------
    # PREPARE REGRESSION & CLASSIFICATION DATA
    # -------------------------------------------------------------
    # Regression Data (Customer Spend)
    X_reg_raw = cust_df[['age', 'total_orders', 'avg_delivery_days', 'gender', 'state']]
    y_reg = cust_df['total_spend']
    X_reg = pd.get_dummies(X_reg_raw, columns=['gender', 'state'], drop_first=True, dtype=int)
    X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Classification Data (Delivery Delay)
    X_clf_raw = order_df[['payment', 'total_order_price', 'item_count', 'age', 'order_month', 'gender', 'state']]
    y_clf = order_df['is_delayed']
    X_clf = pd.get_dummies(X_clf_raw, columns=['gender', 'state'], drop_first=True, dtype=int)
    
    min_class_cnt = y_clf.value_counts().min() if y_clf.nunique() > 1 else 0
    use_stratify = y_clf if (y_clf.nunique() > 1 and (min_class_cnt * 0.2) >= 1) else None
    X_clf_tr, X_clf_te, y_clf_tr, y_clf_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=use_stratify)

    # -------------------------------------------------------------
    # SUB TAB 1: BAGGING ENSEMBLES
    # -------------------------------------------------------------
    with sub_t1:
        st.subheader("1️⃣ Bagging Ensemble Methods (Bootstrap Aggregating)")
        st.markdown("""
        **Bagging (Bootstrap Aggregating)** builds multiple independent models in parallel using random subsets of data (with replacement) and averages their predictions to reduce variance and prevent overfitting.
        - **Random Forest**: Builds multiple randomized decision trees and averages predictions.
        - **Bagging Meta-Estimator**: Wraps base estimators (e.g. decision trees) trained on random subsets of samples and features.
        """)
        
        bag_task = st.radio("Select Prediction Task", ["Regression: Customer Spend ($)", "Classification: Delivery Delay Risk"], horizontal=True, key="bag_task")
        
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            n_estimators_bag = st.slider("Number of Trees/Estimators (n_estimators)", 10, 200, 50, step=10, key="bag_n_est")
        with col_b2:
            max_depth_bag = st.slider("Max Tree Depth (max_depth)", 1, 20, 6, key="bag_depth")
        with col_b3:
            max_samples_bag = st.slider("Subsample Ratio (max_samples)", 0.1, 1.0, 0.8, step=0.1, key="bag_samples")

        if bag_task == "Regression: Customer Spend ($)":
            st.markdown("#### 🪵 Customer Spend Prediction (Bagging)")
            
            rf_reg = RandomForestRegressor(n_estimators=n_estimators_bag, max_depth=max_depth_bag, max_samples=max_samples_bag, random_state=42)
            rf_reg.fit(X_reg_tr, y_reg_tr)
            y_pred_rf = rf_reg.predict(X_reg_te)
            
            base_dt = DecisionTreeRegressor(max_depth=max_depth_bag, random_state=42)
            bag_reg = BaggingRegressor(estimator=base_dt, n_estimators=n_estimators_bag, max_samples=max_samples_bag, random_state=42)
            bag_reg.fit(X_reg_tr, y_reg_tr)
            y_pred_bag = bag_reg.predict(X_reg_te)

            rf_r2 = r2_score(y_reg_te, y_pred_rf)
            rf_mae = mean_absolute_error(y_reg_te, y_pred_rf)
            rf_rmse = np.sqrt(mean_squared_error(y_reg_te, y_pred_rf))

            bag_r2 = r2_score(y_reg_te, y_pred_bag)
            bag_mae = mean_absolute_error(y_reg_te, y_pred_bag)
            bag_rmse = np.sqrt(mean_squared_error(y_reg_te, y_pred_bag))

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("##### 🌲 Random Forest Regressor")
                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{rf_r2:.3f}")
                c2.metric("MAE ($)", f"{rf_mae:.2f}")
                c3.metric("RMSE ($)", f"{rf_rmse:.2f}")
                
            with m_col2:
                st.markdown("##### 🎒 Bagging Meta-Regressor")
                c4, c5, c6 = st.columns(3)
                c4.metric("R² Score", f"{bag_r2:.3f}")
                c5.metric("MAE ($)", f"{bag_mae:.2f}")
                c6.metric("RMSE ($)", f"{bag_rmse:.2f}")

            imp_df = pd.DataFrame({'Feature': X_reg.columns, 'Importance': rf_reg.feature_importances_}).sort_values(by='Importance', ascending=True)
            fig_rf_imp = px.bar(imp_df.tail(10), x='Importance', y='Feature', orientation='h',
                                title="Random Forest Feature Importances", color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig_rf_imp, width='stretch')

        else:
            st.markdown("#### 🍃 Delivery Delay Risk Classification (Bagging)")
            if y_clf_tr.nunique() < 2:
                st.warning("⚠️ Training dataset contains only one class for delivery delay. Broaden sidebar filters.")
            else:
                rf_clf = RandomForestClassifier(n_estimators=n_estimators_bag, max_depth=max_depth_bag, max_samples=max_samples_bag, random_state=42)
                rf_clf.fit(X_clf_tr, y_clf_tr)
                y_pred_rf = rf_clf.predict(X_clf_te)

                base_dt = DecisionTreeClassifier(max_depth=max_depth_bag, random_state=42)
                bag_clf = BaggingClassifier(estimator=base_dt, n_estimators=n_estimators_bag, max_samples=max_samples_bag, random_state=42)
                bag_clf.fit(X_clf_tr, y_clf_tr)
                y_pred_bag = bag_clf.predict(X_clf_te)

                rf_acc = accuracy_score(y_clf_te, y_pred_rf)
                rf_f1 = f1_score(y_clf_te, y_pred_rf, zero_division=0)

                bag_acc = accuracy_score(y_clf_te, y_pred_bag)
                bag_f1 = f1_score(y_clf_te, y_pred_bag, zero_division=0)

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown("##### 🌲 Random Forest Classifier")
                    st.metric("Accuracy", f"{rf_acc*100:.1f}%")
                    st.metric("F1 Score", f"{rf_f1:.3f}")
                    cm_rf = confusion_matrix(y_clf_te, y_pred_rf)
                    fig_cm1 = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues',
                                        x=['On Time', 'Delayed'], y=['On Time', 'Delayed'], title="Confusion Matrix (Random Forest)")
                    st.plotly_chart(fig_cm1, width='stretch')

                with m_col2:
                    st.markdown("##### 🎒 Bagging Meta-Classifier")
                    st.metric("Accuracy", f"{bag_acc*100:.1f}%")
                    st.metric("F1 Score", f"{bag_f1:.3f}")
                    cm_bag = confusion_matrix(y_clf_te, y_pred_bag)
                    fig_cm2 = px.imshow(cm_bag, text_auto=True, color_continuous_scale='Blues',
                                        x=['On Time', 'Delayed'], y=['On Time', 'Delayed'], title="Confusion Matrix (Bagging)")
                    st.plotly_chart(fig_cm2, width='stretch')

    # -------------------------------------------------------------
    # SUB TAB 2: BOOSTING ENSEMBLES
    # -------------------------------------------------------------
    with sub_t2:
        st.subheader("2️⃣ Boosting Ensemble Methods (Sequential Error Correction)")
        st.markdown("""
        **Boosting** trains models sequentially, where each new tree corrects the residual errors of prior trees.
        - **AdaBoost**: Re-weights observations based on misclassifications.
        - **Gradient Boosting (GBM)**: Fits trees on pseudo-residuals using gradient descent.
        - **XGBoost**: Extreme Gradient Boosting optimized with L1/L2 regularization and tree pruning.
        - **LightGBM**: Light Gradient Boosting Machine optimized for speed using leaf-wise tree growth.
        """)

        boost_task = st.radio("Select Prediction Task", ["Regression: Customer Spend ($)", "Classification: Delivery Delay Risk"], horizontal=True, key="boost_task")

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            n_estimators_bt = st.slider("Number of Boosting Stages (n_estimators)", 10, 200, 50, step=10, key="bt_n_est")
        with col_bt2:
            learning_rate_bt = st.select_slider("Learning Rate (learning_rate)", options=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, key="bt_lr")
        with col_bt3:
            max_depth_bt = st.slider("Max Tree Depth", 1, 10, 3, key="bt_depth")

        if boost_task == "Regression: Customer Spend ($)":
            st.markdown("#### 🚀 Customer Spend Prediction (Boosting)")

            # Train AdaBoost
            ada_reg = AdaBoostRegressor(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, random_state=42)
            ada_reg.fit(X_reg_tr, y_reg_tr)
            y_pred_ada = ada_reg.predict(X_reg_te)

            # Train Gradient Boosting
            gb_reg = GradientBoostingRegressor(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42)
            gb_reg.fit(X_reg_tr, y_reg_tr)
            y_pred_gb = gb_reg.predict(X_reg_te)

            # Train XGBoost
            if HAS_XGB:
                xgb_reg = XGBRegressor(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42)
                xgb_reg.fit(X_reg_tr, y_reg_tr)
                y_pred_xgb = xgb_reg.predict(X_reg_te)

            # Train LightGBM
            if HAS_LGB:
                lgb_reg = LGBMRegressor(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42, verbosity=-1)
                lgb_reg.fit(X_reg_tr, y_reg_tr)
                y_pred_lgb = lgb_reg.predict(X_reg_te)

            # Metrics
            bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)
            with bm_c1:
                st.markdown("##### ⚡ AdaBoost")
                st.metric("R² Score", f"{r2_score(y_reg_te, y_pred_ada):.3f}")
                st.metric("MAE ($)", f"{mean_absolute_error(y_reg_te, y_pred_ada):.2f}")
                st.metric("RMSE ($)", f"{np.sqrt(mean_squared_error(y_reg_te, y_pred_ada)):.2f}")

            with bm_c2:
                st.markdown("##### 📈 Gradient Boosting")
                st.metric("R² Score", f"{r2_score(y_reg_te, y_pred_gb):.3f}")
                st.metric("MAE ($)", f"{mean_absolute_error(y_reg_te, y_pred_gb):.2f}")
                st.metric("RMSE ($)", f"{np.sqrt(mean_squared_error(y_reg_te, y_pred_gb)):.2f}")

            with bm_c3:
                st.markdown("##### 🏎️ XGBoost")
                if HAS_XGB:
                    st.metric("R² Score", f"{r2_score(y_reg_te, y_pred_xgb):.3f}")
                    st.metric("MAE ($)", f"{mean_absolute_error(y_reg_te, y_pred_xgb):.2f}")
                    st.metric("RMSE ($)", f"{np.sqrt(mean_squared_error(y_reg_te, y_pred_xgb)):.2f}")
                else:
                    st.info("XGBoost package not available.")

            with bm_c4:
                st.markdown("##### 💡 LightGBM")
                if HAS_LGB:
                    st.metric("R² Score", f"{r2_score(y_reg_te, y_pred_lgb):.3f}")
                    st.metric("MAE ($)", f"{mean_absolute_error(y_reg_te, y_pred_lgb):.2f}")
                    st.metric("RMSE ($)", f"{np.sqrt(mean_squared_error(y_reg_te, y_pred_lgb)):.2f}")
                else:
                    st.info("LightGBM package not available.")

            # Plot Feature Importance comparison for XGBoost or GBM
            target_model = xgb_reg if HAS_XGB else gb_reg
            model_name = "XGBoost" if HAS_XGB else "Gradient Boosting"
            gb_imp = pd.DataFrame({'Feature': X_reg.columns, 'Importance': target_model.feature_importances_}).sort_values(by='Importance', ascending=True)
            fig_gb_imp = px.bar(gb_imp.tail(10), x='Importance', y='Feature', orientation='h',
                                title=f"{model_name} Feature Importances", color='Importance', color_continuous_scale='Purples')
            st.plotly_chart(fig_gb_imp, width='stretch')

        else:
            st.markdown("#### 🍃 Delivery Delay Risk Classification (Boosting)")
            if y_clf_tr.nunique() < 2:
                st.warning("⚠️ Training dataset contains only one class for delivery delay. Broaden sidebar filters.")
            else:
                ada_clf = AdaBoostClassifier(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, random_state=42)
                ada_clf.fit(X_clf_tr, y_clf_tr)
                y_pred_ada = ada_clf.predict(X_clf_te)

                gb_clf = GradientBoostingClassifier(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42)
                gb_clf.fit(X_clf_tr, y_clf_tr)
                y_pred_gb = gb_clf.predict(X_clf_te)

                if HAS_XGB:
                    xgb_clf = XGBClassifier(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42, eval_metric='logloss')
                    xgb_clf.fit(X_clf_tr, y_clf_tr)
                    y_pred_xgb = xgb_clf.predict(X_clf_te)

                if HAS_LGB:
                    lgb_clf = LGBMClassifier(n_estimators=n_estimators_bt, learning_rate=learning_rate_bt, max_depth=max_depth_bt, random_state=42, verbosity=-1)
                    lgb_clf.fit(X_clf_tr, y_clf_tr)
                    y_pred_lgb = lgb_clf.predict(X_clf_te)

                bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)
                with bm_c1:
                    st.markdown("##### ⚡ AdaBoost")
                    st.metric("Accuracy", f"{accuracy_score(y_clf_te, y_pred_ada)*100:.1f}%")
                    st.metric("F1 Score", f"{f1_score(y_clf_te, y_pred_ada, zero_division=0):.3f}")

                with bm_c2:
                    st.markdown("##### 📈 Gradient Boosting")
                    st.metric("Accuracy", f"{accuracy_score(y_clf_te, y_pred_gb)*100:.1f}%")
                    st.metric("F1 Score", f"{f1_score(y_clf_te, y_pred_gb, zero_division=0):.3f}")

                with bm_c3:
                    st.markdown("##### 🏎️ XGBoost")
                    if HAS_XGB:
                        st.metric("Accuracy", f"{accuracy_score(y_clf_te, y_pred_xgb)*100:.1f}%")
                        st.metric("F1 Score", f"{f1_score(y_clf_te, y_pred_xgb, zero_division=0):.3f}")
                    else:
                        st.info("XGBoost unavailable")

                with bm_c4:
                    st.markdown("##### 💡 LightGBM")
                    if HAS_LGB:
                        st.metric("Accuracy", f"{accuracy_score(y_clf_te, y_pred_lgb)*100:.1f}%")
                        st.metric("F1 Score", f"{f1_score(y_clf_te, y_pred_lgb, zero_division=0):.3f}")
                    else:
                        st.info("LightGBM unavailable")

                # Confusion matrix for XGBoost / GBM
                target_pred = y_pred_xgb if HAS_XGB else y_pred_gb
                target_title = "XGBoost" if HAS_XGB else "Gradient Boosting"
                cm_tgt = confusion_matrix(y_clf_te, target_pred)
                fig_cm = px.imshow(cm_tgt, text_auto=True, color_continuous_scale='Purples',
                                   x=['On Time', 'Delayed'], y=['On Time', 'Delayed'], title=f"Confusion Matrix ({target_title})")
                st.plotly_chart(fig_cm, width='stretch')

    # -------------------------------------------------------------
    # SUB TAB 3: BENCHMARK LEADERBOARD
    # -------------------------------------------------------------
    with sub_t3:
        st.subheader("3️⃣ Comprehensive Model Benchmark Leaderboard")
        st.markdown("Compare baseline models against Bagging, AdaBoost, Gradient Boosting, XGBoost, and LightGBM side-by-side.")

        # Fit all Models for Benchmark
        dt_reg_base = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_reg_tr, y_reg_tr)
        rf_reg_base = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42).fit(X_reg_tr, y_reg_tr)
        ada_reg_base = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42).fit(X_reg_tr, y_reg_tr)
        gb_reg_base = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X_reg_tr, y_reg_tr)

        reg_leaderboard = [
            {'Model': 'Decision Tree (Baseline)', 'Family': 'Single Model', 'R² Score': r2_score(y_reg_te, dt_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, dt_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, dt_reg_base.predict(X_reg_te)))},
            {'Model': 'Random Forest', 'Family': 'Bagging Ensemble', 'R² Score': r2_score(y_reg_te, rf_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, rf_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, rf_reg_base.predict(X_reg_te)))},
            {'Model': 'AdaBoost', 'Family': 'Boosting Ensemble', 'R² Score': r2_score(y_reg_te, ada_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, ada_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, ada_reg_base.predict(X_reg_te)))},
            {'Model': 'Gradient Boosting', 'Family': 'Boosting Ensemble', 'R² Score': r2_score(y_reg_te, gb_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, gb_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, gb_reg_base.predict(X_reg_te)))}
        ]

        if HAS_XGB:
            xgb_reg_base = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X_reg_tr, y_reg_tr)
            reg_leaderboard.append({'Model': 'XGBoost', 'Family': 'Advanced Boosting', 'R² Score': r2_score(y_reg_te, xgb_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, xgb_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, xgb_reg_base.predict(X_reg_te)))})

        if HAS_LGB:
            lgb_reg_base = LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, verbosity=-1).fit(X_reg_tr, y_reg_tr)
            reg_leaderboard.append({'Model': 'LightGBM', 'Family': 'Advanced Boosting', 'R² Score': r2_score(y_reg_te, lgb_reg_base.predict(X_reg_te)), 'MAE ($)': mean_absolute_error(y_reg_te, lgb_reg_base.predict(X_reg_te)), 'RMSE ($)': np.sqrt(mean_squared_error(y_reg_te, lgb_reg_base.predict(X_reg_te)))})

        df_reg_lb = pd.DataFrame(reg_leaderboard).sort_values(by='R² Score', ascending=False).reset_index(drop=True)

        st.markdown("#### 💰 Regression Leaderboard: Customer Spend Prediction")
        st.dataframe(df_reg_lb.style.format({'R² Score': '{:.3f}', 'MAE ($)': '${:.2f}', 'RMSE ($)': '${:.2f}'}), width='stretch')

        fig_lb = px.bar(df_reg_lb, x='Model', y='R² Score', color='Family', text_auto='.3f',
                        title="Comprehensive Model R² Score Comparison",
                        color_discrete_map={'Single Model': 'gray', 'Bagging Ensemble': '#2563EB', 'Boosting Ensemble': '#7C3AED', 'Advanced Boosting': '#059669'})
        st.plotly_chart(fig_lb, width='stretch')
