"""
CartIQ Dashboard: Module 6 - Model Evaluation, Calibration & Cross-Validation (M5)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from model_evaluation import (
    evaluate_probability_calibration, plot_calibration_curves,
    evaluate_pr_curve, run_stratified_kfold_cv,
    compute_learning_curve_plot, run_mcnemar_test
)

def render_module6(cust_df, order_df):
    st.title("📐 Module 6: Model Evaluation, Calibration & Statistical Testing (M5)")
    st.markdown("Comprehensive diagnostic evaluation for CartIQ ML models: Probability Calibration (Platt & Isotonic), Precision-Recall Curves, Stratified K-Fold Cross-Validation, Learning Curves, and McNemar's Hypothesis Testing.")
    st.markdown("---")

    if cust_df.empty or len(cust_df) < 10:
        st.warning("⚠️ Insufficient data for rigorous cross-validation and calibration. Please broaden sidebar filters.")
        return

    # User chooses prediction task to evaluate
    task = st.radio("Select Prediction Objective to Evaluate:", [
        "🎯 Customer High-Value Tier (is_high_value)",
        "🚚 Order Delivery Delay (is_delayed)"
    ], horizontal=True)

    if "High-Value" in task:
        feature_cols_num = ['age', 'total_orders', 'total_items', 'avg_delivery_days', 'avg_order_value']
        feature_cols_cat = ['gender', 'state']
        df_target = cust_df.dropna(subset=['is_high_value'])
        X_raw = df_target[feature_cols_num + feature_cols_cat]
        y_eval = df_target['is_high_value']
        X_eval = pd.get_dummies(X_raw, columns=feature_cols_cat, drop_first=True, dtype=int)
    else:
        feature_cols_num = ['total_order_price', 'item_count', 'age', 'order_month']
        feature_cols_cat = ['payment', 'gender', 'state']
        df_target = order_df.dropna(subset=['is_delayed'])
        X_raw = df_target[feature_cols_num + feature_cols_cat]
        y_eval = df_target['is_delayed']
        X_eval = pd.get_dummies(X_raw, columns=feature_cols_cat, drop_first=True, dtype=int)

    # Candidate models
    models_dict = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting (GBM)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree (CART)": DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    if HAS_XGB:
        models_dict["XGBoost Classifier"] = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42, eval_metric="logloss")
    if HAS_LGB:
        models_dict["LightGBM Classifier"] = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42, verbose=-1)

    sub_t1, sub_t2, sub_t3, sub_t4, sub_t5 = st.tabs([
        "🎯 Probability Calibration & Reliability Diagrams",
        "📈 Precision-Recall (PR) Curves",
        "🔁 Stratified K-Fold Cross-Validation",
        "📊 Learning Curves (Bias-Variance)",
        "⚖️ McNemar's Significance Test"
    ])

    # 1. Calibration
    with sub_t1:
        st.markdown("### 🎯 Probability Calibration & Reliability Diagrams")
        st.info("Uncalibrated models output probabilities that do not match empirical positive rates. Platt Scaling (Sigmoid) and Isotonic Regression map raw scores to reliable true risk probabilities.")

        selected_cal_model = st.selectbox("Select Classifier for Calibration:", list(models_dict.keys()), key="c_cal_select")
        clf_cal = models_dict[selected_cal_model]

        X_tr, X_te, y_tr, y_te = train_test_split(X_eval, y_eval, test_size=0.3, random_state=42, stratify=y_eval if y_eval.nunique() > 1 else None)
        clf_cal.fit(X_tr, y_tr)

        with st.spinner("Computing calibration transformations..."):
            cal_res = evaluate_probability_calibration(clf_cal, X_tr, y_tr, X_te, y_te)
            fig_cal = plot_calibration_curves(cal_res, model_name=selected_cal_model)

        c1, c2, c3 = st.columns(3)
        c1.metric("Base Brier Score", f"{cal_res['uncalibrated']['brier_score']:.4f}")
        c2.metric("Platt Scaled (Sigmoid) Brier", f"{cal_res['platt_sigmoid']['brier_score']:.4f}")
        c3.metric("Isotonic Regressed Brier", f"{cal_res['isotonic']['brier_score']:.4f}")

        st.plotly_chart(fig_cal, use_container_width=True)

    # 2. PR Curves
    with sub_t2:
        st.markdown("### 📈 Precision-Recall (PR) Curves & Average Precision (PR-AUC)")
        selected_pr_model = st.selectbox("Select Classifier for PR Analysis:", list(models_dict.keys()), key="c_pr_select")
        clf_pr = models_dict[selected_pr_model]

        X_tr, X_te, y_tr, y_te = train_test_split(X_eval, y_eval, test_size=0.3, random_state=42, stratify=y_eval if y_eval.nunique() > 1 else None)
        clf_pr.fit(X_tr, y_tr)
        y_probs = clf_pr.predict_proba(X_te)[:, 1] if hasattr(clf_pr, "predict_proba") else clf_pr.predict(X_te)

        fig_pr, ap_score = evaluate_pr_curve(y_te, y_probs, model_name=selected_pr_model)
        st.metric("PR-AUC (Average Precision)", f"{ap_score:.4f}", f"Baseline: {np.mean(y_te)*100:.1f}%")
        st.plotly_chart(fig_pr, use_container_width=True)

    # 3. K-Fold CV
    with sub_t3:
        st.markdown("### 🔁 Stratified K-Fold Cross-Validation")
        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            selected_cv_model = st.selectbox("Select Model for CV:", list(models_dict.keys()), key="c_cv_select")
            cv_splits = st.slider("Folds (k):", 3, 10, 5)
            btn_cv = st.button("🚀 Run Stratified K-Fold CV", key="btn_cartiq_cv")

        with col_k2:
            if btn_cv:
                clf_cv = models_dict[selected_cv_model]
                with st.spinner(f"Evaluating across {cv_splits} folds..."):
                    df_folds, summary = run_stratified_kfold_cv(clf_cv, X_eval, y_eval, n_splits=cv_splits, is_classifier=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Accuracy", f"{summary['Accuracy_mean']*100:.2f}%", f"± {summary['Accuracy_std']*100:.2f}%")
                m2.metric("Mean F1-Score", f"{summary['F1_mean']:.4f}", f"± {summary['F1_std']:.4f}")
                m3.metric("Mean ROC-AUC", f"{summary['ROC_AUC_mean']:.4f}", f"± {summary['ROC_AUC_std']:.4f}")

                fig_folds = px.bar(df_folds, x="Fold", y=["Accuracy", "F1", "ROC_AUC"], barmode="group", title=f"{selected_cv_model} Performance Across {cv_splits} Folds")
                st.plotly_chart(fig_folds, use_container_width=True)
                st.dataframe(df_folds, use_container_width=True)

    # 4. Learning Curves
    with sub_t4:
        st.markdown("### 📊 Learning Curves (Bias-Variance Diagnosis)")
        selected_lc_model = st.selectbox("Select Model for Learning Curve:", list(models_dict.keys()), key="c_lc_select")
        if st.button("📈 Compute Learning Curve", key="btn_cartiq_lc"):
            clf_lc = models_dict[selected_lc_model]
            with st.spinner("Computing learning curves across training sizes..."):
                fig_lc = compute_learning_curve_plot(clf_lc, X_eval, y_eval, is_classifier=True, cv=3)
            st.plotly_chart(fig_lc, use_container_width=True)

    # 5. McNemar's Test
    with sub_t5:
        st.markdown("### ⚖️ McNemar's Statistical Significance Test")
        mc_c1, mc_c2 = st.columns(2)
        with mc_c1:
            m_a_name = st.selectbox("Model A:", list(models_dict.keys()), index=0, key="mc_a_c")
        with mc_c2:
            m_b_name = st.selectbox("Model B:", list(models_dict.keys()), index=2, key="mc_b_c")

        if st.button("⚖️ Run Hypothesis Test", key="btn_cartiq_mc"):
            clf_a = models_dict[m_a_name]
            clf_b = models_dict[m_b_name]
            X_tr, X_te, y_tr, y_te = train_test_split(X_eval, y_eval, test_size=0.35, random_state=42, stratify=y_eval if y_eval.nunique() > 1 else None)
            clf_a.fit(X_tr, y_tr)
            clf_b.fit(X_tr, y_tr)
            preds_a = clf_a.predict(X_te)
            preds_b = clf_b.predict(X_te)

            res = run_mcnemar_test(y_te, preds_a, preds_b, name_a=m_a_name, name_b=m_b_name)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Discordant (A✓ B✗)", res["b_count"])
            col_m2.metric("Discordant (A✗ B✓)", res["c_count"])
            col_m3.metric("p-value", f"{res['p_value']:.4e}")

            if res["is_significant"]:
                st.success(f"🎉 **{res['verdict']}**: Statistically significant difference between models at $\\alpha = 0.05$.")
            else:
                st.info(f"ℹ️ **{res['verdict']}**: No statistically significant divergence detected.")
