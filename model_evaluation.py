"""
CartIQ Module 6: Advanced Model Evaluation, Calibration, Cross-Validation & Statistical Testing
Supports:
1. Probability Calibration (Platt Sigmoid & Isotonic Regression, Reliability Diagrams, Brier Score)
2. Precision-Recall Curves & PR-AUC
3. Stratified K-Fold Cross-Validation (Mean +/- Std)
4. Learning Curves (Bias-Variance Diagnosis)
5. Statistical Significance Testing (McNemar's Test for Classifiers)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, KFold, learning_curve, train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    brier_score_loss, accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)

COLOR_PRIMARY = "#2563EB"
COLOR_SUCCESS = "#10B981"
COLOR_DANGER = "#EF4444"
COLOR_WARNING = "#F59E0B"
PLOT_BG = "#FFFFFF"
GRID_COLOR = "#E2E8F0"

def apply_cartiq_layout(fig, title=None, height=450, x_title=None, y_title=None, **kwargs):
    x_final = x_title or kwargs.get("xaxis_title") or kwargs.get("x_label")
    y_final = y_title or kwargs.get("yaxis_title") or kwargs.get("y_label")
    title_final = title or kwargs.get("chart_title")

    fig.update_layout(
        title=dict(text=str(title_final), font=dict(size=16, color="#0F172A", family="sans-serif")) if title_final else None,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#0F172A", size=12),
        height=int(height) if isinstance(height, (int, float)) else 450,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor="#CBD5E1", tickfont=dict(color="#1E293B", size=11), title=dict(text=str(x_final)) if x_final else None),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor="#CBD5E1", tickfont=dict(color="#1E293B", size=11), title=dict(text=str(y_final)) if y_final else None),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#CBD5E1", borderwidth=1, font=dict(color="#0F172A", size=11))
    )
    return fig

def evaluate_probability_calibration(pipeline, X_train, y_train, X_test, y_test, num_bins=10):
    if hasattr(pipeline, "predict_proba"):
        y_prob_uncal = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob_uncal = pipeline.predict(X_test)
        
    brier_uncal = brier_score_loss(y_test, y_prob_uncal)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_prob_uncal, n_bins=num_bins, strategy="uniform")

    # Platt Scaling (Sigmoid)
    platt_cal = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=2)
    platt_cal.fit(X_train, y_train)
    y_prob_platt = platt_cal.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, y_prob_platt)
    prob_true_platt, prob_pred_platt = calibration_curve(y_test, y_prob_platt, n_bins=num_bins, strategy="uniform")

    # Isotonic Regression
    iso_cal = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=2)
    iso_cal.fit(X_train, y_train)
    y_prob_iso = iso_cal.predict_proba(X_test)[:, 1]
    brier_iso = brier_score_loss(y_test, y_prob_iso)
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_prob_iso, n_bins=num_bins, strategy="uniform")

    return {
        "uncalibrated": {
            "brier_score": float(brier_uncal),
            "prob_true": prob_true_uncal.tolist(),
            "prob_pred": prob_pred_uncal.tolist(),
            "y_prob": y_prob_uncal
        },
        "platt_sigmoid": {
            "brier_score": float(brier_platt),
            "prob_true": prob_true_platt.tolist(),
            "prob_pred": prob_pred_platt.tolist(),
            "y_prob": y_prob_platt
        },
        "isotonic": {
            "brier_score": float(brier_iso),
            "prob_true": prob_true_iso.tolist(),
            "prob_pred": prob_pred_iso.tolist(),
            "y_prob": y_prob_iso
        }
    }

def plot_calibration_curves(calib_results, model_name="Classifier"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(color="#64748B", dash="dash", width=2)))

    u = calib_results["uncalibrated"]
    fig.add_trace(go.Scatter(x=u["prob_pred"], y=u["prob_true"], mode="lines+markers", name=f"Base Uncalibrated (Brier: {u['brier_score']:.4f})", line=dict(color=COLOR_DANGER, width=2.5), marker=dict(size=7)))

    p = calib_results["platt_sigmoid"]
    fig.add_trace(go.Scatter(x=p["prob_pred"], y=p["prob_true"], mode="lines+markers", name=f"Platt Scaling (Brier: {p['brier_score']:.4f})", line=dict(color=COLOR_PRIMARY, width=2.5), marker=dict(size=7)))

    i = calib_results["isotonic"]
    fig.add_trace(go.Scatter(x=i["prob_pred"], y=i["prob_true"], mode="lines+markers", name=f"Isotonic Regression (Brier: {i['brier_score']:.4f})", line=dict(color=COLOR_SUCCESS, width=2.5), marker=dict(size=7)))

    fig = apply_cartiq_layout(fig, title=f"Reliability Diagram (Probability Calibration) — {model_name}", x_title="Mean Predicted Probability", y_title="Empirical Positive Fraction")
    return fig

def evaluate_pr_curve(y_test, y_prob, model_name="Classifier"):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = float(np.mean(y_test))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", name=f"No-Skill Baseline ({baseline:.2%})", line=dict(color="#94A3B8", dash="dash", width=1.8)))
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{model_name} (PR-AUC: {ap:.4f})", line=dict(color=COLOR_PRIMARY, width=3), fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.12)"))

    fig = apply_cartiq_layout(fig, title=f"Precision-Recall (PR) Curve — {model_name}", x_title="Recall (Sensitivity)", y_title="Precision (PPV)")
    fig.update_layout(yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1.0]))
    return fig, float(ap)

def run_stratified_kfold_cv(pipeline, X, y, n_splits=5, is_classifier=True):
    if is_classifier:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train_f, y_train_f)
        y_pred = pipeline.predict(X_val_f)

        if is_classifier:
            y_prob = pipeline.predict_proba(X_val_f)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
            acc = accuracy_score(y_val_f, y_pred)
            f1 = f1_score(y_val_f, y_pred, average="weighted", zero_division=0)
            roc = roc_auc_score(y_val_f, y_prob) if len(np.unique(y_val_f)) > 1 else 0.5
            fold_metrics.append({"Fold": fold_idx + 1, "Accuracy": acc, "F1": f1, "ROC_AUC": roc})
        else:
            r2 = r2_score(y_val_f, y_pred)
            mae = mean_absolute_error(y_val_f, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_f, y_pred))
            fold_metrics.append({"Fold": fold_idx + 1, "R2": r2, "MAE": mae, "RMSE": rmse})

    df_folds = pd.DataFrame(fold_metrics)
    summary = {}
    for col in df_folds.columns:
        if col != "Fold":
            summary[f"{col}_mean"] = float(df_folds[col].mean())
            summary[f"{col}_std"] = float(df_folds[col].std())

    return df_folds, summary

def compute_learning_curve_plot(pipeline, X, y, is_classifier=True, cv=3):
    scoring = "accuracy" if is_classifier else "r2"
    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=scoring, n_jobs=-1, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes_abs, y=train_mean, mode="lines+markers", name=f"Training Score ({scoring})", line=dict(color=COLOR_PRIMARY, width=2.5)))
    fig.add_trace(go.Scatter(x=train_sizes_abs, y=test_mean, mode="lines+markers", name=f"Validation Score ({scoring})", line=dict(color=COLOR_SUCCESS, width=2.5)))

    fig = apply_cartiq_layout(fig, title=f"Learning Curve ({scoring.upper()} vs Samples)", x_title="Training Samples", y_title=f"{scoring.capitalize()} Score")
    return fig

def run_mcnemar_test(y_true, preds_model_a, preds_model_b, name_a="Model A", name_b="Model B"):
    y_true = np.array(y_true)
    preds_a = np.array(preds_model_a)
    preds_b = np.array(preds_model_b)

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b_count = np.sum(correct_a & (~correct_b))
    c_count = np.sum((~correct_a) & correct_b)

    total_discordant = b_count + c_count
    if total_discordant == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = (abs(b_count - c_count) - 1.0) ** 2 / total_discordant
        p_value = float(1.0 - chi2.cdf(stat, df=1))

    is_significant = p_value < 0.05
    return {
        "model_a": name_a,
        "model_b": name_b,
        "b_count": int(b_count),
        "c_count": int(c_count),
        "chi2_stat": float(stat),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
        "verdict": (f"Statistically SIGNIFICANT difference (p={p_value:.4f} < 0.05)" if is_significant else f"NO statistically significant difference (p={p_value:.4f} >= 0.05)")
    }
